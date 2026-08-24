#define WKV_LAYOUT_ENTRY wkv_fp32_v2_cuda

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadSize = 64;
constexpr int kWarpThreads = 32;
constexpr int kValueTile = 4;
constexpr int kKeysPerWarp = kWarpThreads / kValueTile;
constexpr float kWScaleLog2E = -0.8750387749145276f;
constexpr float kNLog2E = -1.4426950408889634f;

__device__ __forceinline__ float load_half(const __half* ptr, int64_t index) {
  return __half2float(__ldg(ptr + index));
}

__device__ __forceinline__ float retention(float w) {
  return exp2f(kWScaleLog2E / (1.0f + exp2f(kNLog2E * w)));
}

// Physical state ABI is [B,H,K,V]. Lane v owns one V column and keeps all K
// values in registers. Adjacent lanes therefore issue contiguous state loads.
__global__ __launch_bounds__(kHeadSize, 2) void wkv_kv_large_kernel(
    int T,
    int C,
    int H,
    float* __restrict__ state_ptr,
    const __half* __restrict__ r_ptr,
    const __half* __restrict__ w_ptr,
    const __half* __restrict__ k_ptr,
    const __half* __restrict__ v_ptr,
    const __half* __restrict__ a_ptr,
    const __half* __restrict__ b_ptr,
    __half* __restrict__ y_ptr) {
  const int bh = static_cast<int>(blockIdx.x);
  const int batch = bh / H;
  const int head = bh - batch * H;
  const int value_index = static_cast<int>(threadIdx.x);
  const int channel_base = head * kHeadSize;
  const int64_t token_base =
      static_cast<int64_t>(batch) * T * C + channel_base;
  float* state_base = state_ptr +
      (static_cast<int64_t>(batch) * H + head) * kHeadSize * kHeadSize;

  float state[kHeadSize];
#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state[key_index] = state_base[key_index * kHeadSize + value_index];
  }

  __shared__ float r[kHeadSize];
  __shared__ float decay[kHeadSize];
  __shared__ float k[kHeadSize];
  __shared__ float a[kHeadSize];
  __shared__ float b[kHeadSize];

  for (int token = 0; token < T; ++token) {
    const int64_t index = token_base + static_cast<int64_t>(token) * C + value_index;
    __syncthreads();
    r[value_index] = load_half(r_ptr, index);
    decay[value_index] = retention(load_half(w_ptr, index));
    k[value_index] = load_half(k_ptr, index);
    a[value_index] = load_half(a_ptr, index);
    b[value_index] = load_half(b_ptr, index);
    __syncthreads();

    float a_state = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      a_state += state[key_index] * a[key_index];
    }

    const float value = load_half(v_ptr, index);
    float result = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      const float updated = state[key_index] * decay[key_index] +
          a_state * b[key_index] + k[key_index] * value;
      state[key_index] = updated;
      result += updated * r[key_index];
    }
    y_ptr[index] = __float2half_rn(result);
  }

#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state_base[key_index * kHeadSize + value_index] = state[key_index];
  }
}

// FlashRWKV2's B1T1 path: one warp owns four V columns. Every group of four
// lanes shares K, while the low two lane bits select contiguous V addresses.
__global__ __launch_bounds__(kWarpThreads, 4) void wkv_kv_tile_kernel(
    int C,
    int H,
    float* __restrict__ state_ptr,
    const __half* __restrict__ r_ptr,
    const __half* __restrict__ w_ptr,
    const __half* __restrict__ k_ptr,
    const __half* __restrict__ v_ptr,
    const __half* __restrict__ a_ptr,
    const __half* __restrict__ b_ptr,
    __half* __restrict__ y_ptr) {
  const int head = static_cast<int>(blockIdx.x);
  const int batch = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  const int value_lane = lane & (kValueTile - 1);
  const int key_lane = lane / kValueTile;
  const int value_index = static_cast<int>(blockIdx.z) * kValueTile + value_lane;
  const int64_t token_base = static_cast<int64_t>(batch) * C + head * kHeadSize;
  float* state_base = state_ptr +
      (static_cast<int64_t>(batch) * H + head) * kHeadSize * kHeadSize;

  float a_state = 0.0f;
#pragma unroll
  for (int key_base = 0; key_base < kHeadSize; key_base += kKeysPerWarp) {
    const int key_index = key_base + key_lane;
    float contribution =
        state_base[key_index * kHeadSize + value_index] *
        load_half(a_ptr, token_base + key_index);
#pragma unroll
    for (int offset = kWarpThreads / 2; offset >= kValueTile; offset >>= 1) {
      contribution += __shfl_down_sync(0xffffffffu, contribution, offset);
    }
    if (lane < kValueTile) {
      a_state += contribution;
    }
  }
  a_state = __shfl_sync(0xffffffffu, a_state, value_lane);
  float value = lane < kValueTile ? load_half(v_ptr, token_base + value_index) : 0.0f;
  value = __shfl_sync(0xffffffffu, value, value_lane);

  float result = 0.0f;
#pragma unroll
  for (int key_base = 0; key_base < kHeadSize; key_base += kKeysPerWarp) {
    const int key_index = key_base + key_lane;
    const int64_t input_index = token_base + key_index;
    const int state_index = key_index * kHeadSize + value_index;
    const float updated = retention(load_half(w_ptr, input_index)) *
            state_base[state_index] +
        load_half(b_ptr, input_index) * a_state +
        load_half(k_ptr, input_index) * value;
    state_base[state_index] = updated;
    float contribution = load_half(r_ptr, input_index) * updated;
#pragma unroll
    for (int offset = kWarpThreads / 2; offset >= kValueTile; offset >>= 1) {
      contribution += __shfl_down_sync(0xffffffffu, contribution, offset);
    }
    if (lane < kValueTile) {
      result += contribution;
    }
  }
  if (lane < kValueTile) {
    y_ptr[token_base + value_index] = __float2half_rn(result);
  }
}

// One warp owns ValueTile adjacent V columns. Unlike the T=1 control kernel,
// this path loads each [K,V] state element once and keeps it in registers for
// the complete token loop. The lane layout is intentionally K-major/V-minor:
// lanes in one K group share r/w/k/a/b while lanes with the same V reduce the
// 64-key dot products. Changing this mapping silently changes the state ABI.
template <int ValueTile>
__global__ __launch_bounds__(kWarpThreads, 4) void wkv_kv_tloop_kernel(
    int T,
    int C,
    int H,
    float* __restrict__ state_ptr,
    const __half* __restrict__ r_ptr,
    const __half* __restrict__ w_ptr,
    const __half* __restrict__ k_ptr,
    const __half* __restrict__ v_ptr,
    const __half* __restrict__ a_ptr,
    const __half* __restrict__ b_ptr,
    __half* __restrict__ y_ptr) {
  static_assert(ValueTile == 4 || ValueTile == 8 || ValueTile == 16);
  constexpr int kKeyLanes = kWarpThreads / ValueTile;
  constexpr int kStatesPerLane = kHeadSize / kKeyLanes;

  const int head = static_cast<int>(blockIdx.x);
  const int batch = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  const int value_lane = lane & (ValueTile - 1);
  const int key_lane = lane / ValueTile;
  const int value_index = static_cast<int>(blockIdx.z) * ValueTile + value_lane;
  const int channel_base = head * kHeadSize;
  const int64_t token_base =
      static_cast<int64_t>(batch) * T * C + channel_base;
  float* state_base = state_ptr +
      (static_cast<int64_t>(batch) * H + head) * kHeadSize * kHeadSize;

  float state[kStatesPerLane];
#pragma unroll
  for (int slot = 0; slot < kStatesPerLane; ++slot) {
    const int key_index = slot * kKeyLanes + key_lane;
    state[slot] = state_base[key_index * kHeadSize + value_index];
  }

  for (int token = 0; token < T; ++token) {
    const int64_t input_base = token_base + static_cast<int64_t>(token) * C;
    float a_state = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kStatesPerLane; ++slot) {
      const int key_index = slot * kKeyLanes + key_lane;
      float a_value = value_lane == 0
          ? load_half(a_ptr, input_base + key_index)
          : 0.0f;
      a_value = __shfl_sync(
          0xffffffffu, a_value, key_lane * ValueTile);
      a_state += state[slot] * a_value;
    }
#pragma unroll
    for (int offset = kWarpThreads / 2; offset >= ValueTile; offset >>= 1) {
      a_state += __shfl_down_sync(0xffffffffu, a_state, offset);
    }
    a_state = __shfl_sync(0xffffffffu, a_state, value_lane);

    float value = lane < ValueTile
        ? load_half(v_ptr, input_base + value_index)
        : 0.0f;
    value = __shfl_sync(0xffffffffu, value, value_lane);

    float result = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kStatesPerLane; ++slot) {
      const int key_index = slot * kKeyLanes + key_lane;
      float decay = 0.0f;
      float b_value = 0.0f;
      float k_value = 0.0f;
      float r_value = 0.0f;
      if (value_lane == 0) {
        decay = retention(load_half(w_ptr, input_base + key_index));
        b_value = load_half(b_ptr, input_base + key_index);
        k_value = load_half(k_ptr, input_base + key_index);
        r_value = load_half(r_ptr, input_base + key_index);
      }
      const int source_lane = key_lane * ValueTile;
      decay = __shfl_sync(0xffffffffu, decay, source_lane);
      b_value = __shfl_sync(0xffffffffu, b_value, source_lane);
      k_value = __shfl_sync(0xffffffffu, k_value, source_lane);
      r_value = __shfl_sync(0xffffffffu, r_value, source_lane);

      const float updated = state[slot] * decay +
          a_state * b_value + k_value * value;
      state[slot] = updated;
      result += updated * r_value;
    }
#pragma unroll
    for (int offset = kWarpThreads / 2; offset >= ValueTile; offset >>= 1) {
      result += __shfl_down_sync(0xffffffffu, result, offset);
    }
    if (lane < ValueTile) {
      y_ptr[input_base + value_index] = __float2half_rn(result);
    }
  }

#pragma unroll
  for (int slot = 0; slot < kStatesPerLane; ++slot) {
    const int key_index = slot * kKeyLanes + key_lane;
    state_base[key_index * kHeadSize + value_index] = state[slot];
  }
}

// Four independent value-tile warps share one CTA and one copy of the token
// vectors. This raises resident warps past the one-warp-CTA block-slot limit
// and cuts duplicated r/w/k/v/a/b traffic by 4x. It intentionally remains a
// separate probe: at very small B*H, fewer CTAs can reduce SM coverage.
template <int ValueTile>
__global__ __launch_bounds__(kWarpThreads * 4, 2)
void wkv_kv_tloop_group4_kernel(
    int T,
    int C,
    int H,
    float* __restrict__ state_ptr,
    const __half* __restrict__ r_ptr,
    const __half* __restrict__ w_ptr,
    const __half* __restrict__ k_ptr,
    const __half* __restrict__ v_ptr,
    const __half* __restrict__ a_ptr,
    const __half* __restrict__ b_ptr,
    __half* __restrict__ y_ptr) {
  static_assert(ValueTile == 4 || ValueTile == 8 || ValueTile == 16);
  constexpr int kWarpsPerBlock = 4;
  constexpr int kKeyLanes = kWarpThreads / ValueTile;
  constexpr int kStatesPerLane = kHeadSize / kKeyLanes;

  const int head = static_cast<int>(blockIdx.x);
  const int batch = static_cast<int>(blockIdx.y);
  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / kWarpThreads;
  const int lane = thread & (kWarpThreads - 1);
  const int value_lane = lane & (ValueTile - 1);
  const int key_lane = lane / ValueTile;
  const int value_tile = static_cast<int>(blockIdx.z) * kWarpsPerBlock + warp;
  const int value_index = value_tile * ValueTile + value_lane;
  const int channel_base = head * kHeadSize;
  const int64_t token_base =
      static_cast<int64_t>(batch) * T * C + channel_base;
  float* state_base = state_ptr +
      (static_cast<int64_t>(batch) * H + head) * kHeadSize * kHeadSize;

  float state[kStatesPerLane];
#pragma unroll
  for (int slot = 0; slot < kStatesPerLane; ++slot) {
    const int key_index = slot * kKeyLanes + key_lane;
    state[slot] = state_base[key_index * kHeadSize + value_index];
  }

  __shared__ float shared_r[kHeadSize];
  __shared__ float shared_decay[kHeadSize];
  __shared__ float shared_k[kHeadSize];
  __shared__ float shared_v[kHeadSize];
  __shared__ float shared_a[kHeadSize];
  __shared__ float shared_b[kHeadSize];

  for (int token = 0; token < T; ++token) {
    const int64_t input_base = token_base + static_cast<int64_t>(token) * C;
    // This first barrier is also the consumer barrier for the previous token.
    // Removing it allows early warps to overwrite shared vectors still in use.
    __syncthreads();
    if (thread < kHeadSize) {
      const int64_t input_index = input_base + thread;
      shared_r[thread] = load_half(r_ptr, input_index);
      shared_decay[thread] = retention(load_half(w_ptr, input_index));
      shared_k[thread] = load_half(k_ptr, input_index);
      shared_v[thread] = load_half(v_ptr, input_index);
      shared_a[thread] = load_half(a_ptr, input_index);
      shared_b[thread] = load_half(b_ptr, input_index);
    }
    __syncthreads();

    float a_state = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kStatesPerLane; ++slot) {
      const int key_index = slot * kKeyLanes + key_lane;
      a_state += state[slot] * shared_a[key_index];
    }
#pragma unroll
    for (int offset = kWarpThreads / 2; offset >= ValueTile; offset >>= 1) {
      a_state += __shfl_down_sync(0xffffffffu, a_state, offset);
    }
    a_state = __shfl_sync(0xffffffffu, a_state, value_lane);

    const float value = shared_v[value_index];
    float result = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kStatesPerLane; ++slot) {
      const int key_index = slot * kKeyLanes + key_lane;
      const float updated = state[slot] * shared_decay[key_index] +
          a_state * shared_b[key_index] + shared_k[key_index] * value;
      state[slot] = updated;
      result += updated * shared_r[key_index];
    }
#pragma unroll
    for (int offset = kWarpThreads / 2; offset >= ValueTile; offset >>= 1) {
      result += __shfl_down_sync(0xffffffffu, result, offset);
    }
    if (lane < ValueTile) {
      y_ptr[input_base + value_index] = __float2half_rn(result);
    }
  }

#pragma unroll
  for (int slot = 0; slot < kStatesPerLane; ++slot) {
    const int key_index = slot * kKeyLanes + key_lane;
    state_base[key_index * kHeadSize + value_index] = state[slot];
  }
}

}  // namespace

#ifndef WKV_LAYOUT_ENTRY
#define WKV_LAYOUT_ENTRY wkv_fp32_kv_layout_cuda
#endif

void WKV_LAYOUT_ENTRY(
    int B,
    int T,
    int C,
    int H,
    int mode,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* state_ptr = state.data_ptr<float>();
  auto* r_ptr = reinterpret_cast<const __half*>(r.data_ptr<at::Half>());
  auto* w_ptr = reinterpret_cast<const __half*>(w.data_ptr<at::Half>());
  auto* k_ptr = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
  auto* v_ptr = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
  auto* a_ptr = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
  auto* b_ptr = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
  auto* y_ptr = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

  if ((mode == 2 && T == 1) || (mode == 0 && B == 1 && T == 1)) {
    wkv_kv_tile_kernel<<<dim3(H, B, kHeadSize / kValueTile), kWarpThreads, 0, stream>>>(
        C, H, state_ptr, r_ptr, w_ptr, k_ptr, v_ptr, a_ptr, b_ptr, y_ptr);
    return;
  }
  if (mode == 0) {
    const int batch_heads = B * H;
    if (T == 5 && batch_heads <= 80) {
      mode = 7;
    } else if (T >= 6 && T <= 9 && batch_heads <= 80) {
      mode = batch_heads <= 40 ? 6 : 7;
    } else if (T >= 10 && batch_heads <= 48) {
      mode = 6;
    }
  }
  if (mode >= 3 && mode <= 5) {
    if (mode == 3) {
      wkv_kv_tloop_kernel<4><<<dim3(H, B, kHeadSize / 4), kWarpThreads, 0, stream>>>(
          T, C, H, state_ptr, r_ptr, w_ptr, k_ptr, v_ptr, a_ptr, b_ptr, y_ptr);
    } else if (mode == 4) {
      wkv_kv_tloop_kernel<8><<<dim3(H, B, kHeadSize / 8), kWarpThreads, 0, stream>>>(
          T, C, H, state_ptr, r_ptr, w_ptr, k_ptr, v_ptr, a_ptr, b_ptr, y_ptr);
    } else {
      wkv_kv_tloop_kernel<16><<<dim3(H, B, kHeadSize / 16), kWarpThreads, 0, stream>>>(
          T, C, H, state_ptr, r_ptr, w_ptr, k_ptr, v_ptr, a_ptr, b_ptr, y_ptr);
    }
    return;
  }
  if (mode >= 6 && mode <= 8) {
    if (mode == 6) {
      wkv_kv_tloop_group4_kernel<4><<<
          dim3(H, B, kHeadSize / (4 * 4)), kWarpThreads * 4, 0, stream>>>(
          T, C, H, state_ptr, r_ptr, w_ptr, k_ptr, v_ptr, a_ptr, b_ptr, y_ptr);
    } else if (mode == 7) {
      wkv_kv_tloop_group4_kernel<8><<<
          dim3(H, B, kHeadSize / (8 * 4)), kWarpThreads * 4, 0, stream>>>(
          T, C, H, state_ptr, r_ptr, w_ptr, k_ptr, v_ptr, a_ptr, b_ptr, y_ptr);
    } else {
      wkv_kv_tloop_group4_kernel<16><<<
          dim3(H, B, kHeadSize / (16 * 4)), kWarpThreads * 4, 0, stream>>>(
          T, C, H, state_ptr, r_ptr, w_ptr, k_ptr, v_ptr, a_ptr, b_ptr, y_ptr);
    }
    return;
  }
  wkv_kv_large_kernel<<<B * H, kHeadSize, 0, stream>>>(
      T, C, H, state_ptr, r_ptr, w_ptr, k_ptr, v_ptr, a_ptr, b_ptr, y_ptr);
}
