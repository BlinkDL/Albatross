#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr int N = 64;
constexpr int HALF2_N = N / 2;
constexpr int LDG_ELEMS = sizeof(int4) / sizeof(half);
constexpr float TWO_NEG_41 = 4.547473508864641e-13f;
constexpr float NEXP_HALF_LOG2_E = -0.8750387749145276f;
constexpr float NLOG2_E = -1.4426950408889634f;
constexpr uint32_t ROT1 = 2654435769u;

__device__ __forceinline__ float rotator1(int x) {
  const uint32_t bits = ROT1 * static_cast<uint32_t>(x);
  return TWO_NEG_41 * static_cast<float>(static_cast<int32_t>(bits));
}

__device__ __forceinline__ half make_delta(half raw_w, half w0, int phase) {
  const float w = __half2float(raw_w) + __half2float(w0);
  const float delta =
      exp2f(NEXP_HALF_LOG2_E / (1.0f + exp2f(NLOG2_E * w))) - 1.0f + rotator1(phase);
  return __float2half_rn(delta);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ half warp_dot(half2 left, half2 right) {
  const half2 product = __hmul2(left, right);
  const float local = __half2float(product.x) + __half2float(product.y);
  return __float2half_rn(warp_sum(local));
}

template <int LIVE>
__global__ __launch_bounds__(N, 1) void deltalog_append_kernel(
    int C,
    int H,
    int64_t log_stride,
    const half* __restrict__ base_state,
    half* __restrict__ log_delta,
    half* __restrict__ log_u,
    half* __restrict__ log_b,
    half* __restrict__ log_k,
    half* __restrict__ log_v,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  static_assert(LIVE >= 0 && LIVE <= 6);
  const int h = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int64_t token = static_cast<int64_t>(batch) * C + h * N;
  const int64_t state_offset = token * N;

  __shared__ __align__(256) half2 state_smem[N][HALF2_N];
  __shared__ __align__(128) half2 cur_r[HALF2_N], cur_a[HALF2_N];
  __shared__ __align__(128) half2 cur_b[HALF2_N], cur_k[HALF2_N], cur_delta[HALF2_N];
  __shared__ __align__(128) half2 hist_delta[LIVE > 0 ? LIVE : 1][HALF2_N];
  __shared__ __align__(128) half2 hist_b[LIVE > 0 ? LIVE : 1][HALF2_N];
  __shared__ __align__(128) half2 hist_k[LIVE > 0 ? LIVE : 1][HALF2_N];
  __shared__ __align__(128) half2 final_query[2][HALF2_N];
  __shared__ half coeff_b[2][LIVE > 0 ? LIVE : 1];
  __shared__ half coeff_k[2][LIVE > 0 ? LIVE : 1];
  __shared__ half current_br, current_kr;

#pragma unroll
  for (int j0 = 0; j0 < N / LDG_ELEMS; ++j0) {
    const int4 state_vec = reinterpret_cast<const int4*>(base_state + state_offset)[j0 * N + tid];
#pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; ++j1) {
      const int row = j0 * LDG_ELEMS + tid * LDG_ELEMS / N;
      const int col = tid * LDG_ELEMS % N / 2 + j1;
      state_smem[row][(row & 31) ^ col] = reinterpret_cast<const half2*>(&state_vec)[j1];
    }
  }

  if (tid < HALF2_N) {
    const int64_t idx2 = (token >> 1) + tid;
    cur_r[tid] = reinterpret_cast<const half2*>(r_ptr)[idx2];
    cur_a[tid] = reinterpret_cast<const half2*>(a_ptr)[idx2];
    cur_b[tid] = reinterpret_cast<const half2*>(b_ptr)[idx2];
    cur_k[tid] = reinterpret_cast<const half2*>(k_ptr)[idx2];
    const half2 raw_w = reinterpret_cast<const half2*>(w_ptr)[idx2];
    const half2 w0 = reinterpret_cast<const half2*>(w0_ptr + h * N)[tid];
    const int phase0 = elapsed_t[batch] + h * N + 2 * tid;
    cur_delta[tid] = {
        make_delta(raw_w.x, w0.x, phase0),
        make_delta(raw_w.y, w0.y, phase0 + 1)};
  }
  for (int flat = tid; flat < LIVE * HALF2_N; flat += N) {
    const int slot = flat / HALF2_N;
    const int col2 = flat % HALF2_N;
    const int64_t idx2 = ((static_cast<int64_t>(slot) * log_stride + token) >> 1) + col2;
    hist_delta[slot][col2] = reinterpret_cast<const half2*>(log_delta)[idx2];
    hist_b[slot][col2] = reinterpret_cast<const half2*>(log_b)[idx2];
    hist_k[slot][col2] = reinterpret_cast<const half2*>(log_k)[idx2];
  }
  __syncthreads();

  half2 state[HALF2_N];
#pragma unroll
  for (int col2 = 0; col2 < HALF2_N; ++col2) {
    state[col2] = state_smem[tid][lane ^ col2];
  }

  half2 query = warp == 0 ? cur_a[lane] : __hfma2(cur_r[lane], cur_delta[lane], cur_r[lane]);
#pragma unroll
  for (int slot = LIVE - 1; slot >= 0; --slot) {
    const half cb = warp_dot(hist_b[slot][lane], query);
    const half ck = warp_dot(hist_k[slot][lane], query);
    if (lane == 0) {
      coeff_b[warp][slot] = cb;
      coeff_k[warp][slot] = ck;
    }
    // Correctness-critical: the log stores delta, not half(1 + delta).
    query = __hfma2(query, hist_delta[slot][lane], query);
  }
  final_query[warp][lane] = query;
  const half current_dot = warp == 0
      ? warp_dot(cur_b[lane], cur_r[lane])
      : warp_dot(cur_k[lane], cur_r[lane]);
  if (lane == 0) {
    if (warp == 0) current_br = current_dot;
    else current_kr = current_dot;
  }
  __syncthreads();

  half2 base_u2 = {0.0f, 0.0f};
  half2 base_y2 = {0.0f, 0.0f};
#pragma unroll
  for (int col2 = 0; col2 < HALF2_N; ++col2) {
    base_u2 = __hfma2(state[col2], final_query[0][col2], base_u2);
    base_y2 = __hfma2(state[col2], final_query[1][col2], base_y2);
  }
  half u = base_u2.x + base_u2.y;
  half out = base_y2.x + base_y2.y;
#pragma unroll
  for (int slot = LIVE - 1; slot >= 0; --slot) {
    const int64_t row_idx = static_cast<int64_t>(slot) * log_stride + token + tid;
    const half old_u = log_u[row_idx];
    const half old_v = log_v[row_idx];
    u = __hfma(old_u, coeff_b[0][slot], u);
    u = __hfma(old_v, coeff_k[0][slot], u);
    out = __hfma(old_u, coeff_b[1][slot], out);
    out = __hfma(old_v, coeff_k[1][slot], out);
  }

  const half current_v = v_ptr[token + tid];
  out = __hfma(u, current_br, out);
  out = __hfma(current_v, current_kr, out);
  y_ptr[token + tid] = out;

  const int64_t write_idx = static_cast<int64_t>(LIVE) * log_stride + token + tid;
  log_delta[write_idx] = reinterpret_cast<half*>(cur_delta)[tid];
  log_u[write_idx] = u;
  log_b[write_idx] = reinterpret_cast<half*>(cur_b)[tid];
  log_k[write_idx] = reinterpret_cast<half*>(cur_k)[tid];
  log_v[write_idx] = current_v;
}

template <int M>
__global__ __launch_bounds__(N, 1) void deltalog_merge_kernel(
    int C,
    int H,
    int64_t log_stride,
    half* __restrict__ base_state,
    const half* __restrict__ log_delta,
    const half* __restrict__ log_u,
    const half* __restrict__ log_b,
    const half* __restrict__ log_k,
    const half* __restrict__ log_v,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  static_assert(M >= 2 && M <= 8);
  const int h = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int64_t token = static_cast<int64_t>(batch) * C + h * N;
  const int64_t state_offset = token * N;

  __shared__ __align__(256) half2 state_smem[N][HALF2_N];
  __shared__ __align__(128) half2 cur_r[HALF2_N], cur_a[HALF2_N];
  __shared__ __align__(128) half2 cur_b[HALF2_N], cur_k[HALF2_N], cur_delta[HALF2_N];
  __shared__ __align__(128) half2 hist_delta[M - 1][HALF2_N];
  __shared__ __align__(128) half2 hist_b[M - 1][HALF2_N];
  __shared__ __align__(128) half2 hist_k[M - 1][HALF2_N];

#pragma unroll
  for (int j0 = 0; j0 < N / LDG_ELEMS; ++j0) {
    const int4 state_vec = reinterpret_cast<const int4*>(base_state + state_offset)[j0 * N + tid];
#pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; ++j1) {
      const int row = j0 * LDG_ELEMS + tid * LDG_ELEMS / N;
      const int col = tid * LDG_ELEMS % N / 2 + j1;
      state_smem[row][(row & 31) ^ col] = reinterpret_cast<const half2*>(&state_vec)[j1];
    }
  }
  if (tid < HALF2_N) {
    const int64_t idx2 = (token >> 1) + tid;
    cur_r[tid] = reinterpret_cast<const half2*>(r_ptr)[idx2];
    cur_a[tid] = reinterpret_cast<const half2*>(a_ptr)[idx2];
    cur_b[tid] = reinterpret_cast<const half2*>(b_ptr)[idx2];
    cur_k[tid] = reinterpret_cast<const half2*>(k_ptr)[idx2];
    const half2 raw_w = reinterpret_cast<const half2*>(w_ptr)[idx2];
    const half2 w0 = reinterpret_cast<const half2*>(w0_ptr + h * N)[tid];
    const int phase0 = elapsed_t[batch] + h * N + 2 * tid;
    cur_delta[tid] = {
        make_delta(raw_w.x, w0.x, phase0),
        make_delta(raw_w.y, w0.y, phase0 + 1)};
  }
  for (int flat = tid; flat < (M - 1) * HALF2_N; flat += N) {
    const int slot = flat / HALF2_N;
    const int col2 = flat % HALF2_N;
    const int64_t idx2 = ((static_cast<int64_t>(slot) * log_stride + token) >> 1) + col2;
    hist_delta[slot][col2] = reinterpret_cast<const half2*>(log_delta)[idx2];
    hist_b[slot][col2] = reinterpret_cast<const half2*>(log_b)[idx2];
    hist_k[slot][col2] = reinterpret_cast<const half2*>(log_k)[idx2];
  }
  __syncthreads();

  half2 state[HALF2_N];
#pragma unroll
  for (int col2 = 0; col2 < HALF2_N; ++col2) {
    state[col2] = state_smem[tid][lane ^ col2];
  }
#pragma unroll
  for (int slot = 0; slot < M - 1; ++slot) {
    const int64_t row_idx = static_cast<int64_t>(slot) * log_stride + token + tid;
    const half old_u = log_u[row_idx];
    const half old_v = log_v[row_idx];
    const half2 old_u2 = {old_u, old_u};
    const half2 old_v2 = {old_v, old_v};
#pragma unroll
    for (int col2 = 0; col2 < HALF2_N; ++col2) {
      const half2 old_state = state[col2];
      state[col2] = __hfma2(
          old_state,
          hist_delta[slot][col2],
          __hfma2(
              hist_k[slot][col2], old_v2,
              __hfma2(old_u2, hist_b[slot][col2], old_state)));
    }
  }

  half2 u2 = {0.0f, 0.0f};
#pragma unroll
  for (int col2 = 0; col2 < HALF2_N; ++col2) {
    u2 = __hfma2(cur_a[col2], state[col2], u2);
  }
  const half u = u2.x + u2.y;
  const half2 u_broadcast = {u, u};
  const half current_v = v_ptr[token + tid];
  const half2 v_broadcast = {current_v, current_v};
  half2 y2 = {0.0f, 0.0f};
#pragma unroll
  for (int col2 = 0; col2 < HALF2_N; ++col2) {
    const half2 old_state = state[col2];
    state[col2] = __hfma2(
        old_state,
        cur_delta[col2],
        __hfma2(cur_k[col2], v_broadcast, __hfma2(u_broadcast, cur_b[col2], old_state)));
    y2 = __hfma2(state[col2], cur_r[col2], y2);
  }
  y_ptr[token + tid] = y2.x + y2.y;

#pragma unroll
  for (int col2 = 0; col2 < HALF2_N; ++col2) {
    state_smem[tid][lane ^ col2] = state[col2];
  }
  __syncthreads();
#pragma unroll
  for (int j0 = 0; j0 < N / LDG_ELEMS; ++j0) {
    int4 state_vec;
#pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; ++j1) {
      const int row = j0 * LDG_ELEMS + tid * LDG_ELEMS / N;
      const int col = tid * LDG_ELEMS % N / 2 + j1;
      reinterpret_cast<half2*>(&state_vec)[j1] = state_smem[row][(row & 31) ^ col];
    }
    reinterpret_cast<int4*>(base_state + state_offset)[j0 * N + tid] = state_vec;
  }
}

template <int M>
void launch_step(
    int B,
    int C,
    int H,
    int phase,
    int64_t log_stride,
    at::Tensor base_state,
    half* log_delta,
    half* log_u,
    half* log_b,
    half* log_k,
    half* log_v,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(H, B, 1);
  if (phase == M - 1) {
    deltalog_merge_kernel<M><<<grid, N, 0, stream>>>(
        C, H, log_stride,
        reinterpret_cast<half*>(base_state.data_ptr()),
        log_delta, log_u, log_b, log_k, log_v,
        reinterpret_cast<const half*>(r.data_ptr()),
        reinterpret_cast<const half*>(w.data_ptr()),
        reinterpret_cast<const half*>(w0.data_ptr()),
        reinterpret_cast<const half*>(k.data_ptr()),
        reinterpret_cast<const half*>(v.data_ptr()),
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b.data_ptr()),
        reinterpret_cast<half*>(y.data_ptr()),
        elapsed_t.data_ptr<int>());
  } else {
#define LAUNCH_APPEND(LIVE) \
    deltalog_append_kernel<LIVE><<<grid, N, 0, stream>>>( \
        C, H, log_stride, \
        reinterpret_cast<const half*>(base_state.data_ptr()), \
        log_delta, log_u, log_b, log_k, log_v, \
        reinterpret_cast<const half*>(r.data_ptr()), \
        reinterpret_cast<const half*>(w.data_ptr()), \
        reinterpret_cast<const half*>(w0.data_ptr()), \
        reinterpret_cast<const half*>(k.data_ptr()), \
        reinterpret_cast<const half*>(v.data_ptr()), \
        reinterpret_cast<const half*>(a.data_ptr()), \
        reinterpret_cast<const half*>(b.data_ptr()), \
        reinterpret_cast<half*>(y.data_ptr()), \
        elapsed_t.data_ptr<int>())
    switch (phase) {
      case 0: LAUNCH_APPEND(0); break;
      case 1: LAUNCH_APPEND(1); break;
      case 2: LAUNCH_APPEND(2); break;
      case 3: LAUNCH_APPEND(3); break;
      case 4: LAUNCH_APPEND(4); break;
      case 5: LAUNCH_APPEND(5); break;
      case 6: LAUNCH_APPEND(6); break;
      default: TORCH_CHECK(false, "invalid append phase");
    }
#undef LAUNCH_APPEND
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void deltalog_step_v3a_cuda_impl(
    int B,
    int C,
    int H,
    int merge_interval,
    int phase,
    int64_t log_stride,
    at::Tensor base_state,
    half* log_delta,
    half* log_u,
    half* log_b,
    half* log_k,
    half* log_v,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
#define DISPATCH_M(value) \
  case value: \
    launch_step<value>(B, C, H, phase, log_stride, base_state, log_delta, log_u, log_b, log_k, log_v, \
                       r, w, w0, k, v, a, b, y, elapsed_t); \
    break
  switch (merge_interval) {
    DISPATCH_M(2);
    DISPATCH_M(3);
    DISPATCH_M(4);
    DISPATCH_M(6);
    DISPATCH_M(8);
    default: TORCH_CHECK(false, "unsupported merge interval");
  }
#undef DISPATCH_M
}

void deltalog_step_v3a_cuda(
    int B,
    int C,
    int H,
    int merge_interval,
    int phase,
    at::Tensor base_state,
    at::Tensor log_delta,
    at::Tensor log_u,
    at::Tensor log_b,
    at::Tensor log_k,
    at::Tensor log_v,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  deltalog_step_v3a_cuda_impl(
      B, C, H, merge_interval, phase, static_cast<int64_t>(B) * C,
      base_state,
      reinterpret_cast<half*>(log_delta.data_ptr()),
      reinterpret_cast<half*>(log_u.data_ptr()),
      reinterpret_cast<half*>(log_b.data_ptr()),
      reinterpret_cast<half*>(log_k.data_ptr()),
      reinterpret_cast<half*>(log_v.data_ptr()),
      r, w, w0, k, v, a, b, y, elapsed_t);
}

void deltalog_step_slot_packed_v3a_cuda(
    int B,
    int C,
    int H,
    int L,
    int layer,
    int merge_interval,
    int phase,
    at::Tensor base_state,
    at::Tensor packed_log,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  const int64_t layer_numel = static_cast<int64_t>(B) * C;
  const int64_t kind_stride = static_cast<int64_t>(L) * layer_numel;
  const int64_t slot_stride = 5 * kind_stride;
  half* const packed = reinterpret_cast<half*>(packed_log.data_ptr());
  // Correctness-critical layout: (slot, kind, layer, B, C). Each kind base
  // starts in slot0; slot_stride reaches the same kind in the next slot.
  half* const log_delta = packed + 0 * kind_stride + layer * layer_numel;
  half* const log_u = packed + 1 * kind_stride + layer * layer_numel;
  half* const log_b = packed + 2 * kind_stride + layer * layer_numel;
  half* const log_k = packed + 3 * kind_stride + layer * layer_numel;
  half* const log_v = packed + 4 * kind_stride + layer * layer_numel;
  deltalog_step_v3a_cuda_impl(
      B, C, H, merge_interval, phase, slot_stride, base_state,
      log_delta, log_u, log_b, log_k, log_v,
      r, w, w0, k, v, a, b, y, elapsed_t);
}

void deltalog_step_slot_layer_packed_v3a_cuda(
    int B,
    int C,
    int H,
    int L,
    int layer,
    int merge_interval,
    int phase,
    at::Tensor base_state,
    at::Tensor packed_log,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  const int64_t layer_numel = static_cast<int64_t>(B) * C;
  const int64_t layer_stride = 5 * layer_numel;
  const int64_t slot_stride = static_cast<int64_t>(L) * layer_stride;
  half* const packed = reinterpret_cast<half*>(packed_log.data_ptr());
  // Correctness-critical layout: (slot, layer, kind, B, C). A full slot is
  // still one persisting-L2 window, while each layer's five streams are local.
  half* const layer_base = packed + layer * layer_stride;
  half* const log_delta = layer_base + 0 * layer_numel;
  half* const log_u = layer_base + 1 * layer_numel;
  half* const log_b = layer_base + 2 * layer_numel;
  half* const log_k = layer_base + 3 * layer_numel;
  half* const log_v = layer_base + 4 * layer_numel;
  deltalog_step_v3a_cuda_impl(
      B, C, H, merge_interval, phase, slot_stride, base_state,
      log_delta, log_u, log_b, log_k, log_v,
      r, w, w0, k, v, a, b, y, elapsed_t);
}
