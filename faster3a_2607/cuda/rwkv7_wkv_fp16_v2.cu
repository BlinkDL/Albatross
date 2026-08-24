#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <assert.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr int N = 64;
constexpr int HALF2_N = N / 2;
constexpr int LDG_ELEMS = sizeof(int4) / sizeof(half);

#ifndef RWKV_FP16_KV_STREAM_T1_STATE
#define RWKV_FP16_KV_STREAM_T1_STATE 1
#endif
constexpr float TWO_NEG_41 = 4.547473508864641e-13f;
constexpr float NEXP_HALF_LOG2_E = -0.8750387749145276f;
constexpr float NLOG2_E = -1.4426950408889634f;
constexpr uint32_t ROT1 = 2654435769u;
using F = half;
#define CLONE_N 64

__device__ __forceinline__ float rotator1(int x) {
  const uint32_t bits = ROT1 * static_cast<uint32_t>(x);
  return TWO_NEG_41 * static_cast<float>(static_cast<int32_t>(bits));
}

__device__ __forceinline__ half w_delta(float w, int phase) {
  float d = exp2f(NEXP_HALF_LOG2_E / (1.0f + exp2f(NLOG2_E * w))) - 1.0f + rotator1(phase);
  return __float2half_rn(d);
}

template <bool AddW0>
__device__ __forceinline__ half w_delta_maybe_w0(half w_raw, const half* __restrict__ w0_ptr, int c, int phase) {
  float w = __half2float(w_raw);
  if constexpr (AddW0) {
    w += __half2float(w0_ptr[c]);
  }
  return w_delta(w, phase);
}

// Candidate ABI: state is physically [K,V].  A thread still owns one V row
// for the recurrent arithmetic, so each half2 register combines two K rows.
// The two scalar instructions are warp-coalesced across adjacent V lanes.
__device__ __forceinline__ half2 load_state_kv(
    const half* __restrict__ state_base, int v, int k2) {
  return __halves2half2(
      __ldg(state_base + (2 * k2) * N + v),
      __ldg(state_base + (2 * k2 + 1) * N + v));
}

__device__ __forceinline__ void store_state_kv(
    half* __restrict__ state_base, int v, int k2, half2 value) {
  state_base[(2 * k2) * N + v] = value.x;
  state_base[(2 * k2 + 1) * N + v] = value.y;
}

template <int Bytes>
__device__ __forceinline__ void clone_cp_async(void const* smem_addr, void const* global_ptr, bool cond) {
  static_assert(Bytes == 16 || Bytes == 8 || Bytes == 4);
  int bytes = cond ? Bytes : 0;
  unsigned int addr = __cvta_generic_to_shared(smem_addr);
  if constexpr (Bytes == 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;" ::"r"(addr), "l"(global_ptr), "n"(Bytes), "r"(bytes));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(addr), "l"(global_ptr), "n"(Bytes), "r"(bytes));
  }
}

template <int NWait>
__device__ __forceinline__ void clone_cp_wait() {
  if constexpr (NWait == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NWait));
  }
}

__device__ __forceinline__ void clone_cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <bool Grid2D>
__device__ __forceinline__ void decode_batch_head(int H, int& batch, int& head) {
  if constexpr (Grid2D) {
    head = static_cast<int>(blockIdx.x);
    batch = static_cast<int>(blockIdx.y);
  } else {
    const int bh = static_cast<int>(blockIdx.x);
    batch = bh / H;
    head = bh - batch * H;
  }
}

template <bool Tis1 = false, bool AddW0 = false, bool Grid2D = false>
__global__ void __launch_bounds__(CLONE_N, 2) wkv_fp16_v1_clone_kernel(
    const int B,
    const int T,
    const int C,
    const int H,
    F* __restrict__ state_ptr,
    const F* __restrict__ r_ptr,
    const F* __restrict__ w_ptr,
    const F* __restrict__ w0_ptr,
    const F* __restrict__ k_ptr,
    const F* __restrict__ v_ptr,
    const F* __restrict__ a_ptr,
    const F* __restrict__ b_ptr,
    F* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  if constexpr (Tis1) {
    __builtin_assume(T == 1);
  }
  int b;
  int h;
  decode_batch_head<Grid2D>(H, b, h);
  const int i = threadIdx.x;
  const int lane = i % 32;

  state_ptr += b * C * CLONE_N + h * CLONE_N * CLONE_N;
  half2 state[CLONE_N / 2];
#pragma unroll
  for (int j = 0; j < CLONE_N / 2; j++) {
    state[j] = load_state_kv(state_ptr, i, j);
  }

  __shared__ __align__(128) half2 r[CLONE_N / 2], k[CLONE_N / 2], w[CLONE_N / 2], a[CLONE_N / 2], bvec[CLONE_N / 2], bvec_dummy[CLONE_N / 2];
#pragma unroll
  for (int tt = 0; tt < T; tt++) {
    int t = b * T * C + h * CLONE_N + tt * C;
    __syncthreads();
    clone_cp_async<4>((half2*)(i < 32 ? w : a) + lane, (half2*)((i < 32 ? w_ptr : a_ptr) + t) + lane, true);
    clone_cp_commit();
    clone_cp_async<4>((half2*)(i < 32 ? r : k) + lane, (half2*)((i < 32 ? r_ptr : k_ptr) + t) + lane, true);
    // src-size 0 zero-fills, so warp 1 must target dummy shared instead of
    // racing warp 0's real bvec copy. Keep this branchless for WKV occupancy.
    clone_cp_async<4>((i < 32 ? bvec : bvec_dummy) + lane, (half2*)(b_ptr + t) + lane, i < 32);
    clone_cp_commit();

    half vv = v_ptr[t + i];
    half2 vv2 = {vv, vv};
    half2 y2 = {0.0, 0.0};
    half2 sa2 = {0.0, 0.0};
    clone_cp_wait<1>();
    __syncthreads();
#pragma unroll
    for (int j = 0; j < CLONE_N / 2; j++) {
      sa2 = __hfma2(a[j], state[j], sa2);
    }
    half sa = sa2.x + sa2.y;
    sa2 = {sa, sa};
    ((F*)w)[i] = w_delta_maybe_w0<AddW0>(((F*)w)[i], w0_ptr, h * CLONE_N + i, elapsed_t[b] + h * CLONE_N + i + tt);

    clone_cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int j = 0; j < CLONE_N / 2; j++) {
      half2& s = state[j];
      s = __hfma2(s, w[j], __hfma2(k[j], vv2, __hfma2(sa2, bvec[j], s)));
      y2 = __hfma2(s, r[j], y2);
    }
    y_ptr[t + i] = y2.x + y2.y;
  }

#pragma unroll
  for (int j = 0; j < CLONE_N / 2; j++) {
    store_state_kv(state_ptr, i, j, state[j]);
  }
}

template <int Bytes>
__device__ __forceinline__ void cp_async(void* smem, const void* global, bool pred) {
  static_assert(Bytes == 16 || Bytes == 8 || Bytes == 4);
  int bytes = pred ? Bytes : 0;
  unsigned addr = __cvta_generic_to_shared(smem);
  if constexpr (Bytes == 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;" ::"r"(addr), "l"(global), "n"(Bytes), "r"(bytes));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(addr), "l"(global), "n"(Bytes), "r"(bytes));
  }
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int NWait>
__device__ __forceinline__ void cp_wait() {
  if constexpr (NWait == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NWait));
  }
}

__device__ __forceinline__ void prefetch_token(
    int tid,
    int lane,
    int token,
    half2* r,
    half2* w,
    half2* k,
    half2* a,
    half2* b,
    half2* b_dummy,
    const half* r_ptr,
    const half* w_ptr,
    const half* k_ptr,
    const half* a_ptr,
    const half* b_ptr) {
  cp_async<4>((tid < 32 ? w : a) + lane, (const half2*)(tid < 32 ? w_ptr + token : a_ptr + token) + lane, true);
  cp_commit();
  cp_async<4>((tid < 32 ? r : k) + lane, (const half2*)(tid < 32 ? r_ptr + token : k_ptr + token) + lane, true);
  // A predicated-off cp.async zero-fills its destination. Warp 1 therefore
  // needs a disjoint target; pointing both warps at b would be a shared race.
  cp_async<4>((tid < 32 ? b : b_dummy) + lane, (const half2*)(b_ptr + token) + lane, tid < 32);
  cp_commit();
}

__device__ __forceinline__ float warp_sum_float(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <int RowsPerCta, bool AddW0>
__global__ __launch_bounds__(RowsPerCta * 32, 1) void wkv_fp16_grouped_rows_kernel(
    int T,
    int C,
    int H,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  static_assert(RowsPerCta == 4 || RowsPerCta == 8 || RowsPerCta == 16);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * RowsPerCta + warp;
  const int h = blockIdx.y;
  const int b_id = blockIdx.z;
  const int c0 = lane * 2;

  half* const state_base =
      state_ptr + (static_cast<int64_t>(b_id) * H + h) * N * N;
  // This compatibility implementation is correct for [K,V], but lanes walk
  // K while row is V, so it is intentionally not selected by auto dispatch.
  half2 state = __halves2half2(
      __ldg(state_base + c0 * N + row),
      __ldg(state_base + (c0 + 1) * N + row));

  __shared__ __align__(128) half2 r[HALF2_N];
  __shared__ __align__(128) half2 w[HALF2_N];
  __shared__ __align__(128) half2 k[HALF2_N];
  __shared__ __align__(128) half2 a[HALF2_N];
  __shared__ __align__(128) half2 bvec[HALF2_N];

  for (int tt = 0; tt < T; ++tt) {
    const int64_t token =
        (static_cast<int64_t>(b_id) * T + tt) * C + h * N;
    if (warp == 0) {
      const int64_t idx = token + c0;
      r[lane] = __ldg(reinterpret_cast<const half2*>(r_ptr + idx));
      k[lane] = __ldg(reinterpret_cast<const half2*>(k_ptr + idx));
      a[lane] = __ldg(reinterpret_cast<const half2*>(a_ptr + idx));
      bvec[lane] = __ldg(reinterpret_cast<const half2*>(b_ptr + idx));
      const half2 raw_w = __ldg(reinterpret_cast<const half2*>(w_ptr + idx));
      const int phase = elapsed_t[b_id] + h * N + c0 + tt;
      w[lane] = __halves2half2(
          w_delta_maybe_w0<AddW0>(raw_w.x, w0_ptr, h * N + c0, phase),
          w_delta_maybe_w0<AddW0>(raw_w.y, w0_ptr, h * N + c0 + 1, phase + 1));
    }
    __syncthreads();

    const half2 av = a[lane];
    const half2 product = __hmul2(state, av);
    float sa = __half2float(product.x) + __half2float(product.y);
    sa = warp_sum_float(sa);
    sa = __shfl_sync(0xffffffffu, sa, 0);
    const half2 sa2 = __float2half2_rn(sa);
    const half vv = __ldg(v_ptr + token + row);
    const half2 vv2 = __halves2half2(vv, vv);
    state = __hfma2(
        state,
        w[lane],
        __hfma2(k[lane], vv2, __hfma2(sa2, bvec[lane], state)));

    const half2 yr = __hmul2(state, r[lane]);
    float yy = __half2float(yr.x) + __half2float(yr.y);
    yy = warp_sum_float(yy);
    if (lane == 0) {
      y_ptr[token + row] = __float2half_rn(yy);
    }
    // Every CTA reuses the shared token operands for the next recurrent step.
    // Removing this barrier lets warp 0 overwrite data while peer row owners
    // still consume it, which is a real cross-warp race for T > 1.
    __syncthreads();
  }

  state_base[c0 * N + row] = state.x;
  state_base[(c0 + 1) * N + row] = state.y;
}

template <bool Tis1 = false, bool AddW0 = false, bool Grid2D = false>
__global__ __launch_bounds__(N, 2) void wkv_fp16_v1_exact_kernel(
    const int B,
    const int T,
    const int C,
    const int H,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  if constexpr (Tis1) {
    __builtin_assume(T == 1);
  }
  int b_id;
  int h;
  decode_batch_head<Grid2D>(H, b_id, h);
  const int i = threadIdx.x;
  const int lane = i % 32;

  state_ptr += b_id * C * N + h * N * N;

  half2 state[HALF2_N];
#pragma unroll
  for (int j = 0; j < HALF2_N; j++) {
    state[j] = load_state_kv(state_ptr, i, j);
  }

  __shared__ __align__(128) half2 r[HALF2_N], k[HALF2_N], w[HALF2_N], a[HALF2_N], bvec[HALF2_N], bvec_dummy[HALF2_N];
#pragma unroll
  for (int tt = 0; tt < T; tt++) {
    int t = b_id * T * C + h * N + tt * C;
    __syncthreads();
    cp_async<4>((half2*)(i < 32 ? w : a) + lane, (half2*)((i < 32 ? w_ptr : a_ptr) + t) + lane, true);
    cp_commit();
    cp_async<4>((half2*)(i < 32 ? r : k) + lane, (half2*)((i < 32 ? r_ptr : k_ptr) + t) + lane, true);
    // Do not merge bvec_dummy into bvec: warp 1's src-size 0 copy writes zeros.
    cp_async<4>((i < 32 ? bvec : bvec_dummy) + lane, (half2*)(b_ptr + t) + lane, i < 32);
    cp_commit();

    half vv = v_ptr[t + i];
    half2 vv2 = {vv, vv};
    half2 y2 = {0.0, 0.0};
    half2 sa2 = {0.0, 0.0};
    cp_wait<1>();
    __syncthreads();
#pragma unroll
    for (int j = 0; j < HALF2_N; j++) {
      sa2 = __hfma2(a[j], state[j], sa2);
    }
    half sa = sa2.x + sa2.y;
    sa2 = {sa, sa};
    ((half*)w)[i] = w_delta_maybe_w0<AddW0>(((half*)w)[i], w0_ptr, h * N + i, elapsed_t[b_id] + h * N + i + tt);

    cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int j = 0; j < HALF2_N; j++) {
      half2& s = state[j];
      s = __hfma2(s, w[j], __hfma2(k[j], vv2, __hfma2(sa2, bvec[j], s)));
      y2 = __hfma2(s, r[j], y2);
    }
    y_ptr[t + i] = y2.x + y2.y;
  }

#pragma unroll
  for (int j = 0; j < HALF2_N; j++) {
    store_state_kv(state_ptr, i, j, state[j]);
  }
}

template <bool AddW0 = false, bool Grid2D = false>
__global__ __launch_bounds__(N, 2) void wkv_fp16_seq_v2_kernel(
    int T,
    int C,
    int H,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  int b_id;
  int h;
  decode_batch_head<Grid2D>(H, b_id, h);
  const int i = threadIdx.x;
  const int lane = i & 31;

  state_ptr += static_cast<int64_t>(b_id) * C * N + h * N * N;

  half2 state[HALF2_N];
#pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    state[j] = load_state_kv(state_ptr, i, j);
  }

  __shared__ __align__(128) half2 r[2][HALF2_N], w[2][HALF2_N], k[2][HALF2_N], a[2][HALF2_N], bvec[2][HALF2_N], bvec_dummy[HALF2_N];
  int token = (b_id * T) * C + h * N;
  prefetch_token(i, lane, token, r[0], w[0], k[0], a[0], bvec[0], bvec_dummy, r_ptr, w_ptr, k_ptr, a_ptr, b_ptr);

  for (int tt = 0; tt < T; ++tt) {
    const int cur = tt & 1;
    cp_wait<0>();
    __syncthreads();

    half2 sa2 = {0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < HALF2_N; ++j) {
      sa2 = __hfma2(a[cur][j], state[j], sa2);
    }
    half sa = sa2.x + sa2.y;
    sa2 = {sa, sa};
    ((half*)w[cur])[i] = w_delta_maybe_w0<AddW0>(((half*)w[cur])[i], w0_ptr, h * N + i, elapsed_t[b_id] + h * N + i + tt);
    __syncthreads();

    if (tt + 1 < T) {
      int next_token = token + C;
      prefetch_token(i, lane, next_token, r[cur ^ 1], w[cur ^ 1], k[cur ^ 1], a[cur ^ 1], bvec[cur ^ 1], bvec_dummy, r_ptr, w_ptr, k_ptr, a_ptr, b_ptr);
    }

    half vv = v_ptr[token + i];
    half2 vv2 = {vv, vv};
    half2 y2 = {0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < HALF2_N; ++j) {
      half2 s = state[j];
      s = __hfma2(s, w[cur][j], __hfma2(k[cur][j], vv2, __hfma2(sa2, bvec[cur][j], s)));
      state[j] = s;
      y2 = __hfma2(s, r[cur][j], y2);
    }
    y_ptr[token + i] = y2.x + y2.y;
    token += C;
  }

#pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    store_state_kv(state_ptr, i, j, state[j]);
  }
}

template <bool AddW0 = false, bool Grid2D = false>
__global__ __launch_bounds__(N, 1) void wkv_fp16_one_direct_kernel(
    int C,
    int H,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  int b_id;
  int h;
  decode_batch_head<Grid2D>(H, b_id, h);
  const int i = threadIdx.x;
  const int lane = i & 31;

  half* state_base = state_ptr + static_cast<int64_t>(b_id) * C * N + h * N * N;

  half2 state[HALF2_N];
#pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    state[j] = load_state_kv(state_base, i, j);
  }

  __shared__ __align__(128) half2 r[HALF2_N], w[HALF2_N], k[HALF2_N], a[HALF2_N], bvec[HALF2_N];
  const int token = b_id * C + h * N;
  if (i < HALF2_N) {
    const int idx2 = (token >> 1) + i;
    r[i] = __ldg(reinterpret_cast<const half2*>(r_ptr) + idx2);
    w[i] = __ldg(reinterpret_cast<const half2*>(w_ptr) + idx2);
    k[i] = __ldg(reinterpret_cast<const half2*>(k_ptr) + idx2);
    a[i] = __ldg(reinterpret_cast<const half2*>(a_ptr) + idx2);
    bvec[i] = __ldg(reinterpret_cast<const half2*>(b_ptr) + idx2);
  }
  __syncthreads();

  half2 sa2 = {0.0f, 0.0f};
#pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    sa2 = __hfma2(a[j], state[j], sa2);
  }
  half sa = sa2.x + sa2.y;
  sa2 = {sa, sa};
  ((half*)w)[i] = w_delta_maybe_w0<AddW0>(((half*)w)[i], w0_ptr, h * N + i, elapsed_t[b_id] + h * N + i);
  __syncthreads();

  half vv = __ldg(v_ptr + token + i);
  half2 vv2 = {vv, vv};
  half2 y2 = {0.0f, 0.0f};
#pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    half2 s = state[j];
    s = __hfma2(s, w[j], __hfma2(k[j], vv2, __hfma2(sa2, bvec[j], s)));
    state[j] = s;
    y2 = __hfma2(s, r[j], y2);
  }
  y_ptr[token + i] = y2.x + y2.y;

#pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    store_state_kv(state_base, i, j, state[j]);
  }
}

template <bool AddW0 = false, bool Grid2D = false>
__global__ __launch_bounds__(N, 1) void wkv_fp16_one_cp_kernel(
    int C,
    int H,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  int b_id;
  int h;
  decode_batch_head<Grid2D>(H, b_id, h);
  const int i = threadIdx.x;
  const int lane = i & 31;

  half* state_base = state_ptr + static_cast<int64_t>(b_id) * C * N + h * N * N;

  half2 state[HALF2_N];
#pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    state[j] = load_state_kv(state_base, i, j);
  }

  __shared__ __align__(128) half2 r[HALF2_N], w[HALF2_N], k[HALF2_N], a[HALF2_N], bvec[HALF2_N], bvec_dummy[HALF2_N];
  const int token = b_id * C + h * N;
  cp_async<4>((half2*)(i < 32 ? w : a) + lane, (half2*)((i < 32 ? w_ptr : a_ptr) + token) + lane, true);
  cp_commit();
  cp_async<4>((half2*)(i < 32 ? r : k) + lane, (half2*)((i < 32 ? r_ptr : k_ptr) + token) + lane, true);
  // The dummy allocation is correctness-critical, not padding; see exact/seq.
  cp_async<4>((i < 32 ? bvec : bvec_dummy) + lane, (half2*)(b_ptr + token) + lane, i < 32);
  cp_commit();

  half vv = __ldg(v_ptr + token + i);
  half2 vv2 = {vv, vv};
  half2 sa2 = {0.0f, 0.0f};
  cp_wait<1>();
  __syncthreads();
#pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    sa2 = __hfma2(a[j], state[j], sa2);
  }
  half sa = sa2.x + sa2.y;
  sa2 = {sa, sa};
  ((half*)w)[i] = w_delta_maybe_w0<AddW0>(((half*)w)[i], w0_ptr, h * N + i, elapsed_t[b_id] + h * N + i);

  cp_wait<0>();
  __syncthreads();
  half2 y2 = {0.0f, 0.0f};
#pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    half2 s = state[j];
    s = __hfma2(s, w[j], __hfma2(k[j], vv2, __hfma2(sa2, bvec[j], s)));
    state[j] = s;
    y2 = __hfma2(s, r[j], y2);
  }
  y_ptr[token + i] = y2.x + y2.y;

#pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    store_state_kv(state_base, i, j, state[j]);
  }
}

template <bool AddW0 = false, bool Grid2D = false, bool StreamState = false>
__global__ __launch_bounds__(32, 8) void wkv_fp16_kv_warp_pair_kernel(
    int T,
    int C,
    int H,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  int batch;
  int head;
  decode_batch_head<Grid2D>(H, batch, head);
  const int lane = threadIdx.x;
  const int v0 = lane * 2;
  half* const state_base =
      state_ptr + (static_cast<int64_t>(batch) * H + head) * N * N;

  // Each lane owns two adjacent V columns.  One half2 load per K row gives a
  // full 128-byte coalesced warp request, halving state requests versus the
  // two-warp scalar KV compatibility path.
  half2 state[N];
#pragma unroll
  for (int key = 0; key < N; ++key) {
    const half2* const state_row =
        reinterpret_cast<const half2*>(state_base + key * N) + lane;
#if RWKV_FP16_KV_STREAM_T1_STATE
    // T1 reads each state line once.  Evict-first avoids retaining this large
    // stream over the weights consumed by the following full-model kernels.
    if constexpr (StreamState) {
      state[key] = __ldcs(state_row);
    } else {
      state[key] = __ldg(state_row);
    }
#else
    state[key] = __ldg(state_row);
#endif
  }

  __shared__ __align__(128) half2 r[HALF2_N];
  __shared__ __align__(128) half2 w[HALF2_N];
  __shared__ __align__(128) half2 k[HALF2_N];
  __shared__ __align__(128) half2 a[HALF2_N];
  __shared__ __align__(128) half2 bvec[HALF2_N];

  for (int tt = 0; tt < T; ++tt) {
    const int64_t token =
        (static_cast<int64_t>(batch) * T + tt) * C + head * N;
    const int64_t idx2 = (token >> 1) + lane;
    r[lane] = __ldg(reinterpret_cast<const half2*>(r_ptr) + idx2);
    k[lane] = __ldg(reinterpret_cast<const half2*>(k_ptr) + idx2);
    a[lane] = __ldg(reinterpret_cast<const half2*>(a_ptr) + idx2);
    bvec[lane] = __ldg(reinterpret_cast<const half2*>(b_ptr) + idx2);
    const half2 raw_w = __ldg(reinterpret_cast<const half2*>(w_ptr) + idx2);
    const int phase = elapsed_t[batch] + head * N + 2 * lane + tt;
    w[lane] = __halves2half2(
        w_delta_maybe_w0<AddW0>(raw_w.x, w0_ptr, head * N + 2 * lane, phase),
        w_delta_maybe_w0<AddW0>(raw_w.y, w0_ptr, head * N + 2 * lane + 1, phase + 1));
    __syncwarp();

    half2 sa_even = __float2half2_rn(0.0f);
    half2 sa_odd = __float2half2_rn(0.0f);
#pragma unroll
    for (int key2 = 0; key2 < HALF2_N; ++key2) {
      const half2 av = a[key2];
      sa_even = __hfma2(state[2 * key2], __halves2half2(av.x, av.x), sa_even);
      sa_odd = __hfma2(state[2 * key2 + 1], __halves2half2(av.y, av.y), sa_odd);
    }
    const half2 sa = __hadd2(sa_even, sa_odd);
    const half2 vv = __ldg(reinterpret_cast<const half2*>(v_ptr) + idx2);
    half2 y_even = __float2half2_rn(0.0f);
    half2 y_odd = __float2half2_rn(0.0f);
#pragma unroll
    for (int key2 = 0; key2 < HALF2_N; ++key2) {
      const half2 rv = r[key2];
      const half2 wv = w[key2];
      const half2 kv = k[key2];
      const half2 bv = bvec[key2];
      half2& even = state[2 * key2];
      half2& odd = state[2 * key2 + 1];
      even = __hfma2(
          even, __halves2half2(wv.x, wv.x),
          __hfma2(__halves2half2(kv.x, kv.x), vv,
                  __hfma2(sa, __halves2half2(bv.x, bv.x), even)));
      odd = __hfma2(
          odd, __halves2half2(wv.y, wv.y),
          __hfma2(__halves2half2(kv.y, kv.y), vv,
                  __hfma2(sa, __halves2half2(bv.y, bv.y), odd)));
      y_even = __hfma2(even, __halves2half2(rv.x, rv.x), y_even);
      y_odd = __hfma2(odd, __halves2half2(rv.y, rv.y), y_odd);
    }
    reinterpret_cast<half2*>(y_ptr)[idx2] = __hadd2(y_even, y_odd);
    // All lanes consume the shared token before the next iteration overwrites
    // it.  Keep this consumer barrier if the loop is later software-pipelined.
    __syncwarp();
  }

#pragma unroll
  for (int key = 0; key < N; ++key) {
    half2* const state_row =
        reinterpret_cast<half2*>(state_base + key * N) + lane;
#if RWKV_FP16_KV_STREAM_T1_STATE
    if constexpr (StreamState) {
      __stcs(state_row, state[key]);
    } else {
      *state_row = state[key];
    }
#else
    *state_row = state[key];
#endif
  }
}

template <int KeepKeys, bool AddW0 = false, bool Grid2D = false>
__global__ __launch_bounds__(32, 8) void wkv_fp16_kv_warp_spill_kernel(
    int T,
    int C,
    int H,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  static_assert(KeepKeys == 16 || KeepKeys == 32 || KeepKeys == 48);
  int batch;
  int head;
  decode_batch_head<Grid2D>(H, batch, head);
  const int lane = threadIdx.x;
  half* const state_base =
      state_ptr + (static_cast<int64_t>(batch) * H + head) * N * N;

  half2 state[KeepKeys];
  __shared__ __align__(128) half2 spilled_state[N - KeepKeys][32];
#pragma unroll
  for (int key = 0; key < KeepKeys; ++key) {
    state[key] = __ldg(
        reinterpret_cast<const half2*>(state_base + key * N) + lane);
  }
#pragma unroll
  for (int key = KeepKeys; key < N; ++key) {
    spilled_state[key - KeepKeys][lane] = __ldg(
        reinterpret_cast<const half2*>(state_base + key * N) + lane);
  }

  __shared__ __align__(128) half2 r[HALF2_N];
  __shared__ __align__(128) half2 w[HALF2_N];
  __shared__ __align__(128) half2 k[HALF2_N];
  __shared__ __align__(128) half2 a[HALF2_N];
  __shared__ __align__(128) half2 bvec[HALF2_N];

  for (int tt = 0; tt < T; ++tt) {
    const int64_t token =
        (static_cast<int64_t>(batch) * T + tt) * C + head * N;
    const int64_t idx2 = (token >> 1) + lane;
    r[lane] = __ldg(reinterpret_cast<const half2*>(r_ptr) + idx2);
    k[lane] = __ldg(reinterpret_cast<const half2*>(k_ptr) + idx2);
    a[lane] = __ldg(reinterpret_cast<const half2*>(a_ptr) + idx2);
    bvec[lane] = __ldg(reinterpret_cast<const half2*>(b_ptr) + idx2);
    const half2 raw_w = __ldg(reinterpret_cast<const half2*>(w_ptr) + idx2);
    const int phase = elapsed_t[batch] + head * N + 2 * lane + tt;
    w[lane] = __halves2half2(
        w_delta_maybe_w0<AddW0>(raw_w.x, w0_ptr, head * N + 2 * lane, phase),
        w_delta_maybe_w0<AddW0>(raw_w.y, w0_ptr, head * N + 2 * lane + 1, phase + 1));
    __syncwarp();

    half2 sa_even = __float2half2_rn(0.0f);
    half2 sa_odd = __float2half2_rn(0.0f);
#pragma unroll
    for (int key2 = 0; key2 < KeepKeys / 2; ++key2) {
      const half2 av = a[key2];
      sa_even = __hfma2(state[2 * key2], __halves2half2(av.x, av.x), sa_even);
      sa_odd = __hfma2(state[2 * key2 + 1], __halves2half2(av.y, av.y), sa_odd);
    }
#pragma unroll
    for (int key2 = KeepKeys / 2; key2 < HALF2_N; ++key2) {
      const half2 av = a[key2];
      const int spilled_key = 2 * key2 - KeepKeys;
      sa_even = __hfma2(
          spilled_state[spilled_key][lane], __halves2half2(av.x, av.x), sa_even);
      sa_odd = __hfma2(
          spilled_state[spilled_key + 1][lane], __halves2half2(av.y, av.y), sa_odd);
    }
    const half2 sa = __hadd2(sa_even, sa_odd);
    const half2 vv = __ldg(reinterpret_cast<const half2*>(v_ptr) + idx2);
    half2 y_even = __float2half2_rn(0.0f);
    half2 y_odd = __float2half2_rn(0.0f);
#pragma unroll
    for (int key2 = 0; key2 < KeepKeys / 2; ++key2) {
      const half2 rv = r[key2];
      const half2 wv = w[key2];
      const half2 kv = k[key2];
      const half2 bv = bvec[key2];
      half2& even = state[2 * key2];
      half2& odd = state[2 * key2 + 1];
      even = __hfma2(
          even, __halves2half2(wv.x, wv.x),
          __hfma2(__halves2half2(kv.x, kv.x), vv,
                  __hfma2(sa, __halves2half2(bv.x, bv.x), even)));
      odd = __hfma2(
          odd, __halves2half2(wv.y, wv.y),
          __hfma2(__halves2half2(kv.y, kv.y), vv,
                  __hfma2(sa, __halves2half2(bv.y, bv.y), odd)));
      y_even = __hfma2(even, __halves2half2(rv.x, rv.x), y_even);
      y_odd = __hfma2(odd, __halves2half2(rv.y, rv.y), y_odd);
    }
#pragma unroll
    for (int key2 = KeepKeys / 2; key2 < HALF2_N; ++key2) {
      const half2 rv = r[key2];
      const half2 wv = w[key2];
      const half2 kv = k[key2];
      const half2 bv = bvec[key2];
      const int spilled_key = 2 * key2 - KeepKeys;
      half2 even = spilled_state[spilled_key][lane];
      half2 odd = spilled_state[spilled_key + 1][lane];
      even = __hfma2(
          even, __halves2half2(wv.x, wv.x),
          __hfma2(__halves2half2(kv.x, kv.x), vv,
                  __hfma2(sa, __halves2half2(bv.x, bv.x), even)));
      odd = __hfma2(
          odd, __halves2half2(wv.y, wv.y),
          __hfma2(__halves2half2(kv.y, kv.y), vv,
                  __hfma2(sa, __halves2half2(bv.y, bv.y), odd)));
      spilled_state[spilled_key][lane] = even;
      spilled_state[spilled_key + 1][lane] = odd;
      y_even = __hfma2(even, __halves2half2(rv.x, rv.x), y_even);
      y_odd = __hfma2(odd, __halves2half2(rv.y, rv.y), y_odd);
    }
    reinterpret_cast<half2*>(y_ptr)[idx2] = __hadd2(y_even, y_odd);
    __syncwarp();
  }

#pragma unroll
  for (int key = 0; key < KeepKeys; ++key) {
    reinterpret_cast<half2*>(state_base + key * N)[lane] = state[key];
  }
#pragma unroll
  for (int key = KeepKeys; key < N; ++key) {
    reinterpret_cast<half2*>(state_base + key * N)[lane] =
        spilled_state[key - KeepKeys][lane];
  }
}

template <bool AddW0 = false, bool Grid2D = false, bool StreamState = false>
__global__ __launch_bounds__(N, 2) void wkv_fp16_kv_vector_kernel(
    int T,
    int C,
    int H,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  int batch;
  int head;
  decode_batch_head<Grid2D>(H, batch, head);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  half* const state_base =
      state_ptr + (static_cast<int64_t>(batch) * H + head) * N * N;
  __shared__ __align__(256) half2 state_smem[N][HALF2_N];

  // Pair adjacent K rows before transposing. This mapping is performance- and
  // correctness-critical: handling one K row at a time doubles the shared
  // instruction count because each destination register is a K-pair half2.
#pragma unroll
  for (int pair_segment = tid; pair_segment < HALF2_N * (N / LDG_ELEMS);
       pair_segment += N) {
    const int key2 = pair_segment >> 3;
    const int segment = pair_segment & 7;
    const int4* const lo_ptr = reinterpret_cast<const int4*>(
        state_base + (2 * key2) * N) + segment;
    const int4* const hi_ptr = reinterpret_cast<const int4*>(
        state_base + (2 * key2 + 1) * N) + segment;
#if RWKV_FP16_KV_STREAM_T1_STATE
    const int4 lo = StreamState ? __ldcs(lo_ptr) : __ldg(lo_ptr);
    const int4 hi = StreamState ? __ldcs(hi_ptr) : __ldg(hi_ptr);
#else
    const int4 lo = __ldg(lo_ptr);
    const int4 hi = __ldg(hi_ptr);
#endif
    const int value0 = segment * LDG_ELEMS;
#pragma unroll
    for (int q = 0; q < LDG_ELEMS; ++q) {
      const int value = value0 + q;
      state_smem[value][(value & 31) ^ key2] = __halves2half2(
          reinterpret_cast<const half*>(&lo)[q],
          reinterpret_cast<const half*>(&hi)[q]);
    }
  }
  __syncthreads();

  half2 state[HALF2_N];
#pragma unroll
  for (int key2 = 0; key2 < HALF2_N; ++key2) {
    state[key2] = state_smem[tid][lane ^ key2];
  }

  __shared__ __align__(128) half2 r[HALF2_N], w[HALF2_N], k[HALF2_N];
  __shared__ __align__(128) half2 a[HALF2_N], bvec[HALF2_N], bvec_dummy[HALF2_N];
  for (int tt = 0; tt < T; ++tt) {
    const int64_t token =
        (static_cast<int64_t>(batch) * T + tt) * C + head * N;
    __syncthreads();
    cp_async<4>((tid < 32 ? w : a) + lane,
                reinterpret_cast<const half2*>(tid < 32 ? w_ptr + token : a_ptr + token) + lane,
                true);
    cp_commit();
    cp_async<4>((tid < 32 ? r : k) + lane,
                reinterpret_cast<const half2*>(tid < 32 ? r_ptr + token : k_ptr + token) + lane,
                true);
    cp_async<4>((tid < 32 ? bvec : bvec_dummy) + lane,
                reinterpret_cast<const half2*>(b_ptr + token) + lane, tid < 32);
    cp_commit();

    const half vv = __ldg(v_ptr + token + tid);
    const half2 vv2 = __halves2half2(vv, vv);
    half2 sa2 = __float2half2_rn(0.0f);
    cp_wait<1>();
    __syncthreads();
#pragma unroll
    for (int key2 = 0; key2 < HALF2_N; ++key2) {
      sa2 = __hfma2(a[key2], state[key2], sa2);
    }
    const half sa = sa2.x + sa2.y;
    sa2 = __halves2half2(sa, sa);
    reinterpret_cast<half*>(w)[tid] = w_delta_maybe_w0<AddW0>(
        reinterpret_cast<half*>(w)[tid], w0_ptr, head * N + tid,
        elapsed_t[batch] + head * N + tid + tt);
    cp_wait<0>();
    __syncthreads();

    half2 y2 = __float2half2_rn(0.0f);
#pragma unroll
    for (int key2 = 0; key2 < HALF2_N; ++key2) {
      half2& s = state[key2];
      s = __hfma2(s, w[key2],
                  __hfma2(k[key2], vv2, __hfma2(sa2, bvec[key2], s)));
      y2 = __hfma2(s, r[key2], y2);
    }
    y_ptr[token + tid] = y2.x + y2.y;
  }

#pragma unroll
  for (int key2 = 0; key2 < HALF2_N; ++key2) {
    state_smem[tid][lane ^ key2] = state[key2];
  }
  __syncthreads();
#pragma unroll
  for (int pair_segment = tid; pair_segment < HALF2_N * (N / LDG_ELEMS);
       pair_segment += N) {
    int4 lo;
    int4 hi;
    const int key2 = pair_segment >> 3;
    const int segment = pair_segment & 7;
    const int value0 = segment * LDG_ELEMS;
#pragma unroll
    for (int q = 0; q < LDG_ELEMS; ++q) {
      const int value = value0 + q;
      const half2 packed = state_smem[value][(value & 31) ^ key2];
      reinterpret_cast<half*>(&lo)[q] = packed.x;
      reinterpret_cast<half*>(&hi)[q] = packed.y;
    }
    int4* const lo_ptr = reinterpret_cast<int4*>(
        state_base + (2 * key2) * N) + segment;
    int4* const hi_ptr = reinterpret_cast<int4*>(
        state_base + (2 * key2 + 1) * N) + segment;
#if RWKV_FP16_KV_STREAM_T1_STATE
    if constexpr (StreamState) {
      __stcs(lo_ptr, lo);
      __stcs(hi_ptr, hi);
    } else {
      *lo_ptr = lo;
      *hi_ptr = hi;
    }
#else
    *lo_ptr = lo;
    *hi_ptr = hi;
#endif
  }
}

template <bool AddW0 = false, bool Grid2D = false>
__global__ __launch_bounds__(N, 2) void wkv_fp16_kv_staged_kernel(
    int T,
    int C,
    int H,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  int batch;
  int head;
  decode_batch_head<Grid2D>(H, batch, head);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  half* const state_base =
      state_ptr + (static_cast<int64_t>(batch) * H + head) * N * N;
  __shared__ __align__(256) half state_smem[N][N];

  // Preserve the global [K,V] int4 stream. The earlier vector candidate
  // unpacked every int4 into eight scalar shared stores while transposing it;
  // that overhead erased the DRAM benefit. Here V-owner threads read native K
  // rows coalesced from shared memory while retaining the proven arithmetic.
#pragma unroll
  for (int flat4 = tid; flat4 < N * N / LDG_ELEMS; flat4 += N) {
    reinterpret_cast<int4*>(state_smem)[flat4] =
        __ldg(reinterpret_cast<const int4*>(state_base) + flat4);
  }
  __syncthreads();

  half2 state[HALF2_N];
#pragma unroll
  for (int key2 = 0; key2 < HALF2_N; ++key2) {
    state[key2] = __halves2half2(
        state_smem[2 * key2][tid], state_smem[2 * key2 + 1][tid]);
  }

  __shared__ __align__(128) half2 r[HALF2_N], w[HALF2_N], k[HALF2_N];
  __shared__ __align__(128) half2 a[HALF2_N], bvec[HALF2_N], bvec_dummy[HALF2_N];
  for (int tt = 0; tt < T; ++tt) {
    const int64_t token =
        (static_cast<int64_t>(batch) * T + tt) * C + head * N;
    __syncthreads();
    cp_async<4>((tid < 32 ? w : a) + lane,
                reinterpret_cast<const half2*>(tid < 32 ? w_ptr + token : a_ptr + token) + lane,
                true);
    cp_commit();
    cp_async<4>((tid < 32 ? r : k) + lane,
                reinterpret_cast<const half2*>(tid < 32 ? r_ptr + token : k_ptr + token) + lane,
                true);
    cp_async<4>((tid < 32 ? bvec : bvec_dummy) + lane,
                reinterpret_cast<const half2*>(b_ptr + token) + lane, tid < 32);
    cp_commit();

    const half vv = __ldg(v_ptr + token + tid);
    const half2 vv2 = __halves2half2(vv, vv);
    half2 sa2 = __float2half2_rn(0.0f);
    cp_wait<1>();
    __syncthreads();
#pragma unroll
    for (int key2 = 0; key2 < HALF2_N; ++key2) {
      sa2 = __hfma2(a[key2], state[key2], sa2);
    }
    const half sa = sa2.x + sa2.y;
    sa2 = __halves2half2(sa, sa);
    reinterpret_cast<half*>(w)[tid] = w_delta_maybe_w0<AddW0>(
        reinterpret_cast<half*>(w)[tid], w0_ptr, head * N + tid,
        elapsed_t[batch] + head * N + tid + tt);
    cp_wait<0>();
    __syncthreads();

    half2 y2 = __float2half2_rn(0.0f);
#pragma unroll
    for (int key2 = 0; key2 < HALF2_N; ++key2) {
      half2& s = state[key2];
      s = __hfma2(s, w[key2],
                  __hfma2(k[key2], vv2, __hfma2(sa2, bvec[key2], s)));
      y2 = __hfma2(s, r[key2], y2);
    }
    y_ptr[token + tid] = y2.x + y2.y;
  }

#pragma unroll
  for (int key2 = 0; key2 < HALF2_N; ++key2) {
    state_smem[2 * key2][tid] = state[key2].x;
    state_smem[2 * key2 + 1][tid] = state[key2].y;
  }
  __syncthreads();
#pragma unroll
  for (int flat4 = tid; flat4 < N * N / LDG_ELEMS; flat4 += N) {
    reinterpret_cast<int4*>(state_base)[flat4] =
        reinterpret_cast<const int4*>(state_smem)[flat4];
  }
}

bool use_v2_seq(int B, int T) {
  return (B == 1 && T >= 8) ||
         (B == 4 && T >= 4) ||
         (B == 8 && T >= 8) ||
         (B == 64 && T == 1) ||
         (B == 128 && T == 1);
}

bool use_kv_vector_auto(int B, int T, int H) {
  // The int4/shared transpose only amortizes at T1 once the state stream is at
  // least about 160 MiB.  All H32/H40/H64 points above this boundary were
  // positive on GPU1/2/3 in the two-address gate; do not reuse this rule for
  // T>1, where recurrent work amortizes the direct KV load/store instead.
  return T == 1 && static_cast<int64_t>(B) * H >= 20000;
}

bool use_kv_warp_auto(int B, int T, int H) {
  if (T == 1) {
    // The shared-address multimode gate showed that H16/B960, H32/B512,
    // H40/B384, and H64/B256 all cross over in the same B*H band.  The vector
    // branch above owns B*H >= 20000; one-warp owns the narrower 15000..19999
    // interval without binding dispatch to a particular model width.
    return static_cast<int64_t>(B) * H >= 15000;
  }
  // With enough independent (batch,head) CTAs, the one-warp kernel retains
  // occupancy while halving state memory requests.  Every measured point in
  // this region (T2..T32, H12..H64) was positive on all three GPUs.
  return static_cast<int64_t>(B) * H >= 1280;
}

}  // namespace

void wkv_one_v2_cuda_impl(
    int B,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    const half* w0_ptr,
    bool add_w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t,
    bool grid2d = false);

template <bool Grid2D>
void launch_wkv_seq_v2_cuda_impl(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    const half* w0_ptr,
    bool add_w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t,
    int forced_mode) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (forced_mode < 0 && use_kv_vector_auto(B, T, H)) {
    if (add_w0) {
      wkv_fp16_kv_vector_kernel<true, false, true><<<dim3(B * H), dim3(N), 0, stream>>>(
          T, C, H, reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()), w0_ptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
    } else {
      wkv_fp16_kv_vector_kernel<false, false, true><<<dim3(B * H), dim3(N), 0, stream>>>(
          T, C, H, reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()), nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (forced_mode < 0 && use_kv_warp_auto(B, T, H)) {
    // Always use the measured 2D mapping here.  This branch deliberately owns
    // its launch layout instead of inheriting the caller's flat/2D preference.
    if (add_w0) {
      if (T == 1) {
        wkv_fp16_kv_warp_pair_kernel<true, true, true><<<dim3(H, B), dim3(32), 0, stream>>>(
            T, C, H, reinterpret_cast<half*>(state.data_ptr()),
            reinterpret_cast<const half*>(r.data_ptr()),
            reinterpret_cast<const half*>(w.data_ptr()), w0_ptr,
            reinterpret_cast<const half*>(k.data_ptr()),
            reinterpret_cast<const half*>(v.data_ptr()),
            reinterpret_cast<const half*>(a.data_ptr()),
            reinterpret_cast<const half*>(b.data_ptr()),
            reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
      } else {
        wkv_fp16_kv_warp_pair_kernel<true, true, false><<<dim3(H, B), dim3(32), 0, stream>>>(
            T, C, H, reinterpret_cast<half*>(state.data_ptr()),
            reinterpret_cast<const half*>(r.data_ptr()),
            reinterpret_cast<const half*>(w.data_ptr()), w0_ptr,
            reinterpret_cast<const half*>(k.data_ptr()),
            reinterpret_cast<const half*>(v.data_ptr()),
            reinterpret_cast<const half*>(a.data_ptr()),
            reinterpret_cast<const half*>(b.data_ptr()),
            reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
      }
    } else {
      if (T == 1) {
        wkv_fp16_kv_warp_pair_kernel<false, true, true><<<dim3(H, B), dim3(32), 0, stream>>>(
            T, C, H, reinterpret_cast<half*>(state.data_ptr()),
            reinterpret_cast<const half*>(r.data_ptr()),
            reinterpret_cast<const half*>(w.data_ptr()), nullptr,
            reinterpret_cast<const half*>(k.data_ptr()),
            reinterpret_cast<const half*>(v.data_ptr()),
            reinterpret_cast<const half*>(a.data_ptr()),
            reinterpret_cast<const half*>(b.data_ptr()),
            reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
      } else {
        wkv_fp16_kv_warp_pair_kernel<false, true, false><<<dim3(H, B), dim3(32), 0, stream>>>(
            T, C, H, reinterpret_cast<half*>(state.data_ptr()),
            reinterpret_cast<const half*>(r.data_ptr()),
            reinterpret_cast<const half*>(w.data_ptr()), nullptr,
            reinterpret_cast<const half*>(k.data_ptr()),
            reinterpret_cast<const half*>(v.data_ptr()),
            reinterpret_cast<const half*>(a.data_ptr()),
            reinterpret_cast<const half*>(b.data_ptr()),
            reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
      }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (T == 1) {
    wkv_one_v2_cuda_impl(
        B, C, H, state, r, w, w0_ptr, add_w0, k, v, a, b, y, elapsed_t, Grid2D);
    return;
  }
  const dim3 grid = Grid2D ? dim3(H, B, 1) : dim3(B * H, 1, 1);
  const bool use_seq_v2 = forced_mode == 1 || (forced_mode < 0 && use_v2_seq(B, T));
  if (use_seq_v2) {
    if (add_w0) {
      wkv_fp16_seq_v2_kernel<true, Grid2D><<<grid, dim3(N), 0, stream>>>(
          T, C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()),
          w0_ptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()),
          elapsed_t.data_ptr<int>());
    } else {
      wkv_fp16_seq_v2_kernel<false, Grid2D><<<grid, dim3(N), 0, stream>>>(
          T, C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()),
          nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()),
          elapsed_t.data_ptr<int>());
    }
  } else {
    if (add_w0) {
      wkv_fp16_v1_exact_kernel<false, true, Grid2D><<<grid, dim3(N), 0, stream>>>(
          B, T, C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()),
          w0_ptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()),
          elapsed_t.data_ptr<int>());
    } else {
      wkv_fp16_v1_exact_kernel<false, false, Grid2D><<<grid, dim3(N), 0, stream>>>(
          B, T, C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()),
          nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()),
          elapsed_t.data_ptr<int>());
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wkv_seq_v2_cuda_impl(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    const half* w0_ptr,
    bool add_w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t,
    int forced_mode = -1,
    bool grid2d = false) {
  if (grid2d) {
    launch_wkv_seq_v2_cuda_impl<true>(
        B, T, C, H, state, r, w, w0_ptr, add_w0, k, v, a, b, y, elapsed_t, forced_mode);
  } else {
    launch_wkv_seq_v2_cuda_impl<false>(
        B, T, C, H, state, r, w, w0_ptr, add_w0, k, v, a, b, y, elapsed_t, forced_mode);
  }
}

void wkv_seq_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  wkv_seq_v2_cuda_impl(B, T, C, H, state, r, w, nullptr, false, k, v, a, b, y, elapsed_t);
}

void wkv_seq_w0_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  wkv_seq_v2_cuda_impl(B, T, C, H, state, r, w, reinterpret_cast<const half*>(w0.data_ptr()), true, k, v, a, b, y, elapsed_t);
}

void wkv_seq_grid2d_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  wkv_seq_v2_cuda_impl(
      B, T, C, H, state, r, w, nullptr, false, k, v, a, b, y, elapsed_t, -1, true);
}

void wkv_seq_w0_grid2d_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  wkv_seq_v2_cuda_impl(
      B, T, C, H, state, r, w, reinterpret_cast<const half*>(w0.data_ptr()), true,
      k, v, a, b, y, elapsed_t, -1, true);
}

void wkv_seq_grid2d_forced_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    int mode,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  wkv_seq_v2_cuda_impl(
      B, T, C, H, state, r, w, nullptr, false, k, v, a, b, y, elapsed_t, mode, true);
}

void wkv_seq_w0_grid2d_forced_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    int mode,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  wkv_seq_v2_cuda_impl(
      B, T, C, H, state, r, w, reinterpret_cast<const half*>(w0.data_ptr()), true,
      k, v, a, b, y, elapsed_t, mode, true);
}

void wkv_seq_forced_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    int mode,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  wkv_seq_v2_cuda_impl(B, T, C, H, state, r, w, nullptr, false, k, v, a, b, y, elapsed_t, mode);
}

void wkv_seq_w0_forced_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    int mode,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  wkv_seq_v2_cuda_impl(
      B, T, C, H, state, r, w, reinterpret_cast<const half*>(w0.data_ptr()), true,
      k, v, a, b, y, elapsed_t, mode);
}

void wkv_grouped_w0_forced_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    int rows_per_cta,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_GROUPED_ROWS(Rows) \
  wkv_fp16_grouped_rows_kernel<Rows, true><<<dim3(N / Rows, H, B), dim3(Rows * 32), 0, stream>>>( \
      T, C, H, reinterpret_cast<half*>(state.data_ptr()), \
      reinterpret_cast<const half*>(r.data_ptr()), reinterpret_cast<const half*>(w.data_ptr()), \
      reinterpret_cast<const half*>(w0.data_ptr()), reinterpret_cast<const half*>(k.data_ptr()), \
      reinterpret_cast<const half*>(v.data_ptr()), reinterpret_cast<const half*>(a.data_ptr()), \
      reinterpret_cast<const half*>(b.data_ptr()), reinterpret_cast<half*>(y.data_ptr()), \
      elapsed_t.data_ptr<int>())
  if (rows_per_cta == 4) {
    LAUNCH_GROUPED_ROWS(4);
  } else if (rows_per_cta == 8) {
    LAUNCH_GROUPED_ROWS(8);
  } else {
    assert(rows_per_cta == 16);
    LAUNCH_GROUPED_ROWS(16);
  }
#undef LAUNCH_GROUPED_ROWS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wkv_kv_warp_w0_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    wkv_fp16_kv_warp_pair_kernel<true, true, true><<<dim3(H, B, 1), dim3(32), 0, stream>>>(
        T, C, H, reinterpret_cast<half*>(state.data_ptr()),
        reinterpret_cast<const half*>(r.data_ptr()), reinterpret_cast<const half*>(w.data_ptr()),
        reinterpret_cast<const half*>(w0.data_ptr()), reinterpret_cast<const half*>(k.data_ptr()),
        reinterpret_cast<const half*>(v.data_ptr()), reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b.data_ptr()), reinterpret_cast<half*>(y.data_ptr()),
        elapsed_t.data_ptr<int>());
  } else {
    wkv_fp16_kv_warp_pair_kernel<true, true, false><<<dim3(H, B, 1), dim3(32), 0, stream>>>(
        T, C, H, reinterpret_cast<half*>(state.data_ptr()),
        reinterpret_cast<const half*>(r.data_ptr()), reinterpret_cast<const half*>(w.data_ptr()),
        reinterpret_cast<const half*>(w0.data_ptr()), reinterpret_cast<const half*>(k.data_ptr()),
        reinterpret_cast<const half*>(v.data_ptr()), reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b.data_ptr()), reinterpret_cast<half*>(y.data_ptr()),
        elapsed_t.data_ptr<int>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wkv_kv_warp_spill_w0_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    int keep_keys,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_WARP_SPILL(Keep) \
  wkv_fp16_kv_warp_spill_kernel<Keep, true, false><<<dim3(B * H), dim3(32), 0, stream>>>( \
      T, C, H, reinterpret_cast<half*>(state.data_ptr()), \
      reinterpret_cast<const half*>(r.data_ptr()), \
      reinterpret_cast<const half*>(w.data_ptr()), \
      reinterpret_cast<const half*>(w0.data_ptr()), \
      reinterpret_cast<const half*>(k.data_ptr()), \
      reinterpret_cast<const half*>(v.data_ptr()), \
      reinterpret_cast<const half*>(a.data_ptr()), \
      reinterpret_cast<const half*>(b.data_ptr()), \
      reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>())
  if (keep_keys == 16) {
    LAUNCH_WARP_SPILL(16);
  } else if (keep_keys == 32) {
    LAUNCH_WARP_SPILL(32);
  } else {
    assert(keep_keys == 48);
    LAUNCH_WARP_SPILL(48);
  }
#undef LAUNCH_WARP_SPILL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wkv_kv_vector_w0_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    wkv_fp16_kv_vector_kernel<true, true, true><<<dim3(H, B, 1), dim3(N), 0, stream>>>(
        T, C, H, reinterpret_cast<half*>(state.data_ptr()),
        reinterpret_cast<const half*>(r.data_ptr()), reinterpret_cast<const half*>(w.data_ptr()),
        reinterpret_cast<const half*>(w0.data_ptr()), reinterpret_cast<const half*>(k.data_ptr()),
        reinterpret_cast<const half*>(v.data_ptr()), reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b.data_ptr()), reinterpret_cast<half*>(y.data_ptr()),
        elapsed_t.data_ptr<int>());
  } else {
    wkv_fp16_kv_vector_kernel<true, true, false><<<dim3(H, B, 1), dim3(N), 0, stream>>>(
        T, C, H, reinterpret_cast<half*>(state.data_ptr()),
        reinterpret_cast<const half*>(r.data_ptr()), reinterpret_cast<const half*>(w.data_ptr()),
        reinterpret_cast<const half*>(w0.data_ptr()), reinterpret_cast<const half*>(k.data_ptr()),
        reinterpret_cast<const half*>(v.data_ptr()), reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b.data_ptr()), reinterpret_cast<half*>(y.data_ptr()),
        elapsed_t.data_ptr<int>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wkv_kv_vector_flat_w0_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  wkv_fp16_kv_vector_kernel<true, false><<<dim3(B * H), dim3(N), 0, stream>>>(
      T, C, H, reinterpret_cast<half*>(state.data_ptr()),
      reinterpret_cast<const half*>(r.data_ptr()),
      reinterpret_cast<const half*>(w.data_ptr()),
      reinterpret_cast<const half*>(w0.data_ptr()),
      reinterpret_cast<const half*>(k.data_ptr()),
      reinterpret_cast<const half*>(v.data_ptr()),
      reinterpret_cast<const half*>(a.data_ptr()),
      reinterpret_cast<const half*>(b.data_ptr()),
      reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wkv_kv_staged_w0_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  wkv_fp16_kv_staged_kernel<true, true><<<dim3(H, B, 1), dim3(N), 0, stream>>>(
      T, C, H, reinterpret_cast<half*>(state.data_ptr()),
      reinterpret_cast<const half*>(r.data_ptr()),
      reinterpret_cast<const half*>(w.data_ptr()),
      reinterpret_cast<const half*>(w0.data_ptr()),
      reinterpret_cast<const half*>(k.data_ptr()),
      reinterpret_cast<const half*>(v.data_ptr()),
      reinterpret_cast<const half*>(a.data_ptr()),
      reinterpret_cast<const half*>(b.data_ptr()),
      reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wkv_kv_staged_flat_w0_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  wkv_fp16_kv_staged_kernel<true, false><<<dim3(B * H), dim3(N), 0, stream>>>(
      T, C, H, reinterpret_cast<half*>(state.data_ptr()),
      reinterpret_cast<const half*>(r.data_ptr()),
      reinterpret_cast<const half*>(w.data_ptr()),
      reinterpret_cast<const half*>(w0.data_ptr()),
      reinterpret_cast<const half*>(k.data_ptr()),
      reinterpret_cast<const half*>(v.data_ptr()),
      reinterpret_cast<const half*>(a.data_ptr()),
      reinterpret_cast<const half*>(b.data_ptr()),
      reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wkv_one_v2_cuda(
    int B,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  wkv_one_v2_cuda_impl(B, C, H, state, r, w, nullptr, false, k, v, a, b, y, elapsed_t);
}

void wkv_one_w0_v2_cuda(
    int B,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  wkv_one_v2_cuda_impl(B, C, H, state, r, w, reinterpret_cast<const half*>(w0.data_ptr()), true, k, v, a, b, y, elapsed_t);
}

template <bool Grid2D>
void launch_wkv_one_v2_cuda_impl(
    int B,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    const half* w0_ptr,
    bool add_w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid = Grid2D ? dim3(H, B, 1) : dim3(B * H, 1, 1);
  if (B <= 2) {
    if (add_w0) {
      wkv_fp16_v1_clone_kernel<true, true, Grid2D><<<grid, dim3(N), 0, stream>>>(
          B, 1, C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()),
          w0_ptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()),
          elapsed_t.data_ptr<int>());
    } else {
      wkv_fp16_v1_clone_kernel<true, false, Grid2D><<<grid, dim3(N), 0, stream>>>(
          B, 1, C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()),
          nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()),
          elapsed_t.data_ptr<int>());
    }
  } else if (B <= 64) {
    if (add_w0) {
      wkv_fp16_one_cp_kernel<true, Grid2D><<<grid, dim3(N), 0, stream>>>(
          C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()),
          w0_ptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()),
          elapsed_t.data_ptr<int>());
    } else {
      wkv_fp16_one_cp_kernel<false, Grid2D><<<grid, dim3(N), 0, stream>>>(
          C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()),
          nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()),
          elapsed_t.data_ptr<int>());
    }
  } else if (B <= 128) {
    if (add_w0) {
      wkv_fp16_one_direct_kernel<true, Grid2D><<<grid, dim3(N), 0, stream>>>(
          C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()),
          w0_ptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()),
          elapsed_t.data_ptr<int>());
    } else {
      wkv_fp16_one_direct_kernel<false, Grid2D><<<grid, dim3(N), 0, stream>>>(
          C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()),
          nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()),
          elapsed_t.data_ptr<int>());
    }
  } else {
    if (add_w0) {
      wkv_fp16_v1_clone_kernel<true, true, Grid2D><<<grid, dim3(N), 0, stream>>>(
          B, 1, C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()),
          w0_ptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()),
          elapsed_t.data_ptr<int>());
    } else {
      wkv_fp16_v1_clone_kernel<true, false, Grid2D><<<grid, dim3(N), 0, stream>>>(
          B, 1, C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()),
          nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()),
          elapsed_t.data_ptr<int>());
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wkv_one_v2_cuda_impl(
    int B,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    const half* w0_ptr,
    bool add_w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y,
    at::Tensor elapsed_t,
    bool grid2d) {
  if (grid2d) {
    launch_wkv_one_v2_cuda_impl<true>(
        B, C, H, state, r, w, w0_ptr, add_w0, k, v, a, b, y, elapsed_t);
  } else {
    launch_wkv_one_v2_cuda_impl<false>(
        B, C, H, state, r, w, w0_ptr, add_w0, k, v, a, b, y, elapsed_t);
  }
}
