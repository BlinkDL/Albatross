#include <assert.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#include <vector>

using dtype = at::Half;

namespace {

constexpr int HEAD_SIZE = 64;
constexpr int WARPS_PER_BLOCK = 4;
constexpr float KK_NORMALIZE_EPS = 1.0e-12f;
constexpr float TMIX_LN_X_EPS = 64.0e-5f;
constexpr int FFN_SPMV_THREADS = 128;
constexpr int FFN_TILE = 128;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

__device__ inline __half2 load_h2(const dtype* ptr) {
  return *reinterpret_cast<const __half2*>(ptr);
}

__device__ inline float load_h1(const dtype* ptr) {
  return __half2float(*reinterpret_cast<const __half*>(ptr));
}

__device__ inline void store_h1(dtype* ptr, float value) {
  *reinterpret_cast<__half*>(ptr) = __float2half_rn(value);
}

__device__ inline void store_h2(dtype* ptr, float x0, float x1) {
  *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(x0, x1);
}

__device__ inline float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ inline float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

template <bool UpdateShift>
__global__ void tmix_mix6_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int64_t bt = pair_idx / c_pairs;
  const int c = static_cast<int>(pair_idx - bt * c_pairs) << 1;
  const int b = static_cast<int>(bt / T);
  const int t = static_cast<int>(bt - static_cast<int64_t>(b) * T);
  const int64_t idx = bt * C + c;

  const __half2 cur2 = load_h2(x + idx);
  __half2 prev2;
  if (t == 0) {
    prev2 = load_h2(shift_state + static_cast<int64_t>(b) * C + c);
  } else {
    prev2 = load_h2(x + idx - C);
  }

  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float dx0 = prev.x - cur.x;
  const float dx1 = prev.y - cur.y;

  const float2 xr = __half22float2(load_h2(x_r + c));
  const float2 xw = __half22float2(load_h2(x_w + c));
  const float2 xk = __half22float2(load_h2(x_k + c));
  const float2 xv = __half22float2(load_h2(x_v + c));
  const float2 xa = __half22float2(load_h2(x_a + c));
  const float2 xg = __half22float2(load_h2(x_g + c));

  store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
  store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
  store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
  store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
  store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
  store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);

  if constexpr (UpdateShift) {
    if (t == T - 1) {
      *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = cur2;
    }
  }
}

template <bool UpdateShift>
__global__ void tmix_mix6_3d_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g) {
  const int c_pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int c_pairs = C >> 1;
  if (c_pair >= c_pairs) {
    return;
  }

  const int t = static_cast<int>(blockIdx.y);
  const int b = static_cast<int>(blockIdx.z);
  const int c = c_pair << 1;
  const int64_t bt = static_cast<int64_t>(b) * T + t;
  const int64_t idx = bt * C + c;

  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 = (t == 0)
      ? load_h2(shift_state + static_cast<int64_t>(b) * C + c)
      : load_h2(x + idx - C);
  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float dx0 = prev.x - cur.x;
  const float dx1 = prev.y - cur.y;

  const float2 xr = __half22float2(load_h2(x_r + c));
  const float2 xw = __half22float2(load_h2(x_w + c));
  const float2 xk = __half22float2(load_h2(x_k + c));
  const float2 xv = __half22float2(load_h2(x_v + c));
  const float2 xa = __half22float2(load_h2(x_a + c));
  const float2 xg = __half22float2(load_h2(x_g + c));

  store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
  store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
  store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
  store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
  store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
  store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);

  if constexpr (UpdateShift) {
    // This specialization is only legal for T==1; each half2 has one owner.
    *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = cur2;
  }
}

__global__ void update_shift_state_last_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int b = static_cast<int>(pair_idx / c_pairs);
  const int c = static_cast<int>(pair_idx - static_cast<int64_t>(b) * c_pairs) << 1;
  const int64_t src_idx = (static_cast<int64_t>(b) * T + (T - 1)) * C + c;
  *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = load_h2(x + src_idx);
}

__global__ void update_shift_state_last_2d_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state) {
  const int c_pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int c_pairs = C >> 1;
  if (c_pair >= c_pairs) {
    return;
  }

  const int b = static_cast<int>(blockIdx.y);
  const int c = c_pair << 1;
  const int64_t src_idx = (static_cast<int64_t>(b) * T + (T - 1)) * C + c;
  *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = load_h2(x + src_idx);
}

template <bool HalfMath, int Vec>
__global__ void tmix_mix6_t1_c4096_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t total_pairs) {
  const int64_t base_pair = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * Vec;
#pragma unroll
  for (int u = 0; u < Vec; ++u) {
  const int64_t pair_idx = base_pair + u;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int c = static_cast<int>(pair_idx & 2047) << 1;
  const int64_t idx = pair_idx << 1;
  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 = load_h2(shift_state + idx);
  if constexpr (HalfMath) {
    const __half2 dx = __hsub2(prev2, cur2);
    *reinterpret_cast<__half2*>(out_r + idx) = __hfma2(dx, load_h2(x_r + c), cur2);
    *reinterpret_cast<__half2*>(out_w + idx) = __hfma2(dx, load_h2(x_w + c), cur2);
    *reinterpret_cast<__half2*>(out_k + idx) = __hfma2(dx, load_h2(x_k + c), cur2);
    *reinterpret_cast<__half2*>(out_v + idx) = __hfma2(dx, load_h2(x_v + c), cur2);
    *reinterpret_cast<__half2*>(out_a + idx) = __hfma2(dx, load_h2(x_a + c), cur2);
    *reinterpret_cast<__half2*>(out_g + idx) = __hfma2(dx, load_h2(x_g + c), cur2);
  } else {
    const float2 cur = __half22float2(cur2);
    const float2 prev = __half22float2(prev2);
    const float dx0 = prev.x - cur.x;
    const float dx1 = prev.y - cur.y;
    const float2 xr = __half22float2(load_h2(x_r + c));
    const float2 xw = __half22float2(load_h2(x_w + c));
    const float2 xk = __half22float2(load_h2(x_k + c));
    const float2 xv = __half22float2(load_h2(x_v + c));
    const float2 xa = __half22float2(load_h2(x_a + c));
    const float2 xg = __half22float2(load_h2(x_g + c));
    store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
    store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
    store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
    store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
    store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
    store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);
  }
  *reinterpret_cast<__half2*>(shift_state + idx) = cur2;
  }
}

template <bool UpdateShift>
__global__ void tmix_kk_a_gate_kernel(
    int H,
    const dtype* __restrict__ k,
    const dtype* __restrict__ k_k,
    const dtype* __restrict__ a0,
    const dtype* __restrict__ a12,
    const dtype* __restrict__ k_a,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    dtype* __restrict__ new_k,
    dtype* __restrict__ neg_kk,
    dtype* __restrict__ kka,
    int64_t bth_size) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t bth = static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (bth >= bth_size) {
    return;
  }

  const int64_t h = bth % H;
  const int64_t base = bth * HEAD_SIZE;
  const int64_t c = h * HEAD_SIZE + static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + static_cast<int64_t>(lane) * 2;

  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 kk_scale = __half22float2(load_h2(k_k + c));
  const float u0 = kv.x * kk_scale.x;
  const float u1 = kv.y * kk_scale.y;

  float sum_sq = u0 * u0 + u1 * u1;
  sum_sq = warp_sum(sum_sq);
  const float total = __shfl_sync(0xffffffffu, sum_sq, 0);
  const float inv_d = 1.0f / fmaxf(sqrtf(total), KK_NORMALIZE_EPS);
  const float kk0 = u0 * inv_d;
  const float kk1 = u1 * inv_d;

  const float2 a0v = __half22float2(load_h2(a0 + c));
  const float2 a12v = __half22float2(load_h2(a12 + idx));
  const float av0 = sigmoid_fast(a0v.x + a12v.x);
  const float av1 = sigmoid_fast(a0v.y + a12v.y);
  const float2 ka = __half22float2(load_h2(k_a + c));
  store_h2(new_k + idx, kv.x * fmaf(av0, ka.x, 1.0f - ka.x), kv.y * fmaf(av1, ka.y, 1.0f - ka.y));
  store_h2(neg_kk + idx, -kk0, -kk1);
  store_h2(kka + idx, kk0 * av0, kk1 * av1);
  if constexpr (UpdateShift) {
    *reinterpret_cast<__half2*>(shift_state + idx) = load_h2(x + idx);
  }
}

template <bool UpdateShift>
__global__ void tmix_kk_a_gate_2d_kernel(
    int H,
    const dtype* __restrict__ k,
    const dtype* __restrict__ k_k,
    const dtype* __restrict__ a0,
    const dtype* __restrict__ a12,
    const dtype* __restrict__ k_a,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    dtype* __restrict__ new_k,
    dtype* __restrict__ neg_kk,
    dtype* __restrict__ kka) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int h = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (h >= H) {
    return;
  }

  const int64_t row = blockIdx.y;
  const int64_t bth = row * H + h;
  const int64_t base = bth * HEAD_SIZE;
  const int64_t c = static_cast<int64_t>(h) * HEAD_SIZE + static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + static_cast<int64_t>(lane) * 2;

  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 kk_scale = __half22float2(load_h2(k_k + c));
  const float u0 = kv.x * kk_scale.x;
  const float u1 = kv.y * kk_scale.y;

  float sum_sq = u0 * u0 + u1 * u1;
  sum_sq = warp_sum(sum_sq);
  const float total = __shfl_sync(0xffffffffu, sum_sq, 0);
  const float inv_d = 1.0f / fmaxf(sqrtf(total), KK_NORMALIZE_EPS);
  const float kk0 = u0 * inv_d;
  const float kk1 = u1 * inv_d;

  const float2 a0v = __half22float2(load_h2(a0 + c));
  const float2 a12v = __half22float2(load_h2(a12 + idx));
  const float av0 = sigmoid_fast(a0v.x + a12v.x);
  const float av1 = sigmoid_fast(a0v.y + a12v.y);
  const float2 ka = __half22float2(load_h2(k_a + c));
  store_h2(new_k + idx, kv.x * fmaf(av0, ka.x, 1.0f - ka.x), kv.y * fmaf(av1, ka.y, 1.0f - ka.y));
  store_h2(neg_kk + idx, -kk0, -kk1);
  store_h2(kka + idx, kk0 * av0, kk1 * av1);
  if constexpr (UpdateShift) {
    *reinterpret_cast<__half2*>(shift_state + idx) = load_h2(x + idx);
  }
}

__global__ void tmix_lnx_rkvres_xg_kernel(
    int C,
    int H,
    const dtype* __restrict__ x,
    const dtype* __restrict__ r,
    const dtype* __restrict__ k,
    const dtype* __restrict__ v,
    const dtype* __restrict__ r_k,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ g,
    dtype* __restrict__ out,
    int64_t bth_size) {
  __shared__ float partial[2];
  const int bth = blockIdx.x;
  if (bth >= bth_size) {
    return;
  }
  const int lane = threadIdx.x;
  const int warp = lane >> 5;
  const int warp_lane = lane & 31;
  const int h = bth % H;
  const int64_t base = static_cast<int64_t>(bth) * HEAD_SIZE;
  const int64_t cbase = static_cast<int64_t>(h) * HEAD_SIZE;
  const int64_t idx = base + lane;
  const int64_t c = cbase + lane;

  const float xv = load_h1(x + idx);
  float sum = xv;
  sum = warp_sum(sum);
  if (warp_lane == 0) {
    partial[warp] = sum;
  }
  __syncthreads();
  const float mean = (partial[0] + partial[1]) * (1.0f / 64.0f);
  __syncthreads();

  const float d = xv - mean;
  float ss = d * d;
  ss = warp_sum(ss);
  if (warp_lane == 0) {
    partial[warp] = ss;
  }
  __syncthreads();
  const float var = (partial[0] + partial[1]) * (1.0f / 64.0f);
  const float rstd = rsqrtf(var + TMIX_LN_X_EPS);
  __syncthreads();

  const float rv = load_h1(r + idx);
  const float kv = load_h1(k + idx);
  const float vv = load_h1(v + idx);
  float dot = rv * kv * load_h1(r_k + c);
  dot = warp_sum(dot);
  if (warp_lane == 0) {
    partial[warp] = dot;
  }
  __syncthreads();
  const float rkv = partial[0] + partial[1];
  __syncthreads();

  const float y = (d * rstd * load_h1(weight + c) + load_h1(bias + c) + rkv * vv)
                  * load_h1(g + idx);
  store_h1(out + idx, y);
}

__global__ __launch_bounds__(32) void tmix_lnx_rkvres_xg_warp_kernel(
    int H,
    const dtype* __restrict__ x,
    const dtype* __restrict__ r,
    const dtype* __restrict__ k,
    const dtype* __restrict__ v,
    const dtype* __restrict__ r_k,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ g,
    dtype* __restrict__ out,
    int64_t bth_size) {
  const int64_t bth = blockIdx.x;
  if (bth >= bth_size) {
    return;
  }
  const int lane = threadIdx.x;
  const int h = static_cast<int>(bth % H);
  const int64_t base = bth * HEAD_SIZE;
  const int64_t cbase = static_cast<int64_t>(h) * HEAD_SIZE;
  const int64_t pair = static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + pair;
  const int64_t c = cbase + pair;

  const float2 xv = __half22float2(load_h2(x + idx));
  float sum = warp_sum(xv.x + xv.y);
  const float mean = __shfl_sync(0xffffffffu, sum, 0) * (1.0f / 64.0f);
  const float d0 = xv.x - mean;
  const float d1 = xv.y - mean;
  float ss = warp_sum(d0 * d0 + d1 * d1);
  const float var = __shfl_sync(0xffffffffu, ss, 0) * (1.0f / 64.0f);
  const float rstd = rsqrtf(var + TMIX_LN_X_EPS);

  const float2 rv = __half22float2(load_h2(r + idx));
  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 vv = __half22float2(load_h2(v + idx));
  const float2 rkv_weight = __half22float2(load_h2(r_k + c));
  float dot = warp_sum(rv.x * kv.x * rkv_weight.x + rv.y * kv.y * rkv_weight.y);
  const float rkv = __shfl_sync(0xffffffffu, dot, 0);

  const float2 ln_weight = __half22float2(load_h2(weight + c));
  const float2 ln_bias = __half22float2(load_h2(bias + c));
  const float2 gate = __half22float2(load_h2(g + idx));
  // Pairing adjacent channels changes only the FP32 reduction tree. The model
  // contract permits this small arithmetic-order difference; no precision is removed.
  store_h2(
      out + idx,
      (d0 * rstd * ln_weight.x + ln_bias.x + rkv * vv.x) * gate.x,
      (d1 * rstd * ln_weight.y + ln_bias.y + rkv * vv.y) * gate.y);
}

__global__ __launch_bounds__(32) void tmix_lnx_rkvres_xg_warp_2d_kernel(
    int H,
    const dtype* __restrict__ x,
    const dtype* __restrict__ r,
    const dtype* __restrict__ k,
    const dtype* __restrict__ v,
    const dtype* __restrict__ r_k,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ g,
    dtype* __restrict__ out) {
  const int lane = threadIdx.x;
  const int h = static_cast<int>(blockIdx.x);
  const int64_t row = blockIdx.y;
  const int64_t bth = row * H + h;
  const int64_t base = bth * HEAD_SIZE;
  const int64_t cbase = static_cast<int64_t>(h) * HEAD_SIZE;
  const int64_t pair = static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + pair;
  const int64_t c = cbase + pair;

  const float2 xv = __half22float2(load_h2(x + idx));
  float sum = warp_sum(xv.x + xv.y);
  const float mean = __shfl_sync(0xffffffffu, sum, 0) * (1.0f / 64.0f);
  const float d0 = xv.x - mean;
  const float d1 = xv.y - mean;
  float ss = warp_sum(d0 * d0 + d1 * d1);
  const float var = __shfl_sync(0xffffffffu, ss, 0) * (1.0f / 64.0f);
  const float rstd = rsqrtf(var + TMIX_LN_X_EPS);

  const float2 rv = __half22float2(load_h2(r + idx));
  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 vv = __half22float2(load_h2(v + idx));
  const float2 rkv_weight = __half22float2(load_h2(r_k + c));
  float dot = warp_sum(rv.x * kv.x * rkv_weight.x + rv.y * kv.y * rkv_weight.y);
  const float rkv = __shfl_sync(0xffffffffu, dot, 0);

  const float2 ln_weight = __half22float2(load_h2(weight + c));
  const float2 ln_bias = __half22float2(load_h2(bias + c));
  const float2 gate = __half22float2(load_h2(g + idx));
  store_h2(
      out + idx,
      (d0 * rstd * ln_weight.x + ln_bias.x + rkv * vv.x) * gate.x,
      (d1 * rstd * ln_weight.y + ln_bias.y + rkv * vv.y) * gate.y);
}

__global__ void tmix_vres_gate_kernel(
    int C,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    const dtype* __restrict__ v12,
    dtype* __restrict__ out,
    int64_t total) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int c = static_cast<int>(idx % static_cast<int64_t>(C));
  const float vv = load_h1(v + idx);
  const float gate = sigmoid_fast(load_h1(v0 + c) + load_h1(v12 + idx));
  store_h1(out + idx, fmaf(load_h1(v_first + idx) - vv, gate, vv));
}

template <int Threads>
__global__ __launch_bounds__(Threads) void tmix_vres_gate_vec2_kernel(
    int C,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    const dtype* __restrict__ v12,
    dtype* __restrict__ out,
    int64_t rows) {
  const int pair = static_cast<int>(blockIdx.x) * Threads + threadIdx.x;
  const int pairs_per_row = C >> 1;
  const int64_t row = blockIdx.y;
  if (pair >= pairs_per_row || row >= rows) {
    return;
  }
  const int c = pair << 1;
  const int64_t idx = row * C + c;
  const float2 vv = __half22float2(load_h2(v + idx));
  const float2 vf = __half22float2(load_h2(v_first + idx));
  const float2 base = __half22float2(load_h2(v0 + c));
  const float2 delta = __half22float2(load_h2(v12 + idx));
  const float gate0 = sigmoid_fast(base.x + delta.x);
  const float gate1 = sigmoid_fast(base.y + delta.y);
  store_h2(
      out + idx,
      fmaf(vf.x - vv.x, gate0, vv.x),
      fmaf(vf.y - vv.y, gate1, vv.y));
}

template<int THREADS>
__global__ void cmix_sparse_up_one_kernel(
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ key_fc,
    dtype* __restrict__ act) {
  const int f = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc = 0.0f;

  const auto x2 = reinterpret_cast<const __half2*>(x);
  const auto p2 = reinterpret_cast<const __half2*>(shift_state);
  const auto k2 = reinterpret_cast<const __half2*>(x_k);
  const auto w2 = reinterpret_cast<const __half2*>(key_fc + static_cast<int64_t>(f) * C);
  const int n = C / 2;
  for (int j = tid; j < n; j += THREADS) {
    const float2 xv = __half22float2(x2[j]);
    const float2 pv = __half22float2(p2[j]);
    const float2 kv = __half22float2(k2[j]);
    const float2 wv = __half22float2(w2[j]);
    acc = fmaf(xv.x + (pv.x - xv.x) * kv.x, wv.x, acc);
    acc = fmaf(xv.y + (pv.y - xv.y) * kv.y, wv.y, acc);
  }

  acc = warp_sum(acc);
  __shared__ float warp_sums[THREADS / 32];
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < (THREADS / 32) ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      const float relu = fmaxf(total, 0.0f);
      store_h1(act + f, relu * relu);
    }
  }
}

template<int THREADS>
__global__ void cmix_sparse_up_rows_kernel(
    int T,
    int C,
    int F,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ key_fc,
    dtype* __restrict__ act) {
  const int f = blockIdx.x;
  const int row = blockIdx.y;
  const int b = row / T;
  const int t = row - b * T;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc = 0.0f;

  const auto x2 = reinterpret_cast<const __half2*>(x + static_cast<int64_t>(row) * C);
  const auto p2 = (t == 0)
      ? reinterpret_cast<const __half2*>(shift_state + static_cast<int64_t>(b) * C)
      : reinterpret_cast<const __half2*>(x + static_cast<int64_t>(row - 1) * C);
  const auto k2 = reinterpret_cast<const __half2*>(x_k);
  const auto w2 = reinterpret_cast<const __half2*>(key_fc + static_cast<int64_t>(f) * C);
  const int n = C / 2;
  for (int j = tid; j < n; j += THREADS) {
    const float2 xv = __half22float2(x2[j]);
    const float2 pv = __half22float2(p2[j]);
    const float2 kv = __half22float2(k2[j]);
    const float2 wv = __half22float2(w2[j]);
    acc = fmaf(xv.x + (pv.x - xv.x) * kv.x, wv.x, acc);
    acc = fmaf(xv.y + (pv.y - xv.y) * kv.y, wv.y, acc);
  }

  acc = warp_sum(acc);
  __shared__ float warp_sums[THREADS / 32];
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < (THREADS / 32) ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      const float relu = fmaxf(total, 0.0f);
      store_h1(act + static_cast<int64_t>(row) * F + f, relu * relu);
    }
  }
}

__global__ void cmix_sparse_copy_zero_one_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    dtype* __restrict__ out,
    int C) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n4 = C / 8;
  if (i < n4) {
    reinterpret_cast<int4*>(shift_state)[i] = reinterpret_cast<const int4*>(x)[i];
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
}

__global__ void cmix_sparse_copy_zero_rows_kernel(
    int B,
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    dtype* __restrict__ out,
    int64_t out_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < out_vec4) {
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
  const int64_t state_vec4 = static_cast<int64_t>(B) * (C / 8);
  if (i < state_vec4) {
    const int b = static_cast<int>(i / (C / 8));
    const int c4 = static_cast<int>(i - static_cast<int64_t>(b) * (C / 8));
    reinterpret_cast<int4*>(shift_state)[i] =
        reinterpret_cast<const int4*>(x + (static_cast<int64_t>(b) * T + (T - 1)) * C)[c4];
  }
}

__global__ void zero_vec4_kernel(dtype* __restrict__ out, int64_t n_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_vec4) {
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
}

template <bool UpdateShift>
__global__ void cmix_mix_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int64_t bt = pair_idx / c_pairs;
  const int c = static_cast<int>(pair_idx - bt * c_pairs) << 1;
  const int b = static_cast<int>(bt / T);
  const int t = static_cast<int>(bt - static_cast<int64_t>(b) * T);
  const int64_t idx = bt * C + c;

  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 = (t == 0) ? load_h2(shift_state + static_cast<int64_t>(b) * C + c) : load_h2(x + idx - C);
  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float2 mix = __half22float2(load_h2(x_k + c));
  store_h2(out + idx, cur.x + (prev.x - cur.x) * mix.x, cur.y + (prev.y - cur.y) * mix.y);

  if constexpr (UpdateShift) {
    if (t == T - 1) {
      *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = cur2;
    }
  }
}

template <bool UpdateShift>
__global__ void cmix_mix_3d_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ out) {
  const int c_pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int c_pairs = C >> 1;
  if (c_pair >= c_pairs) {
    return;
  }

  const int t = static_cast<int>(blockIdx.y);
  const int b = static_cast<int>(blockIdx.z);
  const int c = c_pair << 1;
  const int64_t bt = static_cast<int64_t>(b) * T + t;
  const int64_t idx = bt * C + c;

  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 = (t == 0)
      ? load_h2(shift_state + static_cast<int64_t>(b) * C + c)
      : load_h2(x + idx - C);
  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float2 mix = __half22float2(load_h2(x_k + c));
  store_h2(out + idx, cur.x + (prev.x - cur.x) * mix.x, cur.y + (prev.y - cur.y) * mix.y);

  if constexpr (UpdateShift) {
    // Only T==1 launches this specialization. Each half2 owner first reads and
    // then replaces its own state, so no CTA can overwrite state another CTA reads.
    *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = cur2;
  }
}

__global__ void relu_square_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 v = __half22float2(load_h2(x + idx));
  const float x0 = fmaxf(v.x, 0.0f);
  const float x1 = fmaxf(v.y, 0.0f);
  store_h2(out + idx, x0 * x0, x1 * x1);
}

__global__ void act_tanh_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 v = __half22float2(load_h2(x + idx));
  store_h2(out + idx, tanhf(v.x), tanhf(v.y));
}

__global__ void act_sigmoid_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 v = __half22float2(load_h2(x + idx));
  store_h2(out + idx, sigmoid_fast(v.x), sigmoid_fast(v.y));
}

__global__ void add_vec_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ vec,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int c = static_cast<int>((pair_idx % (C >> 1)) << 1);
  const int64_t idx = pair_idx * 2;
  const float2 xv = __half22float2(load_h2(x + idx));
  const float2 vv = __half22float2(load_h2(vec + c));
  store_h2(out + idx, xv.x + vv.x, xv.y + vv.y);
}

__global__ void add_vec_2d_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ vec,
    dtype* __restrict__ out) {
  const int c_pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int c_pairs = C >> 1;
  if (c_pair >= c_pairs) {
    return;
  }
  // The host guard limits rows to grid.y and requires complete C-wide rows.
  // Keeping row in blockIdx.y removes the 64-bit modulo from every half2 owner.
  const int64_t row = blockIdx.y;
  const int c = c_pair << 1;
  const int64_t idx = row * C + c;
  const float2 xv = __half22float2(load_h2(x + idx));
  const float2 vv = __half22float2(load_h2(vec + c));
  store_h2(out + idx, xv.x + vv.x, xv.y + vv.y);
}

__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_one_kernel(
    int C,
    const dtype* __restrict__ act,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half mat_row_smem[2][2 * FFN_SPMV_THREADS];
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;

  if (tid < FFN_TILE / 2) {
    *reinterpret_cast<__half2*>(vec_slice + tid * 2) =
        *reinterpret_cast<const __half2*>(act + start_f + tid * 2);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(reinterpret_cast<__half2*>(out + c_block * (2 * FFN_SPMV_THREADS) + tid * 2), acc);
}

__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_rows_kernel(
    int C,
    int F,
    const dtype* __restrict__ act,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const dtype* act_row = act + static_cast<int64_t>(row) * F;

  if (tid < FFN_TILE / 2) {
    *reinterpret_cast<__half2*>(vec_slice + tid * 2) =
        *reinterpret_cast<const __half2*>(act_row + start_f + tid * 2);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2),
      acc);
}

template <bool Split2>
__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_relu_one_kernel(
    int C,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;

  if (tid < FFN_TILE) {
    const float v = fmaxf(load_h1(preact + start_f + tid), 0.0f);
    vec_slice[tid] = __float2half_rn(v * v);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  if constexpr (Split2) {
    // Two independent chains shorten the loop-carried HFMA2 dependency. This
    // deliberately changes FP16 association order without dropping precision.
    // Keep the odd-nnz guard: nnz_ids[nnz_count] is not an owned element.
    __half2 acc1;
    *reinterpret_cast<int*>(&acc1) = 0;
    for (int i = 0; i < nnz_count; i += 2) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
    }
    acc = __hadd2(acc, acc1);
  } else {
    for (int i = 0; i < nnz_count; ++i) {
      const int actual_f = start_f + nnz_ids[i];
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
    }
  }
  atomicAdd(reinterpret_cast<__half2*>(out + c_block * (2 * FFN_SPMV_THREADS) + tid * 2), acc);
}

template <bool Split2>
__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_relu_rows_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

  if (tid < FFN_TILE) {
    const float v = fmaxf(load_h1(pre_row + start_f + tid), 0.0f);
    vec_slice[tid] = __float2half_rn(v * v);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  if constexpr (Split2) {
    // The two accumulators own the same output half2 and are combined before
    // the original atomicAdd. No extra CTA writes or output races are added.
    __half2 acc1;
    *reinterpret_cast<int*>(&acc1) = 0;
    for (int i = 0; i < nnz_count; i += 2) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
    }
    acc = __hadd2(acc, acc1);
  } else {
    for (int i = 0; i < nnz_count; ++i) {
      const int actual_f = start_f + nnz_ids[i];
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
    }
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2),
      acc);
}

template <int Accumulators>
__global__ __launch_bounds__(256, 2) void cmix_sparse_spmv_relu_rows_t512_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  constexpr int TILE = 512;
  constexpr int THREADS = 256;
  __shared__ __align__(256) __half vec_slice[TILE];
  __shared__ __align__(256) int nnz_ids[TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const float v = fmaxf(load_h1(pre_row + start_f + local_f), 0.0f);
    vec_slice[local_f] = __float2half_rn(v * v);
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    if (lane == 0) {
      warp_counts[warp_id + u * (THREADS / 32)] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    const int local_pos = __popc(mask & ((1u << lane) - 1u));
    const int group = warp_id + u * (THREADS / 32);
    if (nonzero) {
      nnz_ids[warp_prefix[group] + local_pos] = local_f;
    }
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  if constexpr (Accumulators == 1) {
    for (int i = 0; i < nnz_count; ++i) {
      const int local_f = nnz_ids[i];
      const int actual_f = start_f + local_f;
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f]), mat, acc);
    }
  } else if constexpr (Accumulators == 2) {
    __half2 acc1;
    *reinterpret_cast<int*>(&acc1) = 0;
    for (int i = 0; i < nnz_count; i += 2) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      // nnz_count is data-dependent and may be odd. Do not speculatively read
      // nnz_ids[i + 1]: it is not owned by this compacted page in that case.
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
    }
    acc = __hadd2(acc, acc1);
  } else {
    static_assert(Accumulators == 4, "unsupported t512 accumulator count");
    __half2 acc1;
    __half2 acc2;
    __half2 acc3;
    *reinterpret_cast<int*>(&acc1) = 0;
    *reinterpret_cast<int*>(&acc2) = 0;
    *reinterpret_cast<int*>(&acc3) = 0;
    for (int i = 0; i < nnz_count; i += 4) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
      if (i + 2 < nnz_count) {
        const int local_f2 = nnz_ids[i + 2];
        const __half2 mat2 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f2) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc2 = __hfma2(__half2half2(vec_slice[local_f2]), mat2, acc2);
      }
      if (i + 3 < nnz_count) {
        const int local_f3 = nnz_ids[i + 3];
        const __half2 mat3 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f3) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc3 = __hfma2(__half2half2(vec_slice[local_f3]), mat3, acc3);
      }
    }
    acc = __hadd2(__hadd2(acc, acc1), __hadd2(acc2, acc3));
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + c_block * (2 * THREADS) + tid * 2),
      acc);
}

template <int Accumulators>
__global__ __launch_bounds__(256, 2) void cmix_sparse_spmv_relu_rows_t512_reuse_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  static_assert(Accumulators == 1 || Accumulators == 2,
                "unsupported reuse accumulator count");
  constexpr int TILE = 512;
  constexpr int THREADS = 256;
  constexpr int WARPS = THREADS / 32;
  __shared__ __align__(256) __half vec_slice[TILE];
  __shared__ __align__(256) int nnz_ids[TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const float v = fmaxf(load_h1(pre_row + start_f + local_f), 0.0f);
    vec_slice[local_f] = __float2half_rn(v * v);
  }
  __syncthreads();

  // These per-thread masks intentionally remain live across two block
  // barriers. They eliminate the second shared reload/test/ballot, but must
  // never be moved to shared storage where warps would race on ownership.
  unsigned masks[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    masks[u] = mask;
    if (lane == 0) {
      warp_counts[warp_id + u * WARPS] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const unsigned mask = masks[u];
    const bool nonzero = (mask & (1u << lane)) != 0;
    const int local_pos = __popc(mask & ((1u << lane) - 1u));
    const int group = warp_id + u * WARPS;
    if (nonzero) {
      nnz_ids[warp_prefix[group] + local_pos] = local_f;
    }
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  if constexpr (Accumulators == 1) {
    for (int i = 0; i < nnz_count; ++i) {
      const int local_f = nnz_ids[i];
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f]), mat, acc);
    }
  } else {
    __half2 acc1;
    *reinterpret_cast<int*>(&acc1) = 0;
    for (int i = 0; i < nnz_count; i += 2) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
    }
    acc = __hadd2(acc, acc1);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C +
                                 c_block * (2 * THREADS) + tid * 2),
      acc);
}

}  // namespace

std::vector<at::Tensor> tmix_mix6_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    tmix_mix6_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(),
        x_w.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(),
        x_a.data_ptr<dtype>(),
        x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(),
        out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(),
        total_pairs);
  } else {
    tmix_mix6_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(),
        x_w.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(),
        x_a.data_ptr<dtype>(),
        x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(),
        out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(),
        total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> tmix_mix6_cfg_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    int threads) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    tmix_mix6_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(),
        x_w.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(),
        x_a.data_ptr<dtype>(),
        x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(),
        out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(),
        total_pairs);
  } else {
    tmix_mix6_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(),
        x_w.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(),
        x_a.data_ptr<dtype>(),
        x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(),
        out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(),
        total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

void launch_tmix_mix6_flat_out(
    int B,
    int T,
    int C,
    const at::Tensor& x,
    at::Tensor& shift_state,
    const at::Tensor& x_r,
    const at::Tensor& x_w,
    const at::Tensor& x_k,
    const at::Tensor& x_v,
    const at::Tensor& x_a,
    const at::Tensor& x_g,
    at::Tensor& out_r,
    at::Tensor& out_w,
    at::Tensor& out_k,
    at::Tensor& out_v,
    at::Tensor& out_a,
    at::Tensor& out_g) {
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    tmix_mix6_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(), total_pairs);
  } else {
    tmix_mix6_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(), total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_tmix_mix6_3d(
    int B,
    int T,
    int C,
    const at::Tensor& x,
    at::Tensor& shift_state,
    const at::Tensor& x_r,
    const at::Tensor& x_w,
    const at::Tensor& x_k,
    const at::Tensor& x_v,
    const at::Tensor& x_a,
    const at::Tensor& x_g,
    at::Tensor& out_r,
    at::Tensor& out_w,
    at::Tensor& out_k,
    at::Tensor& out_v,
    at::Tensor& out_a,
    at::Tensor& out_g) {
  constexpr int threads = 256;
  const int c_pairs = C / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 mix_grid(static_cast<unsigned int>(ceil_div(c_pairs, threads)),
                      static_cast<unsigned int>(T),
                      static_cast<unsigned int>(B));
  if (T == 1) {
    tmix_mix6_3d_kernel<true><<<mix_grid, threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>());
  } else {
    tmix_mix6_3d_kernel<false><<<mix_grid, threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>());
    // Keep the recurrent write in a second launch. An in-grid t==T-1 store
    // could race a different CTA that has not read old state for t==0 yet.
    const dim3 state_grid(static_cast<unsigned int>(ceil_div(c_pairs, threads)),
                          static_cast<unsigned int>(B));
    update_shift_state_last_2d_kernel<<<state_grid, threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<at::Tensor> tmix_mix6_3d_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  launch_tmix_mix6_3d(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
                      out_r, out_w, out_k, out_v, out_a, out_g);
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

void tmix_mix6_cfg_out_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    at::Tensor out_r,
    at::Tensor out_w,
    at::Tensor out_k,
    at::Tensor out_v,
    at::Tensor out_a,
    at::Tensor out_g) {
  launch_tmix_mix6_flat_out(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
                            out_r, out_w, out_k, out_v, out_a, out_g);
}

void tmix_mix6_3d_out_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    at::Tensor out_r,
    at::Tensor out_w,
    at::Tensor out_k,
    at::Tensor out_v,
    at::Tensor out_a,
    at::Tensor out_g) {
  launch_tmix_mix6_3d(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
                      out_r, out_w, out_k, out_v, out_a, out_g);
}

template <int Vec>
std::vector<at::Tensor> tmix_mix6_t1_c4096_cuda_impl(
    int B,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    int threads,
    bool half_math) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t total_pairs = static_cast<int64_t>(B) * (4096 / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(threads) * Vec));
  if (half_math) {
    tmix_mix6_t1_c4096_kernel<true, Vec><<<blocks, threads, 0, stream>>>(
        x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(),
        total_pairs);
  } else {
    tmix_mix6_t1_c4096_kernel<false, Vec><<<blocks, threads, 0, stream>>>(
        x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(),
        total_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> tmix_mix6_t1_c4096_cuda(
    int B,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    int threads,
    int vec,
    bool half_math) {
  if (vec == 2) {
    return tmix_mix6_t1_c4096_cuda_impl<2>(B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads, half_math);
  }
  if (vec == 4) {
    return tmix_mix6_t1_c4096_cuda_impl<4>(B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads, half_math);
  }
  if (vec == 8) {
    return tmix_mix6_t1_c4096_cuda_impl<8>(B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads, half_math);
  }
  return tmix_mix6_t1_c4096_cuda_impl<1>(B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads, half_math);
}

std::vector<at::Tensor> tmix_kk_a_gate_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor k,
    at::Tensor k_k,
    at::Tensor a0,
    at::Tensor a12,
    at::Tensor k_a,
    at::Tensor x,
    at::Tensor shift_state,
    bool update_shift) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  auto new_k = at::empty_like(k);
  auto neg_kk = at::empty_like(k);
  auto kka = at::empty_like(k);
  const int64_t bth_size = static_cast<int64_t>(B) * T * H;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(ceil_div(bth_size, static_cast<int64_t>(WARPS_PER_BLOCK)));
  if (update_shift) {
    tmix_kk_a_gate_kernel<true><<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
        H, k.data_ptr<dtype>(), k_k.data_ptr<dtype>(), a0.data_ptr<dtype>(), a12.data_ptr<dtype>(),
        k_a.data_ptr<dtype>(), x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        new_k.data_ptr<dtype>(), neg_kk.data_ptr<dtype>(), kka.data_ptr<dtype>(), bth_size);
  } else {
    tmix_kk_a_gate_kernel<false><<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
        H, k.data_ptr<dtype>(), k_k.data_ptr<dtype>(), a0.data_ptr<dtype>(), a12.data_ptr<dtype>(),
        k_a.data_ptr<dtype>(), nullptr, nullptr,
        new_k.data_ptr<dtype>(), neg_kk.data_ptr<dtype>(), kka.data_ptr<dtype>(), bth_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {new_k, neg_kk, kka};
}

std::vector<at::Tensor> tmix_kk_a_gate_2d_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor k,
    at::Tensor k_k,
    at::Tensor a0,
    at::Tensor a12,
    at::Tensor k_a) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  auto new_k = at::empty_like(k);
  auto neg_kk = at::empty_like(k);
  auto kka = at::empty_like(k);
  const int64_t rows = static_cast<int64_t>(B) * T;
  // The C++ boundary requires H%4==0 and rows<=65535. Each 4-warp CTA
  // therefore owns one complete head group with no padded or aliased owner.
  const dim3 grid(static_cast<unsigned int>(ceil_div(H, WARPS_PER_BLOCK)),
                  static_cast<unsigned int>(rows));
  auto stream = at::cuda::getCurrentCUDAStream();
  tmix_kk_a_gate_2d_kernel<false><<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(
      H, k.data_ptr<dtype>(), k_k.data_ptr<dtype>(), a0.data_ptr<dtype>(), a12.data_ptr<dtype>(),
      k_a.data_ptr<dtype>(), nullptr, nullptr,
      new_k.data_ptr<dtype>(), neg_kk.data_ptr<dtype>(), kka.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {new_k, neg_kk, kka};
}

at::Tensor tmix_lnx_rkvres_xg_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor x,
    at::Tensor r,
    at::Tensor k,
    at::Tensor v,
    at::Tensor r_k,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor g) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  auto out = at::empty_like(x);
  const int64_t bth_size = static_cast<int64_t>(B) * T * H;
  auto stream = at::cuda::getCurrentCUDAStream();
  tmix_lnx_rkvres_xg_kernel<<<static_cast<int>(bth_size), HEAD_SIZE, 0, stream>>>(
      C, H,
      x.data_ptr<dtype>(),
      r.data_ptr<dtype>(),
      k.data_ptr<dtype>(),
      v.data_ptr<dtype>(),
      r_k.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      g.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      bth_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor tmix_lnx_rkvres_xg_warp_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor x,
    at::Tensor r,
    at::Tensor k,
    at::Tensor v,
    at::Tensor r_k,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor g) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  auto out = at::empty_like(x);
  const int64_t bth_size = static_cast<int64_t>(B) * T * H;
  auto stream = at::cuda::getCurrentCUDAStream();
  tmix_lnx_rkvres_xg_warp_kernel<<<static_cast<int>(bth_size), 32, 0, stream>>>(
      H,
      x.data_ptr<dtype>(),
      r.data_ptr<dtype>(),
      k.data_ptr<dtype>(),
      v.data_ptr<dtype>(),
      r_k.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      g.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      bth_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor tmix_lnx_rkvres_xg_warp_2d_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor x,
    at::Tensor r,
    at::Tensor k,
    at::Tensor v,
    at::Tensor r_k,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor g) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  auto out = at::empty_like(x);
  const int64_t rows = static_cast<int64_t>(B) * T;
  const dim3 grid(static_cast<unsigned int>(H), static_cast<unsigned int>(rows));
  auto stream = at::cuda::getCurrentCUDAStream();
  tmix_lnx_rkvres_xg_warp_2d_kernel<<<grid, 32, 0, stream>>>(
      H,
      x.data_ptr<dtype>(),
      r.data_ptr<dtype>(),
      k.data_ptr<dtype>(),
      v.data_ptr<dtype>(),
      r_k.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      g.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <int Threads>
void launch_tmix_vres_gate_vec2(
    int B,
    int T,
    int C,
    const at::Tensor& v,
    const at::Tensor& v_first,
    const at::Tensor& v0,
    const at::Tensor& v12,
    at::Tensor& out,
    cudaStream_t stream) {
  const int64_t rows = static_cast<int64_t>(B) * T;
  const int pairs_per_row = C >> 1;
  tmix_vres_gate_vec2_kernel<Threads><<<
      dim3(static_cast<unsigned int>(ceil_div(pairs_per_row, Threads)), static_cast<unsigned int>(rows), 1),
      Threads, 0, stream>>>(
      C,
      v.data_ptr<dtype>(),
      v_first.data_ptr<dtype>(),
      v0.data_ptr<dtype>(),
      v12.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      rows);
}

void launch_tmix_vres_gate_cfg(
    int B,
    int T,
    int C,
    const at::Tensor& v,
    const at::Tensor& v_first,
    const at::Tensor& v0,
    const at::Tensor& v12,
    at::Tensor& out,
    int threads,
    bool vectorized,
    cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(B) * T * C;
  if (!vectorized) {
    tmix_vres_gate_kernel<<<static_cast<int>(ceil_div(total, threads)), threads, 0, stream>>>(
        C,
        v.data_ptr<dtype>(),
        v_first.data_ptr<dtype>(),
        v0.data_ptr<dtype>(),
        v12.data_ptr<dtype>(),
        out.data_ptr<dtype>(),
        total);
    return;
  }
  switch (threads) {
    case 64:
      launch_tmix_vres_gate_vec2<64>(B, T, C, v, v_first, v0, v12, out, stream);
      break;
    case 128:
      launch_tmix_vres_gate_vec2<128>(B, T, C, v, v_first, v0, v12, out, stream);
      break;
    case 256:
      launch_tmix_vres_gate_vec2<256>(B, T, C, v, v_first, v0, v12, out, stream);
      break;
    default:
      launch_tmix_vres_gate_vec2<512>(B, T, C, v, v_first, v0, v12, out, stream);
      break;
  }
}

at::Tensor tmix_vres_gate_cuda(
    int B,
    int T,
    int C,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0,
    at::Tensor v12) {
  auto out = at::empty_like(v);
  if (out.numel() == 0) {
    return out;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_tmix_vres_gate_cfg(B, T, C, v, v_first, v0, v12, out, 256, false, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor tmix_vres_gate_cfg_cuda(
    int B,
    int T,
    int C,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0,
    at::Tensor v12,
    int threads,
    bool vectorized) {
  auto out = at::empty_like(v);
  if (out.numel() == 0) {
    return out;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_tmix_vres_gate_cfg(B, T, C, v, v_first, v0, v12, out, threads, vectorized, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void tmix_vres_gate_cfg_out_cuda(
    int B,
    int T,
    int C,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0,
    at::Tensor v12,
    at::Tensor out,
    int threads,
    bool vectorized) {
  if (out.numel() == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_tmix_vres_gate_cfg(B, T, C, v, v_first, v0, v12, out, threads, vectorized, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor cmix_sparse_one_cuda(
    int C,
    int F,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k,
    at::Tensor key_fc,
    at::Tensor value_fc) {
  auto act = at::empty({F}, x.options());
  auto out = at::empty({1, 1, C}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  cmix_sparse_up_one_kernel<64><<<F, 64, 0, stream>>>(
      C,
      x.data_ptr<dtype>(),
      shift_state.data_ptr<dtype>(),
      x_k.data_ptr<dtype>(),
      key_fc.data_ptr<dtype>(),
      act.data_ptr<dtype>());
  cmix_sparse_copy_zero_one_kernel<<<(C / 8 + 127) / 128, 128, 0, stream>>>(
      x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), out.data_ptr<dtype>(), C);
  cmix_sparse_spmv_one_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), 1), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      act.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_rows_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k,
    at::Tensor key_fc,
    at::Tensor value_fc) {
  const int rows = B * T;
  auto act = at::empty({rows, F}, x.options());
  auto out = at::empty({B, T, C}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  cmix_sparse_up_rows_kernel<64><<<dim3(F, rows, 1), 64, 0, stream>>>(
      T,
      C,
      F,
      x.data_ptr<dtype>(),
      shift_state.data_ptr<dtype>(),
      x_k.data_ptr<dtype>(),
      key_fc.data_ptr<dtype>(),
      act.data_ptr<dtype>());
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  cmix_sparse_copy_zero_rows_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(
      B,
      T,
      C,
      x.data_ptr<dtype>(),
      shift_state.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      out_vec4);
  cmix_sparse_spmv_rows_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      F,
      act.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_one_cuda(
    int C,
    int F,
    at::Tensor act,
    at::Tensor value_fc) {
  auto out = at::empty({1, 1, C}, act.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  zero_vec4_kernel<<<(C / 8 + 127) / 128, 128, 0, stream>>>(out.data_ptr<dtype>(), C / 8);
  cmix_sparse_spmv_one_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), 1), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      act.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_rows_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor act,
    at::Tensor value_fc) {
  const int rows = B * T;
  auto out = at::empty({B, T, C}, act.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(out.data_ptr<dtype>(), out_vec4);
  cmix_sparse_spmv_rows_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      F,
      act.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_one_cuda(
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc) {
  auto out = at::empty({1, 1, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  zero_vec4_kernel<<<(C / 8 + 127) / 128, 128, 0, stream>>>(out.data_ptr<dtype>(), C / 8);
  cmix_sparse_spmv_relu_one_kernel<false><<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), 1), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      preact.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_one_split2_cuda(
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc) {
  auto out = at::empty({1, 1, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  zero_vec4_kernel<<<(C / 8 + 127) / 128, 128, 0, stream>>>(out.data_ptr<dtype>(), C / 8);
  cmix_sparse_spmv_relu_one_kernel<true><<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), 1), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      preact.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_rows_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc) {
  const int rows = B * T;
  auto out = at::empty({B, T, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(out.data_ptr<dtype>(), out_vec4);
  cmix_sparse_spmv_relu_rows_kernel<false><<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      F,
      preact.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_rows_split2_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc) {
  const int rows = B * T;
  auto out = at::empty({B, T, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(out.data_ptr<dtype>(), out_vec4);
  cmix_sparse_spmv_relu_rows_kernel<true><<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      F,
      preact.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void launch_cmix_sparse_down_relu_rows_t512(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc,
    at::Tensor out,
    int accumulators) {
  const int rows = B * T;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(out.data_ptr<dtype>(), out_vec4);
  if (accumulators == 1) {
    cmix_sparse_spmv_relu_rows_t512_kernel<1><<<dim3(F / 512, C / 512, rows), 256, 0, stream>>>(
        C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(), out.data_ptr<dtype>());
  } else if (accumulators == 2) {
    cmix_sparse_spmv_relu_rows_t512_kernel<2><<<dim3(F / 512, C / 512, rows), 256, 0, stream>>>(
        C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(), out.data_ptr<dtype>());
  } else {
    cmix_sparse_spmv_relu_rows_t512_kernel<4><<<dim3(F / 512, C / 512, rows), 256, 0, stream>>>(
        C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(), out.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor cmix_sparse_down_relu_rows_t512_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc) {
  auto out = at::empty({B, T, C}, preact.options());
  launch_cmix_sparse_down_relu_rows_t512(B, T, C, F, preact, value_fc, out, 1);
  return out;
}

at::Tensor cmix_sparse_down_relu_rows_t512_cfg_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc,
    int accumulators) {
  auto out = at::empty({B, T, C}, preact.options());
  launch_cmix_sparse_down_relu_rows_t512(
      B, T, C, F, preact, value_fc, out, accumulators);
  return out;
}

void cmix_sparse_down_relu_rows_t512_cfg_out_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc,
    at::Tensor out,
    int accumulators) {
  launch_cmix_sparse_down_relu_rows_t512(
      B, T, C, F, preact, value_fc, out, accumulators);
}

void launch_cmix_sparse_down_relu_rows_t512_reuse(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc,
    at::Tensor out,
    int accumulators) {
  const int rows = B * T;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(
      out.data_ptr<dtype>(), out_vec4);
  if (accumulators == 1) {
    cmix_sparse_spmv_relu_rows_t512_reuse_kernel<1>
        <<<dim3(F / 512, C / 512, rows), 256, 0, stream>>>(
            C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(),
            out.data_ptr<dtype>());
  } else {
    cmix_sparse_spmv_relu_rows_t512_reuse_kernel<2>
        <<<dim3(F / 512, C / 512, rows), 256, 0, stream>>>(
            C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(),
            out.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor cmix_sparse_down_relu_rows_t512_reuse_cfg_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc,
    int accumulators) {
  auto out = at::empty({B, T, C}, preact.options());
  launch_cmix_sparse_down_relu_rows_t512_reuse(
      B, T, C, F, preact, value_fc, out, accumulators);
  return out;
}

void cmix_sparse_down_relu_rows_t512_reuse_cfg_out_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc,
    at::Tensor out,
    int accumulators) {
  launch_cmix_sparse_down_relu_rows_t512_reuse(
      B, T, C, F, preact, value_fc, out, accumulators);
}

at::Tensor cmix_mix_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    cmix_mix_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T,
        C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        out.data_ptr<dtype>(),
        total_pairs);
  } else {
    cmix_mix_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T,
        C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        out.data_ptr<dtype>(),
        total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_mix_cfg_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k,
    int threads) {
  auto out = at::empty_like(x);
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    cmix_mix_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T,
        C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        out.data_ptr<dtype>(),
        total_pairs);
  } else {
    cmix_mix_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T,
        C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        out.data_ptr<dtype>(),
        total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void cmix_mix_cfg_out_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k,
    at::Tensor out,
    int threads) {
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    cmix_mix_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
  } else {
    cmix_mix_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_cmix_mix_3d(
    int B,
    int T,
    int C,
    const at::Tensor& x,
    at::Tensor& shift_state,
    const at::Tensor& x_k,
    at::Tensor& out) {
  constexpr int threads = 256;
  const int c_pairs = C / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 mix_grid(static_cast<unsigned int>(ceil_div(c_pairs, threads)),
                      static_cast<unsigned int>(T),
                      static_cast<unsigned int>(B));
  if (T == 1) {
    cmix_mix_3d_kernel<true><<<mix_grid, threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(), out.data_ptr<dtype>());
  } else {
    cmix_mix_3d_kernel<false><<<mix_grid, threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(), out.data_ptr<dtype>());
    // T>1 must remain a second launch: an in-grid last-token write can race
    // with a different CTA that is still reading the old recurrent state.
    const dim3 state_grid(static_cast<unsigned int>(ceil_div(c_pairs, threads)),
                          static_cast<unsigned int>(B));
    update_shift_state_last_2d_kernel<<<state_grid, threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor cmix_mix_3d_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k) {
  auto out = at::empty_like(x);
  launch_cmix_mix_3d(B, T, C, x, shift_state, x_k, out);
  return out;
}

void cmix_mix_3d_out_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k,
    at::Tensor out) {
  launch_cmix_mix_3d(B, T, C, x, shift_state, x_k, out);
}

at::Tensor relu_square_cuda(at::Tensor x) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  relu_square_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor act_tanh_cuda(at::Tensor x) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  act_tanh_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor act_sigmoid_cuda(at::Tensor x) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  act_sigmoid_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor add_vec_cuda(int C, at::Tensor x, at::Tensor vec) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  add_vec_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      C,
      x.data_ptr<dtype>(),
      vec.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor add_vec_2d_cuda(int C, at::Tensor x, at::Tensor vec) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int rows = static_cast<int>(x.numel() / C);
  auto stream = at::cuda::getCurrentCUDAStream();
  add_vec_2d_kernel<<<dim3(static_cast<unsigned int>(ceil_div(C / 2, threads)),
                            static_cast<unsigned int>(rows), 1),
                      threads, 0, stream>>>(
      C,
      x.data_ptr<dtype>(),
      vec.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void add_vec_cfg_out_cuda(
    int C,
    at::Tensor x,
    at::Tensor vec,
    at::Tensor out,
    bool grid2d) {
  constexpr int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (grid2d) {
    const int rows = static_cast<int>(x.numel() / C);
    add_vec_2d_kernel<<<dim3(static_cast<unsigned int>(ceil_div(C / 2, threads)),
                              static_cast<unsigned int>(rows), 1),
                        threads, 0, stream>>>(
        C, x.data_ptr<dtype>(), vec.data_ptr<dtype>(), out.data_ptr<dtype>());
  } else {
    const int64_t total_pairs = x.numel() / 2;
    add_vec_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        C, x.data_ptr<dtype>(), vec.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
