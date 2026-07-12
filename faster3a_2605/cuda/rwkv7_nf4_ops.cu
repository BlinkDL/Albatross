// NVFP4 (E2M1) quantized linear kernels for RWKV-7 v3a
// Follows the exact same structure as rwkv7_v3a_ops.cu
// Weight: uint8 [N, K/2] packed (2 E2M1 per byte)
// Block scale: float8_e4m3fn [N, K/16] (per-block, 16 elements along K)
// Tensor scale: float32 (per-tensor)

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <climits>
#include <vector>

using dtype = at::Half;

namespace {

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

// E2M1 4-bit float -> float32 decode table (16 entries)
// Index 0-7: positive values, Index 8-15: negative values
__constant__ float e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// E2M1 __half LUT: same values as __half for __hfma2 accumulation
// FP16 bit patterns: 0=0x0000, 0.5=0x3800, 1.0=0x3C00, 1.5=0x3E00,
// 2.0=0x4000, 3.0=0x4200, 4.0=0x4400, 6.0=0x4600
// Negatives: flip sign bit (bit 15)
__constant__ unsigned short e2m1_raw[16] = {
    0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600,
    0x8000, 0xB800, 0xBC00, 0xBE00, 0xC000, 0xC200, 0xC400, 0xC600
};


// Bitwise E2M1 decode: 4-bit code -> float32 (no LUT, avoids constant memory serialization)
// E2M1 format: bit 3 = sign, bits 2-0 = magnitude code (0-7)
// Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (with sign)
// Special: code 1 (0.5) has mantissa=0 unlike code 3/5/7 (odd codes >= 3 have mant=0.5)
__device__ __forceinline__ float e2m1_decode_f(uint8_t code) {
  const uint32_t sign = (code & 0x8u) ? 0x80000000u : 0u;
  const uint32_t mag = code & 0x7u;
  const uint32_t exp_field = ((mag >> 1) + 126) << 23;
  // mantissa MSB set only for odd codes >= 3 (1.5, 3.0, 6.0), NOT for code 1 (0.5)
  const uint32_t mant = (mag >= 2u && (mag & 1u)) ? (1u << 22) : 0u;
  const uint32_t bits = mag ? (sign | exp_field | mant) : sign;
  return __int_as_float(bits);
}

// Bitwise E2M1 decode: 4-bit code -> __half (no LUT, avoids constant memory serialization)
// FP16 version: bias=15, 5-bit exp, 10-bit mantissa
__device__ __forceinline__ __half e2m1_decode_h(uint8_t code) {
  const uint16_t sign = (code & 0x8u) ? 0x8000u : 0u;
  const uint32_t mag = code & 0x7u;
  const uint16_t exp_field = mag ? static_cast<uint16_t>(((mag >> 1) + 14) << 10) : 0u;
  const uint16_t mant = (mag >= 2u && (mag & 1u)) ? 0x0200u : 0u;
  return __ushort_as_half(sign | exp_field | mant);
}



// E4M3 FP8 -> float32 conversion
// __nv_fp8_e4m3 supports implicit conversion to float in CUDA 12.0+
__device__ __forceinline__ float e4m3_to_float(__nv_fp8_e4m3 v) {
  return float(v);
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

// ═══════════════════════════════════════════════════════════════
// M=1 GEMV: x[K] @ w[N,K]^T -> y[N]
// 2-wide K loop (same as v3a linear_orig_row1_exact_f16_kernel)
// ═══════════════════════════════════════════════════════════════
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_nf4_orig_row1_exact_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const uint8_t* __restrict__ w_nf4,
    const __nv_fp8_e4m3* __restrict__ b_scale,
    float t_scale,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int K2 = K >> 1;    // packed bytes per row
  const int KB = K >> 4;    // blocks per row (K/16)
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const uint8_t packed = w_nf4[static_cast<int64_t>(n0 + j) * K2 + k2];
      const float bs = float(b_scale[static_cast<int64_t>(n0 + j) * KB + (k >> 4)]) * t_scale;
      acc[j] = fmaf(xv.x, e2m1_decode_f(packed & 0x0F) * bs, acc[j]);
      acc[j] = fmaf(xv.y, e2m1_decode_f(packed >> 4) * bs, acc[j]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      y[n0 + j] = __float2half_rn(sum);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// M=1 GEMV: 4-wide K loop (same as v3a linear_orig_row1_exact4_f16_kernel)
// ═══════════════════════════════════════════════════════════════
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_nf4_orig_row1_exact4_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const uint8_t* __restrict__ w_nf4,
    const __nv_fp8_e4m3* __restrict__ b_scale,
    float t_scale,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int K2 = K >> 1;    // packed bytes per row
  const int KB = K >> 4;    // blocks per row
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  // 4-wide: process 4 K elements per iteration (2 packed bytes)
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
    const int k2 = k >> 1;  // byte index
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const uint8_t* wp = w_nf4 + static_cast<int64_t>(n0 + j) * K2 + k2;
      // k, k+1, k+2, k+3 all within same 16-element block (k is 4-aligned)
      const float bs = float(b_scale[static_cast<int64_t>(n0 + j) * KB + (k >> 4)]) * t_scale;
      const float wv0 = e2m1_decode_f(wp[0] & 0x0F) * bs;
      const float wv1 = e2m1_decode_f(wp[0] >> 4)  * bs;
      const float wv2 = e2m1_decode_f(wp[1] & 0x0F) * bs;
      const float wv3 = e2m1_decode_f(wp[1] >> 4)  * bs;
      acc[j] = fmaf(x0.x, wv0, acc[j]);
      acc[j] = fmaf(x0.y, wv1, acc[j]);
      acc[j] = fmaf(x1.x, wv2, acc[j]);
      acc[j] = fmaf(x1.y, wv3, acc[j]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      y[n0 + j] = __float2half_rn(sum);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// M=2 GEMV: x[2,K] @ w[N,K]^T -> y[2,N]
// 2-wide K loop (same as v3a linear_orig_row2_exact_f16_kernel)
// ═══════════════════════════════════════════════════════════════
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_nf4_orig_row2_exact_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const uint8_t* __restrict__ w_nf4,
    const __nv_fp8_e4m3* __restrict__ b_scale,
    float t_scale,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int K2 = K >> 1;
  const int KB = K >> 4;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const uint8_t packed = w_nf4[static_cast<int64_t>(n0 + j) * K2 + k2];
      const float bs = float(b_scale[static_cast<int64_t>(n0 + j) * KB + (k >> 4)]) * t_scale;
      const float wv_x = e2m1_decode_f(packed & 0x0F) * bs;
      const float wv_y = e2m1_decode_f(packed >> 4)  * bs;
      acc0[j] = fmaf(x0.x, wv_x, acc0[j]);
      acc0[j] = fmaf(x0.y, wv_y, acc0[j]);
      acc1[j] = fmaf(x1.x, wv_x, acc1[j]);
      acc1[j] = fmaf(x1.y, wv_y, acc1[j]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (lane == 0) {
      partial[warp][0][j] = v0;
      partial[warp][1][j] = v1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum0 += partial[w][0][j];
        sum1 += partial[w][1][j];
      }
      const int n = n0 + j;
      y[n] = __float2half_rn(sum0);
      y[N + n] = __float2half_rn(sum1);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// M=2 GEMV: 4-wide K loop (same as v3a linear_orig_row2_exact4_f16_kernel)
// ═══════════════════════════════════════════════════════════════
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_nf4_orig_row2_exact4_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const uint8_t* __restrict__ w_nf4,
    const __nv_fp8_e4m3* __restrict__ b_scale,
    float t_scale,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int K2 = K >> 1;
  const int KB = K >> 4;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x00 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x01 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
    const float2 x10 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
    const float2 x11 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 2));
    const int k2 = k >> 1;
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const uint8_t* wp = w_nf4 + static_cast<int64_t>(n0 + j) * K2 + k2;
      const float bs = float(b_scale[static_cast<int64_t>(n0 + j) * KB + (k >> 4)]) * t_scale;
      const float wv0 = e2m1_decode_f(wp[0] & 0x0F) * bs;
      const float wv1 = e2m1_decode_f(wp[0] >> 4)  * bs;
      const float wv2 = e2m1_decode_f(wp[1] & 0x0F) * bs;
      const float wv3 = e2m1_decode_f(wp[1] >> 4)  * bs;
      acc0[j] = fmaf(x00.x, wv0, acc0[j]);
      acc0[j] = fmaf(x00.y, wv1, acc0[j]);
      acc0[j] = fmaf(x01.x, wv2, acc0[j]);
      acc0[j] = fmaf(x01.y, wv3, acc0[j]);
      acc1[j] = fmaf(x10.x, wv0, acc1[j]);
      acc1[j] = fmaf(x10.y, wv1, acc1[j]);
      acc1[j] = fmaf(x11.x, wv2, acc1[j]);
      acc1[j] = fmaf(x11.y, wv3, acc1[j]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (lane == 0) {
      partial[warp][0][j] = v0;
      partial[warp][1][j] = v1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum0 += partial[w][0][j];
        sum1 += partial[w][1][j];
      }
      const int n = n0 + j;
      y[n] = __float2half_rn(sum0);
      y[N + n] = __float2half_rn(sum1);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// M>=3 GEMM: x[M,K] @ w[N,K]^T -> y[M,N]
// 2-wide K loop (same as v3a linear_orig_rows_f16_kernel)
// ═══════════════════════════════════════════════════════════════
template <int Threads, int RowTile, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_nf4_orig_rows_f16_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const uint8_t* __restrict__ w_nf4,
    const __nv_fp8_e4m3* __restrict__ b_scale,
    float t_scale,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m0 = blockIdx.y * RowTile;
  const int K2 = K >> 1;
  const int KB = K >> 4;
  float acc[RowTile][OutTile];
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      acc[r][j] = 0.0f;
    }
  }
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    // Decode weight once, share across RowTile rows (same pattern as v3a)
    float wv_low[OutTile];
    float wv_high[OutTile];
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const uint8_t packed = w_nf4[static_cast<int64_t>(n) * K2 + k2];
        const float bs = float(b_scale[static_cast<int64_t>(n) * KB + (k >> 4)]) * t_scale;
        wv_low[j]  = e2m1_decode_f(packed & 0x0F) * bs;
        wv_high[j] = e2m1_decode_f(packed >> 4)    * bs;
      } else {
        wv_low[j]  = 0.0f;
        wv_high[j] = 0.0f;
      }
    }
    // Reuse decoded weight across all rows
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + static_cast<int64_t>(m) * K + k));
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          acc[r][j] = fmaf(xv.x, wv_low[j], acc[r][j]);
          acc[r][j] = fmaf(xv.y, wv_high[j], acc[r][j]);
        }
      }
    }
  }
  __shared__ float partial[Threads / 32][RowTile][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float v = warp_sum(acc[r][j]);
      if (lane == 0) {
        partial[warp][r][j] = v;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          const int n = n0 + j;
          if (n < N) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < Threads / 32; ++w) {
              sum += partial[w][r][j];
            }
            *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
          }
        }
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// Batch dequant: NF4 [N, K/2] uint8 + b_scale [N, K/16] -> fp16
// Transpose=false: output [N, K] (orig layout)
// Transpose=true:  output [K, N] (non-orig layout)
// ═══════════════════════════════════════════════════════════════
template <int Threads, bool Transpose>
__global__ void dequant_nf4_to_f16_kernel(
    int N,
    int K,
    const uint8_t* __restrict__ w_nf4,
    const __nv_fp8_e4m3* __restrict__ b_scale,
    float t_scale,
    dtype* __restrict__ out) {
  const int n = blockIdx.x;
  if (n >= N) return;
  const int K2 = K >> 1;
  const int KB = K >> 4;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const int k2 = k >> 1;
    const uint8_t packed = w_nf4[static_cast<int64_t>(n) * K2 + k2];
    const float bs = float(b_scale[static_cast<int64_t>(n) * KB + (k >> 4)]) * t_scale;
    float val;
    if ((k & 1) == 0) {
      val = e2m1_decode_f(packed & 0x0F) * bs;
    } else {
      val = e2m1_decode_f(packed >> 4) * bs;
    }
    if (Transpose) {
      out[static_cast<int64_t>(k) * N + n] = __float2half_rn(val);
    } else {
      out[static_cast<int64_t>(n) * K + k] = __float2half_rn(val);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// OPTIMIZED v3: M=1 GEMV — dual-block parallel iteration
//
// Key optimizations vs v2:
// 1. Process 2 K-blocks per iteration with independent accumulators
//    → 2 independent dependency chains, warp scheduler can overlap loads/computes
// 2. Shared mem LUT + __hfma2 (from v2)
// 3. Step by 64 instead of 32, each iter handles blocks kb and kb+32
//    Requires KB % 2 == 0 (true for all RWKV: C is power of 2, KB = C/16)
// ═══════════════════════════════════════════════════════════════
template <int OutTile>
__global__ __launch_bounds__(32, 1) void linear_nvfp4_orig_row1_blk16_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const uint8_t* __restrict__ w_nf4,
    const __nv_fp8_e4m3* __restrict__ b_scale,
    float t_scale,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int KB = K >> 4;
  const int K2 = K >> 1;

  // Shared memory LUT: byte (2 packed E2M1 codes) → __half2
  // 256 entries × 4 bytes = 1 KB
  __shared__ __half2 byte_lut[256];
  for (int i = threadIdx.x; i < 256; i += 32) {
    unsigned short lo = e2m1_raw[i & 0xF];
    unsigned short hi = e2m1_raw[(i >> 4) & 0xF];
    uint32_t bits = lo | (static_cast<uint32_t>(hi) << 16);
    byte_lut[i] = *reinterpret_cast<__half2*>(&bits);
  }
  __syncthreads();

  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }

  // Dual-block: process kb and kb+KB/2 per iteration
  // This gives 2 independent dependency chains for ILP
  const int half_KB = KB >> 1;
  for (int kb = threadIdx.x; kb < half_KB; kb += 32) {
    // ── Block A: kb ──
    const int kA = kb << 4;
    const __half2* xA = reinterpret_cast<const __half2*>(x + kA);
    const __half2 xA0 = xA[0], xA1 = xA[1], xA2 = xA[2], xA3 = xA[3];
    const __half2 xA4 = xA[4], xA5 = xA[5], xA6 = xA[6], xA7 = xA[7];

    // ── Block B: kb + half_KB ──
    const int kB = (kb + half_KB) << 4;
    const __half2* xB = reinterpret_cast<const __half2*>(x + kB);
    const __half2 xB0 = xB[0], xB1 = xB[1], xB2 = xB[2], xB3 = xB[3];
    const __half2 xB4 = xB[4], xB5 = xB[5], xB6 = xB[6], xB7 = xB[7];

#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      // Block A
      const float bsA = float(b_scale[static_cast<int64_t>(n0 + j) * KB + kb]) * t_scale;
      const uint2 pkA = *reinterpret_cast<const uint2*>(
          w_nf4 + static_cast<int64_t>(n0 + j) * K2 + (kb << 3));
      const __half2 wA0 = byte_lut[pkA.x & 0xFF];
      const __half2 wA1 = byte_lut[(pkA.x >> 8) & 0xFF];
      const __half2 wA2 = byte_lut[(pkA.x >> 16) & 0xFF];
      const __half2 wA3 = byte_lut[pkA.x >> 24];
      const __half2 wA4 = byte_lut[pkA.y & 0xFF];
      const __half2 wA5 = byte_lut[(pkA.y >> 8) & 0xFF];
      const __half2 wA6 = byte_lut[(pkA.y >> 16) & 0xFF];
      const __half2 wA7 = byte_lut[pkA.y >> 24];
      __half2 accA = __hfma2(wA0, xA0, __float2half2_rn(0.0f));
      accA = __hfma2(wA1, xA1, accA);
      accA = __hfma2(wA2, xA2, accA);
      accA = __hfma2(wA3, xA3, accA);
      accA = __hfma2(wA4, xA4, accA);
      accA = __hfma2(wA5, xA5, accA);
      accA = __hfma2(wA6, xA6, accA);
      accA = __hfma2(wA7, xA7, accA);

      // Block B (independent chain — can execute while A waits for memory)
      const float bsB = float(b_scale[static_cast<int64_t>(n0 + j) * KB + kb + half_KB]) * t_scale;
      const uint2 pkB = *reinterpret_cast<const uint2*>(
          w_nf4 + static_cast<int64_t>(n0 + j) * K2 + ((kb + half_KB) << 3));
      const __half2 wB0 = byte_lut[pkB.x & 0xFF];
      const __half2 wB1 = byte_lut[(pkB.x >> 8) & 0xFF];
      const __half2 wB2 = byte_lut[(pkB.x >> 16) & 0xFF];
      const __half2 wB3 = byte_lut[pkB.x >> 24];
      const __half2 wB4 = byte_lut[pkB.y & 0xFF];
      const __half2 wB5 = byte_lut[(pkB.y >> 8) & 0xFF];
      const __half2 wB6 = byte_lut[(pkB.y >> 16) & 0xFF];
      const __half2 wB7 = byte_lut[pkB.y >> 24];
      __half2 accB = __hfma2(wB0, xB0, __float2half2_rn(0.0f));
      accB = __hfma2(wB1, xB1, accB);
      accB = __hfma2(wB2, xB2, accB);
      accB = __hfma2(wB3, xB3, accB);
      accB = __hfma2(wB4, xB4, accB);
      accB = __hfma2(wB5, xB5, accB);
      accB = __hfma2(wB6, xB6, accB);
      accB = __hfma2(wB7, xB7, accB);

      // Merge A+B into FP32 accumulator
      const float2 sumA = __half22float2(accA);
      const float2 sumB = __half22float2(accB);
      acc[j] = fmaf(sumA.x + sumA.y, bsA, acc[j]);
      acc[j] = fmaf(sumB.x + sumB.y, bsB, acc[j]);
    }
  }
  // Handle odd KB (shouldn't happen for RWKV, but be safe)
  if (KB & 1) {
    const int kb = threadIdx.x + half_KB * 2;
    if (kb < KB) {
      const int k = kb << 4;
      const __half2* x2 = reinterpret_cast<const __half2*>(x + k);
      const __half2 xv0 = x2[0], xv1 = x2[1], xv2 = x2[2], xv3 = x2[3];
      const __half2 xv4 = x2[4], xv5 = x2[5], xv6 = x2[6], xv7 = x2[7];
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const float bs = float(b_scale[static_cast<int64_t>(n0 + j) * KB + kb]) * t_scale;
        const uint2 packed = *reinterpret_cast<const uint2*>(
            w_nf4 + static_cast<int64_t>(n0 + j) * K2 + (kb << 3));
        const __half2 wv0 = byte_lut[packed.x & 0xFF];
        const __half2 wv1 = byte_lut[(packed.x >> 8) & 0xFF];
        const __half2 wv2 = byte_lut[(packed.x >> 16) & 0xFF];
        const __half2 wv3 = byte_lut[packed.x >> 24];
        const __half2 wv4 = byte_lut[packed.y & 0xFF];
        const __half2 wv5 = byte_lut[(packed.y >> 8) & 0xFF];
        const __half2 wv6 = byte_lut[(packed.y >> 16) & 0xFF];
        const __half2 wv7 = byte_lut[packed.y >> 24];
        __half2 acc_h = __hfma2(wv0, xv0, __float2half2_rn(0.0f));
        acc_h = __hfma2(wv1, xv1, acc_h);
        acc_h = __hfma2(wv2, xv2, acc_h);
        acc_h = __hfma2(wv3, xv3, acc_h);
        acc_h = __hfma2(wv4, xv4, acc_h);
        acc_h = __hfma2(wv5, xv5, acc_h);
        acc_h = __hfma2(wv6, xv6, acc_h);
        acc_h = __hfma2(wv7, xv7, acc_h);
        const float2 bs_sum = __half22float2(acc_h);
        acc[j] = fmaf(bs_sum.x + bs_sum.y, bs, acc[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (threadIdx.x == 0) {
      y[n0 + j] = __float2half_rn(v);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// FUSED v3: 3-way r/k/v M=1 GEMV — dual-block parallel iteration
// blockIdx.z selects which of r/k/v to compute (0=r, 1=k, 2=v)
// Uses shared mem byte→__half2 LUT + __hfma2 + dual-block ILP
// ═══════════════════════════════════════════════════════════════
template <int OutTile>
__global__ __launch_bounds__(32, 1) void linear_nvfp4_rkv_orig_row1_blk16_f16_kernel(
    int K, int N,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const uint8_t* __restrict__ w_r,
    const uint8_t* __restrict__ w_k,
    const uint8_t* __restrict__ w_v,
    const __nv_fp8_e4m3* __restrict__ bs_r,
    const __nv_fp8_e4m3* __restrict__ bs_k,
    const __nv_fp8_e4m3* __restrict__ bs_v,
    float ts_r, float ts_k, float ts_v,
    dtype* __restrict__ y_r,
    dtype* __restrict__ y_k,
    dtype* __restrict__ y_v) {
  // Select pointers based on blockIdx.z (uniform across warp — no divergence)
  const dtype* x;
  const uint8_t* w_nf4;
  const __nv_fp8_e4m3* b_scale;
  float t_scale;
  dtype* y;
  if (blockIdx.z == 0) { x = x_r; w_nf4 = w_r; b_scale = bs_r; t_scale = ts_r; y = y_r; }
  else if (blockIdx.z == 1) { x = x_k; w_nf4 = w_k; b_scale = bs_k; t_scale = ts_k; y = y_k; }
  else { x = x_v; w_nf4 = w_v; b_scale = bs_v; t_scale = ts_v; y = y_v; }

  const int n0 = blockIdx.x * OutTile;
  const int KB = K >> 4;
  const int K2 = K >> 1;

  // Shared memory LUT: byte (2 packed E2M1 codes) → __half2
  __shared__ __half2 byte_lut[256];
  for (int i = threadIdx.x; i < 256; i += 32) {
    unsigned short lo = e2m1_raw[i & 0xF];
    unsigned short hi = e2m1_raw[(i >> 4) & 0xF];
    uint32_t bits = lo | (static_cast<uint32_t>(hi) << 16);
    byte_lut[i] = *reinterpret_cast<__half2*>(&bits);
  }
  __syncthreads();

  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }

  // Dual-block: process kb and kb+KB/2 per iteration
  const int half_KB = KB >> 1;
  for (int kb = threadIdx.x; kb < half_KB; kb += 32) {
    // ── Block A: kb ──
    const int kA = kb << 4;
    const __half2* xA = reinterpret_cast<const __half2*>(x + kA);
    const __half2 xA0 = xA[0], xA1 = xA[1], xA2 = xA[2], xA3 = xA[3];
    const __half2 xA4 = xA[4], xA5 = xA[5], xA6 = xA[6], xA7 = xA[7];

    // ── Block B: kb + half_KB ──
    const int kB = (kb + half_KB) << 4;
    const __half2* xB = reinterpret_cast<const __half2*>(x + kB);
    const __half2 xB0 = xB[0], xB1 = xB[1], xB2 = xB[2], xB3 = xB[3];
    const __half2 xB4 = xB[4], xB5 = xB[5], xB6 = xB[6], xB7 = xB[7];

#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      // Block A
      const float bsA = float(b_scale[static_cast<int64_t>(n0 + j) * KB + kb]) * t_scale;
      const uint2 pkA = *reinterpret_cast<const uint2*>(
          w_nf4 + static_cast<int64_t>(n0 + j) * K2 + (kb << 3));
      const __half2 wA0 = byte_lut[pkA.x & 0xFF];
      const __half2 wA1 = byte_lut[(pkA.x >> 8) & 0xFF];
      const __half2 wA2 = byte_lut[(pkA.x >> 16) & 0xFF];
      const __half2 wA3 = byte_lut[pkA.x >> 24];
      const __half2 wA4 = byte_lut[pkA.y & 0xFF];
      const __half2 wA5 = byte_lut[(pkA.y >> 8) & 0xFF];
      const __half2 wA6 = byte_lut[(pkA.y >> 16) & 0xFF];
      const __half2 wA7 = byte_lut[pkA.y >> 24];
      __half2 accA = __hfma2(wA0, xA0, __float2half2_rn(0.0f));
      accA = __hfma2(wA1, xA1, accA);
      accA = __hfma2(wA2, xA2, accA);
      accA = __hfma2(wA3, xA3, accA);
      accA = __hfma2(wA4, xA4, accA);
      accA = __hfma2(wA5, xA5, accA);
      accA = __hfma2(wA6, xA6, accA);
      accA = __hfma2(wA7, xA7, accA);

      // Block B (independent chain)
      const float bsB = float(b_scale[static_cast<int64_t>(n0 + j) * KB + kb + half_KB]) * t_scale;
      const uint2 pkB = *reinterpret_cast<const uint2*>(
          w_nf4 + static_cast<int64_t>(n0 + j) * K2 + ((kb + half_KB) << 3));
      const __half2 wB0 = byte_lut[pkB.x & 0xFF];
      const __half2 wB1 = byte_lut[(pkB.x >> 8) & 0xFF];
      const __half2 wB2 = byte_lut[(pkB.x >> 16) & 0xFF];
      const __half2 wB3 = byte_lut[pkB.x >> 24];
      const __half2 wB4 = byte_lut[pkB.y & 0xFF];
      const __half2 wB5 = byte_lut[(pkB.y >> 8) & 0xFF];
      const __half2 wB6 = byte_lut[(pkB.y >> 16) & 0xFF];
      const __half2 wB7 = byte_lut[pkB.y >> 24];
      __half2 accB = __hfma2(wB0, xB0, __float2half2_rn(0.0f));
      accB = __hfma2(wB1, xB1, accB);
      accB = __hfma2(wB2, xB2, accB);
      accB = __hfma2(wB3, xB3, accB);
      accB = __hfma2(wB4, xB4, accB);
      accB = __hfma2(wB5, xB5, accB);
      accB = __hfma2(wB6, xB6, accB);
      accB = __hfma2(wB7, xB7, accB);

      // Merge A+B into FP32 accumulator
      const float2 sumA = __half22float2(accA);
      const float2 sumB = __half22float2(accB);
      acc[j] = fmaf(sumA.x + sumA.y, bsA, acc[j]);
      acc[j] = fmaf(sumB.x + sumB.y, bsB, acc[j]);
    }
  }
  // Handle odd KB
  if (KB & 1) {
    const int kb = threadIdx.x + half_KB * 2;
    if (kb < KB) {
      const int k = kb << 4;
      const __half2* x2 = reinterpret_cast<const __half2*>(x + k);
      const __half2 xv0 = x2[0], xv1 = x2[1], xv2 = x2[2], xv3 = x2[3];
      const __half2 xv4 = x2[4], xv5 = x2[5], xv6 = x2[6], xv7 = x2[7];
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const float bs = float(b_scale[static_cast<int64_t>(n0 + j) * KB + kb]) * t_scale;
        const uint2 packed = *reinterpret_cast<const uint2*>(
            w_nf4 + static_cast<int64_t>(n0 + j) * K2 + (kb << 3));
        const __half2 wv0 = byte_lut[packed.x & 0xFF];
        const __half2 wv1 = byte_lut[(packed.x >> 8) & 0xFF];
        const __half2 wv2 = byte_lut[(packed.x >> 16) & 0xFF];
        const __half2 wv3 = byte_lut[packed.x >> 24];
        const __half2 wv4 = byte_lut[packed.y & 0xFF];
        const __half2 wv5 = byte_lut[(packed.y >> 8) & 0xFF];
        const __half2 wv6 = byte_lut[(packed.y >> 16) & 0xFF];
        const __half2 wv7 = byte_lut[packed.y >> 24];
        __half2 acc_h = __hfma2(wv0, xv0, __float2half2_rn(0.0f));
        acc_h = __hfma2(wv1, xv1, acc_h);
        acc_h = __hfma2(wv2, xv2, acc_h);
        acc_h = __hfma2(wv3, xv3, acc_h);
        acc_h = __hfma2(wv4, xv4, acc_h);
        acc_h = __hfma2(wv5, xv5, acc_h);
        acc_h = __hfma2(wv6, xv6, acc_h);
        acc_h = __hfma2(wv7, xv7, acc_h);
        const float2 bs_sum = __half22float2(acc_h);
        acc[j] = fmaf(bs_sum.x + bs_sum.y, bs, acc[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (threadIdx.x == 0) {
      y[n0 + j] = __float2half_rn(v);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// OPTIMIZED: M=2 GEMV — 16-wide block kernel (uint2 vectorized load)
// Same approach as M=1 but with 2 accumulation sets
// ═══════════════════════════════════════════════════════════════
template <int OutTile>
__global__ __launch_bounds__(32, 1) void linear_nvfp4_orig_row2_blk16_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const uint8_t* __restrict__ w_nf4,
    const __nv_fp8_e4m3* __restrict__ b_scale,
    float t_scale,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int KB = K >> 4;
  const int K2 = K >> 1;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  for (int kb = threadIdx.x; kb < KB; kb += 32) {
    const int k = kb << 4;
    const float2 x00 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x01 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
    const float2 x02 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 4));
    const float2 x03 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 6));
    const float2 x04 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 8));
    const float2 x05 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 10));
    const float2 x06 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 12));
    const float2 x07 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 14));
    const float2 x10 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
    const float2 x11 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 2));
    const float2 x12 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 4));
    const float2 x13 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 6));
    const float2 x14 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 8));
    const float2 x15 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 10));
    const float2 x16 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 12));
    const float2 x17 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 14));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float bs = float(b_scale[static_cast<int64_t>(n0 + j) * KB + kb]) * t_scale;
      const uint2 packed = *reinterpret_cast<const uint2*>(
          w_nf4 + static_cast<int64_t>(n0 + j) * K2 + (kb << 3));
      const float wv0  = e2m1_decode_f(packed.x & 0xF) * bs;
      const float wv1  = e2m1_decode_f((packed.x >> 4) & 0xF) * bs;
      const float wv2  = e2m1_decode_f((packed.x >> 8) & 0xF) * bs;
      const float wv3  = e2m1_decode_f((packed.x >> 12) & 0xF) * bs;
      const float wv4  = e2m1_decode_f((packed.x >> 16) & 0xF) * bs;
      const float wv5  = e2m1_decode_f((packed.x >> 20) & 0xF) * bs;
      const float wv6  = e2m1_decode_f((packed.x >> 24) & 0xF) * bs;
      const float wv7  = e2m1_decode_f(packed.x >> 28) * bs;
      const float wv8  = e2m1_decode_f(packed.y & 0xF) * bs;
      const float wv9  = e2m1_decode_f((packed.y >> 4) & 0xF) * bs;
      const float wv10 = e2m1_decode_f((packed.y >> 8) & 0xF) * bs;
      const float wv11 = e2m1_decode_f((packed.y >> 12) & 0xF) * bs;
      const float wv12 = e2m1_decode_f((packed.y >> 16) & 0xF) * bs;
      const float wv13 = e2m1_decode_f((packed.y >> 20) & 0xF) * bs;
      const float wv14 = e2m1_decode_f((packed.y >> 24) & 0xF) * bs;
      const float wv15 = e2m1_decode_f(packed.y >> 28) * bs;
      acc0[j] = fmaf(x00.x, wv0, acc0[j]);
      acc0[j] = fmaf(x00.y, wv1, acc0[j]);
      acc0[j] = fmaf(x01.x, wv2, acc0[j]);
      acc0[j] = fmaf(x01.y, wv3, acc0[j]);
      acc0[j] = fmaf(x02.x, wv4, acc0[j]);
      acc0[j] = fmaf(x02.y, wv5, acc0[j]);
      acc0[j] = fmaf(x03.x, wv6, acc0[j]);
      acc0[j] = fmaf(x03.y, wv7, acc0[j]);
      acc0[j] = fmaf(x04.x, wv8, acc0[j]);
      acc0[j] = fmaf(x04.y, wv9, acc0[j]);
      acc0[j] = fmaf(x05.x, wv10, acc0[j]);
      acc0[j] = fmaf(x05.y, wv11, acc0[j]);
      acc0[j] = fmaf(x06.x, wv12, acc0[j]);
      acc0[j] = fmaf(x06.y, wv13, acc0[j]);
      acc0[j] = fmaf(x07.x, wv14, acc0[j]);
      acc0[j] = fmaf(x07.y, wv15, acc0[j]);
      acc1[j] = fmaf(x10.x, wv0, acc1[j]);
      acc1[j] = fmaf(x10.y, wv1, acc1[j]);
      acc1[j] = fmaf(x11.x, wv2, acc1[j]);
      acc1[j] = fmaf(x11.y, wv3, acc1[j]);
      acc1[j] = fmaf(x12.x, wv4, acc1[j]);
      acc1[j] = fmaf(x12.y, wv5, acc1[j]);
      acc1[j] = fmaf(x13.x, wv6, acc1[j]);
      acc1[j] = fmaf(x13.y, wv7, acc1[j]);
      acc1[j] = fmaf(x14.x, wv8, acc1[j]);
      acc1[j] = fmaf(x14.y, wv9, acc1[j]);
      acc1[j] = fmaf(x15.x, wv10, acc1[j]);
      acc1[j] = fmaf(x15.y, wv11, acc1[j]);
      acc1[j] = fmaf(x16.x, wv12, acc1[j]);
      acc1[j] = fmaf(x16.y, wv13, acc1[j]);
      acc1[j] = fmaf(x17.x, wv14, acc1[j]);
      acc1[j] = fmaf(x17.y, wv15, acc1[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (threadIdx.x == 0) {
      y[n0 + j] = __float2half_rn(v0);
      y[N + n0 + j] = __float2half_rn(v1);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// Host wrappers — optimized blk16 kernels (uint2 vectorized, 1-warp)
// ═══════════════════════════════════════════════════════════════

template <int OutTile>
at::Tensor linear_nvfp4_orig_row1_blk16_f16_cuda_impl(at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = w_nf4.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_nvfp4 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_nvfp4 requires N divisible by out_tile");
  TORCH_CHECK((k64 % 16) == 0, "linear_nvfp4 blk16 requires K divisible by 16");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 1, "linear_nvfp4 row1 requires one row");
  TORCH_CHECK(w_nf4.size(1) == K / 2, "linear_nvfp4 w_nf4 K/2 mismatch");
  TORCH_CHECK(b_scale.size(0) == N && b_scale.size(1) == K / 16, "linear_nvfp4 b_scale shape mismatch");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_nvfp4_orig_row1_blk16_f16_kernel<OutTile><<<N / OutTile, 32, 0, stream>>>(
      K, N, reinterpret_cast<const dtype*>(x.data_ptr()),
      w_nf4.data_ptr<uint8_t>(), reinterpret_cast<const __nv_fp8_e4m3*>(b_scale.data_ptr()),
      static_cast<float>(t_scale),
      reinterpret_cast<dtype*>(y.data_ptr()));
  return y;
}

// Fused 3-way r/k/v GEMV: single launch for r, k, v (M=1, NVFP4)
template <int OutTile>
at::Tensor linear_nvfp4_rkv_orig_row1_blk16_f16_cuda_impl(
    at::Tensor x_r, at::Tensor x_k, at::Tensor x_v,
    at::Tensor w_r, at::Tensor w_k, at::Tensor w_v,
    at::Tensor bs_r, at::Tensor bs_k, at::Tensor bs_v,
    double ts_r, double ts_k, double ts_v) {
  const int64_t k64 = x_r.size(-1);
  const int64_t n64 = w_r.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_nvfp4_rkv K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_nvfp4_rkv requires N divisible by out_tile");
  TORCH_CHECK((k64 % 16) == 0, "linear_nvfp4_rkv blk16 requires K divisible by 16");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  TORCH_CHECK(x_k.size(-1) == K && x_v.size(-1) == K, "rkv x K mismatch");
  TORCH_CHECK(w_k.size(0) == N && w_v.size(0) == N, "rkv w N mismatch");
  TORCH_CHECK(w_r.size(1) == K / 2 && w_k.size(1) == K / 2 && w_v.size(1) == K / 2, "rkv w K/2 mismatch");
  TORCH_CHECK(bs_r.size(0) == N && bs_r.size(1) == K / 16, "rkv bs_r shape mismatch");
  TORCH_CHECK(bs_k.size(0) == N && bs_k.size(1) == K / 16, "rkv bs_k shape mismatch");
  TORCH_CHECK(bs_v.size(0) == N && bs_v.size(1) == K / 16, "rkv bs_v shape mismatch");
  auto y = at::empty({3, N}, x_r.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  dtype* y_ptr = reinterpret_cast<dtype*>(y.data_ptr());
  linear_nvfp4_rkv_orig_row1_blk16_f16_kernel<OutTile><<<dim3(N / OutTile, 1, 3), 32, 0, stream>>>(
      K, N,
      reinterpret_cast<const dtype*>(x_r.data_ptr()),
      reinterpret_cast<const dtype*>(x_k.data_ptr()),
      reinterpret_cast<const dtype*>(x_v.data_ptr()),
      w_r.data_ptr<uint8_t>(), w_k.data_ptr<uint8_t>(), w_v.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_fp8_e4m3*>(bs_r.data_ptr()),
      reinterpret_cast<const __nv_fp8_e4m3*>(bs_k.data_ptr()),
      reinterpret_cast<const __nv_fp8_e4m3*>(bs_v.data_ptr()),
      static_cast<float>(ts_r), static_cast<float>(ts_k), static_cast<float>(ts_v),
      y_ptr, y_ptr + N, y_ptr + 2 * N);
  return y;
}

template <int OutTile>
at::Tensor linear_nvfp4_orig_row2_blk16_f16_cuda_impl(at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = w_nf4.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_nvfp4 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_nvfp4 requires N divisible by out_tile");
  TORCH_CHECK((k64 % 16) == 0, "linear_nvfp4 blk16 requires K divisible by 16");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 2, "linear_nvfp4 row2 requires two rows");
  TORCH_CHECK(w_nf4.size(1) == K / 2, "linear_nvfp4 w_nf4 K/2 mismatch");
  TORCH_CHECK(b_scale.size(0) == N && b_scale.size(1) == K / 16, "linear_nvfp4 b_scale shape mismatch");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_nvfp4_orig_row2_blk16_f16_kernel<OutTile><<<N / OutTile, 32, 0, stream>>>(
      K, N, reinterpret_cast<const dtype*>(x.data_ptr()),
      w_nf4.data_ptr<uint8_t>(), reinterpret_cast<const __nv_fp8_e4m3*>(b_scale.data_ptr()),
      static_cast<float>(t_scale),
      reinterpret_cast<dtype*>(y.data_ptr()));
  return y;
}

// ═══════════════════════════════════════════════════════════════
// Host wrappers — original (2-wide/4-wide) kernels
// ═══════════════════════════════════════════════════════════════

template <int Threads, int OutTile, bool Use4>
at::Tensor linear_nf4_orig_row1_exact_f16_cuda_impl(at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = w_nf4.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_nf4 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_nf4 requires N divisible by out_tile");
  TORCH_CHECK((k64 % (Use4 ? 4 : 2)) == 0, "linear_nf4 unsupported K alignment");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 1, "linear_nf4 row1 requires one row");
  TORCH_CHECK(w_nf4.size(1) == K / 2, "linear_nf4 w_nf4 K/2 mismatch");
  TORCH_CHECK(b_scale.size(0) == N && b_scale.size(1) == K / 16, "linear_nf4 b_scale shape mismatch");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (Use4) {
    linear_nf4_orig_row1_exact4_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, stream>>>(
        K, N, reinterpret_cast<const dtype*>(x.data_ptr()),
        w_nf4.data_ptr<uint8_t>(), reinterpret_cast<const __nv_fp8_e4m3*>(b_scale.data_ptr()),
        static_cast<float>(t_scale),
        reinterpret_cast<dtype*>(y.data_ptr()));
  } else {
    linear_nf4_orig_row1_exact_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, stream>>>(
        K, N, reinterpret_cast<const dtype*>(x.data_ptr()),
        w_nf4.data_ptr<uint8_t>(), reinterpret_cast<const __nv_fp8_e4m3*>(b_scale.data_ptr()),
        static_cast<float>(t_scale),
        reinterpret_cast<dtype*>(y.data_ptr()));
  }
  return y;
}

template <int Threads, int OutTile, bool Use4>
at::Tensor linear_nf4_orig_row2_exact_f16_cuda_impl(at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = w_nf4.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_nf4 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_nf4 requires N divisible by out_tile");
  TORCH_CHECK((k64 % (Use4 ? 4 : 2)) == 0, "linear_nf4 unsupported K alignment");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 2, "linear_nf4 row2 requires two rows");
  TORCH_CHECK(w_nf4.size(1) == K / 2, "linear_nf4 w_nf4 K/2 mismatch");
  TORCH_CHECK(b_scale.size(0) == N && b_scale.size(1) == K / 16, "linear_nf4 b_scale shape mismatch");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (Use4) {
    linear_nf4_orig_row2_exact4_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, stream>>>(
        K, N, reinterpret_cast<const dtype*>(x.data_ptr()),
        w_nf4.data_ptr<uint8_t>(), reinterpret_cast<const __nv_fp8_e4m3*>(b_scale.data_ptr()),
        static_cast<float>(t_scale),
        reinterpret_cast<dtype*>(y.data_ptr()));
  } else {
    linear_nf4_orig_row2_exact_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, stream>>>(
        K, N, reinterpret_cast<const dtype*>(x.data_ptr()),
        w_nf4.data_ptr<uint8_t>(), reinterpret_cast<const __nv_fp8_e4m3*>(b_scale.data_ptr()),
        static_cast<float>(t_scale),
        reinterpret_cast<dtype*>(y.data_ptr()));
  }
  return y;
}

template <int Threads, int RowTile, int OutTile>
at::Tensor linear_nf4_orig_rows_f16_cuda_impl(at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = w_nf4.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_nf4 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  const int M = static_cast<int>(m64);
  TORCH_CHECK(w_nf4.size(1) == K / 2, "linear_nf4 w_nf4 K/2 mismatch");
  TORCH_CHECK(b_scale.size(0) == N && b_scale.size(1) == K / 16, "linear_nf4 b_scale shape mismatch");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(ceil_div(N, OutTile), ceil_div(M, RowTile), 1);
  linear_nf4_orig_rows_f16_kernel<Threads, RowTile, OutTile><<<grid, Threads, 0, stream>>>(
      M, K, N, reinterpret_cast<const dtype*>(x.data_ptr()),
      w_nf4.data_ptr<uint8_t>(), reinterpret_cast<const __nv_fp8_e4m3*>(b_scale.data_ptr()),
      static_cast<float>(t_scale),
      reinterpret_cast<dtype*>(y.data_ptr()));
  return y;
}

// ═══════════════════════════════════════════════════════════════
// NF4 cmix_sparse kernels: sparse mat-vec with NF4-quantized value weights
// Same sparse selection pattern as rwkv7_fast_ops_fp16.cu cmix_sparse_spmv_relu_*
// Weight: uint8_t [F, C/2] packed, b_scale: __half [F, C/16]
// ═══════════════════════════════════════════════════════════════

constexpr int NF4_CMIX_THREADS = 128;
constexpr int NF4_CMIX_TILE = 128;
// Vectorized: each thread reads 4 bytes (uint32 = 8 E2M1 codes = 4 __half2 pairs)
// Block scale: 1 read covers 8 elements (8 < 16, always same block)
constexpr int NF4_CMIX_VEC = 8;  // columns per thread

// rows=1: relu^2 activation + sparse mat-vec with NF4 weights (vectorized)
__global__ __launch_bounds__(NF4_CMIX_THREADS, 4) void cmix_sparse_spmv_relu_one_nf4_kernel(
    int C,
    const dtype* __restrict__ preact,
    const uint8_t* __restrict__ w_nf4,
    const __nv_fp8_e4m3* __restrict__ b_scale,
    float t_scale,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[NF4_CMIX_TILE];
  __shared__ __align__(256) int nnz_ids[NF4_CMIX_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[NF4_CMIX_TILE / 32];
  __shared__ int warp_prefix[NF4_CMIX_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * NF4_CMIX_TILE;

  if (tid < NF4_CMIX_TILE) {
    const float v = fmaxf(__half2float(*reinterpret_cast<const __half*>(preact + start_f + tid)), 0.0f);
    vec_slice[tid] = __float2half_rn(v * v);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < NF4_CMIX_TILE) {
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
    for (int w = 0; w < NF4_CMIX_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < NF4_CMIX_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  // Vectorized NF4 sparse mat-vec: uint32 read = 8 E2M1 codes, __hfma2 accumulation
  const int C2 = C >> 1;
  const int CB = C >> 4;
  const int col = c_block * (NF4_CMIX_VEC * NF4_CMIX_THREADS) + tid * NF4_CMIX_VEC;
  if (col >= C) return;
  const int col2 = col >> 1;   // byte offset (4-byte aligned)
  const int colb = col >> 4;  // block index (8 elements < 16, same block)
  __half2 acc0, acc1, acc2, acc3;
  *reinterpret_cast<int*>(&acc0) = 0;
  *reinterpret_cast<int*>(&acc1) = 0;
  *reinterpret_cast<int*>(&acc2) = 0;
  *reinterpret_cast<int*>(&acc3) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 vs2 = __half2half2(vec_slice[nnz_ids[i]]);
    // Read 4 bytes = 8 E2M1 codes in one transaction
    const uint32_t packed4 = *reinterpret_cast<const uint32_t*>(
        w_nf4 + static_cast<int64_t>(actual_f) * C2 + col2);
    const uint8_t* pp = reinterpret_cast<const uint8_t*>(&packed4);
    // 1 block scale read for all 8 elements
    const __half bs = __float2half(float(b_scale[static_cast<int64_t>(actual_f) * CB + colb]) * t_scale);
    // Decode 4 pairs, apply block scale, accumulate with __hfma2
    acc0 = __hfma2(vs2, __halves2half2(
        __hmul(e2m1_decode_h(pp[0] & 0x0F), bs),
        __hmul(e2m1_decode_h(pp[0] >> 4), bs)), acc0);
    acc1 = __hfma2(vs2, __halves2half2(
        __hmul(e2m1_decode_h(pp[1] & 0x0F), bs),
        __hmul(e2m1_decode_h(pp[1] >> 4), bs)), acc1);
    acc2 = __hfma2(vs2, __halves2half2(
        __hmul(e2m1_decode_h(pp[2] & 0x0F), bs),
        __hmul(e2m1_decode_h(pp[2] >> 4), bs)), acc2);
    acc3 = __hfma2(vs2, __halves2half2(
        __hmul(e2m1_decode_h(pp[3] & 0x0F), bs),
        __hmul(e2m1_decode_h(pp[3] >> 4), bs)), acc3);
  }
  // Bounds-checked atomicAdd
  if (col + 7 < C) {
    atomicAdd(reinterpret_cast<__half2*>(out + col), acc0);
    atomicAdd(reinterpret_cast<__half2*>(out + col + 2), acc1);
    atomicAdd(reinterpret_cast<__half2*>(out + col + 4), acc2);
    atomicAdd(reinterpret_cast<__half2*>(out + col + 6), acc3);
  } else {
    if (col     < C) atomicAdd(reinterpret_cast<__half2*>(out + col),     acc0);
    if (col + 2 < C) atomicAdd(reinterpret_cast<__half2*>(out + col + 2), acc1);
    if (col + 4 < C) atomicAdd(reinterpret_cast<__half2*>(out + col + 4), acc2);
    if (col + 6 < C) atomicAdd(reinterpret_cast<__half2*>(out + col + 6), acc3);
  }
}

// rows=2..19: vectorized, same as one but with row index
__global__ __launch_bounds__(NF4_CMIX_THREADS, 4) void cmix_sparse_spmv_relu_rows_nf4_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const uint8_t* __restrict__ w_nf4,
    const __nv_fp8_e4m3* __restrict__ b_scale,
    float t_scale,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[NF4_CMIX_TILE];
  __shared__ __align__(256) int nnz_ids[NF4_CMIX_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[NF4_CMIX_TILE / 32];
  __shared__ int warp_prefix[NF4_CMIX_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * NF4_CMIX_TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

  if (tid < NF4_CMIX_TILE) {
    const float v = fmaxf(__half2float(*reinterpret_cast<const __half*>(pre_row + start_f + tid)), 0.0f);
    vec_slice[tid] = __float2half_rn(v * v);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < NF4_CMIX_TILE) {
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
    for (int w = 0; w < NF4_CMIX_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < NF4_CMIX_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  const int C2 = C >> 1;
  const int CB = C >> 4;
  const int col = c_block * (NF4_CMIX_VEC * NF4_CMIX_THREADS) + tid * NF4_CMIX_VEC;
  if (col >= C) return;
  const int col2 = col >> 1;
  const int colb = col >> 4;
  __half2 acc0, acc1, acc2, acc3;
  *reinterpret_cast<int*>(&acc0) = 0;
  *reinterpret_cast<int*>(&acc1) = 0;
  *reinterpret_cast<int*>(&acc2) = 0;
  *reinterpret_cast<int*>(&acc3) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 vs2 = __half2half2(vec_slice[nnz_ids[i]]);
    const uint32_t packed4 = *reinterpret_cast<const uint32_t*>(
        w_nf4 + static_cast<int64_t>(actual_f) * C2 + col2);
    const uint8_t* pp = reinterpret_cast<const uint8_t*>(&packed4);
    const __half bs = __float2half(float(b_scale[static_cast<int64_t>(actual_f) * CB + colb]) * t_scale);
    acc0 = __hfma2(vs2, __halves2half2(
        __hmul(e2m1_decode_h(pp[0] & 0x0F), bs),
        __hmul(e2m1_decode_h(pp[0] >> 4), bs)), acc0);
    acc1 = __hfma2(vs2, __halves2half2(
        __hmul(e2m1_decode_h(pp[1] & 0x0F), bs),
        __hmul(e2m1_decode_h(pp[1] >> 4), bs)), acc1);
    acc2 = __hfma2(vs2, __halves2half2(
        __hmul(e2m1_decode_h(pp[2] & 0x0F), bs),
        __hmul(e2m1_decode_h(pp[2] >> 4), bs)), acc2);
    acc3 = __hfma2(vs2, __halves2half2(
        __hmul(e2m1_decode_h(pp[3] & 0x0F), bs),
        __hmul(e2m1_decode_h(pp[3] >> 4), bs)), acc3);
  }
  if (col + 7 < C) {
    atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col), acc0);
    atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col + 2), acc1);
    atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col + 4), acc2);
    atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col + 6), acc3);
  } else {
    if (col     < C) atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col),     acc0);
    if (col + 2 < C) atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col + 2), acc1);
    if (col + 4 < C) atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col + 4), acc2);
    if (col + 6 < C) atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col + 6), acc3);
  }
}

// rows>=8, T=512 tile variant (vectorized)
__global__ __launch_bounds__(256, 2) void cmix_sparse_spmv_relu_rows_t512_nf4_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const uint8_t* __restrict__ w_nf4,
    const __nv_fp8_e4m3* __restrict__ b_scale,
    float t_scale,
    dtype* __restrict__ out) {
  constexpr int TILE = 512;
  constexpr int THREADS = 256;
  constexpr int VEC = 8;  // columns per thread
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
    const float v = fmaxf(__half2float(*reinterpret_cast<const __half*>(pre_row + start_f + local_f)), 0.0f);
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

  const int C2 = C >> 1;
  const int CB = C >> 4;
  const int col = c_block * (VEC * THREADS) + tid * VEC;
  if (col >= C) return;
  const int col2 = col >> 1;
  const int colb = col >> 4;
  __half2 acc0, acc1, acc2, acc3;
  *reinterpret_cast<int*>(&acc0) = 0;
  *reinterpret_cast<int*>(&acc1) = 0;
  *reinterpret_cast<int*>(&acc2) = 0;
  *reinterpret_cast<int*>(&acc3) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int local_f = nnz_ids[i];
    const int actual_f = start_f + local_f;
    const __half2 vs2 = __half2half2(vec_slice[local_f]);
    const uint32_t packed4 = *reinterpret_cast<const uint32_t*>(
        w_nf4 + static_cast<int64_t>(actual_f) * C2 + col2);
    const uint8_t* pp = reinterpret_cast<const uint8_t*>(&packed4);
    const __half bs = __float2half(float(b_scale[static_cast<int64_t>(actual_f) * CB + colb]) * t_scale);
    acc0 = __hfma2(vs2, __halves2half2(
        __hmul(e2m1_decode_h(pp[0] & 0x0F), bs),
        __hmul(e2m1_decode_h(pp[0] >> 4), bs)), acc0);
    acc1 = __hfma2(vs2, __halves2half2(
        __hmul(e2m1_decode_h(pp[1] & 0x0F), bs),
        __hmul(e2m1_decode_h(pp[1] >> 4), bs)), acc1);
    acc2 = __hfma2(vs2, __halves2half2(
        __hmul(e2m1_decode_h(pp[2] & 0x0F), bs),
        __hmul(e2m1_decode_h(pp[2] >> 4), bs)), acc2);
    acc3 = __hfma2(vs2, __halves2half2(
        __hmul(e2m1_decode_h(pp[3] & 0x0F), bs),
        __hmul(e2m1_decode_h(pp[3] >> 4), bs)), acc3);
  }
  if (col + 7 < C) {
    atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col), acc0);
    atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col + 2), acc1);
    atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col + 4), acc2);
    atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col + 6), acc3);
  } else {
    if (col     < C) atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col),     acc0);
    if (col + 2 < C) atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col + 2), acc1);
    if (col + 4 < C) atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col + 4), acc2);
    if (col + 6 < C) atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + col + 6), acc3);
  }
}

// ═══════════════════════════════════════════════════════════════
// NF4 cmix wrapper functions (internal, in anonymous namespace)
// ═══════════════════════════════════════════════════════════════

at::Tensor cmix_sparse_down_relu_one_nf4_impl(
    int C, int F, at::Tensor preact, at::Tensor w_nf4, at::Tensor b_scale, double t_scale) {
  auto out = at::zeros({1, 1, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  cmix_sparse_spmv_relu_one_nf4_kernel<<<dim3(F / NF4_CMIX_TILE, ceil_div(C, NF4_CMIX_VEC * NF4_CMIX_THREADS), 1), NF4_CMIX_THREADS, 0, stream>>>(
      C,
      reinterpret_cast<const dtype*>(preact.data_ptr()),
      w_nf4.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_fp8_e4m3*>(b_scale.data_ptr()),
      static_cast<float>(t_scale),
      reinterpret_cast<dtype*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_rows_nf4_impl(
    int B, int T, int C, int F, at::Tensor preact, at::Tensor w_nf4, at::Tensor b_scale, double t_scale) {
  const int rows = B * T;
  auto out = at::zeros({B, T, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  cmix_sparse_spmv_relu_rows_nf4_kernel<<<dim3(F / NF4_CMIX_TILE, ceil_div(C, NF4_CMIX_VEC * NF4_CMIX_THREADS), rows), NF4_CMIX_THREADS, 0, stream>>>(
      C, F,
      reinterpret_cast<const dtype*>(preact.data_ptr()),
      w_nf4.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_fp8_e4m3*>(b_scale.data_ptr()),
      static_cast<float>(t_scale),
      reinterpret_cast<dtype*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_rows_t512_nf4_impl(
    int B, int T, int C, int F, at::Tensor preact, at::Tensor w_nf4, at::Tensor b_scale, double t_scale) {
  const int rows = B * T;
  auto out = at::zeros({B, T, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  cmix_sparse_spmv_relu_rows_t512_nf4_kernel<<<dim3(F / 512, ceil_div(C, 8 * 256), rows), 256, 0, stream>>>(
      C, F,
      reinterpret_cast<const dtype*>(preact.data_ptr()),
      w_nf4.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_fp8_e4m3*>(b_scale.data_ptr()),
      static_cast<float>(t_scale),
      reinterpret_cast<dtype*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

} // namespace

// ═══════════════════════════════════════════════════════════════
// NF4 cmix entry points (external, called from .cpp)
// ═══════════════════════════════════════════════════════════════

at::Tensor cmix_sparse_down_relu_one_nf4_cuda(
    at::Tensor preact, at::Tensor w_nf4, at::Tensor b_scale, double t_scale, int64_t C, int64_t F) {
  TORCH_CHECK(preact.is_cuda() && preact.is_contiguous(), "preact must be CUDA contiguous");
  TORCH_CHECK(w_nf4.is_cuda() && w_nf4.is_contiguous(), "w_nf4 must be CUDA contiguous");
  TORCH_CHECK(b_scale.is_cuda() && b_scale.is_contiguous(), "b_scale must be CUDA contiguous");
  TORCH_CHECK(preact.scalar_type() == at::kHalf, "preact must be fp16");
  TORCH_CHECK(w_nf4.scalar_type() == at::kByte, "w_nf4 must be uint8");
  TORCH_CHECK(b_scale.scalar_type() == at::kFloat8_e4m3fn, "b_scale must be float8_e4m3fn");
  return cmix_sparse_down_relu_one_nf4_impl(static_cast<int>(C), static_cast<int>(F), preact, w_nf4, b_scale, t_scale);
}

at::Tensor cmix_sparse_down_relu_rows_nf4_cuda(
    at::Tensor preact, at::Tensor w_nf4, at::Tensor b_scale, double t_scale,
    int64_t B, int64_t T, int64_t C, int64_t F) {
  TORCH_CHECK(preact.is_cuda() && preact.is_contiguous(), "preact must be CUDA contiguous");
  TORCH_CHECK(w_nf4.is_cuda() && w_nf4.is_contiguous(), "w_nf4 must be CUDA contiguous");
  TORCH_CHECK(b_scale.is_cuda() && b_scale.is_contiguous(), "b_scale must be CUDA contiguous");
  TORCH_CHECK(preact.scalar_type() == at::kHalf, "preact must be fp16");
  TORCH_CHECK(w_nf4.scalar_type() == at::kByte, "w_nf4 must be uint8");
  TORCH_CHECK(b_scale.scalar_type() == at::kFloat8_e4m3fn, "b_scale must be float8_e4m3fn");
  return cmix_sparse_down_relu_rows_nf4_impl(static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F), preact, w_nf4, b_scale, t_scale);
}

at::Tensor cmix_sparse_down_relu_rows_t512_nf4_cuda(
    at::Tensor preact, at::Tensor w_nf4, at::Tensor b_scale, double t_scale,
    int64_t B, int64_t T, int64_t C, int64_t F) {
  TORCH_CHECK(preact.is_cuda() && preact.is_contiguous(), "preact must be CUDA contiguous");
  TORCH_CHECK(w_nf4.is_cuda() && w_nf4.is_contiguous(), "w_nf4 must be CUDA contiguous");
  TORCH_CHECK(b_scale.is_cuda() && b_scale.is_contiguous(), "b_scale must be CUDA contiguous");
  TORCH_CHECK(preact.scalar_type() == at::kHalf, "preact must be fp16");
  TORCH_CHECK(w_nf4.scalar_type() == at::kByte, "w_nf4 must be uint8");
  TORCH_CHECK(b_scale.scalar_type() == at::kFloat8_e4m3fn, "b_scale must be float8_e4m3fn");
  return cmix_sparse_down_relu_rows_t512_nf4_impl(static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F), preact, w_nf4, b_scale, t_scale);
}

// ═══════════════════════════════════════════════════════════════
// C++ entry points (called from .cpp)
// ═══════════════════════════════════════════════════════════════

at::Tensor linear_nf4_orig_rows_exact_f16_cuda(
    at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale, int64_t threads, int64_t out_tile, bool use4) {
  // Dispatch by rows (same as v3a: x.numel() / x.size(-1))
  const int64_t rows = x.numel() / x.size(-1);
  if (rows == 1) {
    if (!use4 && threads == 128 && out_tile == 2) return linear_nf4_orig_row1_exact_f16_cuda_impl<128, 2, false>(x, w_nf4, b_scale, t_scale);
    if (use4 && threads == 128 && out_tile == 2) return linear_nf4_orig_row1_exact_f16_cuda_impl<128, 2, true>(x, w_nf4, b_scale, t_scale);
  }
  if (rows == 2) {
    if (use4 && threads == 64 && out_tile == 2) return linear_nf4_orig_row2_exact_f16_cuda_impl<64, 2, true>(x, w_nf4, b_scale, t_scale);
    if (use4 && threads == 256 && out_tile == 1) return linear_nf4_orig_row2_exact_f16_cuda_impl<256, 1, true>(x, w_nf4, b_scale, t_scale);
    if (!use4 && threads == 128 && out_tile == 2) return linear_nf4_orig_row2_exact_f16_cuda_impl<128, 2, false>(x, w_nf4, b_scale, t_scale);
  }
  TORCH_CHECK(false, "unsupported linear_nf4_orig_rows_exact_f16 rows/threads/out_tile/use4");
}

// Optimized blk16 entry points (uint2 vectorized, 1-warp, always K%16==0)
at::Tensor linear_nvfp4_orig_row1_blk16_f16_cuda(
    at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale, int64_t out_tile) {
  if (out_tile == 1) return linear_nvfp4_orig_row1_blk16_f16_cuda_impl<1>(x, w_nf4, b_scale, t_scale);
  if (out_tile == 2) return linear_nvfp4_orig_row1_blk16_f16_cuda_impl<2>(x, w_nf4, b_scale, t_scale);
  if (out_tile == 4) return linear_nvfp4_orig_row1_blk16_f16_cuda_impl<4>(x, w_nf4, b_scale, t_scale);
  if (out_tile == 8) return linear_nvfp4_orig_row1_blk16_f16_cuda_impl<8>(x, w_nf4, b_scale, t_scale);
  TORCH_CHECK(false, "unsupported linear_nvfp4_orig_row1_blk16 out_tile");
}

// Fused r/k/v entry point
at::Tensor linear_nvfp4_rkv_orig_row1_blk16_f16_cuda(
    at::Tensor x_r, at::Tensor x_k, at::Tensor x_v,
    at::Tensor w_r, at::Tensor w_k, at::Tensor w_v,
    at::Tensor bs_r, at::Tensor bs_k, at::Tensor bs_v,
    double ts_r, double ts_k, double ts_v, int64_t out_tile) {
  if (out_tile == 1) return linear_nvfp4_rkv_orig_row1_blk16_f16_cuda_impl<1>(x_r, x_k, x_v, w_r, w_k, w_v, bs_r, bs_k, bs_v, ts_r, ts_k, ts_v);
  if (out_tile == 2) return linear_nvfp4_rkv_orig_row1_blk16_f16_cuda_impl<2>(x_r, x_k, x_v, w_r, w_k, w_v, bs_r, bs_k, bs_v, ts_r, ts_k, ts_v);
  if (out_tile == 4) return linear_nvfp4_rkv_orig_row1_blk16_f16_cuda_impl<4>(x_r, x_k, x_v, w_r, w_k, w_v, bs_r, bs_k, bs_v, ts_r, ts_k, ts_v);
  if (out_tile == 8) return linear_nvfp4_rkv_orig_row1_blk16_f16_cuda_impl<8>(x_r, x_k, x_v, w_r, w_k, w_v, bs_r, bs_k, bs_v, ts_r, ts_k, ts_v);
  TORCH_CHECK(false, "unsupported linear_nvfp4_rkv_orig_row1_blk16 out_tile");
}

at::Tensor linear_nvfp4_orig_row2_blk16_f16_cuda(
    at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale, int64_t out_tile) {
  if (out_tile == 1) return linear_nvfp4_orig_row2_blk16_f16_cuda_impl<1>(x, w_nf4, b_scale, t_scale);
  if (out_tile == 2) return linear_nvfp4_orig_row2_blk16_f16_cuda_impl<2>(x, w_nf4, b_scale, t_scale);
  TORCH_CHECK(false, "unsupported linear_nvfp4_orig_row2_blk16 out_tile");
}

at::Tensor linear_nf4_orig_rows_f16_cuda(
    at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale, int64_t row_tile, int64_t out_tile) {
  // Dispatch by row_tile/out_tile (same options as v3a)
  if (row_tile == 1 && out_tile == 2) return linear_nf4_orig_rows_f16_cuda_impl<128, 1, 2>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 1 && out_tile == 4) return linear_nf4_orig_rows_f16_cuda_impl<128, 1, 4>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 1 && out_tile == 8) return linear_nf4_orig_rows_f16_cuda_impl<128, 1, 8>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 1 && out_tile == 16) return linear_nf4_orig_rows_f16_cuda_impl<128, 1, 16>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 2 && out_tile == 2) return linear_nf4_orig_rows_f16_cuda_impl<128, 2, 2>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 2 && out_tile == 4) return linear_nf4_orig_rows_f16_cuda_impl<128, 2, 4>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 2 && out_tile == 8) return linear_nf4_orig_rows_f16_cuda_impl<128, 2, 8>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 3 && out_tile == 2) return linear_nf4_orig_rows_f16_cuda_impl<128, 3, 2>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 3 && out_tile == 4) return linear_nf4_orig_rows_f16_cuda_impl<128, 3, 4>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 4 && out_tile == 2) return linear_nf4_orig_rows_f16_cuda_impl<128, 4, 2>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 4 && out_tile == 4) return linear_nf4_orig_rows_f16_cuda_impl<128, 4, 4>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 8 && out_tile == 2) return linear_nf4_orig_rows_f16_cuda_impl<128, 8, 2>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 8 && out_tile == 4) return linear_nf4_orig_rows_f16_cuda_impl<128, 8, 4>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 16 && out_tile == 2) return linear_nf4_orig_rows_f16_cuda_impl<128, 16, 2>(x, w_nf4, b_scale, t_scale);
  if (row_tile == 16 && out_tile == 4) return linear_nf4_orig_rows_f16_cuda_impl<128, 16, 4>(x, w_nf4, b_scale, t_scale);
  TORCH_CHECK(false, "unsupported linear_nf4_orig_rows_f16 row_tile/out_tile");
}

at::Tensor dequant_nf4_to_f16_cuda(at::Tensor w_nf4, at::Tensor b_scale, double t_scale, bool transpose) {
  const int N = static_cast<int>(w_nf4.size(0));
  const int K = static_cast<int>(w_nf4.size(1)) * 2;
  TORCH_CHECK(b_scale.size(0) == N && b_scale.size(1) == K / 16, "b_scale shape mismatch");
  auto out = at::empty(transpose ? std::vector<int64_t>{K, N} : std::vector<int64_t>{N, K}, w_nf4.options().dtype(at::kHalf));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  if (transpose) {
    dequant_nf4_to_f16_kernel<256, true><<<N, threads, 0, stream>>>(
        N, K, w_nf4.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_fp8_e4m3*>(b_scale.data_ptr()),
        static_cast<float>(t_scale),
        reinterpret_cast<dtype*>(out.data_ptr()));
  } else {
    dequant_nf4_to_f16_kernel<256, false><<<N, threads, 0, stream>>>(
        N, K, w_nf4.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_fp8_e4m3*>(b_scale.data_ptr()),
        static_cast<float>(t_scale),
        reinterpret_cast<dtype*>(out.data_ptr()));
  }
  return out;
}
