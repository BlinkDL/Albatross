// ═══════════════════════════════════════════════════════════════════════
// rwkv7_int8_ops.cu — INT8 量化推理 CUDA kernel
//
// 在原版 v3a fp16 GEMV kernel 基础上，将权重从 fp16 改为 int8 + per-channel scale。
// x 保持 fp16 不变，kernel 内读 int8 权重时即时乘 scale 还原为 float 再做 fmaf。
// 权重读取量减半，GEMV 带宽瓶颈场景下提速明显。
// kernel 结构（grid/block/reduce）与原版完全一致，仅权重读取方式不同。
// ═══════════════════════════════════════════════════════════════════════

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <climits>

using dtype = __half;

// warp 内 butterfly reduce 求和
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    x += __shfl_down_sync(0xffffffffu, x, offset);
  return x;
}

// warp 内求最大值，用于 M>1 fallback kernel
__device__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  return val;
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel A: M=1 GEMV，int8 权重 [N,K] + scale [N]
//   Use4=false → 2-wide K 循环
//   Use4=true  → 4-wide K 循环
//
// 输入：x [K] fp16, w_orig [N,K] int8, scale [N] fp16
// 输出：y [N] fp16
// ═══════════════════════════════════════════════════════════════════════

template <int Threads, int OutTile, bool Use4>
__global__ __launch_bounds__(Threads, 10) void linear_int8_orig_row1_exact_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const int8_t* __restrict__ w_orig,
    const dtype* __restrict__ scale,
    dtype* __restrict__ y) {

  const int n0 = blockIdx.x * OutTile;

  float acc[OutTile];
  float s[OutTile];  // scale 预读到寄存器，避免 K 循环内重复读 global memory

#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
    s[j] = __half2float(scale[n0 + j]);
  }

  if constexpr (Use4) {
    // 4-wide: 每次迭代处理 4 个 K 元素
    for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
      const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
      const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const int8_t* wj = w_orig + static_cast<int64_t>(n0 + j) * K + k;
        acc[j] = fmaf(x0.x, (float)wj[0] * s[j], acc[j]);
        acc[j] = fmaf(x0.y, (float)wj[1] * s[j], acc[j]);
        acc[j] = fmaf(x1.x, (float)wj[2] * s[j], acc[j]);
        acc[j] = fmaf(x1.y, (float)wj[3] * s[j], acc[j]);
      }
    }
  } else {
    // 2-wide: 每次迭代处理 2 个 K 元素
    for (int k2 = threadIdx.x; k2 < (K >> 1); k2 += Threads) {
      const int k = k2 << 1;
      const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + k));
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const int8_t* wj = w_orig + static_cast<int64_t>(n0 + j) * K + k;
        acc[j] = fmaf(xv.x, (float)wj[0] * s[j], acc[j]);
        acc[j] = fmaf(xv.y, (float)wj[1] * s[j], acc[j]);
      }
    }
  }

  // warp 内 reduce → shared memory → thread 0 求和写出
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

// ═══════════════════════════════════════════════════════════════════════
// Kernel B: M=2 GEMV，两行输入共享 int8 权重读取
//
// 输入：x [2,K] fp16, w_orig [N,K] int8, scale [N] fp16
// 输出：y [2,N] fp16
// ═══════════════════════════════════════════════════════════════════════

template <int Threads, int OutTile, bool Use4>
__global__ __launch_bounds__(Threads, 10) void linear_int8_orig_row2_exact_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const int8_t* __restrict__ w_orig,
    const dtype* __restrict__ scale,
    dtype* __restrict__ y) {

  const int n0 = blockIdx.x * OutTile;
  float acc0[OutTile];
  float acc1[OutTile];
  float s[OutTile];

#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
    s[j] = __half2float(scale[n0 + j]);
  }

  if constexpr (Use4) {
    for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
      const float2 x00 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
      const float2 x01 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
      const float2 x10 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
      const float2 x11 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 2));
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const int8_t* wj = w_orig + static_cast<int64_t>(n0 + j) * K + k;
        acc0[j] = fmaf(x00.x, (float)wj[0] * s[j], acc0[j]);
        acc0[j] = fmaf(x00.y, (float)wj[1] * s[j], acc0[j]);
        acc0[j] = fmaf(x01.x, (float)wj[2] * s[j], acc0[j]);
        acc0[j] = fmaf(x01.y, (float)wj[3] * s[j], acc0[j]);
        acc1[j] = fmaf(x10.x, (float)wj[0] * s[j], acc1[j]);
        acc1[j] = fmaf(x10.y, (float)wj[1] * s[j], acc1[j]);
        acc1[j] = fmaf(x11.x, (float)wj[2] * s[j], acc1[j]);
        acc1[j] = fmaf(x11.y, (float)wj[3] * s[j], acc1[j]);
      }
    }
  } else {
    for (int k2 = threadIdx.x; k2 < (K >> 1); k2 += Threads) {
      const int k = k2 << 1;
      const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
      const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const int8_t* wj = w_orig + static_cast<int64_t>(n0 + j) * K + k;
        acc0[j] = fmaf(x0.x, (float)wj[0] * s[j], acc0[j]);
        acc0[j] = fmaf(x0.y, (float)wj[1] * s[j], acc0[j]);
        acc1[j] = fmaf(x1.x, (float)wj[0] * s[j], acc1[j]);
        acc1[j] = fmaf(x1.y, (float)wj[1] * s[j], acc1[j]);
      }
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

// ═══════════════════════════════════════════════════════════════════════
// Kernel C: M>1 fallback，per-token 量化 x 后做 int8×int8 matmul
// 仅用于 M>2 的 prefill 场景，实际推理极少触发
// ═══════════════════════════════════════════════════════════════════════

template <int Threads>
__global__ void linear_int8_f16_kernel(
    const dtype* __restrict__ x,
    const int8_t* __restrict__ w_int8,
    const dtype* __restrict__ w_scale,
    dtype* __restrict__ y,
    int M, int K, int N) {

  __shared__ int8_t xs[16384];
  __shared__ float s_max_abs;

  int row = blockIdx.x;
  if (row >= M) return;

  // per-token 量化 x
  float local_max = 0.0f;
  for (int k = threadIdx.x; k < K; k += Threads) {
    float v = __half2float(x[row * (int64_t)K + k]);
    local_max = fmaxf(local_max, fabsf(v));
  }
  local_max = warp_reduce_max(local_max);
  if (threadIdx.x == 0) s_max_abs = fmaxf(local_max, 1e-10f);
  __syncthreads();

  float scale_x = s_max_abs / 127.0f;

  for (int k = threadIdx.x; k < K; k += Threads) {
    xs[k] = (int8_t)max(-128, min(127, __float2int_rn(__half2float(x[row * (int64_t)K + k]) / scale_x)));
  }
  __syncthreads();

  for (int col_base = threadIdx.x; col_base < N; col_base += Threads) {
    int col = col_base;
    int32_t acc = 0;
    for (int k = 0; k < K; k += 1) {
      acc += (int32_t)xs[k] * (int32_t)w_int8[k * (int64_t)N + col];
    }
    y[row * (int64_t)N + col] = __float2half((float)acc * scale_x * __half2float(w_scale[col]));
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel E: M≥3 GEMM，int8 权重 [N,K] + scale [N]
//   1:1 复制原版 linear_orig_rows_f16_kernel，权重改为 int8 + scale
//   用于 prefill (M>2) 场景，替代 dequant + cuBLAS 路径，消除临时 fp16 张量
//
// 输入：x [M,K] fp16, w_orig [N,K] int8, scale [N] fp16
// 输出：y [M,N] fp16
// ═══════════════════════════════════════════════════════════════════════

template <int Threads, int RowTile, int OutTile, bool Use4>
__global__ __launch_bounds__(Threads, 1) void linear_int8_orig_rows_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const int8_t* __restrict__ w_orig,
    const dtype* __restrict__ scale,
    dtype* __restrict__ y) {

  const int n0 = blockIdx.x * OutTile;
  const int m0 = blockIdx.y * RowTile;

  float acc[RowTile][OutTile];
  float s[OutTile];

#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    s[j] = __half2float(scale[n0 + j]);
  }
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      acc[r][j] = 0.0f;
    }
  }

  if constexpr (Use4) {
    // 4-wide: 每次迭代处理 4 个 K 元素，减少 50% 循环次数
    const int K4 = K >> 2;
    for (int k4 = threadIdx.x; k4 < K4; k4 += Threads) {
      const int k = k4 << 2;
      float wv[OutTile][4];
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const int n = n0 + j;
        if (n < N) {
          const int8_t* wj = w_orig + static_cast<int64_t>(n) * K + k;
          wv[j][0] = (float)wj[0] * s[j];
          wv[j][1] = (float)wj[1] * s[j];
          wv[j][2] = (float)wj[2] * s[j];
          wv[j][3] = (float)wj[3] * s[j];
        } else {
          wv[j][0] = wv[j][1] = wv[j][2] = wv[j][3] = 0.0f;
        }
      }
#pragma unroll
      for (int r = 0; r < RowTile; ++r) {
        const int m = m0 + r;
        if (m < M) {
          const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + static_cast<int64_t>(m) * K + k));
          const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + static_cast<int64_t>(m) * K + k + 2));
#pragma unroll
          for (int j = 0; j < OutTile; ++j) {
            acc[r][j] = fmaf(x0.x, wv[j][0], acc[r][j]);
            acc[r][j] = fmaf(x0.y, wv[j][1], acc[r][j]);
            acc[r][j] = fmaf(x1.x, wv[j][2], acc[r][j]);
            acc[r][j] = fmaf(x1.y, wv[j][3], acc[r][j]);
          }
        }
      }
    }
    // K%4 尾部（最多 3 个元素）
    const int k_tail_start = K4 << 2;
    if (k_tail_start < K && threadIdx.x == 0) {
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const int n = n0 + j;
        if (n < N) {
          const int8_t* wj = w_orig + static_cast<int64_t>(n) * K + k_tail_start;
#pragma unroll
          for (int r = 0; r < RowTile; ++r) {
            const int m = m0 + r;
            if (m < M) {
              const dtype* xr = x + static_cast<int64_t>(m) * K + k_tail_start;
#pragma unroll
              for (int t = 0; t < 4 && k_tail_start + t < K; ++t) {
                acc[r][j] = fmaf(__half2float(xr[t]), (float)wj[t] * s[j], acc[r][j]);
              }
            }
          }
        }
      }
    }
  } else {
    // 2-wide: 每次迭代处理 2 个 K 元素
    const int K2 = K >> 1;
    for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
      const int k = k2 << 1;
      float wv[OutTile][2];
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const int n = n0 + j;
        if (n < N) {
          const int8_t* wj = w_orig + static_cast<int64_t>(n) * K + k;
          wv[j][0] = (float)wj[0] * s[j];
          wv[j][1] = (float)wj[1] * s[j];
        } else {
          wv[j][0] = 0.0f;
          wv[j][1] = 0.0f;
        }
      }
#pragma unroll
      for (int r = 0; r < RowTile; ++r) {
        const int m = m0 + r;
        if (m < M) {
          const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + static_cast<int64_t>(m) * K + k));
#pragma unroll
          for (int j = 0; j < OutTile; ++j) {
            acc[r][j] = fmaf(xv.x, wv[j][0], acc[r][j]);
            acc[r][j] = fmaf(xv.y, wv[j][1], acc[r][j]);
          }
        }
      }
    }
    // K 奇数尾部
    if ((K & 1) && threadIdx.x == 0) {
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const int n = n0 + j;
        if (n < N) {
          const float wv = (float)w_orig[static_cast<int64_t>(n) * K + K - 1] * s[j];
#pragma unroll
          for (int r = 0; r < RowTile; ++r) {
            const int m = m0 + r;
            if (m < M) {
              const float xv = __half2float(*reinterpret_cast<const __half*>(x + static_cast<int64_t>(m) * K + K - 1));
              acc[r][j] = fmaf(xv, wv, acc[r][j]);
            }
          }
        }
      }
    }
  }

  // warp reduce → shared memory → thread 0 写出
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

// ═══════════════════════════════════════════════════════════════════════
// Host wrappers
// ═══════════════════════════════════════════════════════════════════════

template <int Threads, int OutTile, bool Use4>
at::Tensor linear_int8_orig_row1_exact_f16_cuda_impl(at::Tensor x, at::Tensor w_orig, at::Tensor scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = w_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_int8_orig_row1_exact_f16 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_int8_orig_row1_exact_f16 requires N divisible by out_tile");
  TORCH_CHECK((k64 % (Use4 ? 4 : 2)) == 0, "linear_int8_orig_row1_exact_f16 unsupported K alignment");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 1, "linear_int8_orig_row1_exact_f16 requires one row");

  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

  linear_int8_orig_row1_exact_kernel<Threads, OutTile, Use4>
      <<<N / OutTile, Threads, 0, stream>>>(
          K, N,
          reinterpret_cast<const dtype*>(x.data_ptr<at::Half>()),
          w_orig.data_ptr<int8_t>(),
          reinterpret_cast<const dtype*>(scale.data_ptr<at::Half>()),
          reinterpret_cast<dtype*>(y.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

template <int Threads, int OutTile, bool Use4>
at::Tensor linear_int8_orig_row2_exact_f16_cuda_impl(at::Tensor x, at::Tensor w_orig, at::Tensor scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = w_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_int8_orig_row2_exact_f16 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_int8_orig_row2_exact_f16 requires N divisible by out_tile");
  TORCH_CHECK((k64 % (Use4 ? 4 : 2)) == 0, "linear_int8_orig_row2_exact_f16 unsupported K alignment");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 2, "linear_int8_orig_row2_exact_f16 requires two rows");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_int8_orig_row2_exact_kernel<Threads, OutTile, Use4>
      <<<N / OutTile, Threads, 0, stream>>>(
          K, N,
          reinterpret_cast<const dtype*>(x.data_ptr<at::Half>()),
          w_orig.data_ptr<int8_t>(),
          reinterpret_cast<const dtype*>(scale.data_ptr<at::Half>()),
          reinterpret_cast<dtype*>(y.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// 顶层 dispatch：参数选择与原版一致
at::Tensor linear_int8_orig_rows_exact_f16_cuda(
    at::Tensor x, at::Tensor w_orig, at::Tensor scale,
    int64_t threads, int64_t out_tile, bool use4) {
  const int64_t rows = x.numel() / x.size(-1);
  if (rows == 1) {
    if (!use4 && threads == 128 && out_tile == 2) return linear_int8_orig_row1_exact_f16_cuda_impl<128, 2, false>(x, w_orig, scale);
    if (use4 && threads == 128 && out_tile == 2) return linear_int8_orig_row1_exact_f16_cuda_impl<128, 2, true>(x, w_orig, scale);
  }
  if (rows == 2) {
    if (use4 && threads == 64 && out_tile == 2) return linear_int8_orig_row2_exact_f16_cuda_impl<64, 2, true>(x, w_orig, scale);
    if (use4 && threads == 256 && out_tile == 1) return linear_int8_orig_row2_exact_f16_cuda_impl<256, 1, true>(x, w_orig, scale);
    if (!use4 && threads == 128 && out_tile == 2) return linear_int8_orig_row2_exact_f16_cuda_impl<128, 2, false>(x, w_orig, scale);
  }
  TORCH_CHECK(false, "unsupported linear_int8_orig_rows_exact_f16 rows/threads/out_tile/use4");
}

// M>1 fallback wrapper
at::Tensor linear_int8_f16_cuda(
    at::Tensor x,
    at::Tensor w_int8,
    at::Tensor w_scale) {

  int64_t K = x.size(-1);
  int64_t N = w_int8.size(1);
  int64_t M = x.numel() / K;

  auto y = at::empty({M, N}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int Threads = 256;
  linear_int8_f16_kernel<Threads>
      <<<(int)M, Threads, 0, stream>>>(
          reinterpret_cast<const dtype*>(x.data_ptr<at::Half>()),
          w_int8.data_ptr<int8_t>(),
          reinterpret_cast<const dtype*>(w_scale.data_ptr<at::Half>()),
          reinterpret_cast<dtype*>(y.data_ptr<at::Half>()),
          (int)M, (int)K, (int)N);

  return y;
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel D: 批量 dequant int8[N,K] → fp16[N,K]（可选转置为 [K,N]）
// 用于 prefill (M>2) 场景，替代 Python 多步 dequant 链
//
// 输入：w_int8 [N,K] int8, scale [N] fp16
// 输出：w_fp16 [N,K] 或 [K,N] fp16
// transpose=false → 输出 [N,K]（orig 布局）
// transpose=true  → 输出 [K,N]（non-orig 布局，用于 linear/splitk）
// ═══════════════════════════════════════════════════════════════════════

template <int Threads, bool Transpose>
__global__ void dequant_int8_to_f16_kernel(
    const int8_t* __restrict__ w_int8,
    const dtype* __restrict__ scale,
    dtype* __restrict__ w_fp16,
    int N, int K) {

  // 每个 block 处理一行（N 维度的一行）
  int n = blockIdx.x;
  if (n >= N) return;

  float s = __half2float(scale[n]);
  const int8_t* wj = w_int8 + static_cast<int64_t>(n) * K;

  if constexpr (!Transpose) {
    // 非转置：输出 [N,K]，连续写入，可用 half2 向量化
    dtype* out = w_fp16 + static_cast<int64_t>(n) * K;
    // 一次处理 8 个元素（2 个 int4 读 = 16 个 int8，但用 8 更对齐 K%8==0 常见情况）
    int k = threadIdx.x * 8;
    for (; k + 7 < K; k += Threads * 8) {
      // 读 8 个 int8（2 次 int32 加载）
      const int32_t* w32 = reinterpret_cast<const int32_t*>(wj + k);
      int32_t w0 = w32[0];  // 4 个 int8
      int32_t w1 = w32[1];  // 4 个 int8
      // 转换为 float 再乘 scale
      float f0 = static_cast<float>(static_cast<int8_t>(w0 & 0xFF)) * s;
      float f1 = static_cast<float>(static_cast<int8_t>((w0 >> 8) & 0xFF)) * s;
      float f2 = static_cast<float>(static_cast<int8_t>((w0 >> 16) & 0xFF)) * s;
      float f3 = static_cast<float>(static_cast<int8_t>((w0 >> 24) & 0xFF)) * s;
      float f4 = static_cast<float>(static_cast<int8_t>(w1 & 0xFF)) * s;
      float f5 = static_cast<float>(static_cast<int8_t>((w1 >> 8) & 0xFF)) * s;
      float f6 = static_cast<float>(static_cast<int8_t>((w1 >> 16) & 0xFF)) * s;
      float f7 = static_cast<float>(static_cast<int8_t>((w1 >> 24) & 0xFF)) * s;
      // 用 __floats2half2_rn 批量写 2 个 fp16
      *reinterpret_cast<__half2*>(out + k)     = __floats2half2_rn(f0, f1);
      *reinterpret_cast<__half2*>(out + k + 2) = __floats2half2_rn(f2, f3);
      *reinterpret_cast<__half2*>(out + k + 4) = __floats2half2_rn(f4, f5);
      *reinterpret_cast<__half2*>(out + k + 6) = __floats2half2_rn(f6, f7);
    }
    // 尾部
    for (; k < K; k += Threads) {
      out[k] = __float2half_rn(static_cast<float>(wj[k]) * s);
    }
  } else {
    // 转置：输出 [K,N]，非连续写入，无法向量化
    for (int k = threadIdx.x; k < K; k += Threads) {
      float val = static_cast<float>(wj[k]) * s;
      w_fp16[static_cast<int64_t>(k) * N + n] = __float2half_rn(val);
    }
  }
}

// Host wrapper
at::Tensor dequant_int8_to_f16_cuda(
    at::Tensor w_int8, at::Tensor scale, bool transpose) {

  int64_t N = w_int8.size(0);
  int64_t K = w_int8.size(1);

  std::vector<int64_t> out_sizes;
  if (transpose) {
    out_sizes = {K, N};  // [K, N]
  } else {
    out_sizes = {N, K};  // [N, K]
  }
  auto w_fp16 = at::empty(out_sizes, scale.options());
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int Threads = 256;
  if (transpose) {
    dequant_int8_to_f16_kernel<Threads, true>
        <<<(int)N, Threads, 0, stream>>>(
            w_int8.data_ptr<int8_t>(),
            reinterpret_cast<const dtype*>(scale.data_ptr<at::Half>()),
            reinterpret_cast<dtype*>(w_fp16.data_ptr<at::Half>()),
            (int)N, (int)K);
  } else {
    dequant_int8_to_f16_kernel<Threads, false>
        <<<(int)N, Threads, 0, stream>>>(
            w_int8.data_ptr<int8_t>(),
            reinterpret_cast<const dtype*>(scale.data_ptr<at::Half>()),
            reinterpret_cast<dtype*>(w_fp16.data_ptr<at::Half>()),
            (int)N, (int)K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return w_fp16;
}

// ═══════════════════════════════════════════════════════════════════════
// linear_int8_orig_rows_f16 — M≥3 GEMM with int8 weights
//   x      [M,K]    fp16
//   w_orig [N,K]    int8  (orig 布局，不转置)
//   scale  [N]      fp16
//   row_tile, out_tile — 和原版 linear_orig_rows_f16 相同的 tiling 参数
// ═══════════════════════════════════════════════════════════════════════

template <int Threads, int RowTile, int OutTile, bool Use4>
at::Tensor linear_int8_orig_rows_f16_cuda_impl(at::Tensor x, at::Tensor w_orig, at::Tensor scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = w_orig.size(0);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX && m64 <= INT_MAX, "linear_int8_orig_rows_f16 M/K/N too large");
  const int M = static_cast<int>(m64);
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);

  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid((N + OutTile - 1) / OutTile, (M + RowTile - 1) / RowTile, 1);
  linear_int8_orig_rows_kernel<Threads, RowTile, OutTile, Use4>
      <<<grid, Threads, 0, stream>>>(
          M, K, N,
          reinterpret_cast<const dtype*>(x.data_ptr<at::Half>()),
          w_orig.data_ptr<int8_t>(),
          reinterpret_cast<const dtype*>(scale.data_ptr<at::Half>()),
          reinterpret_cast<dtype*>(y.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_int8_orig_rows_f16_cuda(
    at::Tensor x, at::Tensor w_orig, at::Tensor scale,
    int64_t row_tile, int64_t out_tile) {
  // K%4==0 时用 4-wide K 循环（减少 50% 循环次数），否则用 2-wide
  const bool use4 = (x.size(-1) % 4) == 0;
  if (use4) {
    if (row_tile == 4 && out_tile == 4) return linear_int8_orig_rows_f16_cuda_impl<128, 4, 4, true>(x, w_orig, scale);
    if (row_tile == 4 && out_tile == 2) return linear_int8_orig_rows_f16_cuda_impl<128, 4, 2, true>(x, w_orig, scale);
    if (row_tile == 8 && out_tile == 4) return linear_int8_orig_rows_f16_cuda_impl<128, 8, 4, true>(x, w_orig, scale);
    if (row_tile == 8 && out_tile == 2) return linear_int8_orig_rows_f16_cuda_impl<128, 8, 2, true>(x, w_orig, scale);
    if (row_tile == 16 && out_tile == 2) return linear_int8_orig_rows_f16_cuda_impl<128, 16, 2, true>(x, w_orig, scale);
    if (row_tile == 16 && out_tile == 4) return linear_int8_orig_rows_f16_cuda_impl<128, 16, 4, true>(x, w_orig, scale);
  } else {
    if (row_tile == 4 && out_tile == 4) return linear_int8_orig_rows_f16_cuda_impl<128, 4, 4, false>(x, w_orig, scale);
    if (row_tile == 4 && out_tile == 2) return linear_int8_orig_rows_f16_cuda_impl<128, 4, 2, false>(x, w_orig, scale);
    if (row_tile == 8 && out_tile == 4) return linear_int8_orig_rows_f16_cuda_impl<128, 8, 4, false>(x, w_orig, scale);
    if (row_tile == 8 && out_tile == 2) return linear_int8_orig_rows_f16_cuda_impl<128, 8, 2, false>(x, w_orig, scale);
    if (row_tile == 16 && out_tile == 2) return linear_int8_orig_rows_f16_cuda_impl<128, 16, 2, false>(x, w_orig, scale);
    if (row_tile == 16 && out_tile == 4) return linear_int8_orig_rows_f16_cuda_impl<128, 16, 4, false>(x, w_orig, scale);
  }
  TORCH_CHECK(false, "unsupported linear_int8_orig_rows_f16 row_tile/out_tile: ", row_tile, "/", out_tile);
}
