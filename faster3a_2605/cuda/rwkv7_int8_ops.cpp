#include <torch/extension.h>

// Forward declarations (from .cu)
at::Tensor linear_int8_orig_rows_exact_f16_cuda(
    at::Tensor x, at::Tensor w_orig, at::Tensor scale,
    int64_t threads, int64_t out_tile, bool use4);
at::Tensor linear_int8_f16_cuda(
    at::Tensor x, at::Tensor w_int8, at::Tensor w_scale);
at::Tensor dequant_int8_to_f16_cuda(
    at::Tensor w_int8, at::Tensor scale, bool transpose);
at::Tensor linear_int8_orig_rows_f16_cuda(
    at::Tensor x, at::Tensor w_orig, at::Tensor scale,
    int64_t row_tile, int64_t out_tile);

void check_half_cuda_contig(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().type() == torch::kCUDA, name, " must be CUDA");
  TORCH_CHECK(t.dtype() == torch::kHalf, name, " must be half");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_int8_cuda_contig(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().type() == torch::kCUDA, name, " must be CUDA");
  TORCH_CHECK(t.dtype() == torch::kInt8, name, " must be int8");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// ═══════════════════════════════════════════════════════════════
// linear_int8_orig_rows_exact_f16
//   x      [M,K]    fp16  (M=1 或 M=2)
//   w_orig [N,K]    int8  (orig 布局，不转置)
//   scale  [N]      fp16
//   threads, out_tile, use4 — 和原版 linear_orig_rows_exact_f16 相同
// ═══════════════════════════════════════════════════════════════

torch::Tensor linear_int8_orig_rows_exact_f16(
    torch::Tensor x,
    torch::Tensor w_orig,
    torch::Tensor scale,
    int64_t threads,
    int64_t out_tile,
    bool use4) {
  check_half_cuda_contig(x, "x");
  check_int8_cuda_contig(w_orig, "w_orig");
  check_half_cuda_contig(scale, "scale");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(w_orig.dim() == 2, "w_orig must be 2D [N, K]");
  TORCH_CHECK(scale.dim() == 1, "scale must be 1D [N]");
  int64_t K = x.size(-1);
  int64_t N = w_orig.size(0);
  TORCH_CHECK(w_orig.size(1) == K, "w_orig K mismatch: ", w_orig.size(1), " vs ", K);
  TORCH_CHECK(scale.size(0) == N, "scale N mismatch: ", scale.size(0), " vs ", N);
  return linear_int8_orig_rows_exact_f16_cuda(x, w_orig, scale, threads, out_tile, use4);
}

// ═══════════════════════════════════════════════════════════════
// linear_int8_f16 — M>1 fallback（per-token 量化 + int8×int8）
//   x      [M,K]    fp16
//   w_int8 [K,N]    int8  (non-orig 布局，已转置)
//   w_scale [N]     fp16
// ═══════════════════════════════════════════════════════════════

torch::Tensor linear_int8_f16(
    torch::Tensor x,
    torch::Tensor w_int8,
    torch::Tensor w_scale) {
  check_half_cuda_contig(x, "x");
  check_int8_cuda_contig(w_int8, "w_int8");
  check_half_cuda_contig(w_scale, "w_scale");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims, got ", x.dim());
  TORCH_CHECK(w_int8.dim() == 2, "w_int8 must be 2D [K, N]");
  TORCH_CHECK(w_scale.dim() == 1, "w_scale must be 1D [N]");
  int64_t K = x.size(-1);
  int64_t N = w_int8.size(1);
  TORCH_CHECK(w_int8.size(0) == K, "w_int8 K mismatch: ", w_int8.size(0), " vs ", K);
  TORCH_CHECK(w_scale.size(0) == N, "w_scale N mismatch");
  return linear_int8_f16_cuda(x, w_int8, w_scale);
}

// ═══════════════════════════════════════════════════════════════
// dequant_int8_to_f16 — 批量反量化 int8→fp16（可选转置）
//   w_int8  [N,K]   int8
//   scale   [N]     fp16
//   transpose=false → 输出 [N,K] fp16（orig 布局）
//   transpose=true  → 输出 [K,N] fp16（non-orig 布局）
// ═══════════════════════════════════════════════════════════════

torch::Tensor dequant_int8_to_f16(
    torch::Tensor w_int8,
    torch::Tensor scale,
    bool transpose) {
  check_int8_cuda_contig(w_int8, "w_int8");
  check_half_cuda_contig(scale, "scale");
  TORCH_CHECK(w_int8.dim() == 2, "w_int8 must be 2D [N, K]");
  TORCH_CHECK(scale.dim() == 1, "scale must be 1D [N]");
  TORCH_CHECK(scale.size(0) == w_int8.size(0), "scale N mismatch");
  return dequant_int8_to_f16_cuda(w_int8, scale, transpose);
}

// ═══════════════════════════════════════════════════════════════
// linear_int8_orig_rows_f16 — M≥3 GEMM with int8 weights
//   x      [M,K]    fp16
//   w_orig [N,K]    int8  (orig 布局，不转置)
//   scale  [N]      fp16
//   row_tile, out_tile — tiling 参数，和原版 linear_orig_rows_f16 一致
// ═══════════════════════════════════════════════════════════════

torch::Tensor linear_int8_orig_rows_f16(
    torch::Tensor x,
    torch::Tensor w_orig,
    torch::Tensor scale,
    int64_t row_tile,
    int64_t out_tile) {
  check_half_cuda_contig(x, "x");
  check_int8_cuda_contig(w_orig, "w_orig");
  check_half_cuda_contig(scale, "scale");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(w_orig.dim() == 2, "w_orig must be 2D [N, K]");
  TORCH_CHECK(scale.dim() == 1, "scale must be 1D [N]");
  int64_t K = x.size(-1);
  int64_t N = w_orig.size(0);
  int64_t M = x.numel() / K;
  TORCH_CHECK(w_orig.size(1) == K, "w_orig K mismatch: ", w_orig.size(1), " vs ", K);
  TORCH_CHECK(scale.size(0) == N, "scale N mismatch: ", scale.size(0), " vs ", N);
  TORCH_CHECK(M >= 3, "linear_int8_orig_rows_f16 requires M>=3, got ", M);
  return linear_int8_orig_rows_f16_cuda(x, w_orig, scale, row_tile, out_tile);
}

// ═══════════════════════════════════════════════════════════════
// TORCH_LIBRARY 注册
// ═══════════════════════════════════════════════════════════════

TORCH_LIBRARY(rwkv7_int8_ops, m) {
  m.def("linear_int8_orig_rows_exact_f16(Tensor x, Tensor w_orig, Tensor scale, int threads, int out_tile, bool use4) -> Tensor");
  m.def("linear_int8_f16(Tensor x, Tensor w_int8, Tensor w_scale) -> Tensor");
  m.def("dequant_int8_to_f16(Tensor w_int8, Tensor scale, bool transpose) -> Tensor");
  m.def("linear_int8_orig_rows_f16(Tensor x, Tensor w_orig, Tensor scale, int row_tile, int out_tile) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_int8_ops, CUDA, m) {
  m.impl("linear_int8_orig_rows_exact_f16", &linear_int8_orig_rows_exact_f16);
  m.impl("linear_int8_f16", &linear_int8_f16);
  m.impl("dequant_int8_to_f16", &dequant_int8_to_f16);
  m.impl("linear_int8_orig_rows_f16", &linear_int8_orig_rows_f16);
}
