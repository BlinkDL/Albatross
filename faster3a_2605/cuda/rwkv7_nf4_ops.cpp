#include <torch/extension.h>
#include <vector>

// CUDA kernel declarations (defined in .cu)
at::Tensor linear_nf4_orig_rows_exact_f16_cuda(
    at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale, int64_t threads, int64_t out_tile, bool use4);
at::Tensor linear_nvfp4_orig_row1_blk16_f16_cuda(
    at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale, int64_t out_tile);
at::Tensor linear_nvfp4_rkv_orig_row1_blk16_f16_cuda(
    at::Tensor x_r, at::Tensor x_k, at::Tensor x_v,
    at::Tensor w_r, at::Tensor w_k, at::Tensor w_v,
    at::Tensor bs_r, at::Tensor bs_k, at::Tensor bs_v,
    double ts_r, double ts_k, double ts_v, int64_t out_tile);
at::Tensor linear_nvfp4_orig_row2_blk16_f16_cuda(
    at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale, int64_t out_tile);
at::Tensor linear_nf4_orig_rows_f16_cuda(
    at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, double t_scale, int64_t row_tile, int64_t out_tile);
at::Tensor dequant_nf4_to_f16_cuda(at::Tensor w_nf4, at::Tensor b_scale, double t_scale, bool transpose);
at::Tensor cmix_sparse_down_relu_one_nf4_cuda(
    at::Tensor preact, at::Tensor w_nf4, at::Tensor b_scale, double t_scale, int64_t C, int64_t F);
at::Tensor cmix_sparse_down_relu_rows_nf4_cuda(
    at::Tensor preact, at::Tensor w_nf4, at::Tensor b_scale, double t_scale,
    int64_t B, int64_t T, int64_t C, int64_t F);
at::Tensor cmix_sparse_down_relu_rows_t512_nf4_cuda(
    at::Tensor preact, at::Tensor w_nf4, at::Tensor b_scale, double t_scale,
    int64_t B, int64_t T, int64_t C, int64_t F);

namespace {

void check_nf4_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kUInt8, name, " must be uint8");
}

void check_half_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, name, " must be fp16");
}

void check_fp8_e4m3_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat8_e4m3fn, name, " must be float8_e4m3fn");
}

torch::Tensor linear_nf4_orig_rows_exact_f16(
    torch::Tensor x, torch::Tensor w_nf4, torch::Tensor b_scale, double t_scale,
    int64_t threads, int64_t out_tile, bool use4) {
  check_half_cuda_contig(x, "x");
  check_nf4_cuda_contig(w_nf4, "w_nf4");
  check_fp8_e4m3_cuda_contig(b_scale, "b_scale");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(w_nf4.dim() == 2, "w_nf4 must have shape [N, K/2]");
  TORCH_CHECK(b_scale.dim() == 2, "b_scale must have shape [N, K/16]");
  const int64_t K = x.size(-1);
  TORCH_CHECK(w_nf4.size(1) == K / 2, "w_nf4 K/2 mismatch");
  TORCH_CHECK(b_scale.size(0) == w_nf4.size(0), "b_scale N mismatch");
  TORCH_CHECK(b_scale.size(1) == K / 16, "b_scale K/16 mismatch");
  TORCH_CHECK((K % 16) == 0, "K must be divisible by 16");
  return linear_nf4_orig_rows_exact_f16_cuda(x, w_nf4, b_scale, t_scale, threads, out_tile, use4);
}

torch::Tensor linear_nvfp4_orig_row1_blk16_f16(
    torch::Tensor x, torch::Tensor w_nf4, torch::Tensor b_scale, double t_scale,
    int64_t out_tile) {
  check_half_cuda_contig(x, "x");
  check_nf4_cuda_contig(w_nf4, "w_nf4");
  check_fp8_e4m3_cuda_contig(b_scale, "b_scale");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(w_nf4.dim() == 2, "w_nf4 must have shape [N, K/2]");
  TORCH_CHECK(b_scale.dim() == 2, "b_scale must have shape [N, K/16]");
  const int64_t K = x.size(-1);
  TORCH_CHECK(w_nf4.size(1) == K / 2, "w_nf4 K/2 mismatch");
  TORCH_CHECK(b_scale.size(0) == w_nf4.size(0), "b_scale N mismatch");
  TORCH_CHECK(b_scale.size(1) == K / 16, "b_scale K/16 mismatch");
  TORCH_CHECK((K % 16) == 0, "K must be divisible by 16");
  return linear_nvfp4_orig_row1_blk16_f16_cuda(x, w_nf4, b_scale, t_scale, out_tile);
}

torch::Tensor linear_nvfp4_rkv_orig_row1_blk16_f16(
    torch::Tensor x_r, torch::Tensor x_k, torch::Tensor x_v,
    torch::Tensor w_r, torch::Tensor w_k, torch::Tensor w_v,
    torch::Tensor bs_r, torch::Tensor bs_k, torch::Tensor bs_v,
    double ts_r, double ts_k, double ts_v, int64_t out_tile) {
  check_half_cuda_contig(x_r, "x_r");
  check_half_cuda_contig(x_k, "x_k");
  check_half_cuda_contig(x_v, "x_v");
  check_nf4_cuda_contig(w_r, "w_r");
  check_nf4_cuda_contig(w_k, "w_k");
  check_nf4_cuda_contig(w_v, "w_v");
  check_fp8_e4m3_cuda_contig(bs_r, "bs_r");
  check_fp8_e4m3_cuda_contig(bs_k, "bs_k");
  check_fp8_e4m3_cuda_contig(bs_v, "bs_v");
  return linear_nvfp4_rkv_orig_row1_blk16_f16_cuda(x_r, x_k, x_v, w_r, w_k, w_v, bs_r, bs_k, bs_v, ts_r, ts_k, ts_v, out_tile);
}

torch::Tensor linear_nvfp4_orig_row2_blk16_f16(
    torch::Tensor x, torch::Tensor w_nf4, torch::Tensor b_scale, double t_scale,
    int64_t out_tile) {
  check_half_cuda_contig(x, "x");
  check_nf4_cuda_contig(w_nf4, "w_nf4");
  check_fp8_e4m3_cuda_contig(b_scale, "b_scale");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(w_nf4.dim() == 2, "w_nf4 must have shape [N, K/2]");
  TORCH_CHECK(b_scale.dim() == 2, "b_scale must have shape [N, K/16]");
  const int64_t K = x.size(-1);
  TORCH_CHECK(w_nf4.size(1) == K / 2, "w_nf4 K/2 mismatch");
  TORCH_CHECK(b_scale.size(0) == w_nf4.size(0), "b_scale N mismatch");
  TORCH_CHECK(b_scale.size(1) == K / 16, "b_scale K/16 mismatch");
  TORCH_CHECK((K % 16) == 0, "K must be divisible by 16");
  return linear_nvfp4_orig_row2_blk16_f16_cuda(x, w_nf4, b_scale, t_scale, out_tile);
}

torch::Tensor linear_nf4_orig_rows_f16(
    torch::Tensor x, torch::Tensor w_nf4, torch::Tensor b_scale, double t_scale,
    int64_t row_tile, int64_t out_tile) {
  check_half_cuda_contig(x, "x");
  check_nf4_cuda_contig(w_nf4, "w_nf4");
  check_fp8_e4m3_cuda_contig(b_scale, "b_scale");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(w_nf4.dim() == 2, "w_nf4 must have shape [N, K/2]");
  TORCH_CHECK(b_scale.dim() == 2, "b_scale must have shape [N, K/16]");
  const int64_t K = x.size(-1);
  TORCH_CHECK(w_nf4.size(1) == K / 2, "w_nf4 K/2 mismatch");
  TORCH_CHECK(b_scale.size(0) == w_nf4.size(0), "b_scale N mismatch");
  TORCH_CHECK(b_scale.size(1) == K / 16, "b_scale K/16 mismatch");
  TORCH_CHECK((K % 16) == 0, "K must be divisible by 16");
  return linear_nf4_orig_rows_f16_cuda(x, w_nf4, b_scale, t_scale, row_tile, out_tile);
}

torch::Tensor dequant_nf4_to_f16(torch::Tensor w_nf4, torch::Tensor b_scale, double t_scale, bool transpose) {
  check_nf4_cuda_contig(w_nf4, "w_nf4");
  check_fp8_e4m3_cuda_contig(b_scale, "b_scale");
  TORCH_CHECK(w_nf4.dim() == 2, "w_nf4 must be shape [N, K/2]");
  TORCH_CHECK(b_scale.dim() == 2, "b_scale must be shape [N, K/16]");
  const int64_t N = w_nf4.size(0);
  const int64_t K = w_nf4.size(1) * 2;
  TORCH_CHECK(b_scale.size(0) == N, "b_scale N mismatch");
  TORCH_CHECK(b_scale.size(1) == K / 16, "b_scale K/16 mismatch");
  TORCH_CHECK((K % 16) == 0, "K must be divisible by 16");
  return dequant_nf4_to_f16_cuda(w_nf4, b_scale, t_scale, transpose);
}

torch::Tensor cmix_sparse_down_relu_one_nf4(
    torch::Tensor preact, torch::Tensor w_nf4, torch::Tensor b_scale, double t_scale,
    int64_t C, int64_t F) {
  check_half_cuda_contig(preact, "preact");
  check_nf4_cuda_contig(w_nf4, "w_nf4");
  check_fp8_e4m3_cuda_contig(b_scale, "b_scale");
  return cmix_sparse_down_relu_one_nf4_cuda(preact, w_nf4, b_scale, t_scale, C, F);
}

torch::Tensor cmix_sparse_down_relu_rows_nf4(
    torch::Tensor preact, torch::Tensor w_nf4, torch::Tensor b_scale, double t_scale,
    int64_t B, int64_t T, int64_t C, int64_t F) {
  check_half_cuda_contig(preact, "preact");
  check_nf4_cuda_contig(w_nf4, "w_nf4");
  check_fp8_e4m3_cuda_contig(b_scale, "b_scale");
  return cmix_sparse_down_relu_rows_nf4_cuda(preact, w_nf4, b_scale, t_scale, B, T, C, F);
}

torch::Tensor cmix_sparse_down_relu_rows_t512_nf4(
    torch::Tensor preact, torch::Tensor w_nf4, torch::Tensor b_scale, double t_scale,
    int64_t B, int64_t T, int64_t C, int64_t F) {
  check_half_cuda_contig(preact, "preact");
  check_nf4_cuda_contig(w_nf4, "w_nf4");
  check_fp8_e4m3_cuda_contig(b_scale, "b_scale");
  return cmix_sparse_down_relu_rows_t512_nf4_cuda(preact, w_nf4, b_scale, t_scale, B, T, C, F);
}

} // namespace

TORCH_LIBRARY(rwkv7_nf4_ops, m) {
  m.def("linear_nf4_orig_rows_exact_f16(Tensor x, Tensor w_nf4, Tensor b_scale, float t_scale, int threads, int out_tile, bool use4) -> Tensor");
  m.def("linear_nvfp4_orig_row1_blk16_f16(Tensor x, Tensor w_nf4, Tensor b_scale, float t_scale, int out_tile) -> Tensor");
  m.def("linear_nvfp4_rkv_orig_row1_blk16_f16(Tensor x_r, Tensor x_k, Tensor x_v, Tensor w_r, Tensor w_k, Tensor w_v, Tensor bs_r, Tensor bs_k, Tensor bs_v, float ts_r, float ts_k, float ts_v, int out_tile) -> Tensor");
  m.def("linear_nvfp4_orig_row2_blk16_f16(Tensor x, Tensor w_nf4, Tensor b_scale, float t_scale, int out_tile) -> Tensor");
  m.def("linear_nf4_orig_rows_f16(Tensor x, Tensor w_nf4, Tensor b_scale, float t_scale, int row_tile, int out_tile) -> Tensor");
  m.def("dequant_nf4_to_f16(Tensor w_nf4, Tensor b_scale, float t_scale, bool transpose) -> Tensor");
  m.def("cmix_sparse_down_relu_one_nf4(Tensor preact, Tensor w_nf4, Tensor b_scale, float t_scale, int C, int F) -> Tensor");
  m.def("cmix_sparse_down_relu_rows_nf4(Tensor preact, Tensor w_nf4, Tensor b_scale, float t_scale, int B, int T, int C, int F) -> Tensor");
  m.def("cmix_sparse_down_relu_rows_t512_nf4(Tensor preact, Tensor w_nf4, Tensor b_scale, float t_scale, int B, int T, int C, int F) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_nf4_ops, CUDA, m) {
  m.impl("linear_nf4_orig_rows_exact_f16", &linear_nf4_orig_rows_exact_f16);
  m.impl("linear_nvfp4_orig_row1_blk16_f16", &linear_nvfp4_orig_row1_blk16_f16);
  m.impl("linear_nvfp4_rkv_orig_row1_blk16_f16", &linear_nvfp4_rkv_orig_row1_blk16_f16);
  m.impl("linear_nvfp4_orig_row2_blk16_f16", &linear_nvfp4_orig_row2_blk16_f16);
  m.impl("linear_nf4_orig_rows_f16", &linear_nf4_orig_rows_f16);
  m.impl("dequant_nf4_to_f16", &dequant_nf4_to_f16);
  m.impl("cmix_sparse_down_relu_one_nf4", &cmix_sparse_down_relu_one_nf4);
  m.impl("cmix_sparse_down_relu_rows_nf4", &cmix_sparse_down_relu_rows_nf4);
  m.impl("cmix_sparse_down_relu_rows_t512_nf4", &cmix_sparse_down_relu_rows_t512_nf4);
}
