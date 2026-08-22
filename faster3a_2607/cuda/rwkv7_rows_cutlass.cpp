#include <climits>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

torch::Tensor rows_cutlass_orig_cuda(
    torch::Tensor x, torch::Tensor weight_orig, int64_t config);
torch::Tensor rows_cutlass_runtime_cuda(
    torch::Tensor x, torch::Tensor weight, int64_t config);

namespace {

void check_half_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be fp16");
}

void check_common(const torch::Tensor& x, const torch::Tensor& weight) {
  check_half_cuda_contiguous(x, "x");
  check_half_cuda_contiguous(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have shape [..., K]");
  TORCH_CHECK(weight.dim() == 2, "weight must be two-dimensional");
  TORCH_CHECK(x.size(-1) > 0, "x K dimension must be positive");
  const int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(rows > 0 && rows <= INT_MAX, "flattened rows exceed CUTLASS int range");
  TORCH_CHECK(x.get_device() == weight.get_device(), "x and weight must share a CUDA device");
  // CUTLASS Gemm::Arguments uses int dimensions; reject before narrowing.
  TORCH_CHECK(x.size(-1) <= INT_MAX, "x K dimension exceeds CUTLASS int range");
  TORCH_CHECK(
      weight.size(0) <= INT_MAX && weight.size(1) <= INT_MAX,
      "weight dimension exceeds CUTLASS int range");
}

torch::Tensor rows_cutlass_orig(
    torch::Tensor x, torch::Tensor weight_orig, int64_t config) {
  check_common(x, weight_orig);
  TORCH_CHECK(weight_orig.size(1) == x.size(-1), "original-layout weight must be [N, K]");
  TORCH_CHECK(config == 12, "production original-layout config must be 12");
  c10::cuda::CUDAGuard device_guard(x.device());
  return rows_cutlass_orig_cuda(x, weight_orig, config);
}

torch::Tensor rows_cutlass_runtime(
    torch::Tensor x, torch::Tensor weight, int64_t config) {
  check_common(x, weight);
  TORCH_CHECK(weight.size(0) == x.size(-1), "runtime-layout weight must be [K, N]");
  TORCH_CHECK(config == 15, "production runtime-layout config must be 15");
  c10::cuda::CUDAGuard device_guard(x.device());
  return rows_cutlass_runtime_cuda(x, weight, config);
}

}  // namespace

TORCH_LIBRARY(rwkv7_rows_cutlass, m) {
  m.def("linear_orig(Tensor x, Tensor weight_orig, int config) -> Tensor");
  m.def("linear_runtime(Tensor x, Tensor weight, int config) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_rows_cutlass, CUDA, m) {
  m.impl("linear_orig", &rows_cutlass_orig);
  m.impl("linear_runtime", &rows_cutlass_runtime);
}
