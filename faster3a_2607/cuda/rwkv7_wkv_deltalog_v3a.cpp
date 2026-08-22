#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

void deltalog_step_v3a_cuda(
    int B,
    int C,
    int H,
    int merge_interval,
    int phase,
    torch::Tensor base_state,
    torch::Tensor log_delta,
    torch::Tensor log_u,
    torch::Tensor log_b,
    torch::Tensor log_k,
    torch::Tensor log_v,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t);

void deltalog_step_slot_packed_v3a_cuda(
    int B,
    int C,
    int H,
    int L,
    int layer,
    int merge_interval,
    int phase,
    torch::Tensor base_state,
    torch::Tensor packed_log,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t);

void deltalog_step_slot_layer_packed_v3a_cuda(
    int B,
    int C,
    int H,
    int L,
    int layer,
    int merge_interval,
    int phase,
    torch::Tensor base_state,
    torch::Tensor packed_log,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t);

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(
      status == cudaSuccess,
      operation,
      " failed: ",
      cudaGetErrorString(status));
}

void check_half_tensor(
    const torch::Tensor& tensor,
    const char* name,
    int64_t numel,
    const c10::Device& device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.numel() == numel, name, " has wrong numel");
}

void step(
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t merge_interval,
    int64_t phase,
    torch::Tensor base_state,
    torch::Tensor log_delta,
    torch::Tensor log_u,
    torch::Tensor log_b,
    torch::Tensor log_k,
    torch::Tensor log_v,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t) {
  TORCH_CHECK(B > 0 && B <= 65535, "B must fit the 2D CUDA grid");
  TORCH_CHECK(C > 0 && C == H * 64, "DeltaLog v3a requires C == H * 64");
  TORCH_CHECK(
      merge_interval == 2 || merge_interval == 3 || merge_interval == 4 ||
          merge_interval == 6 || merge_interval == 8,
      "merge_interval must be one of 2,3,4,6,8");
  TORCH_CHECK(phase >= 0 && phase < merge_interval, "phase out of range");

  const auto device = base_state.device();
  const int64_t factor_numel = B * C;
  const int64_t state_numel = factor_numel * 64;
  const int64_t log_numel = (merge_interval - 1) * factor_numel;
  check_half_tensor(base_state, "base_state", state_numel, device);
  check_half_tensor(log_delta, "log_delta", log_numel, device);
  check_half_tensor(log_u, "log_u", log_numel, device);
  check_half_tensor(log_b, "log_b", log_numel, device);
  check_half_tensor(log_k, "log_k", log_numel, device);
  check_half_tensor(log_v, "log_v", log_numel, device);
  check_half_tensor(r, "r", factor_numel, device);
  check_half_tensor(w, "w", factor_numel, device);
  check_half_tensor(w0, "w0", C, device);
  check_half_tensor(k, "k", factor_numel, device);
  check_half_tensor(v, "v", factor_numel, device);
  check_half_tensor(a, "a", factor_numel, device);
  check_half_tensor(b, "b", factor_numel, device);
  check_half_tensor(y, "y", factor_numel, device);
  TORCH_CHECK(elapsed_t.is_cuda() && elapsed_t.device() == device, "elapsed_t device mismatch");
  TORCH_CHECK(elapsed_t.scalar_type() == torch::kInt32, "elapsed_t must be int32");
  TORCH_CHECK(elapsed_t.is_contiguous() && elapsed_t.numel() == B, "elapsed_t shape mismatch");

  deltalog_step_v3a_cuda(
      static_cast<int>(B), static_cast<int>(C), static_cast<int>(H),
      static_cast<int>(merge_interval), static_cast<int>(phase),
      base_state, log_delta, log_u, log_b, log_k, log_v,
      r, w, w0, k, v, a, b, y, elapsed_t);
}

void packed_step_impl(
    bool layer_packed,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t L,
    int64_t layer,
    int64_t merge_interval,
    int64_t phase,
    torch::Tensor base_state,
    torch::Tensor packed_log,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t) {
  TORCH_CHECK(B > 0 && B <= 65535, "B must fit the 2D CUDA grid");
  TORCH_CHECK(C > 0 && C == H * 64, "DeltaLog v3a requires C == H * 64");
  TORCH_CHECK(L > 0 && layer >= 0 && layer < L, "invalid layer/L");
  TORCH_CHECK(
      merge_interval == 2 || merge_interval == 3 || merge_interval == 4 ||
          merge_interval == 6 || merge_interval == 8,
      "merge_interval must be one of 2,3,4,6,8");
  TORCH_CHECK(phase >= 0 && phase < merge_interval, "phase out of range");

  const auto device = base_state.device();
  const int64_t factor_numel = B * C;
  check_half_tensor(base_state, "base_state", factor_numel * 64, device);
  check_half_tensor(
      packed_log, "packed_log", (merge_interval - 1) * 5 * L * factor_numel,
      device);
  check_half_tensor(r, "r", factor_numel, device);
  check_half_tensor(w, "w", factor_numel, device);
  check_half_tensor(w0, "w0", C, device);
  check_half_tensor(k, "k", factor_numel, device);
  check_half_tensor(v, "v", factor_numel, device);
  check_half_tensor(a, "a", factor_numel, device);
  check_half_tensor(b, "b", factor_numel, device);
  check_half_tensor(y, "y", factor_numel, device);
  TORCH_CHECK(
      elapsed_t.is_cuda() && elapsed_t.device() == device,
      "elapsed_t device mismatch");
  TORCH_CHECK(elapsed_t.scalar_type() == torch::kInt32, "elapsed_t must be int32");
  TORCH_CHECK(
      elapsed_t.is_contiguous() && elapsed_t.numel() == B,
      "elapsed_t shape mismatch");

  if (layer_packed) {
    deltalog_step_slot_layer_packed_v3a_cuda(
        static_cast<int>(B), static_cast<int>(C), static_cast<int>(H),
        static_cast<int>(L), static_cast<int>(layer),
        static_cast<int>(merge_interval), static_cast<int>(phase),
        base_state, packed_log, r, w, w0, k, v, a, b, y, elapsed_t);
  } else {
    deltalog_step_slot_packed_v3a_cuda(
        static_cast<int>(B), static_cast<int>(C), static_cast<int>(H),
        static_cast<int>(L), static_cast<int>(layer),
        static_cast<int>(merge_interval), static_cast<int>(phase),
        base_state, packed_log, r, w, w0, k, v, a, b, y, elapsed_t);
  }
}

#define PACKED_STEP_ARGS \
    int64_t B, int64_t C, int64_t H, int64_t L, int64_t layer, \
    int64_t merge_interval, int64_t phase, torch::Tensor base_state, \
    torch::Tensor packed_log, torch::Tensor r, torch::Tensor w, \
    torch::Tensor w0, torch::Tensor k, torch::Tensor v, torch::Tensor a, \
    torch::Tensor b, torch::Tensor y, torch::Tensor elapsed_t

void step_slot_packed(PACKED_STEP_ARGS) {
  packed_step_impl(
      false, B, C, H, L, layer, merge_interval, phase, base_state, packed_log,
      r, w, w0, k, v, a, b, y, elapsed_t);
}

void step_slot_layer_packed(PACKED_STEP_ARGS) {
  packed_step_impl(
      true, B, C, H, L, layer, merge_interval, phase, base_state, packed_log,
      r, w, w0, k, v, a, b, y, elapsed_t);
}

#undef PACKED_STEP_ARGS

std::vector<int64_t> apw_device_info(const torch::Tensor& reference) {
  TORCH_CHECK(reference.is_cuda(), "reference must be CUDA");
  const int device = reference.get_device();
  c10::cuda::CUDAGuard guard(device);
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  size_t current_limit = 0;
  check_cuda(
      cudaDeviceGetLimit(&current_limit, cudaLimitPersistingL2CacheSize),
      "cudaDeviceGetLimit(persisting L2)");
  return {
      properties.l2CacheSize,
      properties.persistingL2CacheMaxSize,
      properties.accessPolicyMaxWindowSize,
      static_cast<int64_t>(current_limit),
  };
}

void set_persisting_l2_limit(const torch::Tensor& reference, int64_t bytes) {
  TORCH_CHECK(reference.is_cuda(), "reference must be CUDA");
  TORCH_CHECK(bytes >= 0, "persisting L2 limit must be non-negative");
  c10::cuda::CUDAGuard guard(reference.get_device());
  check_cuda(
      cudaDeviceSetLimit(
          cudaLimitPersistingL2CacheSize, static_cast<size_t>(bytes)),
      "cudaDeviceSetLimit(persisting L2)");
}

void reset_persisting_l2_cache(const torch::Tensor& reference) {
  TORCH_CHECK(reference.is_cuda(), "reference must be CUDA");
  c10::cuda::CUDAGuard guard(reference.get_device());
  check_cuda(cudaCtxResetPersistingL2Cache(), "cudaCtxResetPersistingL2Cache");
}

int64_t set_graph_persisting_window(
    int64_t graph_handle,
    const torch::Tensor& packed_log,
    int64_t bytes,
    double hit_ratio) {
  TORCH_CHECK(graph_handle != 0, "CUDA graph handle must be non-zero");
  TORCH_CHECK(packed_log.is_cuda(), "packed_log must be CUDA");
  TORCH_CHECK(packed_log.is_contiguous(), "packed_log must be contiguous");
  TORCH_CHECK(bytes > 0 && bytes <= packed_log.nbytes(), "invalid APW size");
  TORCH_CHECK(hit_ratio > 0.0 && hit_ratio <= 1.0, "invalid APW hit ratio");
  c10::cuda::CUDAGuard guard(packed_log.get_device());

  auto graph = reinterpret_cast<cudaGraph_t>(graph_handle);
  size_t node_count = 0;
  check_cuda(cudaGraphGetNodes(graph, nullptr, &node_count), "cudaGraphGetNodes(count)");
  std::vector<cudaGraphNode_t> nodes(node_count);
  check_cuda(
      cudaGraphGetNodes(graph, nodes.data(), &node_count),
      "cudaGraphGetNodes(nodes)");

  int64_t kernel_count = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType node_type{};
    check_cuda(cudaGraphNodeGetType(node, &node_type), "cudaGraphNodeGetType");
    if (node_type != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeAttrValue attribute{};
    attribute.accessPolicyWindow.base_ptr = packed_log.data_ptr();
    attribute.accessPolicyWindow.num_bytes = static_cast<size_t>(bytes);
    attribute.accessPolicyWindow.hitRatio = static_cast<float>(hit_ratio);
    attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    check_cuda(
        cudaGraphKernelNodeSetAttribute(
            node, cudaKernelNodeAttributeAccessPolicyWindow, &attribute),
        "cudaGraphKernelNodeSetAttribute(persisting window)");
    ++kernel_count;
  }
  TORCH_CHECK(kernel_count > 0, "CUDA graph has no kernel nodes");
  return kernel_count;
}

}  // namespace

TORCH_LIBRARY(rwkv7_wkv_deltalog_v3a, m) {
  m.def("step", step);
  m.def("step_slot_packed", step_slot_packed);
  m.def("step_slot_layer_packed", step_slot_layer_packed);
  m.def("apw_device_info", apw_device_info);
  m.def("set_persisting_l2_limit", set_persisting_l2_limit);
  m.def("reset_persisting_l2_cache", reset_persisting_l2_cache);
  m.def("set_graph_persisting_window", set_graph_persisting_window);
}
