#include <torch/extension.h>

#include <climits>
#include <cstdint>
#include <initializer_list>
#include <utility>

void wkv_seq_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t);

void wkv_seq_w0_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t);

void wkv_seq_grid2d_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t);

void wkv_seq_w0_grid2d_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t);

void wkv_seq_grid2d_forced_v2_cuda(
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
    torch::Tensor y,
    torch::Tensor elapsed_t);

void wkv_seq_w0_grid2d_forced_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    int mode,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t);

void wkv_seq_forced_v2_cuda(
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
    torch::Tensor y,
    torch::Tensor elapsed_t);

void wkv_seq_w0_forced_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    int mode,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t);

void wkv_one_v2_cuda(
    int B,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t);

void wkv_one_w0_v2_cuda(
    int B,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t);

void wkv_seq(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t) {
  wkv_seq_v2_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      state,
      r,
      w,
      k,
      v,
      a,
      b,
      y,
      elapsed_t);
}

void wkv_seq_w0(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t) {
  wkv_seq_w0_v2_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      state,
      r,
      w,
      w0,
      k,
      v,
      a,
      b,
      y,
      elapsed_t);
}

void check_grid2d_dims(int64_t B, int64_t T, int64_t C, int64_t H) {
  TORCH_CHECK(B > 0 && B <= 65535, "2D WKV grid requires 0 < B <= 65535");
  TORCH_CHECK(T > 0 && T <= INT_MAX, "T must be positive int32");
  TORCH_CHECK(H > 0 && H <= INT_MAX, "H must be positive int32");
  TORCH_CHECK(C > 0 && C <= INT_MAX && C == H * 64, "2D WKV requires C == H * 64");
  // The inherited kernels form token and state offsets in signed int. Keep
  // these bounds even though the 2D launch itself can represent a larger B.
  TORCH_CHECK(T <= INT_MAX / C && B <= INT_MAX / (T * C),
              "2D WKV token indexing exceeds signed int32");
  TORCH_CHECK(B <= INT_MAX / (C * 64),
              "2D WKV state indexing exceeds signed int32");
}

using NamedTensor = std::pair<const char*, const torch::Tensor*>;

void check_grid2d_tensor(
    const torch::Tensor& tensor,
    const char* name,
    at::ScalarType dtype,
    int64_t numel,
    const c10::Device& device,
    uintptr_t alignment) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.numel() == numel, name, " has wrong numel");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0,
              name, " has insufficient pointer alignment");
}

void check_grid2d_no_overlap(std::initializer_list<NamedTensor> tensors) {
  for (auto lhs = tensors.begin(); lhs != tensors.end(); ++lhs) {
    const uintptr_t lhs_begin = reinterpret_cast<uintptr_t>(lhs->second->data_ptr());
    const uintptr_t lhs_end = lhs_begin + lhs->second->nbytes();
    for (auto rhs = lhs + 1; rhs != tensors.end(); ++rhs) {
      const uintptr_t rhs_begin = reinterpret_cast<uintptr_t>(rhs->second->data_ptr());
      const uintptr_t rhs_end = rhs_begin + rhs->second->nbytes();
      TORCH_CHECK(lhs_end <= rhs_begin || rhs_end <= lhs_begin,
                  lhs->first, " must not overlap ", rhs->first,
                  " because the CUDA kernel uses restrict pointers");
    }
  }
}

void check_grid2d_tensors(
    int64_t B,
    int64_t T,
    int64_t C,
    const torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& w,
    const torch::Tensor* w0,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& y,
    const torch::Tensor& elapsed_t) {
  const auto device = state.device();
  const int64_t token_numel = B * T * C;
  check_grid2d_tensor(state, "state", at::kHalf, B * C * 64, device, 16);
  check_grid2d_tensor(r, "r", at::kHalf, token_numel, device, 4);
  check_grid2d_tensor(w, "w", at::kHalf, token_numel, device, 4);
  check_grid2d_tensor(k, "k", at::kHalf, token_numel, device, 4);
  check_grid2d_tensor(v, "v", at::kHalf, token_numel, device, 2);
  check_grid2d_tensor(a, "a", at::kHalf, token_numel, device, 4);
  check_grid2d_tensor(b, "b", at::kHalf, token_numel, device, 4);
  check_grid2d_tensor(y, "y", at::kHalf, token_numel, device, 2);
  check_grid2d_tensor(elapsed_t, "elapsed_t", at::kInt, B, device, 4);
  if (w0 != nullptr) {
    check_grid2d_tensor(*w0, "w0", at::kHalf, C, device, 2);
    check_grid2d_no_overlap({
        {"state", &state}, {"r", &r}, {"w", &w}, {"w0", w0},
        {"k", &k}, {"v", &v}, {"a", &a}, {"b", &b}, {"y", &y},
        {"elapsed_t", &elapsed_t}});
  } else {
    check_grid2d_no_overlap({
        {"state", &state}, {"r", &r}, {"w", &w}, {"k", &k},
        {"v", &v}, {"a", &a}, {"b", &b}, {"y", &y},
        {"elapsed_t", &elapsed_t}});
  }
}

void wkv_seq_grid2d(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t) {
  check_grid2d_dims(B, T, C, H);
  check_grid2d_tensors(B, T, C, state, r, w, nullptr, k, v, a, b, y, elapsed_t);
  wkv_seq_grid2d_v2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      state, r, w, k, v, a, b, y, elapsed_t);
}

void wkv_seq_w0_grid2d(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t) {
  check_grid2d_dims(B, T, C, H);
  check_grid2d_tensors(B, T, C, state, r, w, &w0, k, v, a, b, y, elapsed_t);
  wkv_seq_w0_grid2d_v2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      state, r, w, w0, k, v, a, b, y, elapsed_t);
}

void wkv_seq_grid2d_forced(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    int64_t mode,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t) {
  check_grid2d_dims(B, T, C, H);
  TORCH_CHECK(T > 1, "wkv_seq_grid2d_forced is a T>1 tuning-only entry");
  TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 (exact) or 1 (seq_v2)");
  check_grid2d_tensors(B, T, C, state, r, w, nullptr, k, v, a, b, y, elapsed_t);
  wkv_seq_grid2d_forced_v2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      static_cast<int>(mode), state, r, w, k, v, a, b, y, elapsed_t);
}

void wkv_seq_w0_grid2d_forced(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    int64_t mode,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t) {
  check_grid2d_dims(B, T, C, H);
  TORCH_CHECK(T > 1, "wkv_seq_w0_grid2d_forced is a T>1 tuning-only entry");
  TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 (exact) or 1 (seq_v2)");
  check_grid2d_tensors(B, T, C, state, r, w, &w0, k, v, a, b, y, elapsed_t);
  wkv_seq_w0_grid2d_forced_v2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      static_cast<int>(mode), state, r, w, w0, k, v, a, b, y, elapsed_t);
}

void wkv_one(
    int64_t B,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t) {
  wkv_one_v2_cuda(
      static_cast<int>(B),
      static_cast<int>(C),
      static_cast<int>(H),
      state,
      r,
      w,
      k,
      v,
      a,
      b,
      y,
      elapsed_t);
}

void wkv_one_w0(
    int64_t B,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t) {
  wkv_one_w0_v2_cuda(
      static_cast<int>(B),
      static_cast<int>(C),
      static_cast<int>(H),
      state,
      r,
      w,
      w0,
      k,
      v,
      a,
      b,
      y,
      elapsed_t);
}

void wkv_seq_forced(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    int64_t mode,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t) {
  TORCH_CHECK(T > 1, "wkv_seq_forced is a T>1 tuning-only entry");
  TORCH_CHECK(mode == 0 || mode == 1, "wkv_seq_forced mode must be 0 (exact) or 1 (seq_v2)");
  wkv_seq_forced_v2_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      static_cast<int>(mode),
      state,
      r,
      w,
      k,
      v,
      a,
      b,
      y,
      elapsed_t);
}

void wkv_seq_w0_forced(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    int64_t mode,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor elapsed_t) {
  TORCH_CHECK(T > 1, "wkv_seq_w0_forced is a T>1 tuning-only entry");
  TORCH_CHECK(mode == 0 || mode == 1, "wkv_seq_w0_forced mode must be 0 (exact) or 1 (seq_v2)");
  wkv_seq_w0_forced_v2_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      static_cast<int>(mode),
      state,
      r,
      w,
      w0,
      k,
      v,
      a,
      b,
      y,
      elapsed_t);
}

TORCH_LIBRARY(rwkv7_wkv_fp16_v2, m) {
  m.def("wkv_seq", wkv_seq);
  m.def("wkv_seq_w0", wkv_seq_w0);
  m.def("wkv_seq_grid2d", wkv_seq_grid2d);
  m.def("wkv_seq_w0_grid2d", wkv_seq_w0_grid2d);
  m.def("wkv_one", wkv_one);
  m.def("wkv_one_w0", wkv_one_w0);
  // Forced variants are intentionally tuning-only. Production keeps the auto
  // launcher so unsupported B/T shapes cannot accidentally bypass its policy.
  m.def("wkv_seq_forced", wkv_seq_forced);
  m.def("wkv_seq_w0_forced", wkv_seq_w0_forced);
  m.def("wkv_seq_grid2d_forced", wkv_seq_grid2d_forced);
  m.def("wkv_seq_w0_grid2d_forced", wkv_seq_w0_grid2d_forced);
}
