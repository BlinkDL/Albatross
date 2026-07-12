#include <torch/extension.h>

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

void wkv_lnx_seq_w0_v2_cuda(
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
    torch::Tensor elapsed_t,
    torch::Tensor r_k,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    torch::Tensor g,
    torch::Tensor out);

void wkv_lnx_kkag_seq_w0_v2_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor v,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor y,
    torch::Tensor elapsed_t,
    torch::Tensor r_k,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    torch::Tensor g,
    torch::Tensor out);

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

void wkv_lnx_seq_w0(
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
    torch::Tensor elapsed_t,
    torch::Tensor r_k,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    torch::Tensor g,
    torch::Tensor out) {
  wkv_lnx_seq_w0_v2_cuda(
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
      elapsed_t,
      r_k,
      ln_weight,
      ln_bias,
      g,
      out);
}

void wkv_lnx_kkag_seq_w0(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor v,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor y,
    torch::Tensor elapsed_t,
    torch::Tensor r_k,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    torch::Tensor g,
    torch::Tensor out) {
  wkv_lnx_kkag_seq_w0_v2_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      state,
      r,
      w,
      w0,
      k_raw,
      v,
      k_k,
      a0,
      a12,
      k_a,
      y,
      elapsed_t,
      r_k,
      ln_weight,
      ln_bias,
      g,
      out);
}

TORCH_LIBRARY(rwkv7_wkv_fp16_v2, m) {
  m.def("wkv_seq", wkv_seq);
  m.def("wkv_seq_w0", wkv_seq_w0);
  m.def("wkv_one", wkv_one);
  m.def("wkv_one_w0", wkv_one_w0);
  m.def("wkv_lnx_seq_w0", wkv_lnx_seq_w0);
  m.def("wkv_lnx_kkag_seq_w0", wkv_lnx_kkag_seq_w0);
}
