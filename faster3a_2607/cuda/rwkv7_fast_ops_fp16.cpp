#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

std::vector<torch::Tensor> tmix_mix6_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);
std::vector<torch::Tensor> tmix_mix6_cfg_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    int threads);
std::vector<torch::Tensor> tmix_mix6_3d_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);
void tmix_mix6_cfg_out_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor out_r,
    torch::Tensor out_w,
    torch::Tensor out_k,
    torch::Tensor out_v,
    torch::Tensor out_a,
    torch::Tensor out_g);
void tmix_mix6_3d_out_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor out_r,
    torch::Tensor out_w,
    torch::Tensor out_k,
    torch::Tensor out_v,
    torch::Tensor out_a,
    torch::Tensor out_g);
std::vector<torch::Tensor> tmix_mix6_t1_c4096_cuda(
    int B,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    int threads,
    int vec,
    bool half_math);

std::vector<torch::Tensor> tmix_kk_a_gate_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor x,
    torch::Tensor shift_state,
    bool update_shift);
std::vector<torch::Tensor> tmix_kk_a_gate_2d_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a);

torch::Tensor tmix_lnx_rkvres_xg_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g);
torch::Tensor tmix_lnx_rkvres_xg_warp_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g);
torch::Tensor tmix_lnx_rkvres_xg_warp_2d_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g);

torch::Tensor tmix_vres_gate_cuda(
    int B,
    int T,
    int C,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12);
torch::Tensor tmix_vres_gate_cfg_cuda(
    int B,
    int T,
    int C,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12,
    int threads,
    bool vectorized);
void tmix_vres_gate_cfg_out_cuda(
    int B,
    int T,
    int C,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12,
    torch::Tensor out,
    int threads,
    bool vectorized);

torch::Tensor cmix_sparse_one_cuda(
    int C,
    int F,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_rows_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_one_cuda(
    int C,
    int F,
    torch::Tensor act,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_rows_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor act,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_one_cuda(
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_one_split2_cuda(
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_rows_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_rows_split2_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_rows_t512_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc);
torch::Tensor cmix_sparse_down_relu_rows_t512_cfg_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    int accumulators);
void cmix_sparse_down_relu_rows_t512_cfg_out_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    torch::Tensor out,
    int accumulators);
torch::Tensor cmix_sparse_down_relu_rows_t512_reuse_cfg_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    int accumulators);
void cmix_sparse_down_relu_rows_t512_reuse_cfg_out_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    torch::Tensor out,
    int accumulators);

torch::Tensor cmix_mix_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k);
torch::Tensor cmix_mix_cfg_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    int threads);
void cmix_mix_cfg_out_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor out,
    int threads);
torch::Tensor cmix_mix_3d_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k);
void cmix_mix_3d_out_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor out);

torch::Tensor relu_square_cuda(torch::Tensor x);

torch::Tensor act_tanh_cuda(torch::Tensor x);

torch::Tensor act_sigmoid_cuda(torch::Tensor x);

torch::Tensor add_vec_cuda(int C, torch::Tensor x, torch::Tensor vec);
torch::Tensor add_vec_2d_cuda(int C, torch::Tensor x, torch::Tensor vec);
void add_vec_cfg_out_cuda(
    int C,
    torch::Tensor x,
    torch::Tensor vec,
    torch::Tensor out,
    bool grid2d);

namespace {

void check_half_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, name, " must be fp16");
}

void check_3d(const torch::Tensor& x, int64_t B, int64_t T, int64_t C, const char* name) {
  check_half_cuda_contig(x, name);
  TORCH_CHECK(x.dim() == 3, name, " must have shape [B,T,C]");
  TORCH_CHECK(x.size(0) == B && x.size(1) == T && x.size(2) == C, name, " shape mismatch");
}

void check_vec(const torch::Tensor& x, int64_t C, const char* name) {
  check_half_cuda_contig(x, name);
  TORCH_CHECK(x.dim() == 1 && x.size(0) == C, name, " must have shape [C]");
}

void check_half2_aligned(const torch::Tensor& x, const char* name) {
  TORCH_CHECK((reinterpret_cast<std::uintptr_t>(x.data_ptr()) & 0x3u) == 0,
              name, " must be 4-byte aligned for half2 access");
}

void check_no_storage_overlap(
    const torch::Tensor& output,
    const char* output_name,
    const torch::Tensor& other,
    const char* other_name) {
  const auto output_begin = reinterpret_cast<std::uintptr_t>(output.data_ptr());
  const auto other_begin = reinterpret_cast<std::uintptr_t>(other.data_ptr());
  const auto output_end = output_begin + output.nbytes();
  const auto other_end = other_begin + other.nbytes();
  TORCH_CHECK(output_end <= other_begin || other_end <= output_begin,
              output_name, " must not overlap ", other_name);
}

int64_t check_head_grid_dims(int64_t B, int64_t T, int64_t C, int64_t H, bool grouped_heads) {
  TORCH_CHECK(B > 0 && B <= std::numeric_limits<int>::max(), "B must be positive int32");
  TORCH_CHECK(T > 0 && T <= std::numeric_limits<int>::max(), "T must be positive int32");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max() && (C % 64) == 0,
              "C must be positive int32 divisible by 64");
  TORCH_CHECK(H > 0 && H <= std::numeric_limits<int>::max() && H == C / 64,
              "only head size 64 is supported");
  if (grouped_heads) {
    TORCH_CHECK((H % 4) == 0, "2D KK head grid requires H divisible by 4");
  }
  TORCH_CHECK(B <= 65535 / T, "2D head grid requires B*T <= 65535");
  return B * T;
}

void check_half2_same_device(
    const std::vector<std::pair<const torch::Tensor*, const char*>>& tensors) {
  TORCH_CHECK(!tensors.empty(), "internal error: empty tensor list");
  const int device = tensors.front().first->get_device();
  for (const auto& [tensor, name] : tensors) {
    check_half2_aligned(*tensor, name);
    TORCH_CHECK(tensor->get_device() == device, "all tensors must be on the same CUDA device");
  }
}

}  // namespace

std::vector<torch::Tensor> tmix_mix6(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_r, C, "x_r");
  check_vec(x_w, C, "x_w");
  check_vec(x_k, C, "x_k");
  check_vec(x_v, C, "x_v");
  check_vec(x_a, C, "x_a");
  check_vec(x_g, C, "x_g");
  return tmix_mix6_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g);
}

std::vector<torch::Tensor> tmix_mix6_cfg(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    int64_t threads) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 || threads == 1024, "unsupported threads");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_r, C, "x_r");
  check_vec(x_w, C, "x_w");
  check_vec(x_k, C, "x_k");
  check_vec(x_v, C, "x_v");
  check_vec(x_a, C, "x_a");
  check_vec(x_g, C, "x_g");
  return tmix_mix6_cfg_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, static_cast<int>(threads));
}

void check_tmix_mix6_strict_inputs(
    int64_t B,
    int64_t T,
    int64_t C,
    const torch::Tensor& x,
    const torch::Tensor& shift_state,
    const torch::Tensor& x_r,
    const torch::Tensor& x_w,
    const torch::Tensor& x_k,
    const torch::Tensor& x_v,
    const torch::Tensor& x_a,
    const torch::Tensor& x_g,
    bool grid3d) {
  TORCH_CHECK(B > 0 && B <= std::numeric_limits<int>::max(), "B must be positive int32");
  TORCH_CHECK(T > 0 && T <= std::numeric_limits<int>::max(), "T must be positive int32");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max() && (C % 2) == 0,
              "C must be positive even int32");
  if (grid3d) {
    TORCH_CHECK(B <= 65535 && T <= 65535, "3D TMix requires B/T <= 65535");
  }
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  const std::vector<std::pair<const torch::Tensor*, const char*>> mixes = {
      {&x_r, "x_r"}, {&x_w, "x_w"}, {&x_k, "x_k"},
      {&x_v, "x_v"}, {&x_a, "x_a"}, {&x_g, "x_g"}};
  check_half2_aligned(x, "x");
  check_half2_aligned(shift_state, "shift_state");
  TORCH_CHECK(x.get_device() == shift_state.get_device(), "all tensors must be on the same CUDA device");
  check_no_storage_overlap(shift_state, "shift_state", x, "x");
  for (const auto& [mix, name] : mixes) {
    check_vec(*mix, C, name);
    check_half2_aligned(*mix, name);
    TORCH_CHECK(x.get_device() == mix->get_device(), "all tensors must be on the same CUDA device");
    check_no_storage_overlap(shift_state, "shift_state", *mix, name);
  }
}

void check_tmix_mix6_outputs(
    int64_t B,
    int64_t T,
    int64_t C,
    const torch::Tensor& x,
    const torch::Tensor& shift_state,
    const torch::Tensor& x_r,
    const torch::Tensor& x_w,
    const torch::Tensor& x_k,
    const torch::Tensor& x_v,
    const torch::Tensor& x_a,
    const torch::Tensor& x_g,
    const torch::Tensor& out_r,
    const torch::Tensor& out_w,
    const torch::Tensor& out_k,
    const torch::Tensor& out_v,
    const torch::Tensor& out_a,
    const torch::Tensor& out_g) {
  const std::vector<std::pair<const torch::Tensor*, const char*>> inputs = {
      {&x, "x"}, {&shift_state, "shift_state"}, {&x_r, "x_r"}, {&x_w, "x_w"},
      {&x_k, "x_k"}, {&x_v, "x_v"}, {&x_a, "x_a"}, {&x_g, "x_g"}};
  const std::vector<std::pair<const torch::Tensor*, const char*>> outputs = {
      {&out_r, "out_r"}, {&out_w, "out_w"}, {&out_k, "out_k"},
      {&out_v, "out_v"}, {&out_a, "out_a"}, {&out_g, "out_g"}};
  for (const auto& [output, output_name] : outputs) {
    check_3d(*output, B, T, C, output_name);
    check_half2_aligned(*output, output_name);
    TORCH_CHECK(output->get_device() == x.get_device(), "all outputs must be on the input CUDA device");
    for (const auto& [input, input_name] : inputs) {
      check_no_storage_overlap(*output, output_name, *input, input_name);
    }
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (size_t j = i + 1; j < outputs.size(); ++j) {
      check_no_storage_overlap(*outputs[i].first, outputs[i].second,
                               *outputs[j].first, outputs[j].second);
    }
  }
}

std::vector<torch::Tensor> tmix_mix6_3d(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
  check_tmix_mix6_strict_inputs(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, true);
  return tmix_mix6_3d_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g);
}

void tmix_mix6_cfg_out(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor out_r,
    torch::Tensor out_w,
    torch::Tensor out_k,
    torch::Tensor out_v,
    torch::Tensor out_a,
    torch::Tensor out_g) {
  check_tmix_mix6_strict_inputs(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, false);
  check_tmix_mix6_outputs(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
                          out_r, out_w, out_k, out_v, out_a, out_g);
  tmix_mix6_cfg_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
      out_r, out_w, out_k, out_v, out_a, out_g);
}

void tmix_mix6_3d_out(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor out_r,
    torch::Tensor out_w,
    torch::Tensor out_k,
    torch::Tensor out_v,
    torch::Tensor out_a,
    torch::Tensor out_g) {
  check_tmix_mix6_strict_inputs(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, true);
  check_tmix_mix6_outputs(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
                          out_r, out_w, out_k, out_v, out_a, out_g);
  tmix_mix6_3d_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
      out_r, out_w, out_k, out_v, out_a, out_g);
}

std::vector<torch::Tensor> tmix_mix6_t1_c4096(
    int64_t B,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    int64_t threads,
    int64_t vec,
    bool half_math) {
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 || threads == 1024, "unsupported threads");
  TORCH_CHECK(vec == 1 || vec == 2 || vec == 4 || vec == 8, "unsupported vec");
  check_3d(x, B, 1, 4096, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == 4096,
              "shift_state must have shape [B,4096]");
  check_vec(x_r, 4096, "x_r");
  check_vec(x_w, 4096, "x_w");
  check_vec(x_k, 4096, "x_k");
  check_vec(x_v, 4096, "x_v");
  check_vec(x_a, 4096, "x_a");
  check_vec(x_g, 4096, "x_g");
  return tmix_mix6_t1_c4096_cuda(
      static_cast<int>(B), x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
      static_cast<int>(threads), static_cast<int>(vec), half_math);
}

std::vector<torch::Tensor> tmix_kk_a_gate(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(k, B, T, C, "k");
  check_vec(k_k, C, "k_k");
  check_vec(a0, C, "a0");
  check_3d(a12, B, T, C, "a12");
  check_vec(k_a, C, "k_a");
  return tmix_kk_a_gate_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      k, k_k, a0, a12, k_a, k, k, false);
}

std::vector<torch::Tensor> tmix_kk_a_gate_2d(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a) {
  check_head_grid_dims(B, T, C, H, true);
  check_3d(k, B, T, C, "k");
  check_vec(k_k, C, "k_k");
  check_vec(a0, C, "a0");
  check_3d(a12, B, T, C, "a12");
  check_vec(k_a, C, "k_a");
  check_half2_same_device({
      {&k, "k"}, {&k_k, "k_k"}, {&a0, "a0"}, {&a12, "a12"}, {&k_a, "k_a"}});
  return tmix_kk_a_gate_2d_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      k, k_k, a0, a12, k_a);
}

std::vector<torch::Tensor> tmix_kk_a_gate_update_shift(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor x,
    torch::Tensor shift_state) {
  TORCH_CHECK(T == 1, "tmix_kk_a_gate_update_shift currently requires T=1");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(k, B, T, C, "k");
  check_vec(k_k, C, "k_k");
  check_vec(a0, C, "a0");
  check_3d(a12, B, T, C, "a12");
  check_vec(k_a, C, "k_a");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  return tmix_kk_a_gate_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      k, k_k, a0, a12, k_a, x, shift_state, true);
}

torch::Tensor tmix_lnx_rkvres_xg(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(x, B, T, C, "x");
  check_3d(r, B, T, C, "r");
  check_3d(k, B, T, C, "k");
  check_3d(v, B, T, C, "v");
  check_3d(g, B, T, C, "g");
  check_vec(r_k, C, "r_k");
  check_vec(weight, C, "weight");
  check_vec(bias, C, "bias");
  return tmix_lnx_rkvres_xg_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      x, r, k, v, r_k, weight, bias, g);
}

torch::Tensor tmix_lnx_rkvres_xg_warp(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(x, B, T, C, "x");
  check_3d(r, B, T, C, "r");
  check_3d(k, B, T, C, "k");
  check_3d(v, B, T, C, "v");
  check_3d(g, B, T, C, "g");
  check_vec(r_k, C, "r_k");
  check_vec(weight, C, "weight");
  check_vec(bias, C, "bias");
  return tmix_lnx_rkvres_xg_warp_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      x, r, k, v, r_k, weight, bias, g);
}

torch::Tensor tmix_lnx_rkvres_xg_warp_2d(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g) {
  check_head_grid_dims(B, T, C, H, false);
  check_3d(x, B, T, C, "x");
  check_3d(r, B, T, C, "r");
  check_3d(k, B, T, C, "k");
  check_3d(v, B, T, C, "v");
  check_3d(g, B, T, C, "g");
  check_vec(r_k, C, "r_k");
  check_vec(weight, C, "weight");
  check_vec(bias, C, "bias");
  check_half2_same_device({
      {&x, "x"}, {&r, "r"}, {&k, "k"}, {&v, "v"}, {&r_k, "r_k"},
      {&weight, "weight"}, {&bias, "bias"}, {&g, "g"}});
  return tmix_lnx_rkvres_xg_warp_2d_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      x, r, k, v, r_k, weight, bias, g);
}

torch::Tensor tmix_vres_gate(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12) {
  check_3d(v, B, T, C, "v");
  check_3d(v_first, B, T, C, "v_first");
  check_vec(v0, C, "v0");
  check_3d(v12, B, T, C, "v12");
  return tmix_vres_gate_cuda(static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), v, v_first, v0, v12);
}

torch::Tensor tmix_vres_gate_cfg(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12,
    int64_t threads,
    bool vectorized) {
  check_3d(v, B, T, C, "v");
  check_3d(v_first, B, T, C, "v_first");
  check_vec(v0, C, "v0");
  check_3d(v12, B, T, C, "v12");
  TORCH_CHECK(threads == 64 || threads == 128 || threads == 256 || threads == 512,
              "threads must be 64, 128, 256, or 512");
  TORCH_CHECK(v.get_device() == v_first.get_device() && v.get_device() == v0.get_device() &&
                  v.get_device() == v12.get_device(),
              "all tensors must be on the same CUDA device");
  if (vectorized) {
    TORCH_CHECK((C % 2) == 0, "vectorized vres gate requires even C");
    TORCH_CHECK(B * T <= 65535, "vectorized vres gate requires B*T <= 65535");
    check_half2_aligned(v, "v");
    check_half2_aligned(v_first, "v_first");
    check_half2_aligned(v0, "v0");
    check_half2_aligned(v12, "v12");
  }
  return tmix_vres_gate_cfg_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      v, v_first, v0, v12, static_cast<int>(threads), vectorized);
}

void tmix_vres_gate_cfg_out(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12,
    torch::Tensor out,
    int64_t threads,
    bool vectorized) {
  check_3d(v, B, T, C, "v");
  check_3d(v_first, B, T, C, "v_first");
  check_vec(v0, C, "v0");
  check_3d(v12, B, T, C, "v12");
  check_3d(out, B, T, C, "out");
  TORCH_CHECK(threads == 64 || threads == 128 || threads == 256 || threads == 512,
              "threads must be 64, 128, 256, or 512");
  TORCH_CHECK(v.get_device() == v_first.get_device() && v.get_device() == v0.get_device() &&
                  v.get_device() == v12.get_device() && v.get_device() == out.get_device(),
              "all tensors must be on the same CUDA device");
  // All CUDA pointers are __restrict__; output/input aliasing would invalidate that contract.
  if (out.numel() != 0) {
    TORCH_CHECK(out.data_ptr() != v.data_ptr() && out.data_ptr() != v_first.data_ptr() &&
                    out.data_ptr() != v0.data_ptr() && out.data_ptr() != v12.data_ptr(),
                "out must not alias any input");
  }
  if (vectorized) {
    TORCH_CHECK((C % 2) == 0, "vectorized vres gate requires even C");
    TORCH_CHECK(B * T <= 65535, "vectorized vres gate requires B*T <= 65535");
    check_half2_aligned(v, "v");
    check_half2_aligned(v_first, "v_first");
    check_half2_aligned(v0, "v0");
    check_half2_aligned(v12, "v12");
    check_half2_aligned(out, "out");
  }
  tmix_vres_gate_cfg_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      v, v_first, v0, v12, out, static_cast<int>(threads), vectorized);
}

torch::Tensor cmix_sparse_one(
    int64_t C,
    int64_t F,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor value_fc) {
  check_3d(x, 1, 1, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == 1 && shift_state.size(1) == C,
              "shift_state must have shape [1,C]");
  check_vec(x_k, C, "x_k");
  check_half_cuda_contig(key_fc, "key_fc");
  TORCH_CHECK(key_fc.dim() == 2 && key_fc.size(0) == F && key_fc.size(1) == C,
              "key_fc must have shape [F,C]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_one_cuda(
      static_cast<int>(C), static_cast<int>(F), x, shift_state, x_k, key_fc, value_fc);
}

torch::Tensor cmix_sparse_rows(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor value_fc) {
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  check_half_cuda_contig(key_fc, "key_fc");
  TORCH_CHECK(key_fc.dim() == 2 && key_fc.size(0) == F && key_fc.size(1) == C,
              "key_fc must have shape [F,C]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_rows_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      x, shift_state, x_k, key_fc, value_fc);
}

torch::Tensor cmix_sparse_down_one(
    int64_t C,
    int64_t F,
    torch::Tensor act,
    torch::Tensor value_fc) {
  check_half_cuda_contig(act, "act");
  TORCH_CHECK(act.dim() == 1 && act.size(0) == F, "act must have shape [F]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_one_cuda(static_cast<int>(C), static_cast<int>(F), act, value_fc);
}

torch::Tensor cmix_sparse_down_rows(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor act,
    torch::Tensor value_fc) {
  check_3d(act, B, T, F, "act");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_rows_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      act, value_fc);
}

torch::Tensor cmix_sparse_down_relu_one(
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc) {
  check_half_cuda_contig(preact, "preact");
  TORCH_CHECK(preact.dim() == 1 && preact.size(0) == F, "preact must have shape [F]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_relu_one_cuda(static_cast<int>(C), static_cast<int>(F), preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_one_split2(
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc) {
  check_half_cuda_contig(preact, "preact");
  TORCH_CHECK(preact.dim() == 1 && preact.size(0) == F, "preact must have shape [F]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_relu_one_split2_cuda(
      static_cast<int>(C), static_cast<int>(F), preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_rows(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc) {
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_relu_rows_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_rows_split2(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc) {
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_relu_rows_split2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_rows_t512(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc) {
  TORCH_CHECK(B > 0 && T > 0 && B <= 65535 / T,
              "t512 sparse down requires 0 < B*T <= 65535");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max(), "C must be positive int32");
  TORCH_CHECK(F > 0 && F <= std::numeric_limits<int>::max(), "F must be positive int32");
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  check_half2_same_device({{&preact, "preact"}, {&value_fc, "value_fc"}});
  check_no_storage_overlap(preact, "preact", value_fc, "value_fc");
  return cmix_sparse_down_relu_rows_t512_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_rows_t512_cfg(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    int64_t accumulators) {
  TORCH_CHECK(accumulators == 1 || accumulators == 2 || accumulators == 4,
              "accumulators must be 1, 2, or 4");
  TORCH_CHECK(B > 0 && T > 0 && B <= 65535 / T,
              "t512 sparse down requires 0 < B*T <= 65535");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max(), "C must be positive int32");
  TORCH_CHECK(F > 0 && F <= std::numeric_limits<int>::max(), "F must be positive int32");
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  check_half2_same_device({{&preact, "preact"}, {&value_fc, "value_fc"}});
  check_no_storage_overlap(preact, "preact", value_fc, "value_fc");
  return cmix_sparse_down_relu_rows_t512_cfg_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc, static_cast<int>(accumulators));
}

void cmix_sparse_down_relu_rows_t512_cfg_out(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    torch::Tensor out,
    int64_t accumulators) {
  TORCH_CHECK(accumulators == 1 || accumulators == 2 || accumulators == 4,
              "accumulators must be 1, 2, or 4");
  TORCH_CHECK(B > 0 && T > 0 && B <= 65535 / T,
              "t512 sparse down requires 0 < B*T <= 65535");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max(), "C must be positive int32");
  TORCH_CHECK(F > 0 && F <= std::numeric_limits<int>::max(), "F must be positive int32");
  check_3d(preact, B, T, F, "preact");
  check_3d(out, B, T, C, "out");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  check_half2_same_device(
      {{&preact, "preact"}, {&value_fc, "value_fc"}, {&out, "out"}});
  // All three device pointers are restrict-qualified in the kernel. Read/read
  // overlap is invalid too, not only output aliasing.
  check_no_storage_overlap(preact, "preact", value_fc, "value_fc");
  check_no_storage_overlap(out, "out", preact, "preact");
  check_no_storage_overlap(out, "out", value_fc, "value_fc");
  cmix_sparse_down_relu_rows_t512_cfg_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc, out, static_cast<int>(accumulators));
}

torch::Tensor cmix_sparse_down_relu_rows_t512_reuse_cfg(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    int64_t accumulators) {
  TORCH_CHECK(accumulators == 1 || accumulators == 2,
              "reuse accumulators must be 1 or 2");
  TORCH_CHECK(B > 0 && T > 0 && B <= 65535 / T,
              "t512 reuse sparse down requires 0 < B*T <= 65535");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max(), "C must be positive int32");
  TORCH_CHECK(F > 0 && F <= std::numeric_limits<int>::max(), "F must be positive int32");
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  check_half2_same_device({{&preact, "preact"}, {&value_fc, "value_fc"}});
  check_no_storage_overlap(preact, "preact", value_fc, "value_fc");
  return cmix_sparse_down_relu_rows_t512_reuse_cfg_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc, static_cast<int>(accumulators));
}

void cmix_sparse_down_relu_rows_t512_reuse_cfg_out(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    torch::Tensor out,
    int64_t accumulators) {
  TORCH_CHECK(accumulators == 1 || accumulators == 2,
              "reuse accumulators must be 1 or 2");
  TORCH_CHECK(B > 0 && T > 0 && B <= 65535 / T,
              "t512 reuse sparse down requires 0 < B*T <= 65535");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max(), "C must be positive int32");
  TORCH_CHECK(F > 0 && F <= std::numeric_limits<int>::max(), "F must be positive int32");
  check_3d(preact, B, T, F, "preact");
  check_3d(out, B, T, C, "out");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  check_half2_same_device(
      {{&preact, "preact"}, {&value_fc, "value_fc"}, {&out, "out"}});
  check_no_storage_overlap(preact, "preact", value_fc, "value_fc");
  check_no_storage_overlap(out, "out", preact, "preact");
  check_no_storage_overlap(out, "out", value_fc, "value_fc");
  cmix_sparse_down_relu_rows_t512_reuse_cfg_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc, out, static_cast<int>(accumulators));
}

torch::Tensor cmix_mix(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  return cmix_mix_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), x, shift_state, x_k);
}

torch::Tensor cmix_mix_cfg(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    int64_t threads) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 || threads == 1024, "unsupported threads");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  return cmix_mix_cfg_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), x, shift_state, x_k, static_cast<int>(threads));
}

void cmix_mix_cfg_out(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor out,
    int64_t threads) {
  TORCH_CHECK(B > 0 && T > 0 && C > 0 && (C % 2) == 0,
              "cmix_mix_cfg_out requires positive B/T and positive even C");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 || threads == 1024,
              "unsupported threads");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  check_3d(out, B, T, C, "out");
  TORCH_CHECK(x.get_device() == shift_state.get_device() && x.get_device() == x_k.get_device() &&
                  x.get_device() == out.get_device(),
              "all tensors must be on the same CUDA device");
  check_half2_aligned(x, "x");
  check_half2_aligned(shift_state, "shift_state");
  check_half2_aligned(x_k, "x_k");
  check_half2_aligned(out, "out");
  check_no_storage_overlap(shift_state, "shift_state", x, "x");
  check_no_storage_overlap(shift_state, "shift_state", x_k, "x_k");
  check_no_storage_overlap(out, "out", x, "x");
  check_no_storage_overlap(out, "out", shift_state, "shift_state");
  check_no_storage_overlap(out, "out", x_k, "x_k");
  cmix_mix_cfg_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      x, shift_state, x_k, out, static_cast<int>(threads));
}

void check_cmix_mix_3d_inputs(
    int64_t B,
    int64_t T,
    int64_t C,
    const torch::Tensor& x,
    const torch::Tensor& shift_state,
    const torch::Tensor& x_k) {
  TORCH_CHECK(B > 0 && B <= 65535, "cmix_mix_3d requires 1 <= B <= 65535");
  TORCH_CHECK(T > 0 && T <= 65535, "cmix_mix_3d requires 1 <= T <= 65535");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max() && (C % 2) == 0,
              "cmix_mix_3d requires positive even C within int32 range");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  TORCH_CHECK(x.get_device() == shift_state.get_device() && x.get_device() == x_k.get_device(),
              "all tensors must be on the same CUDA device");
  check_half2_aligned(x, "x");
  check_half2_aligned(shift_state, "shift_state");
  check_half2_aligned(x_k, "x_k");
  // shift_state is restrict read/write. Aliasing a read-only input would make
  // the T>1 two-launch recurrent-state contract invalid.
  check_no_storage_overlap(shift_state, "shift_state", x, "x");
  check_no_storage_overlap(shift_state, "shift_state", x_k, "x_k");
}

torch::Tensor cmix_mix_3d(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k) {
  check_cmix_mix_3d_inputs(B, T, C, x, shift_state, x_k);
  return cmix_mix_3d_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), x, shift_state, x_k);
}

void cmix_mix_3d_out(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor out) {
  check_cmix_mix_3d_inputs(B, T, C, x, shift_state, x_k);
  check_3d(out, B, T, C, "out");
  TORCH_CHECK(out.get_device() == x.get_device(), "out must be on the same CUDA device");
  check_half2_aligned(out, "out");
  // Byte-range checks also catch shifted contiguous views that share storage.
  check_no_storage_overlap(out, "out", x, "x");
  check_no_storage_overlap(out, "out", shift_state, "shift_state");
  check_no_storage_overlap(out, "out", x_k, "x_k");
  cmix_mix_3d_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), x, shift_state, x_k, out);
}

torch::Tensor relu_square(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  TORCH_CHECK((x.numel() % 2) == 0, "x.numel() must be even");
  return relu_square_cuda(x);
}

torch::Tensor act_tanh(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  TORCH_CHECK((x.numel() % 2) == 0, "x.numel() must be even");
  return act_tanh_cuda(x);
}

torch::Tensor act_sigmoid(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  TORCH_CHECK((x.numel() % 2) == 0, "x.numel() must be even");
  return act_sigmoid_cuda(x);
}

int64_t check_add_vec_inputs(
    int64_t C,
    const torch::Tensor& x,
    const torch::Tensor& vec,
    bool grid2d) {
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max() && (C % 2) == 0,
              "C must be positive even int32");
  check_half_cuda_contig(x, "x");
  check_vec(vec, C, "vec");
  TORCH_CHECK(x.numel() > 0 && (x.numel() % C) == 0,
              "x must contain complete C-wide rows");
  TORCH_CHECK(x.size(-1) == C, "x last dim must equal C");
  check_half2_same_device({{&x, "x"}, {&vec, "vec"}});
  // Both CUDA pointers are restrict-qualified, so even read/read aliasing is
  // outside the kernel contract and must be rejected before launch.
  check_no_storage_overlap(x, "x", vec, "vec");
  const int64_t rows = x.numel() / C;
  if (grid2d) {
    TORCH_CHECK(rows <= 65535, "2D add_vec requires rows <= 65535");
  }
  return rows;
}

torch::Tensor add_vec(int64_t C, torch::Tensor x, torch::Tensor vec) {
  check_add_vec_inputs(C, x, vec, false);
  return add_vec_cuda(static_cast<int>(C), x, vec);
}

torch::Tensor add_vec_2d(int64_t C, torch::Tensor x, torch::Tensor vec) {
  check_add_vec_inputs(C, x, vec, true);
  return add_vec_2d_cuda(static_cast<int>(C), x, vec);
}

void add_vec_cfg_out(
    int64_t C,
    torch::Tensor x,
    torch::Tensor vec,
    torch::Tensor out,
    bool grid2d) {
  check_add_vec_inputs(C, x, vec, grid2d);
  check_half_cuda_contig(out, "out");
  TORCH_CHECK(out.sizes() == x.sizes(), "out shape must match x");
  check_half2_same_device({{&x, "x"}, {&vec, "vec"}, {&out, "out"}});
  // The tuning entrypoint writes a caller-owned output; reject even shifted
  // aliases because restrict would otherwise turn overlap into undefined behavior.
  check_no_storage_overlap(out, "out", x, "x");
  check_no_storage_overlap(out, "out", vec, "vec");
  add_vec_cfg_out_cuda(static_cast<int>(C), x, vec, out, grid2d);
}

TORCH_LIBRARY(rwkv7_fast_ops_fp16, m) {
  m.def(
      "tmix_mix6(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
  m.def(
      "tmix_mix6_cfg(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, int threads) -> Tensor[]");
  m.def(
      "tmix_mix6_3d(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
  m.def(
      "tmix_mix6_cfg_out(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, "
      "Tensor(b!) out_r, Tensor(c!) out_w, Tensor(d!) out_k, Tensor(e!) out_v, Tensor(f!) out_a, Tensor(g!) out_g) -> ()");
  m.def(
      "tmix_mix6_3d_out(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, "
      "Tensor(b!) out_r, Tensor(c!) out_w, Tensor(d!) out_k, Tensor(e!) out_v, Tensor(f!) out_a, Tensor(g!) out_g) -> ()");
  m.def(
      "tmix_mix6_t1_c4096(int B, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, int threads, int vec, bool half_math=False) -> Tensor[]");
  m.def(
      "tmix_kk_a_gate(int B, int T, int C, int H, Tensor k, Tensor k_k, Tensor a0, Tensor a12, Tensor k_a) -> Tensor[]");
  m.def(
      "tmix_kk_a_gate_2d(int B, int T, int C, int H, Tensor k, Tensor k_k, Tensor a0, Tensor a12, Tensor k_a) -> Tensor[]");
  m.def(
      "tmix_kk_a_gate_update_shift(int B, int T, int C, int H, Tensor k, Tensor k_k, Tensor a0, Tensor a12, Tensor k_a, Tensor x, Tensor(a!) shift_state) -> Tensor[]");
  m.def(
      "tmix_lnx_rkvres_xg(int B, int T, int C, int H, Tensor x, Tensor r, Tensor k, Tensor v, "
      "Tensor r_k, Tensor weight, Tensor bias, Tensor g) -> Tensor");
  m.def(
      "tmix_lnx_rkvres_xg_warp(int B, int T, int C, int H, Tensor x, Tensor r, Tensor k, Tensor v, "
      "Tensor r_k, Tensor weight, Tensor bias, Tensor g) -> Tensor");
  m.def(
      "tmix_lnx_rkvres_xg_warp_2d(int B, int T, int C, int H, Tensor x, Tensor r, Tensor k, Tensor v, "
      "Tensor r_k, Tensor weight, Tensor bias, Tensor g) -> Tensor");
  m.def(
      "tmix_vres_gate(int B, int T, int C, Tensor v, Tensor v_first, Tensor v0, Tensor v12) -> Tensor");
  m.def(
      "tmix_vres_gate_cfg(int B, int T, int C, Tensor v, Tensor v_first, Tensor v0, Tensor v12, int threads, bool vectorized) -> Tensor");
  m.def(
      "tmix_vres_gate_cfg_out(int B, int T, int C, Tensor v, Tensor v_first, Tensor v0, Tensor v12, Tensor(a!) out, int threads, bool vectorized) -> ()");
  m.def(
      "cmix_sparse_one(int C, int F, Tensor x, Tensor(a!) shift_state, Tensor x_k, Tensor key_fc, Tensor value_fc) -> Tensor");
  m.def(
      "cmix_sparse_rows(int B, int T, int C, int F, Tensor x, Tensor(a!) shift_state, Tensor x_k, Tensor key_fc, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_one(int C, int F, Tensor act, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_rows(int B, int T, int C, int F, Tensor act, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_relu_one(int C, int F, Tensor preact, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_relu_one_split2(int C, int F, Tensor preact, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_relu_rows(int B, int T, int C, int F, Tensor preact, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_relu_rows_split2(int B, int T, int C, int F, Tensor preact, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_relu_rows_t512(int B, int T, int C, int F, Tensor preact, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_relu_rows_t512_cfg(int B, int T, int C, int F, Tensor preact, Tensor value_fc, int accumulators) -> Tensor");
  m.def("cmix_sparse_down_relu_rows_t512_cfg_out(int B, int T, int C, int F, Tensor preact, Tensor value_fc, Tensor(a!) out, int accumulators) -> ()");
  m.def("cmix_sparse_down_relu_rows_t512_reuse_cfg(int B, int T, int C, int F, Tensor preact, Tensor value_fc, int accumulators) -> Tensor");
  m.def("cmix_sparse_down_relu_rows_t512_reuse_cfg_out(int B, int T, int C, int F, Tensor preact, Tensor value_fc, Tensor(a!) out, int accumulators) -> ()");
  m.def("cmix_mix(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_k) -> Tensor");
  m.def("cmix_mix_cfg(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_k, int threads) -> Tensor");
  m.def("cmix_mix_cfg_out(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_k, Tensor(b!) out, int threads) -> ()");
  m.def("cmix_mix_3d(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_k) -> Tensor");
  m.def("cmix_mix_3d_out(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_k, Tensor(b!) out) -> ()");
  m.def("relu_square(Tensor x) -> Tensor");
  m.def("act_tanh(Tensor x) -> Tensor");
  m.def("act_sigmoid(Tensor x) -> Tensor");
  m.def("add_vec(int C, Tensor x, Tensor vec) -> Tensor");
  m.def("add_vec_2d(int C, Tensor x, Tensor vec) -> Tensor");
  m.def("add_vec_cfg_out(int C, Tensor x, Tensor vec, Tensor(a!) out, bool grid2d) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_fast_ops_fp16, CUDA, m) {
  m.impl("tmix_mix6", &tmix_mix6);
  m.impl("tmix_mix6_cfg", &tmix_mix6_cfg);
  m.impl("tmix_mix6_3d", &tmix_mix6_3d);
  m.impl("tmix_mix6_cfg_out", &tmix_mix6_cfg_out);
  m.impl("tmix_mix6_3d_out", &tmix_mix6_3d_out);
  m.impl("tmix_mix6_t1_c4096", &tmix_mix6_t1_c4096);
  m.impl("tmix_kk_a_gate", &tmix_kk_a_gate);
  m.impl("tmix_kk_a_gate_2d", &tmix_kk_a_gate_2d);
  m.impl("tmix_kk_a_gate_update_shift", &tmix_kk_a_gate_update_shift);
  m.impl("tmix_lnx_rkvres_xg", &tmix_lnx_rkvres_xg);
  m.impl("tmix_lnx_rkvres_xg_warp", &tmix_lnx_rkvres_xg_warp);
  m.impl("tmix_lnx_rkvres_xg_warp_2d", &tmix_lnx_rkvres_xg_warp_2d);
  m.impl("tmix_vres_gate", &tmix_vres_gate);
  m.impl("tmix_vres_gate_cfg", &tmix_vres_gate_cfg);
  m.impl("tmix_vres_gate_cfg_out", &tmix_vres_gate_cfg_out);
  m.impl("cmix_sparse_one", &cmix_sparse_one);
  m.impl("cmix_sparse_rows", &cmix_sparse_rows);
  m.impl("cmix_sparse_down_one", &cmix_sparse_down_one);
  m.impl("cmix_sparse_down_rows", &cmix_sparse_down_rows);
  m.impl("cmix_sparse_down_relu_one", &cmix_sparse_down_relu_one);
  m.impl("cmix_sparse_down_relu_one_split2", &cmix_sparse_down_relu_one_split2);
  m.impl("cmix_sparse_down_relu_rows", &cmix_sparse_down_relu_rows);
  m.impl("cmix_sparse_down_relu_rows_split2", &cmix_sparse_down_relu_rows_split2);
  m.impl("cmix_sparse_down_relu_rows_t512", &cmix_sparse_down_relu_rows_t512);
  m.impl("cmix_sparse_down_relu_rows_t512_cfg", &cmix_sparse_down_relu_rows_t512_cfg);
  m.impl("cmix_sparse_down_relu_rows_t512_cfg_out", &cmix_sparse_down_relu_rows_t512_cfg_out);
  m.impl("cmix_sparse_down_relu_rows_t512_reuse_cfg", &cmix_sparse_down_relu_rows_t512_reuse_cfg);
  m.impl("cmix_sparse_down_relu_rows_t512_reuse_cfg_out", &cmix_sparse_down_relu_rows_t512_reuse_cfg_out);
  m.impl("cmix_mix", &cmix_mix);
  m.impl("cmix_mix_cfg", &cmix_mix_cfg);
  m.impl("cmix_mix_cfg_out", &cmix_mix_cfg_out);
  m.impl("cmix_mix_3d", &cmix_mix_3d);
  m.impl("cmix_mix_3d_out", &cmix_mix_3d_out);
  m.impl("relu_square", &relu_square);
  m.impl("act_tanh", &act_tanh);
  m.impl("act_sigmoid", &act_sigmoid);
  m.impl("add_vec", &add_vec);
  m.impl("add_vec_2d", &add_vec_2d);
  m.impl("add_vec_cfg_out", &add_vec_cfg_out);
}
