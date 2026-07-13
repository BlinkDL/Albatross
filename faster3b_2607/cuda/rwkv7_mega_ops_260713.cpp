#include <torch/extension.h>

using T = torch::Tensor;

T emb_ln0_bf16_to_f16_cuda(T emb, T weight, T bias, double eps);
void emb_ln_mix6_into_cuda(T emb, T tokens, T shift, T weight, T bias, T xr, T xw, T xk, T xv, T xa, T xg, T x_out, T out_r, T out_w, T out_k, T out_v, T out_a, T out_g, double eps);
void ln_mix6_into_cuda(T x, T residual, T shift, T weight, T bias, T xr, T xw, T xk, T xv, T xa, T xg, T x_out, T out_r, T out_w, T out_k, T out_v, T out_a, T out_g, double eps);
void rkv_lowrank_pre_into_cuda(T xr, T xk, T xv, T wr, T wk, T wv, T yr, T yk, T yv, T xw, T xa, T xg, T xlr_v, T w1_t, T a1_t, T g1_t, T v1_t, T w1, T a1, T g1, T v1, bool skip_v);
void rankout_into_cuda(T w1, T a1, T g1, T v1, T w2_t, T a2_t, T g2_t, T v2_t, T v, T v_first, T v0, T k_raw, T k_k, T a0, T k_a, T w0, T w, T a, T g, T v_out, T new_k, T neg_kk, T kka, bool skip_v);
void wkv_into_cuda(T state, T r, T w, T k, T v, T a, T b, T y);
void lnx_into_cuda(T x, T r, T k, T v, T r_k, T weight, T bias, T g, T out);
void att_out_into_cuda(T x, T weight, T out);
void ln_cmix_into_cuda(T x, T residual, T shift, T weight, T bias, T x_k, T x_out, T mixed, double eps);
void cmix_key_into_cuda(T x, T weight, T out);
void cmix_value_into_cuda(T act, T weight, T out);
void final_ln_into_cuda(T x, T residual, T weight, T bias, T out, double eps);
void head_into_cuda(T x, T weight, T out);

TORCH_LIBRARY(rwkv7_mega_ops_260713, m) {
    m.def("emb_ln0_bf16_to_f16(Tensor emb, Tensor weight, Tensor bias, float eps) -> Tensor");
    m.def("emb_ln_mix6_into(Tensor emb, Tensor tokens, Tensor(a!) shift, Tensor weight, Tensor bias, Tensor xr, Tensor xw, Tensor xk, Tensor xv, Tensor xa, Tensor xg, Tensor(b!) x_out, Tensor(c!) out_r, Tensor(d!) out_w, Tensor(e!) out_k, Tensor(f!) out_v, Tensor(g!) out_a, Tensor(h!) out_g, float eps) -> ()");
    m.def("ln_mix6_into(Tensor x, Tensor residual, Tensor(a!) shift, Tensor weight, Tensor bias, Tensor xr, Tensor xw, Tensor xk, Tensor xv, Tensor xa, Tensor xg, Tensor(b!) x_out, Tensor(c!) out_r, Tensor(d!) out_w, Tensor(e!) out_k, Tensor(f!) out_v, Tensor(g!) out_a, Tensor(h!) out_g, float eps) -> ()");
    m.def("rkv_lowrank_pre_into(Tensor xr, Tensor xk, Tensor xv, Tensor wr, Tensor wk, Tensor wv, Tensor(a!) yr, Tensor(b!) yk, Tensor(c!) yv, Tensor xw, Tensor xa, Tensor xg, Tensor xlr_v, Tensor w1_t, Tensor a1_t, Tensor g1_t, Tensor v1_t, Tensor(d!) w1, Tensor(e!) a1, Tensor(f!) g1, Tensor(g!) v1, bool skip_v) -> ()");
    m.def("rankout_into(Tensor w1, Tensor a1, Tensor g1, Tensor v1, Tensor w2_t, Tensor a2_t, Tensor g2_t, Tensor v2_t, Tensor v, Tensor v_first, Tensor v0, Tensor k_raw, Tensor k_k, Tensor a0, Tensor k_a, Tensor w0, Tensor(a!) w, Tensor(b!) a, Tensor(c!) g, Tensor(d!) v_out, Tensor(e!) new_k, Tensor(f!) neg_kk, Tensor(g!) kka, bool skip_v) -> ()");
    m.def("wkv_into(Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(c!) y) -> ()");
    m.def("lnx_into(Tensor x, Tensor r, Tensor k, Tensor v, Tensor r_k, Tensor weight, Tensor bias, Tensor g, Tensor(a!) out) -> ()");
    m.def("att_out_into(Tensor x, Tensor weight, Tensor(a!) out) -> ()");
    m.def("ln_cmix_into(Tensor x, Tensor residual, Tensor(a!) shift, Tensor weight, Tensor bias, Tensor x_k, Tensor(b!) x_out, Tensor(c!) mixed, float eps) -> ()");
    m.def("cmix_key_into(Tensor x, Tensor weight, Tensor(a!) out) -> ()");
    m.def("cmix_value_into(Tensor act, Tensor weight, Tensor(a!) out) -> ()");
    m.def("final_ln_into(Tensor x, Tensor residual, Tensor weight, Tensor bias, Tensor(a!) out, float eps) -> ()");
    m.def("head_into(Tensor x, Tensor weight, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_mega_ops_260713, CUDA, m) {
    m.impl("emb_ln0_bf16_to_f16", &emb_ln0_bf16_to_f16_cuda);
    m.impl("emb_ln_mix6_into", &emb_ln_mix6_into_cuda);
    m.impl("ln_mix6_into", &ln_mix6_into_cuda);
    m.impl("rkv_lowrank_pre_into", &rkv_lowrank_pre_into_cuda);
    m.impl("rankout_into", &rankout_into_cuda);
    m.impl("wkv_into", &wkv_into_cuda);
    m.impl("lnx_into", &lnx_into_cuda);
    m.impl("att_out_into", &att_out_into_cuda);
    m.impl("ln_cmix_into", &ln_cmix_into_cuda);
    m.impl("cmix_key_into", &cmix_key_into_cuda);
    m.impl("cmix_value_into", &cmix_value_into_cuda);
    m.impl("final_ln_into", &final_ln_into_cuda);
    m.impl("head_into", &head_into_cuda);
}
