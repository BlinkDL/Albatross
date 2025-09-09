#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>

typedef at::Half dtype;

void cuda_forward_batch(int B, int T, int C, int H, float *state, dtype *r, dtype* w, dtype *k, dtype *v, dtype *a, dtype *b, dtype *y);

void cuda_forward_right_align(int B, int T, int C, int H, float *state, dtype *r, dtype* w, dtype *k, dtype *v, dtype *a, dtype *b, dtype *y, int* Tm);

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y /*, const c10::optional<torch::Tensor>& Tm*/) {
    /*
    if (Tm.has_value()) {
        cuda_forward_right_align(B, T, C, H, state.data_ptr<float>(), r.data_ptr<dtype>(), w.data_ptr<dtype>(), k.data_ptr<dtype>(), v.data_ptr<dtype>(), a.data_ptr<dtype>(), b.data_ptr<dtype>(), y.data_ptr<dtype>(), Tm.value().data_ptr<int>());
    }
    else {
    */
        cuda_forward_batch(B, T, C, H, state.data_ptr<float>(), r.data_ptr<dtype>(), w.data_ptr<dtype>(), k.data_ptr<dtype>(), v.data_ptr<dtype>(), a.data_ptr<dtype>(), b.data_ptr<dtype>(), y.data_ptr<dtype>());
    // }
}

TORCH_LIBRARY(rwkv7_batch_fwd_fp16, m) {
    m.def("forward", forward);
}
