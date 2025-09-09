#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"

typedef at::Half dtype;
typedef const int I;

#define G F *__restrict__ const
#define GG const F *__restrict__ const

template <typename F>
__global__ void kernel_forward_batch(I B, I T, I C, I H, float *__restrict__ _state,
    GG _r, GG _w, GG _k, GG _v, GG _a, GG _b, G _y)
{
    I e = blockIdx.x / H, h = blockIdx.x % H, i = threadIdx.x;
    _state += ((e*H + h) * _N_ + i) * _N_;
    float state[_N_];
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        state[j] = _state[j];

    __shared__ float r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];

#define FL(z,_z) z[i]=float(_z[t])
    for (int _t = 0; _t < T; _t++)
    {
        I t = (e*T + _t) * C + h*_N_ + i;
        __syncthreads();
        FL(r,_r);
        FL(w,_w);
        FL(k,_k);
        FL(a,_a);
        FL(b,_b);
        w[i] = __expf(-__expf(w[i]));
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++){
            sa += a[j] * state[j];
        }
        float vv = float(_v[t]);
        float y = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++){
            float& s = state[j];
            s = s * w[j] + k[j] * vv + sa * b[j];
            y += s * r[j];
        }
        _y[t] = F(y);
    }
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        _state[j] = state[j];    
}

template <typename F>

__global__ void kernel_forward_right_align(I B, I T, I C, I H, float *__restrict__ _state,
    GG _r, GG _w, GG _k, GG _v, GG _a, GG _b, G _y, I* Tm)
    /*
    Tm: batch length - actual length per seq (min 0)
    No initial state, right alignment
    */
{
    I e = blockIdx.x / H, h = blockIdx.x % H, i = threadIdx.x;
    // e=1, h=24, i=13
    _state += ((e*H + h) * _N_ + i) * _N_;
    float state[_N_] = {};
    #pragma unroll
    for (int j = 0; j < _N_; j++){
        state[j] = 0;
        _state[j] = 0;
    }

    __shared__ float r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];

    for (int _t = 0; _t < T; _t++)
    {
        I t = (e*T + _t) * C + h*_N_ + i;
        __syncthreads();
        r[i] = float(_r[t]);
        w[i] = __expf(-__expf(float(_w[t])));
        k[i] = float(_k[t]);
        a[i] = float(_a[t]);
        b[i] = float(_b[t]);
        __syncthreads();
        
        if (_t >= Tm[e]) {
            float sa = 0;
            #pragma unroll
            for (int j = 0; j < _N_; j++){
                sa += a[j] * state[j];
            }

            float vv = float(_v[t]);
            float y = 0;
            #pragma unroll
            for (int j = 0; j < _N_; j++)
            {
                float& s = state[j];
                s = s * w[j] + k[j] * vv + sa * b[j];
                y += s * r[j];
            }
            _y[t] = F(y);
        }
        else {
            _y[t] = 0;
        }
    }
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        _state[j] = state[j];    
}



// template <int block_size>
// static __global__ void rwkv_wkv7_f32(const int B, const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * a, const float * b, const float * s, float * dst) {
//     const int tid = threadIdx.x;
//     const int bid = blockIdx.x;

//     const int head_size = block_size;
//     const int batch_i = bid / H;
//     const int head_i = bid % H;
//     const int state_size = C * head_size;
//     const int n_seq_tokens = T / B;

//     float state[head_size];
//     __shared__ float _r[head_size], _w[head_size], _k[head_size], _a[head_size], _b[head_size];

// #ifndef GGML_USE_MUSA
//     #pragma unroll
// #endif
//     for (int i = 0; i < head_size; i++) {
//         state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
//     }

//     for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
//         __syncthreads();
//         _r[tid] = r[t];
//         _w[tid] = w[t];
//         _k[tid] = k[t];
//         _a[tid] = a[t];
//         _b[tid] = b[t];
//         __syncthreads();

//         float sa = 0;
//         #pragma unroll
//         for (int j = 0; j < head_size; j += 4)
//         {
//             const float4& a = (float4&)(_a[j]);
//             const float4& s = (float4&)(state[j]);
//             sa += a.x * s.x;
//             sa += a.y * s.y;
//             sa += a.z * s.z;
//             sa += a.w * s.w;
//         }

//         const float _v = v[t];
//         float y = 0;
//         for (int j = 0; j < head_size; j += 4) {
//             const float4& r = (float4&)(_r[j]);
//             const float4& w = (float4&)(_w[j]);
//             const float4& k = (float4&)(_k[j]);
//             const float4& b = (float4&)(_b[j]);
//             float4& s = (float4&)(state[j]);
//             float4 kv;

//             kv.x = k.x * _v;
//             kv.y = k.y * _v;
//             kv.z = k.z * _v;
//             kv.w = k.w * _v;

//             s.x = s.x * w.x + kv.x + sa * b.x;
//             s.y = s.y * w.y + kv.y + sa * b.y;
//             s.z = s.z * w.z + kv.z + sa * b.z;
//             s.w = s.w * w.w + kv.w + sa * b.w;

//             y += s.x * r.x;
//             y += s.y * r.y;
//             y += s.z * r.z;
//             y += s.w * r.w;
//         }
//         dst[t] = y;
//     }

//     #pragma unroll
//     for (int i = 0; i < head_size; i++) {
//         dst[T * C + batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
//     }
// }


void cuda_forward_batch(int B, int T, int C, int H, float *state, dtype *r, dtype* w, dtype *k, dtype *v, dtype *a, dtype *b, dtype *y)
{
    assert(H*_N_ == C);
    // assert(B == 1); // only for B=1
    kernel_forward_batch<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, state, r, w, k, v, a, b, y);
}

void cuda_forward_right_align(int B, int T, int C, int H, float *state, dtype *r, dtype* w, dtype *k, dtype *v, dtype *a, dtype *b, dtype *y, int* Tm)
{
    assert(H*_N_ == C);
    // assert(B == 1); // only for B=1
    kernel_forward_right_align<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, state, r, w, k, v, a, b, y, Tm);
}

