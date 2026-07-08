#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <iostream>
#include <assert.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

#ifndef _N_
#define _N_ 64
#endif
#define BLOCKDIM 128
#define MAXNPERBLOCK 128

typedef half F;
typedef curandStatePhilox4_32_10_t RAND;

constexpr float two_to_neg_41 = 4.547473508864641e-13f;
constexpr float nexp_half_log2_e = -0.8750387749145276f, nlog2_e = -1.4426950408889634f;
constexpr int ro1 = (int)2654435769;
#define rotator1(_A) (two_to_neg_41*float(ro1*(_A)))

template <typename T, typename ReduceOp>
__device__ __forceinline__ void warpReduceAll(T& val, ReduceOp op) {
    #pragma unroll
    for (int offset = 16; offset>0; offset/=2) {
        val = op(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
}

// Block-level reduction - ALL threads get the result
template <typename T, typename ReduceOp, int BLOCK_SIZE=1024, bool monotone_sum=false>
__device__ __forceinline__ void blockReduceAll(T& val, ReduceOp op, T identity, void* buf) {
    T* warpResults = reinterpret_cast<T*>(buf);
    const int lane = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;
    const int numWarps = (BLOCK_SIZE + 31) / 32;
    warpReduceAll(val, op);
    if (lane == 31) warpResults[warpId] = val;
    __syncthreads();
    T warpVal;
    if constexpr (!monotone_sum) {
        warpVal = (threadIdx.x < numWarps) ? warpResults[threadIdx.x] : identity;
        if (threadIdx.x < 32) warpReduceAll(warpVal, op);
        if (threadIdx.x == 0) warpResults[0] = warpVal;
    }
    else {
        if (threadIdx.x == 0){
            warpVal = warpResults[0];
            #pragma unroll
            for (int i=1; i<numWarps; i++){
                warpVal += warpResults[i];
            }
            warpResults[0] = warpVal;
        }
    }
    __syncthreads();
    val = warpResults[0];
}

template <typename T>
__device__ __forceinline__ T warpInclusiveScan(T val) {
    #pragma unroll
    for (int offset = 16; offset>0; offset/=2) {
        T n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x % 32 >= offset) {
            val += n;
        }
    }
    return val;
}

// Block-level inclusive scan - each thread gets sum of itself and all preceding threads
template <typename T, int BLOCK_SIZE = 1024>
__device__ __forceinline__ T blockInclusiveScan(T val, void* buf /* shared */ , void* total=nullptr) {
    T* warpSums = reinterpret_cast<T*>(buf);
    
    const int lane = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;
    constexpr int numWarps = (BLOCK_SIZE + 31) / 32;
    
    // Step 1: Inclusive scan within each warp (ok)
    T val1 = warpInclusiveScan(val);
    
    // Step 2: Last lane of each warp stores its total
    if (lane == 31) {
        warpSums[warpId] = val1;
    }
    __syncthreads();
    
    // Step 3: First warp does inclusive scan of warp totals
    // if (threadIdx.x < numWarps) {
    //     T warpTotal = warpSums[threadIdx.x];
    //     warpTotal = warpInclusiveScan(warpTotal);
    //     warpSums[threadIdx.x] = warpTotal;
    // }
    // MUST sum this way to ensure numerical MONOTONICITY (not STABILITY)
    if (threadIdx.x == 0){
        T s = warpSums[0];
        #pragma unroll
        for (int i=1; i<numWarps; i++){
            s += warpSums[i];
            warpSums[i] = s;
        }
    }
    __syncthreads();
    
    // Step 4: Add previous warp's prefix to current value
    if (warpId > 0) {
        val1 += warpSums[warpId - 1];
    }
    if (threadIdx.x == BLOCK_SIZE-1 && total != nullptr) {
        *reinterpret_cast<T*>(total) = val1;
    }
    __syncthreads();
    return val1;
}

// Reduction operation functors
template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
    static constexpr T identity() { return T(0); }
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return max(a, b); }
    static constexpr T identity() { return -INFINITY; }  // For float
};

template <typename T>
struct MinOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return min(a, b); }
    static constexpr T identity() { return INFINITY; }   // For float
};

template <typename T>
struct ProdOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return a * b; }
    static constexpr T identity() { return T(1); }
};

__device__ __forceinline__ float sf(float x){
    float y = isnan(x) ? 0.0f : x;
    return (isinf(y) ? copysignf(FLT_MAX, y) : y);
}

union common128 {
    int4 I;
    struct {int x,y,z,w;} J;
    struct {float x,y,z,w;} F;
    struct {double x,y;} D;
    struct {half2 x,y,z,w;} G;
    struct {half a,b,c,d,e,f,g,h;} H;
    half h[8];
    half2 h2[4];
    unsigned short s[8];
    int i[4];
    float f[4];
};

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

template <int N>
__device__ __forceinline__ void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *const global_ptr, bool cond) {
    static_assert(N == 16 || N == 8 || N == 4);
    int bytes = cond ? N : 0;
    unsigned int addr = __cvta_generic_to_shared(smem_addr);
    if constexpr (N == 16) {
        asm volatile(
            #if ENABLE_L2_PREFETCH
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
            #else
            "cp.async.cg.shared.global [%0], [%1], %2, %3;"
            #endif
            ::"r"(addr),
            "l"(global_ptr), "n"(N), "r"(bytes));
    } else {
        asm volatile(
            #if ENABLE_L2_PREFETCH
            "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;"
            #else
            "cp.async.ca.shared.global [%0], [%1], %2, %3;"
            #endif
            ::"r"(addr),
            "l"(global_ptr), "n"(N), "r"(bytes));
    }
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    if constexpr (N == 0) {
        asm volatile("cp.async.wait_all;\n" ::);
    } else {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <bool Tis1=false>
__global__ void __launch_bounds__(_N_, 2) kernel_forward_w0_fp16_dither(
    const int B, const int T, const int C, const int H,
    F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
    F *__restrict__ const _y, const int *__restrict__ const _elapsed_t){
    
    if constexpr (Tis1) {
        __builtin_assume(T==1);
    }
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    const int L = i%32;

    __shared__ __align__(256) half2 state_smem[_N_][_N_ / 2];

    _state += bbb * C * _N_ + h * _N_ * _N_;
    constexpr int ldg_size = sizeof(int4) / sizeof(F);
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++){
        int4 state_vec = ((int4 *)_state)[j0 * _N_ + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++){
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            state_smem[row][(row % 32) ^ col] = ((half2 *)&state_vec)[j1];
        }
    }
    __syncthreads();
    half2 state[_N_ / 2];
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state[j] = state_smem[i][L^j];
    
    __shared__ __align__(128) half2 r[_N_ / 2], k[_N_ / 2], w[_N_ / 2], a[_N_ / 2], b[_N_ / 2];
    #pragma unroll
    for (int _t = 0; _t < T; _t++){
        int t = bbb*T*C + h*_N_ + _t * C; // + i
        __syncthreads();
        cp_async_gs_conditional<4>((half2*)(i<32?w:a)+L, (half2*)((i<32?_w:_a)+t)+L, true);
        cp_async_commit();
        cp_async_gs_conditional<4>((half2*)(i<32?r:k)+L, (half2*)((i<32?_r:_k)+t)+L, true);
        cp_async_gs_conditional<4>((half2*)b+L, (half2*)(_b+t)+L, i<32);
        cp_async_commit();
        half vv = _v[t+i];
        half2 vv2 = {vv, vv};
        half2 y2 = {0., 0.};
        half2 sa2 = {0., 0.};
        cp_async_wait<1>();
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++)
            sa2 = __hfma2(a[j], state[j], sa2);
        half sa = sa2.x + sa2.y;
        sa2 = {sa, sa};
        ((F*)w)[i] = F(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * (float)(((F*)w)[i])))) - 1.0f + rotator1(_elapsed_t[bbb]+_t));

        cp_async_wait<0>();
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++){
            half2 &s = state[j];
            s = __hfma2(s, w[j], __hfma2(k[j], vv2, __hfma2(sa2, b[j], s)));
            y2 = __hfma2(s, r[j], y2);
        }
        _y[t+i] = y2.x + y2.y;
    }
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state_smem[i][L^j] = state[j];
    __syncthreads();
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++){
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++){
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            ((half2 *)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4 *)_state)[j0 * _N_ + i] = state_vec;
    }
}


void forward_one(int64_t B, int64_t C, int64_t H, at::Tensor &state, at::Tensor &r, at::Tensor &w, at::Tensor &k, at::Tensor &v, at::Tensor &a, at::Tensor &b, at::Tensor &y, at::Tensor &elapsed_t){
    assert(H * _N_ == C);
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward_w0_fp16_dither<1><<<B * H, _N_, 0, stream>>>(
        B, 1, C, H, 
        (F*)state.data_ptr(), 
        (const F*)r.data_ptr(), 
        (const F*)w.data_ptr(),
        (const F*)k.data_ptr(), 
        (const F*)v.data_ptr(),
        (const F*)a.data_ptr(),
        (const F*)b.data_ptr(),
        (F*)y.data_ptr(), 
        elapsed_t.data_ptr<int>()
    );
}

void forward_seq(int64_t B, int64_t T, int64_t C, int64_t H, at::Tensor &state, at::Tensor &r, at::Tensor &w, at::Tensor &k, at::Tensor &v, at::Tensor &a, at::Tensor &b, at::Tensor &y, at::Tensor &elapsed_t){
    assert(H * _N_ == C);
    kernel_forward_w0_fp16_dither<<<B * H, _N_>>>(
        B, T, C, H, 
        (F*)state.data_ptr(), 
        (const F*)r.data_ptr(), 
        (const F*)w.data_ptr(),
        (const F*)k.data_ptr(), 
        (const F*)v.data_ptr(),
        (const F*)a.data_ptr(),
        (const F*)b.data_ptr(),
        (F*)y.data_ptr(), 
        elapsed_t.data_ptr<int>()
    );
}

__global__ void __launch_bounds__(BLOCKDIM, 4) spvecmatmul_noindices(
    const int C,
    const half* __restrict__ vec,
    const half* __restrict__ mat,
    half* __restrict__ out
){
    __builtin_assume(blockDim.x == BLOCKDIM);
    __shared__ __align__(256) half mat_row_smem[2][2*BLOCKDIM];
    __shared__ __align__(256) half vec_slice[MAXNPERBLOCK];
    __shared__ __align__(256) int nnz_ids[MAXNPERBLOCK];
    __shared__ int nnz_count;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int t = threadIdx.x;
    const int warp_id = t >> 5;
    const int lane    = t & 31;
    const int start_pos = bx * MAXNPERBLOCK;

    bool vne0;
    int local_pos;
    constexpr int active_warps = MAXNPERBLOCK/32;
    __shared__ int warp_counts[active_warps], warp_prefix[active_warps];

    if (t < MAXNPERBLOCK/2){
        *(half2*)(vec_slice + t*2) = *(const half2*)(vec + start_pos + t*2);
    }
    __syncthreads();

    if (t < MAXNPERBLOCK){
        vne0 = bool(__half_as_ushort(vec_slice[t]) << 1);
        unsigned mask = __ballot_sync(0xffffffffu, vne0);
        local_pos = __popc(mask & ((1u << lane) - 1u));
        if (lane == 0)
            warp_counts[warp_id] = __popc(mask);
    }
    __syncthreads();

    if (t == 0) {
        int s = 0;
        #pragma unroll
        for (int w = 0; w < active_warps; ++w) {
            warp_prefix[w] = s;
            s += warp_counts[w];
        }
        nnz_count = s;
    }
    __syncthreads();

    if (t < MAXNPERBLOCK && vne0) {
        nnz_ids[warp_prefix[warp_id] + local_pos] = t;
    }
    __syncthreads();

    half2 out_frag;
    *(int*)(&out_frag) = 0;
    // init
    #pragma unroll
    for(int i = 0; i < 2; i++){
        if (i < nnz_count){
            int actual_pos = start_pos + nnz_ids[i];
            cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * C + by * (2*BLOCKDIM) + t*2, true);
            cp_async_commit();
        }
    }
    // main for
    for(int i = 0; i < nnz_count-2; i++){
        // take data
        cp_async_wait<1>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[i%2] + t*2);
        half vec_value = vec_slice[nnz_ids[i]];

        // store
        int actual_pos = start_pos + nnz_ids[i+2];
        cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * C + by * (2*BLOCKDIM) + t*2, true);
        cp_async_commit();

        // compute
        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }

    // end
    if (nnz_count >= 2){
        cp_async_wait<1>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[nnz_count%2] + t*2);
        half vec_value = vec_slice[nnz_ids[nnz_count - 2]];

        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }
    if (nnz_count >= 1){
        cp_async_wait<0>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[(nnz_count+1)%2] + t*2);
        half vec_value = vec_slice[nnz_ids[nnz_count - 1]];

        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }
    atomicAdd((half2*)(out + by*(2*BLOCKDIM) + t*2), out_frag);
}

void spmv_forward(int64_t D, int64_t C, at::Tensor &vec1, at::Tensor &mat, at::Tensor &out) {
    assert(C % (2*BLOCKDIM) == 0);
    assert(D % MAXNPERBLOCK == 0);
    auto stream = at::cuda::getCurrentCUDAStream();
    // cudaMemsetAsync(out, 0, C*sizeof(half), stream);
    spvecmatmul_noindices<<<dim3(D/MAXNPERBLOCK, C/(2*BLOCKDIM), 1), dim3(BLOCKDIM, 1, 1), 0, stream>>>(
        C, (const F*)vec1.data_ptr(), (const F*)mat.data_ptr(), (F*)out.data_ptr()
    );
}

#define XBLOCK 2
#define BLOCKDIMX_CMIX 512
#define NUMWARPS (BLOCKDIMX_CMIX/32)

__global__ void __launch_bounds__(BLOCKDIMX_CMIX, 2) cmix_up_kernel(
    const __half* __restrict__ x_0,
    const __half* __restrict__ x_1,
    const __half* __restrict__ x_k,
    const __half* __restrict__ key,
          __half* __restrict__ out,
    int indim,
    int outdim
){
    const int bx = blockIdx.x;
    const int t = threadIdx.x;
    float acc = 0.f;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    const auto h0 = (const half2(*)[BLOCKDIMX_CMIX])(((const half2*)x_0) + t);
    const auto h1 = (const half2(*)[BLOCKDIMX_CMIX])(((const half2*)x_1) + t);
    const auto h2 = (const half2(*)[BLOCKDIMX_CMIX])(((const half2*)x_k) + t);
    const auto h3 = (const half2(*)[BLOCKDIMX_CMIX])(((const half2*)(key + bx * indim)) + t);

    int N = indim / (BLOCKDIMX_CMIX * 2);

    #pragma unroll
    for (int j=0; j < N; j++) {
        __half2 a0 = *(const __half2*)(h0[j]);
        __half2 a1 = *(const __half2*)(h1[j]);
        __half2 a2 = *(const __half2*)(h2[j]);
        __half2 a3 = *(const __half2*)(h3[j]);
        __half2 diff = __hsub2(a1, a0);
        __half2 interp = __hfma2(diff, a2, a0);
        __half2 prod = __hmul2(interp, a3);
        float2 f = __half22float2(prod);
        acc += f.x + f.y;
    }

    float warp_sum = warp_reduce_sum(acc);

    __shared__ float s[BLOCKDIMX_CMIX/32];
    if (lane == 0) s[warp] = warp_sum;
    __syncthreads();

    float total = 0.f;
    if (warp == 0 && lane < (BLOCKDIMX_CMIX/32))
        total = s[lane];

    if (warp == 0)
        total = warp_reduce_sum(total);

    if (threadIdx.x == 0) {
        float relu = max(total, 0.f);
        out[bx] = __float2half_rn(relu * relu);
    }
}

void cmix_up(
    int64_t indim,
    int64_t outdim,
    at::Tensor &x_0,
    at::Tensor &x_1,
    at::Tensor &x_k,
    at::Tensor &key,
    at::Tensor &out)
{
    // int indim = out.size(1);
    // int outdim = out.size(0);
    // std::cout << indim << ' ' << outdim << std::endl;
    auto stream = at::cuda::getCurrentCUDAStream();
    // (outdim + XBLOCK - 1) / XBLOCK
    cmix_up_kernel<<<outdim, BLOCKDIMX_CMIX, 0, stream>>>(
    // cmix_up_kernel_2<<<((outdim + XBLOCK - 1) / XBLOCK), BLOCKDIMX_CMIX, 0, stream>>>(
        (F*)(x_0.data_ptr()),
        (F*)(x_1.data_ptr()),
        (F*)(x_k.data_ptr()),
        (F*)(key.data_ptr()),
        (F*)(out.data_ptr()),
        indim,
        outdim
    );
}


#define COPY_ZERO_X 128
template <typename T> 
__global__ __launch_bounds__(COPY_ZERO_X, 4) void copy_zero_kernel(const T* __restrict__ a, T* __restrict__ b, T* __restrict__ c, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int4* src  = (const int4*)a;
        int4*       dst  = (int4*)b;
        int4*       dst2 = (int4*)c;
        int4 v = src[i];
        dst[i] = v;
        dst2[i] = make_int4(0, 0, 0, 0);
    }
}

void copy_zero(at::Tensor &a, at::Tensor &b, at::Tensor &c){
    size_t n_uint4 = a.numel() * sizeof(F) / sizeof(int4);
    auto stream = at::cuda::getCurrentCUDAStream();
    copy_zero_kernel<<<(n_uint4+(COPY_ZERO_X-1))/COPY_ZERO_X, COPY_ZERO_X, 0, stream>>>(
        (const F*)a.data_ptr(), (F*)b.data_ptr(), (F*)c.data_ptr(), n_uint4);
}


template <bool Tis1=false>
__global__ void __launch_bounds__(COPY_ZERO_X, 1) shift_conv(
    const int B,
    const int T,
    const int C,
    // Inputs
    const half* __restrict__ x,
          half* __restrict__ x_prev,
    const half* __restrict__ x_mixing,
    // Outputs
    half* __restrict__ xout
) {
    if constexpr (Tis1) {
        __builtin_assume(T == 1);
    }
    const int C2 = C/8;
    const int bt = blockIdx.x;
    const int b = blockIdx.x / T;
    const int t = blockIdx.x % T;
    const int S = B*T*C2;
    for (int c = threadIdx.x; c<C2; c += blockDim.x) {
        const int cur = bt*C2+c;
        common128 x_val;
        x_val.I = ((const int4*)x)[cur]; // [b][t][c]
        common128 x_shifted_val;
        if (t == 0) {
            x_shifted_val.I = ((const int4*)x_prev)[b*C2+c];
        } 
        else {
            x_shifted_val.I = ((const int4*)x)[cur-C2]; // [b][t-1][c]
        }
        common128 x_diff;
        #pragma unroll
        for (int i=0; i<4; i++) {
            x_diff.h2[i] = __hsub2(x_shifted_val.h2[i], x_val.h2[i]);
        }
        #pragma unroll
        for (int q=0; q<6; q++){
            common128 xmix_coeff;
            xmix_coeff.I = ((const int4*)x_mixing)[q*C2+c]; //[q][c]
            common128 result;
            #pragma unroll
            for (int i=0; i<4; i++) {
                result.h2[i] = __hfma2(x_diff.h2[i], xmix_coeff.h2[i], x_val.h2[i]);
            }
            ((int4*)xout)[q*S+bt*C2+c] = result.I;
        }
    }
    if (t == T-1) {
        for (int c = threadIdx.x; c<C2; c += blockDim.x) {
            ((int4*)x_prev)[b*C2+c] = ((const int4*)x)[bt*C2+c];
        }
    }
}


__global__ void setup_rand_kernel(RAND* states, unsigned long long seed) {
    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

at::Tensor setup_rand(int64_t seed, int64_t B){
    at::Tensor state = at::zeros({(long)(B*sizeof(RAND))}, at::TensorOptions().dtype(at::kChar).device(at::kCUDA));
    setup_rand_kernel<<<((int)B), 1>>>((RAND*)(state.data_ptr()), (unsigned long long)seed);
    return state;
}


// #define P0i(x) do{printf(#x":%d\n",x);}while(0)
// #define P0f(x) do{printf(#x":%8e\n",x);}while(0)

__device__ __forceinline__ void print_bits_u32(unsigned v)
{
    for (int bit = 0; bit < 32; ++bit) {
        if(bit%8==0) printf(" ");
        printf("%d", int(bool(v & (1<<bit))));
    }
    printf("\n");
}

__device__ __forceinline__ void dump_thread_states(const unsigned int* s_state, int nthreads)
{
    if (threadIdx.x != 0) return;

    for (int i = 0; i < nthreads; ++i) {
        printf("%3d:", i*32);
        print_bits_u32(s_state[i]);
    }
    printf("\n");
}


#define BLOCKDIM_X_SAMPLE 1024
__global__ void __launch_bounds__(BLOCKDIM_X_SAMPLE, 1) batch_sampling_repetition_temperature_topk_topp_kernel(
    const int B,
    const int T,                         // should be 1 typically; may not be 1 if full output is obtained
    const int V,                         // vocabulary size, 60,000 ~ 120,000
    const float *__restrict__ logits,    // (B, V) if T == 1; If T != 1, only logits[:, T-1, :] is read. This avoids another copying operation
          float *__restrict__ penalties, // (B, V), can set some to -INF for masking
          int   *__restrict__ outputs,   // (B,)
          RAND  *__restrict__ states,    // random state, typedef curandStatePhilox4_32_10_t RAND;
          float *__restrict__ plogits,   // penaltized logits (in L2 cache)
    const float presence_penalty,
    const float repetition_penalty,
    const float penalty_decay,
    const float temperature,
    const int   top_k,
    const float top_p
) {
    const int b = blockIdx.x;
    const int d = blockDim.x;
    const int t = threadIdx.x;
    const int w = t / 32;
    const int l = t % 32;
    // constexpr int W = (BLOCKDIM_X_SAMPLE + 31) / 32;
    __shared__ __align__(256) char reduce_buf[256];
    assert(BLOCKDIM_X_SAMPLE == d);
    assert(V % 4 == 0);
    assert(V <= 1048576);
    assert(temperature > 0.f);
    const int V4 = V / 4;
    float4 l4, p4;

    logits    += (b*T+(T-1)) * V;  // B T V
    penalties +=           b * V;  // B V
    outputs   +=           b    ;  // B
    states    +=           b    ;  // B
    plogits   += (b*T+(T-1)) * V;  // B T V

    float maxu = -INFINITY;
    for (int i=t; i<V4; i+=d) {
        l4 = ((float4*)logits)[i];
        p4 = ((float4*)penalties)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &fl = ((float*)&l4)[j];
            // if (i*4+j < 3){
            //     P0i(i*4+j);
            //     P0f(fl);
            // }
            float &fp = ((float*)&p4)[j];
            fl = sf(sf(fl-fp) / temperature);
            maxu = max(maxu, fl);
            // ((float*)&l4)[j] = fr;
        }
        ((float4*)plogits)[i] = l4;
    }
    blockReduceAll(maxu, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
    __syncthreads();
    // if(t==0){
    //     P0f(maxu);
    //     P0f(temperature);
    // }
    float exp_denom = 0;
    for (int i=t; i<V4; i+=d) {
        l4 = ((float4*)plogits)[i];
        float em = 0.f;
        #pragma unroll
        for (int j=0; j<4; j++){
            float &fr = ((float*)&l4)[j];
            em += expf(fr-maxu);
        }
        exp_denom += em;
    }
    blockReduceAll(exp_denom, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    float pmax = -INFINITY;
    float pmin = +INFINITY;
    for (int i=t; i<V4; i+=d) {
        l4 = ((float4*)plogits)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &fr = ((float*)&l4)[j];
            fr = expf(fr-maxu) / exp_denom;
            pmax = max(pmax, fr);
            pmin = min(pmin, fr);
            // ((float*)&l4)[j] = fr;
        }
        ((float4*)plogits)[i] = l4;
    }
    blockReduceAll(pmax, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
    __syncthreads();
    blockReduceAll(pmin, MinOp<float>{}, MinOp<float>::identity(), reduce_buf);
    __syncthreads();

    // if(t==0) P0f(pmax);
    unsigned left =  __float_as_uint(pmin), right =  __float_as_uint(pmax) + 1;

    uint4 cnt = make_uint4((unsigned)V, 0, 0, 0);
    l4 = make_float4(1, 0, 0, 0);
    uint4 pivot;
    while ((cnt.x > top_k || l4.x > top_p) && left < right-1) {
        // if(t==0){
        //     P0i(top_k);
        //     P0i(left);
        //     P0i(right);
        //     P0i(cnt.x);
        //     printf("\n");
        // }
        pivot.x = left;
        pivot.z = (left            + right) / 2;
        pivot.y = (left  + pivot.z        ) / 2;
        pivot.w = (        pivot.z + right) / 2;
        l4.y = l4.z = l4.w = 0;
        cnt.y = cnt.z = cnt.w = 0;
        for (int i=t; i<V4; i+=d) {
            p4 = ((float4*)plogits)[i];
            #pragma unroll
            for (int j=0; j<4; j++){
                float &p = ((float*)&p4)[j];
                bool u = (p >= __uint_as_float(pivot.y));
                cnt.y += u;
                l4.y = fmaf(p, u, l4.y);
                u = (p >= __uint_as_float(pivot.z));
                cnt.z += u;
                l4.z = fmaf(p, u, l4.z);
                u = (p >= __uint_as_float(pivot.w));
                cnt.w += u;
                l4.w = fmaf(p, u, l4.w);
            }
        }
        blockReduceAll(cnt.y, SumOp<unsigned>{}, SumOp<unsigned>::identity(), reduce_buf);
        __syncthreads();
        blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(l4.y, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
        __syncthreads();
        if (cnt.y < top_k && l4.y < top_p){
            left = pivot.x;
            right = pivot.y;
            // cnt.x = cnt.x;
            // l4.x = l4.x;
            continue;
        }
        blockReduceAll(cnt.z, SumOp<unsigned>{}, SumOp<unsigned>::identity(), reduce_buf);
        __syncthreads();
        blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(l4.z, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
        __syncthreads();
        if (cnt.z < top_k && l4.z < top_p){
            left = pivot.y;
            right = pivot.z;
            cnt.x = cnt.y;
            l4.x = l4.y;
            continue;
        }
        blockReduceAll(cnt.w, SumOp<unsigned>{}, SumOp<unsigned>::identity(), reduce_buf);
        __syncthreads();
        blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(l4.w, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
        __syncthreads();
        if (cnt.w < top_k && l4.w < top_p){
            left = pivot.z;
            right = pivot.w;
            cnt.x = cnt.z;
            l4.x = l4.z;
            continue;
        }
        left = pivot.w;
        // right = right;
        cnt.x = cnt.w;
        l4.x = l4.w;
    }
    // return left
    float threshold =  __uint_as_float(left);
    // if(t==0) P0f(threshold);
    // 5. recompute (read once)
    float gtp=0;
    unsigned eqk=0, gtk=0;
    __shared__ float /* seqp, */ sgtp;
    __shared__ unsigned seqk, sgtk;

    for (int i=t; i<V4; i+=d) {
        p4 = ((float4*)plogits)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &p = ((float*)&p4)[j];
            bool u0 = (p == threshold);
            bool u1 = (p > threshold);
            eqk += u0;
            gtk += u1;
            gtp = fmaf(p, u1, gtp);
        }
    }
    // s: shared all
    // c: cumulative
    // -: per thread
    // __syncthreads();
    float    cgtp = blockInclusiveScan(gtp, reduce_buf, &sgtp);
    __syncthreads();
    unsigned ceqk = blockInclusiveScan(eqk, reduce_buf, &seqk);
    __syncthreads();
    unsigned cgtk = blockInclusiveScan(gtk, reduce_buf, &sgtk);
    __syncthreads();
    // if(t==0) P0f(sgtp);
    // if(t==0) P0i(seqk);
    // if(t==0) P0i(sgtk);

    // compute compensation
    // seqk == total number of tokens that equals threshold
    // _gtp + threshold * _eqk == _eqp
    // (top_p - sgtp) == delta_p
    // delta_p / seqp
    unsigned neqk = seqk;
    float comp=1.0f;
    if (neqk > 0){
        comp = min(sf((top_p - sgtp) / (threshold * neqk)), comp);
        comp = min(sf(float(top_k - sgtk) / neqk), comp);
        comp = max(comp, 0.0f);
    }

    // 6. Yield sampled tokens
    __shared__ float randp, sum_p;
    __shared__ float4 rand4;
    __shared__ int idxt;
    float actual_p = gtp + (threshold * eqk) * comp;
    __syncthreads();
    float cumu_p = blockInclusiveScan(actual_p, reduce_buf, &sum_p);
    __syncthreads();
    if (t==0){
        idxt = 0;
        rand4 = curand_uniform4(states);
        randp = sum_p * rand4.x; // only once 
    }
    __syncthreads();
    
    bool u = (randp <= cumu_p);
    // at last thread: randp = sum_p * rand4.x < cumu_p == sum_p, u == 1
    if(l==31) ((unsigned*)reduce_buf)[w] = u;
    __syncthreads();
    bool u_ = __shfl_up_sync(0xffffffff, u, 1);
    if(t==0) u_=0;
    else if(l==0) u_ = ((unsigned*)reduce_buf)[w-1];
    __syncthreads();

    if (u!=u_) idxt=t;
    __syncthreads();

    // a sub-tile (of no more than 1024)
    int idn = idxt*4 + (t/4)*4*d + (t%4);
    // .... .... (idxt) |||| .... .... .... |||| .... .... .... |||| ....
    float o0 = (idn<V) ? (plogits[idn]) : 0;
    float o = (o0 < threshold) ? 0 : (o0 == threshold) ? (o0 * comp) : o0;

    __shared__ float sum_o;
    float cumu_o = blockInclusiveScan(o, reduce_buf, &sum_o); // monotone
    __syncthreads();
    float rand_2 = sum_o * rand4.y;
    u = (rand_2 <= cumu_o);
    // at last thread: cumu_o == sum_o, rand4.y < 1, sum_o * rand4.y < cumu_o, u == 1
    if(l==31) ((unsigned*)reduce_buf)[w] = u;
    // u: current u_: prev
    // at first thread: u_ == 0
    u_ = __shfl_up_sync(0xffffffff, u, 1);
    __syncthreads();
    if(t==0) u_=0;
    else if(l==0) u_ = ((unsigned*)reduce_buf)[w-1];
    __syncthreads();

    // write idn
    __shared__ int out_id;
    if (u!=u_) out_id = (idn<V)? idn: 0;
    __syncthreads();
    idn = out_id;
    if (t==0) *outputs = idn;
    // 7. Update penalties 
    for (int i=t; i<V4; i+=d) {
        p4 = ((float4*)penalties)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &p = ((float*)&p4)[j];
            int idp = i*4+j;
            p = fmaf(p, penalty_decay, ((idn != idp) ? 0 : (p == 0 ? presence_penalty : repetition_penalty)));
        }
        ((float4*)penalties)[i] = p4;
    }
}

at::Tensor batch_sampling_repetition_temperature_topk_topp(
    at::Tensor& logits,
    at::Tensor& penalties,
    at::Tensor& states,
    double      presence_penalty,
    double      repetition_penalty,
    double      penalty_decay,
    double      temperature,
    int64_t     top_k,
    double      top_p
) {
    int B, T, V;
    V = logits.size(-1);
    B = (penalties.dim() == 2)? penalties.size(0): 1;
    T = (logits.dim() == 3)? logits.size(1): 1;
    // std::cout << "B: " << B << ", T: " << T << " V: " << V << "\n\n";
    auto stream = at::cuda::getCurrentCUDAStream();
    auto plogits = at::empty({B,V}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    if (B*V*4 <= 4194304) {
        cudaStreamAttrValue stream_attribute;
        stream_attribute.accessPolicyWindow.base_ptr  = plogits.data_ptr();
        stream_attribute.accessPolicyWindow.num_bytes = B*V*4;
        stream_attribute.accessPolicyWindow.hitRatio  = 1;
        stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
    }
    auto out = at::empty({B}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
    if (top_k <= 0) top_k = V;
    if (top_p < 0)  top_p = 1;
    if (top_p == 0) {
        top_k = 1;
        top_p = 1;
    }
    batch_sampling_repetition_temperature_topk_topp_kernel<<<B, 1024, 0, stream>>>(
        B, T, V, 
        (float*)logits.data_ptr(),
        (float*)penalties.data_ptr(),
        (int*)  out.data_ptr(),
        (RAND*) states.data_ptr(),
        (float*)plogits.data_ptr(),
        (float) presence_penalty,
        (float) repetition_penalty,
        (float) penalty_decay,
        (float) temperature,
        (int)   top_k,
        (float) top_p
    );
    return out;
}

