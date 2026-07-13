// Fixed C4096/H64 B1T1 release kernels selected on 2026-07-13.


#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace {
namespace cg = cooperative_groups;

constexpr int HEAD_SIZE = 64;
constexpr float KK_NORMALIZE_EPS = 1.0e-12f;
constexpr float TMIX_LN_X_EPS = 64.0e-5f;
constexpr float W_SCALE_LOG2_E = -0.8750387749145276f;
constexpr float NLOG2_E = -1.4426950408889634f;
constexpr int LN_SMALL_C = 4096;


__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_sum_all_xor(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    }
    return v;
}

__device__ __forceinline__ float halfwarp_sum_all_xor(float v) {
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask, 16);
    }
    return v;
}

__device__ __forceinline__ void warp_sum_quad(float& a, float& b, float& c, float& d) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, offset);
        b += __shfl_down_sync(0xffffffffu, b, offset);
        c += __shfl_down_sync(0xffffffffu, c, offset);
        d += __shfl_down_sync(0xffffffffu, d, offset);
    }
}

__device__ __forceinline__ void warp_sum_oct(float (&v)[8]) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] += __shfl_down_sync(0xffffffffu, v[i], offset);
        }
    }
}

__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float rankout_w_eff(float w) {
    return exp2f(W_SCALE_LOG2_E / (1.0f + exp2f(NLOG2_E * w)));
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t x) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = static_cast<uint32_t>(x) << 16;
    return v.f;
}

__device__ __forceinline__ half2 half2_from_u32(uint32_t x) {
    union {
        uint32_t u;
        half2 h;
    } v;
    v.u = x;
    return v.h;
}

__device__ __forceinline__ void load_half4_float2_u64(const half* __restrict__ p, float2& lo, float2& hi) {
    union {
        unsigned long long u64;
        uint2 u32;
    } v;
    v.u64 = *reinterpret_cast<const unsigned long long*>(p);
    lo = __half22float2(half2_from_u32(v.u32.x));
    hi = __half22float2(half2_from_u32(v.u32.y));
}

__device__ __forceinline__ unsigned long long load_u64_evict_first(const void* p) {
    unsigned long long v;
    asm volatile("ld.global.L1::evict_first.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned long long load_u64_evict_last(const void* p) {
    unsigned long long v;
    asm volatile("ld.global.L1::evict_last.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void load_half4_u64_cache(
    const half* __restrict__ p, bool evict_last, float2& lo, float2& hi) {
    union { unsigned long long u64; uint2 u32; } v;
    v.u64 = evict_last ? load_u64_evict_last(p) : load_u64_evict_first(p);
    lo = __half22float2(half2_from_u32(v.u32.x));
    hi = __half22float2(half2_from_u32(v.u32.y));
}

__device__ __forceinline__ void load_half4_v2u32_cache_last(
    const half* __restrict__ p, float2& lo, float2& hi) {
    unsigned int a;
    unsigned int b;
    asm volatile("ld.global.L1::evict_last.v2.u32 {%0, %1}, [%2];"
                 : "=r"(a), "=r"(b) : "l"(p));
    lo = __half22float2(half2_from_u32(a));
    hi = __half22float2(half2_from_u32(b));
}

__device__ __forceinline__ void load_half4_v2u32_l2_64(
    const half* __restrict__ p, float2& lo, float2& hi) {
    unsigned int a;
    unsigned int b;
    asm volatile("ld.global.L1::evict_first.L2::64B.v2.u32 {%0, %1}, [%2];"
                 : "=r"(a), "=r"(b) : "l"(p));
    lo = __half22float2(half2_from_u32(a));
    hi = __half22float2(half2_from_u32(b));
}

template <int THREADS>
__device__ __forceinline__ float block_sum_all(float v) {
    __shared__ float partial[THREADS / 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) {
        partial[warp] = v;
    }
    __syncthreads();
    v = threadIdx.x < (THREADS / 32) ? partial[lane] : 0.0f;
    if (warp == 0) {
        v = warp_sum(v);
    }
    if (threadIdx.x == 0) {
        partial[0] = v;
    }
    __syncthreads();
    return partial[0];
}

template <int THREADS>
__device__ __forceinline__ void block_sum2_all(float& a, float& b) {
    __shared__ float partial_a[THREADS / 32];
    __shared__ float partial_b[THREADS / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, offset);
        b += __shfl_down_sync(0xffffffffu, b, offset);
    }
    if (lane == 0) {
        partial_a[warp] = a;
        partial_b[warp] = b;
    }
    __syncthreads();
    a = threadIdx.x < THREADS / 32 ? partial_a[lane] : 0.0f;
    b = threadIdx.x < THREADS / 32 ? partial_b[lane] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xffffffffu, a, offset);
            b += __shfl_down_sync(0xffffffffu, b, offset);
        }
    }
    if (threadIdx.x == 0) {
        partial_a[0] = a;
        partial_b[0] = b;
    }
    __syncthreads();
    a = partial_a[0];
    b = partial_b[0];
}

__global__ void emb_ln0_bf16_to_f16_kernel(
    int V,
    int C,
    const uint16_t* __restrict__ emb,
    const uint16_t* __restrict__ weight,
    const uint16_t* __restrict__ bias,
    half* __restrict__ out,
    float eps) {
    int tok = blockIdx.x;
    if (tok >= V) {
        return;
    }
    const uint16_t* x = emb + static_cast<int64_t>(tok) * C;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        sum += bf16_bits_to_float(x[c]);
    }
    float mean = block_sum_all<256>(sum) / static_cast<float>(C);

    float var = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float d = bf16_bits_to_float(x[c]) - mean;
        var += d * d;
    }
    float rstd = rsqrtf(block_sum_all<256>(var) / static_cast<float>(C) + eps);

    half* y = out + static_cast<int64_t>(tok) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = (bf16_bits_to_float(x[c]) - mean) * rstd * bf16_bits_to_float(weight[c]) + bf16_bits_to_float(bias[c]);
        y[c] = __float2half_rn(v);
    }
}

__global__ __launch_bounds__(1024, 1) void emb_ln_mix6_release_kernel(
    const half* __restrict__ emb,
    const int64_t* __restrict__ tokens,
    half* __restrict__ shift,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ mix_r,
    const half* __restrict__ mix_w,
    const half* __restrict__ mix_k,
    const half* __restrict__ mix_v,
    const half* __restrict__ mix_a,
    const half* __restrict__ mix_g,
    half* __restrict__ x_out,
    half* __restrict__ out_r,
    half* __restrict__ out_w,
    half* __restrict__ out_k,
    half* __restrict__ out_v,
    half* __restrict__ out_a,
    half* __restrict__ out_g,
    int V,
    float eps) {
    // The executor may be admitted immediately, but its PDL wait prevents any
    // mixed-vector read before this producer finishes the corresponding store.
    asm volatile("griddepcontrol.launch_dependents;");
    int64_t token = tokens[0];
    token = token < 0 ? 0 : (token >= V ? V - 1 : token);
    const half2* emb2 = reinterpret_cast<const half2*>(emb) + token * (LN_SMALL_C / 2);
    float sum = 0.0f;
    float sumsq = 0.0f;
#pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int p = threadIdx.x + step * 1024;
        const float2 e = __half22float2(emb2[p]);
        sum += e.x + e.y;
        sumsq = fmaf(e.x, e.x, sumsq);
        sumsq = fmaf(e.y, e.y, sumsq);
    }
    const float mean = block_sum_all<1024>(sum) * (1.0f / LN_SMALL_C);
    const float second = block_sum_all<1024>(sumsq) * (1.0f / LN_SMALL_C);
    // One-pass C4096 stats passed the full eval_src2 condition scan. This is a
    // release-shape result, not a general cancellation guarantee.
    const float rstd = rsqrtf(fmaxf(second - mean * mean, 0.0f) + eps);

    half2* shift2 = reinterpret_cast<half2*>(shift);
    const half2* weight2 = reinterpret_cast<const half2*>(weight);
    const half2* bias2 = reinterpret_cast<const half2*>(bias);
    const half2* mr2 = reinterpret_cast<const half2*>(mix_r);
    const half2* mw2 = reinterpret_cast<const half2*>(mix_w);
    const half2* mk2 = reinterpret_cast<const half2*>(mix_k);
    const half2* mv2 = reinterpret_cast<const half2*>(mix_v);
    const half2* ma2 = reinterpret_cast<const half2*>(mix_a);
    const half2* mg2 = reinterpret_cast<const half2*>(mix_g);
    half2* xo2 = reinterpret_cast<half2*>(x_out);
    half2* r2 = reinterpret_cast<half2*>(out_r);
    half2* w2 = reinterpret_cast<half2*>(out_w);
    half2* k2 = reinterpret_cast<half2*>(out_k);
    half2* v2 = reinterpret_cast<half2*>(out_v);
    half2* a2 = reinterpret_cast<half2*>(out_a);
    half2* g2 = reinterpret_cast<half2*>(out_g);
#pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int p = threadIdx.x + step * 1024;
        const float2 e = __half22float2(emb2[p]);
        const float2 ww = __half22float2(weight2[p]);
        const float2 bb = __half22float2(bias2[p]);
        const half2 y2 = __floats2half2_rn(
            (e.x - mean) * rstd * ww.x + bb.x,
            (e.y - mean) * rstd * ww.y + bb.y);
        const float2 prev = __half22float2(shift2[p]);
        const float2 y = __half22float2(y2);
        const float dx0 = prev.x - y.x;
        const float dx1 = prev.y - y.y;
#define MIX_STORE(dst, mix) do { \
        const float2 m = __half22float2((mix)[p]); \
        (dst)[p] = __floats2half2_rn(y.x + dx0 * m.x, y.y + dx1 * m.y); \
    } while (0)
        xo2[p] = __floats2half2_rn(e.x, e.y);
        MIX_STORE(r2, mr2);
        MIX_STORE(w2, mw2);
        MIX_STORE(k2, mk2);
        MIX_STORE(v2, mv2);
        MIX_STORE(a2, ma2);
        MIX_STORE(g2, mg2);
#undef MIX_STORE
        shift2[p] = y2;
    }
}

__global__ __launch_bounds__(128, 1) void ln_mix6_cluster8_release_kernel(
    const half* __restrict__ x,
    const half* __restrict__ residual,
    half* __restrict__ shift,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ mix_r,
    const half* __restrict__ mix_w,
    const half* __restrict__ mix_k,
    const half* __restrict__ mix_v,
    const half* __restrict__ mix_a,
    const half* __restrict__ mix_g,
    half* __restrict__ x_out,
    half* __restrict__ out_r,
    half* __restrict__ out_w,
    half* __restrict__ out_k,
    half* __restrict__ out_v,
    half* __restrict__ out_a,
    half* __restrict__ out_g,
    float eps) {
    constexpr int PAIRS_PER_RANK = (LN_SMALL_C / 2) / 8;
    __shared__ float cluster_stats[4];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank());
    const int pair_begin = rank * PAIRS_PER_RANK;
    asm volatile("griddepcontrol.launch_dependents;");

    const half2* x2 = reinterpret_cast<const half2*>(x);
    const half2* residual2 = reinterpret_cast<const half2*>(residual);
    float sum = 0.0f;
    float sumsq = 0.0f;
#pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int p = pair_begin + threadIdx.x + step * 128;
        const float2 xv = __half22float2(x2[p]);
        const float2 rv = __half22float2(residual2[p]);
        const float x0 = xv.x + rv.x;
        const float x1 = xv.y + rv.y;
        sum += x0 + x1;
        sumsq = fmaf(x0, x0, sumsq);
        sumsq = fmaf(x1, x1, sumsq);
    }
    block_sum2_all<128>(sum, sumsq);
    if (threadIdx.x == 0) {
        cluster_stats[0] = sum;
        cluster_stats[1] = sumsq;
    }
    cluster.sync();
    if (rank == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        float total_sq = 0.0f;
#pragma unroll
        for (int peer = 0; peer < 8; ++peer) {
            const float* remote = cluster.map_shared_rank(cluster_stats, peer);
            total += remote[0];
            total_sq += remote[1];
        }
        const float mean = total * (1.0f / LN_SMALL_C);
        // Cluster8 changes the reduction tree but not the C4096 audit scope;
        // re-run the real-activation scan before reusing it for another shape.
        const float variance = fmaxf(total_sq * (1.0f / LN_SMALL_C) - mean * mean, 0.0f);
        cluster_stats[2] = mean;
        cluster_stats[3] = rsqrtf(variance + eps);
    }
    cluster.sync();
    const float* stats = rank == 0 ? cluster_stats : cluster.map_shared_rank(cluster_stats, 0);
    const float mean = stats[2];
    const float rstd = stats[3];

    half2* shift2 = reinterpret_cast<half2*>(shift);
    const half2* weight2 = reinterpret_cast<const half2*>(weight);
    const half2* bias2 = reinterpret_cast<const half2*>(bias);
    const half2* mr2 = reinterpret_cast<const half2*>(mix_r);
    const half2* mw2 = reinterpret_cast<const half2*>(mix_w);
    const half2* mk2 = reinterpret_cast<const half2*>(mix_k);
    const half2* mv2 = reinterpret_cast<const half2*>(mix_v);
    const half2* ma2 = reinterpret_cast<const half2*>(mix_a);
    const half2* mg2 = reinterpret_cast<const half2*>(mix_g);
    half2* xo2 = reinterpret_cast<half2*>(x_out);
    half2* r2 = reinterpret_cast<half2*>(out_r);
    half2* w2 = reinterpret_cast<half2*>(out_w);
    half2* k2 = reinterpret_cast<half2*>(out_k);
    half2* v2 = reinterpret_cast<half2*>(out_v);
    half2* a2 = reinterpret_cast<half2*>(out_a);
    half2* g2 = reinterpret_cast<half2*>(out_g);
#pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int p = pair_begin + threadIdx.x + step * 128;
        const float2 xv = __half22float2(x2[p]);
        const float2 rv = __half22float2(residual2[p]);
        const float x0 = xv.x + rv.x;
        const float x1 = xv.y + rv.y;
        const float2 ww = __half22float2(weight2[p]);
        const float2 bb = __half22float2(bias2[p]);
        const half2 y2 = __floats2half2_rn(
            (x0 - mean) * rstd * ww.x + bb.x,
            (x1 - mean) * rstd * ww.y + bb.y);
        const float2 prev = __half22float2(shift2[p]);
        const float2 y = __half22float2(y2);
        const float dx0 = prev.x - y.x;
        const float dx1 = prev.y - y.y;
#define MIX_STORE(dst, mix) do { \
        const float2 m = __half22float2((mix)[p]); \
        (dst)[p] = __floats2half2_rn(y.x + dx0 * m.x, y.y + dx1 * m.y); \
    } while (0)
        xo2[p] = __floats2half2_rn(x0, x1);
        MIX_STORE(r2, mr2);
        MIX_STORE(w2, mw2);
        MIX_STORE(k2, mk2);
        MIX_STORE(v2, mv2);
        MIX_STORE(a2, ma2);
        MIX_STORE(g2, mg2);
#undef MIX_STORE
        shift2[p] = y2;
    }
    // Protect rank0's shared statistics until every remote rank has consumed
    // them. Removing this final sync is a DSM lifetime race.
    cluster.sync();
}

__global__ __launch_bounds__(1024, 1) void ln_cmix_centered_pack_wait_kernel(
    const half* __restrict__ x,
    const half* __restrict__ residual,
    half* __restrict__ shift,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ mix_k,
    half* __restrict__ x_out,
    half* __restrict__ mixed,
    float eps) {
    constexpr int PAIRS = LN_SMALL_C / 2;
    half2 packed[PAIRS / 1024];
    const half2* x2 = reinterpret_cast<const half2*>(x);
    const half2* residual2 = reinterpret_cast<const half2*>(residual);
    asm volatile("griddepcontrol.wait;");
    float sum = 0.0f;
    float sumsq = 0.0f;
#pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int p = threadIdx.x + step * 1024;
        const float2 xv = __half22float2(x2[p]);
        const float2 rv = __half22float2(residual2[p]);
        const float x0 = xv.x + rv.x;
        const float x1 = xv.y + rv.y;
        packed[step] = __floats2half2_rn(x0, x1);
        sum += x0 + x1;
        sumsq = fmaf(x0, x0, sumsq);
        sumsq = fmaf(x1, x1, sumsq);
    }
    const float mean = block_sum_all<1024>(sum) * (1.0f / LN_SMALL_C);
    const float second = block_sum_all<1024>(sumsq) * (1.0f / LN_SMALL_C);
    // Stats use pre-pack fp32 x+residual. The larger P271 packed-affine delta
    // is a separate admitted compute-order change, not one-pass cancellation.
    const float rstd = rsqrtf(fmaxf(second - mean * mean, 0.0f) + eps);
    half2* shift2 = reinterpret_cast<half2*>(shift);
    const half2* weight2 = reinterpret_cast<const half2*>(weight);
    const half2* bias2 = reinterpret_cast<const half2*>(bias);
    const half2* mk2 = reinterpret_cast<const half2*>(mix_k);
    half2* xo2 = reinterpret_cast<half2*>(x_out);
    half2* mixed2 = reinterpret_cast<half2*>(mixed);
#pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int p = threadIdx.x + step * 1024;
        const float2 xr = __half22float2(packed[step]);
        const float2 ww = __half22float2(weight2[p]);
        const float2 bb = __half22float2(bias2[p]);
        const half2 y2 = __floats2half2_rn(
            (xr.x - mean) * rstd * ww.x + bb.x,
            (xr.y - mean) * rstd * ww.y + bb.y);
        const float2 prev = __half22float2(shift2[p]);
        const float2 y = __half22float2(y2);
        const float2 mk = __half22float2(mk2[p]);
        xo2[p] = packed[step];
        mixed2[p] = __floats2half2_rn(
            y.x + (prev.x - y.x) * mk.x,
            y.y + (prev.y - y.y) * mk.y);
        shift2[p] = y2;
    }
}

template <int THREADS>
__device__ __forceinline__ float row_dot_half2(
    const half* __restrict__ x,
    const half* __restrict__ w,
    int row,
    int C) {
    const half2* x2 = reinterpret_cast<const half2*>(x);
    const half2* w2 = reinterpret_cast<const half2*>(w + static_cast<int64_t>(row) * C);
    int C2 = C >> 1;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < C2; i += THREADS) {
        float2 xv = __half22float2(x2[i]);
        float2 wv = __half22float2(w2[i]);
        sum = fmaf(xv.x, wv.x, sum);
        sum = fmaf(xv.y, wv.y, sum);
    }
    if ((C & 1) && threadIdx.x == 0) {
        int i = C - 1;
        sum = fmaf(__half2float(x[i]), __half2float(w[static_cast<int64_t>(row) * C + i]), sum);
    }
    return block_sum_all<THREADS>(sum);
}


template <int THREADS>
__device__ __forceinline__ void rkv_executor_tile_body_h2stage_hfma2_splitacc_k2pipe(
    const half* __restrict__ xr,
    const half* __restrict__ xk,
    const half* __restrict__ xv,
    const half* __restrict__ wr,
    const half* __restrict__ wk,
    const half* __restrict__ wv,
    half* __restrict__ yr,
    half* __restrict__ yk,
    half* __restrict__ yv,
    int task,
    int C) {
    constexpr int RKV_OUT_TILE = 2;
    __shared__ float partial[THREADS / 32][RKV_OUT_TILE];
    int rows_per_group = C / RKV_OUT_TILE;
    int group = task / rows_per_group;
    int row0 = (task - group * rows_per_group) * RKV_OUT_TILE;
    const half* inp = group == 0 ? xr : (group == 1 ? xk : xv);
    const half* wt = group == 0 ? wr : (group == 1 ? wk : wv);
    half* out = group == 0 ? yr : (group == 1 ? yk : yv);

    half2 acc00h;
    half2 acc01h;
    half2 acc10h;
    half2 acc11h;
    *reinterpret_cast<int*>(&acc00h) = 0;
    *reinterpret_cast<int*>(&acc01h) = 0;
    *reinterpret_cast<int*>(&acc10h) = 0;
    *reinterpret_cast<int*>(&acc11h) = 0;
    for (int k0 = threadIdx.x << 1; k0 < C; k0 += THREADS << 2) {
        const int k1 = k0 + (THREADS << 1);
        const half* w00 = wt + static_cast<int64_t>(row0) * C + k0;
        const half* w10 = w00 + C;
        half2 hx0 = *reinterpret_cast<const half2*>(inp + k0);
        half2 hw00 = *reinterpret_cast<const half2*>(w00);
        half2 hw10 = *reinterpret_cast<const half2*>(w10);
        if (k1 < C) {
            const half* w01 = wt + static_cast<int64_t>(row0) * C + k1;
            const half* w11 = w01 + C;
            half2 hx1 = *reinterpret_cast<const half2*>(inp + k1);
            half2 hw01 = *reinterpret_cast<const half2*>(w01);
            half2 hw11 = *reinterpret_cast<const half2*>(w11);
            acc00h = __hfma2(hx0, hw00, acc00h);
            acc01h = __hfma2(hx1, hw01, acc01h);
            acc10h = __hfma2(hx0, hw10, acc10h);
            acc11h = __hfma2(hx1, hw11, acc11h);
        } else {
            acc00h = __hfma2(hx0, hw00, acc00h);
            acc10h = __hfma2(hx0, hw10, acc10h);
        }
    }

    float2 acc0f = __half22float2(__hadd2(acc00h, acc01h));
    float2 acc1f = __half22float2(__hadd2(acc10h, acc11h));
    float acc0 = acc0f.x + acc0f.y;
    float acc1 = acc1f.x + acc1f.y;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float v0 = warp_sum(acc0);
    float v1 = warp_sum(acc1);
    if (lane == 0) {
        partial[warp][0] = v0;
        partial[warp][1] = v1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
#pragma unroll
        for (int widx = 0; widx < THREADS / 32; ++widx) {
            sum0 += partial[widx][0];
            sum1 += partial[widx][1];
        }
        out[row0] = __float2half_rn(sum0);
        out[row0 + 1] = __float2half_rn(sum1);
    }
}


template <int THREADS>
__device__ __forceinline__ void rkv_executor_tile_body_h2stage_hfma2_splitacc_k2pipe_group_interleave(
    const half* __restrict__ xr,
    const half* __restrict__ xk,
    const half* __restrict__ xv,
    const half* __restrict__ wr,
    const half* __restrict__ wk,
    const half* __restrict__ wv,
    half* __restrict__ yr,
    half* __restrict__ yk,
    half* __restrict__ yv,
    int task,
    int C) {
    constexpr int RKV_OUT_TILE = 2;
    int rows_per_group = C / RKV_OUT_TILE;
    int row_pair = task / 3;
    int group = task - row_pair * 3;
    int mapped_task = group * rows_per_group + row_pair;
    rkv_executor_tile_body_h2stage_hfma2_splitacc_k2pipe<THREADS>(xr, xk, xv, wr, wk, wv, yr, yk, yv, mapped_task, C);
}

template <int THREADS>
__device__ __forceinline__ void lowrank_pre_compact_body(
    const half* __restrict__ xw,
    const half* __restrict__ xa,
    const half* __restrict__ xg,
    const half* __restrict__ xlr_v,
    const half* __restrict__ w1_t,
    const half* __restrict__ a1_t,
    const half* __restrict__ g1_t,
    const half* __restrict__ v1_t,
    half* __restrict__ w1,
    half* __restrict__ a1,
    half* __restrict__ g1,
    half* __restrict__ v1,
    int task,
    int M,
    int C,
    int Rw,
    int Ra,
    int Rg,
    int Rv) {
    int n0 = M * Rw;
    int n1 = n0 + M * Ra;
    int n2 = n1 + M * Rg;
    const half* x = xw;
    const half* wt = w1_t;
    half* y = w1;
    int R = Rw;
    int base = 0;
    if (task >= n2) {
        x = xlr_v;
        wt = v1_t;
        y = v1;
        R = Rv;
        base = n2;
    } else if (task >= n1) {
        x = xg;
        wt = g1_t;
        y = g1;
        R = Rg;
        base = n1;
    } else if (task >= n0) {
        x = xa;
        wt = a1_t;
        y = a1;
        R = Ra;
        base = n0;
    }
    int local = task - base;
    int m = local / R;
    int r = local - m * R;
    if (m < M) {
        float sum = row_dot_half2<THREADS>(x + static_cast<int64_t>(m) * C, wt, r, C);
        if (threadIdx.x == 0) {
            y[static_cast<int64_t>(m) * R + r] = __float2half_rn(sum);
        }
    }
}

template <int THREADS, int RKV_OUT_TILE, bool FORCE_TASK_SYNC, bool SKIP_V, int MIN_BLOCKS = 1, int ROLE_ORDER = -1>
__global__ __launch_bounds__(THREADS, MIN_BLOCKS) void rkv_lowrank_pre_executor_kernel(
    const half* __restrict__ xr,
    const half* __restrict__ xk,
    const half* __restrict__ xv,
    const half* __restrict__ wr,
    const half* __restrict__ wk,
    const half* __restrict__ wv,
    half* __restrict__ yr,
    half* __restrict__ yk,
    half* __restrict__ yv,
    const half* __restrict__ xw,
    const half* __restrict__ xa,
    const half* __restrict__ xg,
    const half* __restrict__ xlr_v,
    const half* __restrict__ w1_t,
    const half* __restrict__ a1_t,
    const half* __restrict__ g1_t,
    const half* __restrict__ v1_t,
    const half* __restrict__ w2_t,
    const half* __restrict__ g2_t,
    half* __restrict__ w1,
    half* __restrict__ a1,
    half* __restrict__ g1,
    half* __restrict__ v1,
    half* __restrict__ w,
    half* __restrict__ g,
    int M,
    int C,
    int Rw,
    int Ra,
    int Rg,
    int Rv,
    int lowrank_worker_budget,
    int role_order) {
    // LN1 published the dependency before writing mixed vectors. Waiting here
    // is mandatory; the immediate release admits rankout while this grid runs.
    asm volatile("griddepcontrol.wait;");
    asm volatile("griddepcontrol.launch_dependents;");
    const int rkv_tasks = 3 * (C / RKV_OUT_TILE);
    const int lowrank_tasks = M * (Rw + Ra + Rg + (SKIP_V ? 0 : Rv));
    int lowrank_workers = lowrank_worker_budget > 0 ? lowrank_worker_budget : static_cast<int>(gridDim.x) / 4;
    if (lowrank_workers < 0) {
        lowrank_workers = 0;
    }
    if (lowrank_workers > lowrank_tasks) {
        lowrank_workers = lowrank_tasks;
    }
    if (lowrank_workers >= static_cast<int>(gridDim.x)) {
        lowrank_workers = static_cast<int>(gridDim.x) - 1;
    }
    const int bid = static_cast<int>(blockIdx.x);
    const int effective_role_order = ROLE_ORDER >= 0 ? ROLE_ORDER : role_order;
    const bool lowrank_first = effective_role_order == 1;
    const bool lowrank_interleave = effective_role_order == 2;
    const int interleave_stride = 8;
    if (lowrank_interleave) {
        const int interleave_slots = (static_cast<int>(gridDim.x) + interleave_stride - 1) / interleave_stride;
        if (lowrank_workers > interleave_slots) {
            lowrank_workers = interleave_slots;
        }
    }
    int rkv_workers = static_cast<int>(gridDim.x) - lowrank_workers;
    const int interleave_span = lowrank_interleave ? lowrank_workers * interleave_stride : 0;
    const bool interleave_lowrank = lowrank_interleave && bid < interleave_span && (bid % interleave_stride) == 0;
    const int interleave_lowrank_before = bid < interleave_span
        ? (bid + interleave_stride - 1) / interleave_stride
        : lowrank_workers;
    const bool lowrank_worker = lowrank_interleave
        ? interleave_lowrank
        : (lowrank_first ? bid < lowrank_workers : bid >= rkv_workers);
    const int worker_rank = lowrank_worker
        ? (lowrank_interleave ? bid / interleave_stride : (lowrank_first ? bid : bid - rkv_workers))
        : (lowrank_interleave ? bid - interleave_lowrank_before : (lowrank_first ? bid - lowrank_workers : bid));
    const int worker_count = lowrank_worker ? lowrank_workers : rkv_workers;
    const bool needs_task_sync = FORCE_TASK_SYNC || worker_count < (lowrank_worker ? lowrank_tasks : rkv_tasks);
    if (lowrank_worker) {
        for (int task = worker_rank; task < lowrank_tasks; task += worker_count) {
            lowrank_pre_compact_body<THREADS>(
                xw, xa, xg, xlr_v, w1_t, a1_t, g1_t, v1_t, w1, a1, g1, v1,
                task, M, C, Rw, Ra, Rg, Rv);
            if (needs_task_sync) {
                __syncthreads();
            }
        }
    } else {
        for (int task = worker_rank; task < rkv_tasks; task += worker_count) {
            rkv_executor_tile_body_h2stage_hfma2_splitacc_k2pipe_group_interleave<THREADS>(
                xr, xk, xv, wr, wk, wv, yr, yk, yv, task, C);
            if (needs_task_sync) {
                __syncthreads();
            }
        }
    }
}



__device__ __forceinline__ void rankout_w8_fixed(
    const half* __restrict__ w1,
    const half* __restrict__ w2_t,
    const half* __restrict__ w0,
    float* __restrict__ out,
    int task,
    bool fp32_w_output) {
    float early[8];
    const int r = threadIdx.x;
    const int n0 = task * 8;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        early[j] = __half2float(w2_t[static_cast<int64_t>(n0 + j) * 128 + r]);
    }
    const float x = tanhf(__half2float(w1[r]));
#pragma unroll
    for (int j = 2; j < 8; ++j) {
        early[j] = __half2float(w2_t[static_cast<int64_t>(n0 + j) * 128 + r]);
    }
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j] = x * early[j];
    }
    warp_sum_oct(acc);
    __shared__ float partial[4][8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            partial[warp][j] = acc[j];
        }
    }
    __syncthreads();
    if (threadIdx.x < 8) {
        const int j = threadIdx.x;
        const float sum = ((partial[0][j] + partial[1][j]) + partial[2][j]) + partial[3][j];
        // Preserve the FP32IO16 low-rank operator's fp16 output contract. This
        // rounding point is not required by decay math; effective W stays fp32.
        const half raw = __float2half_rn(sum);
        const float effective = rankout_w_eff(__half2float(raw) + __half2float(w0[n0 + j]));
        // Keep this as a runtime launch argument even though release passes true.
        // Folding it changed the shared A/W/G/V kernel's codegen and lost B1T1 E2E.
        if (fp32_w_output) {
            out[n0 + j] = effective;
        } else {
            reinterpret_cast<half*>(out)[n0 + j] = __float2half_rn(effective);
        }
    }
}

__device__ __forceinline__ void rankout_g4_fixed(
    const half* __restrict__ g1,
    const half* __restrict__ g2_t,
    half* __restrict__ out,
    int task,
    int rank) {
    const int n0 = task * 4;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    // P485 keeps Rg as a runtime trip count while the row stride is fixed by
    // the validated G480 checkpoint. Folding both to constants changes the
    // whole multi-role kernel's register allocation and real PDL schedule.
    for (int r = threadIdx.x; r < rank; r += 128) {
        const float x = sigmoid_fast(__half2float(g1[r]));
        a0 = fmaf(x, __half2float(g2_t[static_cast<int64_t>(n0) * 480 + r]), a0);
        a1 = fmaf(x, __half2float(g2_t[static_cast<int64_t>(n0 + 1) * 480 + r]), a1);
        a2 = fmaf(x, __half2float(g2_t[static_cast<int64_t>(n0 + 2) * 480 + r]), a2);
        a3 = fmaf(x, __half2float(g2_t[static_cast<int64_t>(n0 + 3) * 480 + r]), a3);
    }
    warp_sum_quad(a0, a1, a2, a3);
    __shared__ float partial[4][4];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        partial[warp][0] = a0;
        partial[warp][1] = a1;
        partial[warp][2] = a2;
        partial[warp][3] = a3;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[n0 + j] = __float2half_rn(
                ((partial[0][j] + partial[1][j]) + partial[2][j]) + partial[3][j]);
        }
    }
}

__device__ __forceinline__ void rankout_v16_tree12(
    const half* __restrict__ v1,
    const half* __restrict__ v2_t,
    const half* __restrict__ v,
    const half* __restrict__ v_first,
    const half* __restrict__ v0,
    half* __restrict__ out,
    int task) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int subgroup = lane >> 3;
    const int sublane = lane & 7;
    const int n = task * 16 + warp * 4 + subgroup;
    const int64_t row = static_cast<int64_t>(n) * 96;
#define VP(offset) (__half2float(v1[sublane + (offset)]) * __half2float(v2_t[row + sublane + (offset)]))
    const float s01 = VP(0) + VP(8);
    const float s23 = VP(16) + VP(24);
    const float s45 = VP(32) + VP(40);
    const float s67 = VP(48) + VP(56);
    const float s89 = VP(64) + VP(72);
    const float s1011 = VP(80) + VP(88);
#undef VP
    float acc = ((s01 + s23) + (s45 + s67)) + (s89 + s1011);
    const unsigned mask = 0xffu << (subgroup * 8);
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(mask, acc, offset, 8);
    }
    if (sublane == 0) {
        const float vv = __half2float(v[n]);
        const float vf = __half2float(v_first[n]);
        const float gate = sigmoid_fast(__half2float(v0[n]) + acc);
        out[n] = __float2half_rn(fmaf(vf - vv, gate, vv));
    }
}

template <bool SKIP_V>
__global__ __launch_bounds__(128, 2) void rankout_fixed_kernel(
    const half* __restrict__ w1,
    const half* __restrict__ a1,
    const half* __restrict__ g1,
    const half* __restrict__ v1,
    const half* __restrict__ w2_t,
    const half* __restrict__ a2_t,
    const half* __restrict__ g2_t,
    const half* __restrict__ v2_t,
    const half* __restrict__ v,
    const half* __restrict__ v_first,
    const half* __restrict__ v0,
    const half* __restrict__ k_raw,
    const half* __restrict__ k_k,
    const half* __restrict__ a0,
    const half* __restrict__ k_a,
    const half* __restrict__ w0,
    float* __restrict__ w,
    half* __restrict__ a,
    half* __restrict__ g,
    half* __restrict__ v_out,
    half* __restrict__ new_k,
    half* __restrict__ neg_kk,
    half* __restrict__ kka,
    int g_rank,
    bool fp32_w_output) {
    constexpr int A_TASKS = 4096 / 16;
    constexpr int W_TASKS = 4096 / 8;
    constexpr int G_TASKS = 4096 / 4;
    const int task = static_cast<int>(blockIdx.x);
    asm volatile("griddepcontrol.wait;");
    if (task >= A_TASKS) {
        int rem = task - A_TASKS;
        if (rem < W_TASKS) {
            rankout_w8_fixed(w1, w2_t, w0, w, rem, fp32_w_output);
        } else if ((rem -= W_TASKS) < G_TASKS) {
            rankout_g4_fixed(g1, g2_t, g, rem, g_rank);
        } else if constexpr (!SKIP_V) {
            rankout_v16_tree12(v1, v2_t, v, v_first, v0, v_out, rem - G_TASKS);
        }
        // All role CTAs own disjoint channels. Release after their final store;
        // WKV waits before reading any rankout result.
        asm volatile("griddepcontrol.launch_dependents;");
        return;
    }

    const int group_base = task * 16;
    const int head_base = (group_base / HEAD_SIZE) * HEAD_SIZE;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float a_total[16];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int r = lane + step * 32;
        const float x = __half2float(a1[r]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = group_base + warp * 4 + j;
            acc[j] = fmaf(x, __half2float(a2_t[static_cast<int64_t>(n) * 128 + r]), acc[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float sum = warp_sum(acc[j]);
        if (lane == 0) {
            const int n = group_base + warp * 4 + j;
            const half ah = __float2half_rn(sum);
            a[n] = ah;
            a_total[n - group_base] = __half2float(ah);
        }
    }
    // Publish all four warps' A stores before warp0 consumes a_total.
    __syncthreads();
    if (warp == 0) {
        const int n0 = head_base + lane;
        const int n1 = n0 + 32;
        const float u0 = __half2float(k_raw[n0]) * __half2float(k_k[n0]);
        const float u1 = __half2float(k_raw[n1]) * __half2float(k_k[n1]);
        const float norm = warp_sum_all_xor(fmaf(u0, u0, u1 * u1));
        const float inv_norm = 1.0f / fmaxf(sqrtf(norm), KK_NORMALIZE_EPS);
        if (lane < 16) {
            const int n = group_base + lane;
            const float kr = __half2float(k_raw[n]);
            const float kk = kr * __half2float(k_k[n]) * inv_norm;
            const float gate = sigmoid_fast(__half2float(a0[n]) + a_total[lane]);
            const float ka = __half2float(k_a[n]);
            new_k[n] = __float2half_rn(kr * fmaf(gate, ka, 1.0f - ka));
            neg_kk[n] = __float2half_rn(-kk);
            kka[n] = __float2half_rn(kk * gate);
        }
    }
    // Uniform late release must remain below every A/KK store.
    asm volatile("griddepcontrol.launch_dependents;");
}

__global__ __launch_bounds__(128, 2) void wkv_halfwarp8_wait_release_kernel(
    float* __restrict__ state,
    const half* __restrict__ r,
    const float* __restrict__ w,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ y) {
    asm volatile("griddepcontrol.wait;");
    asm volatile("griddepcontrol.launch_dependents;");
    __shared__ float sh_r[64];
    __shared__ float sh_w[64];
    __shared__ float sh_k[64];
    __shared__ float sh_a[64];
    __shared__ float sh_b[64];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int sublane = lane & 15;
    const int warp = tid >> 5;
    const int half = lane >> 4;
    const int row = blockIdx.x * 8 + warp * 2 + half;
    const int head = blockIdx.y;
    const int token = head * 64;
    const int state_base = (head * 64 + row) * 64;
    const int j0 = sublane;
    const int j1 = sublane + 16;
    const int j2 = sublane + 32;
    const int j3 = sublane + 48;
    if (tid < 64) {
        const int idx = token + tid;
        sh_r[tid] = __half2float(r[idx]);
        sh_w[tid] = w[idx];
        sh_k[tid] = __half2float(k[idx]);
        sh_a[tid] = __half2float(a[idx]);
        sh_b[tid] = __half2float(b[idx]);
    }
    __syncthreads();
    const float s0old = state[state_base + j0];
    const float s1old = state[state_base + j1];
    const float s2old = state[state_base + j2];
    const float s3old = state[state_base + j3];
    const float vv = __half2float(v[token + row]);
    float sa = s0old * sh_a[j0];
    sa += s1old * sh_a[j1];
    sa += s2old * sh_a[j2];
    sa += s3old * sh_a[j3];
    const float base0 = s0old * sh_w[j0] + vv * sh_k[j0];
    const float base1 = s1old * sh_w[j1] + vv * sh_k[j1];
    const float base2 = s2old * sh_w[j2] + vv * sh_k[j2];
    const float base3 = s3old * sh_w[j3] + vv * sh_k[j3];
    sa = halfwarp_sum_all_xor(sa);
    const float s0 = base0 + sa * sh_b[j0];
    const float s1 = base1 + sa * sh_b[j1];
    const float s2 = base2 + sa * sh_b[j2];
    const float s3 = base3 + sa * sh_b[j3];
    state[state_base + j0] = s0;
    state[state_base + j1] = s1;
    state[state_base + j2] = s2;
    state[state_base + j3] = s3;
    float yy = s0 * sh_r[j0];
    yy += s1 * sh_r[j1];
    yy += s2 * sh_r[j2];
    yy += s3 * sh_r[j3];
    yy = halfwarp_sum_all_xor(yy);
    if (sublane == 0) {
        y[token + row] = __float2half_rn(yy);
    }
}

__global__ __launch_bounds__(64, 1) void lnx_onepass_wait_release_kernel(
    const half* __restrict__ x,
    const half* __restrict__ r,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ r_k,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ g,
    half* __restrict__ out) {
    __shared__ float partial[2];
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const int warp = lane >> 5;
    const int warp_lane = lane & 31;
    const int idx = head * 64 + lane;
    asm volatile("griddepcontrol.wait;");
    asm volatile("griddepcontrol.launch_dependents;");
    const float xv = __half2float(x[idx]);
    const float sumsq = warp_sum(xv * xv);
    const float sum = warp_sum(xv);
    if (warp_lane == 0) partial[warp] = sum;
    __syncthreads();
    const float mean = (partial[0] + partial[1]) * (1.0f / 64.0f);
    __syncthreads();
    if (warp_lane == 0) partial[warp] = sumsq;
    __syncthreads();
    // One-pass stats are release-scoped to H64. The full eval_src2 audit found
    // max rstd relative error 4.91e-7; re-audit before changing shape/model.
    const float variance = fmaxf((partial[0] + partial[1]) * (1.0f / 64.0f) - mean * mean, 0.0f);
    const float rstd = rsqrtf(variance + TMIX_LN_X_EPS);
    __syncthreads();
    const float rv = __half2float(r[idx]);
    const float kv = __half2float(k[idx]);
    const float vv = __half2float(v[idx]);
    const float dot = warp_sum(rv * kv * __half2float(r_k[idx]));
    if (warp_lane == 0) partial[warp] = dot;
    __syncthreads();
    const float rkv = partial[0] + partial[1];
    const float yn = (xv - mean) * rstd * __half2float(weight[idx]) + __half2float(bias[idx]);
    out[idx] = __float2half_rn((yn + rkv * vv) * __half2float(g[idx]));
}

__global__ __launch_bounds__(64, 1) void att_out_pair_wait_release_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out) {
    constexpr int K = 4096;
    const half* wr = weight + static_cast<int64_t>(blockIdx.x) * K;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    asm volatile("griddepcontrol.wait;");
    asm volatile("griddepcontrol.launch_dependents;");
    for (int k0 = threadIdx.x * 4; k0 < K; k0 += 1024) {
        const int k1 = k0 + 256;
        const int k2 = k1 + 256;
        const int k3 = k2 + 256;
        float2 x00, x01, x10, x11, w00, w01, w10, w11;
        load_half4_u64_cache(x + k0, true, x00, x01);
        load_half4_u64_cache(x + k1, true, x10, x11);
        load_half4_u64_cache(wr + k0, false, w00, w01);
        load_half4_u64_cache(wr + k1, false, w10, w11);
        a0 = fmaf(x00.x, w00.x, a0); a1 = fmaf(x10.x, w10.x, a1);
        a0 = fmaf(x00.y, w00.y, a0); a1 = fmaf(x10.y, w10.y, a1);
        a0 = fmaf(x01.x, w01.x, a0); a1 = fmaf(x11.x, w11.x, a1);
        a0 = fmaf(x01.y, w01.y, a0); a1 = fmaf(x11.y, w11.y, a1);
        float2 x20, x21, x30, x31, w20, w21, w30, w31;
        load_half4_u64_cache(x + k2, true, x20, x21);
        load_half4_u64_cache(x + k3, true, x30, x31);
        load_half4_u64_cache(wr + k2, false, w20, w21);
        load_half4_u64_cache(wr + k3, false, w30, w31);
        a2 = fmaf(x20.x, w20.x, a2); a3 = fmaf(x30.x, w30.x, a3);
        a2 = fmaf(x20.y, w20.y, a2); a3 = fmaf(x30.y, w30.y, a3);
        a2 = fmaf(x21.x, w21.x, a2); a3 = fmaf(x31.x, w31.x, a3);
        a2 = fmaf(x21.y, w21.y, a2); a3 = fmaf(x31.y, w31.y, a3);
    }
    const float sum = warp_sum((a0 + a1) + (a2 + a3));
    __shared__ float partial[2];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = __float2half_rn(partial[0] + partial[1]);
}

__global__ __launch_bounds__(128, 1) void cmix_key_outphase_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out) {
    constexpr int K = 4096;
    const int n0 = blockIdx.x * 2;
    const half* w0 = weight + static_cast<int64_t>(n0) * K;
    const half* w1 = w0 + K;
    float a00 = 0.0f, a01 = 0.0f, a10 = 0.0f, a11 = 0.0f;
    for (int k0 = threadIdx.x * 4; k0 < K; k0 += 1024) {
        const int k1 = k0 + 512;
        float2 x00, x01, x10, x11;
        load_half4_float2_u64(x + k0, x00, x01);
        load_half4_float2_u64(x + k1, x10, x11);
        float2 w00, w01, w10, w11;
        load_half4_float2_u64(w0 + k0, w00, w01);
        load_half4_float2_u64(w1 + k0, w10, w11);
        a00 = fmaf(x00.x, w00.x, a00); a10 = fmaf(x00.x, w10.x, a10);
        a00 = fmaf(x00.y, w00.y, a00); a10 = fmaf(x00.y, w10.y, a10);
        a00 = fmaf(x01.x, w01.x, a00); a10 = fmaf(x01.x, w11.x, a10);
        a00 = fmaf(x01.y, w01.y, a00); a10 = fmaf(x01.y, w11.y, a10);
        load_half4_float2_u64(w0 + k1, w00, w01);
        load_half4_float2_u64(w1 + k1, w10, w11);
        a01 = fmaf(x10.x, w00.x, a01); a11 = fmaf(x10.x, w10.x, a11);
        a01 = fmaf(x10.y, w00.y, a01); a11 = fmaf(x10.y, w10.y, a11);
        a01 = fmaf(x11.x, w01.x, a01); a11 = fmaf(x11.x, w11.x, a11);
        a01 = fmaf(x11.y, w01.y, a01); a11 = fmaf(x11.y, w11.y, a11);
    }
    float out0 = warp_sum(a00 + a01);
    float out1 = warp_sum(a10 + a11);
    __shared__ float partial[4][2];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        partial[warp][0] = out0;
        partial[warp][1] = out1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            float sum = 0.0f;
#pragma unroll
            for (int wi = 0; wi < 4; ++wi) sum += partial[wi][j];
            const half raw = __float2half_rn(sum);
            const float activated = fmaxf(__half2float(raw), 0.0f);
            out[n0 + j] = __float2half_rn(activated * activated);
        }
    }
}

__global__ __launch_bounds__(128, 4) void cmix_value_split4_kernel(
    const half* __restrict__ act,
    const half* __restrict__ weight,
    half* __restrict__ out) {
    __shared__ __align__(256) half values[128];
    __shared__ __align__(256) int ids[128];
    __shared__ int count;
    __shared__ int warp_counts[4];
    __shared__ int warp_prefix[4];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int f_block = blockIdx.x;
    const int c_block = blockIdx.y;
    const int start_f = f_block * 128;
    const half value = act[start_f + tid];
    values[tid] = value;
    const bool nonzero = bool(__half_as_ushort(value) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    const int local = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) warp_counts[warp] = __popc(mask);
    __syncthreads();
    if (tid == 0) {
        int total = 0;
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            warp_prefix[wi] = total;
            total += warp_counts[wi];
        }
        count = total;
    }
    __syncthreads();
    if (nonzero) ids[warp_prefix[warp] + local] = tid;
    __syncthreads();
    half2 a0 = __float2half2_rn(0.0f);
    half2 a1 = __float2half2_rn(0.0f);
    half2 a2 = __float2half2_rn(0.0f);
    half2 a3 = __float2half2_rn(0.0f);
    constexpr int C_TILE = 256;
    const int c0 = c_block * C_TILE + tid * 2;
    const int tile_base = ((f_block * 16 + c_block) * 128) * C_TILE;
    for (int i = 0; i < count; i += 4) {
        const int f0 = ids[i];
        const half2 m0 = *reinterpret_cast<const half2*>(weight + static_cast<int64_t>(tile_base) + f0 * C_TILE + tid * 2);
        a0 = __hfma2(__half2half2(values[f0]), m0, a0);
        if (i + 1 < count) {
            const int f1 = ids[i + 1];
            const half2 m1 = *reinterpret_cast<const half2*>(weight + static_cast<int64_t>(tile_base) + f1 * C_TILE + tid * 2);
            a1 = __hfma2(__half2half2(values[f1]), m1, a1);
        }
        if (i + 2 < count) {
            const int f2 = ids[i + 2];
            const half2 m2 = *reinterpret_cast<const half2*>(weight + static_cast<int64_t>(tile_base) + f2 * C_TILE + tid * 2);
            a2 = __hfma2(__half2half2(values[f2]), m2, a2);
        }
        if (i + 3 < count) {
            const int f3 = ids[i + 3];
            const half2 m3 = *reinterpret_cast<const half2*>(weight + static_cast<int64_t>(tile_base) + f3 * C_TILE + tid * 2);
            a3 = __hfma2(__half2half2(values[f3]), m3, a3);
        }
    }
    atomicAdd(reinterpret_cast<half2*>(out + c0), __hadd2(__hadd2(a0, a1), __hadd2(a2, a3)));
}

template <int THREADS, bool PDL_RELEASE = false, bool ONEPASS_STATS = false>
__global__ __launch_bounds__(THREADS, 1) void final_ln_release_kernel(
    const half* __restrict__ x,
    const half* __restrict__ residual,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int B,
    int T,
    int C,
    float eps) {
    const int b = blockIdx.x;
    if (b >= B) {
        return;
    }
    if constexpr (PDL_RELEASE) {
        asm volatile("griddepcontrol.launch_dependents;");
    }
    // B/T/C intentionally remain runtime launch arguments. Specializing the
    // C4096 indexing changed final-LN/HEAD overlap and regressed the full graph.
    const int64_t src = (static_cast<int64_t>(b) * T + (T - 1)) * C;
    const int64_t dst = static_cast<int64_t>(b) * C;
    float mean;
    float rstd;
    if constexpr (ONEPASS_STATS) {
        const half2* x2 = reinterpret_cast<const half2*>(x + src);
        const half2* residual2 = reinterpret_cast<const half2*>(residual + src);
        float sum = 0.0f;
        float sum2 = 0.0f;
        for (int p = threadIdx.x; p < (C >> 1); p += THREADS) {
            const float2 xv = __half22float2(x2[p]);
            const float2 rv = __half22float2(residual2[p]);
            const float v0 = xv.x + rv.x;
            const float v1 = xv.y + rv.y;
            sum += v0 + v1;
            sum2 = fmaf(v0, v0, sum2);
            sum2 = fmaf(v1, v1, sum2);
        }
        mean = block_sum_all<THREADS>(sum) / static_cast<float>(C);
        const float second = block_sum_all<THREADS>(sum2) / static_cast<float>(C);
        // One-pass stats are release-scoped to C4096. See the 2026-07-13
        // numeric precision audit before generalizing this cancellation result.
        rstd = rsqrtf(fmaxf(second - mean * mean, 0.0f) + eps);
    } else {
        float sum = 0.0f;
        for (int c = threadIdx.x; c < C; c += THREADS) {
            sum += __half2float(x[src + c]) + __half2float(residual[src + c]);
        }
        mean = block_sum_all<THREADS>(sum) / static_cast<float>(C);
        float var = 0.0f;
        for (int c = threadIdx.x; c < C; c += THREADS) {
            const float v = __half2float(x[src + c]) + __half2float(residual[src + c]);
            const float d = v - mean;
            var += d * d;
        }
        rstd = rsqrtf(block_sum_all<THREADS>(var) / static_cast<float>(C) + eps);
    }
    for (int c = threadIdx.x; c < C; c += THREADS) {
        const float v = __half2float(x[src + c]) + __half2float(residual[src + c]);
        out[dst + c] = __float2half_rn(
            (v - mean) * rstd * __half2float(weight[c]) + __half2float(bias[c]));
    }
}

__global__ __launch_bounds__(256, 1) void head_mode26_wait_kernel(
    int K,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out) {
    const int n0 = blockIdx.x * 2;
    float acc0[2] = {0.0f, 0.0f};
    float acc1[2] = {0.0f, 0.0f};
    asm volatile("griddepcontrol.wait;");
    // Keep K in the runtime ABI. The fixed-K rewrite was mathematically equal
    // but did not preserve the validated mode26 codegen/PDL schedule.
#pragma unroll
    for (int round = 0; round < 2; ++round) {
#pragma unroll
        for (int stream = 0; stream < 2; ++stream) {
            const int k = threadIdx.x * 4 + (round * 2 + stream) * 1024;
            float2 x0, x1, w00, w01, w10, w11;
            load_half4_v2u32_cache_last(x + k, x0, x1);
            const half* wr0 = weight + static_cast<int64_t>(n0) * K + k;
            const half* wr1 = wr0 + K;
            load_half4_v2u32_l2_64(wr0, w00, w01);
            acc0[stream] = fmaf(x0.x, w00.x, acc0[stream]);
            acc0[stream] = fmaf(x0.y, w00.y, acc0[stream]);
            acc0[stream] = fmaf(x1.x, w01.x, acc0[stream]);
            acc0[stream] = fmaf(x1.y, w01.y, acc0[stream]);
            load_half4_v2u32_l2_64(wr1, w10, w11);
            acc1[stream] = fmaf(x0.x, w10.x, acc1[stream]);
            acc1[stream] = fmaf(x0.y, w10.y, acc1[stream]);
            acc1[stream] = fmaf(x1.x, w11.x, acc1[stream]);
            acc1[stream] = fmaf(x1.y, w11.y, acc1[stream]);
        }
    }
    float out0 = acc0[0] + acc0[1];
    float out1 = acc1[0] + acc1[1];
    // Preserve mode26's paired shuffle issue schedule, not only its sum tree.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other0 = __shfl_down_sync(0xffffffffu, out0, offset);
        const float other1 = __shfl_down_sync(0xffffffffu, out1, offset);
        out0 += other0;
        out1 += other1;
    }
    __shared__ float partial[8][2];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        partial[warp][0] = out0;
        partial[warp][1] = out1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float s00 = (partial[0][0] + partial[1][0]) + (partial[2][0] + partial[3][0]);
        const float s01 = (partial[4][0] + partial[5][0]) + (partial[6][0] + partial[7][0]);
        const float s10 = (partial[0][1] + partial[1][1]) + (partial[2][1] + partial[3][1]);
        const float s11 = (partial[4][1] + partial[5][1]) + (partial[6][1] + partial[7][1]);
        out[n0] = __float2half_rn(s00 + s01);
        out[n0 + 1] = __float2half_rn(s10 + s11);
    }
}

} // namespace

using Tensor = torch::Tensor;

namespace {

void check_half_cuda(const Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kFloat16 && tensor.is_contiguous(),
                name, " must be contiguous CUDA fp16");
}

cudaLaunchConfig_t pdl_config(dim3 grid, dim3 block, cudaStream_t stream, cudaLaunchAttribute& attr) {
    attr = {};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    return config;
}

} // namespace

Tensor emb_ln0_bf16_to_f16_cuda(Tensor emb, Tensor weight, Tensor bias, double eps) {
    TORCH_CHECK(emb.is_cuda() && emb.scalar_type() == torch::kBFloat16 && emb.is_contiguous(),
                "embedding must be contiguous CUDA bf16");
    TORCH_CHECK(emb.dim() == 2 && emb.size(1) == 4096, "embedding must have shape [V,4096]");
    TORCH_CHECK(weight.numel() == 4096 && bias.numel() == 4096, "ln0 shape mismatch");
    auto out = torch::empty(emb.sizes(), emb.options().dtype(torch::kFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    emb_ln0_bf16_to_f16_kernel<<<static_cast<unsigned int>(emb.size(0)), 256, 0, stream>>>(
        static_cast<int>(emb.size(0)), 4096,
        reinterpret_cast<const uint16_t*>(emb.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(bias.data_ptr<at::BFloat16>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void emb_ln_mix6_into_cuda(
    Tensor emb, Tensor tokens, Tensor shift, Tensor weight, Tensor bias,
    Tensor xr, Tensor xw, Tensor xk, Tensor xv, Tensor xa, Tensor xg,
    Tensor x_out, Tensor out_r, Tensor out_w, Tensor out_k, Tensor out_v, Tensor out_a, Tensor out_g,
    double eps) {
    check_half_cuda(emb, "emb");
    TORCH_CHECK(emb.dim() == 2 && emb.size(1) == 4096, "emb shape mismatch");
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == torch::kInt64 && tokens.numel() == 1,
                "tokens must be one CUDA int64 token");
    auto stream = at::cuda::getCurrentCUDAStream();
    emb_ln_mix6_release_kernel<<<1, 1024, 0, stream>>>(
        reinterpret_cast<const half*>(emb.data_ptr<at::Half>()), tokens.data_ptr<int64_t>(),
        reinterpret_cast<half*>(shift.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xr.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xw.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xk.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xv.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xa.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xg.data_ptr<at::Half>()),
        reinterpret_cast<half*>(x_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_r.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_g.data_ptr<at::Half>()),
        static_cast<int>(emb.size(0)), static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ln_mix6_into_cuda(
    Tensor x, Tensor residual, Tensor shift, Tensor weight, Tensor bias,
    Tensor xr, Tensor xw, Tensor xk, Tensor xv, Tensor xa, Tensor xg,
    Tensor x_out, Tensor out_r, Tensor out_w, Tensor out_k, Tensor out_v, Tensor out_a, Tensor out_g,
    double eps) {
    check_half_cuda(x, "x");
    TORCH_CHECK(x.numel() == 4096 && residual.numel() == 4096 && shift.numel() == 4096,
                "LN1 release path requires one C4096 row");
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 8;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(8, 1, 1);
    config.blockDim = dim3(128, 1, 1);
    config.stream = at::cuda::getCurrentCUDAStream();
    config.attrs = &attr;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config, ln_mix6_cluster8_release_kernel,
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<half*>(shift.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xr.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xw.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xk.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xv.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xa.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xg.data_ptr<at::Half>()),
        reinterpret_cast<half*>(x_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_r.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_g.data_ptr<at::Half>()),
        static_cast<float>(eps)));
}

void rkv_lowrank_pre_into_cuda(
    Tensor xr, Tensor xk, Tensor xv, Tensor wr, Tensor wk, Tensor wv,
    Tensor yr, Tensor yk, Tensor yv, Tensor xw, Tensor xa, Tensor xg, Tensor xlr_v,
    Tensor w1_t, Tensor a1_t, Tensor g1_t, Tensor v1_t,
    Tensor w1, Tensor a1, Tensor g1, Tensor v1, bool skip_v) {
    TORCH_CHECK(xr.numel() == 4096 && xk.numel() == 4096 && xv.numel() == 4096,
                "RKV inputs must be C4096");
    TORCH_CHECK(wr.sizes() == torch::IntArrayRef({4096, 4096}) &&
                wk.sizes() == torch::IntArrayRef({4096, 4096}) &&
                wv.sizes() == torch::IntArrayRef({4096, 4096}), "RKV weight shape mismatch");
    TORCH_CHECK(w1_t.size(0) == 128 && a1_t.size(0) == 128 && g1_t.size(0) == 480,
                "fixed low-rank input ranks must be W128/A128/G480");
    TORCH_CHECK(skip_v || v1_t.size(0) == 96, "normal layers require V rank 96");
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr{};
    auto config = pdl_config(dim3(7040), dim3(128), stream, attr);
#define PRE_ARGS \
        reinterpret_cast<const half*>(xr.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(xk.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(xv.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(wr.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(wk.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(wv.data_ptr<at::Half>()), \
        reinterpret_cast<half*>(yr.data_ptr<at::Half>()), \
        reinterpret_cast<half*>(yk.data_ptr<at::Half>()), \
        reinterpret_cast<half*>(yv.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(xw.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(xa.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(xg.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(xlr_v.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(w1_t.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(a1_t.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(g1_t.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(v1_t.data_ptr<at::Half>()), \
        static_cast<const half*>(nullptr), static_cast<const half*>(nullptr), \
        reinterpret_cast<half*>(w1.data_ptr<at::Half>()), \
        reinterpret_cast<half*>(a1.data_ptr<at::Half>()), \
        reinterpret_cast<half*>(g1.data_ptr<at::Half>()), \
        reinterpret_cast<half*>(v1.data_ptr<at::Half>()), \
        static_cast<half*>(nullptr), static_cast<half*>(nullptr), \
        1, 4096, 128, 128, 480, skip_v ? 1 : 96, 896, 2
    if (skip_v) {
        C10_CUDA_CHECK(cudaLaunchKernelEx(
            &config, rkv_lowrank_pre_executor_kernel<128, 2, true, true, 1, 2>, PRE_ARGS));
    } else {
        C10_CUDA_CHECK(cudaLaunchKernelEx(
            &config, rkv_lowrank_pre_executor_kernel<128, 2, true, false, 1, 2>, PRE_ARGS));
    }
#undef PRE_ARGS
}

void rankout_into_cuda(
    Tensor w1, Tensor a1, Tensor g1, Tensor v1, Tensor w2_t, Tensor a2_t, Tensor g2_t, Tensor v2_t,
    Tensor v, Tensor v_first, Tensor v0, Tensor k_raw, Tensor k_k, Tensor a0, Tensor k_a, Tensor w0,
    Tensor w, Tensor a, Tensor g, Tensor v_out, Tensor new_k, Tensor neg_kk, Tensor kka, bool skip_v) {
    TORCH_CHECK(w1.numel() == 128 && a1.numel() == 128 && g1.numel() == 480,
                "fixed rankout requires W128/A128/G480");
    TORCH_CHECK(w2_t.sizes() == torch::IntArrayRef({4096, 128}) &&
                a2_t.sizes() == torch::IntArrayRef({4096, 128}) &&
                g2_t.sizes() == torch::IntArrayRef({4096, 480}), "fixed rankout weight shape mismatch");
    TORCH_CHECK(skip_v || (v1.numel() == 96 && v2_t.sizes() == torch::IntArrayRef({4096, 96})),
                "normal-layer fixed rankout requires V96");
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kFloat32 && w.is_contiguous(),
                "effective W output must be contiguous CUDA fp32");
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr{};
    auto config = pdl_config(dim3(skip_v ? 1792 : 2048), dim3(128), stream, attr);
#define RANK_ARGS \
        reinterpret_cast<const half*>(w1.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(a1.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(g1.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(v1.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(w2_t.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(a2_t.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(g2_t.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(v2_t.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(v_first.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(v0.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(k_raw.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(k_k.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(a0.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(k_a.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(w0.data_ptr<at::Half>()), \
        w.data_ptr<float>(), \
        reinterpret_cast<half*>(a.data_ptr<at::Half>()), \
        reinterpret_cast<half*>(g.data_ptr<at::Half>()), \
        reinterpret_cast<half*>(v_out.data_ptr<at::Half>()), \
        reinterpret_cast<half*>(new_k.data_ptr<at::Half>()), \
        reinterpret_cast<half*>(neg_kk.data_ptr<at::Half>()), \
        reinterpret_cast<half*>(kka.data_ptr<at::Half>()), \
        static_cast<int>(g1.numel()), \
        true
    if (skip_v) {
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, rankout_fixed_kernel<true>, RANK_ARGS));
    } else {
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, rankout_fixed_kernel<false>, RANK_ARGS));
    }
#undef RANK_ARGS
}

void wkv_into_cuda(Tensor state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor y) {
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kFloat32 && state.numel() == 64 * 64 * 64,
                "WKV state must be fp32 [1,64,64,64]");
    TORCH_CHECK(r.numel() == 4096 && w.numel() == 4096 && k.numel() == 4096 && v.numel() == 4096,
                "WKV operands must be C4096");
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kFloat32 && w.is_contiguous(),
                "WKV effective decay must be contiguous CUDA fp32");
    auto stream = at::cuda::getCurrentCUDAStream();
    wkv_halfwarp8_wait_release_kernel<<<dim3(8, 64, 1), 128, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const half*>(r.data_ptr<at::Half>()),
        w.data_ptr<float>(),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lnx_into_cuda(Tensor x, Tensor r, Tensor k, Tensor v, Tensor r_k, Tensor weight, Tensor bias, Tensor g, Tensor out) {
    TORCH_CHECK(x.numel() == 4096 && r_k.numel() == 4096 && out.numel() == 4096,
                "LNX release path requires C4096");
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr{};
    auto config = pdl_config(dim3(64), dim3(64), stream, attr);
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config, lnx_onepass_wait_release_kernel,
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(r.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(r_k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(g.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>())));
}

void att_out_into_cuda(Tensor x, Tensor weight, Tensor out) {
    TORCH_CHECK(x.numel() == 4096 && weight.sizes() == torch::IntArrayRef({4096, 4096}) && out.numel() == 4096,
                "ATT output requires [4096] x [4096,4096]");
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr{};
    auto config = pdl_config(dim3(4096), dim3(64), stream, attr);
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config, att_out_pair_wait_release_kernel,
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>())));
}

void ln_cmix_into_cuda(Tensor x, Tensor residual, Tensor shift, Tensor weight, Tensor bias, Tensor x_k, Tensor x_out, Tensor mixed, double eps) {
    TORCH_CHECK(x.numel() == 4096 && residual.numel() == 4096 && mixed.numel() == 4096,
                "CMIX LN requires C4096");
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr{};
    auto config = pdl_config(dim3(1), dim3(1024), stream, attr);
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config, ln_cmix_centered_pack_wait_kernel,
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<half*>(shift.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(x_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(mixed.data_ptr<at::Half>()), static_cast<float>(eps)));
}

void cmix_key_into_cuda(Tensor x, Tensor weight, Tensor out) {
    TORCH_CHECK(x.numel() == 4096 && weight.dim() == 2 && weight.size(1) == 4096 &&
                weight.size(0) % 2 == 0 && out.numel() == weight.size(0), "CMIX key shape mismatch");
    auto stream = at::cuda::getCurrentCUDAStream();
    cmix_key_outphase_kernel<<<static_cast<unsigned int>(weight.size(0) / 2), 128, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cmix_value_into_cuda(Tensor act, Tensor weight, Tensor out) {
    TORCH_CHECK(act.dim() == 1 && act.numel() == weight.size(0) && weight.dim() == 2 &&
                weight.size(1) == 4096 && act.numel() % 128 == 0 && out.numel() == 4096,
                "CMIX value split4 shape mismatch");
    auto stream = at::cuda::getCurrentCUDAStream();
    cmix_value_split4_kernel<<<dim3(static_cast<unsigned int>(act.numel() / 128), 16, 1), 128, 0, stream>>>(
        reinterpret_cast<const half*>(act.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void final_ln_into_cuda(Tensor x, Tensor residual, Tensor weight, Tensor bias, Tensor out, double eps) {
    TORCH_CHECK(x.numel() == 4096 && residual.numel() == 4096 && out.numel() == 4096,
                "final LN requires C4096");
    auto stream = at::cuda::getCurrentCUDAStream();
    final_ln_release_kernel<256, true, true><<<1, 256, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        1, 1, 4096, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void head_into_cuda(Tensor x, Tensor weight, Tensor out) {
    TORCH_CHECK(x.numel() == 4096 && weight.dim() == 2 && weight.size(1) == 4096 &&
                weight.size(0) % 2 == 0 && out.numel() == weight.size(0), "HEAD shape mismatch");
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr{};
    auto config = pdl_config(dim3(static_cast<unsigned int>(weight.size(0) / 2)), dim3(256), stream, attr);
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config, head_mode26_wait_kernel,
        4096,
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>())));
}
