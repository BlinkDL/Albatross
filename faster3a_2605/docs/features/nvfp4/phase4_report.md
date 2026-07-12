# Phase 4 报告：kk_a_gate + WKV + LNX 三合一融合

## 概述

将 `tmix_kk_a_gate_kernel`、WKV attention kernel 和 `tmix_lnx_rkvres_xg_kernel` 三个 kernel 融合为单一 CUDA kernel。kk_a_gate 的 k 归一化和门控计算直接在 wkv kernel 内部完成，消除了 2 次 kernel launch 和中间结果（gated k, neg_kk, kka）的 HBM 往返。

## 实现内容

### 修改文件
1. **`cuda/rwkv7_wkv_fp16_v2.cu`**: 新增 `wkv_fp16_v1_clone_lnx_kkag_kernel<AddW0>` 模板 kernel
   - 在 WKV 计算前，内联计算 kk_a_gate（k 归一化 + 门控）
   - k_raw, k_k, a0, a12, k_a 从全局内存加载，与 r/w 的 cp.async 并行
   - 归一化使用 warp_sum_f32 + lnx_partial 跨 warp 归约
   - 计算结果直接存入共享内存 k[], a[], bvec[]，跳过 cp.async
   - 复用 Phase 3 的 lnx 融合逻辑

2. **`cuda/rwkv7_wkv_fp16_v2.cpp`**: 新增 C++ wrapper `wkv_lnx_kkag_seq_w0()` 和 TORCH_LIBRARY 注册

3. **`rwkv7_fast_v3a.py`**: 新增 `KKAG_WKV_LNX_FUSE` 环境变量开关
   - 优先级：KKAG_WKV_LNX_FUSE > WKV_LNX_FUSE > 原始分离模式
   - 当启用时，跳过 `tmix_kk_a_gate` 调用，raw k 直接传入融合 kernel

### 融合策略
- **消除的 kernel launch**: 每层减少 2 次（kk_a_gate + lnx），32 层共减少 64 次
- **消除的 HBM 往返**: gated k, neg_kk, kka 的写入+读取（3 × B×C×2B = 15KB/层）
- **kk_a_gate 参数加载**: k_k, a0, k_a 为 per-head [C]；k_raw, a12 为 per-batch [B,T,C]
- **归一化 reduction**: 64 线程跨 warp 归约（warp_sum_f32 + lnx_partial[2]）

## 正确性验证

| 对比 | CosSim | Max Abs Diff | 结论 |
|------|--------|-------------|------|
| Phase 4 vs Baseline | 0.99998 | 0.086 | **PASS** |
| Phase 4 vs Phase 3 | 0.99994 | 0.177 | FP16 精度范围内 |

## 性能基准

### 交替 A/B 测试（3 轮，1x1，warmup=2，iters=5）

| 轮次 | Baseline p50 (ms) | Phase 4 p50 (ms) | 变化 |
|------|-------------------|-------------------|------|
| 1 | 5.787 | 5.853 | -1.1% |
| 2 | 5.912 | 5.801 | +1.9% |
| 3 | 5.897 | 5.809 | +1.5% |

### 最佳 p10 延迟对比

| 指标 | Baseline | Phase 4 | 变化 |
|------|----------|---------|------|
| Best p10 (ms) | 5.766 | 5.784 | -0.3% |
| Best p50 (ms) | 5.787 | 5.801 | -0.2% |

### 多 case 对比（3 轮最佳 p50）

| Case | Baseline tok/s | Phase 4 tok/s | 变化 |
|------|---------------|---------------|------|
| 1x1 | 172.8 | 172.4 | -0.2% |
| 2x1 | 294.4 | 290.3 | -1.4% |
| 4x1 | 275.1 | 274.2 | -0.3% |
| 8x1 | 332.7 | 330.3 | -0.7% |
| 16x1 | 904.7 | 917.7 | +1.4% |
| 1x2 | 344.1 | 334.5 | -2.8% |
| 1x4 | 273.3 | 271.6 | -0.6% |
| 1x8 | 333.7 | 335.4 | +0.5% |

## 分析

### 性能未提升的根因
1. **kk_a_gate kernel 极轻量**: 仅做元素级乘法 + warp 归一化 + sigmoid，执行时间 < 5μs
2. **节省的 launch 开销小**: 64 次 kernel launch × ~5μs = ~320μs，但总延迟 5.8ms 中仅占 5.5%
3. **额外开销抵消收益**:
   - kk_a_gate 参数从全局内存加载（k_k, a0, a12, k_a），增加 4 次全局读取
   - 额外 4 次 __syncthreads() 用于归一化 reduction
   - 寄存器压力增加（k_val, kk_scale, u, a0_val, a12_val, ka 等同时存活）
4. **GPU 热节流影响**: 笔记本 GPU 频率波动大，测量噪声掩盖微小差异

### 融合的价值
- **正确性已验证**: CosSim = 0.99998，可作为后续优化基础
- **代码架构准备**: Phase 5 megakernel 可在此 kernel 基础上扩展
- **减少 Python 调度开销**: 每层减少 2 次 Python→C++ 调用开销

## 备份标签
- `backup-pre-phase4` (已有)
- `backup-phase4-complete`

## 结论
Phase 4 三合一融合 kernel 正确性通过（CosSim=0.99998），但性能与 baseline 基本持平。主要原因是 kk_a_gate kernel 本身非常轻量，融合节省的 launch 开销被额外的全局内存加载和同步开销抵消。此 kernel 为 Phase 5 megakernel 提供了完整的融合基础。
