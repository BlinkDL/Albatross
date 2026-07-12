# Phase 5 分析报告：Megakernel Grid Sync 可行性评估

## 概述

Phase 5 计划使用 `cuda::cooperative_groups::grid_sync()` 将小 grid 内核融合为 megakernel，进一步减少 kernel launch 开销。经过详细分析，**Phase 5 在当前架构下不可实施**，原因如下：

1. `rwkv_tmix_megakernel`（kk_a_gate + wkv + lnx）已在 Phase 4 完成
2. `rwkv_cmix_megakernel`（ln2_cmix_mix + ln1_tmix_mix6）存在数据依赖，不可融合
3. grid_sync 的前提条件（grid blocks ≤ 46 SMs）对关键内核不满足
4. Phase 3/4 实测表明：轻量 elementwise 内核融合的收益低于测量噪声

## 内核 Grid 实测数据（2.9B 模型, C=2560, H=40, F=10240）

| 内核 | Grid 配置 | 总 Block 数 | ≤ 46 SMs? | 可用 grid_sync? |
|------|-----------|------------|-----------|----------------|
| wkv_fp16_v1_clone | B×H = 1×40 | 40 | ✅ | ✅ |
| add_layer_norm_cmix_mix | rows = 1 | 1 | ✅ | ✅ |
| add_layer_norm_tmix_mix6 | rows = 1 | 1 | ✅ | ✅ |
| linear_nvfp4_orig_row1 (att output) | C/OutTile = 2560/128 | 20 | ✅ | ✅ |
| linear_nvfp4_orig_row1 (ffn key) | F/OutTile = 10240/128 | 80 | ❌ | ❌ |
| cmix_sparse_spmv_relu_one_nf4 | (F/128, ceil(C/1024), 1) | 240 | ❌ | ❌ |
| linear_nvfp4_rkv_orig_row1 | (C/OutTile, 1, 3) | 60 | ❌ | ❌ |

**关键发现**: NF4 SpMV 使用 240 个 block，ffn key GEMV 使用 80 个 block，均远超 46 SM 限制。

## 可行性分析

### 1. rwkv_tmix_megakernel（已完成 → Phase 4）

| 项目 | 说明 |
|------|------|
| 原始内核 | #3 kk_a_gate + #4 wkv + #5 lnx_rkvres_xg |
| Grid | B×H = 40 blocks |
| grid_sync 需要? | **不需要** — 三个内核数据流连续，直接在 kernel 内融合 |
| 状态 | ✅ Phase 3+4 已实现，CosSim=0.99998 |

Phase 4 已将 kk_a_gate + wkv + lnx 融合为单一 kernel，无需 grid_sync。这是唯一可行的 megakernel。

### 2. rwkv_cmix_megakernel（不可行 — 数据依赖）

执行流程（每层）:
```
#7 add_layer_norm_cmix_mix: (x, xx) → (x', mixed)
    ↓ mixed 进入 cmix FFN 路径
#8 ffn key GEMV: mixed → hid          [80 blocks, 无法 grid_sync]
    ↓
#9 NF4 SpMV: hid → xx'               [240 blocks, 无法 grid_sync]
    ↓ xx' 回到主路径
#13 add_layer_norm_tmix_mix6: (x', xx') → (x'', pre_mix)
```

**#7 和 #13 之间隔着整个 cmix FFN 路径（#8 + #9），存在严格的数据依赖**:
- #7 输出 `mixed` → #8 输入
- #8 输出 `hid` → #9 输入
- #9 输出 `xx'` → #13 输入

因此 #7 和 #13 **无法合并为单一 kernel**。

### 3. 替代融合方案评估

| 方案 | 融合对象 | 可行? | 原因 |
|------|---------|--------|------|
| NF4 SpMV + tmix_mix6 | #9 + #13 | ❌ | SpMV 240 blocks >> 46 SMs，且使用 atomicAdd |
| cmix_mix + ffn key GEMV | #7 + #8 | ❌ | GEMV 80 blocks >> 46 SMs |
| att output GEMV + cmix_mix | #6 + #7 | ❌ | GEMV 20 blocks + cmix_mix 1 block = 需跨 kernel 同步 |
| att output GEMV 融入 wkv | #6 融入 Phase 4 | ❌ | 40+20=60 blocks > 46，需 grid_sync + -rdc=true |

所有替代方案均因 block 数超限或数据依赖而不可行。

### 4. grid_sync 实施障碍

| 障碍 | 影响 |
|------|------|
| `-rdc=true` 编译标志 | 降低所有 kernel 优化质量（禁止某些内联、延迟优化） |
| `cudaLaunchCooperativeKernel` | 启动开销高于 `<<<>>>` |
| `grid_sync()` 本身 | 全局同步开销（所有 SM 等待最慢 block） |
| 影响范围 | 需修改 `load()` 编译参数，影响所有 4 个 CUDA 扩展 |

### 5. Phase 3/4 实测收益递减

| 阶段 | 融合内容 | 节省 launch | CosSim | p50 性能变化 |
|------|---------|------------|--------|-------------|
| Phase 3 | wkv + lnx | 32 次/层 | 0.99997 | +5% (best case) |
| Phase 4 | + kk_a_gate | 32 次/层 | 0.99998 | 0% (持平) |

**Phase 4 融合 kk_a_gate 后性能无改善**，表明:
- CUDA Graphs 已将 launch 开销降至 ~1.5μs/次
- 每层节省 1 次 launch × 1.5μs × 32 层 = 48μs（< 1% 总延迟）
- 额外的同步和全局内存访问开销抵消了 launch 节省
- 后续 elementwise 融合的收益将继续递减

## 当前每层内核启动数（Phase 4 后）

| # | 内核 | Block 数 | 类型 |
|---|------|---------|------|
| 1 | linear_wagv_rank_in_f16 | — | 低秩投影（Layer 0 跳过） |
| 2 | linear_wagv_rank_out_f16 | — | 低秩输出+激活+vres |
| 3 | linear_nvfp4_rkv_orig_row1 | 60 | r/k/v GEMV（Phase 2 融合） |
| 4 | **wkv_lnx_kkag**（Phase 4 融合） | 40 | kk_a_gate+wkv+lnx |
| 5 | linear_nvfp4_orig_row1 (att output) | 20 | att output GEMV |
| 6 | add_layer_norm_cmix_mix | 1 | 残差+ln2+cmix shift |
| 7 | linear_nvfp4_orig_row1 (ffn key) | 80 | ffn key GEMV |
| 8 | cmix_sparse_spmv_relu_one_nf4 | 240 | relu²+NF4 SpMV |
| 9 | add_layer_norm_tmix_mix6 | 1 | 残差+ln1_next+6路shift |

**每层 9 次 launch**（Layer 0 额外 +1 tmix_mix6，Layer 31 额外 +2 ln_out+head）

## 结论与建议

### Phase 5 不可实施

grid_sync megakernel 路线在当前架构下不可行:
1. 唯一可融合的 megakernel（tmix）已在 Phase 4 完成
2. cmix megakernel 因数据依赖无法实现
3. GEMV 和 SpMV 内核的 block 数远超 SM 限制
4. `-rdc=true` 会降低全局 kernel 性能，得不偿失

### 优化总结（5 阶段）

| 阶段 | 状态 | 效果 | 备注 |
|------|------|------|------|
| Phase 1: CUDA Graphs | ✅ 已有 | launch 5μs→1.5μs | 基线已包含 |
| Phase 2: r/k/v GEMV 融合 | ✅ 完成 | 3→1 launch, +22.8% p50 | 最有效优化 |
| Phase 3: wkv+lnx 融合 | ✅ 完成 | 3→1 launch, ~5% p50 (best) | 消除 y HBM 往返 |
| Phase 4: kk_a_gate 融合 | ✅ 完成 | 2→1 launch, 性能持平 | 为 Phase 5 准备 |
| Phase 5: grid_sync megakernel | ❌ 不可行 | N/A | block 数超限 + 数据依赖 |

### 最终性能基准（Phase 2+3+4 vs Baseline, 2.9B NVFP4, 交替 A/B 2 轮取最佳）

| Case | Baseline p50 (ms) | Phase 2+3+4 p50 (ms) | 变化 | Baseline tok/s | Phase 2+3+4 tok/s |
|------|-------------------|----------------------|------|---------------|-------------------|
| 1x1 | 4.965 | 4.871 | +1.9% | 201.4 | 205.3 |
| 2x1 | 5.980 | 5.704 | +4.8% | 334.5 | 350.6 |
| 4x1 | 14.571 | 14.411 | +1.1% | 274.5 | 277.6 |
| 8x1 | 23.996 | 23.931 | +0.3% | 333.4 | 334.2 |
| 16x1 | 17.595 | 17.515 | +0.5% | 909.3 | 913.5 |
| 1x2 | 5.763 | 5.824 | -1.1% | 347.1 | 343.4 |
| 1x4 | 14.611 | 14.562 | +0.3% | 273.8 | 274.7 |
| 1x8 | 24.174 | 24.633 | -1.9% | 330.9 | 322.2 |

**结论**:
- T=1 decode 路径: +0.3% ~ +4.8%，最佳在 B=2（+4.8%）
- T>1 路径: 变化在噪声范围内（±2%），融合内核仅在 T=1 时激活
- GPU 热节流导致 A/B 交替测试中存在 ±15% 的绝对延迟波动，需取多轮最佳值

### 后续优化方向

1. **CUDA Graph 优化**: 当前 graph 捕获整个 forward，可进一步优化 graph 内的内存分配策略
2. **7.2B 模型测试**: 大模型可能有不同的瓶颈特征（内存带宽 vs 计算密度）
3. **Batch 优化**: 当前优化针对 T=1 decode，大 batch 场景需独立优化
4. **vLLM 集成**: 将 Phase 2-4 的 CUDA kernel 集成到 vLLM 推理框架

### Git 备份标签

| 标签 | 描述 |
|------|------|
| `backup-phase2-complete` | Phase 2 r/k/v GEMV 融合完成 |
| `backup-pre-phase3` | Phase 3 开始前 |
| `backup-phase3-complete` | Phase 3 wkv+lnx 融合完成 |
| `backup-pre-phase4` | Phase 4 开始前 |
| `backup-phase4-complete` | Phase 4 kk_a_gate 融合完成 |
| `backup-pre-phase5` | Phase 5 分析前 |
| `backup-phase5-analysis` | Phase 5 可行性分析完成 |

---

*分析日期: 2026-07-11*
*模型: RWKV-7 2.9B NVFP4 (C=2560, H=40, F=10240)*
*硬件: RTX 5070 Ti Laptop (sm_120, 46 SMs)*
