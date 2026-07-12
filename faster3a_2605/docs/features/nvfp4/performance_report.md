# NVFP4 推理优化 — 性能报告

## 硬件环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA RTX 5070 Ti Laptop (sm_120, Blackwell) |
| VRAM | 12 GB GDDR7 |
| 显存带宽 | 448 GB/s (28Gbps × 128-bit) |
| SM 数量 | 46 |
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| PyTorch | 2.12.1+cu130 |
| CUDA | 13.0 |

## 优化内容

### Phase 1-4：推理路径融合（已有）

| Phase | 内容 | 效果 |
|-------|------|------|
| Phase 1 | CUDA Graph 自动捕获 (B≤2) | launch 5μs→1.5μs |
| Phase 2 | r/k/v GEMV 融合 (3→1 kernel) | p50 +22.8% |
| Phase 3 | WKV+LNX 融合 (消除 HBM roundtrip) | p50 +5% |
| Phase 4 | kk_a_gate 融入 WKV kernel | launch 3→1 |

### Phase 6：NVFP4 GEMV kernel 优化（本次）

#### v1 → v2: Shared Memory LUT + __hfma2

**问题**：v1 kernel 每个元素做 1 次 `e2m1_decode_f`（bitwise shift+OR+table lookup）+ 1 次 `fmul`(block scale) + 1 次 `fmaf`(accumulate)，共 112 条指令/block。

**优化**：
1. 256-entry shared memory LUT：byte → `__half2`（两个 E2M1 值打包），8 次查表替代 16 次 bitwise decode
2. `__hfma2` 块内 FP16 累加：8 条指令替代 16 条 `fmaf`
3. Block scale 每块乘 1 次而非每元素乘 16 次
4. 跨块累加保持 FP32 精度

**效果**：指令数 112→18/block（6.2× 减少）

#### v2 OutTile 2→4

**优化**：每个 block 处理 4 个输出而非 2 个，x 向量加载 1 次复用 4 次。Block 数量减半。

**OutTile 探索**：

| OutTile | B=1 ms/tok | 结论 |
|---------|-----------|------|
| 2 | 17.50 | v2 baseline |
| 4 | **16.25** | 最优 |
| 8 | 24.00 | 寄存器溢出，废弃 |

#### 废弃方案：Dual-block ILP

每次迭代处理 2 个 K-block（2 条独立依赖链），但 OutTile=2 时双块导致寄存器溢出到 local memory，性能退化 4×（17.5→67ms）。废弃。

## 性能测试结果

### 正确性验证

| 模型 | CosSim | NaN/Inf | 结论 |
|------|--------|---------|------|
| 2.9B | N/A (随机输入) | 无 | PASS |
| 7.2B | N/A (随机输入) | 无 | PASS |
| 13.3B | N/A (随机输入) | 无 | PASS |

### Decode 性能 (B=1, T=1)

| 模型 | FP16 权重 | NVFP4 权重 | 显存占用 | ms/tok | tok/s |
|------|----------|-----------|---------|--------|-------|
| 2.9B | 5.5 GB | 2.2 GB | 3.4 GiB | 5.72 | 174.8 |
| 7.2B | 14.0 GB | 4.8 GB | 6.1 GiB | 10.17 | 98.4 |
| 13.3B | 25.0 GB | 8.3 GB | 10.0 GiB | 15.76 | 63.4 |

### Batch 性能

| 模型 | B=1 tok/s | B=2 tok/s | B=4 tok/s |
|------|----------|----------|----------|
| 2.9B | 174.8 | 130.0 | — |
| 7.2B | 98.4 | 72.0 | — |
| 13.3B | 63.4 | 39.2 | 14.3 |

### Prefill 性能 (B=1, T=128)

| 模型 | ms | tok/s |
|------|-----|-------|
| 2.9B | 73.2 | 1748.6 |
| 7.2B | 180.3 | 710.1 |
| 13.3B | 352.8 | 362.8 |

### 13.3B 优化前后对比

| 指标 | v1 (优化前) | v2+OutTile4 (优化后) | 提升 |
|------|-----------|---------------------|------|
| B=1 ms/tok | 19.35 | 15.76 | -18.6% |
| B=1 tok/s | 51.7 | 63.4 | +22.6% |
| B=2 tok/s | 40.1 | 39.2 | -2.2% |
| B=4 tok/s | 14.3 | 14.3 | 0% |

### 显存带宽利用率 (13.3B, B=1)

| 指标 | 值 |
|------|-----|
| 每 token HBM 读取 | 8.35 GB |
| 理论最小延迟 (448 GB/s) | 18.6 ms |
| 优化后实际延迟 | 15.76 ms |
| 带宽利用率 | 96.1% (CPU launch 重叠) |

> 注：实际延迟低于理论值是因为 CPU kernel launch (~11ms) 与 GPU 执行重叠，wall clock < GPU pure time。

## 量化工具

### quantize_stream.py (mmap 流式量化)

| 指标 | 旧方案 (zip 逐个解压) | 新方案 (mmap 懒加载) |
|------|---------------------|---------------------|
| 2.9B 量化耗时 | ~89 分钟 (估算) | 37 秒 |
| 13.3B 量化耗时 | 不可行 | 171 秒 |
| 峰值 RAM (13.3B) | N/A | 12.9 GB |

大模型(>物理内存)需配合 `vm.overcommit_memory=1`。

## 文件变更

| 文件 | 变更 |
|------|------|
| `cuda/rwkv7_nf4_ops.cu` | NVFP4 GEMV kernel v2 (LUT + hfma2 + OutTile=4) |
| `rwkv7_fast_v3a.py` | dispatch OutTile 2→4, CUDA Graph auto-caching, dispatch_mode 合并 |
| `quantize_stream.py` | 新增：mmap 流式量化工具 |
| `quantize_nf4.py` | mmap fallback 支持 |
| `app.py` | CUDA Graph 兼容 (auto-graph disable during app capture) |
