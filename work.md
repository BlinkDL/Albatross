# Albatross 量化项目 work.md

## 项目状态

| 项目 | 状态 | 说明 |
|---|---|---|
| **v1 INT8 离线量化工具 + 精度验证** | ✅ 完成 | quantize.py + test_quant.py |
| **v2 INT8 Python 推理集成 + v3a** | ✅ 完成 | 初步集成 |
| **v2b CUDA INT8 kernel naive** | ✅ 完成 | 1.55ms, 52x 慢 |
| **v3 M=1 CUDA INT8 kernel N-tiled** | ✅ 完成 | 1.19ms, 40x 慢 |
| **v4 fp16×int8 dequant kernel** | ✅ 完成 | 0.05ms 微基准, 但 dispatch bug 导致走 fallback → 6.4 tok/s |
| **v5 修复 dispatch + 1:1 复制原版 exact/exact4** | ✅ **完成** | 85.1 tok/s (vs fp16 4.27 tok/s), VRAM 10.47GB |

## v4 的 4 个 bug（v5 修复）

1. **dequant kernel 从未执行** — `linear()` 调 `linear_int8_f16`（Python fallback），而非 CUDA dequant kernel。`_cuda_int8=False` 因 4 参数 vs 3 参数不匹配被 try/except 静默吞掉
2. **权重布局错误** — orig 组被额外 `.t()` 转成 [K,N]，但 kernel 期望 [N,K]
3. **scale 形状错误** — 存为 [1,N]，kernel 期望 [N]（1D）
4. **ffn.value.weight 量化与 sparse cmix 冲突** — sparse kernel 硬编码 fp16

## v5 改动详情

### 设计原则

**保留原版 dispatch 树完整结构，只在叶子节点插 int8 分支。**

不替换原版 dispatch 逻辑，而是在 `linear_orig_layout` 开头加 int8 分支，处理 rows 1/2（主场景），rows>2 dequant 到 fp16 走原版。

### 1. CUDA kernel（`cuda/rwkv7_int8_ops.cu`, 345 行）

| kernel | 复制自原版 | 用途 |
|---|---|---|
| `linear_int8_orig_row1_exact_kernel<Threads, OutTile, Use4>` | `linear_orig_row1_exact_f16_kernel` / `exact4` | M=1 orig int8 |
| `linear_int8_orig_row2_exact_kernel<Threads, OutTile, Use4>` | `linear_orig_row2_exact_f16_kernel` / `exact4` | M=2 orig int8 |
| `linear_int8_f16_kernel<Threads>` | v4 保留 | M>1 fallback |

改动点（唯一）：权重读取
```
原版: const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(wj));
int8: acc[j] = fmaf(x0.x, (float)wj[0] * s[j], acc[j]);
```

优化：scale 在 K 循环外预读（hoist），不在每次迭代重复读取。

Host dispatch 1:1 复制原版 `linear_orig_rows_exact_f16_cuda`：
```cpp
rows==1: <128, 2, false/true>
rows==2: <64, 2, true> / <256, 1, true> / <128, 2, false>
```

### 2. C++ binding（`cuda/rwkv7_int8_ops.cpp`, 62 行）

```cpp
m.def("linear_int8_orig_rows_exact_f16(Tensor x, Tensor w_orig, Tensor scale, int threads, int out_tile, bool use4) -> Tensor");
m.def("linear_int8_f16(Tensor x, Tensor w_int8, Tensor w_scale) -> Tensor");
```

### 3. Python dispatch（`rwkv7_fast_v3a.py`）

**`linear_orig_layout()`** — int8 分支 1:1 对应原版参数选择：
```python
if weight.dtype == torch.int8:
    if path.rows == 1:
        # 和原版相同的 use4 选择逻辑
        use4 = ...
        return torch.ops.rwkv7_int8_ops.linear_int8_orig_rows_exact_f16(xc, weight, scale, 128, 2, use4)
    if path.rows == 2:
        # 和原版相同的 threads/out_tile/use4 选择
        ...
    # rows > 2: dequant 到 fp16，走原版 dispatch
# 原版 fp16 路径完全不动
```

**`linear()`** — int8 fallback: dequant [N,K] int8 → [K,N] fp16 → splitk/linear_f16

**`linear_head()`** — 移除 int8 redirect，让 int8 head 走 `linear_orig_layout`

**`__init__()`** — 权重量化：
- INT8_KEYS 移除 `.ffn.value.weight`
- orig 组保持 [N,K] 布局（不转置）
- per-channel symmetric: `max_abs per row (dim=1)`, scale=[N] 1D

## 文件清单

| 路径 | 行数 | 说明 |
|---|---|---|
| `cuda/rwkv7_int8_ops.cu` | 345 | 3 个 kernel: row1 exact/exact4, row2 exact/exact4, M>1 fallback |
| `cuda/rwkv7_int8_ops.cpp` | 62 | C++ binding, 2 个 op 注册 |
| `rwkv7_fast_v3a.py` | ~1120 | 基于原版 + int8 dispatch 分支 |
| `rwkv7_v3a_ops.cu` | 3719 | 原版，没动过 |
| `rwkv7_v3a_ops.cpp` | 681 | 原版，没动过 |
| `rwkv7_wkv_fp16_v2.cu` | 901 | 原版，没动过 |
| `rwkv7_fast_ops_fp16.cu` | 1486 | 原版，没动过 |

## 环境

- WSL `~/Albatross/`，uv 管理（`uv run python xxx.py`）
- Shell 是 zsh（别用 `*` glob，用脚本绕）
- 模型 `/home/njzy/model/rwkv7-g1g-7.2b-20260523-ctx8192.pth`
- 备份到 `~/Backups/Albatross-YYYYMMDD_HHMM.tar.gz`
- 原版 `~/Backups/Albatross-SNAP.tar.gz`

## 速度对比

### 微基准（M=1 matmul，v5 测试实测）

| 矩阵形状 | int8 (ms) | fp16 (ms) | 比值 |
|---|---|---|---|
| 4096×4096 (att, use4=False) | 0.021 | 0.020 | 1.05x |
| 4096×4096 (att, use4=True) | 0.020 | 0.020 | 1.02x |
| 16384×4096 (ffn_key, use4=False) | 0.110 | 0.214 | **0.51x (快 2x)** |
| 65536×4096 (head, use4=True) | 0.433 | 0.872 | **0.50x (快 2x)** |

int8 在大矩阵上比 fp16 快约 2 倍：int8 权重内存读取量是 fp16 的一半，GEMV 场景下带宽是瓶颈，减半读取直接提速。

### 完整推理（7.2B, 实测）

| 版本 | B=1,T=1 tok/s | B=1,T=2 tok/s | GPU 占用 |
|---|---|---|---|
| **v5 int8** | **85.10** | **151.93** | **10.47 GB** |
| v4 (dispatch bug) | 6.4 | — | 11.35 GB |
| v3 old INT8 | 2.8 | — | 7.1 GB |
| v2b naive | 2.5 | — | 7.1 GB |
| 原版 fp16 | 4.27 | 5.69 | 11.94 GB (VRAM 满) |

fp16 基线 4.27 tok/s 偏低的原因：7.2B fp16 权重 14.4GB 超过 12GB VRAM，系统用 CPU offload 导致速度下降。int8 量化后权重降到 ~8.3GB，完全放进 VRAM，加上 int8 kernel 的带宽优势，达到 85 tok/s。

### 大批次推理

| B×T | M=B*T | tok/s | p50 (ms) | 走哪条路径 |
|---|---|---|---|---|
| 1×1 | 1 | 85.10 | 11.75 | int8 row1 exact |
| 1×2 | 2 | 151.93 | 13.16 | int8 row2 exact |
| 2×1 | 2 | 168.56 | 11.87 | int8 row2 exact |
| 1×4 | 4 | 21.08 | 189.78 | dequant fallback |
| 1×8 | 8 | 41.95 | 190.71 | dequant fallback |
| 1×16 | 16 | 82.40 | 194.17 | dequant fallback |
| 1×32 | 32 | 191.74 | 166.90 | dequant fallback |
| 4×1 | 4 | 10.90 | 367.00 | dequant fallback |
| 8×1 | 8 | 24.49 | 326.64 | dequant fallback |

M=1/M=2 走 int8 kernel 极快。M≥4 走 dequant fallback（先 int8→fp16 再 cuBLAS），有额外 dequant 开销。逐 token 生成场景（M=1）是 v5 的核心优化目标。

### 精度验证（kernel 级 CosSim）

| 测试 | CosSim | 结果 |
|---|---|---|
| 4096×4096 use4=False | 0.999963 | PASS (≥0.966) |
| 4096×4096 use4=True | 0.999964 | PASS |
| 16384×4096 use4=True | 0.999963 | PASS |
| 65536×4096 use4=True | 0.999963 | PASS |
| 最小尺寸 N=2 K=2 | 0.999999 | PASS |
| 零权重 | 0.0 (输出全零) | PASS |
| M=2 各种参数 | 0.999963+ | PASS |
| M>2 dequant fallback | 0.999962+ | PASS |

## 核心设计决策

| 决策 | 结论 |
|---|---|
| 主力精度 | INT8（per-channel symmetric, max_abs/127） |
| M=1 生成 | int8 orig exact/exact4 kernel（1:1 复制原版结构） |
| M=2 | int8 orig row2 exact/exact4 kernel |
| M>2 | dequant 到 fp16 走原版 cuBLAS |
| ffn.value.weight | 不量化（sparse cmix 要求 fp16） |
| 低秩权重 | 不量化（占 ~5% 参数，收益小） |
| WKV 算子 | 不动 |
| 原版 CUDA kernel | 全部不动 |
| 原版 dispatch 树 | 保留完整结构，只在叶子节点插 int8 分支 |

## 红线

1. 不破坏 FP16 路径 ✅
2. 不写死代码 ✅
3. 不动 WKV 公式 ✅

## 待办

- [x] 同步到 WSL：`cuda/rwkv7_int8_ops.cu`, `cuda/rwkv7_int8_ops.cpp`, `rwkv7_fast_v3a.py`
- [x] 清理 torch_extensions 缓存
- [x] 编译测试
- [x] 正确性验证（CosSim ≥ 0.966）— 全部 PASS，最高 0.999963
- [x] 速度基准（1x1 case）— 85.10 tok/s
- [x] 大批次测试（1x2 ~ 8x1）— M=1/M=2 极快，M≥4 走 dequant fallback

## 离线量化改造（2026-07-04）

### 改动

移除了在线/实时量化功能，改为纯离线量化方式：

| 改动项 | 改动前 | 改动后 |
|---|---|---|
| 量化时机 | 推理引擎加载模型时实时量化 | 独立离线工具 `quantize_int8.py` 预先量化 |
| 参数控制 | `--quant off/int8` 参数 | 无需参数，根据权重 dtype 自动 dispatch |
| 模型文件 | 始终是 FP16 .pth | FP16 .pth 或 INT8 .pth（离线工具生成） |
| `QUANT_MODE` 全局变量 | 存在 | 已移除 |
| `--quant` 命令行参数 | 存在 | 已移除 |
| rkv 融合检查 | `QUANT_MODE == "off"` | `has_int8_att`（dtype 检测） |

### `quantize_int8.py` 开发过程中修复的 3 个 bug

1. **CosSim > 1.0** — `F.cosine_similarity` 在 float32 超大向量（head.weight 268M 元素）上累积误差导致结果略超 1.0。改用 float64 计算点积和范数。
2. **输出文件 18.97GB（应约 10GB）** — `.cpu()` 对已在 CPU 上的 mmap 张量返回张量本身（不复制），导致 `torch.save` 序列化 mmap view 时复制底层文件。改用 `.clone()` 强制复制。
3. **同步失败** — Windows `\\wsl.localhost\` 路径写入和 WSL 内 `/home/njzy/` 可能存在缓存不一致。改为在 WSL 内用 Python `shutil.copy2` 从 `/mnt/d/` 复制。

### 量化结果

| 指标 | 值 |
|---|---|
| 原始模型大小 | 14 GB (bfloat16) |
| 量化后模型大小 | 9.84 GB |
| 节省 | 4.16 GB |
| 量化+验证+保存耗时 | 136s |
| INT8 权重数 | 161（32 层 × 5 + head） |
| Scale 张量数 | 161 |
| CosSim（att 权重） | 0.99994-0.99996 |
| CosSim（ffn.key） | 0.99995-0.99996 |
| CosSim（head.weight） | 0.99989 |
| max error (1-cossim) | 0.000110 |

### 用法

```bash
# 离线量化（只需做一次）
python3 quantize_int8.py --model model.pth --out model-int8.pth --verify

# FP16 推理（原版模型）
python3 rwkv7_fast_v3a.py --model model.pth

# INT8 推理（量化后的模型）
python3 rwkv7_fast_v3a.py --model model-int8.pth
```

### 已知限制

- INT8 模型不支持 `--cmix-sparse auto` 模式（sparse cmix kernel 需要 ffn.key.weight 的 fp16 `.fc` 副本，但量化后不生成）
- 默认 `--cmix-sparse no-fc` 不受影响

## Prefill 优化：CUDA dequant kernel（2026-07-04）

### 问题

M≥4 的 prefill 路径走 dequant fallback，每个权重需要 4 步 Python 操作：
1. `weight.float()` — int8 → float32（4 倍内存膨胀）
2. `* scale.unsqueeze(1).float()` — 逐元素乘
3. `.to(torch.float16)` — 转回 fp16
4. `.t().contiguous()` — 转置 + 拷贝

每步都读写 global memory，32 层 × 5 权重 = 160 次，额外开销约 320-480ms。

### 修复

新增 `dequant_int8_to_f16_kernel` CUDA kernel，一步完成 int8→fp16 反量化（可选转置），替代 Python 4 步链。

改动文件：
- `cuda/rwkv7_int8_ops.cu` — 新增 `dequant_int8_to_f16_kernel<Threads,Transpose>` + host wrapper
- `cuda/rwkv7_int8_ops.cpp` — 注册 `dequant_int8_to_f16` op
- `rwkv7_fast_v3a.py` — `linear()` 和 `linear_orig_layout()` 的 dequant fallback 改用 CUDA kernel

### 性能结果

| 矩阵形状 | CUDA dequant (ms) | 说明 |
|---|---|---|
| 4096×4096 no-trans | 1.249 | att 权重，orig 布局 |
| 4096×4096 trans | 1.534 | att 权重，non-orig 布局 |
| 16384×4096 no-trans | 0.542 | ffn.key 权重 |
| 65536×4096 trans | 18.635 | head 权重 |

Python dequant 链约 2-3ms（4096×4096），CUDA kernel 约 1.2ms，提速约 2 倍。
对于 7.2B 模型 prefill，32 层 × 5 权重 × 节省约 1ms = 约 160ms 总节省。

## GPU 占用率分析与验证（2026-07-04）

### 问题

用户报告 INT8 推理 GPU 占用率仅 ~40%，而 FP16 推理可达 ~80%。

### 诊断过程

#### 第一步：`__launch_bounds__` 分析（误诊→排除）

初始怀疑 INT8 kernel 的 `__launch_bounds__(Threads, 1)` 限制 SM 占用率。对比发现：

| kernel | `__launch_bounds__` | 寄存器 | spills |
|---|---|---|---|
| 原版 FP16 row1 exact | `(Threads, 1)` | 48 | 0 |
| INT8 row1 exact (改前) | `(Threads, 1)` | 60 | 0 |
| INT8 row1 exact (改后) | `(Threads, 10)` | 46 | 0 |

改 `__launch_bounds__(Threads, 1)` 为 `(Threads, 10)` 后寄存器从 60 降至 46，零溢出。但进一步检查发现**原版 FP16 kernel 同样使用 `(Threads, 1)`**（`rwkv7_v3a_ops.cu` 第 521/568/619/677 行），FP16 却能达 80% 占用率。故 `__launch_bounds__` **非根因**。

保留 `(Threads, 10)` 改动：对 INT8 寄存器优化有益（INT8 因 int→float 转换 + scale 乘法，寄存器压力高于 FP16），虽非根因但无害。

#### 第二步：实测验证（找到根因）

用 nvidia-smi 每秒采样 GPU 占用率，分别用 `--iters 3`（默认）和 `--iters 200` 运行 benchmark：

**INT8 `--iters 200`**（benchmark 持续 ~3.4s）：

| 阶段 | 时间 | GPU 占用 |
|---|---|---|
| 模型加载+权重预处理 | ~8.6s | 0-29% |
| CUDAGraph 捕获 | ~0.5s | 68-83% |
| **实际 benchmark (200 iters)** | **~3.4s** | **99%, 99%, 99%** |
| 结束 | — | 0% |

**INT8 `--iters 3`**（benchmark 仅 ~50ms）：

| 阶段 | 时间 | GPU 占用 |
|---|---|---|
| 模型加载+权重预处理 | ~8.6s | 0-29% |
| CUDAGraph 捕获 | ~0.5s | 68-83% |
| **实际 benchmark (3 iters)** | **~50ms** | **nvidia-smi 未采样到** |
| 结束 | — | 0% |

nvidia-smi 采样间隔 1 秒，而 `--iters 3` 的 benchmark 仅 50ms，采样全部落在加载阶段 → 显示 ~10-24%。

**FP16 `--iters 10`**（benchmark 持续 ~3.7s，CPU offload）：

| 阶段 | 时间 | GPU 占用 |
|---|---|---|
| 模型加载+权重预处理 | ~31s | 0-50%（含内存传输） |
| **实际 benchmark (10 iters)** | **~3.7s** | **75-100%（波动大）** |

FP16 每 token 373ms（含 CPU offload 内存传输），10 次迭代 = 3.7s，nvidia-smi 能采样到多次 benchmark 数据 → 显示 ~80%。

### 根因

**40% 占用率是测量假象。**

| 对比 | INT8 | FP16 |
|---|---|---|
| 每 token 耗时 | 16.77ms | 373.10ms |
| `--iters 3` benchmark 总时长 | 50ms | 1119ms |
| nvidia-smi 采样窗口 | 1000ms | 1000ms |
| benchmark 期间采样命中 | 0 次 | 1-2 次 |
| 显示占用率 | ~10-24%（全是加载阶段） | ~80%（含 benchmark） |
| **实际 benchmark 期间占用率** | **99%** | **75-100%** |

FP16 的 80% 含水分：14.4GB FP16 模型超 12GB VRAM，CPU offload 的内存传输被 nvidia-smi 计为 GPU 活跃。波动大（0%, 100%, 55%, 100%）正是内存传输与计算交替的体现。

### `__launch_bounds__` 改动记录

虽非根因，但改动有益（降低 INT8 寄存器压力），保留：

```
// cuda/rwkv7_int8_ops.cu 第 44 行和第 124 行
// 改前: __launch_bounds__(Threads, 1)
// 改后: __launch_bounds__(Threads, 10)
```

ptxas 验证（`--ptxas-options=-v`）：
- row1 use4=true: 60→46 寄存器，0 spills
- row1 use4=false: 48→48 寄存器，0 spills
- row2 use4=true<64,2>: 48 寄存器，0 spills
- row2 use4=true<256,1>: 40 寄存器，0 spills

### `--emb gpu` 对比

| 配置 | VRAM | p50 (ms) | tok/s | benchmark 期间 GPU 占用 |
|---|---|---|---|---|
| `--emb cpu` (默认) | 10.47GB | 16.77 | 59.63 | 99% |
| `--emb gpu` | 11.94GB (满) | 16.77 | 59.62 | 99% |

`--emb gpu` 占满 VRAM 但无速度收益。CUDAGraph 已消除 embedding CPU 开销，benchmark 期间 GPU 持续 99%。保持 `--emb cpu` 默认。

### 验证脚本

`/tmp/verify_gpu_util.py` 和 `/tmp/run_gpu_monitor.sh`（临时文件，不纳入 PR）。

### 结论

**INT8 推理实际 GPU 占用率 99%，高于 FP16 的 75-100%。** 无需任何额外优化。用户观察到的 40% 是因 `--iters 3` benchmark 太短（50ms），nvidia-smi 1 秒采样窗口全部落在模型加载阶段。建议用 `--iters 200` 或更高进行 GPU 占用率测量。

## INT8 GEMM kernel（M≥3 prefill 优化，2026-07-05）

### 背景

M≥4 prefill 场景原来走 dequant int8→fp16 → cuBLAS 路径，每次 forward 产生约 20.6GB 临时 fp16 数据，用完即弃。

### 新 kernel：linear_int8_orig_rows_f16

1:1 复制原版 `linear_orig_rows_f16_kernel` 结构，权重改为 int8 + scale。kernel 内读 int8 权重时即时乘 scale 还原为 float 再做 fmaf。**无 dequant，无临时张量。**

文件改动：
- `cuda/rwkv7_int8_ops.cu`：新增 `linear_int8_orig_rows_kernel` + host wrapper + dispatch
- `cuda/rwkv7_int8_ops.cpp`：注册 `linear_int8_orig_rows_f16` op
- `rwkv7_fast_v3a.py`：`linear_orig_layout()` rows>2 路径改用新 kernel，新增 `_int8_rows_tile()` 辅助方法

### 微基准测试（正确性 + 性能）

CosSim 全部 1.000000（新 kernel vs 旧 dequant+cuBLAS 完全一致）。

| 测试用例 | new_ms | dequant_ms | cublas_ms | speedup |
|---|---|---|---|---|
| att M=4 4096x4096 | 0.033 | 0.167 | 0.027 | 5.02x |
| att M=8 4096x4096 | 0.058 | 0.169 | 0.026 | 2.92x |
| att M=16 4096x4096 | 0.148 | 0.170 | 0.029 | 1.15x |
| att M=64 4096x4096 | 0.443 | 0.173 | 0.047 | 0.39x |
| ffn_key M=4 16384x4096 | 0.150 | 0.565 | 0.228 | 3.78x |
| ffn_key M=8 16384x4096 | 0.289 | 0.571 | 0.241 | 1.97x |
| ffn_key M=16 16384x4096 | 0.730 | 0.673 | 0.226 | 0.92x |
| head M=4 65536x4096 | 0.576 | 2.435 | 0.883 | 4.23x |
| head M=8 65536x4096 | 1.133 | 2.417 | 1.314 | 2.13x |
| head M=16 65536x4096 | 3.693 | 2.482 | 1.102 | 0.67x |

**结论**：M=4-8 场景新 kernel 快 2-5 倍。M≥16 时 cuBLAS 更优（cuBLAS tiling 对大 M 优化更好）。prefill 场景通常 M=4-16（B×T），新 kernel 在 M=4-8 全面碾压。

### 端到端 benchmark

**纯 INT8 kernel（所有 M≥3 都用新 kernel）**：

| 场景 | p50 (ms) | tok/s |
|---|---|---|
| 1x4 | 19.98 | 200.17 |
| 1x8 | 26.89 | 297.49 |
| 1x16 | 60.98 | 262.38 |
| 1x32 | 84.87 | 377.03 |
| 1x64 | 155.92 | 410.46 |

**混合策略（rows≤12 用 INT8 kernel，rows>12 回退 dequant+cuBLAS）**：

| 场景 | p50 (ms) | tok/s | vs 纯 INT8 |
|---|---|---|---|
| 1x4 | 19.92 | 200.82 | 持平 |
| 1x8 | 27.66 | 289.27 | 持平 |
| 1x16 | 60.45 | 264.69 | 持平 |
| 1x32 | 64.09 | 499.27 | **快 32%** |
| 1x64 | 68.60 | 932.90 | **快 127%** |

混合策略在 M≥32 场景大幅提速，M=4-16 场景无退化。

### A/B 对比：纯 INT8 vs 混合策略（cuBLAS）（2026-07-05）

备份：`~/Backups/Albatross-v6-int8gemm-20260705.tar.gz` + `D:\code\workspace\_backups\v6-20260705\`

iters=20, warmup=5, INT8 模型 9.84GB, VRAM 10.47/11.94GB

| 场景 | 混合策略 p50 (ms) | 纯 INT8 p50 (ms) | 混合 tok/s | 纯 INT8 tok/s | cuBLAS 优势 |
|---|---|---|---|---|---|
| 1x1 | 11.78 | 11.77 | 84.88 | 84.93 | 持平（不走 cuBLAS） |
| 1x4 | 26.32 | 26.41 | 151.96 | 151.45 | 持平（M≤12 走 INT8） |
| 1x8 | 25.80 | 25.85 | 310.10 | 309.51 | 持平 |
| 1x16 | 50.88 | 57.30 | 314.50 | 279.23 | **混合快 12%** |
| 1x32 | 54.98 | 83.69 | 582.06 | 382.38 | **混合快 52%** |
| 1x64 | 67.95 | 147.59 | 941.83 | 433.63 | **混合快 117%** |
| 1x128 | 74.73 | 282.50 | 1712.78 | 453.10 | **混合快 278%** |
| 1x256 | 106.01 | 564.02 | 2414.83 | 453.88 | **混合快 432%** |

**结论**：

- M=1-8：两种策略持平（都走 INT8 kernel，cuBLAS 无关）
- M≥16：cuBLAS 碾压纯 INT8 kernel，M 越大优势越大
- 纯 INT8 kernel 在 M≥128 时触顶 ~453 tok/s（kernel tiling 效率限制）
- cuBLAS 在 M=256 时达 2415 tok/s（5.3x 优势）

**cuBLAS 对推理的影响**：
- M=1 生成：无影响（走 INT8 exact kernel，不经过 cuBLAS）
- M≥16 prefill：cuBLAS 带来 12%-432% 加速，不可或缺
- 混合策略是最终方案：M≤12 走 INT8（无 dequant），M>12 走 cuBLAS（tiling 优化）

### 下一步

M=1/2 生成场景的 CUDAGraph 覆盖（方案 C）。

## 4-wide K 循环优化（2026-07-05）

### 改动

`linear_int8_orig_rows_kernel` 添加 `bool Use4` 模板参数。K%4==0 时用 4-wide K 循环（每次迭代处理 4 个 K 元素，减少 50% 循环次数），否则用 2-wide。

dispatch 自动选择：`const bool use4 = (x.size(-1) % 4) == 0;`。K=4096 总是 use4=true。

### GPU 时钟问题（已解决）

测试时 GPU 卡在 180 MHz 空闲时钟（max 3090 MHz），throttle reason = `sw_power_cap`，所有 benchmark 退化 2.5x。v6 备份代码同样慢，排除代码问题。

解决：用户在 Windows 端打开性能模式后 GPU 恢复正常（计算时 99% 利用率，时钟正常升频）。

### 寄存器使用分析（ptxas -v）

4-wide vs 2-wide 寄存器对比（`linear_int8_orig_rows_kernel`）：

| RowTile, OutTile | 2-wide regs | 4-wide regs | 增量 |
|---|---|---|---|
| 4, 4 | 72 | 78 | +6 |
| 8, 4 | 105 | 106 | +1 |
| 8, 2 | 72 | 87 | +15 |
| 16, 4 | 162 | 162 | 0 |
| 16, 2 | 115 | 127 | +12 |

全部 0 spills。4-wide 寄存器增量主要来自 `wv[OutTile][4]` 本地数组。

### Dispatch 策略优化

微基准测试发现 M=8 时 OutTile=4 寄存器压力影响指令调度，OutTile=2 更快：

| M | RowTile | OutTile | 微基准(ms) | 端到端 p50(ms) |
|---|---|---|---|---|
| 4 | 4 | 4 | 0.0358 | 16.94 |
| 4 | 8 | 4 | 0.0456 | — |
| 8 | 8 | 4 | 0.0806 | 27.40 |
| 8 | 8 | 2 | 0.0568 | 23.48 |

最终 `_int8_rows_tile` 策略：M≤4 → (4,4)，M>4 → (8,2)/(16,2)。

### 端到端 benchmark（iters=20, warmup=5, 两次稳定值）

| 场景 | 2-wide 基线 p50 | 4-wide 优化 p50 | 变化 | 说明 |
|---|---|---|---|---|
| 1x1 | 11.78 | 11.77 | 持平 | row1 exact kernel，不受影响 |
| 1x4 | 26.32 | **16.94** | **快 36%** | M=4, OT=4 |
| 1x8 | 25.80 | **23.48** | **快 9%** | M=8, OT=2 |
| 1x16 | 50.88 | 50.73 | 持平 | cuBLAS 路径 |
| 1x32 | 54.98 | 63.95 | -16% | cuBLAS 波动 |
| 1x64 | 67.95 | 59.70 | 快 12% | cuBLAS 波动 |
| 1x128 | 74.73 | 67.59 | 快 9% | cuBLAS 波动 |
| 1x256 | 106.01 | 105.46 | 持平 | cuBLAS 路径 |

M=4/8 prefill 场景（INT8 kernel 路径）稳定提升 9-36%。M≥16 走 cuBLAS 路径，波动为正常 benchmark 噪声。

### 改动文件

- `cuda/rwkv7_int8_ops.cu`：`linear_int8_orig_rows_kernel` 加 `bool Use4` 模板，dispatch 自动选择
- `cuda/rwkv7_int8_ops.cpp`：无改动（use4 在 CUDA dispatch 内自动选择）
- `rwkv7_fast_v3a.py`：`_int8_rows_tile` 优化 — M≤4 用 (4,4)，M>4 用 (8,2)/(16,2)

## Dequant kernel 向量化优化（2026-07-05）

### 问题

大 B×1 场景（M>12 走 cuBLAS 路径）每次 forward 需要对 161 个 int8 权重做 dequant→fp16 临时张量→cuBLAS matmul→丢弃。dequant 占 B=64 forward 的 49%（37ms / 75ms）。

### 优化

`dequant_int8_to_f16_kernel` 非转置路径向量化：一次读 8 个 int8（2 次 int32 加载），用 `__floats2half2_rn` 批量写 4 个 half2，减少循环次数和内存写入次数。

### 微基准结果

| 权重 | 优化前 (ms) | 优化后 (ms) | 提速 |
|---|---|---|---|
| att (4096×4096) no-trans | 1.249 | 0.110 | **11x** |
| ffn_key (16384×4096) no-trans | 0.542 | 0.379 | 1.4x |
| head (65536×4096) no-trans | 18.635 | 1.436 | **13x** |

### 端到端 dequant 总开销

| 权重组 | 优化前 | 优化后 |
|---|---|---|
| att (128× 4096×4096) | 20.69 ms | 22.89 ms |
| ffn_key (32× 16384×4096) | 4.82 ms | 6.26 ms |
| head (1× 65536×4096) | 12.58 ms | **1.89 ms** |
| **总计** | **36.62 ms** | **25.63 ms** |

dequant 占 B=64 forward 从 48.8% 降到 34.2%。

### 大 B×1 端到端 benchmark（单独运行，避免 CUDAGraph VRAM 干扰）

| B | 优化前 p50 (ms) | 优化后 p50 (ms) | 提升 | tok/s |
|---|---|---|---|---|
| 4 | 19.70 | 17.04 | 快 13% | 235 |
| 8 | 24.41 | 23.85 | 快 2% | 335 |
| 16 | 54.31 | 53.74 | 快 1% | 298 |
| 32 | 68.70 | 57.08 | 快 17% | 561 |
| 64 | 75.08 | 63.48 | **快 15%** | **1008** |
| 128 | 228.92 | 223.36 | 快 2% | 573 |
| 256 | 515.76 | 448.54 | 快 13% | 571 |

B=64 达到 1008 tok/s（之前 852 tok/s）。

注：连续运行多个 B case 时 B=64 会出现 670ms 异常，是 CUDAGraph 捕获的临时张量 VRAM 累积导致，非代码问题。单独运行 B=64 结果稳定。

### 改动文件

- `cuda/rwkv7_int8_ops.cu`：`dequant_int8_to_f16_kernel` 非转置路径向量化（8-wide int32 加载 + half2 批量写入）

## INT8 kernel 阈值优化（2026-07-05）

### 改动

微基准显示 M=16 时 att/head 的 INT8 kernel 比 dequant+cuBLAS 快（0.113 vs 0.142ms），但 ffn_key 是 cuBLAS 更快（0.604 vs 0.719ms）。因此按权重组区分阈值：

```python
int8_thresh = 16 if group in ("att_c2c", "head") else 12
```

att_c2c 和 head 在 rows≤16 时走 INT8 direct kernel（无 dequant），ffn_key 保持 rows≤12。

### 最终大 B×1 性能（WSL 重启后干净环境，单独运行）

| B | 初始基线 p50 (ms) | 最终 p50 (ms) | 提升 | tok/s |
|---|---|---|---|---|
| 4 | 19.70 | 17.04 | 13% | 235 |
| 8 | 24.41 | 23.80 | 2% | 336 |
| 16 | 54.31 | 52.19 | 4% | 307 |
| 32 | 68.70 | 53.69 | 22% | 596 |
| 64 | 75.08 | 63.18 | 16% | 1013 |
| 128 | 228.92 | **130.70** | **43%** | **979** |
| 256 | 515.76 | 421.00 | 18% | 608 |

B=128 从 229ms 降到 131ms，tok/s 从 553 提升到 979。

### 累积优化总结

1. **4-wide K 循环**（`linear_int8_orig_rows_kernel`）：M=4 快 36%，M=8 快 9%
2. **dispatch 策略**：M≤4 用 (4,4)，M>4 用 (8,2) 降寄存器压力
3. **dequant 向量化**：head dequant 快 13x，总 dequant 从 37ms 降到 26ms
4. **INT8 阈值优化**：att_c2c/head 在 M≤16 时走 INT8 kernel，省 dequant

### dp4a int8×int8 GEMM 实验（失败）

尝试用 `__dp4a` 做 int8×int8 GEMM 替代 dequant+cuBLAS，失败：
- 精度问题：x 量化为 int8 精度损失太大（cos_sim=0.69）
- 性能问题：`__dp4a` 在 sm_120 (Blackwell) 上比 cuBLAS 慢 28-200 倍
- 结论：per-token 量化 x 不可行，当前 dequant+cuBLAS 方案仍是最优

### 改动文件

- `rwkv7_fast_v3a.py`：`_int8_rows_tile` 优化 + INT8 阈值按权重组区分（att_c2c/head 16，ffn_key 12）
- `cuda/rwkv7_int8_ops.cu`：dequant 向量化 + 4-wide K 循环
- `cuda/rwkv7_int8_ops.cpp`：无改动

### benchmark 注意事项

连续运行多个 B case 时 CUDAGraph 捕获的临时张量会累积导致 OOM（B=64/128 时 VRAM 不足），WSL 可能崩溃。单独运行各 case 结果稳定。实际推理不使用 CUDAGraph，不受此问题影响。
