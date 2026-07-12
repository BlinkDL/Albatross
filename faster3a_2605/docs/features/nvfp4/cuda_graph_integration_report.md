# CUDA Graph 自动集成 + 条件合并报告

## 改进内容

### 1. 合并重复条件判断

**原问题**: `tmix()` 中条件被检查两次（line 592 跳过 kk_a_gate，line 609 选择 dispatch），修改一处忘了另一处会导致 k 被错误处理。

**改进**: 计算一次 `dispatch_mode` 变量（"kkag" / "lnx" / "sep"），全函数复用：

```python
if T == 1 and WKV_MODE == "fp16" and B <= int(os.environ.get("FUSE_MAX_B", "2")):
    if os.environ.get("KKAG_WKV_LNX_FUSE", "1") == "1":
        dispatch_mode = "kkag"
    elif os.environ.get("WKV_LNX_FUSE", "1") == "1":
        dispatch_mode = "lnx"
    else:
        dispatch_mode = "sep"
else:
    dispatch_mode = "sep"
```

新增 `FUSE_MAX_B` 环境变量，允许用户自定义 B 上限（默认 2）。

### 2. CUDA Graph 集成到 forward_from_x

**原问题**: CUDA Graph 仅在 `bench_case()` 和 `app.py` 中使用，直接调用 `forward_from_x()` 不走 graph，性能差 1.78x。

**改进**: `forward_from_x` 成为自动 graph 缓存包装器：

```python
def forward_from_x(self, x, state, path, ...):
    # 检查是否可用 graph
    # 首次调用: warmup → capture → restore state → replay → 返回正确输出
    # 后续调用: copy input → replay → clone output
```

**关键技术点**:
- 使用 `throwaway state` warmup，不扰动用户 state
- 捕获后 `state_backup` + `copy_` 恢复 state（捕获会前进一步）
- 捕获后立即 replay 获取正确输出（PyTorch CUDA Graph 捕获输出不可靠）
- `is_current_stream_capturing()` 检查避免嵌套捕获
- `bench_case` 自动禁用 auto-graph 避免冲突
- `app.py` 在自身捕获期间禁用 auto-graph

## 正确性验证

| 测试 | CosSim | Max Diff | 结果 |
|------|--------|----------|------|
| Step 0 (首次/捕获后replay) | 0.999558 | 0.850 | PASS |
| Step 1 (第1次replay) | 0.999991 | 0.180 | PASS |
| Step 2 (第2次replay) | 0.999981 | 0.242 | PASS |
| State diff | 0.13 shift / 0.16 wkv | — | FP16 精度内 |

## 性能对比

### Auto-Graph vs No Graph (forward_from_x 直接调用)

| B | T | No Graph p50 | Auto Graph p50 | 加速 | No Graph tok/s | Auto Graph tok/s |
|---|---|-------------|---------------|------|----------------|-----------------|
| 1 | 1 | 9.208 ms | 5.024 ms | **1.83x** | 108.6 | **199.0** |
| 2 | 1 | 9.935 ms | 6.423 ms | **1.55x** | 201.3 | **311.4** |
| 4 | 1 | 15.926 ms | 14.858 ms | 1.07x | 251.2 | 269.2 |
| 1 | 8 | 26.991 ms | 24.527 ms | 1.10x | 296.4 | 326.2 |

### Auto-Graph vs Bench-Case Graph (1x1)

| 方式 | p50 (ms) | tok/s |
|------|---------|-------|
| Auto Graph | 5.024 | 199.0 |
| Bench Case Graph | 5.370 | 186.2 |

Auto-graph 比 bench_case graph 快约 7%（bench_case 的 clone 操作有额外开销）。

### Bench_Case 标准基准 (所有融合启用)

| B | T | p50 (ms) | tok/s |
|---|---|---------|-------|
| 1 | 1 | 4.826 | 207.2 |
| 2 | 1 | 5.876 | 340.4 |
| 4 | 1 | 14.504 | 275.8 |
| 8 | 1 | 23.888 | 334.9 |
| 1 | 2 | 5.803 | 344.7 |
| 1 | 4 | 14.541 | 275.1 |
| 1 | 8 | 23.622 | 338.7 |

## 通用性保障

| 保障措施 | 说明 |
|---------|------|
| `is_current_stream_capturing()` | 嵌套捕获时自动跳过，避免冲突 |
| `CUDA_GRAPH_AUTO=0` | 可完全禁用，回退到原始路径 |
| `FUSE_MAX_B` 环境变量 | 可自定义 B 上限 |
| `bench_case` 自动禁用 | 避免与 bench 自身 graph 冲突 |
| `app.py` 捕获期间禁用 | 避免与 app 自身 graph 冲突 |
| 条件不满足自动回退 | T>1, B>FUSE_MAX_B, pp_enabled 等场景走原始路径 |
| `all_logits` 和 `last_indices` 跳过 | 这两种情况输出 shape 或索引变化，不适用 graph |

## 约束

- **State 必须复用**: 同一 (B,T) 的多次调用必须传入相同 state tensor 对象
- **首次调用较慢**: 首次调用需要 warmup + capture + replay（约 3 倍正常延迟）
- **VRAM 额外开销**: static_x + static_out + state_backup ≈ 20MB（2.9B 模型）

---

*日期: 2026-07-11*
*模型: RWKV-7 2.9B NVFP4*
*硬件: RTX 5070 Ti Laptop (sm_120)*
