#!/usr/bin/env python3
"""v5 INT8 kernel 测试 — 边界条件 + 正确性 + 速度"""

import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

THIS_DIR = Path(__file__).resolve().parent
CUDA_DIR = THIS_DIR / "cuda"

def log(msg):
    print(f"[v5test] {msg}", flush=True)

def load_int8_ext():
    """编译加载 int8 ops"""
    cuda_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization", "-Xptxas", "-O3"]
    load(name="rwkv7_int8_ops",
         sources=[str(CUDA_DIR / "rwkv7_int8_ops.cpp"), str(CUDA_DIR / "rwkv7_int8_ops.cu")],
         is_python_module=False, verbose=True, extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags)
    log("int8_ops compiled OK")

def load_v3a_ext():
    """编译加载原版 v3a ops"""
    cuda_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization", "-Xptxas", "-O3"]
    load(name="rwkv7_v3a_ops",
         sources=[str(CUDA_DIR / "rwkv7_v3a_ops.cpp"), str(CUDA_DIR / "rwkv7_v3a_ops.cu")],
         is_python_module=False, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags)
    log("v3a_ops compiled OK")

def quantize_per_channel_symmetric(w_fp16):
    """per-channel symmetric quantization, w [N,K] -> int8 [N,K] + scale [N]"""
    wf = w_fp16.float()
    max_abs = wf.abs().max(dim=1, keepdim=True).values
    max_abs = torch.clamp(max_abs, min=1e-10)
    scale = (max_abs / 127.0).squeeze(1).to(torch.float16)
    w_int8 = (wf / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale

def cos_sim(a, b):
    """cosine similarity for 1D tensors"""
    return torch.nn.functional.cosine_similarity(a.float().flatten().unsqueeze(0),
                                                  b.float().flatten().unsqueeze(0)).item()

def max_rel_err(a, b):
    """max relative error"""
    denom = torch.max(a.abs().float(), b.abs().float())
    denom = torch.clamp(denom, min=1e-8)
    return (torch.abs(a.float() - b.float()) / denom).max().item()

# ═══════════════════════════════════════════════════════════════
# Test 1: M=1 kernel correctness — various sizes
# ═══════════════════════════════════════════════════════════════

def test_m1_correctness():
    log("=" * 60)
    log("Test 1: M=1 kernel correctness")
    log("=" * 60)

    torch.manual_seed(42)
    test_cases = [
        # (N, K, use4, desc)
        (4096, 4096, False, "att_c2c 4096x4096 use4=False"),
        (4096, 4096, True,  "head 4096x4096 use4=True"),
        (16384, 4096, True, "ffn_key 16384x4096 use4=True"),
        (16384, 4096, False, "ffn_key 16384x4096 use4=False"),
        (65536, 4096, True, "head 65536x4096 use4=True"),
        # boundary: small sizes
        (2, 4, True, "min exact4 N=2 K=4"),
        (2, 2, False, "min exact N=2 K=2"),
        (4, 8, True, "small N=4 K=8 use4=True"),
        (4, 8, False, "small N=4 K=8 use4=False"),
        # boundary: K not divisible by 4 but by 2
        (128, 6, False, "K=6 use4=False (not div by 4)"),
        (128, 10, False, "K=10 use4=False (not div by 4)"),
        # boundary: large K
        (128, 8192, True, "K=8192 use4=True"),
        (128, 8192, False, "K=8192 use4=False"),
    ]

    all_pass = True
    for N, K, use4, desc in test_cases:
        # 生成随机数据
        w_fp16 = (torch.randn(N, K, device="cuda") * 0.02).to(torch.float16)
        x_fp16 = torch.randn(1, K, device="cuda", dtype=torch.float16) * 0.5

        # 量化
        w_int8, scale = quantize_per_channel_symmetric(w_fp16)

        # int8 kernel
        try:
            y_int8 = torch.ops.rwkv7_int8_ops.linear_int8_orig_rows_exact_f16(
                x_fp16.contiguous(), w_int8.contiguous(), scale.contiguous(),
                128, 2, use4)
        except Exception as e:
            log(f"  FAIL [{desc}]: kernel error: {e}")
            all_pass = False
            continue

        # fp16 reference (用原版 v3a kernel)
        try:
            y_fp16 = torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                x_fp16.contiguous(), w_fp16.contiguous(), 128, 2, use4)
        except Exception as e:
            # 如果原版 kernel 不支持这个尺寸，用 PyTorch matmul 做 reference
            y_fp16 = (x_fp16 @ w_fp16.t()).to(torch.float16)

        # 比较
        cs = cos_sim(y_int8, y_fp16)
        mre = max_rel_err(y_int8, y_fp16)
        status = "PASS" if cs >= 0.966 else "FAIL"
        if status == "FAIL":
            all_pass = False
        log(f"  {status} [{desc}]: CosSim={cs:.6f} MaxRelErr={mre:.6f}")

    return all_pass

# ═══════════════════════════════════════════════════════════════
# Test 2: M=2 kernel correctness
# ═══════════════════════════════════════════════════════════════

def test_m2_correctness():
    log("=" * 60)
    log("Test 2: M=2 kernel correctness")
    log("=" * 60)

    torch.manual_seed(42)
    test_cases = [
        (4096, 4096, True, "att_c2c M=2 use4=True"),
        (4096, 4096, False, "att_c2c M=2 use4=False"),
        (16384, 4096, True, "ffn_key M=2 use4=True"),
        (128, 8, True, "small M=2 use4=True"),
        (128, 8, False, "small M=2 use4=False"),
    ]

    all_pass = True
    for N, K, use4, desc in test_cases:
        w_fp16 = (torch.randn(N, K, device="cuda") * 0.02).to(torch.float16)
        x_fp16 = torch.randn(2, K, device="cuda", dtype=torch.float16) * 0.5
        w_int8, scale = quantize_per_channel_symmetric(w_fp16)

        try:
            # M=2 dispatch 参数选择（和 Python dispatch 一致）
            if use4:
                threads = 64  # att_c2c M=2 use4=True → <64, 2, True>
            else:
                threads = 128  # M=2 use4=False → <128, 2, False>
            y_int8 = torch.ops.rwkv7_int8_ops.linear_int8_orig_rows_exact_f16(
                x_fp16.contiguous(), w_int8.contiguous(), scale.contiguous(),
                threads, 2, use4)
        except Exception as e:
            log(f"  FAIL [{desc}]: kernel error: {e}")
            all_pass = False
            continue

        # PyTorch reference
        y_ref = (x_fp16 @ w_fp16.t()).to(torch.float16)

        cs = cos_sim(y_int8, y_ref)
        mre = max_rel_err(y_int8, y_ref)
        status = "PASS" if cs >= 0.966 else "FAIL"
        if status == "FAIL":
            all_pass = False
        log(f"  {status} [{desc}]: CosSim={cs:.6f} MaxRelErr={mre:.6f}")

    return all_pass

# ═══════════════════════════════════════════════════════════════
# Test 3: FP16 regression — original path still works
# ═══════════════════════════════════════════════════════════════

def test_fp16_regression():
    log("=" * 60)
    log("Test 3: FP16 regression (original path)")
    log("=" * 60)

    torch.manual_seed(42)
    K, N = 4096, 4096
    w_fp16 = (torch.randn(N, K, device="cuda") * 0.02).to(torch.float16)
    x_fp16 = torch.randn(1, K, device="cuda", dtype=torch.float16) * 0.5

    # 原版 kernel
    try:
        y_v3a = torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
            x_fp16.contiguous(), w_fp16.contiguous(), 128, 2, True)
        log(f"  PASS: v3a linear_orig_rows_exact_f16 works, output shape={y_v3a.shape}")
        return True
    except Exception as e:
        log(f"  FAIL: v3a kernel error: {e}")
        return False

# ═══════════════════════════════════════════════════════════════
# Test 4: M>1 dequant fallback
# ═══════════════════════════════════════════════════════════════

def test_m_gt2_dequant():
    log("=" * 60)
    log("Test 4: M>2 dequant fallback (linear function)")
    log("=" * 60)

    torch.manual_seed(42)
    N, K = 4096, 4096
    w_fp16 = (torch.randn(N, K, device="cuda") * 0.02).to(torch.float16)
    w_int8, scale = quantize_per_channel_symmetric(w_fp16)

    all_pass = True
    for M in [3, 4, 8, 32]:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.5
        # 模拟 linear() 中的 dequant fallback
        w_dequant = (w_int8.float() * scale.unsqueeze(1).float()).to(torch.float16)
        w_for_linear = w_dequant.t().contiguous()  # [K, N]
        y = torch.ops.rwkv7_v3a_ops.linear_f16(x.contiguous(), w_for_linear)

        # reference
        y_ref = (x @ w_fp16.t()).to(torch.float16)
        cs = cos_sim(y, y_ref)
        status = "PASS" if cs >= 0.966 else "FAIL"
        if status == "FAIL":
            all_pass = False
        log(f"  {status} [M={M}]: CosSim={cs:.6f}")

    return all_pass

# ═══════════════════════════════════════════════════════════════
# Test 5: Speed benchmark
# ═══════════════════════════════════════════════════════════════

def test_speed():
    log("=" * 60)
    log("Test 5: Speed benchmark")
    log("=" * 60)

    torch.manual_seed(42)
    shapes = [
        (4096, 4096, False, "att_c2c"),
        (4096, 4096, True, "att_c2c use4"),
        (16384, 4096, False, "ffn_key"),
        (65536, 4096, True, "head"),
    ]

    for N, K, use4, desc in shapes:
        w_fp16 = (torch.randn(N, K, device="cuda") * 0.02).to(torch.float16)
        x = torch.randn(1, K, device="cuda", dtype=torch.float16) * 0.5
        w_int8, scale = quantize_per_channel_symmetric(w_fp16)

        # warmup
        for _ in range(10):
            torch.ops.rwkv7_int8_ops.linear_int8_orig_rows_exact_f16(
                x, w_int8, scale, 128, 2, use4)
        torch.cuda.synchronize()

        # int8 timing
        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.ops.rwkv7_int8_ops.linear_int8_orig_rows_exact_f16(
                x, w_int8, scale, 128, 2, use4)
        torch.cuda.synchronize()
        t_int8 = (time.perf_counter() - t0) / iters * 1000

        # fp16 timing
        for _ in range(10):
            torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                x, w_fp16, 128, 2, use4)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
                x, w_fp16, 128, 2, use4)
        torch.cuda.synchronize()
        t_fp16 = (time.perf_counter() - t0) / iters * 1000

        log(f"  {desc}: int8={t_int8:.3f}ms fp16={t_fp16:.3f}ms ratio={t_int8/t_fp16:.2f}x")

# ═══════════════════════════════════════════════════════════════
# Test 6: Zero weight edge case
# ═══════════════════════════════════════════════════════════════

def test_zero_weight():
    log("=" * 60)
    log("Test 6: Zero weight edge case")
    log("=" * 60)

    N, K = 128, 256
    w_fp16 = torch.zeros(N, K, device="cuda", dtype=torch.float16)
    x = torch.randn(1, K, device="cuda", dtype=torch.float16) * 0.5
    w_int8, scale = quantize_per_channel_symmetric(w_fp16)

    y = torch.ops.rwkv7_int8_ops.linear_int8_orig_rows_exact_f16(
        x, w_int8, scale, 128, 2, True)

    # output should be all zeros
    max_val = y.abs().max().item()
    log(f"  max output abs value: {max_val:.10f} (should be ~0)")
    return max_val < 1e-5

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log("Starting v5 INT8 kernel tests...")
    log(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

    # Clear cache first
    cache_dir = os.path.expanduser("~/.cache/torch_extensions")
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        log(f"Cleared torch extensions cache at {cache_dir}")

    load_int8_ext()
    load_v3a_ext()

    results = []
    results.append(("M=1 correctness", test_m1_correctness()))
    results.append(("M=2 correctness", test_m2_correctness()))
    results.append(("FP16 regression", test_fp16_regression()))
    results.append(("M>2 dequant", test_m_gt2_dequant()))
    results.append(("Zero weight", test_zero_weight()))

    # Speed benchmark (not a pass/fail test)
    test_speed()

    log("=" * 60)
    log("Summary")
    log("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        log(f"  {status}: {name}")

    if all_pass:
        log("ALL TESTS PASSED")
    else:
        log("SOME TESTS FAILED")
        sys.exit(1)
