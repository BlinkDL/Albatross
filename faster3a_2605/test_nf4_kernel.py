#!/usr/bin/env python3
"""NF4 kernel 正确性测试 — 不需要模型文件，用随机权重测试"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

import torch
from torch.utils.cpp_extension import load

CUDA_DIR = Path(__file__).resolve().parent / "cuda"

def load_nf4_ops():
    cuda_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization"]
    if os.name != "nt":
        cuda_flags += ["-Xptxas", "-O3"]
    load(name="rwkv7_nf4_ops",
         sources=[str(CUDA_DIR / "rwkv7_nf4_ops.cpp"), str(CUDA_DIR / "rwkv7_nf4_ops.cu")],
         is_python_module=False, verbose=False,
         extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags)
    print("[test] nf4_ops loaded")

def cosine_sim(a, b):
    a64 = a.double().flatten()
    b64 = b.double().flatten()
    return (torch.dot(a64, b64) / (a64.norm() * b64.norm()).clamp(min=1e-12)).item()

def test_dequant():
    N, K = 256, 4096
    w = torch.randn(N, K, dtype=torch.float16, device="cuda")
    # Quantize on CPU (offline tool pattern)
    from quantize_nf4 import quantize_weight_nf4, _e2m1_dequant
    w_nf4, b_scale = quantize_weight_nf4(w.cpu())
    w_nf4 = w_nf4.cuda()
    b_scale = b_scale.cuda()

    # CUDA dequant (no transpose)
    deq_cuda = torch.ops.rwkv7_nf4_ops.dequant_nf4_to_f16(w_nf4, b_scale, False)
    # CPU reference
    deq_cpu = _e2m1_dequant(w_nf4.cpu(), b_scale.cpu()).to(torch.float16).cuda()
    cos = cosine_sim(deq_cuda.float(), deq_cpu.float())
    print(f"[test] dequant [N,K]: CosSim={cos:.6f}")
    assert cos > 0.999, f"dequant CosSim too low: {cos}"

    # CUDA dequant (transpose)
    deq_cuda_t = torch.ops.rwkv7_nf4_ops.dequant_nf4_to_f16(w_nf4, b_scale, True)
    deq_cpu_t = _e2m1_dequant(w_nf4.cpu(), b_scale.cpu()).to(torch.float16).t().contiguous().cuda()
    cos_t = cosine_sim(deq_cuda_t.float(), deq_cpu_t.float())
    print(f"[test] dequant [K,N]: CosSim={cos_t:.6f}")
    assert cos_t > 0.999, f"dequant transpose CosSim too low: {cos_t}"
    return w_nf4, b_scale, w

def test_gemv_m1(w_nf4, b_scale, w_ref):
    N, K = w_ref.shape
    x = torch.randn(1, K, dtype=torch.float16, device="cuda")
    # NF4 kernel
    y_nf4 = torch.ops.rwkv7_nf4_ops.linear_nf4_orig_rows_exact_f16(x, w_nf4, b_scale, 128, 2, True)
    # FP16 reference
    y_ref = torch.nn.functional.linear(x, w_ref)
    cos = cosine_sim(y_nf4.float(), y_ref.float())
    print(f"[test] GEMV M=1 (4-wide): CosSim={cos:.6f}")

    y_nf4_2 = torch.ops.rwkv7_nf4_ops.linear_nf4_orig_rows_exact_f16(x, w_nf4, b_scale, 128, 2, False)
    cos2 = cosine_sim(y_nf4_2.float(), y_ref.float())
    print(f"[test] GEMV M=1 (2-wide): CosSim={cos2:.6f}")
    return cos

def test_gemv_m2(w_nf4, b_scale, w_ref):
    N, K = w_ref.shape
    x = torch.randn(2, K, dtype=torch.float16, device="cuda")
    y_nf4 = torch.ops.rwkv7_nf4_ops.linear_nf4_orig_rows_exact_f16(x, w_nf4, b_scale, 64, 2, True)
    y_ref = torch.nn.functional.linear(x, w_ref)
    cos = cosine_sim(y_nf4.float(), y_ref.float())
    print(f"[test] GEMV M=2 (4-wide): CosSim={cos:.6f}")
    return cos

def test_gemm(w_nf4, b_scale, w_ref, M):
    N, K = w_ref.shape
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    y_nf4 = torch.ops.rwkv7_nf4_ops.linear_nf4_orig_rows_f16(x, w_nf4, b_scale, 1, 4)
    y_ref = torch.nn.functional.linear(x, w_ref)
    cos = cosine_sim(y_nf4.float(), y_ref.float())
    print(f"[test] GEMM M={M}: CosSim={cos:.6f}")
    return cos

def bench(w_nf4, b_scale, w_ref):
    N, K = w_ref.shape
    x = torch.randn(1, K, dtype=torch.float16, device="cuda")
    # NF4
    for _ in range(10):
        torch.ops.rwkv7_nf4_ops.linear_nf4_orig_rows_exact_f16(x, w_nf4, b_scale, 128, 2, True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        torch.ops.rwkv7_nf4_ops.linear_nf4_orig_rows_exact_f16(x, w_nf4, b_scale, 128, 2, True)
    torch.cuda.synchronize()
    nf4_ms = (time.perf_counter() - t0) / 100 * 1000

    # FP16 cuBLAS
    for _ in range(10):
        torch.nn.functional.linear(x, w_ref)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        torch.nn.functional.linear(x, w_ref)
    torch.cuda.synchronize()
    fp16_ms = (time.perf_counter() - t0) / 100 * 1000

    print(f"[bench] M=1 NF4: {nf4_ms:.3f}ms  FP16 cuBLAS: {fp16_ms:.3f}ms  ratio: {nf4_ms/fp16_ms:.2f}x")
    print(f"[bench] weight mem: NF4={w_nf4.numel()/1e6:.1f}M  FP16={w_ref.numel()/1e6:.1f}M  (4x compression)")

def main():
    load_nf4_ops()
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    w_nf4, b_scale, w_ref = test_dequant()
    print()
    test_gemv_m1(w_nf4, b_scale, w_ref)
    test_gemv_m2(w_nf4, b_scale, w_ref)
    test_gemm(w_nf4, b_scale, w_ref, 8)
    test_gemm(w_nf4, b_scale, w_ref, 16)
    print()
    bench(w_nf4, b_scale, w_ref)
    print("\nAll tests passed!")

if __name__ == "__main__":
    main()
