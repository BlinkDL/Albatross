#!/usr/bin/env python3
"""v4 微基准 + 正确性测试（新 dequant kernel vs PyTorch fallback）"""
import sys, time, torch
sys.path.insert(0, "/home/njzy/Albatross/faster3a_2605")
import rwkv7_fast_v3a as v3a

v3a.load_extensions("fp16")

def bench_dequant(K, N, label, reps=300):
    """测试 dequant kernel (fp16×int8, M=1)"""
    x = torch.randn(1, K, dtype=torch.float16, device="cuda")
    w = torch.randn(K, N, dtype=torch.float16, device="cuda")
    # 离线量化
    wf = w.float()
    max_abs = wf.abs().max(dim=0, keepdim=True).values.clamp(min=1e-10)
    wi = (wf / max_abs * 127).round().clamp(-128, 127).to(torch.int8)
    ws = (max_abs / 127).to(torch.float16).squeeze()

    # 正确的 PyTorch 结果
    y_ref = (x.float() @ wf).to(torch.float16)

    # 测试新 dequant kernel
    w_t = wi.t().contiguous()
    for _ in range(5):
        torch.ops.rwkv7_int8_ops.linear_int8_dequant_row1(x, w_t, ws)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        y = torch.ops.rwkv7_int8_ops.linear_int8_dequant_row1(x, w_t, ws)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / reps * 1000

    # 正确性
    diff = (y.float() - y_ref.float()).abs().max().item()
    print(f"{label:<25} {ms:>7.2f}ms  max_diff={diff:.5f}")

bench_dequant(4096, 4096, "4096x4096 (att)")
bench_dequant(16384, 4096, "16384x4096 (ffn_k)")
bench_dequant(4096, 16384, "4096x16384 (ffn_v)")
bench_dequant(4096, 65536, "4096x65536 (head)")
