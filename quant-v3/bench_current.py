#!/usr/bin/env python3
"""quant-v3 测试：M=1 优化 INT8 kernel vs 原版 fp16"""
import sys, time, torch
sys.path.insert(0, "/home/njzy/Albatross/faster3a_2605")
import rwkv7_fast_v3a as v3a
from quant.int8_linear import linear_int8_f16 as linear_int8

v3a.load_extensions(v3a.WKV_MODE)

def bench(label, K, N, reps=500):
    x = torch.randn(1, K, dtype=torch.float16, device="cuda")
    w = torch.randn(K, N, dtype=torch.float16, device="cuda")
    wa = w.abs().max(dim=0, keepdim=True).values.clamp(min=1e-10)
    wi = (w.float() / wa * 127).round().clamp(-128, 127).to(torch.int8)
    ws = (wa / 127).to(torch.float16)

    for _ in range(5): linear_int8(x, wi, ws)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps): linear_int8(x, wi, ws)
    torch.cuda.synchronize()
    ms = (time.perf_counter()-t0)/reps*1000

    print(f"{label:<25} {ms:>7.2f}ms")

bench("INT8 4096x4096 (att)", 4096, 4096)
bench("INT8 16384x4096 (ffn_k)", 16384, 4096)
bench("INT8 4096x16384 (ffn_v)", 4096, 16384)
bench("INT8 4096x65536 (head)", 4096, 65536)
