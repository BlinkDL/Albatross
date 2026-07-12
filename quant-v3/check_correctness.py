#!/usr/bin/env python3
"""验证 v3 kernel 正确性 vs PyTorch fallback"""
import sys, torch
sys.path.insert(0, "/home/njzy/Albatross/faster3a_2605")
import rwkv7_fast_v3a as v3a
from quant.int8_linear import linear_int8_f16, _int8_t_cache

v3a.load_extensions(v3a.WKV_MODE)

def check(K, N, label):
    x = torch.randn(1, K, dtype=torch.float16, device="cuda")
    w = torch.randn(K, N, dtype=torch.float16, device="cuda")
    wa = w.abs().max(dim=0, keepdim=True).values.clamp(min=1e-10)
    wi = (w.float() / wa * 127).round().clamp(-128, 127).to(torch.int8)
    ws = (wa / 127).to(torch.float16)

    # Force CUDA kernel (set cache first)
    _int8_t_cache.clear()
    y_cuda = linear_int8_f16(x, wi, ws)

    # PyTorch fallback
    xt = x.float()
    max_abs = xt.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-10)
    sx = max_abs / 127.0
    xq = (xt / sx).round().clamp(-128, 127).to(torch.int8)
    acc = (xq.float() @ wi.float()).to(torch.int32)
    y_py = (acc.float() * (sx * ws.unsqueeze(0).float())).to(torch.float16)

    diff = (y_cuda.float() - y_py.float()).abs().max().item()
    print(f"{label:<25} max_diff={diff:.5f}  {'OK' if diff < 0.1 else 'FAIL'}")

check(4096, 4096, "4096x4096 (att)")
check(16384, 4096, "16384x4096 (ffn_k)")
check(4096, 16384, "4096x16384 (ffn_v)")
check(4096, 65536, "4096x65536 (head)")
