#!/usr/bin/env python3
"""v4 dequant kernel 精度排查"""
import sys, torch
sys.path.insert(0, "/home/njzy/Albatross/faster3a_2605")
import rwkv7_fast_v3a as v3a
v3a.load_extensions("fp16")

K, N = 4096, 4096
x = torch.randn(1, K, dtype=torch.float16, device="cuda") * 0.5
w = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.1

# Quantize
wf = w.float()
max_abs = wf.abs().max(dim=0, keepdim=True).values.clamp(min=1e-10)
wi = (wf / max_abs * 127).round().clamp(-128, 127).to(torch.int8)
ws = (max_abs / 127).to(torch.float16).squeeze()

# Reference: fp16 matmul (the "ground truth")
y_ref = (x.float() @ wf).to(torch.float16)

# Dequant kernel
w_t = wi.t().contiguous()
y_deq = torch.ops.rwkv7_int8_ops.linear_int8_dequant_row1(x, w_t, ws)

# PyTorch dequant: simulate what kernel does
w_t_f32 = w_t.float()
ws_f32 = ws.float().unsqueeze(0).t()  # [N, 1]
w_deq = (w_t_f32 * ws_f32).to(torch.float16)  # [N, K] dequantized
w_deq_kn = w_deq.t()  # [K, N] for matmul
y_py_deq = (x.float() @ w_deq_kn.float()).to(torch.float16)

print(f"CUDA dequant vs fp16 ref:  max_diff={(y_deq.float() - y_ref.float()).abs().max().item():.4f}")
print(f"PyTorch dequant vs fp16 ref: max_diff={(y_py_deq.float() - y_ref.float()).abs().max().item():.4f}")
print(f"CUDA dequant vs Py dequant: max_diff={(y_deq.float() - y_py_deq.float()).abs().max().item():.4f}")
