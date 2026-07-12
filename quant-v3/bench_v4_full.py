#!/usr/bin/env python3
"""v4 完整推理速度测试"""
import torch, time, sys
sys.path.insert(0, "/home/njzy/Albatross/faster3a_2605")
import rwkv7_fast_v3a as v3a
v3a.MODEL_PATH = "/home/njzy/model/rwkv7-g1g-7.2b-20260523-ctx8192.pth"
v3a.WKV_MODE = "fp32io16"
v3a.QUANT_MODE = "on"
v3a.load_extensions(v3a.WKV_MODE)
model = v3a.RWKV7()
state = model.zero_state(1)

# Warmup
tok = torch.tensor([[0]], dtype=torch.long)
for _ in range(3):
    logits = model.forward(tok, state)
    tok = logits[0,-1].argmax().unsqueeze(0).unsqueeze(0)

# Benchmark 50 tokens
tok = torch.tensor([[0]], dtype=torch.long)
state = model.zero_state(1)
t0 = time.perf_counter()
for i in range(50):
    logits = model.forward(tok, state)
    tok = logits[0,-1].argmax().unsqueeze(0).unsqueeze(0)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"50 tokens: {dt/50*1000:.1f}ms/tok, {50/dt:.1f} tok/s")
print(f"GPU: {v3a.cuda_mem()}")
