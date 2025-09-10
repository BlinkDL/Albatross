# Albatross
efficient RWKV inference engine

## Result @ 250909

Now with batch inference. 7B fp16 bsz 320 = 5848 token/s decoding (const speed & vram because it's RNN) on 5090. I think 10000 token/s is achievable (full fp16 precision).

## Result @ 250904

Baseline performance for RWKV-7 7.2B bsz=1 @ RTX5090, simply abysmal lol

Let me know if you can find simple methods (such as tuning torch.compile etc.) to improve these a bit
```
Token/s = 75.1 (forward), 73.76 (full) || Bandwidth = 1041.2 GB/s || 3.722s

CTX_LEN 512 : avg loss 1.6548 || prefill 9163 token/s = 127.03 TFLOPS
CTX_LEN 1024 : avg loss 1.5689 || prefill 9742 token/s = 135.06 TFLOPS
CTX_LEN 2048 : avg loss 1.5141 || prefill 10081 token/s = 139.76 TFLOPS
CTX_LEN 4096 : avg loss 1.4824 || prefill 10427 token/s = 144.55 TFLOPS
```
