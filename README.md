# Albatross : efficient RWKV inference engine

UPDATE: faster3a_2607 is faster than faster3a_2605 (only tuned for 7B).
```
RESULT B=1 T=1 iters=3 p10_ms=6.8555 p50_ms=6.8585 p90_ms=6.9575 tok_s_p50=145.80
RESULT B=1 T=2 iters=3 p10_ms=7.1599 p50_ms=7.1634 p90_ms=7.2747 tok_s_p50=279.20
RESULT B=1 T=4 iters=3 p10_ms=7.8409 p50_ms=7.8423 p90_ms=7.9429 tok_s_p50=510.05
RESULT B=1 T=8 iters=3 p10_ms=8.4995 p50_ms=8.5142 p90_ms=8.5864 tok_s_p50=939.60
RESULT B=1 T=16 iters=3 p10_ms=9.1115 p50_ms=9.1417 p90_ms=9.2668 tok_s_p50=1750.22
RESULT B=1 T=32 iters=3 p10_ms=11.1366 p50_ms=11.1400 p90_ms=11.4407 tok_s_p50=2872.53
RESULT B=1 T=64 iters=3 p10_ms=11.7409 p50_ms=11.7417 p90_ms=12.0255 tok_s_p50=5450.68
RESULT B=1 T=128 iters=3 p10_ms=13.4755 p50_ms=13.4777 p90_ms=13.5614 tok_s_p50=9497.19
RESULT B=1 T=256 iters=3 p10_ms=18.0144 p50_ms=18.0292 p90_ms=18.1396 tok_s_p50=14199.20
RESULT B=2 T=1 iters=3 p10_ms=7.1551 p50_ms=7.1580 p90_ms=7.3194 tok_s_p50=279.41
RESULT B=4 T=1 iters=3 p10_ms=7.8659 p50_ms=7.8844 p90_ms=7.9746 tok_s_p50=507.33
RESULT B=8 T=1 iters=3 p10_ms=8.5023 p50_ms=8.5271 p90_ms=8.6361 tok_s_p50=938.19
RESULT B=16 T=1 iters=3 p10_ms=9.1567 p50_ms=9.1705 p90_ms=9.4360 tok_s_p50=1744.72
RESULT B=32 T=1 iters=3 p10_ms=11.1109 p50_ms=11.1118 p90_ms=11.3531 tok_s_p50=2879.83
RESULT B=64 T=1 iters=3 p10_ms=11.6690 p50_ms=11.6721 p90_ms=11.8782 tok_s_p50=5483.15
RESULT B=128 T=1 iters=3 p10_ms=13.4692 p50_ms=13.4731 p90_ms=13.6302 tok_s_p50=9500.42
RESULT B=256 T=1 iters=3 p10_ms=19.4906 p50_ms=19.4907 p90_ms=19.6106 tok_s_p50=13134.48
RESULT B=2 T=2 iters=3 p10_ms=7.8192 p50_ms=7.8250 p90_ms=7.9173 tok_s_p50=511.18
RESULT B=4 T=4 iters=3 p10_ms=8.9666 p50_ms=8.9817 p90_ms=9.1138 tok_s_p50=1781.41
RESULT B=8 T=8 iters=3 p10_ms=11.0728 p50_ms=11.0763 p90_ms=11.3068 tok_s_p50=5778.09
RESULT B=16 T=16 iters=3 p10_ms=14.7504 p50_ms=14.7602 p90_ms=14.8634 tok_s_p50=17343.98
```

UPDATE: faster3b_2607 for MegaKernel B1T1 7B experiment (currently 3~5% faster, 155+ tps on 5090).

UPDATE: faster3a_2605 (please use it as reference) is up to 40% faster than faster3_2605 for small B/T (for better performance, tune linear_orig_layout for your GPU).
```
RESULT B=1 T=1 iters=3 p10_ms=6.9425 p50_ms=6.9427 p90_ms=7.1073 tok_s_p50=144.04
RESULT B=1 T=2 iters=3 p10_ms=7.2224 p50_ms=7.2231 p90_ms=7.3045 tok_s_p50=276.89
RESULT B=1 T=4 iters=3 p10_ms=7.8479 p50_ms=7.8638 p90_ms=8.0480 tok_s_p50=508.66
RESULT B=1 T=8 iters=3 p10_ms=8.9945 p50_ms=8.9973 p90_ms=9.0790 tok_s_p50=889.15
RESULT B=1 T=16 iters=3 p10_ms=9.2388 p50_ms=9.2642 p90_ms=9.3825 tok_s_p50=1727.09
RESULT B=1 T=32 iters=3 p10_ms=11.1926 p50_ms=11.1940 p90_ms=11.4933 tok_s_p50=2858.66
RESULT B=1 T=64 iters=3 p10_ms=11.6656 p50_ms=11.6670 p90_ms=11.9468 tok_s_p50=5485.54
RESULT B=1 T=128 iters=3 p10_ms=13.4997 p50_ms=13.5012 p90_ms=13.6163 tok_s_p50=9480.67
RESULT B=1 T=256 iters=3 p10_ms=18.2705 p50_ms=18.2778 p90_ms=18.3811 tok_s_p50=14006.07
RESULT B=2 T=1 iters=3 p10_ms=7.2577 p50_ms=7.2684 p90_ms=7.3323 tok_s_p50=275.16
RESULT B=4 T=1 iters=3 p10_ms=7.9306 p50_ms=7.9442 p90_ms=8.0348 tok_s_p50=503.51
RESULT B=8 T=1 iters=3 p10_ms=8.7188 p50_ms=8.7593 p90_ms=8.9117 tok_s_p50=913.32
RESULT B=16 T=1 iters=3 p10_ms=9.3525 p50_ms=9.3743 p90_ms=9.6280 tok_s_p50=1706.79
RESULT B=32 T=1 iters=3 p10_ms=11.2196 p50_ms=11.2238 p90_ms=11.4337 tok_s_p50=2851.07
RESULT B=64 T=1 iters=3 p10_ms=11.6686 p50_ms=11.6814 p90_ms=11.8833 tok_s_p50=5478.79
RESULT B=128 T=1 iters=3 p10_ms=13.6054 p50_ms=13.6102 p90_ms=13.7000 tok_s_p50=9404.68
RESULT B=256 T=1 iters=3 p10_ms=19.4996 p50_ms=19.5026 p90_ms=19.6272 tok_s_p50=13126.46
RESULT B=2 T=2 iters=3 p10_ms=7.8615 p50_ms=7.8702 p90_ms=7.9935 tok_s_p50=508.25
RESULT B=4 T=4 iters=3 p10_ms=9.1181 p50_ms=9.1330 p90_ms=9.2556 tok_s_p50=1751.89
RESULT B=8 T=8 iters=3 p10_ms=11.0723 p50_ms=11.0758 p90_ms=11.2748 tok_s_p50=5778.38
RESULT B=16 T=16 iters=3 p10_ms=14.8019 p50_ms=14.8049 p90_ms=14.8631 tok_s_p50=17291.61
```

UPDATE: faster4_2605_cpp (faster for some BnTn, slower for some BnTn) as standalone (no libtorch, no python) C++ inference (for better performance, tune linear_orig_layout_launch for your GPU)

UPDATE: faster3_2605 can reach 17000+ tps prefill (B1T1024), 15000+ tps decode (B1024T1), 21000+ tps batch prefill (B32T32), on single 5090.

---

Demo: enter faster2_251201 (slower than v3a and v4) and run benchmark.py (fastest decode) and demo3.py (fastest batch decode) and demo4.py (write 120 webpages in parallel).

Note: demo3.py has efficient standalone Python GUI and you can simply run it on your GPU computer.

While for demo2.py, you have to SSH to the GPU computer to run demo2.py in a SSH session, such that the GPU won't be affected by slow terminal rendering.

---

## Old Readme

Please check this first: https://github.com/BlinkDL/Albatross/blob/main/benchmark.py

Faster fwd & bwd CUDA kernels: https://github.com/BlinkDL/RWKV-CUDA/tree/main/rwkv7_fast_fused

Full backend: https://github.com/RWKV-Vibe/rwkv_lightning

Fast sampling: https://github.com/Triang-jyed-driung/Rapid-Sampling

## Result @ 251201

145+ token/s RWKV-7 7.2B fp16 bsz1 @ RTX5090

11289 token/s RWKV-7 7.2B fp16 bsz1 prefill @ RTX5090

Code: https://github.com/Triang-jyed-driung/Albatross/tree/fp16

(enable torch.compile in https://github.com/Triang-jyed-driung/Albatross/blob/fp16/reference/rwkv7.py)

## Result @ 251103

10250+ token/s RWKV-7 7.2B fp16 bsz960 @ RTX5090

9650+ token/s RWKV-7 7.2B fp16 bsz320 @ RTX5090

123+ token/s RWKV-7 7.2B fp16 bsz1 @ RTX5090 with CUDAGraph and sparse FFN (lossless)

Code: https://github.com/BlinkDL/Albatross/tree/main/faster_251101

## Result @ 251007

1.3x 7B decoding and 5x 0.1B decoding, with CUDAGraph.

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
