#!/usr/bin/env python3
"""
RWKV-7 v3a 低内存 NVFP4 量化工具

使用 torch.load(mmap=True) 实现逐张量懒加载：
  - mmap 不会一次性加载整个模型到内存
  - 访问 src[key] 时才触发缺页中断，加载对应数据页
  - del 后 OS 可回收页面
  - 峰值内存 ≈ 输出 dict + 1个张量

对于大于物理内存的模型（如 7.2B FP16=14GB > 9.7GB RAM），
需要先设置 vm.overcommit_memory=1：
  wsl -u root -e bash -c "sysctl -w vm.overcommit_memory=1"

用法：
  python3 quantize_stream.py --model model.pth --out model-nvfp4.pth
  python3 quantize_stream.py --model model.pth --out model-nvfp4.pth --verify
"""

import argparse
import gc
import os
import sys
import time

import torch

# ── 量化参数（与 quantize_nf4.py 一致）──
NF4_KEYS = (
    ".att.receptance.weight",
    ".att.key.weight",
    ".att.value.weight",
    ".att.output.weight",
    ".ffn.key.weight",
    ".ffn.value.weight",
)
BLOCK_SIZE = 16
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


# ── 内存追踪 ──
_peak = 0

def _mem_usage_mb() -> float:
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except:
        pass
    return 0

def _peak_mem_mb() -> float:
    global _peak
    cur = _mem_usage_mb()
    if cur > _peak:
        _peak = cur
    return _peak


# ── 量化核心（与 quantize_nf4.py 一致）──

def quantize_weight_nf4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """per-block NVFP4 量化 (E2M1 + E4M3 block scale + FP32 tensor scale)。"""
    N, K = w.shape
    assert K % BLOCK_SIZE == 0
    wf = w.float()
    blocks = wf.reshape(N, K // BLOCK_SIZE, BLOCK_SIZE)
    max_abs = blocks.abs().amax(dim=-1, keepdim=True)
    max_abs = torch.clamp(max_abs, min=1e-10)
    b_scale_fp32 = (max_abs / 6.0).squeeze(-1)
    t_scale = b_scale_fp32.max().item()
    t_scale = max(t_scale, 1e-10)
    b_scale_norm = b_scale_fp32 / t_scale
    b_scale = b_scale_norm.to(torch.float8_e4m3fn)
    normalized = blocks / max_abs
    codes = _e2m1_quantize(normalized)
    codes = codes.reshape(N, K)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.to(torch.uint8).contiguous(), b_scale.contiguous(), t_scale


def _e2m1_quantize(normalized: torch.Tensor) -> torch.Tensor:
    scaled = normalized * 6.0
    sign_bit = (scaled < 0).to(torch.uint8)
    abs_w = scaled.abs()
    idx = torch.bucketize(abs_w, E2M1_BOUNDS)
    codes = idx.to(torch.uint8) | (sign_bit << 3)
    return codes


def _e2m1_dequant(w_nf4, b_scale, t_scale=1.0):
    N, K_half = w_nf4.shape
    K = K_half * 2
    low = (w_nf4 & 0x0F).to(torch.int32)
    high = (w_nf4 >> 4).to(torch.int32)
    codes = torch.stack([low, high], dim=-1).reshape(N, K)
    e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    values = e2m1_lut[codes]
    bs = b_scale.float().repeat_interleave(BLOCK_SIZE, dim=1) * t_scale
    return values * bs


def should_quantize(key: str) -> bool:
    return any(s in key for s in NF4_KEYS)


def cosine_similarity_fp64(a, b) -> float:
    a64 = a.double()
    b64 = b.double()
    dot = torch.dot(a64, b64)
    norm = a64.norm() * b64.norm()
    return (dot / norm.clamp(min=1e-12)).item()


# ── mmap 流式量化 ──

def quantize_model_streaming(input_path: str, output_path: str, verify: bool = False) -> None:
    """mmap 懒加载 + 逐张量量化。

    mmap 模式下 src[key] 返回的是虚拟内存映射的张量，
    只有实际访问时才会触发缺页中断加载到物理内存。
    因此峰值内存 ≈ 输出 dict + 当前正在处理的 1 个张量。
    """
    t0 = time.perf_counter()
    print(f"[mmap] loading (mmap=lazy) {input_path}", flush=True)

    # mmap=True: 不实际加载到内存，创建虚拟内存映射
    # 对于大文件需要 vm.overcommit_memory=1
    try:
        src = torch.load(input_path, map_location="cpu", mmap=True)
    except RuntimeError as e:
        if "mmap" in str(e) or "Cannot allocate memory" in str(e):
            print(f"[mmap] mmap failed: {e}", flush=True)
            print(f"[mmap] for large models, run: sudo sysctl -w vm.overcommit_memory=1", flush=True)
            print(f"[mmap] falling back to direct load (higher memory)...", flush=True)
            src = torch.load(input_path, map_location="cpu")
        else:
            raise

    keys = list(src.keys())
    total_keys = len(keys)
    print(f"[mmap] loaded {total_keys} tensors (lazy)", flush=True)
    print(f"[mmap] RAM after load: {_mem_usage_mb():.0f} MB", flush=True)

    out_dict = {}
    quantized_count = 0
    total_params = 0
    quantized_params = 0

    t1 = time.perf_counter()
    print(f"[mmap] starting quantization of {total_keys} tensors", flush=True)

    for i, key in enumerate(keys):
        # 每 50 个张量打印进度
        if i % 50 == 0:
            print(f"[mmap] [{i}/{total_keys}] processing {key} ...", flush=True)

        # 从 mmap dict 中取出张量（触发缺页中断，加载到物理内存）
        tensor = src[key]
        if tensor.dim() == 0:
            # 标量直接 clone
            out_dict[key] = tensor.clone()
            del tensor
            continue

        tensor = tensor.squeeze()
        total_params += tensor.numel()

        if should_quantize(key) and tensor.dim() == 2:
            # 量化
            w_nf4, b_scale, t_scale = quantize_weight_nf4(tensor)

            if verify:
                deq = _e2m1_dequant(w_nf4, b_scale, t_scale)
                cos = cosine_similarity_fp64(tensor.float().flatten(), deq.flatten())
                print(f"  {key}: CosSim={cos:.6f} t_scale={t_scale:.6f}", flush=True)
                del deq

            out_dict[key] = w_nf4
            out_dict[key + ".nf4_b_scale"] = b_scale
            out_dict[key + ".nvfp4_t_scale"] = torch.tensor(t_scale, dtype=torch.float32)
            quantized_count += 1
            quantized_params += w_nf4.numel() * 2
        else:
            # 不量化，clone 后断开 mmap 引用
            out_dict[key] = tensor.clone()

        # 释放原始 mmap 张量引用，OS 可回收页面
        del tensor
        # 从 src dict 中删除已处理的条目，释放 mmap 引用
        del src[key]

        if (i + 1) % 20 == 0:
            gc.collect()

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t1
            print(f"  [{i+1}/{total_keys}] {quantized_count} quantized, "
                  f"{elapsed:.1f}s elapsed, "
                  f"RAM: {_mem_usage_mb():.0f} MB (peak: {_peak_mem_mb():.0f} MB)",
                  flush=True)

    # 释放 mmap 源
    del src
    gc.collect()

    print(f"\n[mmap] quantized {quantized_count} weights "
          f"({quantized_params / 1e6:.1f}M / {total_params / 1e6:.1f}M params) "
          f"in {time.perf_counter() - t1:.1f}s", flush=True)
    print(f"[mmap] RAM before save: {_mem_usage_mb():.0f} MB (peak: {_peak_mem_mb():.0f} MB)",
          flush=True)

    # 保存
    t2 = time.perf_counter()
    print(f"[mmap] saving to {output_path} ({len(out_dict)} tensors)...", flush=True)
    torch.save(out_dict, output_path)
    file_size = os.path.getsize(output_path) / 1e9
    print(f"[mmap] saved in {time.perf_counter() - t2:.1f}s, file size: {file_size:.2f} GB",
          flush=True)

    # repack: mmap 保存的文件会有对齐填充，需要重新打包
    repack_path = output_path.replace(".pth", "-repack.pth")
    t3 = time.perf_counter()
    print(f"[mmap] repacking (remove mmap padding)...", flush=True)
    z = torch.load(output_path, map_location="cpu", mmap=True)
    for k in list(z.keys()):
        v = z[k]
        if v.dim() > 0:
            v = v.squeeze()
        z[k] = v.clone()
        del v
    gc.collect()
    torch.save(z, repack_path)
    del z
    gc.collect()
    repack_size = os.path.getsize(repack_path) / 1e9
    print(f"[mmap] repacked in {time.perf_counter() - t3:.1f}s, size: {repack_size:.2f} GB",
          flush=True)

    # 替换原文件
    os.remove(output_path)
    os.rename(repack_path, output_path)
    print(f"[mmap] replaced with repacked file (final: {repack_size:.2f} GB)", flush=True)

    print(f"\n[mmap] done in {time.perf_counter() - t0:.1f}s total", flush=True)
    print(f"[mmap] peak RAM: {_peak_mem_mb():.0f} MB", flush=True)


def main():
    parser = argparse.ArgumentParser(description="RWKV-7 低内存 NVFP4 量化工具 (mmap)")
    parser.add_argument("--model", required=True, help="输入 FP16 模型路径")
    parser.add_argument("--out", required=True, help="输出 NVFP4 模型路径")
    parser.add_argument("--verify", action="store_true", help="量化后验证 CosSim")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[error] model not found: {args.model}")
        sys.exit(1)

    quantize_model_streaming(args.model, args.out, args.verify)


if __name__ == "__main__":
    main()
