#!/usr/bin/env python3
"""
RWKV-7 v3a 离线 NVFP4 量化工具

将 FP16 模型量化为 NVFP4 (E2M1 + E4M3 block scale + FP32 tensor scale)，
输出新的 .pth 文件。
推理引擎加载量化后的模型时，根据权重 dtype (uint8) 自动选择 NVFP4 kernel。

NVFP4 格式 (NVIDIA 官方标准):
  - 权重: E2M1 4-bit float, 值集 {0,±0.5,±1,±1.5,±2,±3,±4,±6}
  - 块缩放: E4M3 FP8, 每 16 个元素共享一个 (torch.float8_e4m3fn)
  - 张量缩放: FP32 标量, 每个权重矩阵一个
  - 反量化: value = e2m1[code] * b_scale_e4m3 * t_scale

量化策略：
  - per-block NVFP4（block_size=16）
  - 量化 orig 组大矩阵（att.r/k/v/o, ffn.key, ffn.value, head）
  - ffn.value.weight 量化后走 NVFP4 cmix_sparse 内核
  - 低秩权重不量化（收益小）
  - Embedding / LN / r_k 不量化

用法：
  python3 quantize_nf4.py --model model.pth --out model-nvfp4.pth
  python3 quantize_nf4.py --model model.pth --out model-nvfp4.pth --verify
"""

import argparse
import gc
import os
import sys
import time

import torch

# 量化的权重后缀（和 INT8 相同，全部是 orig 组）
# head.weight 不量化：LM head 必须保持高精度，否则强化学习会训崩
NF4_KEYS = (
    ".att.receptance.weight",
    ".att.key.weight",
    ".att.value.weight",
    ".att.output.weight",
    ".ffn.key.weight",
    ".ffn.value.weight",
)

BLOCK_SIZE = 16

# E2M1 正数值表（索引 0-7 对应编码 0-7）
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# E2M1 量化边界（相邻值的中点）
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def quantize_weight_nf4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """per-block NVFP4 量化 (E2M1 + E4M3 block scale + FP32 tensor scale)。

    Args:
        w: [N, K] fp16/fp32 权重
    Returns:
        w_nf4:   [N, K/2] uint8 (packed, 2 E2M1 per byte)
        b_scale: [N, K/16] float8_e4m3fn (per-block E4M3 scale)
        t_scale: float32 (per-tensor FP32 scale)
    """
    N, K = w.shape
    assert K % BLOCK_SIZE == 0, f"K={K} must be divisible by BLOCK_SIZE={BLOCK_SIZE}"
    wf = w.float()

    # Reshape to blocks: [N, K/16, 16]
    blocks = wf.reshape(N, K // BLOCK_SIZE, BLOCK_SIZE)

    # Per-block max abs
    max_abs = blocks.abs().amax(dim=-1, keepdim=True)  # [N, K/16, 1]
    max_abs = torch.clamp(max_abs, min=1e-10)

    # Block scale = max_abs / 6.0 (E2M1 max value = 6.0)
    b_scale_fp32 = (max_abs / 6.0).squeeze(-1)  # [N, K/16]

    # Per-tensor FP32 scale = global max of all block scales
    t_scale = b_scale_fp32.max().item()
    t_scale = max(t_scale, 1e-10)

    # Normalize block scale by t_scale, then quantize to E4M3
    b_scale_norm = b_scale_fp32 / t_scale  # [N, K/16], range [0, 1]
    b_scale = b_scale_norm.to(torch.float8_e4m3fn)  # [N, K/16]

    # Normalize weights: blocks / max_abs -> [-1, 1] -> E2M1 encode
    # (E2M1 values are [-6, 6], so dividing by max_abs normalizes to [-1, 1],
    #  and the actual value = e2m1[code] * 6.0 * max_abs = e2m1[code] * b_scale_fp32)
    normalized = blocks / max_abs  # [-1, 1]

    # Quantize to E2M1 codes
    codes = _e2m1_quantize(normalized)  # [N, K/16, 16], uint8 (0-15)

    # Reshape back to [N, K]
    codes = codes.reshape(N, K)

    # Pack 2 codes per byte: even index -> low nibble, odd index -> high nibble
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)  # [N, K/2]

    return packed.to(torch.uint8).contiguous(), b_scale.contiguous(), t_scale


def _e2m1_quantize(normalized: torch.Tensor) -> torch.Tensor:
    """Vectorized E2M1 quantization.

    Args:
        normalized: [...] float, range [-1, 1]
    Returns:
        codes: [...] uint8 (4-bit codes, 0-15)
    """
    # E2M1 values are [0, 0.5, 1, 1.5, 2, 3, 4, 6] for positive,
    # normalized input is [-1, 1], so we scale by 6 to get [-6, 6] for E2M1
    scaled = normalized * 6.0  # [-6, 6]
    sign_bit = (scaled < 0).to(torch.uint8)  # 1=negative
    abs_w = scaled.abs()

    # bucketize: 0->0.0, 1->0.5, 2->1.0, ..., 7->6.0
    idx = torch.bucketize(abs_w, E2M1_BOUNDS)  # 0-7

    # code = idx | (sign_bit << 3)
    codes = idx.to(torch.uint8) | (sign_bit << 3)
    return codes


def _e2m1_dequant(w_nf4: torch.Tensor, b_scale: torch.Tensor, t_scale: float = 1.0) -> torch.Tensor:
    """反量化 NVFP4 -> float32（用于验证）。

    Args:
        w_nf4:   [N, K/2] uint8 (packed)
        b_scale: [N, K/16] float8_e4m3fn
        t_scale: float32 (per-tensor scale)
    Returns:
        deq: [N, K] float32
    """
    N, K_half = w_nf4.shape
    K = K_half * 2

    # Unpack: [N, K/2] -> [N, K]
    low = (w_nf4 & 0x0F).to(torch.int32)
    high = (w_nf4 >> 4).to(torch.int32)
    codes = torch.stack([low, high], dim=-1).reshape(N, K)

    # E2M1 decode via lookup
    e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    values = e2m1_lut[codes]  # [N, K]

    # Apply block scale (E4M3 -> float32) and tensor scale (FP32)
    bs = b_scale.float().repeat_interleave(BLOCK_SIZE, dim=1) * t_scale

    return values * bs


def should_quantize(key: str) -> bool:
    return any(s in key for s in NF4_KEYS)


def cosine_similarity_fp64(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.double()
    b64 = b.double()
    dot = torch.dot(a64, b64)
    norm = a64.norm() * b64.norm()
    return (dot / norm.clamp(min=1e-12)).item()


def quantize_model(input_path: str, output_path: str, verify: bool = False) -> None:
    """量化模型并保存。"""
    t0 = time.perf_counter()
    print(f"[quantize] loading {input_path}")
    # Try mmap first (low memory), fall back to direct load (needs more RAM)
    try:
        src = torch.load(input_path, map_location="cpu", mmap=True)
    except RuntimeError as e:
        if "mmap" in str(e) or "Cannot allocate memory" in str(e):
            print(f"[quantize] mmap failed ({e}), falling back to direct load")
            src = torch.load(input_path, map_location="cpu")
        else:
            raise
    print(f"[quantize] loaded {len(src)} tensors in {time.perf_counter() - t0:.1f}s")

    quantized_count = 0
    total_params = 0
    quantized_params = 0

    t1 = time.perf_counter()
    keys = list(src.keys())
    for key in keys:
        value = src[key]
        if value.dim() == 0:
            continue
        value = value.squeeze()
        total_params += value.numel()

        if should_quantize(key) and value.dim() == 2:
            w_nf4, b_scale, t_scale = quantize_weight_nf4(value)

            if verify:
                deq = _e2m1_dequant(w_nf4, b_scale, t_scale)
                cos = cosine_similarity_fp64(value.float().flatten(), deq.flatten())
                print(f"  {key}: CosSim={cos:.6f} t_scale={t_scale:.6f}")
                del deq

            # in-place 替换
            src[key] = w_nf4
            src[key + ".nf4_b_scale"] = b_scale
            src[key + ".nvfp4_t_scale"] = torch.tensor(t_scale, dtype=torch.float32)
            del value
            quantized_count += 1
            quantized_params += src[key].numel() * 2

            if quantized_count % 20 == 0:
                gc.collect()

    gc.collect()
    print(f"[quantize] quantized {quantized_count} weights "
          f"({quantized_params / 1e6:.1f}M / {total_params / 1e6:.1f}M params) "
          f"in {time.perf_counter() - t1:.1f}s")

    t2 = time.perf_counter()
    print(f"[quantize] saving to {output_path}")
    torch.save(src, output_path)
    file_size = os.path.getsize(output_path) / 1e9
    print(f"[quantize] saved in {time.perf_counter() - t2:.1f}s, file size: {file_size:.2f} GB")
    print(f"[quantize] done in {time.perf_counter() - t0:.1f}s total")
    print(f"[quantize] note: mmap causes file bloat, run repack to fix:")

    # 自动 repack
    clean_path = output_path.replace(".pth", "-clean.pth")
    t3 = time.perf_counter()
    print(f"[repack] loading {output_path} (mmap)")
    z = torch.load(output_path, map_location="cpu", mmap=True)
    for k in list(z.keys()):
        v = z[k]
        if v.dim() > 0:
            v = v.squeeze()
        z[k] = v.clone()
        del v
    gc.collect()
    torch.save(z, clean_path)
    clean_size = os.path.getsize(clean_path) / 1e9
    print(f"[repack] saved {clean_path} in {time.perf_counter() - t3:.1f}s, size: {clean_size:.2f} GB")

    # 用 clean 文件替换
    os.remove(output_path)
    os.rename(clean_path, output_path)
    print(f"[repack] replaced {output_path} (final size: {clean_size:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="RWKV-7 v3a 离线 NVFP4 量化工具")
    parser.add_argument("--model", required=True, help="输入 FP16 模型路径")
    parser.add_argument("--out", required=True, help="输出 NF4 模型路径")
    parser.add_argument("--verify", action="store_true", help="量化后验证 CosSim")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[error] model not found: {args.model}")
        sys.exit(1)

    quantize_model(args.model, args.out, args.verify)


if __name__ == "__main__":
    main()
