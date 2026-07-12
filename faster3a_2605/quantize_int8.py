#!/usr/bin/env python3
"""
RWKV-7 v3a 离线 INT8 量化工具

将 FP16 模型量化为 INT8，输出新的 .pth 文件。
推理引擎加载量化后的模型时，根据权重 dtype 自动选择 INT8 或 FP16 kernel。

量化策略：
  - per-channel symmetric INT8（scale = max_abs(row) / 127）
  - 仅量化 orig 组大矩阵（att.r/k/v/o, ffn.key, head）
  - ffn.value.weight 不量化（sparse cmix kernel 硬编码 fp16）
  - 低秩权重不量化（收益小）
  - Embedding / LN / r_k 不量化

内存优化：
  - 用 mmap 加载原始模型（不占内存）
  - 创建全新 target 字典，逐个 .cpu() 复制张量（解除 mmap 映射）
  - torch.save 只写新数据，不复制原始文件

用法：
  python3 quantize_int8.py --model model.pth --out model-int8.pth
  python3 quantize_int8.py --model model.pth --out model-int8.pth --verify
"""

import argparse
import os
import sys
import time

import torch

# 量化的权重后缀（全部是 orig 组）
INT8_KEYS = (
    ".att.receptance.weight",
    ".att.key.weight",
    ".att.value.weight",
    ".att.output.weight",
    ".ffn.key.weight",
    "head.weight",
)


def quantize_weight_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """per-channel symmetric INT8 量化。

    Args:
        w: [N, K] fp16/fp32 权重
    Returns:
        w_int8: [N, K] int8
        scale:  [N] fp16
    """
    wf = w.float()  # [N, K]
    max_abs = wf.abs().max(dim=1, keepdim=True).values  # [N, 1]
    max_abs = torch.clamp(max_abs, min=1e-10)
    scale = (max_abs / 127.0).squeeze(1).to(torch.float16)  # [N]
    w_int8 = (wf / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
    return w_int8.contiguous(), scale.contiguous()


def should_quantize(key: str) -> bool:
    """判断该权重是否需要量化。"""
    return any(s in key for s in INT8_KEYS)


def cosine_similarity_fp64(a: torch.Tensor, b: torch.Tensor) -> float:
    """用 float64 计算 cosine similarity，避免大向量的 float32 累积误差。"""
    a64 = a.double()
    b64 = b.double()
    dot = torch.dot(a64, b64)
    norm = a64.norm() * b64.norm()
    return (dot / norm.clamp(min=1e-12)).item()


def quantize_model(input_path: str, output_path: str, verify: bool = False) -> None:
    """量化模型并保存。"""
    t0 = time.perf_counter()
    print(f"[quantize] loading {input_path}")
    # mmap 加载：不占内存，张量按需从磁盘读取
    src = torch.load(input_path, map_location="cpu", mmap=True)
    print(f"[quantize] loaded {len(src)} tensors in {time.perf_counter() - t0:.1f}s")

    quantized_count = 0
    total_params = 0
    quantized_params = 0

    # 创建全新 target 字典：逐个 .cpu() 复制张量，解除 mmap 映射
    # 这样 torch.save 只写新数据，不会把原始 mmap 文件复制进去
    target: dict[str, torch.Tensor] = {}

    t1 = time.perf_counter()
    for key in list(src.keys()):
        value = src[key].squeeze()
        total_params += value.numel()

        if should_quantize(key) and value.dim() == 2:
            w_int8, scale = quantize_weight_int8(value)
            target[key] = w_int8          # int8，已 .contiguous()，是普通张量
            target[key + "_scale"] = scale  # fp16 [N]
            quantized_count += 1
            quantized_params += value.numel()
        else:
            # 未量化的权重：.clone() 复制解除 mmap 映射
            # 注意：.cpu() 对已在 CPU 上的张量不复制，必须用 .clone()
            target[key] = value.clone()

    print(f"[quantize] quantized {quantized_count} weights "
          f"({quantized_params / 1e6:.1f}M / {total_params / 1e6:.1f}M params) "
          f"in {time.perf_counter() - t1:.1f}s")

    # 验证：反量化后和原始权重的 CosSim（float64 避免精度误差）
    if verify:
        print("[quantize] verifying...")
        z_ref = torch.load(input_path, map_location="cpu", mmap=True)
        max_diff = 0.0
        for key in list(target.keys()):
            if key.endswith("_scale"):
                continue
            if target[key].dtype != torch.int8:
                continue
            ref = z_ref[key].squeeze().float()
            w_int8 = target[key]
            scale = target[key + "_scale"]
            deq = w_int8.float() * scale.unsqueeze(1).float()
            cos = cosine_similarity_fp64(ref.flatten(), deq.flatten())
            max_diff = max(max_diff, 1.0 - cos)
            print(f"  {key}: CosSim={cos:.6f}")
        print(f"[quantize] max error (1-cossim): {max_diff:.6f}")

    # 保存
    t2 = time.perf_counter()
    print(f"[quantize] saving to {output_path}")
    torch.save(target, output_path)
    file_size = os.path.getsize(output_path) / 1e9
    print(f"[quantize] saved in {time.perf_counter() - t2:.1f}s, file size: {file_size:.2f} GB")
    print(f"[quantize] done in {time.perf_counter() - t0:.1f}s total")


def main():
    parser = argparse.ArgumentParser(description="RWKV-7 v3a 离线 INT8 量化工具")
    parser.add_argument("--model", required=True, help="输入 FP16 模型路径")
    parser.add_argument("--out", required=True, help="输出 INT8 模型路径")
    parser.add_argument("--verify", action="store_true", help="量化后验证 CosSim")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[error] model not found: {args.model}")
        sys.exit(1)

    quantize_model(args.model, args.out, args.verify)


if __name__ == "__main__":
    main()
