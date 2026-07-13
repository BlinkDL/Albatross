#!/usr/bin/env python3
"""Independent fixed-path RWKV7 v3b B1T1 release from 2026-07-13."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
MODEL_PATH = "/dev/shm/rwkv7-g1f-7.2b-20260414-ctx8192.pth"
EVAL_JSON = ROOT.parent / "rwkv7-fast-v2-standalone" / "eval_src2.json"
LN_EPS = 1e-5
HEAD_SIZE = 64
C_SIZE = 4096
DTYPE = torch.float16
LOWRANK_SUFFIXES = (
    "att.w1", "att.w2", "att.a1", "att.a2",
    "att.g1", "att.g2", "att.v1", "att.v2",
)


def cuda_flags() -> list[str]:
    flags = ["-O3", "--use_fast_math"]
    if os.environ.get("RWKV7_CUDA_LINEINFO") == "1":
        flags.append("-lineinfo")
    return flags


def load_ops() -> None:
    load(
        name="rwkv7_mega_ops_260713",
        sources=[
            str(ROOT / "cuda/rwkv7_mega_ops_260713.cpp"),
            str(ROOT / "cuda/rwkv7_mega_ops_260713.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_flags(),
        is_python_module=False,
        verbose=True,
    )


def select_token(args: argparse.Namespace, vocab_size: int) -> tuple[int, str]:
    if args.token is not None:
        return int(args.token) % vocab_size, "arg"
    if args.token_source == "v3a-synthetic":
        token = (int(args.token_index) * 1103515245 + 12345) % vocab_size
        return token, f"v3a_synthetic:index={args.token_index}"
    with open(args.eval_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    tokens = obj["tokens"] if isinstance(obj, dict) else obj
    return int(tokens[args.token_index]) % vocab_size, f"eval_json:{Path(args.eval_json).name}:index={args.token_index}"


def tile_cmix_value_weight(weight: torch.Tensor) -> torch.Tensor:
    ffn, channels = weight.shape
    if ffn % 128 or channels % 256:
        raise ValueError(f"CMIX vtile requires F%128==0 and C%256==0, got {tuple(weight.shape)}")
    return (
        weight.reshape(ffn // 128, 128, channels // 256, 256)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(ffn, channels)
    )


def make_layer_work(z: dict[str, torch.Tensor], layer: int, channels: int, ffn: int) -> dict[str, torch.Tensor]:
    p = f"blocks.{layer}."
    work = {
        name: torch.empty((1, 1, channels), device="cuda", dtype=DTYPE)
        for name in (
            "x_base", "xr", "xw", "xk", "xv", "xa", "xg",
            "new_k", "neg_kk", "kka", "wkv_y", "tail_out",
            "cmix_x", "cmix_mixed",
        )
    }
    work.update({
        "r": torch.empty(channels, device="cuda", dtype=DTYPE),
        "k_raw": torch.empty(channels, device="cuda", dtype=DTYPE),
        "v_base": torch.empty(channels, device="cuda", dtype=DTYPE),
        "att": torch.empty(channels, device="cuda", dtype=DTYPE),
        "cmix_hid": torch.empty(ffn, device="cuda", dtype=DTYPE),
        "cmix_out": torch.empty(channels, device="cuda", dtype=DTYPE),
        "lr_w1": torch.empty((1, z[p + "att.w1.t"].size(0)), device="cuda", dtype=DTYPE),
        "lr_a1": torch.empty((1, z[p + "att.a1.t"].size(0)), device="cuda", dtype=DTYPE),
        "lr_g1": torch.empty((1, z[p + "att.g1.t"].size(0)), device="cuda", dtype=DTYPE),
        "lr_v1": torch.empty((1, z[p + "att.v1.t"].size(0)), device="cuda", dtype=DTYPE),
        # Effective decay multiplies recurrent FP32 state on every token. Keep
        # it in FP32; narrowing this post-nonlinearity value accumulated a
        # measurable loss regression over the 8192-token quality gate.
        "gate_w": torch.empty((1, channels), device="cuda", dtype=torch.float32),
        "gate_a": torch.empty((1, channels), device="cuda", dtype=DTYPE),
        "gate_g": torch.empty((1, channels), device="cuda", dtype=DTYPE),
        "gate_v": torch.empty((1, channels), device="cuda", dtype=DTYPE),
    })
    return work


class RWKV7FastB1T1260713:
    """Fixed C4096/H64 B1T1 decode path; no experimental runtime dispatch."""

    def __init__(self, model_path: str) -> None:
        t0 = time.perf_counter()
        src = torch.load(model_path, map_location="cpu", mmap=True)
        self.H, self.N = [int(x) for x in src["blocks.0.att.r_k"].shape]
        self.C = self.H * self.N
        self.V = int(src["emb.weight"].shape[0])
        self.L = max(int(k.split(".")[1]) for k in src if k.startswith("blocks.")) + 1
        if self.N != HEAD_SIZE or self.C != C_SIZE:
            raise ValueError(f"release requires HxN={self.H}x{HEAD_SIZE}, C={C_SIZE}; got H={self.H} N={self.N} C={self.C}")

        self.z: dict[str, torch.Tensor] = {}
        self.load_weights(src)
        ffn = int(self.z["blocks.0.ffn.key.weight"].size(0))
        self.work = [make_layer_work(self.z, layer, self.C, ffn) for layer in range(self.L)]
        self.final_ln = torch.empty((1, self.C), device="cuda", dtype=DTYPE)
        self.logits = torch.empty(self.V, device="cuda", dtype=DTYPE)
        self.initial_residual = torch.zeros((1, 1, self.C), device="cuda", dtype=DTYPE)
        self.cmix_streams = tuple(torch.cuda.Stream() for _ in range(4))
        torch.cuda.synchronize()
        print(
            f"[260713] ready L={self.L} C={self.C} H={self.H} V={self.V} "
            "executor=body21/c7040/w896 rankout=A16-W8-G4-V16-vtree12-wload8x2 "
            "wkv=halfwarp8-late-pdl ln1=cluster8 cmix=kv4-split4 head=t256-k2-mode26 "
            f"load_s={time.perf_counter() - t0:.3f}",
            flush=True,
        )

    def load_weights(self, src: dict[str, torch.Tensor]) -> None:
        ops = torch.ops.rwkv7_mega_ops_260713
        emb = src["emb.weight"].squeeze().to(device="cuda").contiguous()
        ln0_w = src["blocks.0.ln0.weight"].squeeze().to(device="cuda").contiguous()
        ln0_b = src["blocks.0.ln0.bias"].squeeze().to(device="cuda").contiguous()
        self.z["emb.weight"] = ops.emb_ln0_bf16_to_f16(emb, ln0_w, ln0_b, LN_EPS)
        for key, tensor in src.items():
            if key == "emb.weight" or key.startswith("blocks.0.ln0."):
                continue
            value = tensor.squeeze()
            if key.endswith(LOWRANK_SUFFIXES):
                self.z[key + ".t"] = value.to(device="cuda", dtype=DTYPE).t().contiguous()
                continue
            value = value.to(device="cuda", dtype=DTYPE).contiguous()
            if key.endswith("att.r_k"):
                value = value.flatten().contiguous()
            if key.endswith("ffn.value.weight"):
                self.z[key + ".sparse"] = tile_cmix_value_weight(value.t().contiguous())
            else:
                self.z[key] = value

        p0 = "blocks.0.att."
        if p0 + "v1.t" not in self.z:
            self.z[p0 + "v1.t"] = torch.zeros((1, self.C), device="cuda", dtype=DTYPE)
            self.z[p0 + "v2.t"] = torch.zeros((self.C, 1), device="cuda", dtype=DTYPE)
            self.z[p0 + "v0"] = torch.zeros(self.C, device="cuda", dtype=DTYPE)

    def zero_state(self) -> list[torch.Tensor]:
        return [
            torch.zeros((self.L, 2, 1, self.C), device="cuda", dtype=DTYPE),
            torch.zeros((self.L, 1, self.H, self.N, self.N), device="cuda", dtype=torch.float32),
        ]

    def forward(self, tokens: torch.Tensor, state: list[torch.Tensor]) -> torch.Tensor:
        ops = torch.ops.rwkv7_mega_ops_260713
        x: torch.Tensor | None = None
        residual = self.initial_residual
        v_first: torch.Tensor | None = None
        for layer in range(self.L):
            x, residual, v = self.layer(
                layer, x, residual, state[0][layer], state[1][layer], v_first,
                tokens if layer == 0 else None,
            )
            if layer == 0:
                v_first = v
        assert x is not None
        ops.final_ln_into(x, residual, self.z["ln_out.weight"], self.z["ln_out.bias"], self.final_ln, LN_EPS)
        ops.head_into(self.final_ln.view(self.C), self.z["head.weight"], self.logits)
        return self.logits

    def layer(
        self,
        layer: int,
        x: torch.Tensor | None,
        residual: torch.Tensor,
        shift: torch.Tensor,
        wkv_state: torch.Tensor,
        v_first: torch.Tensor | None,
        tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ops = torch.ops.rwkv7_mega_ops_260713
        z = self.z
        p = f"blocks.{layer}."
        w = self.work[layer]
        if layer == 0:
            assert tokens is not None
            ops.emb_ln_mix6_into(
                z["emb.weight"], tokens.contiguous(), shift[0].view(1, 1, self.C),
                z[p + "ln1.weight"], z[p + "ln1.bias"],
                z[p + "att.x_r"], z[p + "att.x_w"], z[p + "att.x_k"],
                z[p + "att.x_v"], z[p + "att.x_a"], z[p + "att.x_g"],
                w["x_base"], w["xr"], w["xw"], w["xk"], w["xv"], w["xa"], w["xg"], LN_EPS,
            )
        else:
            assert x is not None
            ops.ln_mix6_into(
                x, residual, shift[0].view(1, 1, self.C),
                z[p + "ln1.weight"], z[p + "ln1.bias"],
                z[p + "att.x_r"], z[p + "att.x_w"], z[p + "att.x_k"],
                z[p + "att.x_v"], z[p + "att.x_a"], z[p + "att.x_g"],
                w["x_base"], w["xr"], w["xw"], w["xk"], w["xv"], w["xa"], w["xg"], LN_EPS,
            )

        skip_v = layer == 0
        ops.rkv_lowrank_pre_into(
            w["xr"].view(-1), w["xk"].view(-1), w["xv"].view(-1),
            z[p + "att.receptance.weight"], z[p + "att.key.weight"], z[p + "att.value.weight"],
            w["r"], w["k_raw"], w["v_base"],
            w["xw"].view(1, self.C), w["xa"].view(1, self.C), w["xg"].view(1, self.C), w["xv"].view(1, self.C),
            z[p + "att.w1.t"], z[p + "att.a1.t"], z[p + "att.g1.t"], z[p + "att.v1.t"],
            w["lr_w1"], w["lr_a1"], w["lr_g1"], w["lr_v1"], skip_v,
        )
        v_ref = w["v_base"].view(1, self.C) if skip_v else v_first.view(1, self.C)
        ops.rankout_into(
            w["lr_w1"], w["lr_a1"], w["lr_g1"], w["lr_v1"],
            z[p + "att.w2.t"], z[p + "att.a2.t"], z[p + "att.g2.t"], z[p + "att.v2.t"],
            w["v_base"].view(1, self.C), v_ref, z[p + "att.v0"],
            w["k_raw"].view(1, self.C), z[p + "att.k_k"], z[p + "att.a0"], z[p + "att.k_a"], z[p + "att.w0"],
            w["gate_w"], w["gate_a"], w["gate_g"], w["gate_v"],
            w["new_k"].view(1, self.C), w["neg_kk"].view(1, self.C), w["kka"].view(1, self.C), skip_v,
        )
        v = w["v_base"].view(1, 1, self.C) if skip_v else w["gate_v"].view(1, 1, self.C)
        r = w["r"].view(1, 1, self.C)
        ops.wkv_into(
            wkv_state, r, w["gate_w"].view_as(r), w["new_k"], v,
            w["neg_kk"], w["kka"], w["wkv_y"],
        )
        ops.lnx_into(
            w["wkv_y"], r, w["new_k"], v, z[p + "att.r_k"],
            z[p + "att.ln_x.weight"], z[p + "att.ln_x.bias"],
            w["gate_g"].view_as(r), w["tail_out"],
        )
        ops.att_out_into(w["tail_out"].view(-1), z[p + "att.output.weight"], w["att"])
        ops.ln_cmix_into(
            w["x_base"], w["att"].view(1, 1, self.C), shift[1],
            z[p + "ln2.weight"], z[p + "ln2.bias"], z[p + "ffn.x_k"],
            w["cmix_x"], w["cmix_mixed"], LN_EPS,
        )

        main_stream = torch.cuda.current_stream()
        ffn = int(w["cmix_hid"].numel())
        split_f = ffn // 4
        for chain, stream in enumerate(self.cmix_streams):
            stream.wait_stream(main_stream)
            f0 = chain * split_f
            f1 = f0 + split_f
            with torch.cuda.stream(stream):
                ops.cmix_key_into(
                    w["cmix_mixed"].view(-1), z[p + "ffn.key.weight"][f0:f1], w["cmix_hid"][f0:f1]
                )
        # The zero must be ordered after all key launches and before all atomic
        # value launches. Moving it into either loop is a cross-stream race.
        w["cmix_out"].zero_()
        for stream in self.cmix_streams:
            stream.wait_stream(main_stream)
        for chain, stream in enumerate(self.cmix_streams):
            f0 = chain * split_f
            f1 = f0 + split_f
            with torch.cuda.stream(stream):
                ops.cmix_value_into(
                    w["cmix_hid"][f0:f1], z[p + "ffn.value.weight.sparse"][f0:f1], w["cmix_out"]
                )
        for stream in self.cmix_streams:
            main_stream.wait_stream(stream)
        return w["cmix_x"], w["cmix_out"].view(1, 1, self.C), v


def bench(
    model: RWKV7FastB1T1260713,
    token: torch.Tensor,
    warmup: int,
    iters: int,
    rounds: int,
    graph: bool,
) -> list[float]:
    state = model.zero_state()
    for _ in range(warmup):
        model.forward(token, state)
    torch.cuda.synchronize()
    if graph:
        graph_obj = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_obj):
            model.forward(token, state)
        torch.cuda.synchronize()
        run = graph_obj.replay
    else:
        run = lambda: model.forward(token, state)

    times = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end) / iters))
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--eval-json", default=str(EVAL_JSON))
    parser.add_argument("--token-source", choices=("v3a-synthetic", "eval-json"), default="v3a-synthetic")
    parser.add_argument("--token-index", type=int, default=0)
    parser.add_argument("--token", type=int)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    load_ops()
    model = RWKV7FastB1T1260713(args.model)
    token_id, source = select_token(args, model.V)
    token = torch.tensor([[token_id]], device="cuda", dtype=torch.long)
    logits = model.forward(token, model.zero_state())
    torch.cuda.synchronize()
    top = torch.topk(logits.float(), 5)
    print(
        f"CONFIG token={token_id} source={source} graph={int(args.graph)} "
        f"warmup={args.warmup} iters={args.iters} rounds={args.rounds}",
        flush=True,
    )
    print(f"SMOKE top_ids={top.indices.tolist()} top_vals={[round(x, 4) for x in top.values.tolist()]}", flush=True)
    times = bench(model, token, args.warmup, args.iters, args.rounds, args.graph)
    ordered = sorted(times)
    p50 = statistics.median(times)
    p10 = ordered[max(0, int(len(ordered) * 0.10) - 1)]
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))]
    print(f"RESULT p10_ms={p10:.6f} p50_ms={p50:.6f} p90_ms={p90:.6f} tok_s_p50={1000.0 / p50:.2f}", flush=True)


if __name__ == "__main__":
    main()
