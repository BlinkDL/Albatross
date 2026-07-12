#!/usr/bin/env python3
import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HEAD_SIZE = 64
DTYPE = torch.float16
MODEL_PATH = "/dev/shm/rwkv7-g1f-7.2b-20260414-ctx8192.pth"
THIS_DIR = Path(__file__).resolve().parent
CUDA_DIR = THIS_DIR / "cuda"
L,C,H,N,V = 0,0,0,HEAD_SIZE,0
WKV_MODE = "fp16"
EMB_DEVICE = "cpu"
RKV_MODE = "off"
CMIX_SPARSE = "no-fc"
LOWRANK_WEIGHT = "both"
ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
PP_DEVICES: list[int] = []
LOWRANK_SUFFIXES = ("att.w1", "att.w2", "att.a1", "att.a2", "att.g1", "att.g2", "att.v1", "att.v2")
LOWRANK_IN_ROWS_T = 7
LOWRANK_OUT_ROWS_T = 4
LOWRANK_FUSED_MIN_C = 1024
CMIX_NOFC_ROW20_MAX_T = 5
CMIX_NOFC_T512_MIN_ROWS = 8
LN1_TMIX_FUSE = True
CMIX_B1T1_SPARSE = "b1t1_sparse"
CMIX_ROWS2_SPARSE = "rows2_sparse"
CMIX_B1T1_NOFC = "b1t1_nofc"
CMIX_ROWS2_NOFC = "rows2_nofc"
CMIX_DENSE = "dense"

def main() -> None:
    global MODEL_PATH, WKV_MODE, EMB_DEVICE, RKV_MODE, CMIX_SPARSE, LOWRANK_WEIGHT, ORIG_LINEAR_GROUPS, PP_DEVICES
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--cases", default="1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16") # try 1x1024 1024x1 32x32 for extreme tps
    parser.add_argument("--profile-range", action="store_true")
    parser.add_argument("--eval-json", default="")
    parser.add_argument("--eval-out", default="")
    parser.add_argument("--eval-all-logits-out", default="")
    parser.add_argument("--eval-paths", default="b1tn")
    parser.add_argument("--wkv", choices=("fp16", "fp32io16"), default="fp16") # fp16 is faster at large bsz, fp32io16 is more accurate
    parser.add_argument("--emb", choices=("gpu", "cpu"), default="cpu") # cpu is fast too, and saves VRAM
    parser.add_argument("--batched-rkv", choices=("auto", "on", "off"), default="off") # auto is slightly faster but consumes lots of VRAM
    parser.add_argument("--cmix-sparse", choices=("auto", "no-fc", "off"), default="no-fc") # auto is slightly faster but consumes lots of VRAM
    parser.add_argument("--lowrank-weight", choices=("orig", "transpose", "both"), default="both") # orig saves VRAM but slows tiny B*T; transpose saves VRAM but slows large B*T
    parser.add_argument("--orig-linear-groups", default="att_c2c,ffn_key,head") # comma list: none, att_c2c, ffn_key, head
    parser.add_argument("--pp-devices", default="") # comma list, e.g. 0,1. Empty means single GPU.
    args = parser.parse_args()

    MODEL_PATH = args.model
    WKV_MODE = args.wkv
    EMB_DEVICE = args.emb
    RKV_MODE = args.batched_rkv
    CMIX_SPARSE = args.cmix_sparse
    LOWRANK_WEIGHT = args.lowrank_weight
    ORIG_LINEAR_GROUPS = parse_orig_linear_groups(args.orig_linear_groups)
    PP_DEVICES = parse_pp_devices(args.pp_devices)
    groups = ",".join(sorted(ORIG_LINEAR_GROUPS)) if ORIG_LINEAR_GROUPS else "none"
    pp = ",".join(str(x) for x in PP_DEVICES) if PP_DEVICES else "off"
    log(f"start model={MODEL_PATH} wkv={WKV_MODE} emb={EMB_DEVICE} batched_rkv={RKV_MODE} cmix_sparse={CMIX_SPARSE} lowrank_weight={LOWRANK_WEIGHT} orig_linear_groups={groups} pp={pp}")
    log(f"fixed fast path: ln=v3a linear=v3a/splitk lowrank={LOWRANK_IN_ROWS_T}/{LOWRANK_OUT_ROWS_T} nofc_rows=by_C row20_t=by_C nofc_t512_rows>={CMIX_NOFC_T512_MIN_ROWS}")
    load_extensions(WKV_MODE)
    model = RWKV7()
    if args.eval_json:
        run_eval(model, args.eval_json, args.eval_out, args.eval_all_logits_out, args.eval_paths)
        return
    print("csv_header,label,B,T,iters,p10_ms,p50_ms,p90_ms,tok_s_p50", flush=True)
    for item in args.cases.replace(",", " ").split():
        B, T = [int(x) for x in item.lower().split("x", 1)]
        bench_case(model, B, T, args.warmup, args.iters, args.profile_range)

def log(message: str) -> None:
    print(f"[rwkv7_fast_v3a] {message}", flush=True)

def cuda_mem() -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    free, total = torch.cuda.mem_get_info()
    used = total - free
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return f"gpu_mem used={used/2**30:.2f}GiB allocated={allocated/2**30:.2f}GiB reserved={reserved/2**30:.2f}GiB total={total/2**30:.2f}GiB"

def sync_all() -> None:
    if PP_DEVICES:
        for dev_id in PP_DEVICES:
            torch.cuda.synchronize(dev_id)
    else:
        torch.cuda.synchronize()

def pp_enabled() -> bool:
    return len(PP_DEVICES) > 1

def parse_pp_devices(text: str) -> list[int]:
    if not text.strip():
        return []
    out = [int(x) for x in text.replace(",", " ").split()]
    if len(out) != len(set(out)):
        raise ValueError(f"duplicate pp devices: {out}")
    return out

def first_device() -> torch.device:
    return torch.device(f"cuda:{PP_DEVICES[0]}") if PP_DEVICES else torch.device("cuda")

def last_device() -> torch.device:
    return torch.device(f"cuda:{PP_DEVICES[-1]}") if PP_DEVICES else torch.device("cuda")

def layer_device_index(layer: int) -> int:
    if not pp_enabled():
        return 0
    return min(len(PP_DEVICES) - 1, layer * len(PP_DEVICES) // L)

def layer_device(layer: int) -> torch.device:
    return torch.device(f"cuda:{PP_DEVICES[layer_device_index(layer)]}") if PP_DEVICES else torch.device("cuda")

def pp_segments() -> list[tuple[int, int]]:
    if not pp_enabled():
        return [(0, L)]
    out = []
    start = 0
    while start < L:
        idx = layer_device_index(start)
        end = start + 1
        while end < L and layer_device_index(end) == idx:
            end += 1
        out.append((start, end))
        start = end
    return out

def key_device(key: str) -> torch.device:
    if key == "head.weight" or key.startswith("ln_out."):
        return last_device()
    if key == "emb.weight" or key.startswith("blocks.0.ln0."):
        return first_device()
    parts = key.split(".")
    if len(parts) > 2 and parts[0] == "blocks":
        return layer_device(int(parts[1]))
    return first_device()

@dataclass(frozen=True)
class PathConfig:
    rows: int
    use_batched_rkv: bool
    cmix_mode: str

def select_path(B: int, T: int) -> PathConfig:
    """All B/T dependent fast-path choices live here."""
    rows = B*T
    if CMIX_SPARSE == "off":
        cmix_mode = CMIX_DENSE
    elif CMIX_SPARSE == "no-fc":
        use_nofc = rows <= cmix_nofc_max_rows() or (rows == 20 and T <= cmix_nofc_row20_max_t())
        cmix_mode = CMIX_B1T1_NOFC if rows == 1 else (CMIX_ROWS2_NOFC if use_nofc else CMIX_DENSE)
    elif rows == 1:
        cmix_mode = CMIX_B1T1_SPARSE
    elif rows == 2:
        cmix_mode = CMIX_ROWS2_NOFC
    else:
        cmix_mode = CMIX_DENSE
    if RKV_MODE == "auto":
        use_batched_rkv = (rows == 1) or (4 <= rows <= 64)
    elif RKV_MODE == "on":
        use_batched_rkv = True
    else:
        use_batched_rkv = False
    if use_orig_linear("att_c2c"):
        use_batched_rkv = False
    return PathConfig(rows=rows, use_batched_rkv=use_batched_rkv, cmix_mode=cmix_mode)

def cmix_nofc_max_rows() -> int:
    return 19

def cmix_nofc_row20_max_t() -> int:
    return CMIX_NOFC_ROW20_MAX_T

def parse_orig_linear_groups(text: str) -> set[str]:
    groups = {x.strip() for x in text.replace(",", " ").split() if x.strip()}
    if not groups or groups == {"none"}:
        return set()
    unknown = groups - {"att_c2c", "ffn_key", "head"}
    if unknown:
        raise ValueError(f"unknown orig linear groups: {sorted(unknown)}")
    return groups

def use_orig_linear(group: str) -> bool:
    return group in ORIG_LINEAR_GROUPS

def is_lowrank_weight(key: str) -> bool:
    return key.endswith(LOWRANK_SUFFIXES)

def can_use_lowrank_fused(rows: int) -> bool:
    return C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_IN_ROWS_T

def can_use_lowrank_out_fused(rows: int) -> bool:
    return C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T

def is_att_c2c_weight(key: str) -> bool:
    return ".att." in key and key.endswith(("receptance.weight", "key.weight", "value.weight", "output.weight"))

def is_orig_linear_weight(key: str) -> bool:
    return (
        (use_orig_linear("att_c2c") and is_att_c2c_weight(key))
        or (use_orig_linear("ffn_key") and ".ffn.key.weight" in key)
        or (use_orig_linear("head") and key == "head.weight")
    )

def load_extensions(wkv_mode: str = "fp16") -> None:
    t0 = time.perf_counter()
    log(f"loading CUDA extensions v3a_ops + fast_ops + nf4_ops + wkv={wkv_mode}")
    cuda_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization"] + ([] if os.name == "nt" else ["-Xptxas", "-O3"])
    load(name="rwkv7_v3a_ops", sources=[str(CUDA_DIR / "rwkv7_v3a_ops.cpp"), str(CUDA_DIR / "rwkv7_v3a_ops.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags)
    load(name="rwkv7_fast_ops_fp16", sources=[str(CUDA_DIR / "rwkv7_fast_ops_fp16.cpp"), str(CUDA_DIR / "rwkv7_fast_ops_fp16.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags)
    load(name="rwkv7_nf4_ops", sources=[str(CUDA_DIR / "rwkv7_nf4_ops.cpp"), str(CUDA_DIR / "rwkv7_nf4_ops.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags)
    if wkv_mode == "fp16":
        load(name="rwkv7_wkv_fp16_v2", sources=[str(CUDA_DIR / "rwkv7_wkv_fp16_v2.cpp"), str(CUDA_DIR / "rwkv7_wkv_fp16_v2.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas", "-O3"])
    elif wkv_mode == "fp32io16":
        load(name="rwkv7_wkv_fp32_v2", sources=[str(CUDA_DIR / "rwkv7_wkv_fp32_v2.cpp"), str(CUDA_DIR / "rwkv7_wkv_fp32_v2.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3", "-D_IO_FP16_"], extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-O3", "-D_IO_FP16_"])
    else:
        raise ValueError(f"unknown wkv_mode: {wkv_mode}")
    log(f"CUDA extensions loaded in {time.perf_counter() - t0:.3f}s")

class RWKV7:
    def __init__(self) -> None:
        global L, C, H, N, V
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch._C._jit_set_autocast_mode(False)

        t0 = time.perf_counter()
        log(f"loading weights from {MODEL_PATH}")
        z = torch.load(MODEL_PATH, map_location="cpu", mmap=True)
        log(f"weights mmap loaded in {time.perf_counter() - t0:.3f}s tensors={len(z)}")

        H, N = z["blocks.0.att.r_k"].shape
        C, V = H * N, z["emb.weight"].shape[0]
        assert N == HEAD_SIZE
        max_layer = max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks."))
        L = max_layer + 1
        log(f"detected model C={C} H={H} N={N} V={V}")
        log(f"cmix no-fc path: rows<={cmix_nofc_max_rows()} row20_t<={cmix_nofc_row20_max_t()}")

        emb_src = z["emb.weight"].squeeze()
        ln0_w_src = z["blocks.0.ln0.weight"].squeeze()
        ln0_b_src = z["blocks.0.ln0.bias"].squeeze()
        emb_cpu = emb_src if EMB_DEVICE == "cpu" else None
        t0 = time.perf_counter()
        log(f"moving and preprocessing weights to CUDA emb={EMB_DEVICE}")
        for key in list(z.keys()):
            if key == "emb.weight" and emb_cpu is not None:
                continue
            value = z[key].squeeze()
            dev = key_device(key)
            # NVFP4: uint8 weights stay uint8, .nf4_b_scale stays float8_e4m3fn, .nvfp4_t_scale stays float32
            is_nf4_weight = value.dtype == torch.uint8 and key.endswith(".weight")
            is_nf4_scale = key.endswith(".nf4_b_scale")
            is_nf4_t_scale = key.endswith(".nvfp4_t_scale")
            is_lowrank = is_lowrank_weight(key)
            if ".ffn.key.weight" in key and CMIX_SPARSE == "auto":
                z[key + ".fc"] = value.to(device=dev, dtype=DTYPE).contiguous()
            if (
                not is_lowrank
                and not is_nf4_weight
                and not is_nf4_scale
                and not is_nf4_t_scale
                and (("key.weight" in key and not is_orig_linear_weight(key))
                or ("value.weight" in key and not is_orig_linear_weight(key))
                or ("receptance.weight" in key and not is_orig_linear_weight(key))
                or ("output.weight" in key and not is_orig_linear_weight(key))
                or ("head.weight" in key and not is_orig_linear_weight(key)))
            ):
                value = value.t()
            if is_nf4_weight:
                value = value.to(device=dev).contiguous()
            elif is_nf4_scale:
                value = value.to(device=dev).contiguous()
            elif is_nf4_t_scale:
                value = value.to(device=dev).contiguous()
            else:
                value = value.to(device=dev, dtype=DTYPE).contiguous()
            if key.endswith("att.r_k"):
                value = value.flatten().contiguous()
            if is_lowrank:
                if LOWRANK_WEIGHT in ("orig", "both"):
                    z[key] = value
                else:
                    del z[key]
                if LOWRANK_WEIGHT in ("transpose", "both"):
                    z[key + ".t"] = value.t().contiguous()
            else:
                z[key] = value
        emb_dev = first_device()
        ln0_w_bf16 = ln0_w_src.to(device=emb_dev).contiguous()
        ln0_b_bf16 = ln0_b_src.to(device=emb_dev).contiguous()
        if emb_cpu is None:
            with torch.cuda.device(emb_dev):
                z["emb.weight"] = torch.ops.rwkv7_v3a_ops.emb_ln0_bf16_to_f16(
                    emb_src.to(device=emb_dev).contiguous(), ln0_w_bf16, ln0_b_bf16)
        else:
            emb = torch.empty((V,C), dtype=DTYPE, pin_memory=True)
            with torch.cuda.device(emb_dev):
                for start in range(0, V, 4096):
                    end = min(start + 4096, V)
                    chunk = emb_cpu[start:end].to(device=emb_dev).contiguous()
                    chunk = torch.ops.rwkv7_v3a_ops.emb_ln0_bf16_to_f16(chunk, ln0_w_bf16, ln0_b_bf16)
                    emb[start:end].copy_(chunk)
            z["emb.weight"] = emb
        if RKV_MODE != "off" and not use_orig_linear("att_c2c"):
            for layer in range(L):
                p = f"blocks.{layer}.att."
                z[p+"rkv.weight"] = torch.stack((z[p+"receptance.weight"], z[p+"key.weight"], z[p+"value.weight"])).contiguous()
        self.z = z
        self.emb_cpu = EMB_DEVICE == "cpu"
        self.emb_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._graph_cache: dict[tuple[int, int], dict] = {}
        # NVFP4: build block scale + tensor scale lookup tables (weight id -> scale)
        self.nf4_block_scales: dict[int, torch.Tensor] = {}
        self.nf4_t_scales: dict[int, float] = {}
        for key, val in z.items():
            if key.endswith(".nf4_b_scale"):
                base_key = key[:-len(".nf4_b_scale")]
                if base_key in z:
                    self.nf4_block_scales[id(z[base_key])] = val
            elif key.endswith(".nvfp4_t_scale"):
                base_key = key[:-len(".nvfp4_t_scale")]
                if base_key in z:
                    self.nf4_t_scales[id(z[base_key])] = val.item()
        sync_all()
        log(f"model ready in {time.perf_counter() - t0:.3f}s L={L} C={C} H={H} N={N} V={V}")
        log(cuda_mem())

    def zero_state(self, B: int) -> list[torch.Tensor]:
        if pp_enabled():
            shift = []
            wkv = []
            for layer in range(L):
                dev = layer_device(layer)
                shift.append(torch.zeros((2,B,C), dtype=DTYPE, device=dev))
                wkv.append(torch.zeros((B,H,N,N), dtype=torch.float32 if WKV_MODE == "fp32io16" else DTYPE, device=dev))
            elapsed = [torch.zeros((B,), dtype=torch.int32, device=torch.device(f"cuda:{d}")) for d in PP_DEVICES]
            return [shift, wkv, elapsed]
        return [
            torch.zeros((L,2,B,C), dtype=DTYPE, device="cuda"),
            torch.zeros((L,B,H,N,N), dtype=torch.float32 if WKV_MODE == "fp32io16" else DTYPE, device="cuda"),
            torch.zeros((B,), dtype=torch.int32, device="cuda"),
        ]

    def forward(self, tokens: torch.Tensor, state: list[torch.Tensor]) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = select_path(B, T)
        x = self.embed(tokens)
        return self.forward_from_x(x, state, path)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.emb_cpu:
            if tokens.device != self.z["emb.weight"].device:
                tokens = tokens.to(self.z["emb.weight"].device, non_blocking=True)
            return self.z["emb.weight"][tokens]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        host, dev = self.emb_cache.get((B, T), (None, None))
        if host is None:
            host = torch.empty((B*T,C), dtype=DTYPE, pin_memory=True)
            dev = torch.empty((B,T,C), dtype=DTYPE, device=first_device())
            self.emb_cache[(B, T)] = (host, dev)
        flat = tokens.reshape(-1)
        if flat.device.type != "cpu":
            flat = flat.cpu()
        torch.index_select(self.z["emb.weight"], 0, flat, out=host)
        dev.copy_(host.view(B,T,C), non_blocking=True)
        return dev

    def forward_from_x(self, x: torch.Tensor, state: list[torch.Tensor], path: PathConfig, all_logits: bool = False, last_indices=None) -> torch.Tensor:
        """Forward pass with automatic CUDA Graph caching for repeated (B,T) shapes.

        On first call for a given (B,T), performs warmup + graph capture.
        Subsequent calls copy input into static buffer and replay the graph.
        State tensors are updated in-place by the graph replay.

        Disable with CUDA_GRAPH_AUTO=0. State must be the same tensor objects
        across calls for a given (B,T) — create once with zero_state() and reuse.
        """
        B, T, _ = x.shape
        use_graph = (
            os.environ.get("CUDA_GRAPH_AUTO", "1") == "1"
            and not torch.cuda.is_current_stream_capturing()
            and not all_logits
            and not pp_enabled()
            and last_indices is None
        )
        if not use_graph:
            return self._forward_from_x_impl(x, state, path, all_logits, last_indices)

        key = (B, T)
        cached = self._graph_cache.get(key)
        if cached is None:
            log(f"CUDA Graph: capturing B={B} T={T}")
            static_x = x.clone()
            # Warmup with throwaway state so user's state is untouched
            warmup_state = self.zero_state(B)
            for _ in range(2):
                self._forward_from_x_impl(static_x, warmup_state, path, False, None)
            torch.cuda.synchronize()
            # Capture on side stream (same pattern as bench_case)
            # Must warmup on capture stream before graph capture
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._forward_from_x_impl(static_x, warmup_state, path, False, None)
            torch.cuda.current_stream().wait_stream(stream)
            # Save state before capture — capture will advance it, we must undo
            state_backup = [s.clone() for s in state]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                static_out = self._forward_from_x_impl(static_x, state, path, False, None)
            torch.cuda.current_stream().wait_stream(stream)
            # Restore state to pre-capture values (undo capture's 1-step advance)
            for s, backup in zip(state, state_backup):
                s.copy_(backup)
            self._graph_cache[key] = {
                "graph": graph,
                "static_x": static_x,
                "static_out": static_out,
            }
            log(f"CUDA Graph: captured B={B} T={T}")
            # Replay to get correct output (capture output is unreliable)
            # This advances state by exactly 1 step — the expected behavior
            static_x.copy_(x)
            graph.replay()
            return static_out.clone()

        # Replay: copy input, replay graph, clone output
        cached["static_x"].copy_(x)
        cached["graph"].replay()
        return cached["static_out"].clone()

    def _forward_from_x_impl(self, x: torch.Tensor, state: list[torch.Tensor], path: PathConfig, all_logits: bool = False, last_indices=None) -> torch.Tensor:
        if pp_enabled():
            return self.forward_from_x_pp(x, state, path, all_logits, last_indices)
        z = self.z
        B, T, _ = x.shape
        v_first = x
        xx = self.ln(x, z["blocks.0.ln1.weight"], z["blocks.0.ln1.bias"])
        pre_mix = None

        for layer in range(L):
            p = f"blocks.{layer}."
            xx, v_first = self.tmix(layer, xx, state[0][layer], state[1][layer], state[2], v_first, p+"att.", path, pre_mix)
            pre_mix = None
            if T == 1 and path.cmix_mode not in (CMIX_B1T1_SPARSE, CMIX_ROWS2_SPARSE):
                x, mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
                    x.contiguous(), xx.contiguous(), state[0][layer][1], z[p+"ln2.weight"], z[p+"ln2.bias"], z[p+"ffn.x_k"])
                xx = self.cmix_from_mixed(mixed, p+"ffn.", path)
            else:
                x, xx = self.add_ln(x, xx, z[p+"ln2.weight"], z[p+"ln2.bias"])
                xx = self.cmix(xx, state[0][layer], p+"ffn.", path)
            if layer + 1 < L:
                p_next = f"blocks.{layer + 1}."
                if LN1_TMIX_FUSE and B == 1 and T == 1:
                    outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16(
                        x.contiguous(), xx.contiguous(), state[0][layer + 1][0],
                        z[p_next+"ln1.weight"], z[p_next+"ln1.bias"],
                        z[p_next+"att.x_r"], z[p_next+"att.x_w"], z[p_next+"att.x_k"],
                        z[p_next+"att.x_v"], z[p_next+"att.x_a"], z[p_next+"att.x_g"])
                    x, pre_mix = outs[0], outs[1:]
                    xx = x
                else:
                    x, xx = self.add_ln(x, xx, z[p_next+"ln1.weight"], z[p_next+"ln1.bias"])
            elif not all_logits:
                if last_indices is not None:
                    x = self.ln(self.add(x, xx), z["ln_out.weight"], z["ln_out.bias"])
                    x = x[torch.arange(B, device=x.device), last_indices].contiguous()
                else:
                    x = self.add_last_ln(x, xx, z["ln_out.weight"], z["ln_out.bias"])
                torch.ops.rwkv7_v3a_ops.advance_i32(state[2], T) # !!! IMPORTANT FOR WKV16 DITHERING !!!
                return self.linear_head(x)
            else:
                x = self.add(x, xx)

        x = self.ln(x, z["ln_out.weight"], z["ln_out.bias"])
        torch.ops.rwkv7_v3a_ops.advance_i32(state[2], T) # !!! IMPORTANT FOR WKV16 DITHERING !!!
        return self.linear_head(x)

    def forward_from_x_pp(self, x: torch.Tensor, state: list[torch.Tensor], path: PathConfig, all_logits: bool = False, last_indices=None) -> torch.Tensor:
        B, T, _ = x.shape
        v_first = None
        v_first_by_stage: dict[int, torch.Tensor] = {}
        x = x.to(first_device())
        segments = pp_segments()
        for stage, (start, end) in enumerate(segments):
            dev = layer_device(start)
            if x.device != dev:
                x = x.to(dev)
            with torch.cuda.device(dev):
                v_in = None if start == 0 else v_first_by_stage[stage]
                x, v_first = self.forward_pp_segment(x, state, path, start, end, v_in)
            if start == 0 and v_first is not None:
                for next_stage, (next_start, _) in enumerate(segments[1:], 1):
                    next_dev = layer_device(next_start)
                    v_first_by_stage[next_stage] = v_first if next_dev == v_first.device else v_first.to(next_dev)
        with torch.cuda.device(last_device()):
            return self.forward_pp_tail(x, state, T, all_logits, last_indices, advance=True)

    def forward_pp_segment(self, x: torch.Tensor, state: list[torch.Tensor], path: PathConfig, start: int, end: int, v_first: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = self.z
        B, T, _ = x.shape
        out_v_first = None
        xx = self.ln(x, z[f"blocks.{start}.ln1.weight"], z[f"blocks.{start}.ln1.bias"])
        pre_mix = None
        for layer in range(start, end):
            p = f"blocks.{layer}."
            v_in = x if layer == 0 else v_first
            xx, v_out = self.tmix(layer, xx, state[0][layer], state[1][layer], state[2][layer_device_index(layer)], v_in, p+"att.", path, pre_mix)
            pre_mix = None
            if layer == 0:
                v_first = v_out
                out_v_first = v_out
            if T == 1 and path.cmix_mode not in (CMIX_B1T1_SPARSE, CMIX_ROWS2_SPARSE):
                x, mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
                    x.contiguous(), xx.contiguous(), state[0][layer][1], z[p+"ln2.weight"], z[p+"ln2.bias"], z[p+"ffn.x_k"])
                xx = self.cmix_from_mixed(mixed, p+"ffn.", path)
            else:
                x, xx = self.add_ln(x, xx, z[p+"ln2.weight"], z[p+"ln2.bias"])
                xx = self.cmix(xx, state[0][layer], p+"ffn.", path)
            if layer + 1 < end:
                p_next = f"blocks.{layer + 1}."
                if LN1_TMIX_FUSE and B == 1 and T == 1:
                    outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16(
                        x.contiguous(), xx.contiguous(), state[0][layer + 1][0],
                        z[p_next+"ln1.weight"], z[p_next+"ln1.bias"],
                        z[p_next+"att.x_r"], z[p_next+"att.x_w"], z[p_next+"att.x_k"],
                        z[p_next+"att.x_v"], z[p_next+"att.x_a"], z[p_next+"att.x_g"])
                    x, pre_mix = outs[0], outs[1:]
                    xx = x
                else:
                    x, xx = self.add_ln(x, xx, z[p_next+"ln1.weight"], z[p_next+"ln1.bias"])
            else:
                x = self.add(x, xx)
        return x, out_v_first

    def forward_pp_tail(self, x: torch.Tensor, state: list[torch.Tensor], T: int, all_logits: bool = False, last_indices=None, advance: bool = True) -> torch.Tensor:
        B = x.size(0)
        if not all_logits:
            if last_indices is None:
                x = x[:, -1].contiguous()
            else:
                x = x[torch.arange(B, device=x.device), last_indices].contiguous()
        x = self.ln(x, self.z["ln_out.weight"], self.z["ln_out.bias"])
        if advance:
            self.advance_pp_elapsed(state, T)
        return self.linear_head(x)

    def advance_pp_elapsed(self, state: list[torch.Tensor], T: int) -> None:
        for idx, dev_id in enumerate(PP_DEVICES):
            with torch.cuda.device(dev_id):
                torch.ops.rwkv7_v3a_ops.advance_i32(state[2][idx], T)

    def ln(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.layer_norm_f16(x.contiguous(), weight, bias)

    def forward_all_logits(self, tokens: torch.Tensor, state: list[torch.Tensor]) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = select_path(B, T)
        x = self.embed(tokens)
        return self.forward_from_x(x, state, path, all_logits=True)

    def forward_last_at(self, tokens: torch.Tensor, state: list[torch.Tensor], last_indices: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = select_path(B, T)
        x = self.embed(tokens)
        return self.forward_from_x(x, state, path, last_indices=last_indices)

    def _linear_rkv(self, xr: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, p: str, path: PathConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute r, k, v — uses fused NVFP4 kernel for T=1 decode, else 3 separate calls."""
        z = self.z
        w_r = z[p+"receptance.weight"]
        if w_r.dtype == torch.uint8 and path.rows == 1:
            w_k = z[p+"key.weight"]
            w_v = z[p+"value.weight"]
            bs_r = self.nf4_block_scales[id(w_r)]
            bs_k = self.nf4_block_scales[id(w_k)]
            bs_v = self.nf4_block_scales[id(w_v)]
            ts_r = self.nf4_t_scales.get(id(w_r), 1.0)
            ts_k = self.nf4_t_scales.get(id(w_k), 1.0)
            ts_v = self.nf4_t_scales.get(id(w_v), 1.0)
            rkv = torch.ops.rwkv7_nf4_ops.linear_nvfp4_rkv_orig_row1_blk16_f16(
                xr.contiguous(), xk.contiguous(), xv.contiguous(),
                w_r, w_k, w_v, bs_r, bs_k, bs_v, ts_r, ts_k, ts_v, 4)
            B, T = xr.shape[0], xr.shape[1]
            N = w_r.size(0)
            return rkv[0].view(B, T, N), rkv[1].view(B, T, N), rkv[2].view(B, T, N)
        r = self.linear_orig_layout(xr, w_r, path, "att_c2c")
        k = self.linear_orig_layout(xk, z[p+"key.weight"], path, "att_c2c")
        v = self.linear_orig_layout(xv, z[p+"value.weight"], path, "att_c2c")
        return r, k, v

    def tmix(self, layer: int, x: torch.Tensor, shift_state: torch.Tensor, wkv_state: torch.Tensor, elapsed_t: torch.Tensor, v_first: torch.Tensor, p: str, path: PathConfig, pre_mix=None) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape
        if pre_mix is not None:
            xr, xw, xk, xv, xa, xg = pre_mix
        else:
            xr, xw, xk, xv, xa, xg = ops.tmix_mix6(B, T, C, x.contiguous(), shift_state[0], z[p+"x_r"], z[p+"x_w"], z[p+"x_k"], z[p+"x_v"], z[p+"x_a"], z[p+"x_g"])
        if path.use_batched_rkv:
            flat = torch.stack((xr.reshape(-1,C), xk.reshape(-1,C), xv.reshape(-1,C)))
            rkv = torch.bmm(flat, z[p+"rkv.weight"])
            r, k, v = [t.view(B,T,C) for t in rkv.unbind(0)]
        else:
            r, k, v = self._linear_rkv(xr, xk, xv, p, path)

        v1 = None
        if LOWRANK_WEIGHT != "orig" and can_use_lowrank_fused(path.rows) and can_use_lowrank_out_fused(path.rows) and layer != 0:
            w1, a1, g1, v1 = torch.ops.rwkv7_v3a_ops.linear_wagv_rank_in_f16(
                xw.contiguous(), xa.contiguous(), xg.contiguous(), xv.contiguous(),
                z[p+"w1.t"], z[p+"a1.t"], z[p+"g1.t"], z[p+"v1.t"])
        elif LOWRANK_WEIGHT != "orig" and can_use_lowrank_fused(path.rows):
            w1, a1, g1 = torch.ops.rwkv7_v3a_ops.linear_wag_rank_in_f16(
                xw.contiguous(), xa.contiguous(), xg.contiguous(), z[p+"w1.t"], z[p+"a1.t"], z[p+"g1.t"])
        else:
            w1 = self.linear_rank_in(xw, z.get(p+"w1"), z.get(p+"w1.t"), path.rows)
            a1 = self.linear_rank_in(xa, z.get(p+"a1"), z.get(p+"a1.t"), path.rows)
            g1 = self.linear_rank_in(xg, z.get(p+"g1"), z.get(p+"g1.t"), path.rows)
        v_done = False
        if LOWRANK_WEIGHT != "orig" and can_use_lowrank_out_fused(path.rows) and layer != 0 and v1 is not None:
            w, a, g, v = torch.ops.rwkv7_v3a_ops.linear_wagv_rank_out_f16(
                w1.contiguous(), a1.contiguous(), g1.contiguous(), v1.contiguous(),
                z[p+"w2.t"], z[p+"a2.t"], z[p+"g2.t"], z[p+"v2.t"],
                v.contiguous(), v_first.contiguous(), z[p+"v0"])
            v_done = True
        elif LOWRANK_WEIGHT != "orig" and can_use_lowrank_out_fused(path.rows):
            w, a, g = torch.ops.rwkv7_v3a_ops.linear_wag_rank_out_f16(
                w1.contiguous(), a1.contiguous(), g1.contiguous(), z[p+"w2.t"], z[p+"a2.t"], z[p+"g2.t"])
        else:
            w = self.linear_rank_out_act(w1, z.get(p+"w2"), z.get(p+"w2.t"), path.rows, 1)
            a = self.linear_rank_out(a1, z.get(p+"a2"), z.get(p+"a2.t"), path.rows)
            g = self.linear_rank_out_act(g1, z.get(p+"g2"), z.get(p+"g2.t"), path.rows, 2)
        # Determine dispatch mode once (avoids duplicate condition checks)
        if T == 1 and WKV_MODE == "fp16" and B <= int(os.environ.get("FUSE_MAX_B", "2")):
            if os.environ.get("KKAG_WKV_LNX_FUSE", "1") == "1":
                dispatch_mode = "kkag"
            elif os.environ.get("WKV_LNX_FUSE", "1") == "1":
                dispatch_mode = "lnx"
            else:
                dispatch_mode = "sep"
        else:
            dispatch_mode = "sep"

        neg_kk = kka = None
        if dispatch_mode != "kkag":
            k, neg_kk, kka = ops.tmix_kk_a_gate(B, T, C, H, k.contiguous(), z[p+"k_k"], z[p+"a0"], a.contiguous(), z[p+"k_a"])

        if layer == 0:
            v_first = v
        elif not v_done:
            if LOWRANK_WEIGHT != "orig" and can_use_lowrank_out_fused(path.rows):
                if v1 is None:
                    v1 = self.linear_rank_in(xv, z.get(p+"v1"), z.get(p+"v1.t"), path.rows)
                v = torch.ops.rwkv7_v3a_ops.linear_t_vres_f16(v1.contiguous(), z[p+"v2.t"], v.contiguous(), v_first.contiguous(), z[p+"v0"])
            else:
                v12 = self.linear_rank_out(self.linear_rank_in(xv, z.get(p+"v1"), z.get(p+"v1.t"), path.rows), z.get(p+"v2"), z.get(p+"v2.t"), path.rows)
                v = ops.tmix_vres_gate(B, T, C, v.contiguous(), v_first.contiguous(), z[p+"v0"], v12.contiguous())

        y_wkv = torch.empty_like(r)
        y = torch.empty_like(r)
        if dispatch_mode == "kkag":
            torch.ops.rwkv7_wkv_fp16_v2.wkv_lnx_kkag_seq_w0(B, T, C, H, wkv_state, r.contiguous(), w.contiguous(), z[p+"w0"], k.contiguous(), v.contiguous(), z[p+"k_k"], z[p+"a0"], a.contiguous(), z[p+"k_a"], y_wkv, elapsed_t, z[p+"r_k"], z[p+"ln_x.weight"], z[p+"ln_x.bias"], g.contiguous(), y)
        elif dispatch_mode == "lnx":
            torch.ops.rwkv7_wkv_fp16_v2.wkv_lnx_seq_w0(B, T, C, H, wkv_state, r.contiguous(), w.contiguous(), z[p+"w0"], k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y_wkv, elapsed_t, z[p+"r_k"], z[p+"ln_x.weight"], z[p+"ln_x.bias"], g.contiguous(), y)
        else:
            if WKV_MODE == "fp32io16":
                w_raw = ops.add_vec(C, w.contiguous(), z[p+"w0"])
                torch.ops.rwkv7_wkv_fp32_v2.forward(B, T, C, H, wkv_state, r.contiguous(), w_raw.contiguous(), k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y_wkv)
            elif T <= 16:
                torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0(B, T, C, H, wkv_state, r.contiguous(), w.contiguous(), z[p+"w0"], k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y_wkv, elapsed_t)
            else:
                w_raw = ops.add_vec(C, w.contiguous(), z[p+"w0"])
                torch.ops.rwkv7_wkv_fp16_v2.wkv_seq(B, T, C, H, wkv_state, r.contiguous(), w_raw.contiguous(), k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y_wkv, elapsed_t)
            y = ops.tmix_lnx_rkvres_xg(B, T, C, H, y_wkv.contiguous(), r.contiguous(), k.contiguous(), v.contiguous(), z[p+"r_k"], z[p+"ln_x.weight"], z[p+"ln_x.bias"], g.contiguous())
        return self.linear_orig_layout(y, z[p+"output.weight"], path, "att_c2c"), v_first

    def cmix(self, x: torch.Tensor, shift_state: torch.Tensor, p: str, path: PathConfig) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape

        if path.cmix_mode == CMIX_B1T1_SPARSE:
            vw = self._dequant_nf4_value(z[p+"value.weight"])
            return ops.cmix_sparse_one(C, z[p+"key.weight.fc"].size(0), x.contiguous(), shift_state[1], z[p+"x_k"], z[p+"key.weight.fc"], vw)
        if path.cmix_mode == CMIX_ROWS2_SPARSE:
            vw = self._dequant_nf4_value(z[p+"value.weight"])
            return ops.cmix_sparse_rows(B, T, C, z[p+"key.weight.fc"].size(0), x.contiguous(), shift_state[1], z[p+"x_k"], z[p+"key.weight.fc"], vw)

        mixed = ops.cmix_mix(B, T, C, x.contiguous(), shift_state[1], z[p+"x_k"])
        return self.cmix_from_mixed(mixed, p, path)

    def _dequant_nf4_value(self, vw: torch.Tensor) -> torch.Tensor:
        """Dequant NVFP4 value.weight to fp16 for non-NF4 cmix paths (auto mode)."""
        if vw.dtype == torch.uint8:
            b_scale = self.nf4_block_scales.get(id(vw))
            if b_scale is not None:
                t_scale = self.nf4_t_scales.get(id(vw), 1.0)
                return torch.ops.rwkv7_nf4_ops.dequant_nf4_to_f16(vw, b_scale, t_scale, False)
        return vw

    def cmix_from_mixed(self, mixed: torch.Tensor, p: str, path: PathConfig) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = mixed.shape
        hid = self.linear_orig_layout(mixed, z[p+"key.weight"], path, "ffn_key")
        vw = z[p+"value.weight"]
        # NVFP4 value.weight: dispatch to NF4 cmix_sparse kernels
        if vw.dtype == torch.uint8:
            b_scale = self.nf4_block_scales[id(vw)]
            t_scale = self.nf4_t_scales[id(vw)]
            nf4_ops = torch.ops.rwkv7_nf4_ops
            F = vw.size(0)
            if path.cmix_mode == CMIX_B1T1_NOFC:
                return nf4_ops.cmix_sparse_down_relu_one_nf4(hid.view(-1).contiguous(), vw, b_scale, t_scale, C, F)
            if path.cmix_mode == CMIX_ROWS2_NOFC:
                if path.rows >= CMIX_NOFC_T512_MIN_ROWS and C % 512 == 0 and F % 512 == 0:
                    return nf4_ops.cmix_sparse_down_relu_rows_t512_nf4(hid.contiguous(), vw, b_scale, t_scale, B, T, C, F)
                return nf4_ops.cmix_sparse_down_relu_rows_nf4(hid.contiguous(), vw, b_scale, t_scale, B, T, C, F)
            # dense path: dequant + cuBLAS
            k = ops.relu_square(hid.contiguous())
            return self.linear(k, vw)
        # FP16 value.weight: original path
        if path.cmix_mode == CMIX_B1T1_NOFC:
            return ops.cmix_sparse_down_relu_one(C, vw.size(0), hid.view(-1).contiguous(), vw)
        if path.cmix_mode == CMIX_ROWS2_NOFC:
            F = vw.size(0)
            if path.rows >= CMIX_NOFC_T512_MIN_ROWS and C % 512 == 0 and F % 512 == 0:
                return ops.cmix_sparse_down_relu_rows_t512(B, T, C, F, hid.contiguous(), vw)
            return ops.cmix_sparse_down_relu_rows(B, T, C, F, hid.contiguous(), vw)

        k = ops.relu_square(hid.contiguous())
        return self.linear(k, vw)

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if weight.dtype == torch.uint8:
            b_scale = self.nf4_block_scales.get(id(weight))
            if b_scale is not None:
                t_scale = self.nf4_t_scales.get(id(weight), 1.0)
                weight = torch.ops.rwkv7_nf4_ops.dequant_nf4_to_f16(weight, b_scale, t_scale, True)
        if x.numel() == x.size(-1) and weight.size(1) % 64 == 0:
            return torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(x.contiguous(), weight)
        return torch.ops.rwkv7_v3a_ops.linear_f16(x.contiguous(), weight)

    def linear_head(self, x: torch.Tensor) -> torch.Tensor:
        z = self.z
        # head.weight never quantized: LM head must stay FP16 for RL training stability
        if not use_orig_linear("head"):
            return self.linear(x, z["head.weight"])
        rows = x.numel() // C
        return self.linear_orig_layout(x, z["head.weight"], PathConfig(rows, False, CMIX_DENSE), "head")

    def linear_orig_layout(self, x: torch.Tensor, weight: torch.Tensor, path: PathConfig, group: str) -> torch.Tensor:
        if not use_orig_linear(group):
            return self.linear(x, weight)
        # NVFP4 dispatch: uint8 weight -> NVFP4 kernel (same dispatch params as v3a FP16)
        if weight.dtype == torch.uint8:
            b_scale = self.nf4_block_scales.get(id(weight))
            assert b_scale is not None, f"NVFP4 weight has no block scale (id={id(weight)})"
            t_scale = self.nf4_t_scales.get(id(weight), 1.0)
            if path.rows == 1:
                # Optimized blk16 GEMV: uint2 vectorized, 1-warp, always K%16==0
                return torch.ops.rwkv7_nf4_ops.linear_nvfp4_orig_row1_blk16_f16(x.contiguous(), weight, b_scale, t_scale, 4)
            if path.rows == 2:
                return torch.ops.rwkv7_nf4_ops.linear_nvfp4_orig_row2_blk16_f16(x.contiguous(), weight, b_scale, t_scale, 2)
            # M>=3: use NVFP4 GEMM kernel for small M, dequant+cuBLAS for large M
            if path.rows <= 12:
                if group == "att_c2c":
                    if C <= 1024:
                        return torch.ops.rwkv7_nf4_ops.linear_nf4_orig_rows_f16(x.contiguous(), weight, b_scale, t_scale, 2, 2)
                    return torch.ops.rwkv7_nf4_ops.linear_nf4_orig_rows_f16(x.contiguous(), weight, b_scale, t_scale, 3, 2)
                return torch.ops.rwkv7_nf4_ops.linear_nf4_orig_rows_f16(x.contiguous(), weight, b_scale, t_scale, 2, 4)
            # Large M: dequant to fp16 and fall through to cuBLAS
            weight = torch.ops.rwkv7_nf4_ops.dequant_nf4_to_f16(weight, b_scale, t_scale, False)
        if path.rows == 1:
            if group == "ffn_key":
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, True)
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, C <= 1024)
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, group != "att_c2c" or C < 2048)
        if path.rows == 2:
            if group == "att_c2c":
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 64, 2, True)
            if group == "ffn_key":
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, False)
                if C < 4096:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 64, 2, True)
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, False)
            if group == "head" and C == 2560:
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, False)
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 64, 2, True)
        if path.rows == 3:
            if group == "head":
                if C <= 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 3, 2)
            if group == "ffn_key":
                if C <= 1024:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_cfg_f16(x.contiguous(), weight, 64, 3, 4)
                if C == 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if group == "att_c2c":
                if C == 768:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 1, 2)
                if C == 1024:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 2, 2)
                if C == 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 3, 4)
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 3, 2)
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_cfg_f16(x.contiguous(), weight, 64, 3, 4)
        if path.rows == 4:
            if group == "ffn_key":
                if C <= 1024:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_cfg_f16(x.contiguous(), weight, 64, 2, 4)
                if C == 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if group == "att_c2c":
                if C <= 1024:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 2, 2)
                if C == 2048:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 4, 2)
                if C == 2560:
                    return torch.ops.rwkv7_v3a_ops.linear_orig_rows_f16(x.contiguous(), weight, 4, 2)
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
        if group == "head":
            if C == 768:
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 3)
                if 96 <= path.rows < 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
            if C == 1024:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
                if 96 <= path.rows < 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if C == 2048:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 6)
                if 128 <= path.rows < 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if C == 2560:
                if path.rows >= 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
                if path.rows >= 192:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 5)
                if path.rows >= 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 5)
                if path.rows >= 128:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
                if path.rows >= 96:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
                if path.rows >= 80:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
                if path.rows >= 72:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 1024:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
            if path.rows >= 512:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
            if path.rows >= 384:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
            if path.rows >= 256:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
            if path.rows >= 192:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
            if path.rows >= 160:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 128:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
            if path.rows >= 112:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 96:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 80:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 2)
            if path.rows >= 72:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
        if group == "att_c2c":
            if C == 2560 and 17 <= path.rows <= 20:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if C == 768:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 1)
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 3)
            if C == 1024:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 6)
            if C == 2048:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 3)
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
            if C == 2560:
                if path.rows >= 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
                if path.rows >= 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
                if path.rows >= 128:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
                if path.rows >= 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 3)
                if path.rows >= 96:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 2)
                if path.rows >= 72:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
                if path.rows >= 5:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
            if path.rows >= 1024:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
            if path.rows >= 768:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 512:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 384:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
            if path.rows >= 256:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
            if path.rows >= 192:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows >= 160:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 1)
            if path.rows >= 112:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
            if path.rows >= 96:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 5)
            if path.rows >= 72:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 48:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 6)
            if path.rows >= 32:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows >= 24:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 6)
            if path.rows >= 12:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows >= 5:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
        if group == "ffn_key":
            if C == 2560 and 17 <= path.rows <= 20:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if C == 768:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                if 96 <= path.rows < 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
            if C == 1024:
                if 256 <= path.rows < 384:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 2)
                if 192 <= path.rows < 256:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
                if 96 <= path.rows < 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 2)
            if C == 2048 and 128 <= path.rows < 160:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 3)
            if C == 2560:
                if path.rows >= 192:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 5)
                if path.rows >= 160:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 4)
                if path.rows >= 128:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 5)
                if path.rows >= 112:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 4)
                if path.rows >= 96:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 4)
                if path.rows >= 80:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 3)
                if path.rows >= 72:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
                if path.rows >= 3:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
            if path.rows >= 1024:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows >= 768:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 512:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 3)
            if path.rows >= 384:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 256:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 4)
            if path.rows >= 192:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
            if path.rows >= 160:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
            if path.rows >= 128:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 112:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 3)
            if path.rows >= 96:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 72:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 1)
            if path.rows >= 48:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
            if path.rows >= 12:
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows in (5, 6):
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
        return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)

    def linear_rank_in(self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int) -> torch.Tensor:
        if weight_t is not None and rows <= LOWRANK_IN_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        return self.linear_lowrank_orig(x, weight) if weight is not None else self.linear_t_orig(x, weight_t)

    def linear_rank_out(self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int) -> torch.Tensor:
        if weight_t is not None and C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        return self.linear_lowrank_orig(x, weight) if weight is not None else self.linear_t_orig(x, weight_t)

    def linear_rank_out_act(self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int, act: int) -> torch.Tensor:
        if weight_t is not None and C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_act_f16(x.contiguous(), weight_t, act)
        ops = torch.ops.rwkv7_fast_ops_fp16
        x = ops.act_tanh(x.contiguous()) if act == 1 else ops.act_sigmoid(x.contiguous())
        return self.linear_lowrank_orig(x.contiguous(), weight) if weight is not None else self.linear_t_orig(x, weight_t)

    def linear_lowrank_orig(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.linear_f16(x.contiguous(), weight)

    def linear_t_orig(self, x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight_t)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.add_f16(x.contiguous(), y.contiguous())

    def add_ln(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_f16(x.contiguous(), residual.contiguous(), weight, bias)
        return outs[0], outs[1]

    def add_last_ln(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.add_last_layer_norm_f16(x.contiguous(), residual.contiguous(), weight, bias)

def bench_case(model: RWKV7, B: int, T: int, warmup: int, iters: int, profile_range: bool) -> None:
    def percentile(values: list[float], q: float) -> float:
        return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q / 100.0).item())

    # bench_case does its own graph capture — disable auto-graph to avoid conflict
    old_graph_auto = os.environ.get("CUDA_GRAPH_AUTO", "1")
    os.environ["CUDA_GRAPH_AUTO"] = "0"
    state = model.zero_state(B)
    token_device = "cpu" if model.emb_cpu else first_device()
    tokens = torch.arange(B*T, dtype=torch.long, device=token_device).view(B,T)
    tokens = (tokens * 1103515245 + 12345) % V
    path = select_path(B, T)
    x = model.embed(tokens) if (model.emb_cpu or pp_enabled()) else None
    for _ in range(warmup):
        if x is None:
            model.forward(tokens, state)
        else:
            model.forward_from_x(x, state, path)
    sync_all()

    if pp_enabled():
        segments = pp_segments()
        stage_inputs = []
        stage_vfirst = []
        stage_outputs = []
        stage_graphs = []
        v_first_out = None
        prev = x
        for stage, (start, end) in enumerate(segments):
            dev = layer_device(start)
            with torch.cuda.device(dev):
                inp = torch.empty((B,T,C), dtype=DTYPE, device=dev)
                inp.copy_(prev.to(dev))
                vin = None
                if start > 0:
                    vin = torch.empty((B,T,C), dtype=DTYPE, device=dev)
                    vin.copy_(v_first_out.to(dev))
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream(device=dev)
                stream.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(stream):
                    warm_out, warm_vf = model.forward_pp_segment(inp, state, path, start, end, vin)
                    if end == L:
                        model.forward_pp_tail(warm_out, state, T, advance=False)
                torch.cuda.current_stream(dev).wait_stream(stream)
                with torch.cuda.graph(graph, stream=stream):
                    seg_out, vf = model.forward_pp_segment(inp, state, path, start, end, vin)
                    out = model.forward_pp_tail(seg_out, state, T, advance=False) if end == L else seg_out
                stage_inputs.append(inp)
                stage_vfirst.append(vin)
                stage_outputs.append((out, vf, end == L))
                stage_graphs.append(graph)
                prev = seg_out
                if vf is not None:
                    v_first_out = vf
        sync_all()

        times = []
        if profile_range:
            torch.cuda.cudart().cudaProfilerStart()
        for _ in range(iters):
            t0 = time.perf_counter()
            with torch.cuda.device(layer_device(segments[0][0])):
                stage_inputs[0].copy_(x, non_blocking=True)
            for stage, graph in enumerate(stage_graphs):
                dev = layer_device(segments[stage][0])
                with torch.cuda.device(dev):
                    graph.replay()
                out, vf, final_stage = stage_outputs[stage]
                if vf is not None:
                    v_first_out = vf
                if not final_stage:
                    next_start = segments[stage + 1][0]
                    next_dev = layer_device(next_start)
                    with torch.cuda.device(next_dev):
                        stage_inputs[stage + 1].copy_(out, non_blocking=True)
                        if stage_vfirst[stage + 1] is not None:
                            stage_vfirst[stage + 1].copy_(v_first_out, non_blocking=True)
            model.advance_pp_elapsed(state, T)
            sync_all()
            times.append((time.perf_counter() - t0) * 1000.0)
        if profile_range:
            torch.cuda.cudart().cudaProfilerStop()
        p10 = percentile(times, 10)
        p50 = percentile(times, 50)
        p90 = percentile(times, 90)
        tok_s = B*T*1000.0 / p50
        print(f"RESULT B={B} T={T} iters={iters} p10_ms={p10:.4f} p50_ms={p50:.4f} p90_ms={p90:.4f} tok_s_p50={tok_s:.2f}", flush=True)
        print(f"csv,rwkv7_fast_v3a_pp,{B},{T},{iters},{p10:.6f},{p50:.6f},{p90:.6f},{tok_s:.6f}", flush=True)
        os.environ["CUDA_GRAPH_AUTO"] = old_graph_auto
        return

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if x is None:
            model.forward(tokens, state)
        else:
            model.forward_from_x(x, state, path)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        if x is None:
            model.forward(tokens, state)
        else:
            model.forward_from_x(x, state, path)
    torch.cuda.synchronize()

    times = []
    if profile_range:
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    if profile_range:
        torch.cuda.cudart().cudaProfilerStop()

    p10 = percentile(times, 10)
    p50 = percentile(times, 50)
    p90 = percentile(times, 90)
    tok_s = B*T*1000.0 / p50
    print(f"RESULT B={B} T={T} iters={iters} p10_ms={p10:.4f} p50_ms={p50:.4f} p90_ms={p90:.4f} tok_s_p50={tok_s:.2f}", flush=True)
    print(f"csv,rwkv7_fast_v3a,{B},{T},{iters},{p10:.6f},{p50:.6f},{p90:.6f},{tok_s:.6f}", flush=True)
    os.environ["CUDA_GRAPH_AUTO"] = old_graph_auto

def run_eval(model: RWKV7, eval_json: str, eval_out: str, logits_out: str, paths: str) -> None:
    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = data["tokens"]
    outputs = {}
    for path in paths.replace(",", " ").split():
        token_device = "cpu" if model.emb_cpu else first_device()
        targets = torch.tensor(ids[1:], dtype=torch.long, device=last_device() if pp_enabled() else "cuda")
        state = model.zero_state(1)
        sync_all()
        t0 = time.perf_counter()
        if path == "b1tn":
            tokens = torch.tensor(ids[:-1], dtype=torch.long, device=token_device).view(1, -1)
            logits = model.forward_all_logits(tokens, state).squeeze(0).float()
            loss = F.cross_entropy(logits, targets, reduction="none")
        elif path == "b1t1":
            losses = []
            for i, tok in enumerate(ids[:-1]):
                token = torch.tensor([[tok]], dtype=torch.long, device=token_device)
                logits = model.forward(token, state).float()
                losses.append(F.cross_entropy(logits, targets[i:i + 1], reduction="none"))
            loss = torch.cat(losses)
            logits = None
        else:
            raise ValueError(f"unknown eval path: {path}")
        sync_all()
        dt = time.perf_counter() - t0
        loss_cpu = loss.detach().cpu()
        p90 = torch.quantile(loss_cpu.float(), 0.90).item()
        p99 = torch.quantile(loss_cpu.float(), 0.99).item()
        tok_s = loss_cpu.numel() / dt
        print(
            f"EVAL label=rwkv7_fast_v3a path={path} positions={loss_cpu.numel()} "
            f"mean_loss={loss_cpu.mean().item():.8f} p90_loss={p90:.8f} p99_loss={p99:.8f} "
            f"max_loss={loss_cpu.max().item():.8f} min_loss={loss_cpu.min().item():.8f} "
            f"time_s={dt:.3f} tok_s={tok_s:.3f}",
            flush=True,
        )
        if logits_out and path == "b1tn":
            torch.save(logits.detach().cpu(), logits_out)
        outputs[path] = {
            "label": "rwkv7_fast_v3a",
            "path": path,
            "tokens": ids,
            "loss": loss_cpu,
            "mean_loss": float(loss_cpu.mean().item()),
            "p90_loss": float(p90),
            "p99_loss": float(p99),
            "max_loss": float(loss_cpu.max().item()),
            "min_loss": float(loss_cpu.min().item()),
            "time_s": float(dt),
        }
    if eval_out:
        torch.save(outputs, eval_out)

if __name__ == "__main__":
    main()
