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
WKV_FP16_POLICY = "tuned"
WKV_BH_GRID_MODE = "tuned"
ADD_VEC_MODE = "tuned"
LAST_LN_MODE = "indexed"
LNX_MODE = "tuned"
LN_OWNER_MODE = "tuned"
LN_STATS_MODE = "tuned"
CMIX_LN_STATS_MODE = "tuned"
CMIX_MIX_MODE = "tuned"
CMIX_VALUE_LOOP_MODE = "tuned"
CMIX_T512_ACCUM_MODE = "tuned"
CMIX_T512_REUSE_MODE = "tuned"
TMIX_MIX_MODE = "tuned"
HEAD_GRID_MODE = "tuned"
HEAD_ALL_LOGITS_GEMM_MODE = "tuned"
HEAD_LAST_LOGITS_GEMM_MODE = "tuned"
FFN_DOWN_GEMM_MODE = "tuned"
ORIG_DENSE_GEMM_MODE = "tuned"
VRES_GATE_MODE = "tuned"
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
LOWRANK_GEMM_MODE = "tuned"
CMIX_NOFC_ROW20_MAX_T = 5
CMIX_NOFC_T512_MIN_ROWS = 8
LN1_TMIX_FUSE = True
CMIX_B1T1_SPARSE = "b1t1_sparse"
CMIX_ROWS2_SPARSE = "rows2_sparse"
CMIX_B1T1_NOFC = "b1t1_nofc"
CMIX_ROWS2_NOFC = "rows2_nofc"
CMIX_DENSE = "dense"
LNX_WARP_MIN_HEAD_TASKS = 4096
LNX_WARP_B1_T_4096 = frozenset((64, 96, 128, 160, 192, 240, 248, 264, 512))
CMIX_MIX_3D_B1_T_4096 = frozenset((2, 4, 16, 64, 512))
TMIX_MIX_3D_B1_T_4096 = frozenset((2, 4, 16, 64, 512))
CMIX_VALUE_SPLIT2_BT_4096 = frozenset({
    (1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (2, 2),
    (1, 5), (5, 1), (3, 2), (1, 7), (7, 1),
})
CMIX_T512_ACC2_BT_4096 = frozenset({(8, 1), (4, 2), (3, 3), (19, 1)})
CMIX_T512_REUSE_BT_4096 = frozenset({
    (1, 8), (2, 4), (4, 2), (8, 1),
    (3, 3),
    (1, 11), (11, 1),
    (1, 12), (3, 4), (4, 3), (12, 1),
    (1, 13), (13, 1),
    (1, 14), (14, 1),
    (1, 15),
    (1, 16), (2, 8), (4, 4), (8, 2), (16, 1),
    (1, 17),
    (1, 19), (19, 1),
    (4, 5), (5, 4), (10, 2), (20, 1),
})

# Current-environment all-logits head winners. These must not affect the usual
# last-logits path: the same row count can mean either B or B*T. Lt indices are
# GPU/CUDA-specific, so linear_f16_orig_lt_cfg deliberately falls back to algo 0
# when a requested index is unavailable after an environment change.
HEAD_ALL_LOGITS_GEMM_4096 = {
    24: (0, 0),
    32: (0, 0),
    160: (0, 7),
    192: (0, 5),
}

# Ordinary last-logits head winners measured with the complete real model in
# both graph-capture orders.  The exact (B,T) guard is intentional: this GEMM
# has B rows regardless of T, while its fixed saving falls below graph-level
# noise as the body grows.  Do not broaden this to a B-only dispatch table.
HEAD_LAST_LOGITS_GEMM_4096 = {
    (24, 1): (0, 0),
    (24, 2): (0, 0),
    (24, 4): (0, 0),
    (24, 8): (0, 0),
    (32, 1): (0, 0),
    (32, 2): (0, 0),
    (32, 4): (0, 0),
    (32, 8): (0, 0),
    (32, 16): (0, 0),
    (160, 1): (0, 7),
    (160, 2): (0, 7),
    (160, 4): (0, 7),
    (160, 32): (0, 7),
    (192, 1): (0, 5),
    (192, 2): (0, 5),
    (192, 4): (0, 5),
    (192, 8): (0, 5),
    (192, 16): (0, 5),
    (192, 32): (0, 5),
}

# Runtime-layout FFN down winners on Blackwell/CUDA 13.2.  The heuristic index
# is environment-specific, so the production op intentionally falls back to
# algo 0 if an index disappears.  Exact shape/rows guards prevent accidental
# reuse by another model size; this table adds no persistent weight copy.
FFN_DOWN_GEMM_4096 = {
    48: (32, 1),
    256: (32, 5),
}

# Exact original-layout dense winners on Blackwell/CUDA 13.2.  These replace
# severe heuristic cliffs in the older rows-range table.  Keep both the C/N/K
# guards and the non-strict production Lt call: heuristic indices are not ABI
# and may disappear after a CUDA, driver, or GPU change.
ORIG_ATT_C2C_GEMM_4096 = {
    24: ("lt", 0, 0),
    32: ("gemmex", 0, 0),
    48: ("lt", 32, 0),
    64: ("lt", 32, 0),
    96: ("gemmex", 0, 0),
    192: ("lt", 32, 2),
}
ORIG_FFN_KEY_GEMM_4096 = {
    24: ("lt", 0, 2),
    32: ("lt", 0, 2),
    128: ("lt", 32, 3),
    192: ("lt", 0, 3),
    384: ("lt", 0, 2),
}

# Exact B/T overrides admitted on the C=4096,H=64 model only. Keep this sparse:
# WKV task supply depends on B*H while the recurrent critical path depends on T,
# so neither a rows-only threshold nor cross-model reuse is justified.
WKV_FP16_TUNED_OVERRIDES = {
    (2, 32): ("fused", "exact"),
    (4, 16): ("fused", "exact"),
    (4, 64): ("fused", "exact"),
    (8, 8): ("fused", "exact"),
}

# Tuned table for the 4096-wide 7B model, admitted by real B/T E2E and eval_src2
# on 2026-07-14. cuBLASLt heuristic indices are CUDA/GPU specific; production
# ops must keep the algo-0 fallback because an index can disappear after a CUDA,
# driver, or GPU change. Operator-group scans alone are not an admission test.
LOWRANK_IN_GEMM_4096 = {
    (128, 8): ("orig_lt", 32, 2),
    (128, 16): ("orig_lt", 32, 1),
    (128, 48): ("orig_lt", 32, 1),
    (128, 64): ("orig_lt", 128, 2),
    (128, 96): ("orig_gemmex", 0, 0),
    (128, 128): ("orig_gemmex", 0, 0),
    (128, 192): ("orig_lt", 128, 1),
    (128, 256): ("orig_lt", 128, 6),
    (128, 512): ("orig_lt", 128, 0),
    (128, 1024): ("orig_lt", 32, 1),
    (480, 8): ("orig_lt", 128, 0),
    (480, 16): ("runtime_lt", 128, 1),
    (480, 24): ("orig_lt", 128, 5),
    (480, 32): ("orig_lt", 32, 1),
    (480, 48): ("orig_lt", 128, 5),
    (480, 96): ("orig_lt", 128, 4),
    (480, 128): ("orig_lt", 32, 4),
    (480, 192): ("orig_lt", 128, 0),
    (480, 256): ("orig_lt", 32, 1),
    (480, 512): ("orig_lt", 128, 1),
    (96, 8): ("orig_lt", 32, 2),
    (96, 16): ("orig_lt", 128, 1),
    (96, 96): ("orig_lt", 128, 2),
    (96, 128): ("orig_lt", 32, 0),
    (96, 192): ("runtime_lt", 32, 2),
    (96, 256): ("orig_lt", 32, 1),
    (96, 512): ("orig_gemmex", 0, 0),
    (96, 1024): ("orig_lt", 128, 1),
}
LOWRANK_OUT_GEMM_4096 = {
    (128, 8): ("runtime_lt", 128, 3),
    (128, 16): ("runtime_lt", 128, 2),
    (128, 24): ("orig_lt", 128, 0),
    (128, 32): ("orig_lt", 128, 2),
    (128, 48): ("orig_lt", 0, 5),
    (128, 64): ("orig_lt", 0, 2),
    (128, 96): ("orig_lt", 32, 3),
    (128, 192): ("orig_lt", 128, 2),
    (128, 256): ("runtime_lt", 128, 3),
    (128, 512): ("runtime_lt", 128, 1),
    (128, 1024): ("runtime_lt", 128, 1),
    (480, 8): ("orig_lt", 0, 1),
    (480, 16): ("orig_lt", 0, 1),
    (480, 24): ("orig_lt", 128, 0),
    (480, 32): ("orig_gemmex", 0, 0),
    (480, 96): ("runtime_lt", 128, 1),
    (480, 128): ("runtime_lt", 32, 0),
    (480, 256): ("runtime_lt", 32, 2),
    (480, 512): ("runtime_lt", 128, 1),
    (480, 1024): ("runtime_lt", 0, 1),
    (96, 8): ("runtime_lt", 128, 5),
    (96, 16): ("runtime_lt", 32, 4),
    (96, 24): ("orig_gemmex", 0, 0),
    (96, 32): ("orig_lt", 128, 1),
    (96, 48): ("orig_lt", 128, 3),
    (96, 64): ("orig_lt", 128, 2),
    (96, 96): ("orig_lt", 32, 3),
    (96, 128): ("runtime_lt", 32, 2),
    (96, 256): ("runtime_lt", 128, 4),
    (96, 512): ("runtime_lt", 128, 0),
    (96, 1024): ("runtime_lt", 128, 2),
}

def main() -> None:
    global MODEL_PATH, WKV_MODE, WKV_FP16_POLICY, WKV_BH_GRID_MODE, ADD_VEC_MODE, LAST_LN_MODE, LNX_MODE, LN_OWNER_MODE, LN_STATS_MODE, CMIX_LN_STATS_MODE, CMIX_MIX_MODE, CMIX_VALUE_LOOP_MODE, CMIX_T512_ACCUM_MODE, CMIX_T512_REUSE_MODE, TMIX_MIX_MODE, HEAD_GRID_MODE, HEAD_ALL_LOGITS_GEMM_MODE, HEAD_LAST_LOGITS_GEMM_MODE, FFN_DOWN_GEMM_MODE, ORIG_DENSE_GEMM_MODE, VRES_GATE_MODE, EMB_DEVICE, RKV_MODE, CMIX_SPARSE, LOWRANK_WEIGHT, LOWRANK_GEMM_MODE, ORIG_LINEAR_GROUPS, PP_DEVICES
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
    parser.add_argument("--wkv", choices=("fp16", "fp32io16"), default="fp16") # fp32io16 is more accurate
    parser.add_argument("--wkv-fp16-policy", choices=("current", "tuned"), default=WKV_FP16_POLICY)
    parser.add_argument("--wkv-bh-grid", choices=("current", "tuned"), default=WKV_BH_GRID_MODE)
    parser.add_argument("--add-vec", choices=("current", "tuned"), default=ADD_VEC_MODE)
    parser.add_argument("--last-ln", choices=("current", "indexed"), default=LAST_LN_MODE)
    parser.add_argument("--lnx", choices=("current", "tuned", "warp"), default=LNX_MODE)
    parser.add_argument("--ln-owner", choices=("current", "tuned"), default=LN_OWNER_MODE)
    parser.add_argument(
        "--ln-stats", choices=("current", "tuned", "welford", "welford-cache", "centered-cache"),
        default=LN_STATS_MODE)
    parser.add_argument(
        "--cmix-ln-stats", choices=("current", "tuned", "welford", "welford-cache"),
        default=CMIX_LN_STATS_MODE)
    parser.add_argument("--cmix-mix", choices=("current", "tuned", "grid3d"), default=CMIX_MIX_MODE)
    parser.add_argument(
        "--cmix-value-loop", choices=("current", "tuned", "split2"),
        default=CMIX_VALUE_LOOP_MODE)
    parser.add_argument(
        "--cmix-t512-accum", choices=("current", "tuned", "acc2", "acc4"),
        default=CMIX_T512_ACCUM_MODE)
    parser.add_argument(
        "--cmix-t512-reuse", choices=("current", "tuned", "reuse"),
        default=CMIX_T512_REUSE_MODE)
    parser.add_argument("--tmix-mix", choices=("current", "tuned", "grid3d"), default=TMIX_MIX_MODE)
    parser.add_argument(
        "--head-grid", choices=("current", "tuned", "kk2d", "lnx2d", "grid2d"),
        default=HEAD_GRID_MODE)
    parser.add_argument(
        "--head-all-logits-gemm", choices=("current", "tuned"),
        default=HEAD_ALL_LOGITS_GEMM_MODE)
    parser.add_argument(
        "--head-last-logits-gemm", choices=("current", "tuned"),
        default=HEAD_LAST_LOGITS_GEMM_MODE)
    parser.add_argument(
        "--ffn-down-gemm", choices=("current", "tuned"),
        default=FFN_DOWN_GEMM_MODE)
    parser.add_argument(
        "--orig-dense-gemm", choices=("current", "tuned"),
        default=ORIG_DENSE_GEMM_MODE)
    parser.add_argument("--vres-gate", choices=("current", "tuned"), default=VRES_GATE_MODE)
    parser.add_argument("--emb", choices=("gpu", "cpu"), default="cpu") # cpu is fast too, and saves VRAM
    parser.add_argument("--batched-rkv", choices=("auto", "on", "off"), default="off") # auto is slightly faster but consumes lots of VRAM
    parser.add_argument("--cmix-sparse", choices=("auto", "no-fc", "off"), default="no-fc") # auto is slightly faster but consumes lots of VRAM
    parser.add_argument("--lowrank-weight", choices=("orig", "transpose", "both"), default="both") # orig saves VRAM but slows tiny B*T; transpose saves VRAM but slows large B*T
    parser.add_argument("--lowrank-gemm", choices=("current", "tuned"), default=LOWRANK_GEMM_MODE)
    parser.add_argument("--orig-linear-groups", default="att_c2c,ffn_key,head") # comma list: none, att_c2c, ffn_key, head
    parser.add_argument("--pp-devices", default="") # comma list, e.g. 0,1. Empty means single GPU.
    args = parser.parse_args()

    MODEL_PATH = args.model
    WKV_MODE = args.wkv
    WKV_FP16_POLICY = args.wkv_fp16_policy
    WKV_BH_GRID_MODE = args.wkv_bh_grid
    ADD_VEC_MODE = args.add_vec
    LAST_LN_MODE = args.last_ln
    LNX_MODE = args.lnx
    LN_OWNER_MODE = args.ln_owner
    LN_STATS_MODE = args.ln_stats
    CMIX_LN_STATS_MODE = args.cmix_ln_stats
    CMIX_MIX_MODE = args.cmix_mix
    CMIX_VALUE_LOOP_MODE = args.cmix_value_loop
    CMIX_T512_ACCUM_MODE = args.cmix_t512_accum
    CMIX_T512_REUSE_MODE = args.cmix_t512_reuse
    TMIX_MIX_MODE = args.tmix_mix
    HEAD_GRID_MODE = args.head_grid
    HEAD_ALL_LOGITS_GEMM_MODE = args.head_all_logits_gemm
    HEAD_LAST_LOGITS_GEMM_MODE = args.head_last_logits_gemm
    FFN_DOWN_GEMM_MODE = args.ffn_down_gemm
    ORIG_DENSE_GEMM_MODE = args.orig_dense_gemm
    VRES_GATE_MODE = args.vres_gate
    EMB_DEVICE = args.emb
    RKV_MODE = args.batched_rkv
    CMIX_SPARSE = args.cmix_sparse
    LOWRANK_WEIGHT = args.lowrank_weight
    LOWRANK_GEMM_MODE = args.lowrank_gemm
    ORIG_LINEAR_GROUPS = parse_orig_linear_groups(args.orig_linear_groups)
    PP_DEVICES = parse_pp_devices(args.pp_devices)
    groups = ",".join(sorted(ORIG_LINEAR_GROUPS)) if ORIG_LINEAR_GROUPS else "none"
    pp = ",".join(str(x) for x in PP_DEVICES) if PP_DEVICES else "off"
    log(f"start model={MODEL_PATH} wkv={WKV_MODE} wkv_fp16_policy={WKV_FP16_POLICY} wkv_bh_grid={WKV_BH_GRID_MODE} add_vec={ADD_VEC_MODE} last_ln={LAST_LN_MODE} lnx={LNX_MODE} ln_owner={LN_OWNER_MODE} ln_stats={LN_STATS_MODE} cmix_ln_stats={CMIX_LN_STATS_MODE} cmix_mix={CMIX_MIX_MODE} cmix_value_loop={CMIX_VALUE_LOOP_MODE} cmix_t512_accum={CMIX_T512_ACCUM_MODE} cmix_t512_reuse={CMIX_T512_REUSE_MODE} tmix_mix={TMIX_MIX_MODE} head_grid={HEAD_GRID_MODE} head_all_logits_gemm={HEAD_ALL_LOGITS_GEMM_MODE} head_last_logits_gemm={HEAD_LAST_LOGITS_GEMM_MODE} ffn_down_gemm={FFN_DOWN_GEMM_MODE} orig_dense_gemm={ORIG_DENSE_GEMM_MODE} vres_gate={VRES_GATE_MODE} emb={EMB_DEVICE} batched_rkv={RKV_MODE} cmix_sparse={CMIX_SPARSE} lowrank_weight={LOWRANK_WEIGHT} lowrank_gemm={LOWRANK_GEMM_MODE} orig_linear_groups={groups} pp={pp}")
    log(f"fixed fast path: ln=v3a linear=v3a/splitk lowrank={LOWRANK_IN_ROWS_T}/{LOWRANK_OUT_ROWS_T} nofc_rows=by_C row20_t=by_C nofc_t512_rows>={CMIX_NOFC_T512_MIN_ROWS} t512_acc2=exact_BT t512_reuse=exact_BT")
    load_extensions(WKV_MODE)
    model = RWKV7()
    if args.eval_json:
        run_eval(model, args.eval_json, args.eval_out, args.eval_all_logits_out, args.eval_paths)
        return
    print("csv_header,label,B,T,iters,p10_ms,p50_ms,p90_ms,tok_s_p50", flush=True)
    for item in args.cases.replace(",", " ").split():
        B, T = [int(x) for x in item.lower().split("x", 1)]
        bench_case(model, B, T, args.warmup, args.iters, args.profile_range)


def use_wkv_bh_grid_2d(B: int, T: int, C: int, H: int) -> bool:
    if (
        WKV_BH_GRID_MODE != "tuned"
        or C != 4096
        or H != 64
        or B <= 0
        or B > 65535
        # CUDA bodies inherited signed-int token/state address arithmetic.
        or B * T * C > 2**31 - 1
        or B * C * 64 > 2**31 - 1
    ):
        return False
    if T == 1:
        return B <= 16
    if (B, T) in WKV_FP16_TUNED_OVERRIDES:
        return True
    # This must mirror use_v2_seq() in rwkv7_wkv_fp16_v2.cu. The 2D grid is
    # faster for exact but slower for seq; seq also gains one register, but no
    # occupancy change is claimed without profiler evidence. Keep it flat.
    return not (
        (B == 1 and T >= 8)
        or (B == 4 and T >= 4)
        or (B == 8 and T >= 8)
    )


def use_warp_lnx(B: int, T: int, C: int, H: int) -> bool:
    if LNX_MODE == "warp":
        return True
    if LNX_MODE != "tuned" or C != 4096 or H != 64 or B * T * H < LNX_WARP_MIN_HEAD_TASKS:
        return False
    # B1 long-context E2E is unusually graph-pool/capture-order sensitive even
    # when the operator is faster. Admit only T values positive in both orders.
    return B >= 2 or T in LNX_WARP_B1_T_4096

def use_add_vec_2d(rows: int, channels: int) -> bool:
    return ADD_VEC_MODE == "tuned" and channels == 4096 and 17 <= rows <= 65535

def use_tuned_ln_owner(rows: int, channels: int) -> bool:
    return LN_OWNER_MODE == "tuned" and channels == 4096 and rows < 1024

def tuned_ln_stats_mode(batch: int, rows: int, channels: int) -> int | None:
    if LN_STATS_MODE == "current" or channels != 4096:
        return None
    if LN_STATS_MODE == "welford":
        return 0
    if LN_STATS_MODE == "welford-cache":
        return 1
    if LN_STATS_MODE == "centered-cache":
        return 2
    # Same-graph E2E is stable across BnTn/BnT1 factorizations for B>=2 and
    # rows<=1024. B1Tn remains capture-layout sensitive, so tuned must not
    # silently change that path until it has an independent stable gate.
    if batch < 2 or rows > 1024:
        return None
    # Direct Welford retains unrounded FP32 sums and is strongest on small row
    # sets. At higher task supply, caching the final FP16 residual removes one
    # more input pass and has a substantially larger operator margin.
    return 0 if rows < 192 else 1

def tuned_cmix_ln_stats_mode(batch: int, channels: int) -> int | None:
    if CMIX_LN_STATS_MODE == "current" or channels != 4096:
        return None
    if CMIX_LN_STATS_MODE == "welford":
        return 0
    if CMIX_LN_STATS_MODE == "welford-cache":
        return 1
    # The fused op is only called at T=1, so rows == batch. Operator gains at
    # smaller B do not survive the full graph; both capture orders are positive
    # only from B=192 onward.
    return 1 if 192 <= batch <= 1024 else None

def use_cmix_mix_3d(batch: int, tokens: int, channels: int) -> bool:
    if CMIX_MIX_MODE == "current" or tokens == 1:
        return False
    if CMIX_MIX_MODE == "grid3d":
        return batch <= 65535 and tokens <= 65535
    # The production model is C=4096. Keep tuned conservative until other
    # widths have their own real-model E2E evidence, even though the kernel is generic.
    if channels != 4096 or batch > 65535 or tokens > 65535:
        return False
    # B1 long-context is capture-pool/order sensitive. T=256 flipped sign by
    # capture order despite a bitwise-exact, 1.2x operator, so only admit the
    # B1 points that were positive in both orders. B>=2 had broad factorization coverage.
    return batch >= 2 or tokens in CMIX_MIX_3D_B1_T_4096

def use_cmix_value_split2(
    batch: int, tokens: int, channels: int, hidden: int
) -> bool:
    if CMIX_VALUE_LOOP_MODE == "current":
        return False
    if CMIX_VALUE_LOOP_MODE == "split2":
        return True
    # Arithmetic association and sparse task supply both vary with B/T. Admit
    # only shapes positive in both real-model capture orders; B4T1 and B2T3
    # each failed one order, and unmeasured factorizations are not extrapolated.
    return (
        channels == 4096 and hidden == 16384 and
        (batch, tokens) in CMIX_VALUE_SPLIT2_BT_4096)

def cmix_t512_accumulators(
    batch: int, tokens: int, channels: int, hidden: int
) -> int:
    if channels != 4096 or hidden != 16384:
        return 1
    if CMIX_T512_ACCUM_MODE == "acc2":
        return 2
    if CMIX_T512_ACCUM_MODE == "acc4":
        return 4
    # The fixed savings are only a few microseconds per full graph. Keep the
    # production table exact: these four shapes survived two independent
    # 8x5 rounds in both capture orders, while nearby B/T factorizations did not.
    if CMIX_T512_ACCUM_MODE == "tuned" and (batch, tokens) in CMIX_T512_ACC2_BT_4096:
        return 2
    return 1

def use_cmix_t512_reuse(
    batch: int, tokens: int, channels: int, hidden: int
) -> bool:
    if CMIX_T512_REUSE_MODE == "current":
        return False
    if CMIX_T512_REUSE_MODE == "reuse":
        return True
    # Retaining each warp's ballot removes one shared reload/test/ballot and
    # has no register or occupancy cost. The graph-level saving is nevertheless
    # B/T-sensitive, so only exact shapes positive in both balanced slot orders
    # are admitted; nearby factorizations are intentionally not extrapolated.
    return (
        channels == 4096 and hidden == 16384 and
        (batch, tokens) in CMIX_T512_REUSE_BT_4096)

def use_tmix_mix6_3d(batch: int, tokens: int, channels: int) -> bool:
    if TMIX_MIX_MODE == "current" or tokens == 1:
        return False
    if TMIX_MIX_MODE == "grid3d":
        return batch <= 65535 and tokens <= 65535
    if channels != 4096 or batch > 65535 or tokens > 65535:
        return False
    # The generic kernel is exact and broadly faster, but only admit shapes
    # that survived full-model timing. B1 points outside this set have no such
    # evidence yet; keep their launch layout unchanged instead of extrapolating.
    return batch >= 2 or tokens in TMIX_MIX_3D_B1_T_4096

def use_kk_head_grid_2d(batch: int, tokens: int, channels: int, heads: int) -> bool:
    if HEAD_GRID_MODE in ("current", "lnx2d"):
        return False
    if HEAD_GRID_MODE == "tuned":
        return (
            channels == 4096 and heads == 64 and
            batch > 0 and tokens > 0 and batch * tokens <= 65535)
    return (
        channels == heads * 64 and heads % 4 == 0 and
        batch > 0 and tokens > 0 and batch * tokens <= 65535)

def use_lnx_head_grid_2d(batch: int, tokens: int, channels: int, heads: int) -> bool:
    if HEAD_GRID_MODE in ("current", "kk2d"):
        return False
    if HEAD_GRID_MODE == "tuned":
        return False
    return channels == heads * 64 and batch > 0 and tokens > 0 and batch * tokens <= 65535

def tuned_vres_gate_threads(rows: int, channels: int) -> int | None:
    if VRES_GATE_MODE != "tuned" or channels != 4096 or rows < 64 or rows > 65535:
        return None
    # At C=4096 vec2/128 preserves the scalar kernel's 16 CTAs per row while
    # halving warps per CTA. Once rows supply enough work, vec2/256 halves the
    # CTA count too. Keep this rows-based; per-layer tables are not justified.
    return 128 if rows < 256 else 256

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
    log(f"loading CUDA extensions v3a_ops + fast_ops + wkv={wkv_mode}")
    cuda_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization"] + ([] if os.name == "nt" else ["-Xptxas", "-O3"])
    load(name="rwkv7_v3a_ops", sources=[str(CUDA_DIR / "rwkv7_v3a_ops.cpp"), str(CUDA_DIR / "rwkv7_v3a_ops.cu")], is_python_module=False, verbose=True, extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags)
    load(name="rwkv7_fast_ops_fp16", sources=[str(CUDA_DIR / "rwkv7_fast_ops_fp16.cpp"), str(CUDA_DIR / "rwkv7_fast_ops_fp16.cu")], is_python_module=False, verbose=True, extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags)
    if wkv_mode == "fp16":
        load(name="rwkv7_wkv_fp16_v2", sources=[str(CUDA_DIR / "rwkv7_wkv_fp16_v2.cpp"), str(CUDA_DIR / "rwkv7_wkv_fp16_v2.cu")], is_python_module=False, verbose=True, extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas", "-O3"])
    elif wkv_mode == "fp32io16":
        load(name="rwkv7_wkv_fp32_v2", sources=[str(CUDA_DIR / "rwkv7_wkv_fp32_v2.cpp"), str(CUDA_DIR / "rwkv7_wkv_fp32_v2.cu")], is_python_module=False, verbose=True, extra_cflags=["-O3", "-D_IO_FP16_"], extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-O3", "-D_IO_FP16_"])
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
            is_lowrank = is_lowrank_weight(key)
            if ".ffn.key.weight" in key and CMIX_SPARSE == "auto":
                z[key + ".fc"] = value.to(device=dev, dtype=DTYPE).contiguous()
            if (
                not is_lowrank
                and (("key.weight" in key and not is_orig_linear_weight(key))
                or ("value.weight" in key and not is_orig_linear_weight(key))
                or ("receptance.weight" in key and not is_orig_linear_weight(key))
                or ("output.weight" in key and not is_orig_linear_weight(key))
                or ("head.weight" in key and not is_orig_linear_weight(key)))
            ):
                value = value.t()
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
        self.batch_rows_cache: dict[tuple[int, int], torch.Tensor] = {}
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
                x, mixed = self.add_ln_cmix_mix(
                    x, xx, state[0][layer][1], z[p+"ln2.weight"], z[p+"ln2.bias"], z[p+"ffn.x_k"])
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
                    if LAST_LN_MODE == "indexed":
                        x = torch.ops.rwkv7_v3a_ops.add_last_layer_norm_indexed_f16(
                            x.contiguous(), xx.contiguous(), last_indices.contiguous(),
                            z["ln_out.weight"], z["ln_out.bias"])
                    else:
                        x = self.ln(self.add(x, xx), z["ln_out.weight"], z["ln_out.bias"])
                        x = x[self.batch_rows(B, x.device), last_indices].contiguous()
                else:
                    x = self.add_last_ln(x, xx, z["ln_out.weight"], z["ln_out.bias"])
                torch.ops.rwkv7_v3a_ops.advance_i32(state[2], T) # !!! IMPORTANT FOR WKV16 DITHERING !!!
                return self.linear_head_last(x, T)
            else:
                x = self.add(x, xx)

        x = self.ln(x, z["ln_out.weight"], z["ln_out.bias"])
        torch.ops.rwkv7_v3a_ops.advance_i32(state[2], T) # !!! IMPORTANT FOR WKV16 DITHERING !!!
        return self.linear_head(x, all_logits=True)

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
                x, mixed = self.add_ln_cmix_mix(
                    x, xx, state[0][layer][1], z[p+"ln2.weight"], z[p+"ln2.bias"], z[p+"ffn.x_k"])
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
                x = x[self.batch_rows(B, x.device), last_indices].contiguous()
        x = self.ln(x, self.z["ln_out.weight"], self.z["ln_out.bias"])
        if advance:
            self.advance_pp_elapsed(state, T)
        if all_logits:
            return self.linear_head(x, all_logits=True)
        return self.linear_head_last(x, T)

    def advance_pp_elapsed(self, state: list[torch.Tensor], T: int) -> None:
        for idx, dev_id in enumerate(PP_DEVICES):
            with torch.cuda.device(dev_id):
                torch.ops.rwkv7_v3a_ops.advance_i32(state[2][idx], T)

    def batch_rows(self, B: int, device: torch.device) -> torch.Tensor:
        key = (device.index if device.index is not None else torch.cuda.current_device(), B)
        rows = self.batch_rows_cache.get(key)
        if rows is None:
            rows = torch.arange(B, dtype=torch.long, device=device)
            self.batch_rows_cache[key] = rows
        return rows

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

    def tmix(self, layer: int, x: torch.Tensor, shift_state: torch.Tensor, wkv_state: torch.Tensor, elapsed_t: torch.Tensor, v_first: torch.Tensor, p: str, path: PathConfig, pre_mix=None) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape
        if pre_mix is not None:
            xr, xw, xk, xv, xa, xg = pre_mix
        else:
            tmix_mix_op = ops.tmix_mix6_3d if use_tmix_mix6_3d(B, T, C) else ops.tmix_mix6
            xr, xw, xk, xv, xa, xg = tmix_mix_op(B, T, C, x.contiguous(), shift_state[0], z[p+"x_r"], z[p+"x_w"], z[p+"x_k"], z[p+"x_v"], z[p+"x_a"], z[p+"x_g"])
        if pre_mix is not None:
            if path.use_batched_rkv:
                flat = torch.stack((xr.reshape(-1,C), xk.reshape(-1,C), xv.reshape(-1,C)))
                rkv = torch.bmm(flat, z[p+"rkv.weight"])
                r, k, v = [t.view(B,T,C) for t in rkv.unbind(0)]
            else:
                r = self.linear_orig_layout(xr, z[p+"receptance.weight"], path, "att_c2c")
                k = self.linear_orig_layout(xk, z[p+"key.weight"], path, "att_c2c")
                v = self.linear_orig_layout(xv, z[p+"value.weight"], path, "att_c2c")
        else:
            if path.use_batched_rkv:
                flat = torch.stack((xr.reshape(-1,C), xk.reshape(-1,C), xv.reshape(-1,C)))
                rkv = torch.bmm(flat, z[p+"rkv.weight"])
                r, k, v = [t.view(B,T,C) for t in rkv.unbind(0)]
            else:
                r = self.linear_orig_layout(xr, z[p+"receptance.weight"], path, "att_c2c")
                k = self.linear_orig_layout(xk, z[p+"key.weight"], path, "att_c2c")
                v = self.linear_orig_layout(xv, z[p+"value.weight"], path, "att_c2c")

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
        kk_gate_op = ops.tmix_kk_a_gate_2d if use_kk_head_grid_2d(B, T, C, H) else ops.tmix_kk_a_gate
        k, neg_kk, kka = kk_gate_op(B, T, C, H, k.contiguous(), z[p+"k_k"], z[p+"a0"], a.contiguous(), z[p+"k_a"])

        if layer == 0:
            v_first = v
        elif not v_done:
            if LOWRANK_WEIGHT != "orig" and can_use_lowrank_out_fused(path.rows):
                if v1 is None:
                    v1 = self.linear_rank_in(xv, z.get(p+"v1"), z.get(p+"v1.t"), path.rows)
                v = torch.ops.rwkv7_v3a_ops.linear_t_vres_f16(v1.contiguous(), z[p+"v2.t"], v.contiguous(), v_first.contiguous(), z[p+"v0"])
            else:
                v12 = self.linear_rank_out(self.linear_rank_in(xv, z.get(p+"v1"), z.get(p+"v1.t"), path.rows), z.get(p+"v2"), z.get(p+"v2.t"), path.rows)
                v_contig = v.contiguous()
                v_first_contig = v_first.contiguous()
                v12_contig = v12.contiguous()
                vres_threads = tuned_vres_gate_threads(path.rows, C)
                if vres_threads is None:
                    v = ops.tmix_vres_gate(B, T, C, v_contig, v_first_contig, z[p+"v0"], v12_contig)
                else:
                    v = ops.tmix_vres_gate_cfg(
                        B, T, C, v_contig, v_first_contig, z[p+"v0"], v12_contig, vres_threads, True)

        y = torch.empty_like(r)
        if WKV_MODE == "fp32io16":
            w_raw = ops.add_vec(C, w.contiguous(), z[p+"w0"])
            torch.ops.rwkv7_wkv_fp32_v2.forward(B, T, C, H, wkv_state, r.contiguous(), w_raw.contiguous(), k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y)
        elif WKV_FP16_POLICY == "tuned" and C == 4096 and H == 64 and (B, T) in WKV_FP16_TUNED_OVERRIDES:
            w0_mode, kernel_mode = WKV_FP16_TUNED_OVERRIDES[(B, T)]
            forced_mode = 0 if kernel_mode == "exact" else 1
            if w0_mode == "fused":
                wkv_op = (
                    torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0_grid2d_forced
                    if use_wkv_bh_grid_2d(B, T, C, H)
                    else torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0_forced)
                wkv_op(
                    B, T, C, H, forced_mode, wkv_state, r.contiguous(), w.contiguous(), z[p+"w0"],
                    k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y, elapsed_t)
            else:
                w_contig = w.contiguous()
                add_vec_op = ops.add_vec_2d if use_add_vec_2d(path.rows, C) else ops.add_vec
                w_raw = add_vec_op(C, w_contig, z[p+"w0"])
                wkv_op = (
                    torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_grid2d_forced
                    if use_wkv_bh_grid_2d(B, T, C, H)
                    else torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_forced)
                wkv_op(
                    B, T, C, H, forced_mode, wkv_state, r.contiguous(), w_raw.contiguous(),
                    k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y, elapsed_t)
        elif T <= 16:
            wkv_op = (
                torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0_grid2d
                if use_wkv_bh_grid_2d(B, T, C, H)
                else torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0)
            wkv_op(B, T, C, H, wkv_state, r.contiguous(), w.contiguous(), z[p+"w0"], k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y, elapsed_t)
        else:
            w_contig = w.contiguous()
            add_vec_op = ops.add_vec_2d if use_add_vec_2d(path.rows, C) else ops.add_vec
            w_raw = add_vec_op(C, w_contig, z[p+"w0"])
            wkv_op = (
                torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_grid2d
                if use_wkv_bh_grid_2d(B, T, C, H)
                else torch.ops.rwkv7_wkv_fp16_v2.wkv_seq)
            wkv_op(B, T, C, H, wkv_state, r.contiguous(), w_raw.contiguous(), k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y, elapsed_t)
        if use_warp_lnx(B, T, C, H):
            lnx_op = ops.tmix_lnx_rkvres_xg_warp_2d if use_lnx_head_grid_2d(B, T, C, H) else ops.tmix_lnx_rkvres_xg_warp
        else:
            lnx_op = ops.tmix_lnx_rkvres_xg
        y = lnx_op(B, T, C, H, y.contiguous(), r.contiguous(), k.contiguous(), v.contiguous(), z[p+"r_k"], z[p+"ln_x.weight"], z[p+"ln_x.bias"], g.contiguous())
        return self.linear_orig_layout(y, z[p+"output.weight"], path, "att_c2c"), v_first

    def cmix(self, x: torch.Tensor, shift_state: torch.Tensor, p: str, path: PathConfig) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape

        if path.cmix_mode == CMIX_B1T1_SPARSE:
            return ops.cmix_sparse_one(C, z[p+"key.weight.fc"].size(0), x.contiguous(), shift_state[1], z[p+"x_k"], z[p+"key.weight.fc"], z[p+"value.weight"])
        if path.cmix_mode == CMIX_ROWS2_SPARSE:
            return ops.cmix_sparse_rows(B, T, C, z[p+"key.weight.fc"].size(0), x.contiguous(), shift_state[1], z[p+"x_k"], z[p+"key.weight.fc"], z[p+"value.weight"])

        cmix_mix_op = ops.cmix_mix_3d if use_cmix_mix_3d(B, T, C) else ops.cmix_mix
        mixed = cmix_mix_op(B, T, C, x.contiguous(), shift_state[1], z[p+"x_k"])
        return self.cmix_from_mixed(mixed, p, path)

    def cmix_from_mixed(self, mixed: torch.Tensor, p: str, path: PathConfig) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = mixed.shape
        hid = self.linear_orig_layout(mixed, z[p+"key.weight"], path, "ffn_key")
        if path.cmix_mode == CMIX_B1T1_NOFC:
            F = z[p+"value.weight"].size(0)
            down_op = (
                ops.cmix_sparse_down_relu_one_split2
                if use_cmix_value_split2(B, T, C, F)
                else ops.cmix_sparse_down_relu_one)
            return down_op(C, F, hid.view(-1).contiguous(), z[p+"value.weight"])
        if path.cmix_mode == CMIX_ROWS2_NOFC:
            F = z[p+"value.weight"].size(0)
            if path.rows >= CMIX_NOFC_T512_MIN_ROWS and C % 512 == 0 and F % 512 == 0:
                accumulators = cmix_t512_accumulators(B, T, C, F)
                if accumulators <= 2 and use_cmix_t512_reuse(B, T, C, F):
                    return ops.cmix_sparse_down_relu_rows_t512_reuse_cfg(
                        B, T, C, F, hid.contiguous(), z[p+"value.weight"],
                        accumulators)
                if accumulators == 1:
                    return ops.cmix_sparse_down_relu_rows_t512(
                        B, T, C, F, hid.contiguous(), z[p+"value.weight"])
                return ops.cmix_sparse_down_relu_rows_t512_cfg(
                    B, T, C, F, hid.contiguous(), z[p+"value.weight"], accumulators)
            down_op = (
                ops.cmix_sparse_down_relu_rows_split2
                if use_cmix_value_split2(B, T, C, F)
                else ops.cmix_sparse_down_relu_rows)
            return down_op(B, T, C, F, hid.contiguous(), z[p+"value.weight"])

        k = ops.relu_square(hid.contiguous())
        return self.linear_ffn_down(k, z[p+"value.weight"], path.rows)

    def linear_ffn_down(self, x: torch.Tensor, weight: torch.Tensor, rows: int) -> torch.Tensor:
        if (
            FFN_DOWN_GEMM_MODE == "tuned"
            and C == 4096
            and rows in FFN_DOWN_GEMM_4096
            and tuple(weight.shape) == (16384, 4096)
        ):
            workspace_mb, algo_index = FFN_DOWN_GEMM_4096[rows]
            return torch.ops.rwkv7_v3a_ops.linear_f16_lt_cfg(
                x.contiguous(), weight, workspace_mb, algo_index)
        return self.linear(x, weight)

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.numel() == x.size(-1) and weight.size(1) % 64 == 0:
            return torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(x.contiguous(), weight)
        return torch.ops.rwkv7_v3a_ops.linear_f16(x.contiguous(), weight)

    def linear_head(self, x: torch.Tensor, all_logits: bool = False) -> torch.Tensor:
        z = self.z
        if not use_orig_linear("head"):
            return self.linear(x, z["head.weight"])
        rows = x.numel() // C
        if all_logits and HEAD_ALL_LOGITS_GEMM_MODE == "tuned" and C == 4096:
            cfg = HEAD_ALL_LOGITS_GEMM_4096.get(rows)
            if cfg is not None:
                workspace_mb, algo_index = cfg
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), z["head.weight"], workspace_mb, algo_index)
        return self.linear_orig_layout(x, z["head.weight"], PathConfig(rows, False, CMIX_DENSE), "head")

    def linear_head_last(self, x: torch.Tensor, tokens_count: int) -> torch.Tensor:
        z = self.z
        if not use_orig_linear("head"):
            return self.linear(x, z["head.weight"])
        rows = x.numel() // C
        if (
            HEAD_LAST_LOGITS_GEMM_MODE == "tuned"
            and C == 4096
            and tuple(z["head.weight"].shape) == (65536, 4096)
        ):
            cfg = HEAD_LAST_LOGITS_GEMM_4096.get((rows, tokens_count))
            if cfg is not None:
                workspace_mb, algo_index = cfg
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), z["head.weight"], workspace_mb, algo_index)
        return self.linear_orig_layout(
            x, z["head.weight"], PathConfig(rows, False, CMIX_DENSE), "head")

    def linear_orig_layout(self, x: torch.Tensor, weight: torch.Tensor, path: PathConfig, group: str) -> torch.Tensor:
        if not use_orig_linear(group):
            return self.linear(x, weight)
        if ORIG_DENSE_GEMM_MODE == "tuned" and C == 4096:
            if group == "att_c2c" and tuple(weight.shape) == (4096, 4096):
                cfg = ORIG_ATT_C2C_GEMM_4096.get(path.rows)
            elif group == "ffn_key" and tuple(weight.shape) == (16384, 4096):
                cfg = ORIG_FFN_KEY_GEMM_4096.get(path.rows)
            else:
                cfg = None
            if cfg is not None:
                kind, workspace_mb, algo_index = cfg
                if kind == "gemmex":
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
                return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                    x.contiguous(), weight, workspace_mb, algo_index)
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
        return self.linear_lowrank_large(x, weight, weight_t, rows, "in")

    def linear_rank_out(self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int) -> torch.Tensor:
        if weight_t is not None and C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        return self.linear_lowrank_large(x, weight, weight_t, rows, "out")

    def linear_rank_out_act(self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int, act: int) -> torch.Tensor:
        if weight_t is not None and C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_act_f16(x.contiguous(), weight_t, act)
        ops = torch.ops.rwkv7_fast_ops_fp16
        x = ops.act_tanh(x.contiguous()) if act == 1 else ops.act_sigmoid(x.contiguous())
        return self.linear_lowrank_large(x.contiguous(), weight, weight_t, rows, "out")

    def linear_lowrank_large(self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int, direction: str) -> torch.Tensor:
        if LOWRANK_GEMM_MODE == "tuned" and C == 4096:
            if direction == "in":
                rank = weight.size(1) if weight is not None else weight_t.size(0)
                cfg = LOWRANK_IN_GEMM_4096.get((rank, rows))
            else:
                rank = weight.size(0) if weight is not None else weight_t.size(1)
                cfg = LOWRANK_OUT_GEMM_4096.get((rank, rows))
            if cfg is not None:
                backend, workspace_mb, algo_index = cfg
                if backend == "runtime_lt" and weight is not None:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_lt_cfg(
                        x.contiguous(), weight, workspace_mb, algo_index)
                if backend == "orig_lt" and weight_t is not None:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(
                        x.contiguous(), weight_t, workspace_mb, algo_index)
                if backend == "orig_gemmex" and weight_t is not None:
                    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight_t)
        return self.linear_lowrank_orig(x, weight) if weight is not None else self.linear_t_orig(x, weight_t)

    def linear_lowrank_orig(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.linear_f16(x.contiguous(), weight)

    def linear_t_orig(self, x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight_t)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.add_f16(x.contiguous(), y.contiguous())

    def add_ln(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x.numel() // x.size(-1)
        stats_mode = tuned_ln_stats_mode(x.size(0), rows, x.size(-1))
        if stats_mode is not None:
            outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_f16_stats_cfg(
                x.contiguous(), residual.contiguous(), weight, bias, 1.0e-5, stats_mode)
        elif use_tuned_ln_owner(rows, x.size(-1)):
            outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_f16_cfg(
                x.contiguous(), residual.contiguous(), weight, bias, 1.0e-5, 256, True)
        else:
            outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_f16(
                x.contiguous(), residual.contiguous(), weight, bias)
        return outs[0], outs[1]

    def add_ln_cmix_mix(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        shift_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        x_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = tuned_cmix_ln_stats_mode(x.size(0), x.size(-1))
        if mode is None:
            outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
                x.contiguous(), residual.contiguous(), shift_state, weight, bias, x_k)
        else:
            outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16_stats_cfg(
                x.contiguous(), residual.contiguous(), shift_state, weight, bias, x_k, 1.0e-5, mode)
        return outs[0], outs[1]

    def add_last_ln(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.add_last_layer_norm_f16(x.contiguous(), residual.contiguous(), weight, bias)

def bench_case(model: RWKV7, B: int, T: int, warmup: int, iters: int, profile_range: bool) -> None:
    def percentile(values: list[float], q: float) -> float:
        return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q / 100.0).item())

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
