#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rwkv.utils import PIPELINE

ROOT = Path(__file__).resolve().parent
LETTERS = "ABCD"
ANSWER_PATTERNS = (
    re.compile(r"\\boxed\s*\{\s*(?:\\(?:text|mathrm)\s*\{\s*)?\(?\s*([ABCD])\s*\)?\s*\}?\s*\}", re.I),
    re.compile(r"(?:final\s+answer|correct\s+answer|answer)\s*(?:(?:choice|option)\s*)?(?:is\s*|[:=]\s*)(?:(?:choice|option)\s*)?\(?\s*([ABCD])\s*\)?", re.I),
    re.compile(r"(?:(?:choice|option)\s*)?\(?\s*([ABCD])\s*\)?\s+is\s+(?:the\s+)?(?:final\s+|correct\s+)?answer", re.I),
)


@dataclass(frozen=True)
class Task:
    index: int
    problem: str
    answer: str
    domain: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple single-GPU GPQA Diamond evaluation for RWKV-7")
    p.add_argument("--model", default="/dev/shm/rwkv-g1i-7b-3275.pth")
    p.add_argument("--gpu", default="0")
    p.add_argument("--dataset", default="hendrydong/gpqa_diamond_mc")
    p.add_argument("--split", default="test")
    p.add_argument("--rollout", type=int, default=4)
    p.add_argument("--bsz", type=int, default=320)
    p.add_argument("--max-new-tokens", type=int, default=12288)
    p.add_argument("--ctx-limit", type=int, default=12288)
    p.add_argument("--temperature", type=float, default=0.96)
    p.add_argument("--top-p", type=float, default=0.76)
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--presence-penalty", type=float, default=1.0)
    p.add_argument("--frequency-penalty", type=float, default=0.1)
    p.add_argument("--penalty-decay", type=float, default=0.988)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--report-seconds", type=float, default=30.0)
    p.add_argument("--out-dir", default="")
    a = p.parse_args()
    if any(separator in str(a.gpu) for separator in (",", " ", "\t")):
        p.error("--gpu accepts exactly one GPU")
    if min(a.rollout, a.bsz, a.max_new_tokens, a.ctx_limit) < 1:
        p.error("--rollout, --bsz, --max-new-tokens, and --ctx-limit must be positive")
    if a.limit < 0 or a.report_seconds <= 0:
        p.error("--limit must be non-negative and --report-seconds must be positive")
    if a.temperature < 0 or not 0 <= a.top_p <= 1 or not 1 <= a.top_k <= 65536:
        p.error("sampling requires temperature >= 0, 0 <= top-p <= 1, and top-k >= 1")
    if a.presence_penalty < 0 or a.frequency_penalty < 0 or not 0 <= a.penalty_decay <= 1:
        p.error("penalties must be non-negative and penalty-decay must be in [0, 1]")
    if not a.out_dir:
        a.out_dir = str(ROOT / "gpqa_diamond_simple_runs" / time.strftime("%Y%m%d-%H%M%S"))
    return a


def extract_answer(text: str, require_think_close: bool = False) -> str | None:
    if require_think_close and "</think>" not in text:
        return None
    answer_text = text.rsplit("</think>", 1)[-1]
    matches = [(m.start(), m.group(1).upper()) for pattern in ANSWER_PATTERNS for m in pattern.finditer(answer_text)]
    if matches:
        return max(matches, key=lambda item: item[0])[1]
    for line in reversed(answer_text.splitlines()):
        match = re.fullmatch(r"\s*(?:final\s+answer\s*[:=]?\s*)?[\[(]?([ABCD])[\])]?[.!]?\s*", line, re.I)
        if match:
            return match.group(1).upper()
    return None


def load_tasks(dataset: str, split: str, limit: int) -> list[Task]:
    from datasets import load_dataset

    tasks = []
    for index, row in enumerate(load_dataset(dataset, split=split)):
        problem = str(row["problem"]).strip()
        answer = extract_answer(str(row["solution"]))
        if not problem or answer is None:
            raise ValueError(f"invalid GPQA row {index}")
        tasks.append(Task(index, problem, answer, str(row.get("domain", "")).strip()))
        if limit and len(tasks) >= limit:
            break
    if not tasks:
        raise ValueError("GPQA dataset is empty")
    return tasks


def prompt(task: Task) -> str:
    return f"User: {task.problem}\n\nAssistant: <think" # !!! Default !!! avg 36+%
    # return f"User✿{task.problem}✿\nBot✿<think" # !!! Better !!! avg 37+%
    # return f"User: {task.problem}\n\nAssistant: <think></think" # !!! Fast !!!


@torch.jit.script
def sample_logits(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    k = min(max(1, int(top_k)), logits.size(-1))
    if temperature <= 0.0 or top_p <= 0.0 or k == 1:
        return torch.argmax(logits, dim=-1)
    values, ids = torch.topk(logits.float(), k=k, dim=-1, sorted=True)
    probabilities = torch.softmax(values if temperature == 1.0 else values / temperature, dim=-1)
    cdf = torch.cumsum(probabilities, dim=-1)
    if top_p < 1.0:
        last = torch.argmax((cdf >= top_p).to(torch.int32), dim=-1)
        mass = cdf.gather(1, last.view(-1, 1))
    else:
        mass = cdf[:, -1:]
    picked = torch.searchsorted(cdf, torch.rand_like(mass) * mass).clamp_max(k - 1)
    return ids.gather(1, picked).view(-1)


def score(row: dict[str, Any]) -> float:
    return 1.0 if row["correct"] else 0.25 if row["prediction"] is None else 0.0


def majority(rows: list[dict[str, Any]]) -> str | None:
    predictions = [row["prediction"] for row in sorted(rows, key=lambda x: x["sample_id"]) if isinstance(row["prediction"], str) and row["prediction"] in LETTERS]
    if not predictions:
        return None
    counts = Counter(predictions)
    return max(counts, key=lambda x: (counts[x], -predictions.index(x)))


def metrics(rows: list[dict[str, Any]], rollout: int) -> dict[str, Any]:
    by_task: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task_index"], []).append(row)
    complete = [group for group in by_task.values() if len(group) == rollout]
    majority_scores = []
    pass_scores = []
    for group in complete:
        prediction = majority(group)
        majority_scores.append(1.0 if prediction == group[0]["answer"] else 0.25 if prediction is None else 0.0)
        missing = sum(row["prediction"] is None for row in group)
        pass_scores.append(1.0 if any(row["correct"] for row in group) else 1.0 - 0.75**missing)
    total = len(rows)
    return {
        "routes": total,
        "tasks": len(complete),
        f"avg@{rollout}": sum(map(score, rows)) / total if total else None,
        f"majority@{rollout}": sum(majority_scores) / len(majority_scores) if majority_scores else None,
        f"pass@{rollout}": sum(pass_scores) / len(pass_scores) if pass_scores else None,
        "exact": sum(row["correct"] for row in rows) / total if total else None,
        "valid": sum(row["prediction"] is not None for row in rows) / total if total else None,
        "think_closed": sum(row["think_closed"] for row in rows) / total if total else None,
        "truncated": sum(row["truncated"] for row in rows) / total if total else None,
        "mean_tokens": sum(row["generated_tokens"] for row in rows) / total if total else None,
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
    }


def percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def copy_state(dst: list[torch.Tensor], row: int, src: list[torch.Tensor]) -> None:
    dst[0][:, :, row : row + 1].copy_(src[0])
    dst[1][:, row : row + 1].copy_(src[1])
    dst[2][row : row + 1].copy_(src[2])


def prefill(tasks: list[Task], model: Any, tokenizer: PIPELINE, token_device: str, ctx_limit: int, report_seconds: float) -> dict[int, tuple[list[torch.Tensor], torch.Tensor, int]]:
    cache = {}
    started = last_report = time.perf_counter()
    for done, task in enumerate(tasks, 1):
        ids = [0] + tokenizer.encode(prompt(task).replace("\r\n", "\n"))
        if len(ids) >= ctx_limit:
            raise ValueError(f"task {task.index} prompt has {len(ids)} tokens, ctx-limit is {ctx_limit}")
        state = model.zero_state(1)
        logits = model.forward(torch.tensor(ids, dtype=torch.long, device=token_device), state).view(-1)
        cache[task.index] = ([x.clone() for x in state], logits.clone(), len(ids))
        now = time.perf_counter()
        if now - last_report >= report_seconds:
            print(f"prefill={done}/{len(tasks)} elapsed={now - started:.1f}s", flush=True)
            last_report = now
    torch.cuda.synchronize()
    print(f"prefill={len(tasks)}/{len(tasks)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return cache


def generate(a: argparse.Namespace, tasks: list[Task], model: Any, tokenizer: PIPELINE, token_device: str, cache: dict[int, tuple[list[torch.Tensor], torch.Tensor, int]], output: Any) -> tuple[list[dict[str, Any]], int, float]:
    task_order = [task.index for task in tasks]
    random.Random(a.seed).shuffle(task_order)
    work = [(task_index, sample_id) for task_index in task_order for sample_id in range(a.rollout)]
    task_by_index = {task.index: task for task in tasks}
    unassigned = Counter(task_index for task_index, _sample_id in work)
    B = min(a.bsz, len(work))
    state = model.zero_state(B)
    device = next(iter(cache.values()))[1].device
    vocab = next(iter(cache.values()))[1].numel()
    frequency = torch.zeros((B, vocab), dtype=next(iter(cache.values()))[1].dtype, device=device)
    presence = torch.zeros_like(frequency)
    all_rows = torch.arange(B, device=device)
    token_tensor = torch.zeros((B, 1), dtype=torch.long, device=token_device)
    slots: list[tuple[int, int] | None] = [None] * B
    prompt_lengths = [0] * B
    generated = [[] for _ in range(B)]
    texts = ["" for _ in range(B)]
    decoded_to = [0] * B
    next_tokens = [0] * B
    token_counts = [0] * B
    max_tokens = [0] * B
    active = [False] * B
    rows: list[dict[str, Any]] = []
    pending = 0
    token_events = 0
    started = last_report = time.perf_counter()

    def report(force: bool = False) -> None:
        nonlocal last_report
        now = time.perf_counter()
        if not force and now - last_report < a.report_seconds:
            return
        m = metrics(rows, a.rollout)
        print(
            f"done={m['routes']}/{len(work)} tasks={m['tasks']}/{len(tasks)} "
            f"active={sum(active)}/{B} pending={len(work) - pending} "
            f"avg@{a.rollout}={percent(m[f'avg@{a.rollout}'])} exact={percent(m['exact'])} "
            f"majority@{a.rollout}={percent(m[f'majority@{a.rollout}'])} pass@{a.rollout}={percent(m[f'pass@{a.rollout}'])} "
            f"valid={percent(m['valid'])} think={percent(m['think_closed'])} trunc={percent(m['truncated'])} "
            f"mean_tok={m['mean_tokens'] or 0:.0f} tok/s={token_events / max(now - started, 1e-9):.1f} elapsed={now - started:.1f}s",
            flush=True,
        )
        output.flush()
        last_report = now

    def finish(row: int, reason: str) -> None:
        item = slots[row]
        assert item is not None
        task_index, sample_id = item
        if decoded_to[row] < len(generated[row]):
            tail = tokenizer.decode(generated[row][decoded_to[row] :])
            if "\ufffd" not in tail:
                texts[row] += tail
        completion = texts[row].split("\nUser:", 1)[0]
        completion = completion[1:] if completion.startswith(">") else completion
        completion = completion.strip()
        task = task_by_index[task_index]
        prediction = extract_answer(completion)
        result = {
            "task_index": task.index,
            "sample_id": sample_id,
            "domain": task.domain,
            "problem": task.problem,
            "answer": task.answer,
            "prediction": prediction,
            "correct": prediction == task.answer,
            "score": 1.0 if prediction == task.answer else 0.25 if prediction is None else 0.0,
            "think_closed": "</think>" in completion,
            "prompt_tokens": prompt_lengths[row],
            "generated_tokens": len(generated[row]),
            "tokens_including_stop": token_counts[row],
            "stop_reason": reason,
            "truncated": reason == "max_tokens",
            "completion": completion,
        }
        rows.append(result)
        output.write(json.dumps(result, ensure_ascii=False) + "\n")
        slots[row] = None
        generated[row] = []
        texts[row] = ""
        decoded_to[row] = 0
        next_tokens[row] = 0
        active[row] = False

    def refill(refill_rows: list[int]) -> list[int]:
        nonlocal pending
        assigned, first_logits = [], []
        for row in refill_rows:
            if pending == len(work):
                break
            task_index, sample_id = work[pending]
            pending += 1
            task_state, task_logits, prompt_length = cache[task_index]
            copy_state(state, row, task_state)
            slots[row] = (task_index, sample_id)
            prompt_lengths[row] = prompt_length
            generated[row] = []
            texts[row] = ""
            decoded_to[row] = 0
            token_counts[row] = 0
            max_tokens[row] = min(a.max_new_tokens, a.ctx_limit - prompt_length)
            active[row] = True
            frequency[row].zero_()
            presence[row].zero_()
            assigned.append(row)
            first_logits.append(task_logits)
            unassigned[task_index] -= 1
            if not unassigned[task_index]:
                del cache[task_index]
        if assigned:
            sampled = sample_logits(torch.stack(first_logits), a.temperature, a.top_p, a.top_k)
            index = torch.tensor(assigned, device=device)
            frequency[index, sampled] = a.frequency_penalty
            presence[index, sampled] = a.presence_penalty
            for row, token in zip(assigned, sampled.cpu().tolist()):
                next_tokens[row] = int(token)
        return assigned

    def consume(row: int) -> bool:
        nonlocal token_events
        token = next_tokens[row]
        token_counts[row] += 1
        token_events += 1
        if token == 0:
            finish(row, "eod")
            return False
        generated[row].append(token)
        tail = tokenizer.decode(generated[row][decoded_to[row] :])
        if "\ufffd" not in tail:
            texts[row] += tail
            decoded_to[row] = len(generated[row])
            if "\nUser:" in texts[row]:
                finish(row, "user_stop")
                return False
            if extract_answer(texts[row], require_think_close=True) is not None:
                finish(row, "answer")
                return False
        if token_counts[row] >= max_tokens[row]:
            finish(row, "max_tokens")
            return False
        return True

    refill(list(range(B)))
    while any(active) or pending < len(work):
        scan = list(range(B))
        forward_rows = []
        while scan:
            empty = []
            for row in scan:
                if active[row]:
                    if consume(row):
                        forward_rows.append(row)
                    else:
                        empty.append(row)
                elif pending < len(work):
                    empty.append(row)
            scan = refill(empty)
        report()
        if not forward_rows:
            continue
        token_tensor[:, 0] = torch.tensor(next_tokens, dtype=torch.long, device=token_device)
        logits = model.forward(token_tensor, state).view(B, -1)
        logits.sub_(frequency).sub_(presence)
        sampled = sample_logits(logits, a.temperature, a.top_p, a.top_k)
        frequency.mul_(a.penalty_decay)
        frequency[all_rows, sampled] += a.frequency_penalty
        presence[all_rows, sampled] = a.presence_penalty
        sampled_cpu = sampled.cpu().tolist()
        for row in forward_rows:
            next_tokens[row] = int(sampled_cpu[row])
    torch.cuda.synchronize()
    report(True)
    if len(rows) != len(work):
        raise RuntimeError(f"completed {len(rows)} routes, expected {len(work)}")
    return rows, token_events, time.perf_counter() - started


def main() -> None:
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.set_grad_enabled(False)
    tasks = load_tasks(a.dataset, a.split, a.limit)
    if not a.limit and len(tasks) != 198:
        print(f"warning: expected 198 GPQA Diamond tasks, loaded {len(tasks)}", flush=True)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "config.json").write_text(json.dumps(vars(a), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"tasks={len(tasks)} rollout={a.rollout} routes={len(tasks) * a.rollout} bsz={a.bsz} gpu={a.gpu}\n"
        f"sampling: temperature={a.temperature:g} top_p={a.top_p:g} top_k={a.top_k} "
        f"presence={a.presence_penalty:g} frequency={a.frequency_penalty:g} decay={a.penalty_decay:g}\n"
        f"output: {out_dir}",
        flush=True,
    )
    os.chdir(ROOT)
    import rwkv7_fast_v3a as v3a

    v3a.MODEL_PATH = a.model
    v3a.WKV_MODE = "fp32io16"
    v3a.EMB_DEVICE = "cpu"
    v3a.RKV_MODE = "off"
    v3a.CMIX_SPARSE = "no-fc"
    v3a.LOWRANK_WEIGHT = "both"
    v3a.ORIG_LINEAR_GROUPS = v3a.parse_orig_linear_groups("att_c2c,ffn_key,head")
    v3a.load_extensions(v3a.WKV_MODE)
    model = v3a.RWKV7()
    tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")
    token_device = "cpu" if model.emb_cpu else "cuda"
    with torch.inference_mode():
        cache = prefill(tasks, model, tokenizer, token_device, a.ctx_limit, a.report_seconds)
        with (out_dir / "generations.jsonl").open("w", encoding="utf-8", buffering=1) as output:
            rows, tokens, elapsed = generate(a, tasks, model, tokenizer, token_device, cache, output)
    final = metrics(rows, a.rollout)
    summary = {
        **final,
        "num_tasks": len(tasks),
        "rollout": a.rollout,
        "tokens_including_stop": tokens,
        "elapsed_seconds": elapsed,
        "token_per_second": tokens / elapsed,
        "generations_jsonl": str(out_dir / "generations.jsonl"),
        "model": a.model,
        "sampling": {key: getattr(a, key) for key in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "penalty_decay")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("GPQA_DIAMOND_RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
