import codecs
import os
import queue
import sys
import time
from pathlib import Path

from constants import TITLE_GPU_NAME, TITLE_MODEL_NAME, TITLE_PRECISION
from parsing import HtmlCompletionScanner
from shared import (
    make_model_args,
    open_jsonl_log,
    put_drop,
    put_reliable,
    send_finished_captions,
    send_page_finished,
    snapshot_dirty,
    write_jsonl_line,
    write_unlogged_jsonl,
)


def model_process(out_q, ctrl_q, shutdown_event, cfg):
    model_dir = Path(cfg["model_dir"]).resolve()
    sys.path.insert(0, str(model_dir))

    import torch
    from reference.rwkv7 import RWKV_x070
    from reference.utils import TRIE_TOKENIZER, sampler_top_p_fast

    page_count = cfg["cols"] * cfg["rows"] * cfg["groups"]
    current_prompt = cfg["prompt"]
    flush_dt = 1.0 / max(1, cfg["producer_flush_hz"])
    perf_interval = max(1, cfg["perf_interval"])
    perf_sync_interval = max(0, cfg["perf_sync_interval"])
    sampler_top_p = cfg["sampler_top_p"]
    sampler_temp = cfg["sampler_temp"]
    sampler_top_k = cfg["sampler_top_k"]
    presence_penalty = cfg["presence_penalty"]
    generation_length = cfg["generation_length"]

    if cfg.get("model_nice", 0) > 0:
        try:
            os.nice(int(cfg["model_nice"]))
        except OSError:
            pass

    state = None
    raw_pages = []
    logged_pages = []
    log_file = None
    dropped = 0
    try:
        model = RWKV_x070(make_model_args(cfg["model_name"]))
        tokenizer = TRIE_TOKENIZER(str(model_dir / "reference" / "rwkv_vocab_v20230424.txt"))
        generation_id = 0
        seed_base = int(time.time_ns() & 0x7FFFFFFF)
        baseline_prompt = None
        baseline_state = None
        baseline_out = None

        def clone_tensor_list(items):
            return [item.detach().clone() if torch.is_tensor(item) else item for item in items]

        def compare_tensor_lists(lhs, rhs, label):
            if lhs is None or rhs is None:
                return True, f"{label}: missing baseline"
            if len(lhs) != len(rhs):
                return False, f"{label}: len {len(lhs)} != {len(rhs)}"
            for idx, (a, b) in enumerate(zip(lhs, rhs)):
                if torch.is_tensor(a) or torch.is_tensor(b):
                    if not (torch.is_tensor(a) and torch.is_tensor(b)):
                        return False, f"{label}[{idx}]: tensor/type mismatch"
                    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                        return False, f"{label}[{idx}]: meta {tuple(a.shape)} {a.dtype} {a.device} != {tuple(b.shape)} {b.dtype} {b.device}"
                    if not torch.equal(a, b):
                        max_diff = (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()
                        return False, f"{label}[{idx}]: value mismatch max_abs_diff={max_diff:.6g}"
                elif a != b:
                    return False, f"{label}[{idx}]: value mismatch"
            return True, f"{label}: exact match"

        def compare_tensor(lhs, rhs, label):
            if lhs is None or rhs is None:
                return True, f"{label}: missing baseline"
            if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype or lhs.device != rhs.device:
                return False, f"{label}: meta {tuple(lhs.shape)} {lhs.dtype} {lhs.device} != {tuple(rhs.shape)} {rhs.dtype} {rhs.device}"
            if torch.equal(lhs, rhs):
                return True, f"{label}: exact match"
            max_diff = (lhs.to(torch.float32) - rhs.to(torch.float32)).abs().max().item()
            return False, f"{label}: value mismatch max_abs_diff={max_diff:.6g}"

        while not shutdown_event.is_set():
            if state is not None:
                del state
                torch.cuda.empty_cache()
            if log_file is not None:
                log_file.close()
                log_file = None

            generation_id += 1
            seed = seed_base + generation_id
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            state = model.generate_zero_state(page_count)
            raw_pages = ["" for _ in range(page_count)]
            sent_lengths = [0 for _ in range(page_count)]
            finished_pages = [False for _ in range(page_count)]
            logged_pages = [False for _ in range(page_count)]
            completion_scanners = [HtmlCompletionScanner(current_prompt) for _ in range(page_count)]
            log_path, log_file = open_jsonl_log(cfg)
            dirty = set()
            put_reliable(out_q, ("clear_all", None), shutdown_event)
            put_drop(out_q, ("status", "Loading prompts..."))
            if log_path is not None:
                put_drop(out_q, ("status", f"Writing JSONL log to {log_path}"))

            prompts = [current_prompt for _ in range(page_count)]
            out = model.forward_batch([tokenizer.encode(prompt) for prompt in prompts], state)
            if baseline_state is None or baseline_prompt != current_prompt:
                baseline_prompt = current_prompt
                baseline_state = clone_tensor_list(state)
                baseline_out = out.detach().clone()
                put_drop(out_q, ("prefill_check", {"ok": True, "message": f"Prefill baseline captured | generation {generation_id}"}))
            else:
                state_ok, state_msg = compare_tensor_lists(state, baseline_state, "prefill state")
                out_ok, out_msg = compare_tensor(out, baseline_out, "prefill logits")
                if state_ok and out_ok:
                    put_drop(out_q, ("prefill_check", {"ok": True, "message": f"Prefill check OK | generation {generation_id}"}))
                else:
                    put_reliable(
                        out_q,
                        (
                            "prefill_check",
                            {
                                "ok": False,
                                "message": f"Prefill check FAILED | generation {generation_id} | {state_msg} | {out_msg}",
                            },
                        ),
                        shutdown_event,
                    )
            token_presence = None
            if presence_penalty != 0.0:
                token_presence = torch.zeros(
                    (page_count, out.shape[-1]),
                    dtype=torch.bool,
                    device=out.device,
                )
            decoders = [codecs.getincrementaldecoder("utf-8")("strict") for _ in range(page_count)]
            last_flush = time.perf_counter()
            perf_start = last_flush
            perf_tokens = 0
            stop_requested = False
            restart_requested = False
            paused = False

            def handle_control_message(msg):
                nonlocal baseline_out
                nonlocal baseline_prompt
                nonlocal baseline_state
                nonlocal current_prompt
                nonlocal paused
                nonlocal restart_requested
                nonlocal stop_requested
                if msg == "stop":
                    stop_requested = True
                elif msg == "restart":
                    restart_requested = True
                elif msg == "pause":
                    paused = True
                elif msg == "resume":
                    paused = False
                elif isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "prompt":
                    current_prompt = str(msg[1])
                    baseline_prompt = None
                    baseline_state = None
                    baseline_out = None
                    restart_requested = True

            step = 0
            while generation_length <= 0 or step < generation_length:
                if shutdown_event.is_set():
                    break

                try:
                    while True:
                        handle_control_message(ctrl_q.get_nowait())
                        if stop_requested or restart_requested:
                            break
                except queue.Empty:
                    pass
                if stop_requested:
                    put_drop(out_q, ("status", "Stopping model worker"))
                    break
                if restart_requested:
                    put_drop(out_q, ("status", "Restarting generation"))
                    break
                if paused:
                    put_drop(out_q, ("status", "Generation paused"))
                    while paused and not stop_requested and not restart_requested and not shutdown_event.is_set():
                        try:
                            handle_control_message(ctrl_q.get(timeout=0.1))
                        except queue.Empty:
                            continue
                    if stop_requested:
                        put_drop(out_q, ("status", "Stopping model worker"))
                        break
                    if restart_requested:
                        put_drop(out_q, ("status", "Restarting generation"))
                        break
                    if shutdown_event.is_set():
                        break
                    put_drop(out_q, ("status", "Generation resumed"))

                new_tokens_tensor = sampler_top_p_fast(
                    out,
                    sampler_top_p,
                    sampler_temp,
                    sampler_top_k,
                    token_presence,
                    presence_penalty,
                )
                flat_tokens = new_tokens_tensor.view(-1).detach().cpu().tolist()

                if hasattr(model, "forward_seq_batch_1"):
                    out = model.forward_seq_batch_1(new_tokens_tensor, state, False)
                else:
                    out = model.forward_batch([[int(x)] for x in flat_tokens], state)
                perf_tokens += page_count

                for idx, token_id in enumerate(flat_tokens):
                    if token_id == 0:
                        if not finished_pages[idx]:
                            finished_pages[idx] = True
                            write_jsonl_line(log_file, raw_pages[idx])
                            logged_pages[idx] = True
                            if send_page_finished(out_q, idx, raw_pages[idx]):
                                dirty.add(idx)
                            else:
                                sent_lengths[idx] = len(raw_pages[idx])
                                dirty.discard(idx)
                        continue
                    text = decoders[idx].decode(tokenizer.idx2token[token_id], final=False)
                    if text and not finished_pages[idx]:
                        raw_pages[idx] += text
                        frozen = completion_scanners[idx].update(raw_pages[idx])
                        if frozen is not None:
                            raw_pages[idx] = frozen
                            finished_pages[idx] = True
                            write_jsonl_line(log_file, frozen)
                            logged_pages[idx] = True
                            if send_page_finished(out_q, idx, frozen):
                                dirty.add(idx)
                            else:
                                sent_lengths[idx] = len(frozen)
                                dirty.discard(idx)
                        else:
                            dirty.add(idx)

                now = time.perf_counter()
                if now - last_flush >= flush_dt:
                    dropped = snapshot_dirty(out_q, raw_pages, dirty, sent_lengths, finished_pages, dropped)
                    last_flush = now

                if (step + 1) % perf_interval == 0:
                    if perf_sync_interval and (step + 1) % perf_sync_interval == 0:
                        torch.cuda.synchronize()
                    now = time.perf_counter()
                    elapsed = max(1e-9, now - perf_start)
                    tps = round(perf_tokens / elapsed)
                    perf_tokens = 0
                    perf_start = now
                    put_drop(
                        out_q,
                        (
                            "status",
                            f"{TITLE_MODEL_NAME} {TITLE_PRECISION} bsz{page_count} @ {TITLE_GPU_NAME} | "
                            f"Token/s {tps}",
                        ),
                    )
                if all(finished_pages):
                    put_drop(out_q, ("status", f"All {page_count} pages completed"))
                    break
                step += 1

            if stop_requested:
                return
            if restart_requested:
                continue
            dropped = snapshot_dirty(out_q, raw_pages, dirty, sent_lengths, finished_pages, dropped, force=True)
            write_unlogged_jsonl(log_file, raw_pages, logged_pages)
            completed = sum(finished_pages)
            if completed == page_count:
                send_finished_captions(out_q, raw_pages)
                put_drop(out_q, ("status", f"Generation finished: all {page_count} pages completed"))
            elif shutdown_event.is_set():
                put_drop(out_q, ("status", f"Generation stopped: {completed}/{page_count} pages completed"))
            else:
                put_drop(out_q, ("status", f"Token budget reached: {completed}/{page_count} pages completed"))
            if log_file is not None:
                log_file.close()
                log_file = None
            while not shutdown_event.is_set():
                try:
                    msg = ctrl_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if msg == "stop":
                    return
                if msg == "restart":
                    break
                if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "prompt":
                    current_prompt = str(msg[1])
                    baseline_prompt = None
                    baseline_state = None
                    baseline_out = None
                    break
            if shutdown_event.is_set():
                return
    except Exception as exc:
        put_drop(out_q, ("error", f"{type(exc).__name__}: {exc}"))
    finally:
        write_unlogged_jsonl(log_file, raw_pages, logged_pages)
        if log_file is not None:
            log_file.close()
        put_drop(out_q, ("done", None))
