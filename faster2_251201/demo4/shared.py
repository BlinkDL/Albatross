import json
import queue
import types
from pathlib import Path

from constants import DEFAULT_PROMPT


def put_drop(out_q, msg):
    try:
        out_q.put_nowait(msg)
        return False
    except queue.Full:
        return True


def put_reliable(out_q, msg, shutdown_event=None, timeout=0.1):
    while shutdown_event is None or not shutdown_event.is_set():
        try:
            out_q.put(msg, timeout=timeout)
            return True
        except queue.Full:
            continue
    return False


def make_model_args(model_name):
    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    args.MODEL_NAME = model_name
    return args


def load_prompt(cli):
    if cli.prompt:
        return cli.prompt.replace("\\n", "\n")
    if cli.prompt_file:
        return Path(cli.prompt_file).read_text(encoding="utf-8")
    return DEFAULT_PROMPT


def open_jsonl_log(cfg):
    log_path = cfg.get("jsonl_log")
    if not log_path:
        return None, None
    path = Path(log_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, path.open("w", encoding="utf-8", buffering=1)


def write_jsonl_line(log_file, text):
    if log_file is None:
        return
    log_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def write_unlogged_jsonl(log_file, raw_pages, logged_pages):
    if log_file is None:
        return
    for idx, text in enumerate(raw_pages):
        if not logged_pages[idx]:
            write_jsonl_line(log_file, text)
            logged_pages[idx] = True


def snapshot_dirty(out_q, raw_pages, dirty, sent_lengths, finished_pages, dropped, force=False):
    if not dirty and not force:
        return dropped
    updates = []
    sent_indices = []
    for idx in sorted(dirty):
        raw = raw_pages[idx]
        start = sent_lengths[idx]
        finished = finished_pages[idx]
        if len(raw) <= start and not finished:
            sent_indices.append(idx)
            continue
        updates.append(
            {
                "index": idx,
                "delta": raw[start:],
                "finished": finished,
                "totalBytes": len(raw.encode("utf-8")) if finished else None,
            }
        )
        sent_indices.append(idx)
    if not updates:
        dirty.difference_update(sent_indices)
        return dropped
    if put_drop(out_q, ("pages_delta", updates)):
        dropped += 1
        return dropped
    for idx in sent_indices:
        sent_lengths[idx] = len(raw_pages[idx])
    dirty.difference_update(sent_indices)
    return dropped


def send_finished_captions(out_q, raw_pages):
    payload = [
        {
            "index": idx,
            "captionStage": "finish",
            "totalBytes": len(text.encode("utf-8")),
        }
        for idx, text in enumerate(raw_pages)
    ]
    put_drop(out_q, ("pages_finished", payload))


def send_page_finished(out_q, idx, text):
    msg = (
        "page_finished",
        {
            "index": idx,
            "text": text,
            "captionStage": "finish",
            "totalBytes": len(text.encode("utf-8")),
        },
    )
    try:
        out_q.put(msg, timeout=0.02)
        return False
    except queue.Full:
        return put_drop(out_q, msg)
