import json
import os
import queue
import random
import sys
import time

from constants import WINDOW_TITLE
from host_html import build_host_html
from parsing import parse_generated_page


def gui_process(in_q, ctrl_q, shutdown_event, cfg):
    if cfg.get("gui_nice", 0) > 0 and hasattr(os, "nice"):
        try:
            os.nice(int(cfg["gui_nice"]))
        except OSError:
            pass

    if cfg.get("disable_webengine_gpu", True):
        flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
        if "--disable-gpu" not in flags:
            os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (flags + " --disable-gpu --disable-gpu-compositing").strip()

    from PySide6 import QtCore, QtWidgets
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineCore import QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView

    class BridgeObject(QtCore.QObject):
        def __init__(self, owner):
            super().__init__(owner)
            self.owner = owner

        @QtCore.Slot(str)
        def handleGroup(self, action):
            QtCore.QTimer.singleShot(0, lambda action=action: self.owner.handle_group_action(action))

        @QtCore.Slot()
        def handleRestart(self):
            QtCore.QTimer.singleShot(0, self.owner.request_restart)

        @QtCore.Slot(str)
        def handlePrompt(self, prompt):
            QtCore.QTimer.singleShot(0, lambda prompt=prompt: self.owner.handle_prompt(prompt))

        @QtCore.Slot(bool)
        def handlePromptEditor(self, opened):
            QtCore.QTimer.singleShot(0, lambda opened=opened: self.owner.handle_prompt_editor(opened))

    class HtmlGridView(QWebEngineView):
        def __init__(self):
            super().__init__()
            self.visible_count = cfg["cols"] * cfg["rows"]
            self.group_count = cfg["groups"]
            self.page_count = self.visible_count * self.group_count
            self.current_group = 0
            self.raw_pages = ["" for _ in range(self.page_count)]
            self.raw_bytes = [0 for _ in range(self.page_count)]
            self.byte_integrity_error = False
            self.byte_integrity_message = ""
            self.prefill_check_error = False
            self.prefill_check_message = ""
            self.dirty = set()
            self.loaded = False
            self.pending_status = "Starting demo4..."
            self.current_prompt = cfg["prompt"]
            self.current_scale = cfg["page_scale"]
            self.last_rendered_stage = ["" for _ in range(self.page_count)]
            self.last_rendered_html_bytes = [0 for _ in range(self.page_count)]
            self.last_rendered_at = [0.0 for _ in range(self.page_count)]
            self.next_render_at = [0.0 for _ in range(self.page_count)]
            self.page_phase = self.make_page_phases()
            self.render_cursor = 0
            self.force_render = set()
            self.finished_pages = [False for _ in range(self.page_count)]
            self.seen_error = False
            self.pending_group = None
            self.ignore_until_clear_all = False
            self.view_epoch = 0
            self.render_defer_until = 0.0
            self.group_render_delay_ms = max(0, cfg["group_render_delay_ms"])
            self.channel = QWebChannel(self.page())
            self.bridge_object = BridgeObject(self)
            self.channel.registerObject("demo4Bridge", self.bridge_object)
            self.page().setWebChannel(self.channel)
            self.refresh_title()
            self.resize(cfg["window_w"], cfg["window_h"])

            settings = self.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, bool(cfg["allow_scripts"]))
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, False)

            self.loadFinished.connect(self.on_loaded)
            self.setHtml(
                build_host_html(
                    cfg["cols"],
                    cfg["rows"],
                    cfg["page_scale"],
                    cfg["allow_scripts"],
                    self.group_count,
                    self.current_prompt,
                    cfg.get("auto_bounce_scroll", False),
                    cfg.get("auto_bounce_scroll_step", 20),
                    cfg.get("auto_bounce_scroll_tick_ms", 16),
                    cfg.get("auto_bounce_scroll_edge_pause_ms", 350),
                ),
                QtCore.QUrl("about:blank"),
            )

            self.poll_timer = QtCore.QTimer(self)
            self.poll_timer.timeout.connect(self.poll_messages)
            self.poll_timer.start(max(1, cfg["ui_poll_ms"]))

            self.render_timer = QtCore.QTimer(self)
            self.render_timer.timeout.connect(self.render_dirty)
            self.render_timer.start(max(1, int(1000 / max(1, cfg["render_hz"]))))

            self.auto_group_timer = None
            auto_group_seconds = float(cfg.get("auto_switch_group_seconds", 0.0) or 0.0)
            if self.group_count > 1 and auto_group_seconds > 0.0:
                self.auto_group_timer = QtCore.QTimer(self)
                self.auto_group_timer.timeout.connect(self.auto_switch_group)
                self.auto_group_timer.start(int(max(3.0, auto_group_seconds) * 1000))

        def set_status(self, text):
            self.pending_status = text or ""
            warning = ""
            if self.byte_integrity_error:
                warning = f" | BYTE MISMATCH: {self.byte_integrity_message}"
            if self.prefill_check_error:
                warning += f" | PREFILL MISMATCH: {self.prefill_check_message}"
            title = (
                f"{WINDOW_TITLE}{warning} | finish {sum(self.finished_pages):,}/{self.page_count:,} "
                f"| bytes {sum(self.raw_bytes):,}"
            )
            if self.pending_status:
                title += " | " + self.pending_status
            self.setWindowTitle(title)

        def refresh_title(self):
            self.set_status(self.pending_status)

        def warn_byte_integrity(self, message):
            if not self.byte_integrity_error:
                self.byte_integrity_error = True
                self.byte_integrity_message = message
            self.refresh_title()

        def verify_finished_page_bytes(self, idx, source, expected_total=None):
            actual = len(self.raw_pages[idx].encode("utf-8"))
            displayed = self.raw_bytes[idx]
            mismatch = []
            if expected_total is not None and expected_total != actual:
                mismatch.append(f"msg={expected_total:,} actual={actual:,}")
            if displayed != actual:
                mismatch.append(f"display={displayed:,} actual={actual:,}")
            if mismatch:
                self.warn_byte_integrity(f"page {idx + 1} {source} " + " ".join(mismatch))
                self.raw_bytes[idx] = actual
                return True
            return False

        def verify_total_bytes(self, source):
            displayed = sum(self.raw_bytes)
            actual = sum(len(text.encode("utf-8")) for text in self.raw_pages)
            if displayed != actual:
                self.warn_byte_integrity(f"{source} total display={displayed:,} actual={actual:,}")
                return False
            return True

        def visible_start(self):
            return self.current_group * self.visible_count

        def visible_indices(self):
            start = self.visible_start()
            return range(start, min(start + self.visible_count, self.page_count))

        def is_visible(self, idx):
            start = self.visible_start()
            return start <= idx < start + self.visible_count

        def local_index(self, idx):
            return idx - self.visible_start()

        def caption_item(self, idx, stage=None):
            return {
                "index": self.local_index(idx),
                "globalIndex": idx,
                "captionStage": "finish" if self.finished_pages[idx] else stage or self.last_rendered_stage[idx] or "pending",
                "totalBytes": self.raw_bytes[idx],
            }

        def visible_caption_items(self):
            return [self.caption_item(idx) for idx in self.visible_indices()]

        def reset_visible_frames(self):
            script = (
                "window.resetFrames("
                + json.dumps(self.visible_caption_items(), ensure_ascii=False)
                + ","
                + json.dumps(self.view_epoch)
                + ");"
            )
            self.run_js(script)

        def reset_local_generation_state(self, status_text):
            self.raw_pages = ["" for _ in range(self.page_count)]
            self.raw_bytes = [0 for _ in range(self.page_count)]
            self.byte_integrity_error = False
            self.byte_integrity_message = ""
            self.dirty.clear()
            self.last_rendered_stage = ["" for _ in range(self.page_count)]
            self.last_rendered_html_bytes = [0 for _ in range(self.page_count)]
            self.last_rendered_at = [0.0 for _ in range(self.page_count)]
            self.force_render.clear()
            self.finished_pages = [False for _ in range(self.page_count)]
            self.seen_error = False
            self.reset_render_schedule()
            target_group = self.current_group if self.pending_group is None else self.pending_group
            self.current_group = max(0, min(self.group_count - 1, target_group))
            self.pending_group = None
            self.view_epoch += 1
            self.render_defer_until = 0.0
            self.run_js(
                "window.setGroup("
                + json.dumps(self.current_group)
                + ","
                + json.dumps(self.group_count)
                + ");"
            )
            self.reset_visible_frames()
            self.dirty.update(self.visible_indices())
            self.set_status(status_text)

        def request_restart(self):
            if self.ignore_until_clear_all:
                return
            try:
                ctrl_q.put_nowait("restart")
            except queue.Full:
                self.set_status("Restart request queue is full")
                return
            self.ignore_until_clear_all = True
            self.reset_local_generation_state("Restarting generation...")

        def handle_prompt(self, prompt):
            if prompt == self.current_prompt:
                return
            if self.ignore_until_clear_all:
                self.set_status("Prompt change ignored while restart is pending")
                self.run_js("window.setPrompt(" + json.dumps(self.current_prompt, ensure_ascii=False) + ");")
                return
            try:
                ctrl_q.put_nowait(("prompt", prompt))
            except queue.Full:
                self.set_status("Prompt update queue is full")
                self.run_js("window.setPrompt(" + json.dumps(self.current_prompt, ensure_ascii=False) + ");")
                try:
                    ctrl_q.put_nowait("resume")
                except queue.Full:
                    pass
                return
            self.current_prompt = prompt
            cfg["prompt"] = prompt
            self.ignore_until_clear_all = True
            self.reset_local_generation_state("Restarting generation with new prompt...")

        def handle_prompt_editor(self, opened):
            try:
                ctrl_q.put_nowait("pause" if opened else "resume")
            except queue.Full:
                self.set_status("Prompt editor control queue is full")
                return
            self.set_status("Generation paused for prompt editing" if opened else "Generation resumed")

        def switch_group(self, group):
            group = max(0, min(self.group_count - 1, group))
            if group == self.current_group:
                return
            previous_visible = set(self.visible_indices())
            self.current_group = group
            self.view_epoch += 1
            self.render_cursor = 0
            visible = set(self.visible_indices())
            self.dirty.difference_update(previous_visible)
            self.force_render.difference_update(previous_visible)
            self.dirty.update(visible)
            self.force_render.update(visible)
            now = time.perf_counter()
            self.render_defer_until = now + self.group_render_delay_ms / 1000.0
            for idx in visible:
                self.last_rendered_stage[idx] = ""
                self.last_rendered_html_bytes[idx] = 0
                self.last_rendered_at[idx] = 0.0
                self.next_render_at[idx] = self.render_defer_until
            self.run_js(
                "window.setGroup("
                + json.dumps(self.current_group)
                + ","
                + json.dumps(self.group_count)
                + ");"
            )
            self.reset_visible_frames()
            QtCore.QTimer.singleShot(self.group_render_delay_ms, self.render_dirty)

        def request_switch_group(self, group):
            group = max(0, min(self.group_count - 1, group))
            if not self.loaded:
                self.pending_group = group
                return
            self.pending_group = None
            self.switch_group(group)

        def handle_group_action(self, action):
            base_group = self.current_group if self.pending_group is None else self.pending_group
            if action == "prev":
                self.request_switch_group((base_group - 1) % self.group_count)
            elif action == "next":
                self.request_switch_group((base_group + 1) % self.group_count)
            else:
                try:
                    self.request_switch_group(int(action) - 1)
                except ValueError:
                    pass

        def auto_switch_group(self):
            if self.group_count <= 1:
                return
            base_group = self.current_group if self.pending_group is None else self.pending_group
            self.request_switch_group((base_group + 1) % self.group_count)

        def make_page_phases(self):
            if self.page_count <= 0:
                return []
            window = max(0.0, cfg["html_stagger_ms"] / 1000.0)
            if window <= 0:
                return [0.0 for _ in range(self.page_count)]
            rng = random.Random(0xD4E4F00D)
            slot = window / self.page_count
            phases = []
            for idx in range(self.page_count):
                jitter = rng.uniform(0.0, slot * 0.55)
                phases.append(idx * slot + jitter)
            return phases

        def reset_render_schedule(self, start_now=False):
            base = time.perf_counter()
            for idx in range(self.page_count):
                self.next_render_at[idx] = base if start_now else base + self.page_phase[idx]
            self.render_cursor = 0

        def on_loaded(self, ok):
            self.loaded = ok
            self.run_js(f"window.setScale({json.dumps(self.current_scale)});")
            self.run_js(
                "window.setGroup("
                + json.dumps(self.current_group)
                + ","
                + json.dumps(self.group_count)
                + ");"
            )
            self.set_status(self.pending_status)
            self.reset_render_schedule()
            self.dirty.update(self.visible_indices())
            if self.pending_group is not None:
                group = self.pending_group
                self.pending_group = None
                self.switch_group(group)
            else:
                self.reset_visible_frames()

        def run_js(self, script):
            if self.loaded:
                self.page().runJavaScript(script)

        def poll_messages(self):
            processed = 0
            while processed < cfg["max_gui_messages"]:
                try:
                    msg_type, payload = in_q.get_nowait()
                except queue.Empty:
                    break
                processed += 1
                if self.ignore_until_clear_all and msg_type not in ("clear_all", "error"):
                    continue
                if msg_type == "pages_delta":
                    caption_updates = []
                    finished_changed = False
                    bytes_changed = False
                    for item in payload:
                        idx = item.get("index")
                        if not isinstance(idx, int) or not 0 <= idx < self.page_count:
                            continue
                        delta = item.get("delta", "")
                        if delta:
                            self.raw_pages[idx] += delta
                            self.raw_bytes[idx] += len(delta.encode("utf-8"))
                            bytes_changed = True
                        if item.get("totalBytes") is not None:
                            if self.raw_bytes[idx] != item["totalBytes"]:
                                bytes_changed = True
                            self.raw_bytes[idx] = item["totalBytes"]
                        if item.get("finished"):
                            if not self.finished_pages[idx]:
                                finished_changed = True
                            self.finished_pages[idx] = True
                            self.force_render.add(idx)
                            self.next_render_at[idx] = 0.0
                            if self.verify_finished_page_bytes(idx, "pages_delta", item.get("totalBytes")):
                                bytes_changed = True
                        if self.is_visible(idx):
                            caption_updates.append(self.caption_item(idx))
                        self.dirty.add(idx)
                    if caption_updates:
                        self.run_js("window.setPageCaptions(" + json.dumps(caption_updates, ensure_ascii=False) + ");")
                    if finished_changed or bytes_changed:
                        self.refresh_title()
                elif msg_type == "page_finished":
                    idx = payload.get("index")
                    raw = payload.get("text", "")
                    if isinstance(idx, int) and 0 <= idx < self.page_count:
                        self.raw_pages[idx] = raw
                        new_bytes = payload.get("totalBytes", len(raw.encode("utf-8")))
                        bytes_changed = self.raw_bytes[idx] != new_bytes
                        self.raw_bytes[idx] = new_bytes
                        finished_changed = not self.finished_pages[idx]
                        self.finished_pages[idx] = True
                        if self.verify_finished_page_bytes(idx, "page_finished", payload.get("totalBytes")):
                            bytes_changed = True
                        self.dirty.add(idx)
                        self.force_render.add(idx)
                        self.next_render_at[idx] = 0.0
                        if self.is_visible(idx):
                            self.run_js(
                                "window.setPageCaptions("
                                + json.dumps([self.caption_item(idx)], ensure_ascii=False)
                                + ");"
                            )
                        if finished_changed or bytes_changed:
                            self.refresh_title()
                elif msg_type == "pages_finished":
                    finished_changed = False
                    bytes_changed = False
                    for item in payload:
                        idx = item.get("index")
                        if isinstance(idx, int) and 0 <= idx < self.page_count:
                            if not self.finished_pages[idx]:
                                finished_changed = True
                            self.finished_pages[idx] = True
                            if item.get("totalBytes") is not None:
                                if self.raw_bytes[idx] != item["totalBytes"]:
                                    bytes_changed = True
                                self.raw_bytes[idx] = item["totalBytes"]
                            if self.verify_finished_page_bytes(idx, "pages_finished", item.get("totalBytes")):
                                bytes_changed = True
                    self.run_js("window.setPageCaptions(" + json.dumps(self.visible_caption_items(), ensure_ascii=False) + ");")
                    self.verify_total_bytes("pages_finished")
                    if finished_changed or bytes_changed:
                        self.refresh_title()
                elif msg_type == "clear_all":
                    self.ignore_until_clear_all = False
                    self.reset_local_generation_state(self.pending_status)
                elif msg_type == "prefill_check":
                    ok = bool(payload.get("ok")) if isinstance(payload, dict) else False
                    message = payload.get("message", "") if isinstance(payload, dict) else str(payload)
                    if ok:
                        self.prefill_check_error = False
                        self.prefill_check_message = ""
                    else:
                        self.prefill_check_error = True
                        self.prefill_check_message = message
                    self.set_status(message)
                elif msg_type == "status":
                    self.set_status(payload)
                elif msg_type == "error":
                    self.seen_error = True
                    self.set_status(f"Error: {payload}")
                elif msg_type == "done":
                    if not self.seen_error:
                        self.set_status(payload or "Done")
            if shutdown_event.is_set():
                self.close()

        def render_dirty(self):
            if not self.loaded or not self.dirty:
                return
            batch = []
            keep_dirty = set()
            now = time.perf_counter()
            if now < self.render_defer_until:
                return
            max_html_loads = max(1, cfg["max_page_loads_per_tick"])
            max_text_loads = max(1, cfg["max_text_page_loads_per_tick"])
            html_min_delta = max(0, cfg["html_min_delta"])
            html_max_stale = max(0.05, cfg["html_max_stale_ms"] / 1000.0)
            html_refresh_interval = max(0.05, cfg["html_refresh_interval_ms"] / 1000.0)
            preview_refresh_interval = max(0.05, cfg["preview_refresh_interval_ms"] / 1000.0)
            defer_interval = max(0.03, min(0.25, html_refresh_interval / 3.0))
            html_loads = 0
            text_loads = 0
            visible = list(self.visible_indices())
            indexes = visible[self.render_cursor :] + visible[: self.render_cursor]
            next_cursor = self.render_cursor
            for idx in indexes:
                if idx not in self.dirty:
                    continue
                forced = idx in self.force_render
                if not forced and now < self.next_render_at[idx]:
                    keep_dirty.add(idx)
                    continue
                parsed = parse_generated_page(self.raw_pages[idx])
                html_bytes = len(parsed.html_text)
                total_bytes = self.raw_bytes[idx]
                is_finished = self.finished_pages[idx]
                is_page_render = parsed.page_render
                if is_page_render:
                    if html_loads >= max_html_loads:
                        self.next_render_at[idx] = now + defer_interval + self.page_phase[idx] * 0.05
                        keep_dirty.add(idx)
                        continue
                elif text_loads >= max_text_loads:
                    self.next_render_at[idx] = now + preview_refresh_interval * 0.5 + self.page_phase[idx] * 0.03
                    keep_dirty.add(idx)
                    continue
                should_render = True
                if not forced and is_page_render and self.last_rendered_stage[idx] == "html":
                    if is_finished:
                        should_render = html_bytes != self.last_rendered_html_bytes[idx]
                    else:
                        byte_delta = html_bytes - self.last_rendered_html_bytes[idx]
                        stale = now - self.last_rendered_at[idx]
                        should_render = byte_delta >= html_min_delta or stale >= html_max_stale
                if not should_render:
                    self.next_render_at[idx] = now + defer_interval + self.page_phase[idx] * 0.05
                    keep_dirty.add(idx)
                    continue
                batch.append(
                    {
                        "index": self.local_index(idx),
                        "globalIndex": idx,
                        "html": parsed.render_html,
                        "stage": parsed.stage,
                        "captionStage": "finish" if is_finished else parsed.stage,
                        "totalBytes": total_bytes,
                        "autoScroll": not is_page_render,
                        "epoch": self.view_epoch,
                    }
                )
                self.last_rendered_stage[idx] = parsed.stage
                self.last_rendered_html_bytes[idx] = html_bytes
                self.last_rendered_at[idx] = now
                self.force_render.discard(idx)
                if is_page_render:
                    html_loads += 1
                    interval = html_refresh_interval
                else:
                    text_loads += 1
                    interval = preview_refresh_interval
                self.next_render_at[idx] = now + interval + self.page_phase[idx] * 0.08
                next_cursor = (self.local_index(idx) + 1) % self.visible_count
            self.dirty = keep_dirty
            self.render_cursor = next_cursor
            if batch:
                self.run_js("window.updatePages(" + json.dumps(batch, ensure_ascii=False) + ");")

    app = QtWidgets.QApplication(sys.argv[:1])
    view = HtmlGridView()
    screen = app.primaryScreen()
    if screen is not None:
        view.setGeometry(screen.availableGeometry())
    maximized = getattr(getattr(QtCore.Qt, "WindowState", QtCore.Qt), "WindowMaximized")
    view.setWindowState(view.windowState() | maximized)
    view.show()
    QtCore.QTimer.singleShot(0, view.showMaximized)
    QtCore.QTimer.singleShot(100, view.showMaximized)
    rc = app.exec()
    shutdown_event.set()
    return rc
