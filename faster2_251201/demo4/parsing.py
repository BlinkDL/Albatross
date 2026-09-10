import html
import re
from dataclasses import dataclass


@dataclass
class ParsedPage:
    thinking_text: str
    answer_text: str
    html_text: str
    render_html: str
    stage: str
    page_render: bool


def wrap_partial_html(fragment, caption="HTML is still streaming", show_caption=True):
    escaped_caption = html.escape(caption)
    caption_html = f'<div class="pending"><h1>{escaped_caption}</h1></div>' if show_caption else ""
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html, body {{ margin: 0; min-height: 100%; font-family: Inter, system-ui, sans-serif; }}
    body {{ background: #f6f4ef; color: #202020; }}
    .pending {{ padding: 24px; border: 2px dashed #999; margin: 18px; border-radius: 12px; }}
    .pending h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .raw-output {{ margin: 0; padding: 18px 20px 28px; white-space: pre-wrap; overflow-wrap: anywhere; font: 24px/1.0 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  </style>
</head>
<body>
  {caption_html}
  {fragment}
</body>
</html>"""


def render_raw_text_preview(raw):
    text = compact_stream_text("<think" + raw, 9000)
    fragment = f'<pre class="raw-output">{html.escape(text)}</pre>'
    return wrap_partial_html(fragment, show_caption=False)


def compact_stream_text(text, limit=6000):
    text = text.strip()
    if len(text) <= limit:
        return text
    keep_head = max(800, limit // 4)
    keep_tail = max(1200, limit - keep_head)
    omitted = len(text) - keep_head - keep_tail
    return text[:keep_head] + f"\n\n... omitted {omitted} characters ...\n\n" + text[-keep_tail:]


def render_html_source_preview(raw, html_text):
    fragment = f'<pre class="raw-output">{html.escape(compact_stream_text(raw, 9000))}</pre>'
    return wrap_partial_html(fragment, show_caption=False)


def html_has_closed_document(html_text):
    lower = html_text.lower()
    return "</html>" in lower or "</body>" in lower


def html_visual_ready(html_text):
    lower = html_text.lower()
    body_match = re.search(r"<body\b[^>]*>", lower)
    if not body_match:
        return html_has_closed_document(html_text)
    after_body = html_text[body_match.end() :]
    if len(after_body.strip()) < 80 and not html_has_closed_document(html_text):
        return False
    return re.search(r"<(header|main|section|article|nav|div|h1|h2|p|ul|ol|img|button|footer)\b", after_body, re.I) is not None


FENCE_RE = re.compile(r"```[ \t]*([a-zA-Z0-9_-]*)[^\S\r\n]*(?:\r?\n|$)")
CLOSING_FENCE_RE = re.compile(r"(^|\r?\n)```[ \t]*(?=\r?\n|$)")
OPENING_FENCE_LINE_RE = re.compile(r"```[ \t]*([a-zA-Z0-9_-]*)[^\S\r\n]*$")
ONLY_CLOSING_FENCE_LINE_RE = re.compile(r"^[ \t]*```[ \t]*(?:\r?\n)?$")
MARKER_BACKTRACK = 32


def find_closing_fence(text, start):
    match = CLOSING_FENCE_RE.search(text, start)
    if not match:
        return None
    return match.start() + len(match.group(1)), match.end()


def find_html_candidate(text):
    for match in FENCE_RE.finditer(text):
        content_start = match.end()
        close = find_closing_fence(text, content_start)
        content = text[content_start:] if close is None else text[content_start : close[0]]
        lang = match.group(1).lower()
        if lang == "html" or "<html" in content.lower() or "<!doctype" in content.lower():
            return content, match.start()

    lower = text.lower()
    starts = [pos for pos in (lower.find("<!doctype"), lower.find("<html")) if pos >= 0]
    if starts:
        start = min(starts)
        return text[start:], start
    return "", None


@dataclass
class HtmlCompletionScanner:
    initial_text: str = ""
    scan_pos: int = 0
    marker_scan_pos: int = 0
    in_fence: bool = False
    html_candidate: bool = False
    html_started: bool = False
    think_closed: bool = False
    content_offset: int = 0
    raw_content_offset: int = 0

    def update(self, text):
        scan_text = self.initial_text + text
        if not self.think_closed:
            close = scan_text.find("</think>")
            if close < 0:
                return None
            self.think_closed = True
            self.content_offset = close + len("</think>")
            self.raw_content_offset = max(0, self.content_offset - len(self.initial_text))
            self.scan_pos = 0
            self.marker_scan_pos = 0
            self.in_fence = False
            self.html_candidate = False
            self.html_started = False

        content = scan_text[self.content_offset :]
        frozen_content = self._update_html_content(content)
        if frozen_content is None:
            return None
        return text[: self.raw_content_offset + len(frozen_content)]

    def _update_html_content(self, text):
        direct_end = self._scan_direct_markers(text)
        if direct_end is not None:
            return text[:direct_end]

        while True:
            line_end = text.find("\n", self.scan_pos)
            if line_end < 0:
                break
            end = line_end + 1
            line = text[self.scan_pos:end]
            frozen = self._scan_line(text, line, end)
            if frozen is not None:
                return frozen
            self.scan_pos = end

        trailing = text[self.scan_pos :]
        if self.in_fence and trailing:
            if not self.html_candidate and self._line_has_html_start(trailing):
                self.html_candidate = True
                self.html_started = True
            if self.html_candidate and ONLY_CLOSING_FENCE_LINE_RE.match(trailing):
                return text
        return None

    def _scan_line(self, full_text, line, end):
        if not self.in_fence:
            match = OPENING_FENCE_LINE_RE.search(line.rstrip("\r\n"))
            if match:
                self.in_fence = True
                self.html_candidate = match.group(1).lower() == "html"
                self.html_started = self.html_started or self.html_candidate
            return None

        if not self.html_candidate and self._line_has_html_start(line):
            self.html_candidate = True
            self.html_started = True
        if ONLY_CLOSING_FENCE_LINE_RE.match(line):
            if self.html_candidate:
                return full_text[:end]
            self.in_fence = False
            self.html_candidate = False
        return None

    @staticmethod
    def _line_has_html_start(line):
        lower = line.lower()
        return "<html" in lower or "<!doctype" in lower

    def _scan_direct_markers(self, text):
        start = max(0, self.marker_scan_pos - MARKER_BACKTRACK)
        lower = text[start:].lower()
        if not self.html_started and ("<html" in lower or "<!doctype html" in lower):
            self.html_started = True
        close = lower.find("</html>") if self.html_started else -1
        self.marker_scan_pos = max(0, len(text) - MARKER_BACKTRACK)
        if close >= 0:
            return start + close + len("</html>")
        return None


def parse_generated_page(raw):
    raw = raw.split("<|endoftext|>", 1)[0]
    assistant_text = "<think" + raw
    body = assistant_text[len("<think") :]
    if body.startswith(">"):
        body = body[1:]

    close = body.find("</think>")
    if close < 0:
        fallback = body
        html_text, html_start = find_html_candidate(fallback)
        if html_start is not None:
            answer = fallback[:html_start].strip()
            page_render = html_visual_ready(html_text)
            render_html = html_text.strip() if page_render else render_html_source_preview(raw, html_text)
            if page_render and "<html" not in render_html.lower() and "<!doctype" not in render_html.lower():
                render_html = wrap_partial_html(render_html, "Partial HTML")
            return ParsedPage("", answer, html_text, render_html, "html", page_render)
        thinking = body
        render = render_raw_text_preview(raw)
        return ParsedPage(thinking, "", "", render, "think", False)

    thinking = body[:close]
    after = body[close + len("</think>") :]
    html_text, html_start = find_html_candidate(after)
    if html_start is None:
        answer = after.strip()
    else:
        answer = after[:html_start].strip()

    if html_text.strip():
        page_render = html_visual_ready(html_text)
        if page_render:
            render_html = html_text.strip()
            if "<html" not in render_html.lower() and "<!doctype" not in render_html.lower():
                render_html = wrap_partial_html(render_html, "Partial HTML")
            stage = "html"
        else:
            render_html = render_html_source_preview(raw, html_text)
            stage = "html"
    elif answer:
        render_html = render_raw_text_preview(raw)
        stage = "answer"
        page_render = False
    else:
        render_html = render_raw_text_preview(raw)
        stage = "answer"
        page_render = False

    return ParsedPage(thinking, answer, html_text, render_html, stage, page_render)
