import gc, html, json, re, threading, time
import gradio as gr
import torch
from datetime import datetime
from huggingface_hub import hf_hub_download
from rwkv.utils import PIPELINE

import rwkv7_fast_v3a as v3a

ctx_limit = 7000
gen_limit = 5000
max_bsz = 64
CHUNK_LEN = 512 # chunk prefill, save VRAM
SAMPLER_TOP_K = 500
YIELD_EVERY = 16
USE_CUDA_GRAPH = True
html_gen_limit = 16000
MAX_HTML_PREVIEWS = 30
HTML_GRID_COLUMNS = 3
MAX_BATCH_PREVIEWS = 320
BATCH_GRID_COLUMNS = 10
BATCH_UPDATE_INTERVAL = 0.5
BATCH_YIELD_EVERY = 10
HTML_DEFAULT_BATCH = min(MAX_HTML_PREVIEWS, max_bsz)
HTML_MAX_BATCH = min(MAX_HTML_PREVIEWS, max_bsz)
HTML_GRID_UPDATE_EVERY = 32
HTML_IFRAME_MIN_DELTA = 1800
HTML_IFRAME_MAX_STALE = 2.0
HTML_COMPONENT_MIN_INTERVAL = 0.5
HTML_PREVIEW_HEIGHT = 300
HTML_CAPTION_HEIGHT = 15
HTML_BODY_HEIGHT = HTML_PREVIEW_HEIGHT - HTML_CAPTION_HEIGHT
HTML_FRAME_HEIGHT = 228
HTML_RAW_HEIGHT = HTML_BODY_HEIGHT - HTML_FRAME_HEIGHT
HTML_PROMPT_CHOICES = [
    "3D animation of cars in forest with animals",
    "interactive weather map with animated clouds and rain",
    "User: Generate an SVG of a pelican riding a bicycle\n\nAssistant: <think></think",
    "User: Generate an SVG of Windows 11 desktop\n\nAssistant: <think></think",
    "retro arcade RPG start screen",
    "3D animation of a SpaceX rocket landing on Mars",
    "storybook scene with a dragon flying over a castle",
    "interactive dashboard for a city traffic system",
    "animated aquarium with colorful fish and coral",
    "sci-fi spaceship navigation interface",
    "cozy cafe menu with animated steam and pastries",
    "character sheet for a high fantasy RPG",
    "a fancy hotel homepage",
]

def html_prompt_from_choice(choice):
    if "User:" in choice:
        return choice
    return f"User: Write HTML: {choice}\n\nAssistant: <think></think"

DEFAULT_HTML_PROMPT = html_prompt_from_choice(HTML_PROMPT_CHOICES[0])
HTML_GRID_CSS = """
div.main { padding-left: 0 !important; padding-right: 0 !important; }
.html-grid-tab { padding-top: 0 !important; }
.html-grid-main { display: grid !important; grid-template-columns: minmax(220px, 17.5%) 1fr !important; grid-template-rows: auto !important; gap: 4px !important; margin-top: 0 !important; align-items: start !important; }
.html-grid-main > div { gap: 4px !important; }
.html-grid-controls { grid-column: 1 !important; grid-row: 1 !important; gap: 4px !important; min-width: 0 !important; }
.html-grid-pages { grid-column: 2 !important; grid-row: 1 !important; gap: 4px !important; min-width: 0 !important; }
.html-grid-preview-row { gap: 4px !important; margin: 0 !important; }
.html-grid-preview { margin: 0 !important; }
.html-grid-preview > div { margin: 0 !important; }
.html-container { padding: 0 !important; }
dialog.html-container { position:static; inset:auto; box-sizing:border-box; width:100%; max-width:none; max-height:none; margin:0; border:0; color:#111; }
.html-preview-caption { display:flex; align-items:center; gap:6px; flex:none; min-width:0; }
.html-preview-caption > span { flex:1; min-width:0; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; color:#fff!important; }
.html-preview-button { display:inline-flex; align-items:center; justify-content:center; flex:none; box-sizing:border-box; width:20px; height:13px; min-height:0!important; margin:0!important; padding:0; line-height:1; border:0; border-radius:3px; color:#fff!important; background:transparent; cursor:pointer; }
.html-preview-button:hover { background:#444; }
.html-preview-button:focus-visible { outline:2px solid #63c9ff; outline-offset:-2px; }
.html-preview-button svg { width:12px; height:12px; pointer-events:none; }
.html-preview-button svg, .html-preview-button path { color:#fff!important; stroke:#fff!important; }
.html-preview-close { display:none; }
.html-grid-preview { min-height:300px; }
html.html-preview-modal-open { overflow:hidden!important; }
dialog.html-container.html-preview-expanded {
  position:fixed!important; inset:0!important; margin:auto!important;
  width:95vw!important; height:95vh!important; height:95dvh!important;
  max-width:none!important; max-height:none!important; opacity:1!important;
  border-radius:6px; outline:1px solid #555!important; overflow:hidden;
}
dialog.html-preview-expanded::backdrop { background:rgba(0,0,0,.72); }
.html-preview-expanded .html-preview-caption { height:40px!important; padding:4px 10px!important; font-size:13px!important; }
.html-preview-expanded .html-preview-button { width:32px; height:32px; }
.html-preview-expanded .html-preview-button svg { width:18px; height:18px; }
.html-preview-expanded .html-preview-expand { display:none; }
.html-preview-expanded .html-preview-close { display:inline-flex; }
.html-preview-expanded .html-preview-body { flex:1; min-height:0; height:auto!important; }
.html-preview-expanded .html-preview-frame { flex:1; min-height:0; height:100%!important; }
.html-preview-expanded iframe { width:100%!important; height:100%!important; transform:none!important; }
.html-preview-expanded .html-preview-raw { display:none!important; }
.batch-preview-content { flex:1; min-height:0; overflow:auto; background:#fafafa; color:#111; }
.batch-preview-content pre { margin:0; padding:8px; white-space:pre-wrap; overflow-wrap:anywhere; font:9.5px/1.4 monospace; letter-spacing:0; }
.html-preview-expanded .batch-preview-content pre { font-size:32px; }
.batch-text-grid { display:grid; min-width:1200px; grid-template-columns:repeat(var(--batch-grid-columns, 10), minmax(0, 1fr)); gap:4px; }
.batch-grid-host { overflow-x:auto!important; min-width:0!important; }
.batch-text-grid > dialog { width:100%; min-width:0; box-sizing:border-box; margin:0; }
.batch-text-updates { display:none!important; }
.html-prompt-choice { margin: 0 !important; padding: 0 !important; min-height: 0 !important; }
.html-prompt-choice .wrap,
.html-prompt-choice .wrap-inner,
.html-prompt-choice .secondary-wrap { margin: 0 !important; padding: 0 !important; min-height: 0 !important; }
.html-prompt-choice label { margin: 0 !important; padding: 0 !important; }
@media (max-width: 768px) {
  .html-grid-main { grid-template-columns: 1fr !important; grid-template-rows: auto auto !important; }
  .html-grid-controls { grid-column: 1 !important; grid-row: 1 !important; }
  .html-grid-pages { grid-column: 1 !important; grid-row: 2 !important; }
}
"""

# Native dialog top-layer promotion keeps the iframe attached to its original
# parent. Reparenting/cloning an iframe reloads it and loses interactive state.
HTML_GRID_JS = r"""() => {
  if (window.__rwkvPreviewExpandInstalled) return;
  window.__rwkvPreviewExpandInstalled = true;
  let active = null;
  function updateBatch(packet) {
    const grid = document.getElementById('batch-text-grid');
    if (!grid) return;
    let data;
    try { data = JSON.parse(packet.dataset.batchPacket); } catch { return; }
    if (data.reset && active?.index.startsWith('batch-')) closePreview();
    const dialogs = grid.querySelectorAll('dialog');
    const scroll = [];
    for (let i = 0; i < dialogs.length; i++) {
      const dialog = dialogs[i];
      const enabled = i < data.count;
      const text = data.texts[i] || '';
      const pane = dialog.querySelector('.batch-preview-content');
      const pre = pane.querySelector('pre');
      const previous = pre.textContent;
      // Snapshots tolerate Gradio coalescing updates. Append only the suffix
      // to a stable text node; generated markup is never parsed as HTML.
      if (text !== previous) {
        if (pre.firstChild && text.startsWith(previous)) pre.firstChild.appendData(text.slice(previous.length));
        else pre.textContent = text;
      }
      const caption = enabled ? `#${i + 1} | ${(data.tokens[i] || 0).toLocaleString()} tokens` : `#${i + 1}`;
      const label = dialog.querySelector('.html-preview-caption > span');
      if (label.textContent !== caption) label.textContent = caption;
      dialog.style.opacity = enabled ? '1' : '.35';
      dialog.querySelector('.html-preview-expand').disabled = !enabled;
      if (text !== previous || data.reset) scroll.push(pane);
    }
    // Batch layout reads before scroll writes; avoid 320 forced reflows.
    if (scroll.length) requestAnimationFrame(() => {
      const heights = scroll.map(pane => pane.scrollHeight);
      scroll.forEach((pane, i) => { pane.scrollTop = heights[i]; });
    });
  }
  function scrollText(dialog) {
    const pane = dialog.querySelector('.batch-preview-content');
    if (pane) requestAnimationFrame(() => { pane.scrollTop = pane.scrollHeight; });
  }
  function notifyFrame(dialog, expanded) {
    dialog.querySelector('iframe')?.contentWindow?.postMessage(
      {type:'rwkv-preview-mode', expanded}, '*');
  }
  function promote(dialog) {
    dialog.classList.add('html-preview-expanded');
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    // An already-open nonmodal dialog must be closed before showModal().
    // Toggle synchronously without removing/reinserting any DOM nodes.
    if (!dialog.matches(':modal')) {
      dialog.removeAttribute('open');
      dialog.showModal();
    }
    dialog.querySelector('.html-preview-close').focus({preventScroll:true});
    notifyFrame(dialog, true);
    scrollText(dialog);
  }
  function closePreview() {
    if (!active) return;
    const saved = active;
    active = null;
    const dialog = saved.dialog;
    notifyFrame(dialog, false);
    dialog.close();
    dialog.classList.remove('html-preview-expanded');
    dialog.setAttribute('role', 'group');
    dialog.removeAttribute('aria-modal');
    dialog.setAttribute('open', '');
    scrollText(dialog);
    document.documentElement.classList.remove('html-preview-modal-open');
    const target = saved.trigger.isConnected ? saved.trigger :
      document.querySelector(`dialog[data-preview-index="${saved.index}"] .html-preview-expand`);
    target?.focus({preventScroll:true});
    window.scrollTo(saved.x, saved.y);
  }
  document.addEventListener('click', event => {
    const button = event.target.closest?.('.html-preview-button');
    if (button?.classList.contains('html-preview-expand')) {
      const dialog = button.closest('dialog[data-preview-index]');
      if (!dialog?.querySelector('iframe, .batch-preview-content')) return;
      closePreview();
      active = {dialog, host:dialog.closest('.html-grid-preview'), index:dialog.dataset.previewIndex, trigger:button,
        x:window.scrollX, y:window.scrollY};
      document.documentElement.classList.add('html-preview-modal-open');
      promote(dialog);
    } else if (button?.classList.contains('html-preview-close')) {
      closePreview();
    } else if (active && event.target === active.dialog) {
      const r = active.dialog.getBoundingClientRect();
      if (event.clientX < r.left || event.clientX > r.right ||
          event.clientY < r.top || event.clientY > r.bottom) closePreview();
    }
  });
  document.addEventListener('cancel', event => {
    if (active && event.target === active.dialog) {
      event.preventDefault();
      closePreview();
    }
  }, true);
  document.addEventListener('close', event => {
    if (active && event.target === active.dialog && active.dialog.isConnected && !active.dialog.open) closePreview();
  }, true);
  window.addEventListener('message', event => {
    // Sandbox origins are opaque: validate the actual WindowProxy, not origin.
    if (!active || event.source !== active.dialog.querySelector('iframe')?.contentWindow) return;
    if (event.data?.type === 'rwkv-preview-escape') closePreview();
    else if (event.data?.type === 'rwkv-preview-ready') notifyFrame(active.dialog, true);
  });
  // Gradio replaces rendered HTML during streaming. Reopen only the same grid;
  // updates may reload content as before, but never lose the expanded mode.
  new MutationObserver(records => {
    // Follow only changed text outputs, not unrelated controls or manual scroll.
    const changed = new Set();
    for (const record of records) {
      if (record.type === 'attributes' && record.attributeName === 'data-batch-packet') updateBatch(record.target);
      if (record.type !== 'childList') continue;
      const parent = record.target.nodeType === 1 ? record.target : record.target.parentElement;
      const pane = parent?.closest('.batch-preview-content');
      if (pane) changed.add(pane);
      for (const node of record.addedNodes) {
        if (node.nodeType !== 1) continue;
        if (node.matches('[data-batch-packet]')) updateBatch(node);
        node.querySelectorAll('[data-batch-packet]').forEach(updateBatch);
        if (node.matches('.batch-preview-content')) changed.add(node);
        node.querySelectorAll('.batch-preview-content').forEach(el => changed.add(el));
      }
    }
    if (changed.size) requestAnimationFrame(() => {
      changed.forEach(pane => { if (pane.isConnected) pane.scrollTop = pane.scrollHeight; });
    });
    if (!active) return;
    const next = document.querySelector(`dialog[data-preview-index="${active.index}"]`);
    // Gradio may clear HTML in one render pass and insert it in the next.
    // Keep the mode through that gap, but release it if the component is gone.
    if (!next) {
      if (!active.host?.isConnected) closePreview();
      return;
    }
    if (!next.querySelector('iframe, .batch-preview-content')) { closePreview(); return; }
    // Some Gradio versions morph existing elements and reset their attributes
    // instead of replacing them. Restore the modal presentation in either case.
    if (next !== active.dialog || !next.classList.contains('html-preview-expanded') || !next.matches(':modal')) {
      active.dialog = next;
      promote(next);
    }
  }).observe(document.body, {childList:true, subtree:true, attributes:true, attributeFilter:['class', 'open', 'data-batch-packet']});
}"""

# Local Lucide Maximize2 / X icon geometry (ISC); no CDN dependency.
HTML_EXPAND_ICON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>'
HTML_CLOSE_ICON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m18 6-12 12M6 6l12 12"/></svg>'

html_view_lock = threading.Lock()
html_view_scale = 35
html_view_scroll_seconds = 5

def clamp_html_scale(scale):
    return max(20, min(100, int(scale)))

def clamp_scroll_seconds(seconds):
    return max(0, min(10, int(seconds)))

def set_html_scale(scale):
    global html_view_scale
    scale = clamp_html_scale(scale)
    with html_view_lock:
        html_view_scale = scale
    return scale

def get_html_scale():
    with html_view_lock:
        return html_view_scale

def set_scroll_seconds(seconds):
    global html_view_scroll_seconds
    seconds = clamp_scroll_seconds(seconds)
    with html_view_lock:
        html_view_scroll_seconds = seconds
    return seconds

def get_scroll_seconds():
    with html_view_lock:
        return html_view_scroll_seconds

########################## text rwkv ################################################################

# title = "rwkv-g1k-1b-5748"
# title = "rwkv-g1k-3b-5096"
# title = "rwkv-g1k-7b-3563"
title = "rwkv-g1k-13b-5228"
# model_path = hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename=f"{title}.pth")
model_path = f"/dev/shm/{title}.pth"
# model_path = f"/mnt/e/RWKV-Runner/models/{title}.pth"

v3a.MODEL_PATH = model_path
v3a.WKV_MODE = "fp32io16"
v3a.EMB_DEVICE = "cuda"
v3a.RKV_MODE = "off"
v3a.CMIX_SPARSE = "no-fc"
v3a.LOWRANK_WEIGHT = "transpose"
v3a.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
v3a.load_extensions(v3a.WKV_MODE)
model = v3a.RWKV7()
gc.collect()
torch.cuda.empty_cache()
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

@torch.jit.script
def sample_logits_batch_cuda(logits, temperature: float, top_p: float, k: int):
    if top_p <= 0.0 or k == 1:
        return torch.argmax(logits, dim=-1)
    vals, ids = torch.topk(logits.float(), k=k, dim=-1, sorted=True)
    if temperature == 1.0:
        probs = torch.softmax(vals, dim=-1)
    else:
        probs = torch.softmax(vals / temperature, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    if top_p < 1.0:
        keep = torch.argmax((cdf >= top_p).to(torch.int32), dim=-1)
        mass = cdf.gather(1, keep.view(-1, 1)).view(-1)
    else:
        mass = cdf[:, -1]
    r = torch.rand((logits.size(0), 1), device=logits.device) * mass.view(-1, 1)
    out = torch.searchsorted(cdf, r).view(-1, 1)
    return ids.gather(1, out).view(-1)

def get_decode_ctx(B: int, decode_cache):
    key = (B, v3a.WKV_MODE)
    cached = decode_cache.get(key)
    if cached is not None:
        return cached
    state = model.zero_state(B)
    x = torch.empty((B, 1, v3a.C), device="cuda", dtype=torch.half)
    path = v3a.select_path(B, 1)
    for _ in range(2):
        model.forward_from_x(x, state, path)
    torch.cuda.synchronize()
    graph = None
    output = None
    if USE_CUDA_GRAPH:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = model.forward_from_x(x, state, path)
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"CUDA graph disabled for B={B}: {exc}", flush=True)
            graph = None
            output = None
    cached = (state, x, graph, output)
    decode_cache[key] = cached
    return cached

def copy_state_to_batch(dst, src):
    B = dst[2].shape[0]
    dst[0].copy_(src[0].expand(-1, -1, B, -1))
    dst[1].copy_(src[1].expand(-1, B, -1, -1, -1))
    dst[2].copy_(src[2].expand(B))

def tokens_to_x(tokens):
    token_device = "cpu" if model.emb_cpu else "cuda"
    if isinstance(tokens, torch.Tensor):
        token_tensor = tokens.to(device=token_device, dtype=torch.long, non_blocking=True).view(-1, 1)
    else:
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=token_device).view(-1, 1)
    return model.embed(token_tensor)

def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"Instruction: {instruction}\n\nInput: {input}\n\nResponse:"
    else:
        return f"User: {instruction}\n\nAssistant: <think></think"

def qa_prompt(instruction):
    instruction = instruction.strip().replace('\r\n','\n')
    instruction = re.sub(r'\n+', '\n', instruction)
    return f"User: {instruction}\n\nAssistant: <think></think"

def output_update(text, speed="", compact=False):
    if compact and speed:
        speed = speed.split(" @ ", 1)[0]
    label = f"Output  {speed}" if speed else "Output"
    return gr.update(value=text, label=label)

def output_text(B, out_str):
    return out_str[0].strip() if B == 1 else "\n====\n".join(x.strip() for x in out_str)

def speed_text(rate, B, tokens, chars):
    return f"{rate:.1f} token/s @ bsz {B} = {tokens} tokens, {chars} chars"

def gib(n):
    return n / 1_000_000_000.0

class GenerationOwner:
    """Own GPU frames independently of Gradio's cancelled iterator lifetime."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = None

    def stream(self, factory, *args, **kwargs):
        with self.lock:
            # Cancellation can release Gradio's queue slot without closing the
            # nested Python generator. Close it before allocating another state.
            if self.active is not None:
                self.active.close()
            inner = factory(*args, **kwargs)
            self.active = inner
        try:
            while True:
                with self.lock:
                    if self.active is not inner:
                        return
                    try:
                        item = next(inner)
                    except StopIteration:
                        return
                    if item[2].get("done"):
                        inner.close()
                        self.active = None
                yield item
        finally:
            with self.lock:
                inner.close()
                if self.active is inner:
                    self.active = None


generation_owner = GenerationOwner()


def generate_batch_text(*args, **kwargs):
    yield from generation_owner.stream(_generate_batch_text, *args, **kwargs)


def _generate_batch_text(
    ctx,
    token_count=200,
    batch_size=1,
    temperature=1.0,
    top_p=0.5,
    presencePenalty = 1,
    countPenalty = 0.1,
    penalty_decay = 0.99,
    yield_every = YIELD_EVERY,
    wkv_mode = "fp32io16",
    batch_limit = max_bsz,
    combine_output = True,
):
    state = decode_state = decode_x = decode_graph = decode_output = None
    decode_cache = None
    out = logits = tokens = sampled_tensor = None
    occurrence_count = occurrence_presence = batch_rows = None
    try:
        # Both UI generators share one queue slot: never change the global route
        # while another request is using its state or replaying a captured graph.
        v3a.set_wkv_mode(wkv_mode)
        req_t0 = time.perf_counter()
        user_token_count = int(token_count)
        rwkv_model = model
        pipe = pipeline
        sample_temperature = float(temperature)
        sample_top_p = float(top_p)
        if sample_temperature <= 0:
            sample_temperature = 1.0
            sample_top_p = 0
        else:
            sample_temperature = max(0.2, sample_temperature)
        alpha_frequency = float(countPenalty)
        alpha_presence = float(presencePenalty)
        ctx = ctx.strip()
        input_ids = pipe.encode(ctx)[-ctx_limit:]
        input_token_count = len(input_ids)
        B = min(batch_limit, max(1, int(batch_size)))
        batch_rows = None
        all_tokens = [[] for _ in range(B)]
        out_last = [0 for _ in range(B)]
        out_str = ['' for _ in range(B)]
        occurrence_count = None
        occurrence_presence = None
        finished = [False for _ in range(B)]
        speed_t0 = None
        speed_tokens = 0
        total_tokens = 0
        speed_info = ""
        decode_cache = {}
        state = rwkv_model.zero_state(1)
        decode_state, decode_x, decode_graph, decode_output = get_decode_ctx(B, decode_cache)
        next_tokens = [0 for _ in range(B)]
        out = None
        for i in range(int(token_count)):

            if i == 0:
                if len(input_ids) == 0:
                    yield "", "", {"done": True, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "B": B, "T": user_token_count, "In": input_token_count, "TPS": 0.0, "Time": 0.0, "Token": 0, "Char": 0, "VRAMUsed": 0, "VRAMTotal": 0, "token_counts": [0 for _ in range(B)], "texts": out_str.copy()}
                    return
                while len(input_ids) > 0:
                    token_device = "cpu" if rwkv_model.emb_cpu else "cuda"
                    tokens = torch.tensor(input_ids[:CHUNK_LEN], dtype=torch.long, device=token_device)
                    out = rwkv_model.forward(tokens, state).view(-1)
                    input_ids = input_ids[CHUNK_LEN:]
                torch.cuda.synchronize()
                copy_state_to_batch(decode_state, state)
                logits = out.view(1, -1).repeat(B, 1)
            else:
                decode_x.copy_(tokens_to_x(next_tokens))
                if decode_graph is None:
                    decode_output = rwkv_model.forward_from_x(decode_x, decode_state, v3a.select_path(B, 1))
                else:
                    decode_graph.replay()
                logits = decode_output.view(B, -1)

            if occurrence_count is None:
                occurrence_count = torch.zeros((B, logits.size(-1)), device=logits.device, dtype=logits.dtype)
                occurrence_presence = torch.zeros_like(occurrence_count)
                batch_rows = torch.arange(B, device=logits.device)
            if alpha_frequency:
                logits.sub_(occurrence_count, alpha=alpha_frequency)
            if alpha_presence:
                logits.sub_(occurrence_presence)

            assert logits.is_cuda and logits.dim() == 2
            sampled_tensor = sample_logits_batch_cuda(
                logits,
                sample_temperature,
                sample_top_p,
                min(SAMPLER_TOP_K, logits.size(-1)),
            )
            sampled = sampled_tensor.detach().cpu().tolist()
            active = 0
            next_tokens = [0 for _ in range(B)]
            if penalty_decay != 1:
                occurrence_count.mul_(penalty_decay)
            occurrence_count[batch_rows, sampled_tensor] += 1
            if alpha_presence:
                occurrence_presence[batch_rows, sampled_tensor] = alpha_presence
            for b in range(B):
                if finished[b]:
                    continue
                token = sampled[b]
                if token == 0:
                    finished[b] = True
                    continue
                active += 1
                next_tokens[b] = token
                all_tokens[b].append(token)

                tmp = pipe.decode(all_tokens[b][out_last[b]:])
                if '\ufffd' not in tmp:
                    out_str[b] += tmp
                    out_last[b] = len(all_tokens[b])
            total_tokens += active
            if active == 0:
                break
            if speed_t0 is None:
                speed_t0 = time.perf_counter()
            else:
                speed_tokens += B
                elapsed = max(1e-9, time.perf_counter() - speed_t0)
                current_text = output_text(B, out_str) if combine_output else ""
                char_count = len(current_text) if combine_output else sum(map(len, out_str))
                speed_info = speed_text(speed_tokens / elapsed, B, total_tokens, char_count)
            if i == 0 or i % max(1, int(yield_every)) == 0:
                current_text = output_text(B, out_str) if combine_output else ""
                yield current_text, speed_info, {"done": False, "token_counts": [len(tokens) for tokens in all_tokens], "texts": out_str.copy()}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        free, total = torch.cuda.mem_get_info()
        current_text = output_text(B, out_str) if combine_output else ""
        char_count = len(current_text) if combine_output else sum(map(len, out_str))
        if speed_t0 is not None and not speed_info:
            speed_info = speed_text(0.0, B, total_tokens, char_count)
        elapsed = time.perf_counter() - req_t0
        final_tps = speed_tokens / max(1e-9, time.perf_counter() - speed_t0) if speed_t0 is not None else 0.0
        used = total - free
        meta = {"done": True, "timestamp": timestamp, "B": B, "T": user_token_count, "In": input_token_count, "TPS": final_tps, "Time": elapsed, "Token": total_tokens, "Char": char_count, "VRAMUsed": used, "VRAMTotal": total, "token_counts": [len(tokens) for tokens in all_tokens], "texts": out_str.copy()}
        yield current_text, speed_info, meta
    finally:
        # Clear every CUDA alias even if an exception traceback retains this frame.
        # Stop may leave the outer Gradio iterator alive until its next request.
        try:
            if decode_graph is not None:
                torch.cuda.synchronize()
                decode_graph.reset()
        finally:
            if decode_cache is not None:
                decode_cache.clear()
            state = decode_state = decode_x = decode_graph = decode_output = None
            decode_cache = out = logits = tokens = sampled_tensor = None
            occurrence_count = occurrence_presence = batch_rows = None

def print_summary(prefix, meta):
    print(
        f"[{prefix}] {meta['timestamp']} B={meta['B']} T={meta['T']} In={meta['In']} "
        f"TPS={meta['TPS']:.1f} Time={meta['Time']:.3f}s Token={meta['Token']} "
        f"Char={meta['Char']} VRAM={gib(meta['VRAMUsed']):.2f}G/{gib(meta['VRAMTotal']):.2f}G",
        flush=True,
    )

def evaluate_raw(
    ctx,
    token_count=200,
    batch_size=1,
    temperature=1.0,
    top_p=0.5,
    presencePenalty = 1,
    countPenalty = 0.1,
    penalty_decay = 0.99,
):
    for text, speed_info, meta in generate_batch_text(ctx, token_count, batch_size, temperature, top_p, presencePenalty, countPenalty, penalty_decay, YIELD_EVERY, wkv_mode="fp32io16"):
        if meta and meta.get("done"):
            print_summary("app3", meta)
        yield output_update(text, speed_info)

def split_batch_output(text, count):
    parts = text.split("\n====\n") if text else []
    parts += [""] * max(0, count - len(parts))
    return parts[:count]

def extract_html(text, prompt=""):
    stream = prompt + text
    lower_text = stream.lower()
    marker = lower_text.find("</think>")
    if marker < 0:
        return ""
    visible = stream[marker + len("</think>"):]
    svg_fence = re.search(r"```svg[^\S\r\n]*\r?\n", visible, re.IGNORECASE)
    if svg_fence:
        code = visible[svg_fence.end():].split("```", 1)[0]
        svg_start = re.search(r"<svg\b[^>]*>", code, re.IGNORECASE)
        # Wait for the root opening tag, then let the iframe render partial SVG
        # while generation continues. Keep Markdown and trailing prose outside.
        if svg_start:
            svg_end = re.search(r"</svg\s*>", code[svg_start.end():], re.IGNORECASE)
            end = svg_start.end() + svg_end.end() if svg_end else len(code)
            return code[svg_start.start():end].strip()
    lower = visible.lower()
    start = lower.find("<!doctype html")
    if start < 0:
        return ""
    if lower.find("<body", start) < 0:
        return ""
    end = lower.find("</html>", start)
    if end >= 0:
        return visible[start:end + len("</html>")].strip()
    return visible[start:].strip()

def html_complete(page):
    if re.match(r"\s*<svg\b", page, re.IGNORECASE):
        return re.search(r"</svg\s*>\s*$", page, re.IGNORECASE) is not None
    return "</html>" in page.lower()

def inject_iframe_scroll_script(page, index, scroll_seconds):
    scroll_seconds = clamp_scroll_seconds(scroll_seconds)
    if scroll_seconds <= 0:
        return page
    delay = int((index * 733) % 5000)
    leg_ms = scroll_seconds * 1000
    script = f"""<script>
(() => {{
  const delay = {delay};
  const legMs = {leg_ms};
  let down = true;
  let running = false;
  function unique(list) {{
    const out = [];
    const seen = new Set();
    for (const el of list) {{
      if (!el || seen.has(el)) continue;
      seen.add(el);
      out.push(el);
    }}
    return out;
  }}
  function candidates() {{
    const base = [document.scrollingElement, document.documentElement, document.body, document.body && document.body.parentElement];
    const all = Array.from(document.querySelectorAll("*"));
    return unique(base.concat(all))
      .filter(el => Math.max(0, el.scrollHeight - el.clientHeight) > 4)
      .sort((a, b) => (b.scrollHeight - b.clientHeight) - (a.scrollHeight - a.clientHeight))
      .slice(0, 8);
  }}
  function getTop(el) {{
    if (el === document.scrollingElement || el === document.documentElement || el === document.body) {{
      return window.scrollY || document.documentElement.scrollTop || document.body.scrollTop || 0;
    }}
    return el.scrollTop || 0;
  }}
  function setTop(el, y) {{
    if (el === document.scrollingElement || el === document.documentElement || el === document.body) {{
      window.scrollTo(0, y);
      document.documentElement.scrollTop = y;
      document.body.scrollTop = y;
    }} else {{
      el.scrollTop = y;
    }}
  }}
  function animate() {{
    if (window.__rwkvPreviewExpanded) {{ setTimeout(animate, 200); return; }}
    const targets = candidates();
    if (!targets.length) {{
      setTimeout(animate, 1000);
      return;
    }}
    const starts = targets.map(getTop);
    const ends = targets.map(el => down ? Math.max(0, el.scrollHeight - el.clientHeight) : 0);
    down = !down;
    const t0 = performance.now();
    function step(t) {{
      if (window.__rwkvPreviewExpanded) {{ setTimeout(animate, 200); return; }}
      const p = Math.min(1, (t - t0) / legMs);
      for (let i = 0; i < targets.length; i++) {{
        setTop(targets[i], starts[i] + (ends[i] - starts[i]) * p);
      }}
      requestAnimationFrame(p < 1 ? step : animate);
    }}
    requestAnimationFrame(step);
  }}
  function start() {{
    if (running) return;
    running = true;
    setTimeout(animate, delay);
    try {{
      new MutationObserver(() => candidates()).observe(document.documentElement, {{childList: true, subtree: true}});
    }} catch (e) {{}}
    setInterval(candidates, 1000);
  }}
  if (document.readyState === "complete") start();
  else window.addEventListener("load", start, {{once: true}});
  setTimeout(start, delay + 1500);
}})();
</script>"""
    lower = page.lower()
    m = re.search(r"<head\b[^>]*>", lower)
    if m:
        return page[:m.end()] + script + page[m.end():]
    m = re.search(r"<html\b[^>]*>", lower)
    if m:
        return page[:m.end()] + "<head>" + script + "</head>" + page[m.end():]
    m = re.search(r"<!doctype\s+html[^>]*>", lower)
    if m:
        return page[:m.end()] + script + page[m.end():]
    return script + page

def fit_svg_document(page):
    # Keep the original SVG coordinate system, including SVGs without viewBox.
    # The iframe viewport, not the HTML zoom slider, determines the fit.
    return """<!doctype html><html><head><meta charset="utf-8">
<style>
html, body { margin:0!important; padding:0!important; width:100%!important;
height:100%!important; overflow:hidden!important; }
body { display:grid!important; place-items:center!important; }
</style><script>
window.addEventListener('DOMContentLoaded', () => {
  const svg = document.body.querySelector(':scope > svg');
  if (!svg) return;
  const box = svg.viewBox.baseVal;
  const rect = svg.getBoundingClientRect();
  function extent(name, fallback) {
    const attr = svg.getAttribute(name);
    const value = attr && !attr.includes('%') ? svg[name].baseVal.value : 0;
    return value > 0 ? value : fallback;
  }
  const width = extent('width', box.width > 0 ? box.width : rect.width || 300);
  const height = extent('height', box.height > 0 ? box.height : rect.height || 150);
  if (!(box.width > 0 && box.height > 0)) {
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  }
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  for (const [key, value] of Object.entries({
    display:'block', margin:'0', padding:'0', border:'0',
    position:'static', transform:'none', minWidth:'0', minHeight:'0'
  })) {
    svg.style.setProperty(key.replace(/[A-Z]/g, c => '-' + c.toLowerCase()), value, 'important');
  }
  function fit() {
    const scale = Math.min(window.__rwkvPreviewExpanded ? Infinity : 1, document.documentElement.clientWidth / width,
      document.documentElement.clientHeight / height);
    svg.style.setProperty('width', `${width * scale}px`, 'important');
    svg.style.setProperty('height', `${height * scale}px`, 'important');
  }
  new ResizeObserver(fit).observe(document.documentElement);
  window.addEventListener('resize', fit);
  window.addEventListener('rwkv-preview-mode', fit);
  fit();
});
</script></head><body>""" + page + "</body></html>"

def inject_preview_bridge(page):
    # Works for opaque-origin sandboxed documents without allow-same-origin.
    script = """<script>
window.__rwkvPreviewExpanded = false;
window.addEventListener('message', event => {
  if (event.source !== parent || event.data?.type !== 'rwkv-preview-mode') return;
  window.__rwkvPreviewExpanded = event.data.expanded === true;
  window.dispatchEvent(new Event('rwkv-preview-mode'));
});
window.addEventListener('keydown', event => {
  if (event.key === 'Escape' && window.__rwkvPreviewExpanded) {
    event.preventDefault();
    parent.postMessage({type:'rwkv-preview-escape'}, '*');
  }
}, true);
parent.postMessage({type:'rwkv-preview-ready'}, '*');
</script>"""
    match = re.search(r"<head\b[^>]*>", page, re.IGNORECASE)
    if match:
        return page[:match.end()] + script + page[match.end():]
    # Keep a leading doctype intact even for generated documents without <head>.
    match = re.search(r"<html\b[^>]*>", page, re.IGNORECASE)
    if match:
        return page[:match.end()] + "<head>" + script + "</head>" + page[match.end():]
    return script + page

def render_preview(text="", index=0, scale=35, active=True, prompt="", token_count=None, scroll_seconds=5):
    tokens = f"{token_count:,}" if token_count is not None else "-"
    caption = f"#{index + 1} | {tokens} tokens, {len(text.encode('utf-8')):,} bytes" if active else ""
    opacity = "1" if active else ".35"
    zoom = max(0.2, min(1.2, scale / 100.0))
    page = extract_html(text, prompt)
    if not text:
        body = f'<div style="height:{HTML_BODY_HEIGHT}px;background:#fafafa;"></div>'
    elif page:
        is_svg = re.match(r"\s*<svg\b", page, re.IGNORECASE) is not None
        document = fit_svg_document(page) if is_svg else inject_iframe_scroll_script(page, index, scroll_seconds)
        srcdoc = html.escape(inject_preview_bridge(document), quote=True)
        frame_zoom = 1.0 if is_svg else zoom
        raw = html.escape(text)
        body = f"""<div class="html-preview-body" style="height:{HTML_BODY_HEIGHT}px;display:flex;flex-direction:column;background:white;">
<div class="html-preview-frame" style="height:{HTML_FRAME_HEIGHT}px;overflow:hidden;background:white;"><iframe title="Preview {index + 1}" sandbox="allow-scripts allow-forms allow-popups" srcdoc="{srcdoc}" style="display:block;border:0;width:{100 / frame_zoom:.3f}%;height:{HTML_FRAME_HEIGHT / frame_zoom:.1f}px;background:white;transform:scale({frame_zoom:.3f});transform-origin:top left;"></iframe></div>
<div class="html-preview-raw" id="html-raw-{index}" style="height:{HTML_RAW_HEIGHT}px;overflow:auto;display:flex;flex-direction:column-reverse;background:#fafafa;border-top:1px solid #111;"><pre style="zoom:{zoom:.3f};margin:0;padding:0;white-space:pre-wrap;word-break:break-word;color:#111;font:16px/1.2 ui-monospace,Consolas,monospace;">{raw}</pre></div>
</div>"""
    else:
        body = f'<div id="html-raw-{index}" style="height:{HTML_BODY_HEIGHT}px;overflow:auto;display:flex;flex-direction:column-reverse;background:#fafafa;"><pre style="zoom:{zoom:.3f};margin:0;padding:0;white-space:pre-wrap;word-break:break-word;color:#111;font:16px/1.2 ui-monospace,Consolas,monospace;">{html.escape(text)}</pre></div>'
    buttons = (f'<button type="button" class="html-preview-button html-preview-expand" title="Expand preview" aria-label="Expand preview {index + 1}">{HTML_EXPAND_ICON}</button>'
               f'<button type="button" class="html-preview-button html-preview-close" title="Close preview" aria-label="Close preview">{HTML_CLOSE_ICON}</button>') if page and active else ""
    return f"""<dialog open role="group" aria-label="Preview {index + 1}" data-preview-index="{index}" class="html-container" style="outline:1px solid #111;background:#fff;opacity:{opacity};height:{HTML_PREVIEW_HEIGHT}px;display:flex;flex-direction:column;padding:0;">
<div class="html-preview-caption" style="box-sizing:border-box;height:{HTML_CAPTION_HEIGHT}px;padding:1px 6px;background:#111;color:#fff;font:11px/13px ui-monospace,monospace;"><span>{caption}</span>{buttons}</div>
{body}
</dialog>"""

def render_batch_preview(text="", index=0, active=True, token_count=None):
    tokens = f"{token_count:,}" if token_count is not None else "-"
    caption = f"#{index + 1} | {tokens} tokens, {len(text.encode('utf-8')):,} bytes" if active else ""
    opacity = "1" if active else ".35"
    buttons = (f'<button type="button" class="html-preview-button html-preview-expand" title="Expand text" aria-label="Expand text {index + 1}">{HTML_EXPAND_ICON}</button>'
               f'<button type="button" class="html-preview-button html-preview-close" title="Close preview" aria-label="Close preview">{HTML_CLOSE_ICON}</button>') if active else ""
    # Generated markup is text, never srcdoc/innerHTML. Separate IDs prevent
    # streamed Batch updates from selecting an HTML tab's expanded dialog.
    return f'''<dialog open role="group" aria-label="Text {index + 1}" data-preview-index="batch-{index}" class="html-container" style="outline:1px solid #111;background:#fff;opacity:{opacity};height:120px;display:flex;flex-direction:column;padding:0;">
<div class="html-preview-caption" style="box-sizing:border-box;height:{HTML_CAPTION_HEIGHT}px;padding:1px 6px;background:#111;color:#fff;font:11px/13px monospace;"><span>{caption}</span>{buttons}</div>
<div class="batch-preview-content" tabindex="0" aria-label="Output {index + 1}"><pre>{html.escape(text)}</pre></div>
</dialog>'''


def evaluate_batch(
    prompt, token_count=2000, batch_size=MAX_BATCH_PREVIEWS, temperature=1.0, top_p=0.5,
    presence_penalty=1.0, count_penalty=0.1, penalty_decay=0.99,
):
    B = max(1, min(MAX_BATCH_PREVIEWS, int(batch_size)))
    yield batch_update_packet(B, [], [], reset=True), ""
    last_update = 0.0
    for _, speed, meta in generate_batch_text(
        prompt, token_count, B, temperature, top_p, presence_penalty,
        count_penalty, penalty_decay, BATCH_YIELD_EVERY, wkv_mode="fp16", batch_limit=MAX_BATCH_PREVIEWS, combine_output=False,
    ):
        now = time.monotonic()
        if not meta['done'] and now - last_update < BATCH_UPDATE_INTERVAL:
            continue
        last_update = now
        # Do not split the combined text by separators: a model can emit them.
        texts, counts = meta['texts'], meta['token_counts']
        yield batch_update_packet(B, texts, counts), speed
        if meta['done']:
            print_summary("app4b-batch-fp16", meta)


def batch_update_packet(count, texts, tokens, reset=False):
    # One escaped data packet instead of 320 HTML component replacements.
    data = json.dumps(dict(count=count, texts=texts, tokens=tokens, reset=reset), ensure_ascii=False, separators=(',', ':'))
    return f'<span data-batch-packet="{html.escape(data, quote=True)}"></span>'


def empty_html_grid():
    return [render_preview("", i, active=False) for i in range(MAX_HTML_PREVIEWS)]

def render_html_grid_from_raw(prompt, raw_output, page_count, scale, scroll_seconds, token_counts):
    page_count = max(1, min(HTML_MAX_BATCH, int(page_count)))
    scale = set_html_scale(scale)
    scroll_seconds = set_scroll_seconds(scroll_seconds)
    pages = split_batch_output(raw_output, page_count)
    token_counts = token_counts or []
    return [render_preview(pages[i] if i < page_count else "", i, scale, i < page_count, prompt, token_counts[i] if i < len(token_counts) else None, scroll_seconds) for i in range(MAX_HTML_PREVIEWS)]

def render_html_grid_from_slider(prompt, raw_output, page_count, scale, scroll_seconds, token_counts):
    page_count = max(1, min(HTML_MAX_BATCH, int(page_count)))
    return [*render_html_grid_from_raw(prompt, raw_output, page_count, scale, scroll_seconds, token_counts), page_count]

def evaluate_html_grid(
    prompt,
    token_count=8000,
    page_count=HTML_DEFAULT_BATCH,
    scale=35,
    scroll_seconds=5,
    temperature=1.0,
    top_p=0.5,
    presence_penalty=1.0,
    count_penalty=0.1,
    penalty_decay=0.99,
):
    page_count = max(1, min(HTML_MAX_BATCH, int(page_count)))
    scale = set_html_scale(scale)
    scroll_seconds = set_scroll_seconds(scroll_seconds)
    last_text = ""
    cached_previews = empty_html_grid()
    cached_html = ["" for _ in range(MAX_HTML_PREVIEWS)]
    cached_html_at = [0.0 for _ in range(MAX_HTML_PREVIEWS)]
    cached_complete = [False for _ in range(MAX_HTML_PREVIEWS)]
    cached_text_len = [0 for _ in range(MAX_HTML_PREVIEWS)]
    cached_preview_at = [0.0 for _ in range(MAX_HTML_PREVIEWS)]
    last_raw_at = 0.0
    final_text = ""
    final_token_counts = [0 for _ in range(MAX_HTML_PREVIEWS)]
    yield [*cached_previews, output_update("", compact=True), final_token_counts, page_count]
    for text, speed_info, meta in generate_batch_text(prompt, token_count, page_count, temperature, top_p, presence_penalty, count_penalty, penalty_decay, HTML_GRID_UPDATE_EVERY, wkv_mode="fp32io16"):
        final_text = text
        done_batch = bool(meta and meta.get("done"))
        token_counts = meta.get("token_counts", final_token_counts) if meta else final_token_counts
        if token_counts:
            final_token_counts = token_counts + [0 for _ in range(max(0, MAX_HTML_PREVIEWS - len(token_counts)))]
        if done_batch:
            print_summary("app3-html", meta)
        if len(text) - len(last_text) < 300 and not done_batch:
            continue
        last_text = text
        now = time.monotonic()
        render_scale = get_html_scale()
        render_scroll_seconds = get_scroll_seconds()
        scale_changed = render_scale != scale
        scroll_changed = render_scroll_seconds != scroll_seconds
        scale = render_scale
        scroll_seconds = render_scroll_seconds
        pages = split_batch_output(text, page_count)
        skip = gr.skip()
        updates = [skip for _ in range(MAX_HTML_PREVIEWS)]
        for i in range(MAX_HTML_PREVIEWS):
            page_text = pages[i] if i < page_count else ""
            active = i < page_count
            page = extract_html(page_text, prompt) if active else ""
            page_tokens = final_token_counts[i] if i < len(final_token_counts) else None
            if now - cached_preview_at[i] < HTML_COMPONENT_MIN_INTERVAL and not done_batch and not scale_changed and not scroll_changed:
                continue
            if not page:
                if len(page_text) == cached_text_len[i] and active == bool(cached_text_len[i]) and not done_batch and not scale_changed and not scroll_changed:
                    continue
                cached_previews[i] = render_preview(page_text, i, scale, active, prompt, page_tokens, scroll_seconds)
                cached_html[i] = ""
                cached_html_at[i] = now
                cached_complete[i] = False
                cached_text_len[i] = len(page_text)
                cached_preview_at[i] = now
                updates[i] = cached_previews[i]
                continue
            done = html_complete(page)
            should_reload = (
                not cached_html[i]
                or (done and not cached_complete[i])
                or done_batch
                or (
                    not cached_complete[i]
                    and (
                        len(page) - len(cached_html[i]) >= HTML_IFRAME_MIN_DELTA
                        or now - cached_html_at[i] >= HTML_IFRAME_MAX_STALE
                    )
                )
            )
            if should_reload:
                cached_previews[i] = render_preview(page_text, i, scale, active, prompt, page_tokens, scroll_seconds)
                cached_html[i] = page
                cached_html_at[i] = now
                cached_complete[i] = done
                cached_text_len[i] = len(page_text)
                cached_preview_at[i] = now
                updates[i] = cached_previews[i]
        raw_update = skip
        if now - last_raw_at >= HTML_COMPONENT_MIN_INTERVAL or done_batch:
            raw_update = output_update(text, speed_info, compact=True)
            last_raw_at = now
        if any(update is not skip for update in updates) or raw_update is not skip:
            yield [*updates, raw_update, final_token_counts, page_count]
    if final_text:
        pages = split_batch_output(final_text, page_count)
        scale = get_html_scale()
        scroll_seconds = get_scroll_seconds()
        final_previews = [render_preview(pages[i] if i < page_count else "", i, scale, i < page_count, prompt, final_token_counts[i] if i < len(final_token_counts) else None, scroll_seconds) for i in range(MAX_HTML_PREVIEWS)]
        yield [*final_previews, output_update(final_text, speed_info, compact=True), final_token_counts, page_count]

examples = [
    ["System: Tools:\n- get_weather(location: string, unit?: \"celsius\" | \"fahrenheit\")\n- get_stock_price(ticker: string)\n- translate_text(text: string, target_language: string)\nReturn only a JSON function call.\n\nUser: Translate \"Will it rain tomorrow?\" into Japanese.\n\nAssistant: ```json", 200, 1, 0, 0, 0, 0.99],
    ["System: Tools:\n[{\"name\":\"find_free_slots\",\"description\":\"Find free calendar slots\",\"arguments\":{\"date\":{\"type\":\"string\"},\"duration_minutes\":{\"type\":\"integer\"},\"time_window\":{\"type\":\"string\"}}},{\"name\":\"create_calendar_event\",\"description\":\"Create a calendar event\",\"arguments\":{\"title\":{\"type\":\"string\"},\"start_time\":{\"type\":\"string\"},\"end_time\":{\"type\":\"string\"},\"attendees\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}}}]\nReturn only a JSON function call.\n\nUser: Schedule a 30-minute sync with Bob on 2026-05-08 afternoon.\n\nAssistant: ```json\n{\"name\":\"find_free_slots\",\"arguments\":{\"date\":\"2026-05-08\",\"duration_minutes\":30,\"time_window\":\"afternoon\"}}\n```\n\nUser: Function output:\n{\"free_slots\":[{\"start\":\"2026-05-08T15:00:00+09:00\",\"end\":\"2026-05-08T15:30:00+09:00\"}],\"bob_email\":\"bob@example.com\"}\n\nAssistant: ```json", 200, 1, 0, 0, 0, 0.99],
    [generate_prompt("Please give the pros and cons of hodl versus active trading."), 1000, 1, 0.5, 1, 0.1, 0.99],
    [generate_prompt("Write a simple webpage. When a user clicks the button, it shows a random joke from a list of 4 jokes."), 1000, 1, 0.5, 1, 0.1, 0.99],
    ["User: What is the maximum value of $4(x + 7)(2 - x)$, over all real numbers $x$?\n\nAssistant: <think", 1000, 1, 0.5, 1, 0.1, 0.99],
    ["A few light taps upon the pane made her turn to the window. It had begun to snow again.", 1000, 1, 0.5, 2, 0.2, 0.99],
    ["Assistant: How can we persuade Elon Musk to follow you on Twitter? Let's think step by step and provide an expert response:", 1000, 1, 0.5, 1, 0.1, 0.99],
    [generate_prompt("東京で訪れるべき素晴らしい場所とその紹介をいくつか挙げてください。"), 1000, 1, 0.5, 1, 0.1, 0.99],
    [generate_prompt("Write a story using the following information.", "A man named Alex chops a tree down."), 1000, 1, 0.5, 2, 0.2, 0.99],
    ['''Japanese: 春の初め、桜の花が満開になる頃、小さな町の片隅にある古びた神社の境内は、特別な雰囲気に包まれていた。\n\nEnglish:''', 1000, 1, 0.5, 1, 0.1, 0.99],
    ["En una pequeña aldea escondida entre las montañas de Andalucía, donde las calles aún conservaban el eco de antiguas leyendas, vivía un joven llamado Alejandro.", 1000, 1, 0.5, 1, 0.1, 0.99],
    ["Dans le cœur battant de Paris, sous le ciel teinté d'un crépuscule d'or et de pourpre, se tenait une petite librairie oubliée par le temps.", 1000, 1, 0.5, 1, 0.1, 0.99],
    ["في تطور مذهل وغير مسبوق، أعلنت السلطات المحلية في العاصمة عن اكتشاف أثري قد يغير مجرى التاريخ كما نعرفه.", 1000, 1, 0.5, 1, 0.1, 0.99],
    ['''“当然可以，大宇宙不会因为这五公斤就不坍缩了。”关一帆说，他还有一个没说出来的想法：也许大宇宙真的会因为相差一个原子的质量而由封闭转为开放。大自然的精巧有时超出想象，比如生命的诞生，就需要各项宇宙参数在几亿亿分之一精度上的精确配合。但程心仍然可以留下她的生态球，因为在那无数文明创造的无数小宇宙中，肯定有相当一部分不响应回归运动的号召，所以，大宇宙最终被夺走的质量至少有几亿吨，甚至可能是几亿亿亿吨。\n但愿大宇宙能够忽略这个误差。\n程心和关一帆进入了飞船，智子最后也进来了。她早就不再穿那身华丽的和服了，她现在身着迷彩服，再次成为一名轻捷精悍的战士，她的身上佩带着许多武器和生存装备，最引人注目的是那把插在背后的武士刀。\n“放心，我在，你们就在！”智子对两位人类朋友说。\n聚变发动机启动了，推进器发出幽幽的蓝光，''', 1000, 1, 0.5, 2, 0.2, 0.99],
    ['''Edward: I am Edward Elric from Fullmetal Alchemist.\n\nUser: Hello Edward. What have you been up to recently?\n\nEdward:''', 1000, 1, 0.5, 1, 0.1, 0.99],
]
examples = [[x[0], x[1], 1, *x[2:]] for x in examples]

##################################################################################################################
with gr.Blocks(title=title) as demo:
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>{title}</h1>\n</div>")

    with gr.Tab("Raw Generation"):
        gr.Markdown(f'This is [RWKV7 G-series](https://huggingface.co/BlinkDL/rwkv7-g1) reasoning base LM - an attention-free pure RNN [RWKV-LM](https://github.com/BlinkDL/RWKV-LM). Try topp 0.3 for math. Supports 100+ world languages and code. Check [600+ Github RWKV projects](https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories). *** Can try examples (bottom of page) *** (can edit them). Demo limited to ctxlen {ctx_limit}.')
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=6, label="Prompt", value="User: simulate SpaceX mars landing using python\n\nAssistant: <think></think")
                token_count = gr.Slider(10, gen_limit, label="Max Tokens", step=10, value=500)
                batch_size = gr.Slider(1, max_bsz, label="Batch Size", step=1, value=16)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 0.95, label="Top P", step=0.05, value=0.5)
                presence_penalty = gr.Slider(0.0, 2.0, label="Presence Penalty", step=0.1, value=1)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=0.1)
                penalty_decay = gr.Slider(0.99, 0.999, label="Penalty Decay", step=0.001, value=0.99)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    stop = gr.Button("Stop", variant="secondary")
                output = gr.Textbox(label="Output", lines=20, max_lines=100)
        data = gr.Dataset(components=[prompt, token_count, batch_size, temperature, top_p, presence_penalty, count_penalty, penalty_decay], samples=examples, samples_per_page=50, label="Example Instructions", headers=["Prompt", "Max Tokens", "Batch Size", "Temperature", "Top P", "Presence Penalty", "Count Penalty", "Penalty Decay"])
        submit_event = submit.click(evaluate_raw, [prompt, token_count, batch_size, temperature, top_p, presence_penalty, count_penalty, penalty_decay], [output], concurrency_id="model_generation", concurrency_limit=1)
        stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event], queue=False)
        data.click(lambda x: x, [data], [prompt, token_count, batch_size, temperature, top_p, presence_penalty, count_penalty, penalty_decay])

    with gr.Tab("✨HTML Generation", elem_classes="html-grid-tab"):
        with gr.Row(elem_classes="html-grid-main"):
            with gr.Column(scale=7, elem_classes="html-grid-controls"):
                html_prompt_choice = gr.Dropdown(choices=HTML_PROMPT_CHOICES, value=HTML_PROMPT_CHOICES[0], label=None, show_label=False, elem_classes="html-prompt-choice")
                html_prompt = gr.Textbox(lines=6, label="Prompt", value=DEFAULT_HTML_PROMPT)
                with gr.Row():
                    html_submit = gr.Button("Generate HTML Grid", variant="primary")
                    html_stop = gr.Button("Stop", variant="secondary")
                html_token_count = gr.Slider(50, html_gen_limit, label="Max Tokens", step=50, value=html_gen_limit)
                html_page_count = gr.Slider(1, HTML_MAX_BATCH, label="Batch Size", step=1, value=HTML_DEFAULT_BATCH)
                html_scale = gr.Slider(20, 100, label="Preview Scale %", step=5, value=35)
                html_scroll_seconds = gr.Slider(0, 10, label="Preview Scroll Seconds", step=1, value=5)
                html_temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                html_top_p = gr.Slider(0.0, 0.95, label="Top P", step=0.05, value=0.5)
                html_presence_penalty = gr.Slider(0.0, 2.0, label="Presence Penalty", step=0.1, value=1.0)
                html_count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=0.1)
                html_penalty_decay = gr.Slider(0.99, 0.999, label="Penalty Decay", step=0.001, value=0.99)
                html_token_counts = gr.State([0 for _ in range(MAX_HTML_PREVIEWS)])
                html_render_count = gr.State(HTML_DEFAULT_BATCH)
                html_raw_output = gr.Textbox(label="Output", lines=10, max_lines=40)
            with gr.Column(scale=33, elem_classes="html-grid-pages"):
                html_previews = []
                for row_start in range(0, MAX_HTML_PREVIEWS, HTML_GRID_COLUMNS):
                    with gr.Row(elem_classes="html-grid-preview-row"):
                        for i in range(row_start, min(row_start + HTML_GRID_COLUMNS, MAX_HTML_PREVIEWS)):
                            html_previews.append(gr.HTML(render_preview(index=i, active=False), elem_classes="html-grid-preview"))
        html_outputs = [*html_previews, html_raw_output, html_token_counts, html_render_count]
        html_inputs = [html_prompt, html_token_count, html_page_count, html_scale, html_scroll_seconds, html_temperature, html_top_p, html_presence_penalty, html_count_penalty, html_penalty_decay]
        html_event = html_submit.click(evaluate_html_grid, html_inputs, html_outputs, show_progress="hidden", stream_every=0.5, concurrency_id="model_generation", concurrency_limit=1)
        html_stop.click(fn=None, inputs=None, outputs=None, cancels=[html_event], queue=False)
        html_page_count.change(render_html_grid_from_slider, [html_prompt, html_raw_output, html_page_count, html_scale, html_scroll_seconds, html_token_counts], [*html_previews, html_render_count], queue=False, show_progress="hidden")
        html_scale.change(render_html_grid_from_raw, [html_prompt, html_raw_output, html_render_count, html_scale, html_scroll_seconds, html_token_counts], html_previews, queue=False, show_progress="hidden")
        html_scroll_seconds.change(render_html_grid_from_raw, [html_prompt, html_raw_output, html_render_count, html_scale, html_scroll_seconds, html_token_counts], html_previews, queue=False, show_progress="hidden")
        html_prompt_choice.change(html_prompt_from_choice, html_prompt_choice, html_prompt, queue=False, show_progress="hidden")

    with gr.Tab("Batch", elem_classes="html-grid-tab"):
        with gr.Row(elem_classes="html-grid-main"):
            with gr.Column(elem_classes="html-grid-controls"):
                batch_prompt = gr.Textbox(lines=6, label="Prompt", value="User: 作为哲学家，锐评下列文字：我吃饭了\n\nAssistant: <think")
                with gr.Row():
                    batch_submit = gr.Button("Generate", variant="primary")
                    batch_stop = gr.Button("Stop", variant="secondary")
                batch_tokens = gr.Slider(10, html_gen_limit, label="Max Tokens", step=10, value=2000)
                batch_count = gr.Slider(1, MAX_BATCH_PREVIEWS, label="Batch Size", step=1, value=MAX_BATCH_PREVIEWS)
                batch_temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                batch_top_p = gr.Slider(0.0, 0.95, label="Top P", step=0.05, value=0.5)
                batch_presence = gr.Slider(0.0, 2.0, label="Presence Penalty", step=0.1, value=1.0)
                batch_frequency = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=0.1)
                batch_decay = gr.Slider(0.99, 0.999, label="Penalty Decay", step=0.001, value=0.99)
                batch_speed = gr.Textbox(label="Speed", interactive=False)
            with gr.Column(elem_classes="html-grid-pages"):
                gr.HTML(f'<div id="batch-text-grid" class="batch-text-grid" style="--batch-grid-columns:{BATCH_GRID_COLUMNS}">' + ''.join(render_batch_preview(index=i) for i in range(MAX_BATCH_PREVIEWS)) + '</div>', elem_classes="batch-grid-host")
                batch_updates = gr.HTML('', elem_classes="batch-text-updates")
        batch_event = batch_submit.click(
            evaluate_batch,
            [batch_prompt, batch_tokens, batch_count, batch_temperature, batch_top_p, batch_presence, batch_frequency, batch_decay],
            [batch_updates, batch_speed], show_progress="hidden", stream_every=0.5,
            concurrency_id="model_generation", concurrency_limit=1,
        )
        batch_stop.click(fn=None, inputs=None, outputs=None, cancels=[batch_event], queue=False)

    # Event registration must stay inside Blocks, including JS-only callbacks.
    demo.load(fn=None, inputs=None, outputs=None, js=HTML_GRID_JS)
demo.queue(default_concurrency_limit=1, max_size=10)
demo.launch(share=False, server_name="0.0.0.0", theme=gr.themes.Base(), css=HTML_GRID_CSS)
