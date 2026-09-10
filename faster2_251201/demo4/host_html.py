import html
import json

from constants import PROMPT_PRESETS


def build_host_html(
    cols,
    rows,
    page_scale,
    allow_scripts,
    group_count,
    initial_prompt,
    auto_bounce_scroll=False,
    auto_bounce_scroll_step=20,
    auto_bounce_scroll_tick_ms=16,
    auto_bounce_scroll_edge_pause_ms=350,
):
    frame_count = cols * rows
    sandbox = "allow-same-origin allow-scripts" if allow_scripts else "allow-same-origin"
    cells = []
    for idx in range(frame_count):
        cells.append(
            f"""
      <section class="cell" id="cell-{idx}">
        <div class="viewport">
          <iframe class="page-frame active" id="frame-{idx}-0" sandbox="{sandbox}" scrolling="yes"></iframe>
          <iframe class="page-frame hidden" id="frame-{idx}-1" sandbox="{sandbox}" scrolling="yes"></iframe>
        </div>
        <div class="caption" id="caption-{idx}">#{idx + 1} | pending | 0 bytes</div>
      </section>"""
        )
    cells_html = "\n".join(cells)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    :root {{ --page-scale: {page_scale}; }}
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #000; }}
    body {{ font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #grid {{
      box-sizing: border-box; height: 100vh; padding: 1px; background: #000;
      display: grid; grid-template-columns: repeat({cols}, 1fr); grid-template-rows: repeat({rows}, 1fr);
      gap: 1px;
    }}
    .cell {{ position: relative; min-width: 0; min-height: 0; background: #11151d; border: 0; overflow: hidden; }}
    .viewport {{ position: absolute; inset: 0; overflow: hidden; background: #f8f6ef; }}
    .cell.blanked .viewport {{ background: #fff; }}
    .cell.blanked .page-frame {{ visibility: hidden !important; opacity: 0 !important; pointer-events: none; }}
    .page-frame {{
      position: absolute; left: 0; top: 0;
      width: calc(100% / var(--page-scale)); height: calc(100% / var(--page-scale));
      transform: scale(var(--page-scale)); transform-origin: top left; border: 0; background: white;
      overflow: auto; scrollbar-gutter: stable;
    }}
    .page-frame.active {{ visibility: visible; opacity: 1; }}
    .page-frame.hidden {{ visibility: hidden; opacity: 0; pointer-events: none; }}
    .caption {{
      position: absolute; left: 0; right: 0; bottom: 0; box-sizing: border-box;
      padding: 3px 6px; color: #d6d9df; background: rgba(0,0,0,.62);
      font: 11px/15px ui-monospace, monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      pointer-events: none;
    }}
    .cell:hover .caption {{ white-space: normal; max-height: 46%; overflow: auto; pointer-events: auto; }}
    #pager {{
      position: fixed; right: 12px; top: 12px; z-index: 20; display: flex; align-items: center; gap: 7px;
      padding: 5px 7px; color: #f2f2f2; background: rgba(0,0,0,.58); border: 1px solid rgba(255,255,255,.28);
      border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,.24);
      font: 18px/24px ui-monospace, monospace; pointer-events: auto;
    }}
    #pager button {{
      display: grid; place-items: center; text-decoration: none;
      width: 54px; height: 33px; padding: 0; color: #fff; background: rgba(255,255,255,.14);
      border: 1px solid rgba(255,255,255,.30); border-radius: 8px; cursor: pointer;
      font: 700 21px/27px ui-monospace, monospace;
    }}
    #pager button:hover {{ background: rgba(255,255,255,.22); }}
    #pager button:disabled {{ opacity: .55; cursor: default; }}
    #groupLabel {{ min-width: 42px; text-align: center; }}
    #bounceButton {{
      position: fixed; right: 12px; top: 58px; z-index: 20;
      min-width: 108px; height: 40px; padding: 0 14px;
      color: #fff; background: rgba(0,0,0,.58);
      border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
      box-shadow: 0 8px 24px rgba(0,0,0,.24);
      font: 700 13px/13px ui-monospace, monospace; cursor: pointer;
      pointer-events: auto;
    }}
    #bounceButton.is-off {{ background: rgba(60,60,60,.72); color: #d7d7d7; }}
    #bounceButton:hover {{ background: rgba(255,255,255,.18); }}
    #bounceButton:disabled {{ opacity: .55; cursor: default; }}
    #restartButton {{
      position: fixed; right: 12px; bottom: 12px; z-index: 20;
      display: grid; place-items: center;
      width: 48px; height: 48px; padding: 0;
      color: #fff; background: rgba(0,0,0,.58);
      border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
      box-shadow: 0 8px 24px rgba(0,0,0,.24);
      font: 700 25px/25px ui-monospace, monospace; cursor: pointer;
      pointer-events: auto;
    }}
    #restartButton:hover {{ background: rgba(255,255,255,.18); }}
    #restartButton:disabled {{ opacity: .55; cursor: default; }}
    #promptButton {{
      position: fixed; left: 12px; top: 12px; z-index: 20;
      display: grid; place-items: center;
      width: 48px; height: 48px; padding: 0;
      color: #fff; background: rgba(0,0,0,.58);
      border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
      box-shadow: 0 8px 24px rgba(0,0,0,.24);
      font: 700 22px/22px ui-monospace, monospace; cursor: pointer;
      pointer-events: auto;
    }}
    #promptButton:hover {{ background: rgba(255,255,255,.18); }}
    #promptButton:disabled {{ opacity: .55; cursor: default; }}
    #promptPresetButton {{
      position: fixed; left: 12px; top: 68px; z-index: 20;
      display: grid; place-items: center;
      width: 48px; height: 48px; padding: 0;
      color: #fff; background: rgba(0,0,0,.58);
      border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
      box-shadow: 0 8px 24px rgba(0,0,0,.24);
      font: 700 22px/22px ui-monospace, monospace; cursor: pointer;
      pointer-events: auto;
    }}
    #promptPresetButton:hover {{ background: rgba(255,255,255,.18); }}
    #promptPresetButton:disabled {{ opacity: .55; cursor: default; }}
    #promptPresetOverlay {{
      position: fixed; inset: 0; z-index: 30; display: none;
      align-items: center; justify-content: center;
      background: rgba(0,0,0,.38); pointer-events: auto;
    }}
    #promptPresetOverlay.open {{ display: flex; }}
    #promptPresetDialog {{
      box-sizing: border-box; width: min(1100px, calc(100vw - 96px)); height: min(720px, calc(100vh - 96px));
      display: grid; grid-template-rows: auto 1fr auto; gap: 12px;
      padding: 16px; color: #f4f4f4; background: rgba(18,20,24,.94);
      border: 1px solid rgba(255,255,255,.24); border-radius: 12px;
      box-shadow: 0 18px 50px rgba(0,0,0,.40);
      font: 14px/20px ui-monospace, monospace;
    }}
    #promptPresetDialog h2 {{ margin: 0; font: 700 18px/24px ui-monospace, monospace; }}
    #promptPresetItems {{
      display: grid; gap: 10px; overflow: auto; padding-right: 12px; scrollbar-gutter: stable;
    }}
    #promptPresetItems button {{
      box-sizing: border-box; width: 100%; height: 40px; padding: 0 12px;
      color: #111; background: #fff; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      border: 1px solid rgba(255,255,255,.35); border-radius: 8px;
      font: 14px/20px ui-monospace, monospace; cursor: pointer;
    }}
    #promptPresetItems button:hover {{ outline: 2px solid #ffcf70; }}
    #promptPresetPreview {{
      position: fixed; left: 0; top: 0; z-index: 31; display: none;
      box-sizing: border-box; width: min(720px, calc(100vw - 36px)); overflow: visible;
      padding: 12px 14px; color: #111; background: #fff;
      border: 1px solid rgba(255,255,255,.35); border-radius: 8px;
      box-shadow: 0 18px 50px rgba(0,0,0,.40);
      font: 14px/20px ui-monospace, monospace; white-space: pre-wrap; overflow-wrap: anywhere;
      pointer-events: none;
    }}
    #promptPresetPreview.open {{ display: block; }}
    #promptOverlay {{
      position: fixed; inset: 0; z-index: 30; display: none;
      align-items: center; justify-content: center;
      background: rgba(0,0,0,.38); pointer-events: auto;
    }}
    #promptOverlay.open {{ display: flex; }}
    #promptDialog {{
      box-sizing: border-box; width: min(900px, calc(100vw - 64px)); height: min(560px, calc(100vh - 64px));
      display: grid; grid-template-rows: auto 1fr auto; gap: 12px;
      padding: 16px; color: #f4f4f4; background: rgba(18,20,24,.94);
      border: 1px solid rgba(255,255,255,.24); border-radius: 12px;
      box-shadow: 0 18px 50px rgba(0,0,0,.40);
      font: 14px/20px ui-monospace, monospace;
    }}
    #promptDialog h2 {{ margin: 0; font: 700 18px/24px ui-monospace, monospace; }}
    #promptText {{
      box-sizing: border-box; width: 100%; height: 100%; resize: none;
      padding: 12px; color: #111; background: #fff;
      border: 1px solid rgba(255,255,255,.35); border-radius: 8px;
      font: 14px/20px ui-monospace, monospace;
    }}
    #promptActions, #promptPresetActions {{ display: flex; justify-content: flex-end; gap: 10px; }}
    #promptActions button, #promptPresetActions button {{
      min-width: 88px; height: 36px; padding: 0 14px;
      color: #fff; background: rgba(255,255,255,.14);
      border: 1px solid rgba(255,255,255,.30); border-radius: 8px;
      font: 700 14px/20px ui-monospace, monospace; cursor: pointer;
    }}
    #promptActions button:hover, #promptPresetActions button:hover {{ background: rgba(255,255,255,.22); }}
  </style>
</head>
<body>
  <main id="grid">
    {cells_html}
  </main>
  <nav id="pager">
    <button id="pagerPrev" type="button" onclick="changeGroup('prev')" disabled>&lt;</button>
    <span id="groupLabel">1/{group_count}</span>
    <button id="pagerNext" type="button" onclick="changeGroup('next')" disabled>&gt;</button>
  </nav>
  <button id="bounceButton" type="button" onclick="toggleBounceScroll()" title="Toggle auto bounce scroll" aria-label="Toggle auto bounce scroll" disabled>Scroll: On</button>
  <button id="promptButton" type="button" onclick="openPromptDialog()" title="Edit prompt" aria-label="Edit prompt" disabled>{html.escape("✏️")}</button>
  <button id="promptPresetButton" type="button" onclick="openPromptPresets()" title="Prompt presets" aria-label="Prompt presets" disabled>{html.escape("☰")}</button>
  <button id="restartButton" type="button" onclick="restartGeneration()" title="Restart" aria-label="Restart" disabled>{html.escape("🔄")}</button>
  <div id="promptPresetOverlay">
    <section id="promptPresetDialog" role="dialog" aria-modal="true" aria-label="Prompt presets">
      <h2>Prompt Presets</h2>
      <div id="promptPresetItems"></div>
      <pre id="promptPresetPreview"></pre>
      <div id="promptPresetActions">
        <button type="button" onclick="closePromptPresets()">Cancel</button>
      </div>
    </section>
  </div>
  <div id="promptOverlay">
    <section id="promptDialog" role="dialog" aria-modal="true" aria-label="Edit prompt">
      <h2>Edit Prompt</h2>
      <textarea id="promptText" spellcheck="false"></textarea>
      <div id="promptActions">
        <button type="button" onclick="closePromptDialog()">Cancel</button>
        <button type="button" onclick="applyPromptDialog()">OK</button>
      </div>
    </section>
  </div>
  <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
  <script>
    let currentPrompt = {json.dumps(initial_prompt, ensure_ascii=False)};
    const promptPresets = {json.dumps(PROMPT_PRESETS, ensure_ascii=False)};
    let autoBounceScrollEnabled = {json.dumps(bool(auto_bounce_scroll))};
    const autoBounceScrollStep = {json.dumps(max(1, int(auto_bounce_scroll_step)))};
    const autoBounceScrollTickMs = {json.dumps(max(8, int(auto_bounce_scroll_tick_ms)))};
    const autoBounceScrollEdgePauseMs = {json.dumps(max(0, int(auto_bounce_scroll_edge_pause_ms)))};
    window.setScale = function(scale) {{
      document.documentElement.style.setProperty('--page-scale', String(scale));
    }};
    let demo4Bridge = null;
    function enablePager(enabled) {{
      const prev = document.getElementById('pagerPrev');
      const next = document.getElementById('pagerNext');
      if (prev) prev.disabled = !enabled;
      if (next) next.disabled = !enabled;
      const restart = document.getElementById('restartButton');
      if (restart) restart.disabled = !enabled;
      const prompt = document.getElementById('promptButton');
      if (prompt) prompt.disabled = !enabled;
      const presets = document.getElementById('promptPresetButton');
      if (presets) presets.disabled = !enabled;
      const bounce = document.getElementById('bounceButton');
      if (bounce) bounce.disabled = !enabled;
    }}
    function updateBounceButton() {{
      const bounce = document.getElementById('bounceButton');
      if (!bounce) return;
      bounce.textContent = autoBounceScrollEnabled ? 'Scroll: On' : 'Scroll: Off';
      bounce.classList.toggle('is-off', !autoBounceScrollEnabled);
    }}
    function restartVisibleBounceScroll() {{
      for (let index = 0; index < pageState.length; index++) {{
        const state = pageState[index];
        if (!state) continue;
        if (!autoBounceScrollEnabled) {{
          stopFrameAutoBounce(index);
          continue;
        }}
        const activeFrame = frameFor(index, state.active);
        if (!activeFrame || !state.currentHtml) continue;
        startFrameAutoBounce(index, activeFrame, state.loadToken);
      }}
    }}
    function toggleBounceScroll(forceEnabled) {{
      autoBounceScrollEnabled = typeof forceEnabled === 'boolean' ? forceEnabled : !autoBounceScrollEnabled;
      updateBounceButton();
      restartVisibleBounceScroll();
    }}
    function changeGroup(action) {{
      if (demo4Bridge) demo4Bridge.handleGroup(action);
    }}
    function restartGeneration() {{
      if (demo4Bridge) demo4Bridge.handleRestart();
    }}
    function openPromptPresets() {{
      const overlay = document.getElementById('promptPresetOverlay');
      if (!overlay) return;
      overlay.classList.add('open');
      if (demo4Bridge) demo4Bridge.handlePromptEditor(true);
    }}
    function closePromptPresets(notify) {{
      const overlay = document.getElementById('promptPresetOverlay');
      if (overlay) overlay.classList.remove('open');
      hidePromptPresetPreview();
      if (notify !== false && demo4Bridge) demo4Bridge.handlePromptEditor(false);
    }}
    function showPromptPresetPreview(prompt, anchor) {{
      const preview = document.getElementById('promptPresetPreview');
      if (!preview) return;
      preview.textContent = String(prompt || '');
      preview.classList.add('open');
      if (!anchor) return;
      const rect = anchor.getBoundingClientRect();
      const dialog = document.getElementById('promptPresetDialog');
      const dialogRect = dialog ? dialog.getBoundingClientRect() : rect;
      const gap = 12;
      const margin = 18;
      const rightWidth = window.innerWidth - dialogRect.right - gap - margin;
      const leftWidth = dialogRect.left - gap - margin;
      let width = Math.min(720, Math.max(280, rightWidth));
      let left = dialogRect.right + gap;
      if (rightWidth < 280 && leftWidth > rightWidth) {{
        width = Math.min(720, Math.max(280, leftWidth));
        left = dialogRect.left - gap - width;
      }}
      preview.style.width = width + 'px';
      let top = rect.top;
      const height = preview.offsetHeight;
      top = Math.max(margin, Math.min(top, window.innerHeight - height - margin));
      preview.style.left = left + 'px';
      preview.style.top = top + 'px';
    }}
    function hidePromptPresetPreview() {{
      const preview = document.getElementById('promptPresetPreview');
      if (preview) preview.classList.remove('open');
    }}
    function choosePromptPreset(index) {{
      const nextPrompt = promptPresets[index];
      if (typeof nextPrompt !== 'string') return;
      if (nextPrompt === currentPrompt) {{
        closePromptPresets(true);
        return;
      }}
      closePromptPresets(false);
      closePromptDialog(false);
      currentPrompt = nextPrompt;
      if (demo4Bridge) demo4Bridge.handlePrompt(nextPrompt);
    }}
    function buildPromptPresetList() {{
      const list = document.getElementById('promptPresetItems');
      if (!list) return;
      list.innerHTML = '';
      promptPresets.forEach(function(prompt, index) {{
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = prompt.replaceAll('\\n', '\\\\n');
        button.onmouseenter = function() {{ showPromptPresetPreview(prompt, button); }};
        button.onfocus = function() {{ showPromptPresetPreview(prompt, button); }};
        button.onmouseleave = hidePromptPresetPreview;
        button.onblur = hidePromptPresetPreview;
        button.onclick = function() {{ choosePromptPreset(index); }};
        list.appendChild(button);
      }});
    }}
    function openPromptDialog() {{
      const overlay = document.getElementById('promptOverlay');
      const text = document.getElementById('promptText');
      if (!overlay || !text) return;
      text.value = currentPrompt;
      overlay.classList.add('open');
      if (demo4Bridge) demo4Bridge.handlePromptEditor(true);
      setTimeout(function() {{ text.focus(); }}, 0);
    }}
    function closePromptDialog(notify) {{
      const overlay = document.getElementById('promptOverlay');
      if (overlay) overlay.classList.remove('open');
      if (notify !== false && demo4Bridge) demo4Bridge.handlePromptEditor(false);
    }}
    function applyPromptDialog() {{
      const text = document.getElementById('promptText');
      if (!text) return;
      const nextPrompt = text.value;
      if (nextPrompt !== currentPrompt) {{
        closePromptDialog(false);
        currentPrompt = nextPrompt;
        if (demo4Bridge) demo4Bridge.handlePrompt(nextPrompt);
      }} else {{
        closePromptDialog(true);
      }}
    }}
    window.setPrompt = function(prompt) {{
      currentPrompt = String(prompt || '');
      const text = document.getElementById('promptText');
      if (text && document.getElementById('promptOverlay').classList.contains('open')) {{
        text.value = currentPrompt;
      }}
    }};
    if (window.qt && window.qt.webChannelTransport) {{
      new QWebChannel(window.qt.webChannelTransport, function(channel) {{
        demo4Bridge = channel.objects.demo4Bridge;
        enablePager(true);
        updateBounceButton();
        buildPromptPresetList();
      }});
    }}
    let viewEpoch = 0;
    const pageState = Array.from({{ length: {frame_count} }}, () => ({{
      active: 0,
      loading: false,
      loadToken: 0,
      loadTimer: null,
      currentHtml: '',
      pendingItem: null,
      scrollTimer: null,
      scrollDirection: 1,
      scrollBounceTimer: null
    }}));
    function frameFor(index, slot) {{
      return document.getElementById('frame-' + index + '-' + slot);
    }}
    function captionFor(index) {{
      return document.getElementById('caption-' + index);
    }}
    function cellFor(index) {{
      return document.getElementById('cell-' + index);
    }}
    function setCaption(item) {{
      const caption = captionFor(item.index);
      if (!caption) return;
      const pageNumber = item.globalIndex == null ? item.index + 1 : item.globalIndex + 1;
      const parts = [
        '#' + pageNumber,
        item.captionStage || item.stage || 'pending',
        (item.totalBytes || 0) + ' bytes'
      ];
      caption.textContent = parts.join(' | ');
    }}
    function isTextStage(item) {{
      return item && item.autoScroll === true;
    }}
    function scrollFrameToBottom(frame) {{
      try {{
        const doc = frame.contentDocument;
        const win = frame.contentWindow;
        if (!doc || !win) return false;
        const body = doc.body;
        const root = doc.documentElement;
        const height = Math.max(
          body ? body.scrollHeight : 0,
          root ? root.scrollHeight : 0
        );
        if (body) body.scrollTop = height;
        if (root) root.scrollTop = height;
        win.scrollTo(0, height);
        return true;
      }} catch (err) {{
      }}
      return false;
    }}
    function clearBounceTimer(state) {{
      if (state.scrollBounceTimer) {{
        clearTimeout(state.scrollBounceTimer);
        state.scrollBounceTimer = null;
      }}
    }}
    function stopFrameAutoBounce(index) {{
      const state = pageState[index];
      if (!state) return;
      if (state.scrollTimer) {{
        clearInterval(state.scrollTimer);
        state.scrollTimer = null;
      }}
      clearBounceTimer(state);
      state.scrollDirection = 1;
    }}
    function readScrollMetrics(frame) {{
      try {{
        const doc = frame.contentDocument;
        const win = frame.contentWindow;
        if (!doc || !win) return null;
        const body = doc.body;
        const root = doc.documentElement;
        const top = win.scrollY || (root ? root.scrollTop : 0) || (body ? body.scrollTop : 0) || 0;
        const viewport = win.innerHeight || (root ? root.clientHeight : 0) || (body ? body.clientHeight : 0) || 0;
        const fullHeight = Math.max(
          body ? body.scrollHeight : 0,
          root ? root.scrollHeight : 0
        );
        return {{
          top: top,
          viewport: viewport,
          maxTop: Math.max(0, fullHeight - viewport)
        }};
      }} catch (err) {{
      }}
      return null;
    }}
    function queueBounceResume(index, token, nextDirection) {{
      const state = pageState[index];
      if (!state) return;
      clearBounceTimer(state);
      state.scrollDirection = 0;
      state.scrollBounceTimer = setTimeout(function() {{
        if (token !== state.loadToken) return;
        state.scrollBounceTimer = null;
        state.scrollDirection = nextDirection;
      }}, autoBounceScrollEdgePauseMs);
    }}
    function startFrameAutoBounce(index, frame, token) {{
      if (!autoBounceScrollEnabled) return;
      const state = pageState[index];
      if (!state) return;
      stopFrameAutoBounce(index);
      state.scrollDirection = 1;
      state.scrollTimer = setInterval(function() {{
        if (token !== state.loadToken) {{
          stopFrameAutoBounce(index);
          return;
        }}
        if (frameFor(index, state.active) !== frame) {{
          stopFrameAutoBounce(index);
          return;
        }}
        const metrics = readScrollMetrics(frame);
        if (!metrics) return;
        if (metrics.maxTop <= 0) return;
        if (state.scrollDirection === 0) return;
        const nextTop = Math.max(0, Math.min(metrics.maxTop, metrics.top + state.scrollDirection * autoBounceScrollStep));
        try {{
          frame.contentWindow.scrollTo(0, nextTop);
        }} catch (err) {{
          return;
        }}
        if (nextTop >= metrics.maxTop && state.scrollDirection > 0) {{
          queueBounceResume(index, token, -1);
        }} else if (nextTop <= 0 && state.scrollDirection < 0) {{
          queueBounceResume(index, token, 1);
        }}
      }}, autoBounceScrollTickMs);
    }}
    function frameReady(frame) {{
      try {{
        const doc = frame.contentDocument;
        return !!(doc && doc.readyState !== 'loading' && doc.body);
      }} catch (err) {{
        return false;
      }}
    }}
    function keepTextFrameAtBottom(frame, item, state, token) {{
      if (!isTextStage(item)) return;
      const delays = [0, 16, 40, 90, 180, 320];
      for (const delay of delays) {{
        setTimeout(function() {{
          if (token !== state.loadToken) return;
          if (item.epoch !== undefined && item.epoch !== viewEpoch) return;
          if (frameFor(item.index, state.active) !== frame) return;
          scrollFrameToBottom(frame);
        }}, delay);
      }}
    }}
    window.setPageCaptions = function(batch) {{
      for (const item of batch) setCaption(item);
    }};
    window.setGroup = function(current, total) {{
      const label = document.getElementById('groupLabel');
      if (label) label.textContent = String(current + 1) + '/' + String(total);
    }};
    window.resetFrames = function(captions, epoch) {{
      if (epoch !== undefined && epoch !== null) viewEpoch = epoch;
      for (let index = 0; index < pageState.length; index++) {{
        const state = pageState[index];
        state.loadToken += 1;
        if (state.loadTimer) clearTimeout(state.loadTimer);
        state.loadTimer = null;
        state.loading = false;
        state.currentHtml = '';
        state.pendingItem = null;
        stopFrameAutoBounce(index);
        const cell = cellFor(index);
        if (cell) cell.classList.add('blanked');
        for (let slot = 0; slot < 2; slot++) {{
          const frame = frameFor(index, slot);
          if (!frame) continue;
          frame.onload = null;
        }}
      }}
      for (const item of captions || []) setCaption(item);
    }};
    function startBufferedLoad(item) {{
      if (item.epoch !== undefined && item.epoch !== viewEpoch) return;
      setCaption(item);
      if (typeof item.html !== 'string') {{
        return;
      }}
      const state = pageState[item.index];
      if (!state || item.html === state.currentHtml) {{
        return;
      }}
      if (state.loading) {{
        state.pendingItem = item;
        return;
      }}
      const nextSlot = 1 - state.active;
      const hidden = frameFor(item.index, nextSlot);
      const visible = frameFor(item.index, state.active);
      if (!hidden || !visible) return;
      state.loading = true;
      state.loadToken += 1;
      const token = state.loadToken;
      if (state.loadTimer) clearTimeout(state.loadTimer);
      state.pendingItem = null;
      const finishLoad = function(fromTimeout) {{
        if (token !== state.loadToken) return;
        if (item.epoch !== undefined && item.epoch !== viewEpoch) return;
        if (fromTimeout && isTextStage(item) && !frameReady(hidden)) {{
          state.loadTimer = setTimeout(function() {{ finishLoad(true); }}, 80);
          return;
        }}
        if (state.loadTimer) {{
          clearTimeout(state.loadTimer);
          state.loadTimer = null;
        }}
        hidden.onload = null;
        if (isTextStage(item)) scrollFrameToBottom(hidden);
        visible.classList.remove('active');
        visible.classList.add('hidden');
        hidden.classList.remove('hidden');
        hidden.classList.add('active');
        state.active = nextSlot;
        state.currentHtml = item.html;
        state.loading = false;
        setCaption(item);
        const cell = cellFor(item.index);
        if (cell) cell.classList.remove('blanked');
        keepTextFrameAtBottom(hidden, item, state, token);
        if (item.autoScroll !== true) startFrameAutoBounce(item.index, hidden, token);
        if (state.pendingItem && state.pendingItem.html !== state.currentHtml) {{
          const pending = state.pendingItem;
          state.pendingItem = null;
          startBufferedLoad(pending);
        }}
      }};
      hidden.onload = function() {{ finishLoad(false); }};
      state.loadTimer = setTimeout(function() {{ finishLoad(true); }}, isTextStage(item) ? 700 : 1200);
      hidden.srcdoc = item.html;
    }}
    window.updatePages = function(batch) {{
      for (const item of batch) {{
        if (item.epoch !== undefined && item.epoch !== viewEpoch) continue;
        const caption = document.getElementById('caption-' + item.index);
        if (!caption) continue;
        startBufferedLoad(item);
      }}
    }};
    updateBounceButton();
  </script>
</body>
</html>"""
