import curses, time, threading, random
from collections import deque
from typing import List, Optional
import math
import multiprocessing as mp
import queue

import torch, types, copy
import numpy as np
from torch.nn import functional as F

GRID_W, GRID_H = 16, 8
NUM_PANELS = GRID_W * GRID_H
BATCH_SIZE = NUM_PANELS
GENERATION_LENGTH = 4000
SAMPLER_NOISE = 3.0 # here we use simple (fast) sampling = greedy(logits + noise)

# find some random tokens as first token?
# words = []
# with open("reference/rwkv_vocab_v20230424.txt", "r", encoding="utf-8") as f:
#     lines = f.readlines()
#     for l in lines:
#         x = eval(l[l.index(' '):l.rindex(' ')])
#         if isinstance(x, str) and all(c.isalpha() for c in x) and x[0].isupper() and all(c.islower() for c in x[1:]) and ' ' not in x:
#             words.append(x)
# words = random.sample(words, BATCH_SIZE)

# or, use "The" for all panels?
words = ["The" for _ in range(BATCH_SIZE)]

########################################################################################################

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
#
# model download: https://huggingface.co/BlinkDL/rwkv7-g1
#
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.1b-20250728-ctx4096"
# args.n_layer = 12
# args.n_embd = 768
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.4b-20250905-ctx4096"
# args.n_layer = 24
# args.n_embd = 1024
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-1.5b-20250429-ctx4096"
# args.n_layer = 24
# args.n_embd = 2048
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-2.9b-20250519-ctx4096"
# args.n_layer = 32
# args.n_embd = 2560
args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g0a-7.2b-20250829-ctx4096"
args.n_layer = 32
args.n_embd = 4096

from reference.rwkv7 import RWKV_x070
from reference.utils import TRIE_TOKENIZER, sampler_simple_batch

########################################################################################################

def computation_process(text_queue, shutdown_event):
    """Computation process that runs the model inference and sends results to UI"""
    try:
        # Initialize model
        model = RWKV_x070(args)
        tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")
        
        # Initialize state
        state = [None for _ in range(args.n_layer * 3)]
        for i in range(args.n_layer):
            state[i*3+0] = torch.zeros((BATCH_SIZE, args.n_embd), dtype=torch.half, requires_grad=False, device="cuda")
            state[i*3+1] = torch.zeros((BATCH_SIZE, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
            state[i*3+2] = torch.zeros((BATCH_SIZE, args.n_embd), dtype=torch.half, requires_grad=False, device="cuda")
        
        # Send initial words to UI
        for i, word in enumerate(words):
            text_queue.put(("text", i, word))
        
        # Initial state with initial words
        tokens = [tokenizer.encode(prompt) for prompt in words]
        out, state = model.forward_batch(tokens, state)
        
        perf_interval = 10
        times = []
        all_times = []
        
        for i in range(GENERATION_LENGTH):
            if shutdown_event.is_set():
                break
                
            t00 = time.perf_counter()
            token = sampler_simple_batch(out, SAMPLER_NOISE).tolist()
            
            # Send decoded tokens to UI
            for n in range(BATCH_SIZE):
                decoded_text = tokenizer.decode(token[n])
                text_queue.put(("text", n, decoded_text))
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, state = model.forward_batch(token, state)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            times.append(t1 - t0)
            all_times.append(t1 - t00)

            if i % perf_interval == 0:
                times_tmp = np.percentile(times, 10) if times else 0
                all_times_tmp = np.percentile(all_times, 10) if all_times else 0
                times.clear()
                all_times.clear()
                # Send performance info to main process
                if times_tmp > 0 and all_times_tmp > 0:
                    text_queue.put(("perf", -1, f'Token/s = {round(BATCH_SIZE/times_tmp,2)} (forward), {round(BATCH_SIZE/all_times_tmp,2)} (full)'))
                    
    except Exception as e:
        text_queue.put(("error", -1, f"Error: {str(e)}"))
        print(e)
    finally:
        text_queue.put(("done", -1, "Finished"))

########################################################################################################

_ui_singleton = None
_singleton_lock = threading.Lock()

class PanelState:
    __slots__ = ("width","height","lines","cur","pending","lock","dirty")
    def __init__(self,w:int,h:int,history_lines:int=4096):
        self.width=max(4,w)
        self.height=max(1,h)
        self.lines=deque(maxlen=history_lines)
        self.cur=""
        self.pending=deque()
        self.lock=threading.Lock()
        self.dirty=True
    def set_size(self,w:int,h:int):
        self.width=max(4,w)
        self.height=max(1,h)
    def enqueue(self,s:str):
        with self.lock:
            self.pending.append(s)
    def drain_and_wrap(self)->bool:
        buf=None
        with self.lock:
            if self.pending:
                buf="".join(self.pending); self.pending.clear()
        if not buf: return False
        w=self.width; cur=self.cur; i=0; L=len(buf)
        while i<L:
            ch=buf[i]
            if ch=="\n":
                self.lines.append(cur); cur=""
            else:
                cur+=ch
                if len(cur)>=w:
                    self.lines.append(cur); cur=""
            i+=1
        self.cur=cur
        self.dirty=True
        return True
    def visible_tail(self)->List[str]:
        tail=list(self.lines)
        if self.cur: tail.append(self.cur)
        h=self.height
        if len(tail)>=h: return tail[-h:]
        return [""]*(h-len(tail))+tail

class TextGridUI:
    def __init__(self,fps:int=30,history_lines:int=4096,text_queue=None):
        self.fps=max(1,fps)
        self.history_lines=history_lines
        self.stop_event=threading.Event()
        self.stdscr=None
        self.panels:List[PanelState]=[]
        self.windows:List[curses.window]=[]
        self.grid_cell_w=None
        self.grid_cell_h=None
        self._started=False
        self.text_queue=text_queue
        self.perf_info="Loading..."
        self.perf_dirty=True
        self.first_text_received=False
    def add_text(self,idx:int,s:str):
        if 0<=idx<NUM_PANELS: self.panels[idx].enqueue(s)
    
    def stop(self):
        """Stop the UI and clean up"""
        self.stop_event.set()

    def start(self):
        if self._started: return
        self._started=True
        curses.wrapper(self._curses_main)
    def _compute_layout(self):
        max_y,max_x=self.stdscr.getmaxyx()
        # Reserve 1 line at the top for performance info
        available_height = max_y - 1
        cell_w=max_x//GRID_W
        cell_h=available_height//GRID_H
        # Account for borders: need minimum space for border + content
        if cell_w<10 or cell_h<4:
            raise RuntimeError(f"Terminal too small: need >= {10*GRID_W}x{4*GRID_H}, current={max_x}x{max_y}")
        self.grid_cell_w, self.grid_cell_h = cell_w, cell_h
    def _init_windows(self):
        self.windows=[]; self.panels=[]
        cw=self.grid_cell_w; ch=self.grid_cell_h
        for r in range(GRID_H):
            for c in range(GRID_W):
                # Offset by 1 row to make space for performance info at top
                top=r*ch+1; left=c*cw
                win=self.stdscr.derwin(ch,cw,top,left)
                win.scrollok(False); win.nodelay(True)
                self.windows.append(win)
                # Account for border space: 1 char on each side horizontally, 1 line for top border, 1 for bottom
                # Content area is reduced by 2 chars width and 2 lines height
                p=PanelState(cw-2,ch-3,history_lines=self.history_lines)
                self.panels.append(p)
        for i,win in enumerate(self.windows):
            try:
                win.erase()
                # Draw border
                win.box()
                # Add label inside the border
                label=f"[{i:03d}]"
                win.addnstr(1,1,label,len(label))
                win.noutrefresh()
            except curses.error:
                pass
        curses.doupdate()
    def _curses_main(self,stdscr):
        self.stdscr=stdscr
        curses.curs_set(0); curses.noecho(); curses.cbreak()
        stdscr.nodelay(True); stdscr.keypad(True)
        
        # Initialize colors if supported
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_WHITE, -1)    # idle - white
            curses.init_pair(2, curses.COLOR_YELLOW, -1)   # processing - yellow
            curses.init_pair(3, curses.COLOR_GREEN, -1)    # completed - green
            curses.init_pair(4, curses.COLOR_RED, -1)      # error - red
            curses.init_pair(5, curses.COLOR_CYAN, -1)     # header - cyan
        try:
            self._compute_layout(); self._init_windows()
        except Exception as e:
            curses.endwin(); print(e); return
        target_dt=1.0/self.fps; last=time.perf_counter()
        while not self.stop_event.is_set():
            # Process messages from computation process
            if self.text_queue:
                try:
                    while True:
                        msg_type, panel_id, content = self.text_queue.get_nowait()
                        if msg_type == "text" and 0 <= panel_id < NUM_PANELS:
                            # Clear screen on first text to remove any warnings
                            if not self.first_text_received:
                                self.stdscr.clear()
                                self.first_text_received = True
                                # Force re-initialization of windows and performance display
                                self._init_windows()
                                self.perf_dirty = True
                            self.add_text(panel_id, content)
                        elif msg_type == "perf":
                            # Store performance info for display at top
                            self.perf_info = content
                            self.perf_dirty = True
                        elif msg_type == "error":
                            self.perf_info = f"Error: {content}"
                            self.perf_dirty = True
                        elif msg_type == "done":
                            # Computation finished, we can exit
                            self.perf_info = "Computation finished"
                            self.perf_dirty = True
                            self.stop_event.set()
                            break
                except queue.Empty:
                    pass
            
            # Display performance info at the top
            if self.perf_dirty:
                try:
                    max_y, max_x = self.stdscr.getmaxyx()
                    # Clear the top line
                    self.stdscr.move(0, 0)
                    self.stdscr.clrtoeol()
                    # Display performance info with color
                    perf_text = self.perf_info[:max_x-1]  # Truncate if too long
                    if curses.has_colors():
                        self.stdscr.addnstr(0, 0, perf_text, max_x-1, curses.color_pair(2) | curses.A_BOLD)  # Yellow and bold
                    else:
                        self.stdscr.addnstr(0, 0, perf_text, max_x-1)
                    self.perf_dirty = False
                except curses.error:
                    pass
            
            any_dirty=False
            for idx,p in enumerate(self.panels):
                changed=p.drain_and_wrap()
                if not (changed or p.dirty): continue
                win=self.windows[idx]
                cw=p.width; ch=self.grid_cell_h
                try:
                    win.erase()
                    
                    # Draw border
                    win.box()
                    
                    # Simple header with batch info only - inside the border
                    header = f"[{idx:03d}]"
                    
                    # Choose color based on status for header only
                    header_color = 5  # cyan for header
                    if curses.has_colors():
                        win.addnstr(1, 1, header, cw, curses.color_pair(header_color) | curses.A_BOLD)
                    else:
                        win.addnstr(1, 1, header, cw)
                    
                    # Content - display inside border area
                    visible=p.visible_tail()
                    row=2  # Start from row 2 (after border and header)
                    max_rows=ch-3  # Account for top border, header, and bottom border
                    for line in visible[-max_rows:]:
                        if len(line)>cw: line=line[:cw]
                        # Use normal color for content text, positioned inside border
                        win.addnstr(row,1,line,cw)
                        row+=1
                        if row>=ch-1: break  # Stop before bottom border
                    win.noutrefresh()
                    p.dirty=False
                    any_dirty=True
                except curses.error:
                    pass
            # Also mark as dirty if performance info was updated
            if self.perf_dirty:
                any_dirty = True
            if any_dirty: curses.doupdate()
            try:
                ch=self.stdscr.getch()
                if ch in (ord('q'),ord('Q'),27): self.stop_event.set()
            except curses.error:
                pass
            now=time.perf_counter(); dt=now-last
            if dt<target_dt: time.sleep(target_dt-dt)
            last=now
        curses.nocbreak(); self.stdscr.keypad(False); curses.echo(); curses.endwin()

def start_ui(fps:int=30,history_lines:int=4096,text_queue=None):
    global _ui_singleton
    with _singleton_lock:
        if _ui_singleton is None:
            _ui_singleton=TextGridUI(fps=fps,history_lines=history_lines,text_queue=text_queue)
    _ui_singleton.start()

def add_text(n:int,s:str):
    ui=_ui_singleton
    if ui is None: raise RuntimeError("UI not started. Call start_ui().")
    ui.add_text(n,s)

def stop_ui():
    """Stop the UI and clean up resources"""
    global _ui_singleton
    if _ui_singleton is not None:
        _ui_singleton.stop()
        time.sleep(0.1)  # Give a moment for the UI thread to clean up

if __name__=="__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Create communication queue and shutdown event
    text_queue = mp.Queue()
    shutdown_event = mp.Event()
    
    # Start computation process
    comp_process = mp.Process(
        target=computation_process, 
        args=(text_queue, shutdown_event),
        daemon=False
    )
    comp_process.start()
    # computation_process(text_queue, shutdown_event)
    # quit()
    
    try:
        # Start UI in main process
        start_ui(fps=30, text_queue=text_queue)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Signal computation process to shutdown
        shutdown_event.set()
        
        # Wait for computation process to finish
        comp_process.join(timeout=5.0)
        if comp_process.is_alive():
            print("Terminating computation process...")
            comp_process.terminate()
            comp_process.join(timeout=2.0)
            if comp_process.is_alive():
                comp_process.kill()
                comp_process.join()
        
        # Always stop UI and restore terminal
        stop_ui()
        
        # Ensure terminal is fully restored
        import os
        os.system('stty sane')  # Reset terminal settings
        print("\nDemo completed! Terminal restored.")
