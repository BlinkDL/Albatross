import curses, time, threading, random
from collections import deque
from typing import List, Optional
import math
import multiprocessing as mp
import queue
import unicodedata

import torch, types, copy
import numpy as np
from torch.nn import functional as F

GRID_W, GRID_H = 20, 16
NUM_PANELS = GRID_W * GRID_H
BATCH_SIZE = NUM_PANELS
GENERATION_LENGTH = 4000
SAMPLER_NOISE = 3.0 # here we use simple (fast) sampling = greedy(logits + noise)

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
# args.MODEL_NAME = "/home/molly/rwkv7-g0a-7.2b-20250829-ctx4096"
# args.n_layer = 32
# args.n_embd = 4096

TITLE_MODEL_NAME = "RWKV-7 7.2B"
TITLE_PRECISION = "FP16"
TITLE_GPU_NAME = "RTX 5090"

# find some random tokens as first token?
# words = []
# with open("reference/rwkv_vocab_v20230424.txt", "r", encoding="utf-8") as f:
#     lines = f.readlines()
#     for l in lines:
#         x = eval(l[l.index(' '):l.rindex(' ')])
#         if isinstance(x, str) and all(c.isalpha() for c in x) and x[0].isupper() and all(c.islower() for c in x[1:]) and ' ' not in x:
#             words.append(x)
# words = random.sample(words, BATCH_SIZE)

prompt_list = [
    # ["The" for _ in range(BATCH_SIZE)],
    ["Assistant: <think" for _ in range(BATCH_SIZE)],
    ["Assistant: <think>嗯" for _ in range(BATCH_SIZE)],
    ["Assistant: <think>私" for _ in range(BATCH_SIZE)]
]
current_prompt = 0

########################################################################################################

from reference.rwkv7 import RWKV_x070
from reference.utils import TRIE_TOKENIZER, sampler_simple_batch

def computation_process(text_queue, shutdown_event, control_queue=None):
    """Computation process that runs the model inference and sends results to UI"""
    global current_prompt
    try:
        # Initialize model
        model = RWKV_x070(args)
        tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

        while True:
            if shutdown_event.is_set():
                break
            
            try:
                # Initialize state
                state = [None for _ in range(args.n_layer * 3)]
                for i in range(args.n_layer):
                    state[i*3+0] = torch.zeros((BATCH_SIZE, args.n_embd), dtype=torch.half, requires_grad=False, device="cuda")
                    state[i*3+1] = torch.zeros((BATCH_SIZE, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
                    state[i*3+2] = torch.zeros((BATCH_SIZE, args.n_embd), dtype=torch.half, requires_grad=False, device="cuda")
                
                # clear all panels
                for i in range(NUM_PANELS):
                    text_queue.put(("clear", i, ""))

                # Send initial words to UI
                for i, word in enumerate(prompt_list[current_prompt]):
                    text_queue.put(("text", i, word))
                
                # Initial state with initial words
                tokens = [tokenizer.encode(prompt) for prompt in prompt_list[current_prompt]]
                out, state = model.forward_batch(tokens, state)
                
                perf_interval = 10
                times = []
                all_times = []
                
                for i in range(GENERATION_LENGTH):
                    if shutdown_event.is_set():
                        break
                    
                    # Check for control messages (like prompt switch)
                    if control_queue:
                        try:
                            while True:
                                msg_type, _, _ = control_queue.get_nowait()
                                if msg_type == "switch_prompt":
                                    # Switch to next prompt and restart generation
                                    current_prompt = (current_prompt + 1) % len(prompt_list)
                                    text_queue.put(("perf", -1, f"Switched to prompt {current_prompt + 1}: {prompt_list[current_prompt][0][:50]}..."))
                                    # Break from generation loop to restart with new prompt
                                    raise StopIteration("prompt_switch")
                        except queue.Empty:
                            pass
                            
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
                            text_queue.put(("perf", -1, f'{TITLE_MODEL_NAME} {TITLE_PRECISION} bsz{BATCH_SIZE} inference @ {TITLE_GPU_NAME} || Token/s = {round(BATCH_SIZE/times_tmp,2)} (forward), {round(BATCH_SIZE/all_times_tmp,2)} (full) || Const speed & VRAM because this is RNN || https://github.com/BlinkDL/Albatross https://rwkv.com || Press "a" to switch prompt'))
                            
            except StopIteration as e:
                if str(e) == "prompt_switch":
                    # Continue with the outer loop to restart with new prompt
                    continue
                else:
                    raise
                    
    except Exception as e:
        text_queue.put(("error", -1, f"Error: {str(e)}"))
        print(e)
    finally:
        text_queue.put(("done", -1, "Finished"))

########################################################################################################

_ui_singleton = None
_singleton_lock = threading.Lock()

def get_display_width(text: str) -> int:
    """Calculate the actual display width of text in terminal, considering CJK characters"""
    width = 0
    for char in text:
        # CJK characters (Chinese, Japanese, Korean) occupy 2 terminal columns
        if unicodedata.east_asian_width(char) in ('F', 'W'):
            width += 2
        else:
            width += 1
    return width

def truncate_to_width(text: str, max_width: int) -> str:
    """Truncate text to fit within max_width terminal columns"""
    if get_display_width(text) <= max_width:
        return text
    
    result = ""
    current_width = 0
    for char in text:
        char_width = 2 if unicodedata.east_asian_width(char) in ('F', 'W') else 1
        if current_width + char_width > max_width:
            break
        result += char
        current_width += char_width
    return result

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
    
    def clear(self):
        """Clear all content from the panel"""
        with self.lock:
            self.lines.clear()
            self.cur = ""
            self.pending.clear()
            self.dirty = True
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
                # Use display width instead of character count for proper CJK handling
                if get_display_width(cur)>=w:
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
    def __init__(self,fps:int=30,history_lines:int=4096,text_queue=None,control_queue=None):
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
        self.control_queue=control_queue
        self.perf_info=f"Loading {args.MODEL_NAME} ..."
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
        
        # Calculate base cell size
        base_cell_w = max_x // GRID_W
        base_cell_h = available_height // GRID_H
        
        # Calculate remaining space that would be wasted
        remaining_width = max_x - (base_cell_w * GRID_W)
        remaining_height = available_height - (base_cell_h * GRID_H)
        
        # Distribute remaining space among cells to maximize usage
        # Some cells will be 1 character/line larger to use all available space
        self.cell_widths = [base_cell_w + (1 if c < remaining_width else 0) for c in range(GRID_W)]
        self.cell_heights = [base_cell_h + (1 if r < remaining_height else 0) for r in range(GRID_H)]
        
        # No borders, no headers: need minimum space for content only
        if base_cell_w < 4 or base_cell_h < 1:
            raise RuntimeError(f"Terminal too small: need >= {4*GRID_W}x{1*GRID_H}, current={max_x}x{max_y}")
        
        # For backward compatibility, store the base cell size
        self.grid_cell_w, self.grid_cell_h = base_cell_w, base_cell_h
    def _init_windows(self):
        self.windows=[]; self.panels=[]
        
        # Calculate actual positions using variable cell sizes
        for r in range(GRID_H):
            for c in range(GRID_W):
                # Calculate actual cell dimensions
                cw = self.cell_widths[c]
                ch = self.cell_heights[r]
                
                # Calculate position - sum of previous cell sizes
                left = sum(self.cell_widths[:c])
                top = sum(self.cell_heights[:r]) + 1  # +1 for performance info at top
                
                win=self.stdscr.derwin(ch,cw,top,left)
                win.scrollok(False); win.nodelay(True)
                self.windows.append(win)
                # No borders, no header - use full panel space
                p=PanelState(cw,ch,history_lines=self.history_lines)
                self.panels.append(p)
        for i,win in enumerate(self.windows):
            try:
                win.erase()
                # No panel label - just clear the window
                win.noutrefresh()
            except curses.error:
                pass
        curses.doupdate()
    def _curses_main(self,stdscr):
        self.stdscr=stdscr
        curses.curs_set(0); curses.noecho(); curses.cbreak()
        stdscr.nodelay(True); stdscr.keypad(True)
        
        # Initialize colors if supported - using simple 3-color scheme for panel differentiation
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            # Simple 3-color scheme that ensures adjacent panels are always different colors
            # Using (row + col) % 3 pattern guarantees no adjacent panels have same color
            curses.init_pair(1, curses.COLOR_WHITE, -1)     # Color 0: white
            curses.init_pair(2, curses.COLOR_CYAN, -1)      # Color 1: cyan
            curses.init_pair(3, curses.COLOR_GREEN, -1)     # Color 2: green
            curses.init_pair(4, curses.COLOR_RED, -1)       # performance info - red for attention
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
                        elif msg_type == "clear" and 0 <= panel_id < NUM_PANELS:
                            # Clear the specified panel
                            self.panels[panel_id].clear()
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
                    # Display performance info with color - properly truncate CJK characters
                    perf_text = truncate_to_width(self.perf_info, max_x-1)
                    if curses.has_colors():
                        self.stdscr.addnstr(0, 0, perf_text, max_x-1, curses.color_pair(4) | curses.A_BOLD)  # Red and bold
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
                
                # Calculate row and column for this panel to get actual dimensions
                row_pos = idx // GRID_W  # Which row (0-15)
                col_pos = idx % GRID_W   # Which column (0-19)
                cw = self.cell_widths[col_pos]
                ch = self.cell_heights[row_pos]
                try:
                    win.erase()
                    
                    # Simple 3-color pattern: (row + col) % 3 ensures adjacent panels differ
                    # This guarantees any panel differs from its 4 neighbors (up/down/left/right)
                    color_index = (row_pos + col_pos) % 3
                    panel_color = color_index + 1  # color_pair IDs: 1=white, 2=cyan, 3=green
                    
                    # Add some visual variety with alternating bold pattern
                    use_bold = (row_pos + col_pos) % 2 == 0
                    
                    # Content - display without header, using full panel area
                    visible=p.visible_tail()
                    row=0  # Start from row 0 (no header)
                    max_rows=ch  # Use full panel height
                    for line in visible[-max_rows:]:
                        # Use proper truncation for CJK characters
                        if get_display_width(line)>cw: 
                            line=truncate_to_width(line, cw)
                        # Use panel color for content text with row-based bold variation
                        if curses.has_colors():
                            color_attr = curses.color_pair(panel_color)
                            if use_bold:
                                color_attr |= curses.A_BOLD
                            win.addnstr(row,0,line,cw,color_attr)
                        else:
                            win.addnstr(row,0,line,cw)
                        row+=1
                        if row>=ch: break  # Stop at panel bottom
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
                if ch in (ord('q'),ord('Q'),27): 
                    self.stop_event.set()
                elif ch in (ord('a'),ord('A')):
                    # Send prompt switch message to computation process
                    if self.control_queue:
                        self.control_queue.put(("switch_prompt", -1, ""))
            except curses.error:
                pass
            now=time.perf_counter(); dt=now-last
            if dt<target_dt: time.sleep(target_dt-dt)
            last=now
        curses.nocbreak(); self.stdscr.keypad(False); curses.echo(); curses.endwin()

def start_ui(fps:int=30,history_lines:int=4096,text_queue=None,control_queue=None):
    global _ui_singleton
    with _singleton_lock:
        if _ui_singleton is None:
            _ui_singleton=TextGridUI(fps=fps,history_lines=history_lines,text_queue=text_queue,control_queue=control_queue)
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
    
    # Create communication queues and shutdown event
    text_queue = mp.Queue()
    control_queue = mp.Queue()
    shutdown_event = mp.Event()
    
    # Start computation process
    comp_process = mp.Process(
        target=computation_process, 
        args=(text_queue, shutdown_event, control_queue),
        daemon=False
    )
    comp_process.start()
    # computation_process(text_queue, shutdown_event)
    # quit()
    
    try:
        # Start UI in main process
        start_ui(fps=30, text_queue=text_queue, control_queue=control_queue)
        
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
