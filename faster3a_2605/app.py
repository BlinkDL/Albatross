import gc, os, re
import gradio as gr
import torch
from datetime import datetime
from huggingface_hub import hf_hub_download
from pynvml import *
from rwkv.utils import PIPELINE, PIPELINE_ARGS

import rwkv7_fast_v3a as v3a

nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)

ctx_limit = 7000
gen_limit = 1000
max_bsz = 8
CHUNK_LEN = 512 # chunk prefill, save VRAM
SAMPLER_TOP_K = 500

########################## text rwkv ################################################################

title = "rwkv7-g1f-7.2b-20260414-ctx8192"
# model_path = hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename=f"{title}.pth")
model_path = "/dev/shm/rwkv7-g1f-7.2b-20260414-ctx8192.pth"

v3a.MODEL_PATH = model_path
v3a.WKV_MODE = "fp32io16"
v3a.EMB_DEVICE = "cpu"
v3a.RKV_MODE = "off"
v3a.CMIX_SPARSE = "no-fc"
v3a.LOWRANK_WEIGHT = "transpose"
v3a.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
v3a.load_extensions(v3a.WKV_MODE)
model = v3a.RWKV7()
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

decode_cache = {}

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

def get_decode_ctx(B: int):
    cached = decode_cache.get(B)
    if cached is not None:
        return cached
    # app.py does its own graph capture — disable auto-graph in forward_from_x
    os.environ["CUDA_GRAPH_AUTO"] = "0"
    state = model.zero_state(B)
    x = torch.empty((B, 1, v3a.C), device="cuda", dtype=torch.half)
    path = v3a.select_path(B, 1)
    for _ in range(2):
        model.forward_from_x(x, state, path)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model.forward_from_x(x, state, path)
    os.environ["CUDA_GRAPH_AUTO"] = "1"
    cached = (state, x, graph, output)
    decode_cache[B] = cached
    return cached

def copy_state_to_batch(dst, src):
    B = dst[2].shape[0]
    dst[0].copy_(src[0].expand(-1, -1, B, -1))
    dst[1].copy_(src[1].expand(-1, B, -1, -1, -1))
    dst[2].copy_(src[2].expand(B))

def tokens_to_x(tokens):
    token_tensor = torch.tensor(tokens, dtype=torch.long, device="cpu" if model.emb_cpu else "cuda").view(-1, 1)
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

def evaluate(
    ctx,
    token_count=200,
    batch_size=1,
    temperature=1.0,
    top_p=0.5,
    presencePenalty = 2,
    countPenalty = 0.2,
    penalty_decay = 0.99,
):
    sample_temperature = float(temperature)
    sample_top_p = float(top_p)
    if sample_temperature <= 0:
        sample_temperature = 1.0
        sample_top_p = 0
    else:
        sample_temperature = max(0.2, sample_temperature)
    args = PIPELINE_ARGS(temperature = sample_temperature, top_p = sample_top_p,
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    ctx = ctx.strip()
    B = min(max_bsz, max(1, int(batch_size)))
    all_tokens = [[] for _ in range(B)]
    out_last = [0 for _ in range(B)]
    out_str = ['' for _ in range(B)]
    occurrence = [{} for _ in range(B)]
    finished = [False for _ in range(B)]
    state = model.zero_state(1)
    decode_state, decode_x, decode_graph, decode_output = get_decode_ctx(B)
    next_tokens = [0 for _ in range(B)]
    out = None
    for i in range(int(token_count)):

        if i == 0:
            input_ids = pipeline.encode(ctx)[-ctx_limit:]
            if len(input_ids) == 0:
                yield ""
                return
            while len(input_ids) > 0:
                token_device = "cpu" if model.emb_cpu else "cuda"
                tokens = torch.tensor(input_ids[:CHUNK_LEN], dtype=torch.long, device=token_device)
                out = model.forward(tokens, state).view(-1)
                input_ids = input_ids[CHUNK_LEN:]
            copy_state_to_batch(decode_state, state)
            logits = out.view(1, -1).repeat(B, 1)
        else:
            decode_x.copy_(tokens_to_x(next_tokens))
            decode_graph.replay()
            logits = decode_output.view(B, -1)

        for b in range(B):
            if finished[b]:
                continue
            row = logits[b]
            for n in occurrence[b]:
                row[n] -= (args.alpha_presence + occurrence[b][n] * args.alpha_frequency)

        assert logits.is_cuda and logits.dim() == 2
        sampled = sample_logits_batch_cuda(
            logits,
            sample_temperature,
            sample_top_p,
            min(SAMPLER_TOP_K, logits.size(-1)),
        ).detach().cpu().tolist()
        active = 0
        next_tokens = [0 for _ in range(B)]
        for b in range(B):
            if finished[b]:
                continue
            token = sampled[b]
            if token in args.token_stop:
                finished[b] = True
                continue
            active += 1
            next_tokens[b] = token
            all_tokens[b] += [token]
            for xxx in occurrence[b]:
                occurrence[b][xxx] *= penalty_decay

            ttt = pipeline.decode([token])
            www = 1
            #if ttt in ' \t0123456789':
            #    www = 0
            #elif ttt in '\r\n,.;?!"\':+-*/=#@$%^&_`~|<>\\()[]{}，。；“”：？！（）【】':
            #    www = 0.5
            if token not in occurrence[b]:
                occurrence[b][token] = www
            else:
                occurrence[b][token] += www

            tmp = pipeline.decode(all_tokens[b][out_last[b]:])
            if '\ufffd' not in tmp:
                out_str[b] += tmp
                out_last[b] = len(all_tokens[b])
        if active == 0:
            break
        yield out_str[0].strip() if B == 1 else "\n====\n".join(x.strip() for x in out_str)

    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'{timestamp} - vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    del out
    del state
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str[0].strip() if B == 1 else "\n====\n".join(x.strip() for x in out_str)

examples = [
    ["System: Tools:\n- get_weather(location: string, unit?: \"celsius\" | \"fahrenheit\")\n- get_stock_price(ticker: string)\n- translate_text(text: string, target_language: string)\nReturn only a JSON function call.\n\nUser: Translate \"Will it rain tomorrow?\" into Japanese.\n\nAssistant: ```json", 200, 1, 0, 0, 0, 0.99],
    ["System: Tools:\n[{\"name\":\"find_free_slots\",\"description\":\"Find free calendar slots\",\"arguments\":{\"date\":{\"type\":\"string\"},\"duration_minutes\":{\"type\":\"integer\"},\"time_window\":{\"type\":\"string\"}}},{\"name\":\"create_calendar_event\",\"description\":\"Create a calendar event\",\"arguments\":{\"title\":{\"type\":\"string\"},\"start_time\":{\"type\":\"string\"},\"end_time\":{\"type\":\"string\"},\"attendees\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}}}]\nReturn only a JSON function call.\n\nUser: Schedule a 30-minute sync with Bob on 2026-05-08 afternoon.\n\nAssistant: ```json\n{\"name\":\"find_free_slots\",\"arguments\":{\"date\":\"2026-05-08\",\"duration_minutes\":30,\"time_window\":\"afternoon\"}}\n```\n\nUser: Function output:\n{\"free_slots\":[{\"start\":\"2026-05-08T15:00:00+09:00\",\"end\":\"2026-05-08T15:30:00+09:00\"}],\"bob_email\":\"bob@example.com\"}\n\nAssistant: ```json", 200, 1, 0, 0, 0, 0.99],
    [generate_prompt("Please give the pros and cons of hodl versus active trading."), gen_limit, 1, 0.5, 2, 0.2, 0.99],
    [generate_prompt("Write a simple webpage. When a user clicks the button, it shows a random joke from a list of 4 jokes."), gen_limit, 1, 0.5, 2, 0.2, 0.99],
    ["User: What is the maximum value of $4(x + 7)(2 - x)$, over all real numbers $x$?\n\nAssistant: <think", gen_limit, 1, 0.5, 2, 0.2, 0.99],
    ["A few light taps upon the pane made her turn to the window. It had begun to snow again.", gen_limit, 1, 0.5, 2, 0.2, 0.99],
    ["Assistant: How can we persuade Elon Musk to follow you on Twitter? Let's think step by step and provide an expert response:", gen_limit, 1, 0.5, 2, 0.2, 0.99],
    [generate_prompt("東京で訪れるべき素晴らしい場所とその紹介をいくつか挙げてください。"), gen_limit, 1, 0.5, 2, 0.2, 0.99],
    [generate_prompt("Write a story using the following information.", "A man named Alex chops a tree down."), gen_limit, 1, 0.5, 2, 0.2, 0.99],
    ['''Japanese: 春の初め、桜の花が満開になる頃、小さな町の片隅にある古びた神社の境内は、特別な雰囲気に包まれていた。\n\nEnglish:''', gen_limit, 1, 0.5, 2, 0.2, 0.99],
    ["En una pequeña aldea escondida entre las montañas de Andalucía, donde las calles aún conservaban el eco de antiguas leyendas, vivía un joven llamado Alejandro.", gen_limit, 1, 0.5, 2, 0.2, 0.99],
    ["Dans le cœur battant de Paris, sous le ciel teinté d'un crépuscule d'or et de pourpre, se tenait une petite librairie oubliée par le temps.", gen_limit, 1, 0.5, 2, 0.2, 0.99],
    ["في تطور مذهل وغير مسبوق، أعلنت السلطات المحلية في العاصمة عن اكتشاف أثري قد يغير مجرى التاريخ كما نعرفه.", gen_limit, 1, 0.5, 2, 0.2, 0.99],
    ['''“当然可以，大宇宙不会因为这五公斤就不坍缩了。”关一帆说，他还有一个没说出来的想法：也许大宇宙真的会因为相差一个原子的质量而由封闭转为开放。大自然的精巧有时超出想象，比如生命的诞生，就需要各项宇宙参数在几亿亿分之一精度上的精确配合。但程心仍然可以留下她的生态球，因为在那无数文明创造的无数小宇宙中，肯定有相当一部分不响应回归运动的号召，所以，大宇宙最终被夺走的质量至少有几亿吨，甚至可能是几亿亿亿吨。\n但愿大宇宙能够忽略这个误差。\n程心和关一帆进入了飞船，智子最后也进来了。她早就不再穿那身华丽的和服了，她现在身着迷彩服，再次成为一名轻捷精悍的战士，她的身上佩带着许多武器和生存装备，最引人注目的是那把插在背后的武士刀。\n“放心，我在，你们就在！”智子对两位人类朋友说。\n聚变发动机启动了，推进器发出幽幽的蓝光，''', gen_limit, 1, 0.5, 2, 0.2, 0.99],
    ['''Edward: I am Edward Elric from Fullmetal Alchemist.\n\nUser: Hello Edward. What have you been up to recently?\n\nEdward:''', gen_limit, 1, 0.5, 2, 0.2, 0.99],
]
examples = [[x[0], x[1], 1, *x[2:]] for x in examples]

##################################################################################################################
with gr.Blocks(title=title, theme=gr.themes.Base()) as demo:
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>{title}</h1>\n</div>")

    with gr.Tab("=== Base Model (Raw Generation) ==="):
        gr.Markdown(f'This is [RWKV7 G-series](https://huggingface.co/BlinkDL/rwkv7-g1) reasoning base LM - an attention-free pure RNN [RWKV-LM](https://github.com/BlinkDL/RWKV-LM). Try topp 0.3 for math. Supports 100+ world languages and code. Check [600+ Github RWKV projects](https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories). *** Can try examples (bottom of page) *** (can edit them). Demo limited to ctxlen {ctx_limit}.')
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=6, label="Prompt", value="User: simulate SpaceX mars landing using python\n\nAssistant: <think></think")
                token_count = gr.Slider(10, gen_limit, label="Max Tokens", step=10, value=gen_limit)
                batch_size = gr.Slider(1, max_bsz, label="Batch Size", step=1, value=max_bsz)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 0.95, label="Top P", step=0.05, value=0.5)
                presence_penalty = gr.Slider(0.0, 2.0, label="Presence Penalty", step=0.1, value=2)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=0.2)
                penalty_decay = gr.Slider(0.99, 0.999, label="Penalty Decay", step=0.001, value=0.99)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=20, max_lines=100)
        data = gr.Dataset(components=[prompt, token_count, batch_size, temperature, top_p, presence_penalty, count_penalty, penalty_decay], samples=examples, samples_per_page=50, label="Example Instructions", headers=["Prompt", "Max Tokens", "Batch Size", "Temperature", "Top P", "Presence Penalty", "Count Penalty", "Penalty Decay"])
        submit.click(evaluate, [prompt, token_count, batch_size, temperature, top_p, presence_penalty, count_penalty, penalty_decay], [output])
        clear.click(lambda: None, [], [output])
        data.click(lambda x: x, [data], [prompt, token_count, batch_size, temperature, top_p, presence_penalty, count_penalty, penalty_decay])

demo.queue(default_concurrency_limit=1, max_size=10)
demo.launch(share=False, server_name="0.0.0.0")
