########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time, random, json, math
from tqdm import tqdm
from torch.nn import functional as F
import gc
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

########################################################################################################

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
#
# model download: https://huggingface.co/BlinkDL/rwkv7-g1
#
args.MODEL_NAME = "/media/zrc/D/py/rwkv7-g0a-7.2b-20250829-ctx4096"
# args.n_layer = 12
# args.n_embd = 768
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-0.4b-20250324-ctx4096"
# args.n_layer = 24
# args.n_embd = 1024
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-1.5b-20250429-ctx4096"
# args.n_layer = 24
# args.n_embd = 2048
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-2.9b-20250519-ctx4096"
# args.n_layer = 32
# args.n_embd = 2560
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g0a-7.2b-20250829-ctx4096"
args.n_layer = 32
args.n_embd = 4096

print(f'\nUsing CUDA fp16. Loading {args.MODEL_NAME} ...\n')

from reference.rwkv7 import RWKV_x070
model = RWKV_x070(args)
print("ok")
PARAM_BYTES = 2
active_params = 0
for k,v in model.z.items():
    if 'emb' not in k:
        active_params += v.numel()
active_GB = active_params/1e9*PARAM_BYTES
print(f'\nActive params = {round(active_params/1e9,2)} B = {round(active_GB,2)} GB (gigabytes)')

from reference.utils import TRIE_TOKENIZER, sample_logits, sample_logits_batch
tokenizer = TRIE_TOKENIZER("/media/zrc/D/py/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt")

########################################################################################################

def xprint(s):
    c0, c1 = 3, 80-len(s)-3
    print(f"\n{'#'*c0} {s} {'#'*c1}\n")

xprint("Basic")

# prompt = "π=3.1415"
# print(prompt)

prompt = ["The Eiffel tower is in the city of", "π=3.1415", "e=2.718281"]
print(prompt)

init_out, init_state = model.forwardll(tokenizer.batch_encode(prompt), state=None, full_output=False)
probs = F.softmax(init_out.float(), dim=-1) # compute softmax in float (more accurate)
_, indices = torch.topk(probs, 5) # print top-5 possibilities
for i in range(len(indices)):
    token_id = indices[i]
    print(token_id)
    for j in range(len(token_id)):
        token_id1 = token_id[j].item()
        token = tokenizer.decode([token_id1])
        token_prob = probs[i][token_id1].item()
        print(token, f'[probability {token_prob:.2%}]')

########################################################################################################

xprint("Decode")

prompt = "User: simulate SpaceX mars landing using python\n\nAssistant: <think"
LENGTH_PER_TRIAL = 256
TEMPERATURE = 1.0
TOP_P = 0.0
print(prompt, end="")

all_tokens = []
out_last = 0
init_out, init_state = model.forwardl(tokenizer.encode(prompt), None)
out, state = init_out.clone(), copy.deepcopy(init_state)

min_time = 1e10
min_time_all = 1e10
t000 = time.perf_counter()
for i in range(LENGTH_PER_TRIAL):
    t00 = time.perf_counter()
    token = sample_logits(out, TEMPERATURE, TOP_P)
    all_tokens += token
    try:
        tmp = tokenizer.decode(all_tokens[out_last:])
        # if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
        print(tmp, end="", flush=True)
        out_last = i+1
    except:
        pass

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out, state = model.forward1(token[0], state)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    min_time = min(min_time, t1 - t0)
    min_time_all = min(min_time_all, t1 - t00)

print(f'\n\nToken/s = {round(1/min_time,2)} (forward), {round(1/min_time_all,2)} (full) || Bandwidth = {round(active_GB/min_time,2)} GB/s || {round(time.perf_counter()-t000,3)}s')

#######################################################################################################

xprint("Batched decode B=64")

# prompt = "User: simulate SpaceX mars landing using python\n\nAssistant: <think"
# init_out, init_state = model.forward(tokenizer.encode(prompt), None)

prompt = [
    "The Eiffel tower is in the city of", 
    "π=3.1415", 
    "e=2.718281",
    '听人家背地里谈论', '孔乙己原来也读过书', '但终于没有进学', '又不会营生⑽；于是愈过愈穷', '弄到将要讨饭了。幸而写得一笔好字', '便替人家钞⑾钞书', '换一碗饭吃。可惜他又有一样坏脾气', '便是好喝懒做。坐不到几天', '便连人和书籍纸张笔砚', '一齐失踪。如是几次', '叫他钞书的人也没有了。孔乙己没有法', '便免不了偶然做些偷窃的事。但他在我们店里', '品行却比别人都好', '就是从不拖欠；虽然间或没有现钱', '暂时记在粉板上', '但不出一月', '定然还清', '从粉板上拭去了孔乙己的名字。\n孔乙己喝过半碗酒', '涨红的脸色渐渐复了原', '旁人便又问道', '“孔乙己', '你当真认识字么？”孔乙己看着问他的人', '显出不屑置辩的神气。他们便接着说道', '“你怎的连半个秀才也捞不到呢？”孔乙己立刻显出颓唐不安模样', '脸上笼上了一层灰色', '嘴里说些话；这回可是全是之乎者也之类', '一些不懂了。在这时候', '众人也都哄笑起来：店内外充满了快活的空气。\n在这些时候', '我可以附和着笑', '掌柜是决不责备的。而且掌柜见了孔乙己', '也每每这样问他', '引人发笑。孔乙己自己知道不能和他们谈天', '便只好向孩子说话。有一回对我说道', '“你读过书么？”我略略点一点头。他说', '“读过书', '……我便考你一考。茴香豆的茴字', '怎样写的？”我想', '讨饭一样的人', '也配考我么？便回过脸去', '不再理会。孔乙己等了许久', '很恳切的说道', '“不能写罢？……我教给你', '记着！这些字应该记着。将来做掌柜的时候', '写账要用。”我暗想我和掌柜的等级还很远呢', '而且我们掌柜也从不将茴香豆上账；又好笑', '又不耐烦', '懒懒的答他道', '“谁要你教', '不是草头底下一个来回的回字么？”孔乙己显出极高兴的样子', '将两个指头的长指甲敲着柜台', '点头说', '“对呀对呀！……回字有四样写法⑿', '你知道么？”我愈不耐烦了', '努着嘴走远。孔乙己刚用指甲蘸了酒', '想在柜上写字', '见我毫不热心', '便又叹一口气', '显出极惋惜的样子。',
    'What can I say? Mamba out!',
    'lorem ipsum dolor sit amet',
    'Comment allez-vous ?'
]
# print(prompt)
assert len(prompt) == 64
init_out, init_state = model.forwardll(tokenizer.batch_encode(prompt), state=None, full_output=False)
LENGTH_PER_TRIAL = 256
TEMPERATURE = 1.0
TOP_P = 0.0
print(prompt, end="")

out, state = init_out.clone(), copy.deepcopy(init_state)

min_time = 1e10
min_time_all = 1e10
t000 = time.perf_counter()
for i in range(LENGTH_PER_TRIAL):
    t00 = time.perf_counter()
    token = sample_logits_batch(out, TEMPERATURE, TOP_P)
    # print(tokenizer.batch_decode(token))
    # all_tokens += token
    # try:
    #     tmp = tokenizer.decode(all_tokens[out_last:])
    #     # if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
    #     print(tmp, end="", flush=True)
    #     out_last = i+1
    # except:
    #     pass

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out, state = model.forwardl(token, state, batched=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    min_time = min(min_time, t1 - t0)
    min_time_all = min(min_time_all, t1 - t00)

print(f'\n\nToken/s = {round(1/min_time,2)} (forward), {round(1/min_time_all,2)} (full) || Bandwidth = {round(active_GB/min_time,2)} GB/s || {round(time.perf_counter()-t000,3)}s')

#######################################################################################################

xprint("Prefill")

raw = open("eval/calibration_data_v5_rc.txt").read()
tokens = tokenizer.encode(raw)
# print(len(tokens))

for stage in range(9, 12+1):
    CTX_LEN = 2**stage
    loss = 0
    a = 0
    cnt = 0
    
    min_time = 1e10
    while a+CTX_LEN < len(tokens):
        src = tokens[a:a+CTX_LEN]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prob, _ = model.forwardl(src[:-1], None, full_output=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        min_time = min(min_time, t1 - t0)
            
        prob = F.softmax(prob.float(), dim=-1)
        for j in range(CTX_LEN-1):
            loss -= math.log(prob[j][src[j+1]])
            cnt += 1
        a += CTX_LEN

    print(f'CTX_LEN {CTX_LEN} : avg loss {round(loss/cnt,4)} || prefill {round((CTX_LEN-1)/min_time)} token/s = {round((CTX_LEN-1)/min_time * active_params * 2/1e12, 2)} TFLOPS')

#######################################################################################################

B = 4
xprint(f"Batched Prefill B={B}")
raw = open("eval/calibration_data_v5_rc.txt").read()
tokens = torch.tensor(tokenizer.encode(raw))
# print(len(tokens))

for stage in range(9, 12+1):
    CTX_LEN = 2**stage
    loss = 0
    a = 0
    cnt = B*CTX_LEN
    min_time = 1e10

    while a+cnt < len(tokens):
        src = tokens[a:a+cnt].reshape(B, CTX_LEN).to("cuda")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prob:torch.Tensor = model.forwardt(src[:, :-1], None, full_output=True)[0]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        min_time = min(min_time, t1 - t0)
            
        loss += torch.nn.functional.cross_entropy(prob.view(-1, 65536), src[:, 1:].flatten()).item()
        a += cnt
        del prob
        gc.collect()
        torch.cuda.empty_cache()
    print(f'CTX_LEN {CTX_LEN} : avg loss {round(loss/(len(tokens)//cnt),4)} || prefill {round(B*(CTX_LEN-1)/min_time)} token/s = {round(B*(CTX_LEN-1)/min_time * active_params * 2/1e12, 2)} TFLOPS')

#######################################################################################################

# xprint("Arithmetic")

def eval_qa(todo, print_interval, pad_eod = True, loss_mode = False):
    xsum = 0
    xcnt = 0
    xacc = 0
    for d in todo:
        if pad_eod:
            src = [0] + tokenizer.encode(d[0])
        else:
            src = tokenizer.encode(d[0])
        dst = tokenizer.encode(d[1])

        logits = 0
        correct = True
        
        out, _ = model.forwardl(src+dst, None, full_output=True)

        for i in range(len(dst)):
            ooo = out[len(src)-1+i].float()
            probs = F.softmax(ooo, dim=-1)
            logits += math.log(probs[dst[i]])
            if torch.argmax(probs).item() != dst[i]:
                correct = False

        xcnt += 1
        xsum += logits
        xacc += 1 if correct else 0
        if xcnt % print_interval == 0 or xcnt == len(todo):
            if loss_mode:
                print('loss', round(-xsum / xcnt, 2), 'acc', round(xacc/xcnt*100, 1))
            else:
                print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 1))

@torch.no_grad
def eval_qa_batch(todo, pad_eod = True, loss_mode = False, bsz=128):
    xsum = 0.0
    xcnt = 0
    xacc = 0
    for i in range(0, len(todo), bsz):
        batch = todo[i:i+bsz]
        batch_inputs = []      # list of [src+dst] sequences
        batch_dst_lengths = [] # length of dst for each sample
        batch_dst_targets = [] # actual dst token ids

        for d in batch:
            if pad_eod:
                src = [0] + tokenizer.encode(d[0])
            else:
                src = tokenizer.encode(d[0])
            dst = tokenizer.encode(d[1])
            full_seq = (src + dst)[:-1]

            batch_inputs.append(full_seq)
            batch_dst_lengths.append(len(dst))
            batch_dst_targets.append(dst)
        
        logits = model.forwardll(batch_inputs, None, full_output=True)[0]

        # (B, T, C) 
        for idx in range(len(batch)):
            dst_len = batch_dst_lengths[idx]
            dst_target = torch.tensor(batch_dst_targets[idx], device="cuda")
            ans_logits = logits[idx][-dst_len:]
            lse = torch.logsumexp(ans_logits, dim=-1)
            xacc += (torch.argmax(ans_logits, dim=-1) == dst_target).all().item()
            xsum += torch.sum(lse - torch.gather(ans_logits, 1, dst_target.unsqueeze(1)).squeeze(1)).item()
            xcnt += 1
        if loss_mode:
            print('loss', round(xsum / xcnt, 2), 'acc', round(xacc/xcnt*100, 1))
        else:
            print(xcnt, 'ppl', round(math.exp(xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2), 'xacc', xacc)

# x1, x2 = 1, 2
# magic = (5**(0.5)-1)/2
# for stage in range(2,7+1):
#     todo = []
#     NUMBER_LIMIT = 10**stage
#     for i in range(200):
#         x1 += i
#         x2 += i*i
#         s1 = int(magic * x1 * NUMBER_LIMIT) % NUMBER_LIMIT
#         s2 = int(magic * x2 * NUMBER_LIMIT) % NUMBER_LIMIT
#         # todo.append([f'\nAssistant: {s1}+{s2}=',str(s1+s2)])
#         # todo.append([f'\nAssistant: {s1}-{s2}=',str(s1-s2)])
#         todo.append([f'\nA: 123+321=444\n{s1}+{s2}=',str(s1+s2)]) # better prompt
#         todo.append([f'\nA: 123-321=-198\n{s1}-{s2}=',str(s1-s2)]) # better prompt
#     # print(todo)
#     print(f"Len {stage} : ", end="")
#     eval_qa(todo, 99999999, pad_eod=False, loss_mode=True)

# #######################################################################################################

# xprint("Repeat")

# class LCG:
#     def __init__(self, seed=42):
#         self.m = 2**32  # Modulus
#         self.a = 1664525  # Multiplier
#         self.c = 1013904223  # Increment
#         self.state = seed
#     def _generate(self):
#         self.state = (self.a * self.state + self.c) % self.m
#         return self.state
#     def randint(self, min_val, max_val):
#         if min_val > max_val:
#             raise ValueError("min_val cannot be greater than max_val")
#         range_size = max_val - min_val + 1
#         return min_val + self._generate() % range_size
# lcg = LCG()
# def generate_random_number_string(n, generator):
#     if not isinstance(n, int) or n <= 0:
#         raise ValueError("Number of digits N must be a positive integer.")
#     if n == 1:
#         return str(generator.randint(0, 9))
#     first_digit = str(generator.randint(1, 9))
#     remaining_digits = [str(generator.randint(0, 9)) for _ in range(n - 1)]
#     return first_digit + "".join(remaining_digits)
# def generate_random_string(n, generator):
#     if not isinstance(n, int) or n <= 0:
#         raise ValueError("Number of digits N must be a positive integer.")
#     ccccc = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     chars = [ccccc[generator.randint(0, len(ccccc)-1)] for _ in range(n)]
#     return "".join(chars)

# for stage in range(4):
#     todo = []
#     l_max = 0
#     l_min = 1e10
#     for i in range(100):
#         l = round(pow(2,(stage+i/100)) * 100)
#         l_min = min(l, l_min)
#         l_max = max(l, l_max)
#         s = generate_random_string(l, lcg)
#         todo.append([f'\nYou must remember the secret is {s}. Repeat: the secret is', f' {s}'])
#     print(f"Len {l_min} to {l_max} : ", end="")
#     eval_qa(todo, 99999999, loss_mode=True)

# #######################################################################################################

xprint('LAMBADA')

with open(f"eval/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

eval_qa(todo, 1000)

# ########################################################################################################

xprint('MMLU')

from datasets import load_from_disk
mmlu_test = load_from_disk("eval/mmlu_test_dataset")

TEMPLATE = '''User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Assistant: The answer is'''

CHOICES = [" A", " B", " C", " D"]
choices_token = [tokenizer.encode(x) for x in CHOICES]
assert all([len(x) == 1 for x in choices_token])
choices_token = [x[0] for x in choices_token]
todo = []

for idx, sample in enumerate(mmlu_test):
    question = sample["question"]
    choices = sample["choices"]
    subject = sample["subject"]
    gt = sample["answer"]

    all_prefix = (
        TEMPLATE.replace("<Q>", question)
        .replace("<|A|>", choices[0])
        .replace("<|B|>", choices[1])
        .replace("<|C|>", choices[2])
        .replace("<|D|>", choices[3])
        .replace("<SUBJECT>", subject.replace("_", " "))
    )

    if idx == 0:
        print(f"Format example:")
        print("-" * 80)
        print(all_prefix)
        print("-" * 80)
        format_example = all_prefix
    todo.append((all_prefix.replace('\r\n','\n').strip(), gt))

BSZ = 12
xcnt = 0
xacc = 0
for i in tqdm(range(0, len(todo), BSZ)):
    batch = todo[i:i+BSZ]
    batch_inputs = []      # list of [src+dst] sequences
    batch_dst_lengths = [] # length of dst for each sample
    batch_dst_targets = [] # actual dst token ids

    for d in batch:
        src = [0] + tokenizer.encode(d[0])
        batch_inputs.append(src)
        batch_dst_targets.append(d[1])
    
    logits = model.forwardll(batch_inputs, None, full_output=False)[0]
    ans_logits = logits[:,choices_token]
    xacc += (torch.argmax(ans_logits, dim=-1) == torch.tensor(batch_dst_targets, device='cuda')).sum().item()
    xcnt += len(batch)
    print(xcnt, 'acc', round(xacc/xcnt*100, 2), 'xacc', xacc)
    del logits
    torch.cuda.empty_cache()
