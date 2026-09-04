########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

from typing import List
import os
current_path = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.library
from torch.library import register_fake
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

import torch.nn as nn
from torch.nn import functional as F

from torch.library import Library, impl

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script
# ⬇️ Need to install Triton
# MyModule = nn.Module
# MyFunction = torch.compile(mode='max-autotune-no-cudagraphs')
# MyStatic = torch.compile(mode='max-autotune-no-cudagraphs')
def __nop(ob): return ob
# MyFunction = __nop
# MyStatic = __nop

DTYPE = torch.half
HEAD_SIZE = 64

from torch.utils.cpp_extension import load

# sm_X, sm_Y = torch.cuda.get_device_capability()
load(
    name="rwkv7_state_fwd_fp16", 
    sources=[f"{current_path}/cuda/rwkv7_state_fwd_fp16.cpp", f"{current_path}/cuda/rwkv7_state_fwd_fp16.cu"], 
    is_python_module=False,
    verbose=True, 
    extra_cuda_cflags=[
        "-res-usage", 
        "--use_fast_math", 
        "-O3", 
        "--extra-device-vectorization", 
        f"-D_N_={HEAD_SIZE}", 
        # f"-gencode=arch=compute_{sm_X}{sm_Y},code=sm_{sm_X}{sm_Y}"
    ] + (
        ["-Xptxas -O3"] if os.name != "nt" else ["-Xcompiler", "/Zc:preprocessor"]
    )
)

class SAMPLING(torch.autograd.Function):
    def forward(ctx, logits, penalties, states, presence_penalty, repetition_penalty, penalty_decay, temperature, top_k, top_p):
        return torch.ops.rwkv7_state_fwd_fp16.batch_sampling_repetition_temperature_topk_topp(logits, penalties, states, presence_penalty, repetition_penalty, penalty_decay, temperature, top_k, top_p)
@torch.library.custom_op("mylib::sampling", mutates_args=("penalties","states"))
def Sampling(logits:torch.Tensor, penalties:torch.Tensor, states:torch.Tensor, presence_penalty:float=0.0, repetition_penalty:float=0.0, penalty_decay:float=0.0, temperature:float=1.0, top_k:int=-1, top_p:float=0.0) -> torch.Tensor:
    return SAMPLING.apply(logits, penalties, states, presence_penalty, repetition_penalty, penalty_decay, temperature, top_k, top_p)
@Sampling.register_fake
def _(logits:torch.Tensor, penalties:torch.Tensor, states:torch.Tensor, presence_penalty:float=0.0, repetition_penalty:float=0.0, penalty_decay:float=0.0, temperature:float=1.0, top_k:int=-1, top_p:float=0.0) -> torch.Tensor:
    V = logits.size(-1)
    B = penalties.size(0) if (penalties.dim() == 2) else 1
    T = logits.size(1) if (logits.dim() == 3) else 1
    return torch.empty((B,), dtype=torch.int32, device=penalties.device)

class CMIXONE(torch.autograd.Function):
    def forward(ctx, x, x_1, x_k, K, V):
        return torch.ops.rwkv7_state_fwd_fp16.cmix_one(x, x_1, x_k, K, V)

@torch.library.custom_op("mylib::RWKV_x070_CMix_one", mutates_args=("x_prev",))
def RWKV_x070_CMix_one(x:torch.Tensor, x_prev:torch.Tensor, x_k:torch.Tensor, K_:torch.Tensor, V_:torch.Tensor) -> torch.Tensor:
    return CMIXONE.apply(x, x_prev, x_k, K_, V_)


@RWKV_x070_CMix_one.register_fake
def _(x:torch.Tensor, x_prev:torch.Tensor, x_k:torch.Tensor, K_:torch.Tensor, V_:torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class WKV_7_ONE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            C = r.size()[0]
            H = C // HEAD_SIZE
            y = torch.empty((C,), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward_one(1, C, H, state, r, w, k, v, a, b, y, elapsed_t)
            return y

@torch.library.custom_op("mylib::RWKV7_ONE_OP", mutates_args=("state",))
# @MyDisable
def RWKV7_ONE_OP(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return WKV_7_ONE.apply(state, r, w, k, v, a, b, elapsed_t)
@RWKV7_ONE_OP.register_fake
def _(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return torch.empty_like(r)

class WKV_7_SEQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            T, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward_seq(1, T, C, H, state, r, w, k, v, a, b, y, elapsed_t)
            return y

@torch.library.custom_op("mylib::RWKV7_SEQ_OP", mutates_args=("state",))
# @MyDisable
def RWKV7_SEQ_OP(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return WKV_7_SEQ.apply(state, r, w, k, v, a, b, elapsed_t)
@RWKV7_SEQ_OP.register_fake
def _(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return torch.empty_like(r)

class WKV_7_BATCH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            B, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((B, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward_one(B, C, H, state, r, w, k, v, a, b, y, elapsed_t)
            return y

@torch.library.custom_op("mylib::RWKV7_ONE_BATCH_OP", mutates_args=("state",))
# @MyDisable
def RWKV7_ONE_BATCH_OP(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return WKV_7_BATCH.apply(state, r, w, k, v, a, b, elapsed_t)
@RWKV7_ONE_BATCH_OP.register_fake
def _(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return torch.empty_like(r)


class WKV_7_SEQ_BATCH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward_seq(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t)
            return y

@torch.library.custom_op("mylib::RWKV7_BATCH_OP", mutates_args=("state",))
# @MyDisable
def RWKV7_BATCH_OP(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return WKV_7_SEQ_BATCH.apply(state, r, w, k, v, a, b, elapsed_t)
@RWKV7_BATCH_OP.register_fake
def _(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return torch.empty_like(r)

class RWKV_x070(MyModule):
    def __init__(self, args):
        super().__init__()
        args.vocab_size = 65536
        args.head_size = 64
        self.args = args
        self.eval()
        
        self.z = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
        args.n_embd = self.n_head * self.head_size

        assert HEAD_SIZE == self.head_size
        assert self.head_size == args.head_size

        keys = list(z.keys())
        max_layer = -1
        for k in keys:
            kk = k.split('.')
            # if kk[0] == 'blocks' and int(kk[1]) >= 10:
            #     continue
            if 'att.g1' in k or 'att.g2' in k or 'att.a1' in k or 'att.a2' in k or 'att.w1' in k or 'att.w2' in k or 'att.v1' in k or 'att.v2' in k or 'ffn.value.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE, device="cuda")
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
            z[k] = z[k].contiguous()
            if kk[0] == 'blocks':
                max_layer = max(max_layer, int(kk[1]))
        args.n_layer = max_layer + 1
        print(args)
        self.n_layer, self.n_embd, self.head_size = args.n_layer, args.n_embd, self.head_size

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored

    def generate_zero_state(self, bsz=0):
        if bsz == 0:
            state = [None for _ in range(self.n_layer * 3 + 3)]
            state[self.n_layer*3] = torch.zeros((), dtype=torch.int32, requires_grad=False, device="cuda")
            state[self.n_layer*3+2] = torch.ops.rwkv7_state_fwd_fp16.setup_rand(42, 1)
            state[self.n_layer*3+1] = torch.zeros((self.args.vocab_size,), dtype=torch.float32, requires_grad=False, device="cuda")
            for i in range(self.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                state[i*3+0] = torch.zeros(self.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
                state[i*3+1] = torch.zeros((self.n_embd // self.head_size, self.head_size, self.head_size), dtype=DTYPE, requires_grad=False, device="cuda")
                state[i*3+2] = torch.zeros(self.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
        else:
            state = [None for _ in range(self.n_layer * 3 + 1)]
            state[self.n_layer*3] = torch.zeros((bsz,), dtype=torch.int32, requires_grad=False, device="cuda")
            # state[self.n_layer*3+1] = 
            # state[self.n_layer*3+2] = torch.ops.rwkv7_state_fwd_fp16.setup_rand(42, bsz)
            for i in range(self.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                state[i*3+0] = torch.zeros((bsz, self.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
                state[i*3+1] = torch.zeros((bsz, self.n_embd // self.head_size, self.head_size, self.head_size), dtype=DTYPE, requires_grad=False, device="cuda")
                state[i*3+2] = torch.zeros((bsz, self.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
        return state

    def forward(self, idx, state, full_output=False, with_sampling=False): # will modify state in-place
        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(torch.tensor(idx), state, full_output)
            else:
                x = self.z['emb.weight'][idx[0]]
                return self.forward_one(x, state, with_sampling)
        elif type(idx) is torch.Tensor:
            return self.forward_one(idx, state, with_sampling)
        else:
            x = self.z['emb.weight'][idx]
            return self.forward_one(x, state, with_sampling)
        
    def forward_batch(self, tokens, state, full_output=False): # will modify state in-place
        assert type(tokens) is list
        lengths = [len(x) for x in tokens]
        if len(set(lengths)) == 1 and full_output == False:
            return self.forward_batch_same_length(tokens, state, full_output)

        raise NotImplementedError("varlen not implemented")


    def forward_batch_same_length(self, tokens, state, full_output=False):
        assert type(tokens) is list
        assert len(set([len(x) for x in tokens])) == 1, 'here all sequences must have the same length'
        return self.forward_seq_batch(tokens, state, full_output)

    @MyFunction
    def forward_one(self, x:torch.Tensor, state:List[torch.Tensor], with_sampling:bool=False):
        with torch.no_grad(): 
            z = self.z
            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                xx, v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[3*i], v_first, state[3*i+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[3*self.n_layer])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_one(xx, state[3*i+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            state[3*self.n_layer] += 1
            if with_sampling:
                y = Sampling(x.to(torch.float32), state[self.n_layer*3+1], state[self.n_layer*3+2])
                # print("x:", x)
                # breakpoint()
                return y
            return x
        
    @MyFunction
    def forward_seq(self, idx:torch.Tensor, state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                xx, v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[3*i], v_first, state[3*i+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[3*self.n_layer])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq(xx, state[3*i+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            # state[2] += len(idx)
            state[3*self.n_layer] += len(idx)
            return x
        
    @MyFunction
    def forward_seq_batch(self, idxs:List[List[int]], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][torch.tensor(idxs, device=z['emb.weight'].device)]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                xx, v_first = RWKV_x070_TMix_seq_batch(i, self.n_head, self.head_size, xx, state[3*i], v_first, state[3*i+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[3*self.n_layer])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq_batch(xx, state[3*i+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[:,-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            state[3*self.n_layer] += len(idxs[0])
            return x
    
    @MyFunction
    def forward_seq_batch_1(self, idxs:torch.Tensor, state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idxs]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                xx, v_first = RWKV_x070_TMix_seq_batch(i, self.n_head, self.head_size, xx, state[3*i], v_first, state[3*i+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[3*self.n_layer])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq_batch(xx, state[3*i+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[:,-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            state[3*self.n_layer] += len(idxs[0])
            return x
        
    @MyFunction
    def forward_seq_batch_right(self, idxs:torch.Tensor, state:List[torch.Tensor], lens:torch.Tensor, full_output:bool=False):
        with torch.no_grad():
            L = idxs.size(1)
            att_mask = (torch.arange(L, device="cuda", dtype=torch.int32).unsqueeze(0) < (L - lens).unsqueeze(1)).unsqueeze(2)
            state[3*self.n_layer] = lens - L
            z = self.z
            x = z['emb.weight'][idxs]
            x.masked_fill_(att_mask, 0)
            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                xx.masked_fill_(att_mask, 0)
                xx, v_first = RWKV_x070_TMix_seq_batch_right(i, self.n_head, self.head_size, xx, state[3*i], v_first, state[3*i+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[3*self.n_layer], att_mask)
                x = x + xx
                x.masked_fill_(att_mask, 0)

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                xx.masked_fill_(att_mask, 0)
                xx = RWKV_x070_CMix_seq_batch(xx, state[3*i+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
                x.masked_fill_(att_mask, 0)

            if not full_output: 
                x = x[:,-1,:]
                x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            else:
                x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
                x.masked_fill_(att_mask, 0)
            x = F.linear(x, z['head.weight'])
            state[3*self.n_layer] += len(idxs[0])
            return x

########################################################################################################

@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    xx = x_prev - x
    x_prev.copy_(x)
    # x_prev = x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0))

    xx = RWKV7_ONE_OP(state, r, w, k, v, -kk, kka, elapsed_t) # !!! using CUDA to modify state in-place !!! (faster too)

    xx = F.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return F.linear((xx * g), O_), v_first

@MyStatic
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    T = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    x_prev.copy_(x[-1])
    # x_prev = x[-1]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0))

    xx = RWKV7_SEQ_OP(state, r, w, k, v, -kk, kka, elapsed_t)

    xx = F.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return F.linear((xx * g), O_), v_first

@MyStatic
def RWKV_x070_TMix_seq_batch(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    B,T,C = x.shape
    xx = torch.cat((x_prev.unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev.copy_(x[:,-1,:])
    # x_prev = x[:,-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)

    kk = F.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
    k = k * (1 + (a-1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0))

    xx = RWKV7_BATCH_OP(state, r, w, k, v, -kk, kka, elapsed_t).view(B*T,H*N)

    xx = F.group_norm(xx.view(B*T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,T,H*N)
    xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
    return F.linear((xx * g), O_), v_first


@MyStatic
def RWKV_x070_TMix_seq_batch_right(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t, att_mask):
    B,T,C = x.shape
    xx = torch.cat((x_prev.unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev.copy_(x[:,-1,:])
    # x_prev = x[:,-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)

    kk = F.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
    kk.masked_fill_(att_mask, 0)
    k = k * (1 + (a-1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0))

    xx = RWKV7_BATCH_OP(state, r, w, k, v, -kk, kka, elapsed_t).view(B*T,H*N)

    xx = F.group_norm(xx.view(B*T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,T,H*N)
    xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
    return F.linear((xx * g), O_), v_first

@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    x_prev.copy_(x[-1])
    # x_prev = x[-1]
    k = x + xx * x_k
    k = torch.relu(F.linear(k, K_)) ** 2
    # print("Sparsity:", (k == 0).float().mean().item())
    return k @ V_ # F.linear(k, V_)

@MyStatic
def RWKV_x070_CMix_seq_batch(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev.unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev.copy_(x[:,-1,:])
    # x_prev = x[:,-1,:]
    k = x + xx * x_k
    k = torch.relu(F.linear(k, K_)) ** 2
    return k @ V_ # F.linear(k, V_)
