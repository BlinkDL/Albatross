########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################
from typing import List, Optional
import os
current_path = os.path.dirname(os.path.abspath(__file__))

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

import torch.nn as nn
from torch.nn import functional as F

# MyModule = torch.jit.ScriptModule
# MyFunction = torch.jit.script_method
# MyStatic = torch.jit.script
# MyOverload = torch.jit.script_method
MyModule = nn.Module
def __nop(ob): return ob
MyFunction = torch.compile(mode='max-autotune-no-cudagraphs')
MyStatic = torch.compile(mode='max-autotune-no-cudagraphs')
MyOverload = torch.compile(mode='max-autotune-no-cudagraphs')

DTYPE = torch.float16

from torch.utils.cpp_extension import load
HEAD_SIZE = 64

load(name="rwkv7_batch_fwd_fp16", sources=[f"{current_path}/cuda/rwkv7_batch_fwd_fp16.cpp", f"{current_path}/cuda/rwkv7_batch_fwd_fp16.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
class WKV_7(torch.autograd.Function):
    @torch.compiler.disable
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            assert all(x.dtype == torch.float16 for x in [r,w,k,v,a,b])
            assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
            T, C = r.size()[-2:]
            B = 1 if r.dim() == 2 else r.size(0)
            H = C // HEAD_SIZE
            assert HEAD_SIZE == C // H
            y = torch.empty(r.size(), device=k.device, dtype=torch.float16, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_batch_fwd_fp16.forward(B, T, C, H, state, r, w, k, v, a, b, y)
            return y
def RWKV7_OP(state, r, w, k, v, a, b):
    return WKV_7.apply(state, r, w, k, v, a, b)

########################################################################################################

class RWKV_x070(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        self.head_size = args.head_size
        self.eval()
        
        self.z = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape

        assert HEAD_SIZE == self.head_size
        assert self.head_size == args.head_size

        keys = list(z.keys())
        for k in keys:
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=torch.float16, device="cuda")
            if k.endswith('att.r_k'): z[k] = z[k].flatten()

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])

    # @MyFunction
    # def forward(self, idx:int|List[int]|List[List[int]], state:Optional[torch.Tensor], full_output:bool=False, align_mode:str="right"):
    #     if type(idx) is list and len(idx) > 1 and type(idx[0]) is list:
    #         state: List[Optional[torch.Tensor]] = [None for _ in range(self.n_layer * 3)]
    #         B = len(idx)
    #         for i in range(self.n_layer):
    #             state[i*3+0] = torch.zeros((B, self.n_embd), dtype=torch.float16, requires_grad=False, device="cuda")
    #             state[i*3+1] = torch.zeros((B, self.n_embd // self.head_size, self.head_size, self.head_size), dtype=torch.float, requires_grad=False, device="cuda")
    #             state[i*3+2] = torch.zeros((B, self.n_embd), dtype=torch.float16, requires_grad=False, device="cuda")
    #         return self.forward_prefill(B, idx, state, full_output)
    #     else:
    #         if state == None:
    #             # determine state size
    #             state = [None for _ in range(self.n_layer * 3)]
    #             for i in range(self.n_layer):
    #                 state[i*3+0] = torch.zeros(self.n_embd, dtype=torch.float16, requires_grad=False, device="cuda")
    #                 state[i*3+1] = torch.zeros((self.n_embd // self.head_size, self.head_size, self.head_size), dtype=torch.float, requires_grad=False, device="cuda")
    #                 state[i*3+2] = torch.zeros(self.n_embd, dtype=torch.float16, requires_grad=False, device="cuda")
            
    #         if isinstance(idx, List):
    #             if len(idx) > 1:
    #                 return self.forward_seq(idx, state, full_output)
    #             elif len(idx) == 1 and type(idx[0]) is list:
    #                 return self.forward_one_batch(idx[0], state)
    #             else:
    #                 assert isinstance(idx[0], int)
    #                 return self.forward_one(idx[0], state)
    #         else:
    #             return self.forward_one(idx, state)
    
    # @MyFunction
    def get_state(self, state:Optional[List[torch.Tensor]], B: int=0) -> List[torch.Tensor]:
        if state is not None:
            return state
        elif B > 0:
            state = []
            
            for i in range(self.n_layer):
                state.append(torch.zeros((B, self.n_embd), dtype=torch.float16, device="cuda"))
                state.append(torch.zeros((B, self.n_embd // self.head_size, self.head_size, self.head_size), dtype=torch.float, device="cuda"))
                state.append(torch.zeros((B, self.n_embd), dtype=torch.float16, device="cuda"))
        else:
            state = []
            for i in range(self.n_layer):
                state.append(torch.zeros(self.n_embd, dtype=torch.float16, device="cuda"))
                state.append(torch.zeros((self.n_embd // self.head_size, self.head_size, self.head_size), dtype=torch.float, device="cuda"))
                state.append(torch.zeros(self.n_embd, dtype=torch.float16, device="cuda"))
        return state

    # @torch.compile(mode="max-autotune")
    def forward1(self, idx: int, state: Optional[List[torch.Tensor]]=None):
        state = self.get_state(state, B=0)
        return self.forward_one(idx, state)
    
    # @MyOverload
    def forwardl(self, idx: List[int], state: Optional[List[torch.Tensor]]=None, batched:bool=False, full_output:bool=False):
        if batched:
            B = len(idx)
            state = self.get_state(state, B=B)
            return self.forward_one_batch(idx, state)
        else:
            state = self.get_state(state, B=0)
            return self.forward_seq(idx, state, full_output=full_output)
    
    # @MyOverload
    def forwardll(self, idx: List[List[int]], state: Optional[List[torch.Tensor]]=None, right_pad:bool=True, full_output:bool=False):
        att_mask: Optional[torch.Tensor] = None
        B = len(idx)
        if right_pad:
            # right_pad
            state = self.get_state(None, B=B)
            TT = max(len(h) for h in idx)
            idx1 = torch.full((B, TT), 0, dtype=torch.long, device="cuda")
            Tm = torch.zeros(B, dtype=torch.int, device="cuda")
            for i in range(B):
                t = TT - len(idx[i])
                Tm[i] = t
                idx1[i, t:] = torch.tensor(idx[i], dtype=torch.long, device="cuda")
            att_mask = (torch.arange(TT, device="cuda") < Tm.unsqueeze(1)).unsqueeze(-1)
            return self.forward_prefill(B, idx1, state, att_mask=att_mask, full_output=full_output)
        else:
            state = self.get_state(state, B=B)
            idx1 = torch.tensor(idx)
            return self.forward_prefill(B, idx1, state, att_mask=None, full_output=full_output)
    
    # @MyOverload
    def forwardt(self, idx: torch.Tensor, state: Optional[List[torch.Tensor]]=None, att_mask:Optional[torch.Tensor]=None, full_output:bool=False):
        B = idx.size(0)
        state = self.get_state(state, B=B)
        return self.forward_prefill(B, idx, state, att_mask=att_mask, full_output=full_output)

    # @MyOverload
    # def forward(self, idx: torch.Tensor, state: Optional[List[torch.Tensor]], ):
    #     state = self.get_state(state)
    #     return self.forward_seq(idx, state)

    @torch.compile(mode="max-autotune-no-cudagraphs")
    def forward_one(self, idx:int, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            # v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx1 = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                if i==0:
                    xx2, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one_0(self.n_head, self.head_size, xx1, state[i*3+0], state[i*3+1],
                        z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                        z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                        z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                        z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                        z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                else:
                    xx2, state[i*3+0], state[i*3+1] = RWKV_x070_TMix_one_1(self.n_head, self.head_size, xx1, state[i*3+0], v_first, state[i*3+1],
                        z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                        z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                        z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                        z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                        z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx2

                xx3 = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx4, state[i*3+2] = RWKV_x070_CMix_one(xx3, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx4
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state
    
    @MyFunction
    def forward_one_batch(self, idx:List[int], state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]
            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'
                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one_batch(i, len(idx), self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                xx, state[i*3+2] = RWKV_x070_CMix_one(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state
        
    @MyFunction
    def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = RWKV_x070_CMix_seq(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state
        
    @MyFunction
    def forward_prefill(self, B: int, idx:torch.Tensor, state:List[torch.Tensor], att_mask:Optional[torch.Tensor]=None, full_output:bool=False):
        TT = idx.size(1)
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]
            if att_mask is not None: x.masked_fill_(att_mask, 0)
            v_first = torch.empty_like(x)

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'
                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                if att_mask is not None: xx.masked_fill_(att_mask, 0)
                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_prefill(i, B, TT, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], att_mask)
                x = x + xx
                if att_mask is not None: x.masked_fill_(att_mask, 0)

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                if att_mask is not None: xx.masked_fill_(att_mask, 0)

                xx, state[i*3+2] = RWKV_x070_CMix_prefill(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
                if att_mask is not None: x.masked_fill_(att_mask, 0)

            if not full_output: 
                x = x[:,-1,:]
                x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            else:
                x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
                if att_mask is not None: x.masked_fill_(att_mask, 0)
            x = x @ z['head.weight']
            
            return x, state

########################################################################################################

@torch.compile(mode="max-autotune")
def RWKV_x070_TMix_one_0(H:int, N:int, x, x_prev, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(H,N,1))

    xx = torch.nn.functional.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, x, state, v

@torch.compile(mode="max-autotune")
def RWKV_x070_TMix_one_1(H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(H,N,1))

    xx = torch.nn.functional.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, x, state

@MyStatic
def RWKV_x070_TMix_one_batch(layer_id: int, B: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(B,H,N), dim=-1, p=2.0).view(B,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float()))

    vk = v.view(B,H,N,1) @ k.view(B,H,1,N)
    ab = (-kk).view(B,H,N,1) @ (kk*a).view(B,H,1,N)
    state = state * w.view(B,H,1,N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(B,H,N,1))

    xx = torch.nn.functional.group_norm(xx.view(B,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,H*N)    
    xx = xx + ((r * k * r_k).view(B,H,N).sum(dim=-1, keepdim=True) * v.view(B,H,N)).view(B,H*N)
    return (xx * g) @ O_, x, state, v_first

@MyStatic
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    T = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = -torch.nn.functional.softplus(-(w0 + w)) - 0.5

    xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a)

    xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return (xx * g) @ O_, x[-1,:], state, v_first

@MyStatic
def RWKV_x070_TMix_prefill(layer_id: int, B:int, T:int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, att_mask: Optional[torch.Tensor]):
    xx = torch.cat((x_prev.unsqueeze(1), x[:,:-1,:]), dim=1) - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
    if att_mask is not None: kk.masked_fill_(att_mask, 0)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = -torch.nn.functional.softplus(-(w0 + w)) - 0.5

    xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a)

    xx = torch.nn.functional.group_norm(xx.view(B*T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,T,H*N)
    xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
    oo = (xx * g) @ O_ # [B, T, C]
    return oo, x[:,-1,:], state, v_first

@MyStatic
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x

@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x[-1,:]

@MyStatic
def RWKV_x070_CMix_prefill(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev.unsqueeze(1), x[:,:-1,:]), dim=1) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    o = k @ V_
    return o, x[:,-1,:]
