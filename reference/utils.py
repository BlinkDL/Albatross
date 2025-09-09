########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

import torch
from torch.nn import functional as F

# MyModule = torch.jit.ScriptModule
# MyFunction = torch.jit.script_method
MyStatic = torch.jit.script
# MyModule = nn.Module
# def __nop(ob): return ob
# MyFunction = __nop
# MyStatic = __nop

@MyStatic
# @torch.compile("max-autotune")
def sample_logits(logits, temperature:float=1.0, top_p:float=1.0, top_k:int=0):
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    
    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0

    if top_p < 1:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0

        if top_p > 0:
            idx = torch.where(probs == cutoff)[0]
            if len(idx) > 0:
                probs[idx] = cutoff + (top_p - torch.sum(probs).item()) / len(idx)
                # assert abs(torch.sum(probs).item() - top_p) < 1e-6
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    result_list: list[int] = torch.multinomial(probs, num_samples=1).tolist()
    return result_list

def sample_logits_batch(logits, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0):
    """
    根据 logits 进行采样，支持批量输入和多种采样策略。
    
    参数:
        logits (Tensor): 输入 logits，形状为 (batch_size, vocab_size)。
        temperature (float): 温度参数，默认为 1.0。
        top_p (float): Top-p 截断阈值，默认为 1.0（无截断）。
        top_k (int): Top-k 截断数量，默认为 0（无截断）。
    
    返回:
        Tensor: 采样结果，形状为 (batch_size,)。
    """
    # 应用温度缩放
    if temperature != 1.0:
        logits = logits / temperature
    
    # 应用 top-k 过滤
    if top_k > 0:
        # 获取每个样本中 top-k 位置的值
        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        
        # 创建掩码，将非 top-k 的位置设为负无穷
        min_values = top_k_logits[:, -1:]  # 每个样本中第 k 小的值
        logits = torch.where(logits >= min_values, logits, torch.full_like(logits, float('-inf')))
    
    # 应用 top-p (nucleus) 过滤
    if top_p < 1.0:
        # 对 logits 进行排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 找到累积概率超过 top_p 的位置
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # 至少保留第一个（概率最高的）token
        sorted_indices_to_remove[:, 0] = False
        
        # 将需要移除的位置设为负无穷
        sorted_logits.masked_fill_(sorted_indices_to_remove, float('-inf'))
        
        # 将排序后的 logits 转换回原始顺序
        logits = torch.gather(sorted_logits, dim=-1, index=sorted_indices.argsort(dim=-1))
    
    # 从处理后的 logits 中采样
    probs = F.softmax(logits, dim=-1)
    samples = torch.multinomial(probs, 1).squeeze(-1)
    
    return samples


class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)
    
    def add(self, key:bytes, idx:int=0, val=None):
        if(idx == len(key)):
            if(val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]
        
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret

class TRIE_TOKENIZER():
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k,v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src:bytes):
        idx:int = 0
        tokens = []
        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))            
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))
    def batch_encode(self, src):
        return [self.encodeBytes(s.encode("utf-8")) for s in src]

    def decode(self, tokens):
        try:
            return self.decodeBytes(tokens).decode('utf-8')
        except:
            return '\ufffd' # bad utf-8
        
    def batch_decode(self, src):
        return [self.decode([int(s)]) for s in src]

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()
