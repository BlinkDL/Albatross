########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

import numpy as np
from collections import deque

np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time, random, json, math, gc
from tqdm import tqdm
from torch.nn import functional as F
import flashinfer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

########################################################################################################

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64

# model download: https://huggingface.co/BlinkDL/rwkv7-g1

args.MODEL_NAME = "../models/rwkv7-g0a-7.2b-20250829-ctx4096"

print(f"\nUsing CUDA fp16. Loading {args.MODEL_NAME} ...\n")

from reference.rwkv7 import RWKV_x070

model = RWKV_x070(args)

from reference.utils import TRIE_TOKENIZER, sampler_simple_batch, sampler_simple

tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

########################################################################################################


def continuous_batching(
    model,
    tokenizer,
    inputs,
    stop_tokens,
    max_generate_tokens,
    batch_size,
    pad_zero=True,
    temperature=1,
    top_k=50,
    top_p=0.3,
    alpha_presence=0.5,
    alpha_frequency=0.5,
    alpha_decay=0.996,
):

    assert len(inputs) >= batch_size, "The number of inputs must be greater than or equal to the batch size"

    STOP_TOKENS = stop_tokens
    MAX_GENERATE_TOKENS = max_generate_tokens
    BATCH_SIZE = batch_size
    PAD_ZERO = pad_zero

    total_inputs = len(inputs)

    print("Preparing inputs...")
    encoded_inputs = []
    for prompt in inputs:
        input_token = tokenizer.encode(prompt)
        if PAD_ZERO:
            input_token = [0] + input_token
        encoded_inputs.append((prompt, input_token))
    inputs = deque(encoded_inputs)

    prompt_idx = 0
    states = model.generate_zero_state(BATCH_SIZE)
    task_pool = []
    for i in range(BATCH_SIZE):
        prompt, input_token = inputs.popleft()
        task_pool.append(
            {
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "input_token": input_token,
                "state_pos": i,
                "last_logits": None,
                "generated_tokens": [],
                "new_token": None,
            }
        )
        prompt_idx += 1

    pbar = tqdm(
        total=total_inputs, desc="Processing", unit=" Sequence", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    occurrence = torch.zeros((BATCH_SIZE, args.vocab_size), dtype=torch.float32, device=model.z["head.weight"].device)
    no_penalty_token_ids = set([33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])  # ' \t0123456789'
    alpha_presence_vector = torch.zeros((BATCH_SIZE, args.vocab_size), dtype=torch.float32, device=model.z["head.weight"].device)

    outputs = []
    while True:
        accomplished_task_indices = []
        state_slots_to_remove = set()
        for task_idx, task in enumerate(task_pool):
            if len(task["input_token"]) == 0:  # this means the task is in decoding stage

                new_token = task["new_token"]

                token_in_stop = new_token in STOP_TOKENS
                length_exceed = len(task["generated_tokens"]) >= MAX_GENERATE_TOKENS

                if not token_in_stop:
                    task["input_token"].append(new_token)
                    task["generated_tokens"].append(new_token)

                if token_in_stop or length_exceed:  # task is finished
                    outputs.append(
                        {
                            "prompt_idx": task["prompt_idx"],
                            "prompt": task["prompt"],
                            "generated_tokens": task["generated_tokens"],
                        }
                    )
                    pbar.update(1)

                    if len(inputs) > 0:  # add a new task
                        prompt, input_token = inputs.popleft()
                        task_pool[task_idx] = {
                            "prompt_idx": prompt_idx,
                            "prompt": prompt,
                            "input_token": input_token,
                            "state_pos": task["state_pos"],
                            "last_logits": None,
                            "generated_tokens": [],
                            "new_token": None,
                        }
                        prompt_idx += 1
                        states[0][:, :, task["state_pos"], :] = 0
                        states[1][:, task["state_pos"], :, :] = 0
                        occurrence[task["state_pos"], :] = 0
                        alpha_presence_vector[task["state_pos"], :] = 0
                    else:  # no more new task
                        accomplished_task_indices.append(task_idx)
                        state_slots_to_remove.add(task["state_pos"])
                else:  # task is not finished, update the occurrence and alpha_presence_vector
                    www = 0.0 if new_token in no_penalty_token_ids else 1.0
                    occurrence[task["state_pos"], new_token] += www
                    alpha_presence_vector[task["state_pos"], new_token] = alpha_presence

        if accomplished_task_indices:
            sorted_slots_to_remove = sorted(list(state_slots_to_remove), reverse=True)

            for slot in sorted_slots_to_remove:
                part1_s0 = states[0][:, :, :slot, :]
                part2_s0 = states[0][:, :, slot + 1 :, :]
                states[0] = torch.cat([part1_s0, part2_s0], dim=2)

                part1_s1 = states[1][:, :slot, :, :, :]
                part2_s1 = states[1][:, slot + 1 :, :, :, :]
                states[1] = torch.cat([part1_s1, part2_s1], dim=1)

                occ_part1 = occurrence[:slot, :]
                occ_part2 = occurrence[slot + 1 :, :]
                occurrence = torch.cat([occ_part1, occ_part2], dim=0)

                alpha_presence_part1 = alpha_presence_vector[:slot, :]
                alpha_presence_part2 = alpha_presence_vector[slot + 1 :, :]
                alpha_presence_vector = torch.cat([alpha_presence_part1, alpha_presence_part2], dim=0)

            # Remove the accomplished tasks from the task_pool
            for task_idx in sorted(accomplished_task_indices, reverse=True):
                del task_pool[task_idx]

            # Re-index the state_pos for all remaining tasks
            remaining_slots = sorted([t["state_pos"] for t in task_pool])
            pos_map = {old_pos: new_pos for new_pos, old_pos in enumerate(remaining_slots)}
            for task in task_pool:
                task["state_pos"] = pos_map[task["state_pos"]]

        if len(task_pool) == 0:
            break

        max_state_idx = max(task["state_pos"] for task in task_pool)
        next_tokens = [None] * (max_state_idx + 1)
        for task in task_pool:
            next_tokens[task["state_pos"]] = [task["input_token"].pop(0)]

        out = model.forward_batch(next_tokens, states)

        # repetition penalty
        occurrence *= alpha_decay
        out -= alpha_presence_vector + occurrence * alpha_frequency

        if temperature == 0:  # greedy sampling
            temperature = 1.0
            top_k = 1
        if temperature != 1.0:
            out /= temperature

        new_tokens = flashinfer.sampling.top_k_top_p_sampling_from_logits(out, top_k, top_p)
        new_tokens = new_tokens.tolist()

        for task in task_pool:
            state_pos = task["state_pos"]
            tok = new_tokens[state_pos]
            task["new_token"] = tok

        for task in task_pool:
            state_pos = task["state_pos"]
            task["new_token"] = new_tokens[state_pos]

    pbar.close()
    print("Decoding outputs...")
    for output in outputs:
        generated_tokens = output["generated_tokens"]
        while True:
            if len(generated_tokens) == 0:
                output["generated_text"] = ""
                break
            try:
                text = tokenizer.decode(generated_tokens)
                output["generated_text"] = text
                break
            except:
                generated_tokens = generated_tokens[:-1]
    outputs = sorted(outputs, key=lambda x: x["prompt_idx"])
    return outputs


if __name__ == "__main__":

    inputs = [f"User: 为什么 {i} 是一个有趣的数字?\n\nAssistant:" for i in range(256)]
    TEMPERATURE = 1
    STOP_TOKENS = [0, 261]
    MAX_GENERATE_TOKENS = 2048
    BATCH_SIZE = 128
    TOP_K = 50
    TOP_P = 0.3
    PAD_ZERO = False
    # Set temperature=0 or top_k=1 for greedy sampling

    outputs = continuous_batching(model, tokenizer, inputs, STOP_TOKENS, MAX_GENERATE_TOKENS, BATCH_SIZE, PAD_ZERO, TEMPERATURE, TOP_K, TOP_P)

    print(outputs[-1]["generated_text"])
