import os
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import math
from task import input_t, output_t
from torch.profiler import profile, record_function, ProfilerActivity
import triton.language as tl
import triton


class Expert(nn.Module):
    def __init__(self, config: Dict, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden: int = config["d_hidden"]
        self.d_expert: int = config["d_expert"] if d_expert is None else d_expert

        self.W_gate = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_up = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_down = nn.Linear(self.d_expert, self.d_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.W_gate(x))
        out = self.W_down(gate * self.W_up(x))
        return out


class MoEGate(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.top_k: int = config["n_experts_per_token"]
        self.num_experts: int = config["n_routed_experts"]
        self.d_hidden: int = config["d_hidden"]

        self.W_g = nn.Linear(self.d_hidden, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # logits will be of shape [batch, seq, num_experts]
        # and we apply softmax on the last dimension i.e. the number of experts
        logits = self.W_g(x)
        scores = logits.softmax(dim=-1)
        # torch.topk returns a namedtuple of (values, indices)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        return topk_indices, topk_scores


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            Expert(config)
            for _ in range(config["n_routed_experts"])
        ])
        self.gating_network = MoEGate(config)
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(config=config, d_expert=shared_expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_output = self.shared_expert(x)
        expert_indices, expert_scores = self.gating_network(x)
        batch_size, seq_len, hidden_dim = x.shape
        orig_shape = x.shape
        x_flat = x.view(-1, hidden_dim)
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_scores.view(-1, 1)
        routed_output_flat = self.moe_infer(x_flat,
                                            flat_expert_indices,
                                            flat_expert_weights)

        routed_output = routed_output_flat.view(*orig_shape)
        return routed_output + shared_output

    @torch.no_grad()
    def moe_infer(self,
                  x: torch.Tensor,
                  flat_expert_indices: torch.Tensor,
                  flat_expert_weights: torch.Tensor
                 ) -> torch.Tensor:
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount().cpu().numpy()
        tokens_per_expert = counts.cumsum()
        num_per_tok = self.config["n_experts_per_token"]
        token_idxs = idxs // num_per_tok
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue

            expert = self.experts[expert_id]
            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idxs]
            expert_out    = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce='sum'
            )

        return expert_cache


def ref_custom_kernel(data: input_t) -> output_t:  # type: ignore
    """
    Reference implementation of DeepSeek-style Mixture of Experts using PyTorch.
    
    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_dim]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
            
    Returns:
        Tuple containing:
            - output: Processed tensor [batch_size, seq_len, d_model]
            - aux_data: Dictionary with auxiliary data
    """
    input_tensor, weights, config = data
    num_experts = config["n_routed_experts"]
    moe = MoE(config)

    # Fill in the given weights of the model
    moe.gating_network.W_g.weight = nn.Parameter(weights['router.weight'])

    for i in range(num_experts):
        gate_proj_weight = weights[f'experts.{i}.0.weight']
        up_proj_weight = weights[f'experts.{i}.1.weight']
        down_proj_weight = weights[f'experts.{i}.2.weight']

        # Transpose weights to match expected shape for nn.Linear
        moe.experts[i].W_gate.weight = nn.Parameter(gate_proj_weight.t()) # type: ignore
        moe.experts[i].W_up.weight = nn.Parameter(up_proj_weight.t()) # pyright: ignore[reportAttributeAccessIssue]
        moe.experts[i].W_down.weight = nn.Parameter(down_proj_weight.t()) # type: ignore

    moe.shared_expert.W_gate.weight = nn.Parameter(weights['shared_experts.0.weight'].t())
    moe.shared_expert.W_up.weight = nn.Parameter(weights['shared_experts.1.weight'].t())
    moe.shared_expert.W_down.weight = nn.Parameter(weights['shared_experts.2.weight'].t())

    output = moe(input_tensor)

    return output


def generate_input(
    d_hidden: int,
    dexpert: int,
    nroutedexperts: int,
    nsharedexperts: int,
    nexpertspertoken: int,
    bs: int,
    seqlen: int,
    seed: int
):

    # Really dumb but for now _ isn't parsing correctly.
    d_hidden = d_hidden
    d_expert = dexpert
    n_routed_experts = nroutedexperts
    n_shared_experts = nsharedexperts
    n_experts_per_token = nexpertspertoken
    batch_size = bs
    seq_len = seqlen

    config = {
        "d_hidden": d_hidden,
        "d_expert": d_expert,
        "n_routed_experts": n_routed_experts,
        "n_shared_experts": n_shared_experts,
        "n_experts_per_token": n_experts_per_token,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    num_experts = n_routed_experts
    expert_dim = d_expert
    weights = {}

    input_tensor = torch.randn(
        (batch_size, seq_len, d_hidden),
        device='cuda',
        dtype=torch.float16,
        generator=gen
    ).contiguous()

    # Initialize router weights
    weights['router.weight'] = torch.randn(
        (num_experts, d_hidden),
        device="cuda",
        dtype=torch.float16,
        generator=gen
    ) / math.sqrt(d_hidden)

    for i in range(num_experts):
        weights[f'experts.{i}.0.weight'] = torch.randn(
            (d_hidden, expert_dim),
            device='cuda',
            dtype=torch.float16,
            generator=gen
        ) / math.sqrt(expert_dim)

        weights[f'experts.{i}.1.weight'] = torch.randn(
            (d_hidden, expert_dim),
            device='cuda',
            dtype=torch.float16,
            generator=gen
        ) / math.sqrt(expert_dim)

        weights[f'experts.{i}.2.weight'] = torch.randn(
            (expert_dim, d_hidden),
            device='cuda',
            dtype=torch.float16,
            generator=gen
        ) / math.sqrt(d_hidden)
    
    weights['shared_experts.0.weight'] = torch.randn(
        (d_hidden, expert_dim * n_shared_experts),
        device='cuda',
        dtype=torch.float16,
        generator=gen
    ) / math.sqrt(expert_dim * n_shared_experts)
    weights['shared_experts.1.weight'] = torch.randn(
        (d_hidden, expert_dim * n_shared_experts),
        device='cuda',
        dtype=torch.float16,
        generator=gen
    ) / math.sqrt(expert_dim * n_shared_experts)
    weights['shared_experts.2.weight'] = torch.randn(
        (expert_dim * n_shared_experts, d_hidden),
        device='cuda',
        dtype=torch.float16,
        generator=gen
    ) / math.sqrt(d_hidden)

    return (input_tensor, weights, config)


def configs():
    return [
        triton.Config({
            "BLOCKSIZE_M": BM, "BLOCKSIZE_N": BN,
            "BLOCKSIZE_K": BK, "GROUPSIZE_M": GS,
        }, num_warps=4, num_stages=3)
        for BM in [32]
        for BN in [64]
        for BK in [64]
        for GS in [4]

    ]

@triton.autotune(
    configs= configs(),
    key= ["num_expert_token_pairs", "d_hidden", "d_expert"],
    reset_to_zero=["out"],
)
@triton.jit
def expert_kernel(
    W_gate,
    W_up,
    W_down,
    x,
    out,
    max_tokens_per_expert,
    d_hidden,
    d_expert,
    sorted_expert_scores_ptr,
    token_idxs_ptr,
    expert_counts_ptr,
    expert_offsets_ptr,
    BLOCKSIZE_M: tl.constexpr,
    BLOCKSIZE_N: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    GROUPSIZE_M: tl.constexpr,
):

    # W_gate : (n_routed_experts, d_hidden, d_expert) : (d_hidden * d_expert, d_expert, 1)
    # W_up : (n_routed_experts, d_hidden, d_expert) : (d_hidden * d_expert, d_expert, 1)
    # W_down : (n_routed_experts, d_expert, d_hidden) : (d_expert * d_hidden, d_hidden, 1)
    # x (batched_input) layout: (num_experts, max_tokens_per_Expert, d_hidden) : (max_tokens_per_expert * d_hidden, d_hidden, 1)
    # out (output_tensor) layout: (num_experts, max_tokens_per_Expert, d_hidden) : (max_tokens_per_expert * d_hidden, d_hidden, 1)
    # max_tokens_per_expert : M
    # d_hidden : N
    # d_expert : intermediate
    # sorted_exp_scores : (batch_size * seq_len * n_experts_per_token, ) : (1, 1)
    # token_idxs : (batch_size * seq_len * n_experts_per_token, )
    # expert_counts : (num_experts, ) : (1, )
    # grid : (n_experts, cdiv(max_tokens_per_expert, BM) * cdiv(d_expert, BN))

    expert_id = tl.program_id(axis=0)
    expert_count = tl.load(expert_counts_ptr + expert_id)
    expert_start_offset = tl.load(expert_offsets_ptr + expert_id)

    pid_expert_flat = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(max_tokens_per_expert, BLOCKSIZE_M)
    num_pid_n = tl.cdiv(d_expert, BLOCKSIZE_N)
    num_pid_in_group = GROUPSIZE_M * num_pid_n
    group_id = pid_expert_flat // num_pid_in_group
    first_pid_m = group_id * GROUPSIZE_M
    group_size_m = min(GROUPSIZE_M, num_pid_m - first_pid_m)
    pid_m = first_pid_m + (pid_expert_flat % group_size_m)
    pid_n = (pid_expert_flat % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCKSIZE_M
    start_n = pid_n * BLOCKSIZE_N

    offs_am = start_m + tl.arange(0, BLOCKSIZE_M)
    offs_b0n = start_n + tl.arange(0, BLOCKSIZE_N)           # W_gate
    offs_b1n = start_n + tl.arange(0, BLOCKSIZE_N)           # W_up
    offs_b2n = start_n + tl.arange(0, BLOCKSIZE_N)           # W_down
    m_mask = offs_am < max_tokens_per_expert
    offs_am = tl.where(m_mask, offs_am, 0)
    offs_b0n = tl.where(offs_b0n < d_expert, offs_b0n, 0)
    offs_b1n = tl.where(offs_b1n < d_expert, offs_b1n, 0)
    offs_b2n = tl.where(offs_b2n < d_expert, offs_b2n, 0)
    offs_cm = start_m + tl.arange(0, BLOCKSIZE_M)
    c_mask = offs_cm < expert_count
    offs_cm = tl.where(c_mask, offs_cm, 0)
    offs_k = tl.arange(0, BLOCKSIZE_K)
    token_ids = tl.load(token_idxs_ptr + expert_start_offset + offs_cm)
    

    
    x_expert_offset = expert_id * max_tokens_per_expert * d_hidden
    weight_expert_offset = expert_id * d_hidden * d_expert
    
    # Pointer Arithmetic
    expert_scores = tl.load(sorted_expert_scores_ptr + expert_start_offset + offs_cm)
    a_ptrs = x + x_expert_offset + offs_am[:, None] * d_hidden + offs_k[None, :]               # [BM, BK]
    b0_ptrs = W_gate + weight_expert_offset + offs_k[:, None] * d_expert + offs_b0n[None, :]          # [BK, BN]
    b1_ptrs = W_up + weight_expert_offset + offs_k[:, None] * d_expert + offs_b1n[None, :]            # [BK, BN]
    b2_ptrs = W_down + weight_expert_offset + offs_b2n[:, None] * d_hidden + offs_k[None, :]          # [BN, BK]
    
    gate_acc = tl.zeros((BLOCKSIZE_M, BLOCKSIZE_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCKSIZE_M, BLOCKSIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(d_hidden, BLOCKSIZE_K)):
        a = tl.load(a_ptrs, mask= m_mask[:, None] & (offs_k[None, :] < d_hidden - k * BLOCKSIZE_K), other=0.0)
        b0 = tl.load(b0_ptrs, mask=offs_k[:, None] < d_hidden - k * BLOCKSIZE_K, other=0.0)
        
        gate_acc = tl.dot(a, b0, gate_acc, input_precision="ieee")         # [BM, BN]

        b1 = tl.load(b1_ptrs, mask=offs_k[:, None] < d_hidden - k * BLOCKSIZE_K, other=0.0)
        up_acc = tl.dot(a, b1, up_acc, input_precision="ieee")             # [BM, BN]
        a_ptrs += BLOCKSIZE_K
        b0_ptrs += BLOCKSIZE_K * d_expert
        b1_ptrs += BLOCKSIZE_K * d_expert

    #SiLu
    gate_acc = gate_acc * tl.sigmoid(gate_acc)
    temp = gate_acc * up_acc        # [BM,BN]
    temp = temp.to(tl.float16)
    
    for k in range(0, tl.cdiv(d_hidden,BLOCKSIZE_K)):
        b2 = tl.load(b2_ptrs,
                     mask= offs_k[None, :] < d_hidden - k * BLOCKSIZE_K, other=0.0
                    )
        v = tl.dot(temp,b2, input_precision="ieee")   # [BM, BK]
        v = v * expert_scores[:, None]               # [BM, BK] * [BM, ]

        # writing to the out matrix
        tl.atomic_add(
            out + token_ids[:, None] * d_hidden + (k*BLOCKSIZE_K + offs_k[None, :]),
            v,
            mask= c_mask[:, None] & (offs_k[None, :] < d_hidden - k * BLOCKSIZE_K),
        )
        b2_ptrs += BLOCKSIZE_K


def _stack_tensor(weights, key_tmpl: str, n):
    mats = [weights[key_tmpl.format(i)] for i in range(n)]
    return torch.stack(mats, dim=0).contiguous()

# config: {'d_hidden': 7168, 'd_expert': 2048, 'n_routed_experts': 32, 'n_shared_experts': 1, 'n_experts_per_token': 4, 'batch_size': 1, 'seq_len': 2048}
# input_tensor layout: (batch_size, seq_len, d_hidden) : (d_hidden * seq_len, d_hidden, 1)
# router.weight layout: (n_routed_experts,  d_hidden) : (d_hidden, 1)
# experts.{i}.0.weight layout: (d_hidden, d_expert) : (d_expert, 1)
# experts.{i}.1.weight layout: (d_hidden, d_expert) : (d_expert, 1)
# experts.{i}.2.weight layout: (d_expert, d_hidden) : (d_hidden, 1)
# shared_experts.{i}.0.weight layout: (d_hidden, d_expert * n_shared_experts) : (d_expert * n_shared_experts, 1)
# shared_experts.{i}.1.weight layout: (d_hidden, d_expert * n_shared_experts) : (d_expert * n_shared_experts, 1)
# shared_experts.{i}.2.weight layout: (d_expert * n_shared_experts, d_hidden) : (d_hidden, 1)
def moe_forward(input_tensor: torch.Tensor,
                weights: Dict[str, torch.Tensor],
                d_hidden: int,
                d_expert: int,
                n_routed_experts: int,
                n_shared_experts: int,
                n_experts_per_token: int,
                batch_size: int,
                seq_len: int,
                ) -> torch.Tensor:
    
    input_tensor = input_tensor.reshape(-1, d_hidden)
    num_tokens, _ = input_tensor.shape

    out = torch.zeros_like(input_tensor, device= input_tensor.device, dtype=torch.float16)

    num_experts = n_routed_experts

    # Combined weight tensors for all routed experts
    W_gate = _stack_tensor(weights, "experts.{}.0.weight", num_experts)
    # W_gate : (n_routed_experts, d_hidden, d_expert) : (d_hidden * d_expert, d_expert, 1) 
    W_up = _stack_tensor(weights, "experts.{}.1.weight", num_experts)
    # W_up : (n_routed_experts, d_hidden, d_expert) : (d_hidden * d_expert, d_expert, 1)
    W_down = _stack_tensor(weights, "experts.{}.2.weight", num_experts)
    # W_down : (n_routed_experts, d_expert, d_hidden) : (d_expert * d_hidden, d_hidden, 1)

    # Shared Expert/s
    shared_w_gate = weights['shared_experts.0.weight']
    shared_w_up = weights['shared_experts.1.weight']
    shared_w_down = weights['shared_experts.2.weight']

    shared_temp = F.silu(input_tensor @ shared_w_gate) * (input_tensor @ shared_w_up)     # (batch_size * seq_len, d_expert * n_shared_experts)
    shared_out = shared_temp @ shared_w_down        # (batch_Size * seq_len, d_hidden)

    logits = F.linear(input_tensor, weights["router.weight"])       # (batch_size * seq_len, n_routed_experts)
    # print(f"logits dtype : {logits.dtype}")
    scores = logits.softmax(dim=-1)     # (batch_size * seq_len, n_routed_experts)
    topk_scores, topk_indices = torch.topk(scores, k=n_experts_per_token, dim=-1, sorted=False)
    # topk_scores: (batch_size * seq_len, n_experts_per_token)
    # topk_indices: (batch_size * seq_len, n_experts_per_token)
    # print(f"topk_scores dtype : {topk_scores.dtype}")

    flat_expert_indices = topk_indices.view(-1)
    # flat_expert_indices -> (batch_size*seq_len*n_experts_per_token, ) : (1, )
    flat_expert_scores = topk_scores.view(-1)
    # flat_expert_scores -> (batch_size*seq_len*n_experts_per_token, 1) : (1, 1)
    idxs = flat_expert_indices.argsort()    
    # (batch_size * seq_len * n_experts_per_token, )
    sorted_expert_ids = flat_expert_indices[idxs]
    sorted_expert_scores = flat_expert_scores[idxs]
    token_idxs = idxs // n_experts_per_token  
    # (batch_size * seq_len * n_experts_per_token, )
    
    counts = sorted_expert_ids.to(torch.int32).bincount(minlength=num_experts)      # [num_experts]
    expert_offsets = torch.zeros_like(counts)
    expert_offsets[1:] = torch.cumsum(counts[:-1], dim=0)
    max_tokens_per_expert = counts.max().item()

    sorted_input = input_tensor[token_idxs]   
    # (batch_size * seq_len * n_experts_per_token, d_hidden)

    batched_input = torch.zeros(size=(num_experts, max_tokens_per_expert, d_hidden),  # type: ignore
                                device=input_tensor.device,
                                dtype= input_tensor.dtype)

    sorted_input_token_indices = torch.arange(sorted_input.shape[0], device=input_tensor.device)
    token_group_offset = expert_offsets[sorted_expert_ids]
    token_offset_inside_token_group = sorted_input_token_indices - token_group_offset
    batched_input[sorted_expert_ids, token_offset_inside_token_group, :] = sorted_input
    
    grid = lambda META: (num_experts, triton.cdiv(max_tokens_per_expert, META["BLOCKSIZE_M"]) * triton.cdiv(d_expert, META["BLOCKSIZE_N"]), )

    expert_kernel[grid](
        W_gate,
        W_up,
        W_down,
        batched_input,
        out,
        max_tokens_per_expert,
        d_hidden,
        d_expert,
        sorted_expert_scores,
        token_idxs,
        counts,
        expert_offsets,
    )
    
    return (shared_out + out).reshape(batch_size, seq_len, d_hidden)


def custom_kernel(data: input_t) -> output_t: # type: ignore

    input_tensor, weights, config = data
    output = moe_forward(input_tensor, weights, **config)

    return output # type: ignore


def profile_moe():
    # Configuration values
    dhidden = 7168
    dexpert = 2048
    nroutedexperts = 8
    nsharedexperts = 1
    nexpertspertoken = 4
    bs = 2
    seqlen = 8192
    seed = 81934

    # Generate input, weights, config
    input_tensor, weights, config = generate_input(
        dhidden,
        dexpert,
        nroutedexperts,
        nsharedexperts,
        nexpertspertoken,
        bs,
        seqlen,
        seed
    )

    def step():
        with torch.profiler.record_function("moe_fwd"):
            return custom_kernel((input_tensor, weights, config))
        
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule= schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            step()
            prof.step()
    torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    prof.export_chrome_trace("moe_profile.json")

def run_test(expect, actual, enabled=True):
    if enabled:
        passed = torch.allclose(expect, actual, atol=1e-2, rtol=1e-2)
        if passed:
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")
    else:
        icon = "⭕"
        print(f"\r  Disabled: {icon}  ")

def benchmark(fn, data, warmup: int = 10, iterations: int = 50):
    for _ in range(warmup):
        fn(data)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        fn(data)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations

if __name__ == "__main__":
    torch.cuda.set_device(0)
    dhidden = 7168
    dexpert = 2048
    nroutedexperts = 8
    nsharedexperts = 1
    nexpertspertoken = 4
    bs = 2
    seqlen = 8192
    seed = 81934

    input_tensor, weights, config = generate_input(
        dhidden,
        dexpert,
        nroutedexperts,
        nsharedexperts,
        nexpertspertoken,
        bs,
        seqlen,
        seed
    )
    print("Input tensor shape:", input_tensor.shape)
    # print("Weights dictionary keys:", list(weights.keys()))
    for key, value in weights.items():
        print(f"{key}: {value.shape}")
    print("Config dictionary:", config)

    my_moe_out = custom_kernel((input_tensor, weights, config))
    ref_moe_out = ref_custom_kernel((input_tensor, weights, config))
    diff = torch.abs(my_moe_out - ref_moe_out)
    print("Reference MoE output shape,dtype:", ref_moe_out.shape, ref_moe_out.dtype)
    print("My MoE output shape, dtype", my_moe_out.shape, my_moe_out.dtype)
    run_test(my_moe_out, ref_moe_out)
    print(f"Final difference: {diff}")

    # data = (input_tensor, weights, config)
    # latency_pytorch = benchmark(ref_custom_kernel, data, )
    # latency_triton = benchmark(custom_kernel, data, )

    # print(f"Approach ref latency: {latency_pytorch:.4f} ms")
    # print(f"Approach custom latency: {latency_triton:.4f} ms")
    # print(f"Speedup: {latency_pytorch / latency_triton:.2f}x")


    

    # profiel the custom kernel
    # profile_moe()
