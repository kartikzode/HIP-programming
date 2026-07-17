import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import math
from task import input_t, output_t
from torch.profiler import profile, record_function, ProfilerActivity
import triton.language as tl
import triton

## need to rework on the weight matrices, we have written 
# the pointer arithmatic assuming the shape as
# [d_hidden, d_expert] but it is wrong.


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
            "BLOCKSIZE_K": BK, "GROUPSIZE_M": 8,
        }, num_warps= w)
        for BM in [16]
        for BN in [16]
        for BK in [64]
        for w in [4]
    ]

@triton.autotune(
    configs= configs(),
    key= ["M", "N", "K"],
)
@triton.jit
def expert_kernel(
    exp_gate,
    exp_up,
    exp_down,
    x,
    out,
    d_hidden,
    d_expert,
    idxs_len,
    exp_token_idxs,
    exp_weights,
    BLOCKSIZE_M: tl.constexpr,
    BLOCKSIZE_N: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    GROUPSIZE_M: tl.constexpr,
):

    # expert_gate layout: (d_hidden, d_expert) : (d_expert, 1)
    # expert_up layout: (d_hidden, d_expert) : (d_expert, 1)
    # expert_down layout: (d_expert, d_hidden) : (d_hidden, 1)
    # x (input_tensor) layout: (batch_size * seq_len, d_hidden) : (d_hidden, 1)
    # out (output_tensor) layout: (batch_size * seq_len, d_hidden) : (d_hidden, 1)
    # d_hidden : N
    # d_expert : intermediate 
    # idxs_len : M
    # exp_token_idxs : indexes of tokens assigned to a particular expert
    # exp_weights: scores of that expert for teh above tokens
    # grid : (cdiv(idxs_len, BM), cdiv(d_expert, BN))

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(idxs_len, BLOCKSIZE_M)
    num_pid_n = tl.cdiv(d_expert, BLOCKSIZE_N)
    num_pid_in_group = GROUPSIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUPSIZE_M
    group_size_m = min(GROUPSIZE_M, num_pid_m - first_pid_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCKSIZE_M
    start_n = pid_n * BLOCKSIZE_N

    offs_am = start_m + tl.arange(0, BLOCKSIZE_M)
    offs_b0n = start_n + tl.arange(0, BLOCKSIZE_N)           # W_gate
    offs_b1n = start_n + tl.arange(0, BLOCKSIZE_N)           # W_up
    offs_b2n = start_n + tl.arange(0, BLOCKSIZE_N)           # W_down
    offs_am = tl.where(offs_am < idxs_len, offs_am, 0)
    offs_b0n = tl.where(offs_b0n < d_expert, offs_b0n, 0)
    offs_b1n = tl.where(offs_b1n < d_expert, offs_b1n, 0)
    offs_b2n = tl.where(offs_b2n < d_expert, offs_b2n, 0)
    offs_k = tl.arange(0, BLOCKSIZE_K)

    # laod the token ids and the expert scores to be fetched from X
    offs_m_mask = tl.arange(0, BLOCKSIZE_M) < idxs_len - start_m
    token_ids = tl.load(
        exp_token_idxs + offs_am
        )

    scales = tl.load(
        exp_weights + offs_am,
        )
    
    # Pointer Arithmetic
    a_ptrs = x + (token_ids[:, None] * d_hidden + offs_k[None, :])               # [BM, BK]
    b0_ptrs = exp_gate + offs_k[:, None] * d_expert + offs_b0n[None, :]          # [BK, BN]
    b1_ptrs = exp_up + offs_k[:, None] * d_expert + offs_b1n[None, :]            # [BK, BN]
    b2_ptrs = exp_down + offs_b2n[:, None] * d_hidden + offs_k[None, :]          # [BN, BK]
    
    gate_acc = tl.zeros((BLOCKSIZE_M, BLOCKSIZE_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCKSIZE_M, BLOCKSIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(d_hidden, BLOCKSIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < d_hidden - k * BLOCKSIZE_K, other=0.0)
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

    for k in range(0, tl.cdiv(d_hidden,BLOCKSIZE_K)):
        b2 = tl.load(b2_ptrs,
                     mask= offs_k[None, :] < d_hidden - k * BLOCKSIZE_K, other=0.0
                    ).to(tl.float32)
        v = tl.dot(temp,b2, input_precision="ieee")   # [BM, BK]
        v = v * scales[:, None]         # [BM, BK] * [BM, BK]

        # writing to the out matrix
        tl.atomic_add(
            out + token_ids[:, None] * d_hidden + (k*BLOCKSIZE_K + offs_k[None, :]),
            v,
            mask=offs_m_mask[:, None] & (offs_k[None, :] < d_hidden - k * BLOCKSIZE_K),
        )
        b2_ptrs += BLOCKSIZE_K

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
    out = torch.zeros_like(input_tensor, device= input_tensor.device, dtype=torch.float32)

    num_experts = n_routed_experts
    experts = [{} for _ in range(num_experts)]
    for i in range(num_experts):
        experts[i]["w_gate"] = weights[f"experts.{i}.0.weight"]     # (d_hidden, d_expert) : (d_expert, 1)
        experts[i]["w_up"] = weights[f"experts.{i}.1.weight"]       # (d_hidden, d_expert) : (d_expert, 1) 
        experts[i]["w_down"] = weights[f"experts.{i}.2.weight"]     # (d_expert, d_hidden) : (d_hidden, 1)


    shared_w_gate = weights['shared_experts.0.weight']
    shared_w_up = weights['shared_experts.1.weight']
    shared_w_down = weights['shared_experts.2.weight']

    shared_temp = F.silu(input_tensor @ shared_w_gate) * (input_tensor @ shared_w_up)     # (d_hidden, d_expert * n_shared_experts)
    shared_out = shared_temp @ shared_w_down        # (batch_Size * seq_len, d_hidden)

    logits = F.linear(input_tensor, weights["router.weight"])       # (batch_size * seq_len, n_routed_experts)
    scores = logits.softmax(dim=-1)     # (batch_size * seq_len, n_routed_experts)
    topk_scores, topk_indices = torch.topk(scores, k=n_experts_per_token, dim=-1, sorted=False)
    # topk_scores: token_id -> topk scores (batch_size * seq_len, n_experts_per_token)
    # topk_indices: token_id -> topk expert_id (batch_size * seq_len, n_experts_per_token)

    flat_expert_indices = topk_indices.view(-1)
    flat_expert_scores = topk_scores.view(-1, 1)
    idxs = flat_expert_indices.argsort()
    counts = flat_expert_indices.bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    token_idxs = idxs // n_experts_per_token

    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
            continue

        expert = experts[expert_id]
        exp_token_idxs = token_idxs[start_idx:end_idx]

        grid = lambda META: (triton.cdiv(len(exp_token_idxs), META["BLOCKSIZE_M"]) * triton.cdiv(d_expert, META["BLOCKSIZE_N"]), )
        expert_kernel[grid](
            expert["w_gate"],
            expert["w_up"],
            expert["w_down"],
            input_tensor,
            out,
            d_hidden,
            d_expert,
            len(exp_token_idxs),
            exp_token_idxs,
            flat_expert_scores[idxs[start_idx:end_idx]],
        )
    
    torch.cuda.synchronize()
    return (out + shared_out).reshape(batch_size, seq_len, d_hidden)


def custom_kernel(data: input_t) -> output_t: # type: ignore
    """
    Submission template for DeepSeek-style Mixture of Experts using PyTorch.
    
    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_size]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
            
    Returns:
        Tuple containing:
            - output: Processed tensor [batch_size, seq_len, d_model]
            - aux_data: Dictionary with auxiliary data
    """
    input_tensor, weights, config = data
    output = moe_forward(input_tensor, weights, **config)

    return output


def profile_moe():
    # Configuration values
    d_hidden = 512
    dexpert = 1024
    nroutedexperts = 4
    nsharedexperts = 1
    nexpertspertoken = 2
    bs = 1
    seqlen = 4
    seed = 42

    # Generate input, weights, config
    input_tensor, weights, config = generate_input(
        d_hidden,
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

def run_test(expect, actual, label, enabled=True):
    print(f"  {label}: ...", end="")
    if enabled:
        passed = torch.allclose(expect, actual.to(expect.dtype), atol=1.0)
        icon = "✅" if passed else "❌"
    else:
        icon = "⭕"
    print(f"\r  {label}: {icon}  ")

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
    print("Reference MoE output shape:", ref_moe_out.shape)
    print("My MoE output shape:", my_moe_out.shape)
    run_test(my_moe_out, ref_moe_out, "triton")
    print(f"Final difference: {diff}")

    

    # profiel the custom kernel
    # profile_moe()
