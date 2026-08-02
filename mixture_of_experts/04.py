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

def num_sms():
    return 16

def configs():
    return [
        triton.Config({
            "BLOCKSIZE_M": BM, "BLOCKSIZE_N": BN,
            "BLOCKSIZE_K": BK, "GROUPSIZE_M": GS,
            "NUM_SMS": num_sms(),
        }, num_warps=4, num_stages=3)
        for BM in [32]
        for BN in [64]
        for BK in [64]
        for GS in [4]

    ]


@triton.jit
def silu(x): return (x * tl.sigmoid(x.to(tl.float32))).to(tl.float16)


@triton.autotune(
    configs= configs(),
    key= [],
)
@triton.jit
def gate_up_silu_kernel(
    W_gate_ptrs,
    W_up_ptrs,
    x,
    routed_temp,
    expert_counts_ptr,
    expert_offsets_ptr,
    num_experts,
    d_hidden,
    d_expert,
    BLOCKSIZE_M: tl.constexpr,
    BLOCKSIZE_N: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    GROUPSIZE_M: tl.constexpr,
    NUM_SMS:tl.constexpr,
):
    # W_gate_ptrs : [n_routed_experts, ] addresses -->  (d_hidden, d_expert) : (d_expert, 1) 
    # W_up_ptrs : [n_routed_experts, ] addresses --> (d_hidden, d_expert) : (d_expert, 1)
    # x (grouped_input) layout: (expert_token_pairs, d_hidden) : (d_hidden, 1)
    # routed_temp layout: (expert_token_pairs, d_expert) : (d_expert, 1)
    # expert_counts_ptr : [num_experts, ]
    # expert_offsets_ptr : [num_experts, ]
    # expert_token_pairs : M
    # d_hidden : K
    # d_expert : N
    # grid : (NUM_SMS, )

    tile_idx = tl.program_id(axis=0)
    last_problem_end = 0

    for g in range(num_experts):
        gm = tl.load(expert_counts_ptr + g)
        row_offset = tl.load(expert_offsets_ptr + g)
        num_m_tiles = tl.cdiv(gm, BLOCKSIZE_M)
        num_n_tiles = tl.cdiv(d_expert, BLOCKSIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:

            W_gate_base = tl.load(W_gate_ptrs + g).to(tl.pointer_type(tl.float16))
            W_up_base = tl.load(W_up_ptrs + g).to(tl.pointer_type(tl.float16))

            tile_idx_in_gemm = tile_idx - last_problem_end
            num_pid_in_group = GROUPSIZE_M * num_n_tiles
            group_id = tile_idx_in_gemm // num_pid_in_group
            first_pid_m = group_id * GROUPSIZE_M
            group_size_m = min(GROUPSIZE_M, num_m_tiles - first_pid_m)
            pid_m = first_pid_m + (tile_idx_in_gemm % group_size_m)
            pid_n = (tile_idx_in_gemm % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCKSIZE_M
            start_n = pid_n * BLOCKSIZE_N

            offs_am = start_m + tl.arange(0, BLOCKSIZE_M)
            offs_bn = start_n + tl.arange(0, BLOCKSIZE_N)
            m_mask = offs_am < gm
            n_mask = offs_bn < d_expert
            offs_am = tl.where(m_mask, offs_am, 0)
            offs_bn = tl.where(n_mask, offs_bn, 0)
            offs_k = tl.arange(0, BLOCKSIZE_K)

            
            # Pointer Arithmetic
            a_ptrs = x + (row_offset + offs_am)[:, None] * d_hidden + offs_k[None, :]               # [BM, BK]
            b0_ptrs = W_gate_base + offs_k[:, None] * d_expert + offs_bn[None, :]          # [BK, BN]
            b1_ptrs = W_up_base + offs_k[:, None] * d_expert + offs_bn[None, :]            # [BK, BN]
            
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

            offs_cm = start_m + tl.arange(0, BLOCKSIZE_M)
            offs_cn = start_n + tl.arange(0, BLOCKSIZE_N)
            c_ptrs = routed_temp + (row_offset + offs_cm)[:, None] * d_expert + offs_cn[None, :] 
            c_mask = (offs_cm[:, None] < gm) & (offs_cn[None, :] < d_expert)
            tl.store(c_ptrs, temp, mask=c_mask)

            tile_idx += tl.num_programs(0)

        last_problem_end = last_problem_end + num_tiles


@triton.autotune(
    configs= configs(),
    key= [],
)
@triton.jit
def down_proj_kernel(
    W_down_ptrs,
    routed_temp,
    out,
    d_hidden,
    d_expert,
    sorted_expert_scores_ptr,
    dest_row_ptr,
    expert_counts_ptr,
    expert_offsets_ptr,
    num_experts,
    BLOCKSIZE_M: tl.constexpr,
    BLOCKSIZE_N: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    GROUPSIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    # W_down_ptrs : [n_routed_experts, ]  addresses --> (d_expert, d_hidden) : (d_hidden, 1)
    # routed_temp layout: (expert_token_pairs, d_expert) : (d_expert, 1)
    # out layout: (expert_token_pairs, d_hidden) : (d_hidden, 1)
    # expert_token_pairs : M
    # d_hidden : N
    # d_expert : K
    # sorted_exp_scores : (batch_size * seq_len * n_experts_per_token, ) : (1, 1)
    # dest_row_ptr : (batch_size * seq_len * n_experts_per_token, )
    # expert_counts : (num_experts, ) : (1, )
    # expert_offsets : (num_experts, ) : (1, )
    # grid : (NUM_SMS, )

    tile_idx = tl.program_id(0)
    last_problem_end = 0
 
    for g in range(num_experts):
        gm = tl.load(expert_counts_ptr + g)
        row_offset = tl.load(expert_offsets_ptr + g)
        num_m_tiles = tl.cdiv(gm, BLOCKSIZE_M)
        num_n_tiles = tl.cdiv(d_hidden, BLOCKSIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
 
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            W_down_base = tl.load(W_down_ptrs + g).to(tl.pointer_type(tl.float16))

            tile_idx_in_gemm = tile_idx - last_problem_end  
            num_pid_in_group = GROUPSIZE_M * num_n_tiles
            group_id = tile_idx_in_gemm // num_pid_in_group
            first_pid_m = group_id * GROUPSIZE_M
            group_size_m = min(GROUPSIZE_M, num_m_tiles - first_pid_m)
            pid_m = first_pid_m + (tile_idx_in_gemm % group_size_m)
            pid_n = (tile_idx_in_gemm % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCKSIZE_M
            start_n = pid_n * BLOCKSIZE_N

            offs_am = start_m + tl.arange(0, BLOCKSIZE_M)           
            offs_bn = start_n + tl.arange(0, BLOCKSIZE_N)           # W_down
            m_mask = offs_am < gm
            n_mask = offs_bn < d_hidden
            offs_am = tl.where(m_mask, offs_am, 0)
            offs_bn = tl.where(n_mask, offs_bn, 0)
            offs_k = tl.arange(0, BLOCKSIZE_K)

            dest_rows = tl.load(dest_row_ptr + row_offset + offs_am, mask=m_mask, other=0)
            expert_scores = tl.load(sorted_expert_scores_ptr + row_offset + offs_am, mask=m_mask, other= 0.0)

            # Pointer Arithmetic
            a_ptrs = routed_temp + (row_offset + offs_am)[:, None] * d_expert + offs_k[None, :]               # [BM, BK]         # [BK, BN]
            b_ptrs = W_down_base + offs_k[:, None] * d_hidden + offs_bn[None, :]                    # [BN, BK]

            acc = tl.zeros((BLOCKSIZE_M,BLOCKSIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(d_expert, BLOCKSIZE_K)):
                a = tl.load(a_ptrs, mask= m_mask[:, None] & (offs_k[None, :] < d_expert - k * BLOCKSIZE_K), other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < d_expert - k * BLOCKSIZE_K, other=0.0)
                
                acc = tl.dot(a, b, acc, input_precision="ieee")         # [BM, BN]

                a_ptrs += BLOCKSIZE_K
                b_ptrs += BLOCKSIZE_K * d_hidden

            acc = acc * expert_scores[:, None]            # [BM, BK] * [BM, ]

            tl.store(
                out + dest_rows[:, None] * d_hidden + offs_bn[None, :],
                acc,
                mask= m_mask[:, None] & n_mask[None, :],
            )

            tile_idx += tl.num_programs(0)
        last_problem_end = last_problem_end + num_tiles

def _weight_ptrs(weights: Dict[str, torch.Tensor], key_tmpl: str, n: int, device) -> torch.Tensor:
    """Builds a (n,) uint64 tensor of each expert's weight tensor address """
    addrs = [weights[key_tmpl.format(i)].data_ptr() for i in range(n)]
    return torch.tensor(addrs, dtype=torch.uint64, device=device)

@torch.compile
def compute_shared_expert(weights, tokens):
    w_gate = weights['shared_experts.0.weight']
    w_up = weights['shared_experts.1.weight']
    w_down = weights['shared_experts.2.weight']
    gate = F.silu(tokens @ w_gate)  # (batch_size * seq_len, d_expert * n_shared_experts)
    up = tokens @ w_up
    down = (gate * up) @ w_down     # (batch_Size * seq_len, d_hidden)
    return down

@torch.compile
def compute_routing(weights, tokens, topk):
    logits = F.linear(tokens, weights['router.weight'])   # x @ W.T (batch_size * seq_len, n_routed_experts)
    scores = logits.softmax(dim=-1)
    topk_scores, topk_indices = torch.topk(scores, k=topk, dim=-1, sorted=False)
    return topk_scores, topk_indices

@torch.compile
def permute_for_experts(
    expert_indices: torch.Tensor,
    expert_scores: torch.Tensor,
    tokens: torch.Tensor,
    topk: int,
    num_experts: int,
):
    flat_expert_indices = expert_indices.view(-1)
    # flat_expert_indices -> (batch_size*seq_len*n_experts_per_token, ) : (1, )
    flat_expert_scores = expert_scores.view(-1)
    # flat_expert_scores -> (batch_size*seq_len*n_experts_per_token, 1) : (1, 1)
    idxs = flat_expert_indices.argsort()    # permutation of 0..tokens*top_k-1
    # (batch_size * seq_len * n_experts_per_token, )
                      
    sorted_expert_ids = flat_expert_indices[idxs]
    sorted_scores = flat_expert_scores[idxs]
 
    counts = sorted_expert_ids.to(torch.int32).bincount(minlength=num_experts).to(torch.int32)
    expert_offsets = torch.zeros_like(counts)
    expert_offsets[1:] = torch.cumsum(counts[:-1], dim=0)
 
    token_idxs = idxs // topk
    grouped_input = tokens[token_idxs]
    # (batch_size * seq_len * n_experts_per_token, d_hidden) no padding
    dest_row = idxs.to(torch.int32)              # unique destination row per assignment
 
    return grouped_input, sorted_scores, dest_row, counts, expert_offsets

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
    device = input_tensor.device
    num_experts = n_routed_experts

    # Combined weight tensors for all routed experts
    W_gate_ptrs = _weight_ptrs(weights, "experts.{}.0.weight", num_experts, device)
    # W_gate_ptrs : (n_routed_experts, ) 
    W_up_ptrs = _weight_ptrs(weights, "experts.{}.1.weight", num_experts, device)
    # W_up_ptrs : (n_routed_experts, )
    W_down_ptrs = _weight_ptrs(weights, "experts.{}.2.weight", num_experts, device)
    # W_down : (n_routed_experts, )

    # shared expert output
    shared_out = compute_shared_expert(weights, input_tensor)    

    # expert routing
    topk_scores, topk_indices = compute_routing(weights, input_tensor, n_experts_per_token)
    # topk_scores: (batch_size * seq_len, n_experts_per_token)
    # topk_indices: (batch_size * seq_len, n_experts_per_token)
    # print(f"topk_scores dtype : {topk_scores.dtype}")

    grouped_input, sorted_expert_scores, dest_row, counts, expert_offsets = permute_for_experts(
        topk_indices, topk_scores, input_tensor, n_experts_per_token, num_experts
    )
    # grouped_input : (batch_size * seq_len * n_experts_per_token, d_hidden)
    # sorted_expert_scores : (batch_size*seq_len*n_experts_per_token, 1) : (1, 1)
    # dest_row : (batch_size * seq_len * n_experts_per_token, )
    # counts : [num_experts]
    # expert_offsets : [num_experts]

    exp_token_pairs = grouped_input.shape[0]  # [batch_size * seq_len * n_experts_per_token]

    # kernel 1: gate + up + silu
    routed_temp = torch.zeros(
        size= (exp_token_pairs, d_expert),
        device= device, dtype= torch.float16,
    )
    
    grid1 = lambda META: (META["NUM_SMS"], )
    gate_up_silu_kernel[grid1](
        W_gate_ptrs,
        W_up_ptrs,
        grouped_input,
        routed_temp,
        counts,
        expert_offsets,
        num_experts,
        d_hidden,
        d_expert,
    )

    # kernel 2: down projection
    routed_out = torch.zeros(
        size=(num_tokens * n_experts_per_token, d_hidden),
        device= input_tensor.device,
        dtype= torch.float32,
    )
    grid2 = lambda META: (META["NUM_SMS"], )
    down_proj_kernel[grid2](
        W_down_ptrs,
        routed_temp,
        routed_out,
        d_hidden,
        d_expert,
        sorted_expert_scores,
        dest_row,
        counts,
        expert_offsets,
        num_experts,
    )
 
    routed_out = routed_out.view(num_tokens, n_experts_per_token, d_hidden).sum(dim=1).to(torch.float16)
    
    return (shared_out + routed_out).reshape(batch_size, seq_len, d_hidden)


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
        passed = torch.allclose(expect, actual, atol=1e-2, rtol=0)
        if passed:
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")
    else:
        icon = "⭕"
        print(f"\r  Disabled: {icon}  ")

if __name__ == "__main__":
    torch.cuda.set_device(0)
    dhidden = 512
    dexpert = 128
    nroutedexperts = 4
    nsharedexperts = 1
    nexpertspertoken = 2
    bs = 2
    seqlen = 8
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

    

    # profiel the custom kernel
    # profile_moe()
