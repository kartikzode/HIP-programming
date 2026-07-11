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

def get_configs(pre_hook=None):
    return [
        triton.Config({ 'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 
                        "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8,
                        'loop_unroll_factor': 1},
                        num_stages=s,
                        num_warps=w, pre_hook=pre_hook,)
        for BM in [1]
        for BN in [32]
        for BK in [64]
        for s in ([2])
        for w in [4]
    ]

@triton.autotune(
    configs= get_configs(),
    key= ["total_expert_token_pairs", "d_expert", "d_hidden"],
)
@triton.jit
def _expert_kernel(
            x,
            token_idxs,
            sorted_expert_ids,
            exp_gate,
            exp_up,
            sort_order_indices,
            out,
            total_expert_token_pairs,
            d_hidden,
            d_expert,
            BLOCKSIZE_M: tl.constexpr,
            BLOCKSIZE_N: tl.constexpr,
            BLOCKSIZE_K: tl.constexpr,
            GROUPSIZE_M: tl.constexpr,
            ):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(total_expert_token_pairs, BLOCKSIZE_M)
    num_pid_n = tl.cdiv(d_expert, BLOCKSIZE_N)
    num_pid_in_group = GROUPSIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUPSIZE_M
    group_size_m = min(GROUPSIZE_M, num_pid_m - first_pid_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    expert_id = tl.load(sorted_expert_ids + pid_m)
    token_idx_x = tl.load(token_idxs + pid_m)
    offs_k = tl.arange(0, BLOCKSIZE_K)                      

    # token pointer arithmatic
    token_ids =  token_idx_x + tl.arange(0, BLOCKSIZE_M)
    a_ptrs = x + (token_ids[:, None] * d_hidden + offs_k[None, :])

    # Expert W_gate/W_up pointer arithmatic
    exp_offset = expert_id * d_hidden * d_expert
    col_offset = pid_n * BLOCKSIZE_N                        # start_n
    offs_bn = col_offset + tl.arange(0, BLOCKSIZE_N)
    offs_bn = tl.where(offs_bn < d_expert, offs_bn, 0)
    gate_tile_ptrs = exp_gate + exp_offset + (offs_k[:, None] * d_expert + offs_bn[None, :])
    up_tile_ptrs = exp_up + exp_offset + (offs_k[:, None] * d_expert + offs_bn[None, :])

    # Expert W_down pointer arithmatic
    
    
    gate_acc = tl.zeros((BLOCKSIZE_M, BLOCKSIZE_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCKSIZE_M, BLOCKSIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(d_hidden, BLOCKSIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < d_hidden - k * BLOCKSIZE_K, other=0.0)
        b0 = tl.load(gate_tile_ptrs, mask=offs_k[:, None] < d_hidden - k * BLOCKSIZE_K, other=0.0)
        
        gate_acc = tl.dot(a, b0, gate_acc, input_precision="ieee")

        b1 = tl.load(up_tile_ptrs, mask=offs_k[:, None] < d_hidden - k * BLOCKSIZE_K, other=0.0)
        up_acc = tl.dot(a, b1, up_acc, input_precision="ieee")
        a_ptrs += BLOCKSIZE_K
        gate_tile_ptrs += BLOCKSIZE_K * d_expert
        up_tile_ptrs += BLOCKSIZE_K * d_expert

    #SiLu
    gate_acc = gate_acc * tl.sigmoid(gate_acc)
    temp = gate_acc * up_acc                        # [BM, BN]

    for k in range(0, tl.cdiv(d_hidden,BLOCKSIZE_K)):
        b2 = tl.load(b2_ptrs,
                     mask= offs_k[None, :] < d_hidden - k * BLOCKSIZE_K, other=0.0
                    ).to(tl.float32)
        v = tl.dot(temp,b2, input_precision="ieee")
        v = v * scales[None, :]

        # writing to the out matrix
        tl.atomic_add(
            out + token_ids[:, None] * d_hidden + (k*BLOCKSIZE_K + offs_k[None, :]),
            v,
            mask= offs_k[None, :] < d_hidden - k * BLOCKSIZE_K,
        )
        b2_ptrs += BLOCKSIZE_K

    # out_offset = tl.load(sort_order_indices + pid_m)

    # offs_cm = out_offset * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # c_ptrs = temp_buffer + d_expert * offs_cm[:, None] + offs_cn[None, :]
    # c_mask = (offs_cm[:, None] < expert_token_pairs) & (offs_cn[None, :] < d_expert)
    # tl.store(c_ptrs, temp, mask=c_mask)


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

    def __init__(self, config: Dict, device):
        super().__init__()
        self.config = config
        self.device = device
        self.d_hidden = self.config["d_hidden"]
        self.d_expert = self.config["d_expert"]
        self.n_routed_experts = self.config["n_routed_experts"]
        self.n_shared_expeerts = self.config["n_shared_experts"]
        self.n_experts_per_token = self.config["n_experts_per_token"]
        self.expert_dtype = torch.float16
        # self.expert_token_pairs = self.config["batch_size"] * self.config["seq_len"]

        self.gating_network = MoEGate(config)
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(config=config, d_expert=shared_expert_dim)
        self.W_gate = nn.Parameter(
            torch.empty(self.n_routed_experts, self.d_expert, self.d_hidden, device=device, dtype=self.expert_dtype)
        )
        self.W_up = nn.Parameter(
            torch.empty(self.n_routed_experts, self.d_expert, self.d_hidden, device=device, dtype=self.expert_dtype)
        )
        self.W_down = nn.Parameter(
            torch.empty(self.n_routed_experts, self.d_hidden, self.d_expert, device=device, dtype=self.expert_dtype)
        )

    def _stack_t(self, weights, key_tmpl: str, n, out_shape):
        mats = [weights[key_tmpl.format(i)].t() for i in range(n)]
        return torch.stack(mats, dim=0).reshape(out_shape).contiguous()

    def load_data(self, weights: Dict[str, torch.Tensor]):
        self.gating_network.W_g.weight.data.copy_(
            weights["router.weight"].to(self.device, dtype=self.expert_dtype))
        self.W_gate.data.copy_(
            self._stack_t(weights, "experts.{}.0.weight", self.n_routed_experts, (self.n_routed_experts, self.d_expert, self.d_hidden)).to(
                self.device, dtype=self.expert_dtype
            )
        )
        self.W_up.data.copy_(
            self._stack_t(weights, "experts.{}.1.weight", self.n_routed_experts, (self.n_routed_experts, self.d_expert, self.d_hidden)).to(
                self.device, dtype=self.expert_dtype
            )
        )
        self.W_down.data.copy_(
            self._stack_t(weights, "experts.{}.2.weight", self.n_routed_experts, (self.n_routed_experts, self.d_hidden, self.d_expert)).to(
                self.device, dtype=self.expert_dtype
            )
        )

        self.shared_expert.W_gate.weight.data.copy_(
            weights["shared_experts.0.weight"].t().to(self.device, self.expert_dtype)
        )
        self.shared_expert.W_up.weight.data.copy_(
            weights["shared_experts.1.weight"].t().to(self.device, self.expert_dtype)
        )
        self.shared_expert.W_down.weight.data.copy_(
            weights["shared_experts.2.weight"].t().to(self.device, self.expert_dtype)
        )


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
        num_tokens, hidden_dim = x.shape
        total_exp_token_pairs = num_tokens * self.n_routed_experts
        sort_order_indices = flat_expert_indices.argsort()
        sorted_expert_ids = flat_expert_indices[sort_order_indices]
        num_per_tok = self.config["n_experts_per_token"]
        token_idxs = sort_order_indices // num_per_tok

        out = torch.empty((num_tokens, hidden_dim), device=x.device, dtype=x.dtype)


        k1_grid = lambda META: (triton.cdiv(total_exp_token_pairs, META["BLOCK_SIZE_M"]) * triton.cdiv(self.config["d_expert"], META["BLOCKSIZE_N"]), )
        _expert_kernel[k1_grid](
            x,
            token_idxs,
            sorted_expert_ids,
            self.W_gate,
            self.W_up,
            sort_order_indices,
            out,
            total_exp_token_pairs,
            hidden_dim,
            self.d_expert,
            )

        return out


def custom_kernel(data: input_t) -> output_t:
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
    device = input_tensor.device
    moe = MoE(config, device)
    moe.load_data(weights)

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

    
    # # Run the reference kernel
    # output = ref_kernel((input_tensor, weights, config))

    # # Print only shapes
    # print("Input shape:", input_tensor.shape)
    # print("Router weight shape:", weights['router.weight'].shape)

    # for i in range(nroutedexperts):
    #     print(f"Expert {i} gate weight shape:", weights[f'experts.{i}.0.weight'].shape)
    #     print(f"Expert {i} up weight shape:",   weights[f'experts.{i}.1.weight'].shape)
    #     print(f"Expert {i} down weight shape:", weights[f'experts.{i}.2.weight'].shape)

    # print("Shared expert gate weight shape:", weights['shared_experts.0.weight'].shape)
    # print("Shared expert up weight shape:",   weights['shared_experts.1.weight'].shape)
    # print("Shared expert down weight shape:", weights['shared_experts.2.weight'].shape)

    # print("Output shape:", output.shape)
    

if __name__ == "__main__":
    profile_moe()

