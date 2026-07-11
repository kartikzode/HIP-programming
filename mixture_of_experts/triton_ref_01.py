import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from task import input_t, output_t
import triton
import triton.language as tl


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
        logits = self.W_g(x)
        scores = logits.softmax(dim=-1)
        topk_scores, topk_indices = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )

        return topk_indices, topk_scores


def configs():
    return [
        triton.Config(
            kwargs={
                "k_idx": k_idx,
                "k_hid": k_hid,
                "k_exp": k_exp,
            },
            num_warps=8,
        )
        for k_idx in [16]
        for k_hid in [32]
        for k_exp in [32]
    ]


@triton.autotune(
    configs=configs(),
    key=[],  # the two above configs will be evaluated anytime
)
@triton.jit
def expert_kernel(
    expert_gate,
    expert_up,
    expert_down,
    x,
    out,
    d_hidden,
    d_expert,
    idxs_len,
    expert_token_idxs,
    expert_weights,
    k_idx: tl.constexpr,
    k_hid: tl.constexpr,
    k_exp: tl.constexpr,
):
    tl.multiple_of(d_hidden, 32)
    tl.multiple_of(d_expert, 32)

    start_id = tl.program_id(0) * k_idx
    start_exp = tl.program_id(1) * k_exp

    ids_mask = tl.arange(0, k_idx) < idxs_len - start_id
    exp_mask = tl.arange(0, k_exp) < d_expert - start_exp

    ids = tl.load(
        expert_token_idxs + start_id + tl.arange(0, k_idx),
        mask=ids_mask,
    )
    scales = tl.load(expert_weights + start_id + tl.arange(0, k_idx), mask=ids_mask)

    gate_sum = tl.zeros((k_exp, k_idx), dtype=tl.float32)
    up_sum = tl.zeros((k_exp, k_idx), dtype=tl.float32)

    for hid in tl.range(0, d_hidden, k_hid):
        d_hidden_mask = tl.arange(0, k_hid) < d_hidden - hid
        input = tl.load(
            x + (hid + tl.arange(0, k_hid))[:, None] + (d_hidden * ids)[None, :],
            mask=d_hidden_mask[:, None] & ids_mask[None, :],
        )
        gate = tl.load(
            expert_gate
            + (start_exp + tl.arange(0, k_exp))[:, None]
            + d_expert * (hid + tl.arange(0, k_hid))[None, :],
            mask=exp_mask[:, None] & d_hidden_mask[None, :],
        )
        gate_sum = tl.dot(gate, input, acc=gate_sum, input_precision="ieee")
        up = tl.load(
            expert_up
            + (start_exp + tl.arange(0, k_exp))[:, None]
            + d_expert * (hid + tl.arange(0, k_hid))[None, :],
            mask=exp_mask[:, None] & d_hidden_mask[None, :],
        )
        up_sum = tl.dot(up, input, acc=up_sum, input_precision="ieee")
    # silu
    gate_sum = gate_sum * tl.sigmoid(gate_sum)
    right = gate_sum * up_sum
    for hid in tl.range(0, d_hidden, k_hid):
        d_hidden_mask = tl.arange(0, k_hid) < d_hidden - hid
        down = tl.load(
            expert_down
            + (hid + tl.arange(0, k_hid))[:, None]
            + d_hidden * (start_exp + tl.arange(0, k_exp))[None, :],
            mask=d_hidden_mask[:, None] & exp_mask[None, :],
        ).to(tl.float32)

        v = tl.dot(down, right, input_precision="ieee")
        v = v * scales[None, :]

        tl.atomic_add(
            out + (hid + tl.arange(0, k_hid))[:, None] + (d_hidden * ids)[None, :],
            v,
            mask=d_hidden_mask[:, None] & ids_mask[None, :],
        )


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [Expert(config) for _ in range(config["n_routed_experts"])]
        )
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
        routed_output_flat = self.moe_infer(
            x_flat, flat_expert_indices, flat_expert_weights
        )

        routed_output = routed_output_flat.view(*orig_shape)
        return routed_output + shared_output

    @torch.no_grad()
    def moe_infer(
        self,
        x: torch.Tensor,
        flat_expert_indices: torch.Tensor,
        flat_expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.zeros_like(x, device=x.device, dtype=torch.float32)
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

            grid = lambda META: (
                triton.cdiv(len(exp_token_idxs), META["k_idx"]),
                triton.cdiv(self.config["d_expert"], META["k_exp"]),
            )
            expert_kernel[grid](
                expert.W_gate.weight,
                expert.W_up.weight,
                expert.W_down.weight,
                x,
                out,
                self.config["d_hidden"],
                self.config["d_expert"],
                len(exp_token_idxs),
                exp_token_idxs,  # flat_expert_indices[idxs[start_idx:end_idx]],
                flat_expert_weights[idxs[start_idx:end_idx]],
            )
        torch.cuda.synchronize()
        return out

    """
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
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )

        return expert_cache
    """


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
    num_experts = config["n_routed_experts"]

    experts = [{} for _ in range(num_experts)]
    for i in range(num_experts):
        gate_proj_weight = weights[f"experts.{i}.0.weight"]
        up_proj_weight = weights[f"experts.{i}.1.weight"]
        down_proj_weight = weights[f"experts.{i}.2.weight"]

        # Transpose weights to match expected shape for nn.Linear
        experts[i]["W_gate"] = gate_proj_weight
        experts[i]["W_up"] = up_proj_weight
        experts[i]["W_down"] = down_proj_weight

    shared_W_gate = weights["shared_experts.0.weight"].t()
    shared_W_up = weights["shared_experts.1.weight"].t()
    shared_W_down = weights["shared_experts.2.weight"].t()

    gate = F.silu(F.linear(input_tensor, shared_W_gate), inplace=True)
    shared_out = F.linear(gate * F.linear(input_tensor, shared_W_up), shared_W_down)

    logits = F.linear(input_tensor, weights["router.weight"])
    scores = logits.softmax(dim=-1)
    expert_scores, expert_indices = torch.topk(
        scores, k=config["n_experts_per_token"], dim=-1, sorted=False
    )

    flat_expert_indices = expert_indices.view(-1)
    flat_expert_weights = expert_scores.view(-1, 1)
    out = torch.zeros_like(
        input_tensor, device=input_tensor.device, dtype=torch.float32
    )
    idxs = flat_expert_indices.argsort()
    counts = flat_expert_indices.bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    num_per_tok = config["n_experts_per_token"]
    token_idxs = idxs // num_per_tok
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if start_idx == end_idx:
            continue

        expert = experts[expert_id]
        exp_token_idxs = token_idxs[start_idx:end_idx]

        grid = lambda META: (
            triton.cdiv(len(exp_token_idxs), META["k_idx"]),
            triton.cdiv(config["d_expert"], META["k_exp"]),
        )
        expert_kernel[grid](
            expert["W_gate"],
            expert["W_up"],
            expert["W_down"],
            input_tensor,
            out,
            config["d_hidden"],
            config["d_expert"],
            len(exp_token_idxs),
            exp_token_idxs,  # flat_expert_indices[idxs[start_idx:end_idx]],
            flat_expert_weights[idxs[start_idx:end_idx]],
        )
        # print(expert_kernel.best_config)
    torch.cuda.synchronize()
    return out + shared_out
    """
    moe = MoE(config)

    # Fill in the given weights of the model
    moe.gating_network.W_g.weight = nn.Parameter(weights["router.weight"])

    for i in range(num_experts):
        gate_proj_weight = weights[f"experts.{i}.0.weight"]
        up_proj_weight = weights[f"experts.{i}.1.weight"]
        down_proj_weight = weights[f"experts.{i}.2.weight"]

        # Transpose weights to match expected shape for nn.Linear
        moe.experts[i].W_gate.weight = nn.Parameter(gate_proj_weight)
        moe.experts[i].W_up.weight = nn.Parameter(up_proj_weight)
        moe.experts[i].W_down.weight = nn.Parameter(down_proj_weight)

    moe.shared_expert.W_gate.weight = nn.Parameter(
        weights["shared_experts.0.weight"].t()
    )
    moe.shared_expert.W_up.weight = nn.Parameter(weights["shared_experts.1.weight"].t())
    moe.shared_expert.W_down.weight = nn.Parameter(
        weights["shared_experts.2.weight"].t()
    )

    # Run the model
    output = moe(input_tensor)

    return output
    """
