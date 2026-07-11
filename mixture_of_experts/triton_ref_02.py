import os

os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
os.environ["TRITON_HIP_GLOBAL_PREFETCH"] = "3"

import math
import sys
import os
from typing import Dict, Tuple

import triton
import triton.language as tl
import torch
import torch.nn as nn

from datetime import datetime
import pytz


# --- Global Timezone ---
IST = pytz.timezone("Asia/Kolkata")

# --- MI300X Tuned Parameters ---
# Kernel 1: _compute_fused_scatter_mi300x_kernel
K1_BLOCK_D_EXP_OUT_TUNED = 64
K1_BLOCK_H_SLICING_TUNED = 64
K1_NUM_WARPS_TUNED = 4
K1_NUM_STAGES_TUNED = 4  # For kernel launch `num_stages` (e.g., dot pipelining)
K1_LOOP_NUM_STAGES_TUNED = (
    1  # MODIFIED: For tl.range(..., num_stages=...) in Kernel 1 (based on benchmarks)
)
K1_LOOP_UNROLL_FACTOR_TUNED = 8  # NEW: For tl.range(..., loop_unroll_factor=...) in Kernel 1

# Kernel 2: _project_and_accumulate_mi300x_kernel
K2_BLOCK_H_OUT_TUNED = 32
K2_BLOCK_D_STREAM_TUNED = 64
K2_NUM_WARPS_TUNED = 4
K2_NUM_STAGES_TUNED = 4  # For kernel launch `num_stages`
K2_LOOP_NUM_STAGES_TUNED = (
    1  # MODIFIED: For tl.range(..., num_stages=...) in Kernel 2 (based on benchmarks)
)
K2_LOOP_UNROLL_FACTOR_TUNED = 1  # NEW: For tl.range(..., loop_unroll_factor=...) in Kernel 2


# --- Profiling Utility ---
def _maybe_profile(msg: str):
    if os.getenv("MIXTURE_OF_EXPERTS_PROFILE"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ist_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ist_time}] {msg}", file=sys.stderr)


# --- Standard Expert Module (Unchanged) ---
class Expert(nn.Module):
    def __init__(self, config: Dict, d_expert: int = None, device=None, dtype=None):
        super().__init__()
        self.d_hidden_expert_in = config["d_hidden"]
        self.d_expert_out = config["d_expert"] if d_expert is None else d_expert
        self.act = nn.SiLU()
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.W_gate = nn.Linear(self.d_hidden_expert_in, self.d_expert_out, bias=False, **kwargs)
        self.W_up = nn.Linear(self.d_hidden_expert_in, self.d_expert_out, bias=False, **kwargs)
        self.W_down = nn.Linear(self.d_expert_out, self.d_hidden_expert_in, bias=False, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.W_gate(x))
        up = self.W_up(x)
        return self.W_down(gate * up)


# --- Kernel 1: Rewritten for MI300X with tl.range ---
@triton.jit
def _compute_fused_scatter_mi300x_kernel(
    x_flat_ptr,
    indices_to_gather_x_ptr,
    sorted_expert_ids_ptr,
    w_gate_ptr,
    w_up_ptr,
    sort_order_indices_ptr,
    fused_out_original_order_ptr,
    P_total_pairs: tl.constexpr,
    H_DIM: tl.constexpr,
    D_EXP: tl.constexpr,
    BLOCK_D_EXP_OUT: tl.constexpr,
    BLOCK_H_SLICING: tl.constexpr,
    LOOP_NUM_STAGES_K1: tl.constexpr,
    LOOP_UNROLL_FACTOR_K1: tl.constexpr,  # New parameter for unroll factor
) -> None:
    pid_pair_sorted = tl.program_id(0)
    pid_d_exp_tile_idx = tl.program_id(1)

    if pid_pair_sorted >= P_total_pairs:
        return

    expert_id = tl.load(sorted_expert_ids_ptr + pid_pair_sorted)
    original_token_idx_for_x = tl.load(indices_to_gather_x_ptr + pid_pair_sorted)
    original_pair_idx_for_output = tl.load(sort_order_indices_ptr + pid_pair_sorted)

    ptr_x_token_base = x_flat_ptr + original_token_idx_for_x * H_DIM

    d_exp_start_offset = pid_d_exp_tile_idx * BLOCK_D_EXP_OUT
    offs_d_exp_prog = d_exp_start_offset + tl.arange(0, BLOCK_D_EXP_OUT)
    mask_d_exp_prog_valid = offs_d_exp_prog < D_EXP

    expert_offset_in_weights = expert_id * D_EXP * H_DIM
    row_offsets_in_expert_matrix = offs_d_exp_prog[:, None] * H_DIM

    base_w_gate_expert_tile_rows = (
        w_gate_ptr + expert_offset_in_weights + row_offsets_in_expert_matrix
    )
    base_w_up_expert_tile_rows = w_up_ptr + expert_offset_in_weights + row_offsets_in_expert_matrix

    gate_acc = tl.zeros((BLOCK_D_EXP_OUT,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_D_EXP_OUT,), dtype=tl.float32)

    offs_h_stream_elements = tl.arange(0, BLOCK_H_SLICING)
    # MODIFIED: Use tl.range with loop_unroll_factor
    for h_start in tl.range(
        0,
        H_DIM,
        step=BLOCK_H_SLICING,
        num_stages=LOOP_NUM_STAGES_K1,
        loop_unroll_factor=LOOP_UNROLL_FACTOR_K1,
    ):
        current_h_offs = h_start + offs_h_stream_elements
        mask_h_slice_valid = current_h_offs < H_DIM

        x_slice = tl.load(
            ptr_x_token_base + current_h_offs,
            mask=mask_h_slice_valid,
            other=0.0,
            eviction_policy="evict_last",
        )

        ptr_w_gate_tile = base_w_gate_expert_tile_rows + current_h_offs[None, :]
        mask_w_tile_valid = mask_d_exp_prog_valid[:, None] & mask_h_slice_valid[None, :]
        w_gate_tile = tl.load(
            ptr_w_gate_tile, mask=mask_w_tile_valid, other=0.0, eviction_policy="evict_first"
        )

        ptr_w_up_tile = base_w_up_expert_tile_rows + current_h_offs[None, :]
        w_up_tile = tl.load(
            ptr_w_up_tile, mask=mask_w_tile_valid, other=0.0, eviction_policy="evict_first"
        )

        x_slice_casted = x_slice.to(w_gate_tile.dtype)
        gate_acc += tl.sum(w_gate_tile * x_slice_casted[None, :], axis=1)
        up_acc += tl.sum(w_up_tile * x_slice_casted[None, :], axis=1)

    gate_activated_f32 = gate_acc * tl.sigmoid(gate_acc)
    fused_result_f32 = gate_activated_f32 * up_acc
    fused_result_f16 = fused_result_f32.to(fused_out_original_order_ptr.dtype.element_ty)

    out_ptr_for_tile = (
        fused_out_original_order_ptr + original_pair_idx_for_output * D_EXP + offs_d_exp_prog
    )
    tl.store(out_ptr_for_tile, fused_result_f16, mask=mask_d_exp_prog_valid)


# --- Kernel 2: Adapted for MI300X with Cache Hints and tl.range ---
@triton.jit
def _project_and_accumulate_mi300x_kernel(
    fused_in_ptr,
    original_expert_ids_ptr,
    original_routing_w_ptr,
    w_down_ptr,
    out_ptr,
    NUM_TOKENS: tl.constexpr,
    H_DIM: tl.constexpr,
    D_EXP: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H_OUT: tl.constexpr,
    BLOCK_D_STREAM: tl.constexpr,
    LOOP_NUM_STAGES_K2: tl.constexpr,
    LOOP_UNROLL_FACTOR_K2: tl.constexpr,  # New parameter for unroll factor
) -> None:
    pid_tok = tl.program_id(0)
    pid_h_out_tile_idx = tl.program_id(1)

    if pid_tok >= NUM_TOKENS:
        return

    h_out_start_offset = pid_h_out_tile_idx * BLOCK_H_OUT
    offs_h_out_prog = h_out_start_offset + tl.arange(0, BLOCK_H_OUT)
    mask_h_out_prog_valid = offs_h_out_prog < H_DIM

    out_acc_h_slice = tl.zeros((BLOCK_H_OUT,), dtype=tl.float32)

    for k_idx in range(TOP_K):
        original_pair_id = pid_tok * TOP_K + k_idx

        expert_id_for_w_down = tl.load(original_expert_ids_ptr + original_pair_id)
        routing_weight_fp16 = tl.load(original_routing_w_ptr + original_pair_id)
        routing_weight_f32 = routing_weight_fp16.to(tl.float32)

        fused_data_base_for_pair = fused_in_ptr + original_pair_id * D_EXP

        expert_offset_in_w_down = expert_id_for_w_down * H_DIM * D_EXP
        row_offsets_in_expert_w_down = offs_h_out_prog[:, None] * D_EXP
        base_w_down_expert_tile_rows = (
            w_down_ptr + expert_offset_in_w_down + row_offsets_in_expert_w_down
        )

        current_expert_acc_h_slice = tl.zeros((BLOCK_H_OUT,), dtype=tl.float32)

        offs_d_stream_elements = tl.arange(0, BLOCK_D_STREAM)
        # MODIFIED: Use tl.range with loop_unroll_factor
        for d_start in tl.range(
            0,
            D_EXP,
            step=BLOCK_D_STREAM,
            num_stages=LOOP_NUM_STAGES_K2,
            loop_unroll_factor=LOOP_UNROLL_FACTOR_K2,
        ):
            current_d_offs = d_start + offs_d_stream_elements
            mask_d_stream_valid = current_d_offs < D_EXP

            f_slice = tl.load(
                fused_data_base_for_pair + current_d_offs,
                mask=mask_d_stream_valid,
                other=0.0,
                eviction_policy="evict_first",
            )

            ptr_w_down_tile = base_w_down_expert_tile_rows + current_d_offs[None, :]
            mask_w_tile_valid = mask_h_out_prog_valid[:, None] & mask_d_stream_valid[None, :]
            w_down_tile = tl.load(
                ptr_w_down_tile, mask=mask_w_tile_valid, other=0.0, eviction_policy="evict_first"
            )

            f_slice_casted = f_slice.to(w_down_tile.dtype)
            current_expert_acc_h_slice += tl.sum(w_down_tile * f_slice_casted[None, :], axis=1)

        out_acc_h_slice += current_expert_acc_h_slice * routing_weight_f32

    output_ptr_for_tile = out_ptr + pid_tok * H_DIM + offs_h_out_prog
    tl.store(
        output_ptr_for_tile,
        out_acc_h_slice.to(out_ptr.dtype.element_ty),
        mask=mask_h_out_prog_valid,
    )


# --- TritonMoE Module ---
class TritonMoE(nn.Module):
    def __init__(self, config: Dict, device: torch.device):
        super().__init__()
        self.E = config["n_routed_experts"]
        self.K = config["n_experts_per_token"]
        self.H_dim = config["d_hidden"]
        self.D_exp = config["d_expert"]
        self.device = device
        expert_dtype_arg = torch.float16

        self.W_g = nn.Linear(self.H_dim, self.E, bias=False, device=device, dtype=expert_dtype_arg)
        self.W_gate = nn.Parameter(
            torch.empty(self.E, self.D_exp, self.H_dim, device=device, dtype=expert_dtype_arg)
        )
        self.W_up = nn.Parameter(
            torch.empty(self.E, self.D_exp, self.H_dim, device=device, dtype=expert_dtype_arg)
        )
        self.W_down = nn.Parameter(
            torch.empty(self.E, self.H_dim, self.D_exp, device=device, dtype=expert_dtype_arg)
        )
        shared_expert_output_dim = self.D_exp * config["n_shared_experts"]
        self.shared_expert = Expert(
            config, d_expert=shared_expert_output_dim, device=device, dtype=expert_dtype_arg
        )

    def load_weights_fast(self, weights: Dict[str, torch.Tensor]):
        dev = self.device
        td = torch.float16
        self.W_g.weight.data.copy_(weights["router.weight"].to(dev, td))
        self.W_gate.data.copy_(
            _stack_t(weights, "experts.{}.0.weight", self.E, (self.E, self.D_exp, self.H_dim)).to(
                dev, td
            )
        )
        self.W_up.data.copy_(
            _stack_t(weights, "experts.{}.1.weight", self.E, (self.E, self.D_exp, self.H_dim)).to(
                dev, td
            )
        )
        self.W_down.data.copy_(
            _stack_t(weights, "experts.{}.2.weight", self.E, (self.E, self.H_dim, self.D_exp)).to(
                dev, td
            )
        )

        self.shared_expert.W_gate.weight.data.copy_(
            weights["shared_experts.0.weight"].t().to(dev, td)
        )
        self.shared_expert.W_up.weight.data.copy_(
            weights["shared_experts.1.weight"].t().to(dev, td)
        )
        self.shared_expert.W_down.weight.data.copy_(
            weights["shared_experts.2.weight"].t().to(dev, td)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        assert H == self.H_dim, "Input H_dim mismatch"
        P_total_pairs = B * T * self.K
        NUM_TOKENS = B * T

        current_dtype = x.dtype
        if hasattr(self.W_g.weight, "dtype") and self.W_g.weight.dtype == torch.float16:
            current_dtype = torch.float16
        x_compute_dtype = x.to(current_dtype)

        _maybe_profile("start routing ↘")
        logits = self.W_g(x_compute_dtype)
        scores = logits.softmax(dim=-1)
        topk_scores, topk_idx = torch.topk(scores, self.K, dim=-1, sorted=False)

        original_expert_flat = topk_idx.reshape(-1).int()
        original_rout_flat = topk_scores.reshape(-1).to(current_dtype)
        x_flat = x_compute_dtype.reshape(NUM_TOKENS, H).contiguous()
        _maybe_profile("routing done ↗")

        _maybe_profile("start permutation ops (argsort only) ↘")
        token_indices_for_flat_pairs = (
            torch.arange(NUM_TOKENS, device=x.device, dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, self.K)
            .reshape(-1)
        )
        perm_indices = torch.argsort(original_expert_flat, stable=True).int()
        sorted_expert_ids = original_expert_flat[perm_indices].int()
        indices_to_gather_x = token_indices_for_flat_pairs[perm_indices].int()
        _maybe_profile("permutation ops (argsort only) done ↗")

        fbuf_original_order = torch.empty(
            (P_total_pairs, self.D_exp), device=self.device, dtype=current_dtype
        )
        obuf = torch.empty((NUM_TOKENS, H), device=self.device, dtype=current_dtype)

        # --- Kernel 1 Launch ---
        k1_grid_dim0 = P_total_pairs
        k1_grid_dim1 = (self.D_exp + K1_BLOCK_D_EXP_OUT_TUNED - 1) // K1_BLOCK_D_EXP_OUT_TUNED
        k1_grid = (k1_grid_dim0, k1_grid_dim1, 1)

        _maybe_profile(
            f"launch Kernel 1 (MI300X) grid={k1_grid} "
            f"BLOCK_D_EXP_OUT={K1_BLOCK_D_EXP_OUT_TUNED}, BLOCK_H_SLICING={K1_BLOCK_H_SLICING_TUNED} "
            f"NW={K1_NUM_WARPS_TUNED}, NS_kernel={K1_NUM_STAGES_TUNED}, "
            f"NS_loop={K1_LOOP_NUM_STAGES_TUNED}, UNROLL_loop={K1_LOOP_UNROLL_FACTOR_TUNED}↘"
        )
        _compute_fused_scatter_mi300x_kernel[k1_grid](
            x_flat,
            indices_to_gather_x,
            sorted_expert_ids,
            self.W_gate,
            self.W_up,
            perm_indices,
            fbuf_original_order,
            P_total_pairs=P_total_pairs,
            H_DIM=H,
            D_EXP=self.D_exp,
            BLOCK_D_EXP_OUT=K1_BLOCK_D_EXP_OUT_TUNED,
            BLOCK_H_SLICING=K1_BLOCK_H_SLICING_TUNED,
            LOOP_NUM_STAGES_K1=K1_LOOP_NUM_STAGES_TUNED,
            LOOP_UNROLL_FACTOR_K1=K1_LOOP_UNROLL_FACTOR_TUNED,  # Pass unroll factor
            num_warps=K1_NUM_WARPS_TUNED,
            num_stages=K1_NUM_STAGES_TUNED,
        )
        _maybe_profile("Kernel 1 (MI300X) done ↗")

        # --- Kernel 2 Launch ---
        k2_grid_dim0 = NUM_TOKENS
        k2_grid_dim1 = (H + K2_BLOCK_H_OUT_TUNED - 1) // K2_BLOCK_H_OUT_TUNED
        k2_grid = (k2_grid_dim0, k2_grid_dim1, 1)

        _maybe_profile(
            f"launch Kernel 2 (MI300X) grid={k2_grid} "
            f"BLOCK_H_OUT={K2_BLOCK_H_OUT_TUNED}, BLOCK_D_STREAM={K2_BLOCK_D_STREAM_TUNED} "
            f"NW={K2_NUM_WARPS_TUNED}, NS_kernel={K2_NUM_STAGES_TUNED}, "
            f"NS_loop={K2_LOOP_NUM_STAGES_TUNED}, UNROLL_loop={K2_LOOP_UNROLL_FACTOR_TUNED}↘"
        )
        _project_and_accumulate_mi300x_kernel[k2_grid](
            fbuf_original_order,
            original_expert_flat,
            original_rout_flat,
            self.W_down,
            obuf,
            NUM_TOKENS=NUM_TOKENS,
            H_DIM=H,
            D_EXP=self.D_exp,
            TOP_K=self.K,
            BLOCK_H_OUT=K2_BLOCK_H_OUT_TUNED,
            BLOCK_D_STREAM=K2_BLOCK_D_STREAM_TUNED,
            LOOP_NUM_STAGES_K2=K2_LOOP_NUM_STAGES_TUNED,
            LOOP_UNROLL_FACTOR_K2=K2_LOOP_UNROLL_FACTOR_TUNED,  # Pass unroll factor
            num_warps=K2_NUM_WARPS_TUNED,
            num_stages=K2_NUM_STAGES_TUNED,
        )
        _maybe_profile("Kernel 2 (MI300X) done ↗")

        routed_output = obuf.reshape(B, T, H)
        shared_output = self.shared_expert(x_compute_dtype)
        return routed_output + shared_output


def _stack_t(weights, key_tmpl: str, n, out_shape):
    mats = [weights[key_tmpl.format(i)].t() for i in range(n)]
    return torch.stack(mats, dim=0).reshape(out_shape).contiguous()


def custom_kernel(data):
    x, weights, config = data
    device = x.device
    _maybe_profile("construct module")
    moe = TritonMoE(config, device=device)
    _maybe_profile("module initialized")
    moe.load_weights_fast(weights)
    _maybe_profile("weights loaded")
    with torch.no_grad():
        out = moe(x)
    _maybe_profile("forward done")
    return out