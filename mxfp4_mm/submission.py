#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
os.environ['CXX'] = 'clang++'

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# TILE_K must equal SCALE_GROUP_SIZE (32) so each K-tile maps to exactly one E8M0 scale.
CUDA_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_ext_ocp.h>

// One MFMA tile: 16x16x128
#define TILE_M 16
#define TILE_N 16
#define TILE_K 128   // one MFMA instruction consumes 128 FP4 elements along K

using fp4x2_t  = __amd_fp4x2_storage_t;
using fp4x64_t = fp4x2_t __attribute__((ext_vector_type(32)));   // 32 bytes = 64 FP4
using fp32x4_t = float __attribute__((ext_vector_type(4)));      // MFMA accumulator
using mfma_input_t = int __attribute__((ext_vector_type(8)));    // 32 bytes for MFMA src

// Naive MFMA MXFP4 GEMM — no shared memory.
// Each wavefront (64 threads) computes one 16×16 output tile.
// K is processed in chunks of TILE_K=128 (one MFMA per chunk).
//
// Layout:
//   A_q   : [M, K/2]  fp4x2 packed, row-major
//   B_q   : [N, K/2]  fp4x2 packed, row-major  (C = A @ B^T)
//   A_scl : [M, K/32] E8M0 scales
//   B_scl : [N, K/32] E8M0 scales
//   C     : [M, N]    bf16 output
//
// Thread mapping within a wavefront (64 lanes):
//   row_in_tile = lane % 16    -> which of the 16 M-rows in the tile
//   row_group   = lane / 16    -> which of the 4 groups (each loads 32 FP4 = 16 bytes)
//   Together the 4 groups cover 128 FP4 elements along K.

__global__ void mxfp4_gemm_naive_kernel(
    const fp4x2_t*  __restrict__ A_q,
    const fp4x2_t*  __restrict__ B_q,
    const uint8_t*  __restrict__ A_scl,
    const uint8_t*  __restrict__ B_scl,
    __hip_bfloat16* __restrict__ C,
    const int M,
    const int N,
    const int K
) {
    const int lane = threadIdx.x;              // [0, 63]
    const int row_in_tile = lane & 15;         // [0, 15]
    const int row_group   = lane >> 4;         // [0, 3]

    // Which 16×16 output tile this wavefront computes
    const int tile_row = blockIdx.x * TILE_M;  // starting M-row
    const int tile_col = blockIdx.y * TILE_N;  // starting N-col

    const int global_m = tile_row + row_in_tile;
    const int global_n = tile_col + row_in_tile;

    // Byte strides
    const int K_half  = K / 2;    // bytes per row in fp4x2 layout
    const int K_scale = K / 32;   // number of E8M0 scale groups per row

    // Each row_group loads 16 bytes (32 FP4 values) starting at this byte offset
    // within the 64-byte (128 FP4) K-tile
    const int k_byte_offset = row_group * 16;

    // Accumulator — 4 floats, maps to 4 output rows: (lane/16)*4 + i, col = lane%16
    fp32x4_t c_reg = {0.0f, 0.0f, 0.0f, 0.0f};

    // Loop over K in chunks of TILE_K=128
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // --- Load A fragment directly from global memory ---
        // This thread loads 16 bytes from A_q[global_m, (k0/2) + k_byte_offset]
        fp4x64_t a_reg = {};
        if (global_m < M) {
            const fp4x2_t* a_row = A_q + global_m * K_half + (k0 / 2);
            *reinterpret_cast<uint4*>(&a_reg) =
                *reinterpret_cast<const uint4*>(a_row + k_byte_offset);
        }

        // --- Load B fragment directly from global memory ---
        fp4x64_t b_reg = {};
        if (global_n < N) {
            const fp4x2_t* b_row = B_q + global_n * K_half + (k0 / 2);
            *reinterpret_cast<uint4*>(&b_reg) =
                *reinterpret_cast<const uint4*>(b_row + k_byte_offset);
        }

        // --- Load scales ---
        // 4 E8M0 scale bytes per row for this K-tile (one per group of 32 FP4 elements).
        // Pack them into a uint32 for the MFMA scale operand.
        // scale_idx is the index of the first scale group in this TILE_K chunk.
        int scale_idx = k0 / 32;  // 4 scale groups per TILE_K

        // Each lane loads only its own scale byte (for its row_group's K range).
        // The MFMA scale operand is a uint32 but with op_sel=0, only the low byte is used.
        uint32_t scale_a = 0;
        if (global_m < M) {
            scale_a = A_scl[global_m * K_scale + scale_idx + row_group];
        }

        uint32_t scale_b = 0;
        if (global_n < N) {
            scale_b = B_scl[global_n * K_scale + scale_idx + row_group];
        }

        // --- MFMA: 16×16×128 FP4 with E8M0 block scaling ---
        // cbsz=4, blgp=4 selects FP4 operand type for both A and B
        c_reg = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            reinterpret_cast<mfma_input_t&>(a_reg),
            reinterpret_cast<mfma_input_t&>(b_reg),
            c_reg,
            4, 4,       // cbsz=4 (FP4 A), blgp=4 (FP4 B)
            0, scale_a,
            0, scale_b
        );
    }

    // --- Write-back ---
    // MFMA output mapping: c_reg[i] -> row = (lane/16)*4 + i, col = lane%16
    int out_col = tile_col + (lane % 16);
    for (int i = 0; i < 4; i++) {
        int out_row = tile_row + (lane / 16) * 4 + i;
        if (out_row < M && out_col < N) {
            C[out_row * N + out_col] = __float2bfloat16(c_reg[i]);
        }
    }
}

// Host-callable wrapper
void mxfp4_gemm(
    torch::Tensor A_q, torch::Tensor B_q,
    torch::Tensor A_scale, torch::Tensor B_scale,
    torch::Tensor C, int M, int N, int K
) {
    dim3 block(64);   // one wavefront per block
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    mxfp4_gemm_naive_kernel<<<grid, block>>>(
        reinterpret_cast<const fp4x2_t*>(A_q.data_ptr()),
        reinterpret_cast<const fp4x2_t*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(A_scale.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
        M, N, K
    );
}
"""

CPP_SRC = """
void mxfp4_gemm(
    torch::Tensor A_q, torch::Tensor B_q,
    torch::Tensor A_scale, torch::Tensor B_scale,
    torch::Tensor C, int M, int N, int K);
"""

module = load_inline(
    name='mxfp4_gemm_module',
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=['mxfp4_gemm'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++20"],
)


def custom_kernel(data: input_t) -> output_t:
    import aiter
    from aiter import QuantType, dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant 
    from aiter.utility.fp4_utils import e8m0_shuffle

    def _quant_mxfp4(x, shuffle=True):
        x_fp4, bs_e8m0 = dynamic_mxfp4_quant(x)
        if shuffle:
            bs_e8m0 = e8m0_shuffle(bs_e8m0)
        return x_fp4.view(dtypes.fp4x2), bs_e8m0.view(dtypes.fp8_e8m0)

    A, B, _B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    B = B.contiguous()
    m, k = A.shape
    n, _ = B.shape

    A_q, A_scale = _quant_mxfp4(A, shuffle=False)
    B_q, B_scale = _quant_mxfp4(B, shuffle=False)

    # dynamic_mxfp4_quant may return padded tensors — slice to exact shapes
    k_half  = k // 2
    k_scale = k // 32
    A_q     = A_q[:m, :k_half].contiguous()
    A_scale = A_scale[:m, :k_scale].contiguous()
    B_q     = B_q[:n, :k_half].contiguous()
    B_scale = B_scale[:n, :k_scale].contiguous()

    C = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    module.mxfp4_gemm(A_q, B_q, A_scale, B_scale, C, m, n, k)
    return C
