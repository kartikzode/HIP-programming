#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <cstring>

// FP4 E2M1 dequant lookup table (indices 0–15)
__constant__ float FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,
     2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f,
    -2.0f, -3.0f, -4.0f, -6.0f
};

// E8M0 -> float: exponent-only format, value = 2^(e - 127)
// e=0 gives 0.0f (MX spec: zero scale)
__device__ __forceinline__ float e8m0_to_float(uint8_t e) {
    uint32_t bits = (uint32_t)e << 23;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

// Unpack k-th FP4 nibble from a row of fp4x2-packed bytes
// Even k -> low nibble, odd k -> high nibble
__device__ __forceinline__ float unpack_fp4(const uint8_t* row, int k) {
    uint8_t byte   = row[k >> 1];
    uint8_t nibble = (k & 1) ? (byte >> 4) : (byte & 0xF);
    return FP4_LUT[nibble];
}

// Most naive MXFP4 GEMM: one thread per output element.
// C[M,N] = dequant(A_q) @ dequant(B_q)^T
//
// A_q   : [M, K/2]  fp4x2 packed, row-major
// B_q   : [N, K/2]  fp4x2 packed, row-major
// A_scl : [M, K/32] E8M0 block scales
// B_scl : [N, K/32] E8M0 block scales
// C     : [M, N]    bf16 output
__global__ void mxfp4_naive_gemm_kernel(
    const uint8_t* __restrict__ A_q,
    const uint8_t* __restrict__ B_q,
    const uint8_t* __restrict__ A_scl,
    const uint8_t* __restrict__ B_scl,
    __hip_bfloat16* __restrict__ C,
    const int M,
    const int N,
    const int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    const int K_half  = K / 2;   // bytes per row
    const int K_scale = K / 32;  // scales per row

    const uint8_t* a_row = A_q   + row * K_half;
    const uint8_t* b_row = B_q   + col * K_half;
    const uint8_t* a_scl = A_scl + row * K_scale;
    const uint8_t* b_scl = B_scl + col * K_scale;

    float acc = 0.0f;

    for (int k = 0; k < K; k++) {
        float a_val = unpack_fp4(a_row, k) * e8m0_to_float(a_scl[k / 32]);
        float b_val = unpack_fp4(b_row, k) * e8m0_to_float(b_scl[k / 32]);
        acc += a_val * b_val;
    }

    C[row * N + col] = __float2bfloat16(acc);
}

// Host wrapper
void mxfp4_gemm(
    torch::Tensor A_q, torch::Tensor B_q,
    torch::Tensor A_scale, torch::Tensor B_scale,
    torch::Tensor C, int M, int N, int K
) {
    dim3 block(16, 16);  // 256 threads
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    mxfp4_naive_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const uint8_t*>(A_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(A_scale.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
        M, N, K
    );
}
