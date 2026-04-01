#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_ext_ocp.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 128
#define BLOCKSIZE 16

using fp4x2_t = __amd_fp4x2_storage_t;
using fp4x64_t = fp4x2_t __attribute__((ext_vector_type(32)));  // 32 elements = 32 bytes = 64 FP4
using fp4x32_t = fp4x2_t __attribute__((vector_size(16)));      // 16 elements = 16 bytes = 32 FP4
using fp32x4_t = float __attribute__((ext_vector_type(4)));     // 4 floats
using mfma_input_t = int __attribute__((ext_vector_type(8)));   // 8 ints = 32 bytes

//Helper: load 32 fp4 values (16B/ 128 bits) from address p
static inline __device__ fp4x32_t load_fp4x32(const fp4x2_t* p) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    return *reinterpret_cast<const fp4x32_t*>(&v);
}


__global__
void mxfp4_gemm_kernel(
    const __amd_fp4x2_storage_t* __restrict__ A,
    const __amd_fp4x2_storage_t* __restrict__ B,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_scale,
    __hip_bfloat16* __restrict__ C,
    const int M,
    const int N,
    const int K
)  {

    __shared__ uint8_t A_tile[TILE_M][TILE_K / 2];         // [16][64]
    __shared__ uint8_t B_tile[TILE_N][TILE_K / 2];         // [16][64]

    __shared__ uint8_t A_scale_tile[TILE_M][TILE_K / 32];  // [16][4]
    __shared__ uint8_t B_scale_tile[TILE_N][TILE_K / 32];  // [16][4]

    const int lane = threadIdx.x & 63;          // [0, 63]
    const int row_in_tile = lane & 15;          // [0, 15]  (for MFMA register mapping)
    const int row_group = lane >> 4;            // [0, 3]   (for MFMA register mapping)
    const int k_byte_offset = row_group * 16;

    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    const int K_half = K / 2;      // bytes per row in A/B
    const int K_scale = K / 32;    // scale bytes per row

    // Coalesced load mapping: 4 consecutive lanes load one row of 64 bytes
    // lane / 4 = which row (0-15), lane % 4 = which 16-byte chunk (0-3)
    const int load_row = lane / 4;          // 0-15
    const int load_col = (lane % 4) * 16;   // 0, 16, 32, 48

    //acc registers
    fp32x4_t c_reg = {0.0f, 0.0f, 0.0f, 0.0f};

    // loop over the K dimension
    for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_K) {

        // --- Coalesced global → LDS loads ---
        // Lanes 0-3 load row 0 (64 contiguous bytes), lanes 4-7 load row 1, etc.
        {
            const int gm = cRow * BLOCKSIZE + load_row;
            if (gm < M) {
                const fp4x32_t a_vec = load_fp4x32(A + gm * K_half + (bkIdx / 2) + load_col);
                *reinterpret_cast<fp4x32_t*>(&A_tile[load_row][load_col]) = a_vec;
            } else {
                *reinterpret_cast<fp4x32_t*>(&A_tile[load_row][load_col]) = {};
            }
        }
        {
            const int gn = cCol * BLOCKSIZE + load_row;
            if (gn < N) {
                const fp4x32_t b_vec = load_fp4x32(B + gn * K_half + (bkIdx / 2) + load_col);
                *reinterpret_cast<fp4x32_t*>(&B_tile[load_row][load_col]) = b_vec;
            } else {
                *reinterpret_cast<fp4x32_t*>(&B_tile[load_row][load_col]) = {};
            }
        }

        // Scales: A_scale_tile is 16 rows × 4 bytes = 64 bytes
        // 64 lanes, each loads 1 byte
        {
            const int scl_row = lane / 4;   // 0-15
            const int scl_col = lane % 4;   // 0-3
            const int gm = cRow * BLOCKSIZE + scl_row;
            if (gm < M) {
                A_scale_tile[scl_row][scl_col] = A_scale[gm * K_scale + (bkIdx / 32) + scl_col];
            } else {
                A_scale_tile[scl_row][scl_col] = 0;
            }
        }
        {
            const int scl_row = lane / 4;
            const int scl_col = lane % 4;
            const int gn = cCol * BLOCKSIZE + scl_row;
            if (gn < N) {
                B_scale_tile[scl_row][scl_col] = B_scale[gn * K_scale + (bkIdx / 32) + scl_col];
            } else {
                B_scale_tile[scl_row][scl_col] = 0;
            }
        }

        __syncthreads();

        // --- LDS → register loads (MFMA mapping) ---
        fp4x64_t a_reg {};
        fp4x64_t b_reg {};

        const fp4x2_t* ldg_a = &A_tile[row_in_tile][k_byte_offset];
        const fp4x2_t* ldg_b = &B_tile[row_in_tile][k_byte_offset];
        for (int i = 0; i < 16; i++) {
            a_reg[i] = *(ldg_a + i);
            b_reg[i] = *(ldg_b + i);
        }

        // Each lane loads only its own scale byte
        uint32_t scale_a = A_scale_tile[row_in_tile][row_group];
        uint32_t scale_b = B_scale_tile[row_in_tile][row_group];

        //mfma
        c_reg = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            reinterpret_cast<mfma_input_t&>(a_reg),
            reinterpret_cast<mfma_input_t&>(b_reg),
            c_reg, 4, 4, 0, scale_a, 0, scale_b);

        __syncthreads();

    }
    // write-back
    int out_col = cCol * BLOCKSIZE + (threadIdx.x % 16);
    for(int i=0; i < 4; i++) {
        int out_row = cRow * BLOCKSIZE + (threadIdx.x / 16) * 4 + i;
        if (out_row < M && out_col < N) {
            C[out_row * N + out_col] = __float2bfloat16(c_reg[i]);
        }
    }
}

void mxfp4_gemm(
    torch::Tensor A_q, torch::Tensor B_q,
    torch::Tensor A_scale, torch::Tensor B_scale,
    torch::Tensor C, int M, int N, int K
) {
    dim3 block(64);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    mxfp4_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const __amd_fp4x2_storage_t*>(A_q.data_ptr()),
        reinterpret_cast<const __amd_fp4x2_storage_t*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(A_scale.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
        M, N, K
    );
}


// ## Benchmarks:
// ```
// k: 512; m: 4; n: 2880; seed: 4565
//  ⏱ 20.8 ± 0.02 µs
//  ⚡ 19.9 µs 🐌 25.0 µs

// k: 7168; m: 16; n: 2112; seed: 15
//  ⏱ 64.5 ± 0.06 µs
//  ⚡ 63.1 µs 🐌 67.8 µs

// k: 512; m: 32; n: 4096; seed: 457
//  ⏱ 21.6 ± 0.02 µs
//  ⚡ 20.7 µs 🐌 26.2 µs

// k: 512; m: 32; n: 2880; seed: 54
//  ⏱ 21.0 ± 0.03 µs
//  ⚡ 20.1 µs 🐌 26.5 µs

// k: 2048; m: 64; n: 7168; seed: 687
//  ⏱ 44.4 ± 0.04 µs
//  ⚡ 42.8 µs 🐌 49.7 µs

// k: 1536; m: 256; n: 3072; seed: 7856
//  ⏱ 37.4 ± 0.04 µs
//  ⚡ 36.4 µs 🐌 47.9 µs
// ```