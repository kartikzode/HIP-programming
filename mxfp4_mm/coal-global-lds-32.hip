#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_ext_ocp.h>

#define TILE_M 32
#define TILE_N 32
#define TILE_K 64
#define BLOCKSIZE 32

using fp4x2_t = __amd_fp4x2_storage_t;
using fp4x64_t = fp4x2_t __attribute__((ext_vector_type(32)));  // 32 elements = 32 bytes = 64 FP4
using fp4x32_t = fp4x2_t __attribute__((vector_size(16)));      // 16 elements = 16 bytes = 32 FP4
using fp32x16_t = __attribute__((vector_size(16 * sizeof(float)))) float;     // 4 floats
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

    __shared__ uint8_t A_tile[TILE_M][TILE_K / 2];
    __shared__ uint8_t B_tile[TILE_N][TILE_K / 2];

    __shared__ uint8_t A_scale_tile[TILE_M][TILE_K / 32];  // [32][2]
    __shared__ uint8_t B_scale_tile[TILE_N][TILE_K / 32];  // [32][2]


    const int lane = threadIdx.x & 63;          // [0, 63]
    const int row_in_tile = lane & 31;          // [0, 32]
    const int row_group = lane >> 5;            // [0, 1]
    const int k_byte_offset = row_group * 16;

    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    int global_m = cRow * BLOCKSIZE + row_in_tile;
    int global_n = cCol * BLOCKSIZE + row_in_tile;

    const int K_half = K / 2;     // bytes per row in A/B
    const int K_scale = K / 32;   // scale bytes per row

    //acc registers
    fp32x16_t c_reg = {};

    // loop over the K dimension
    for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_K) {

        // --- Coalesced global → LDS loads ---
        // A_tile is 32 rows × 32 bytes = 1024 bytes total
        // 64 lanes × 16 bytes each = 1024 bytes — one load covers the whole tile
        // Lane i loads 16 bytes starting at flat byte offset i*16
        {
            const int flat_offset = lane * 16;              // [0, 1008] step 16
            const int lds_row = flat_offset / (TILE_K / 2); // which row in tile (0-31)
            const int lds_col = flat_offset % (TILE_K / 2); // byte offset within row (0 or 16)
            const int gm = cRow * BLOCKSIZE + lds_row;
            if (gm < M) {
                const fp4x32_t a_vec = load_fp4x32(A + gm * K_half + (bkIdx / 2) + lds_col);
                *reinterpret_cast<fp4x32_t*>(&A_tile[lds_row][lds_col]) = a_vec;
            } else {
                *reinterpret_cast<fp4x32_t*>(&A_tile[lds_row][lds_col]) = {};
            }
        }
        {
            const int flat_offset = lane * 16;
            const int lds_row = flat_offset / (TILE_K / 2);
            const int lds_col = flat_offset % (TILE_K / 2);
            const int gn = cCol * BLOCKSIZE + lds_row;
            if (gn < N) {
                const fp4x32_t b_vec = load_fp4x32(B + gn * K_half + (bkIdx / 2) + lds_col);
                *reinterpret_cast<fp4x32_t*>(&B_tile[lds_row][lds_col]) = b_vec;
            } else {
                *reinterpret_cast<fp4x32_t*>(&B_tile[lds_row][lds_col]) = {};
            }

        // Scales: A_scale_tile is 32 rows × 2 bytes = 64 bytes
        // 64 lanes, each loads 1 byte
        {
            const int scl_row = lane / 2;   // 0-31
            const int scl_col = lane % 2;   // 0 or 1
            const int gm = cRow * BLOCKSIZE + scl_row;
            if (gm < M) {
                A_scale_tile[scl_row][scl_col] = A_scale[gm * K_scale + (bkIdx / 32) + scl_col];
            } else {
                A_scale_tile[scl_row][scl_col] = 0;
            }
        }
        {
            const int scl_row = lane / 2;
            const int scl_col = lane % 2;
            const int gn = cCol * BLOCKSIZE + scl_row;
            if (gn < N) {
                B_scale_tile[scl_row][scl_col] = B_scale[gn * K_scale + (bkIdx / 32) + scl_col];
            } else {
                B_scale_tile[scl_row][scl_col] = 0;
            }
        }

        __syncthreads();

        fp4x64_t a_reg {};
        fp4x64_t b_reg {};

        // load a_fragment and b_fragment
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
        c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            reinterpret_cast<mfma_input_t&>(a_reg),
            reinterpret_cast<mfma_input_t&>(b_reg),
            c_reg, 4, 4, 0, scale_a, 0, scale_b);
        
        __syncthreads();

        }
        // write-back
        // For 32x32x64 MFMA, c_reg has 16 floats.
        // Output layout (from blog): c_reg[i*4 + j] maps to:
        //   row = row_group * 4 + j + i * 8
        //   col = lane % 32
        // row_group=0: rows 0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27
        // row_group=1: rows 4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31
        int out_col = cCol * BLOCKSIZE + (threadIdx.x % 32);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int out_row = cRow * BLOCKSIZE + row_group * 4 + j + i * 8;
                if (out_row < M && out_col < N) {
                    C[out_row * N + out_col] = __float2bfloat16(c_reg[i * 4 + j]);
                }
            }
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


// EXTRA ALU eats the GAINs by lane pairwise coalescing
// ## Benchmarks:
// ```
// k: 512; m: 4; n: 2880; seed: 4565
//  ⏱ 23.8 ± 0.03 µs
//  ⚡ 23.0 µs 🐌 32.2 µs

// k: 7168; m: 16; n: 2112; seed: 15
//  ⏱ 115 ± 0.1 µs
//  ⚡ 113 µs 🐌 118 µs

// k: 512; m: 32; n: 4096; seed: 457
//  ⏱ 25.4 ± 0.03 µs
//  ⚡ 24.5 µs 🐌 32.1 µs

// k: 512; m: 32; n: 2880; seed: 54
//  ⏱ 25.1 ± 0.03 µs
//  ⚡ 24.2 µs 🐌 30.0 µs

// k: 2048; m: 64; n: 7168; seed: 687
//  ⏱ 62.1 ± 0.06 µs
//  ⚡ 60.8 µs 🐌 69.4 µs

// k: 1536; m: 256; n: 3072; seed: 7856
//  ⏱ 50.2 ± 0.05 µs
//  ⚡ 49.0 µs 🐌 55.3 µs
// ```

