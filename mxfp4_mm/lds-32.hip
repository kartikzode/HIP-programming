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

    // pointers advanced to starting positions
    const fp4x2_t* a_ptr = A + (cRow * BLOCKSIZE) * K/2;
    const fp4x2_t* b_ptr = B + (cCol * BLOCKSIZE) * K/2;
    // const __hip_bfloat16* C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    // pointers advanced to strating scale positions
    const uint8_t* a_scale_ptr = A_scale + (cRow * BLOCKSIZE) * K/32;
    const uint8_t* b_scale_ptr = B_scale + (cCol * BLOCKSIZE) * K/32;

    //acc registers
    fp32x16_t c_reg = {};    

    // loop over the K dimension
    for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_K) {

        const fp4x2_t* a_row = a_ptr + (row_in_tile * K/2);
        const fp4x2_t* b_row = b_ptr + (row_in_tile * K/2);
        
        //load A_tile : global to lds
        if (global_m < M) {
            const fp4x32_t a_vec = load_fp4x32(a_row + k_byte_offset);
            *reinterpret_cast<fp4x32_t*>(&A_tile[row_in_tile][k_byte_offset]) = a_vec;
        } else {
            *reinterpret_cast<fp4x32_t*>(&A_tile[row_in_tile][k_byte_offset]) = {};
        }

        // load b_tile : global to lds
        if (global_n < N) {
            const fp4x32_t b_vec = load_fp4x32(b_row + k_byte_offset);
            *reinterpret_cast<fp4x32_t*>(&B_tile[row_in_tile][k_byte_offset]) = b_vec;
        } else {
            *reinterpret_cast<fp4x32_t*>(&B_tile[row_in_tile][k_byte_offset]) = {};
        }

        //load scales
        if (global_m < M) {
            A_scale_tile[row_in_tile][row_group] = a_scale_ptr[row_in_tile * K/32 + row_group];
        } else {
            A_scale_tile[row_in_tile][row_group] = 0;
        }

        if (global_n < N) {
            B_scale_tile[row_in_tile][row_group] = b_scale_ptr[row_in_tile * K/32 + row_group];
        } else {
            B_scale_tile[row_in_tile][row_group] = 0;
        }

        __syncthreads();

        a_ptr += TILE_K/2;
        b_ptr += TILE_K/2;
        a_scale_ptr += TILE_K/32;
        b_scale_ptr += TILE_K/32;

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
//  ⏱ 22.6 ± 0.02 µs
//  ⚡ 21.8 µs 🐌 29.5 µs

// k: 7168; m: 16; n: 2112; seed: 15
//  ⏱ 89.9 ± 0.09 µs
//  ⚡ 88.5 µs 🐌 92.7 µs

// k: 512; m: 32; n: 4096; seed: 457
//  ⏱ 23.7 ± 0.02 µs
//  ⚡ 22.9 µs 🐌 28.9 µs

// k: 512; m: 32; n: 2880; seed: 54
//  ⏱ 23.2 ± 0.02 µs
//  ⚡ 22.4 µs 🐌 28.5 µs

// k: 2048; m: 64; n: 7168; seed: 687
//  ⏱ 50.0 ± 0.05 µs
//  ⚡ 48.8 µs 🐌 55.2 µs

// k: 1536; m: 256; n: 3072; seed: 7856
//  ⏱ 41.6 ± 0.04 µs
//  ⚡ 40.6 µs 🐌 46.9 µs
// ```