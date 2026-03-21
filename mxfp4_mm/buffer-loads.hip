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
using i32x4_t = int32_t __attribute__((ext_vetcor_tyoe(4)));
using u32x4_t = uint32x_t __attribute__((ext_vector_type(4)));
using as3 uint32_ptr = uint32_t __attribute__((address_space(3)))*;

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

    __shared__ uint8_t A_scale_tile[TILE_M][TILE_K / 32];  // [16][4]
    __shared__ uint8_t B_scale_tile[TILE_N][TILE_K / 32];  // [16][4]


    const int lane = threadIdx.x & 63;          // [0, 63]
    const int row_in_tile = lane & 15;          // [0, 15]
    const int row_group = lane >> 4;            // [0, 3]
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

    // // stride = number of bytes per row in the input tile
    // const int stride = TILE_K / 2;
    // // scale_stride; one scale value per 32 elements 
    // const int scale_stride = TILE_K / 32;\

    //acc registers
    fp32x4_t c_reg = {0.0f, 0.0f, 0.0f, 0.0f};    

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