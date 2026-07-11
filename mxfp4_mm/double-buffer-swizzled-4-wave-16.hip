#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_ext_ocp.h>


#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

constexpr int NUM_WARPS = 8;

using fp4x2_t = __amd_fp4x2_storage_t;
using fp4x64_t  = fp4x2_t __attribute__((ext_vector_type(32)));
using fp32x4_t = float __attribute__((ext_vector_type(4)));     // 4 floats
using mfma_input_t = int __attribute__((ext_vector_type(8)));   // 8 ints = 32 bytes
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;


// Buffer Descriptor
struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
};

__device__ inline int32x4_t make_buffer_resource(const void* ptr, uint32_t range_bytes) {
    buffer_resource rsrc = {
        reinterpret_cast<uint64_t>(ptr),
        range_bytes,
        0x110000};
    return *reinterpret_cast<const int32x4_t*>(&rsrc);
}

extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    int32x4_t rsrc, as3_uint32_ptr lds_ptr, int size, int voffset, int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");


// Loads one half-tile (HALF_BLOCK_SIZE_ROW x BLOCK_K fp4 elements = HALF_BLOCK_SIZE_ROW * BLOCK_K/2 bytes)
// from global memory into LDS using pre-computed swizzled offsets.
template<int HALF_BLOCK_SIZE, int BLOCK_K, int MEMCPY_PER_TILE>
__device__ inline void load_tile(fp4x2_t* dst, const int32x4_t src_rsrc, const uint32_t* swizzled_offsets,
                    const int block_idx, const int half, const int k, const int K) {
    constexpr int bytes_per_thread  = 16;
    constexpr int bytes_per_warp = bytes_per_thread * 64;

    const int warpid = threadIdx.x >> 6;
    fp4x2_t* warp_dst = dst + (warpid * bytes_per_warp);

    const uint32_t s_offset = (block_idx * 2 + half) * HALF_BLOCK_SIZE * K/2 + k*(BLOCK_K/2);

    #pragma unroll
    for (int i = 0; i < MEMCPY_PER_TILE; i++) {
        fp4x2_t* iter_ptr = warp_dst + (i * NUM_WARPS * bytes_per_warp);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(reinterpret_cast<uintptr_t>(iter_ptr));
        llvm_amdgcn_raw_buffer_load_lds(src_rsrc, lds_ptr, bytes_per_thread, swizzled_offsets[i], s_offset, 0, 0);
    }
}


// Loads a warp's slice of a half-tile from LDS into register tile
// src    : pointer to start of warp's subtile in LDS
// dst    : register tile e.g. a_reg[4] or b_reg[2]
// HEIGHT : number of 16-row base tiles (4 for A, 2 for B)
template<int HEIGHT>
__device__ inline void load_st_to_rt(fp4x64_t dst[HEIGHT], const fp4x2_t* src,
                        const int row_in_tile, const int row_group) {
    constexpr int subtile_bytes  = 16 * 64;
    constexpr int row_bytes      = 64;

    #pragma unroll
    for (int i = 0; i < HEIGHT; i++) {
        const uint32_t linear  = row_in_tile * row_bytes + row_group * 16;
        const uint32_t swizzled = linear ^ (((linear >> 8) & 3) << 4);
        const fp4x2_t* subtile_base = src + i * subtile_bytes;
        const fp4x2_t* ptr = subtile_base + swizzled;
        *reinterpret_cast<uint4*>(&dst[i])[0] = *reinterpret_cast<const uint4*>(ptr);
    }
}


// mfma_fp4_ABt: computes D = A * B^T + C for fp4 tiles with microscaling
// A_HEIGHT: 4 (a_reg rows), B_HEIGHT: 2 (b_reg rows), each base tile is 16×128 fp4
// c_reg[A_HEIGHT * B_HEIGHT][4] — 8 accumulators, each holding 4 floats/thread
template<int A_HEIGHT, int B_HEIGHT>
__device__ inline void mma_fp4_ABt(
    float             c_reg[A_HEIGHT * B_HEIGHT][4],
    const fp4x2_t     a_reg[A_HEIGHT],
    const fp4x2_t     b_reg[B_HEIGHT],
    const uint32_t    scale_a[A_HEIGHT],
    const uint32_t    scale_b[B_HEIGHT]
)
{
    #pragma unroll
    for (int i = 0; i < A_HEIGHT; i++) {
        #pragma unroll
        for (int j = 0; j < B_HEIGHT; j++) {

            *reinterpret_cast<fp32x4_t*>(c_reg[i * B_HEIGHT + j]) =
                __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                    *reinterpret_cast<const mfma_input_t*>(&a_reg[i]),
                    *reinterpret_cast<const mfma_input_t*>(&b_reg[j]),
                    *reinterpret_cast<const fp32x4_t*>(c_reg[i * B_HEIGHT + j]),
                    4, 4, 0, scale_a[i], 0, scale_b[j]
                );
        }
    }
}


//kernel
__global__ void mxfp4_gemm_kernel(
    const __amd_fp4x2_storage_t* __restrict__ A,
    const __amd_fp4x2_storage_t* __restrict__ B,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_scale,
    __hip_bfloat16* __restrict__ C,
    const int M,
    const int N,
    const int K
)
{
    constexpr int WARPS_COL = 4;
    constexpr int WARPS_ROW = 2;
    constexpr int BLOCK_SIZE_ROW = 256;
    constexpr int BLOCK_SIZE_COL = 256;
    constexpr int SUB_M = 16;
    constexpr int SUB_N = 16;
    constexpr int BLOCK_K = 128;
    const int gridDim_y = M / BLOCK_SIZE_ROW;
    const int gridDim_x = N / BLOCK_SIZE_COL;
    const int k_iters = K / BLOCK_K;
    constexpr int NUM_THREADS = NUM_WARPS * 64;
    constexpr int HALF_BLOCK_SIZE_ROW = BLOCK_SIZE_ROW / 2;         // 128
    constexpr int HALF_BLOCK_SIZE_COL = BLOCK_SIZE_COL / 2;         // 128
    constexpr int REG_BLOCK_M = BLOCK_SIZE_ROW / WARPS_ROW / 2;     // 64
    constexpr int REG_BLOCK_N = BLOCK_SIZE_COL / WARPS_COL / 2;     // 32

    // Shared Memory Allocation
    // Data Tiles 
    __shared__ fp4x2_t As[2][2][HALF_BLOCK_SIZE_ROW * BLOCK_K/2];           // 2x2x128x64 = 32KB
    __shared__ fp4x2_t Bs[2][2][HALF_BLOCK_SIZE_COL * BLOCK_K/2];
    //Scale tiles
    __shared__ uint8_t a_scale[2][2][HALF_BLOCK_SIZE_ROW * BLOCK_K/32];     // 2x2x128x4 = 2KB
    __shared__ uint8_t b_scale[2][2][HALF_BLOCK_SIZE_COL * BLOCK_K/32];

    // a_reg: 64x64 byte tile (4 x subtile_16x64)
    fp4x64_t a_reg[4] {};
    uint32_t scale_a[4];
    
    // b_reg: 32x64 byte tile (2 x subtile_16x64)
    fp4x64_t b_reg0[2] {};
    uint32_t scale_b0[2];
    fp4x64_t b_reg1[2] {};
    uint32_t scale_b1[2];

    // C: 64x32 fp32 tile (4x2 subtile_16x16 >> 8x4 vgprs per thread)
    float c_reg0[8][4] = {};
    float c_reg1[8][4] = {};
    float c_reg2[8][4] = {};
    float c_reg3[8][4] = {};

    //Indices
    uint block_row = blockIdx.x / gridDim_x;
    uint block_col = blockIdx.x % gridDim_x;
    const int warpid = (threadIdx.x >> 6) % NUM_WARPS;
    const int warp_m = warpid / WARPS_COL;
    const int warp_n = warpid % WARPS_COL;

    int tic = 0, toc = 1;

    const int laneid = threadIdx.x & 63;          // [0, 63]
    const int row_in_tile = laneid & 15;          // [0, 15]  (for MFMA register mapping)
    const int row_group = laneid >> 4;            // [0, 3]   (for MFMA register mapping)
    const int k_byte_offset = row_group * 16;
    
    // A/B scales cover 128 * 4 = 64 bytes per half tile (512 threads >> 1 byte each)
    const int scale_row = laneid / 4 + (warpid * 16);   // 0-127 across all warps
    const int scale_col = laneid % 4;                   // 0-3

    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int bytes_per_warp = bytes_per_thread * 64;

    // number of iterations to copy one tile
    constexpr int memcpy_per_tile_A = HALF_BLOCK_SIZE_ROW * BLOCK_K/2 / bytes_per_memcpy;
    constexpr int memcpy_per_tile_B = HALF_BLOCK_SIZE_COL * BLOCK_K/2 / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile_A];
    uint32_t swizzled_offsets_B[memcpy_per_tile_B];
    
    auto prefill_swizzled_offsets = [&](uint32_t* swizzled_offsets, int memcpy_per_tile) {
        for (int i = 0; i < memcpy_per_tile; i++){
            // one 16x64 byte subtile is fully consumed by one warp
            const uint32_t lane_byte_offset = (laneid * bytes_per_thread) + 
                                                (warpid * bytes_per_warp) + (i * NUM_WARPS * bytes_per_warp);
            const int subtile_id = lane_byte_offset / bytes_per_warp;
            const int subtile_row = subtile_id * SUB_M;
            // const int subtile_col = subtile_id % 1;
            const int linear = lane_byte_offset % bytes_per_warp;           //within the 16x64 subtile
            const uint32_t swizzled = linear ^ (((linear >> 8) & 3) << 4);

            // realtive to the half-tile
            const int row = swizzled / (BLOCK_K/2) + subtile_row;
            const int col = swizzled % (BLOCK_K/2);
            // final global byte offset
            swizzled_offsets[i] = row * K/2 + col;
        }
    };

    prefill_swizzled_offsets(swizzled_offsets_A, memcpy_per_tile_A);
    prefill_swizzled_offsets(swizzled_offsets_B, memcpy_per_tile_B);
    // Buffer descriptors
    int32x4_t src_a = make_buffer_resource(A, M*K/2);
    int32x4_t src_b = make_buffer_resource(B, N*K/2);

    auto load_scale_tiles = [&](int buf, int half, int k) {
        const int gk = k*(BLOCK_K/32) + scale_col;
        const int lds_idx = scale_row * (BLOCK_K/32) + scale_col;
        const int gm = block_row * BLOCK_SIZE_ROW + half * HALF_BLOCK_SIZE_ROW + scale_row;
        const int gn = block_col * BLOCK_SIZE_COL + half * HALF_BLOCK_SIZE_COL + scale_row;

        a_scale[buf][half][lds_idx] = (gm < M) ? A_scale[gm * (K/32) + gk] : 0;
        b_scale[buf][half][lds_idx] = (gn < N) ? B_scale[gn * (K/32) + gk] : 0;
    };

    auto load_scale_regs = [&](uint32_t* dst, const uint8_t* src, const int warp_offset, const int HEIGHT) {
        for (int i = 0; i < HEIGHT; i++) {
            const int row = warp_offset + i * 16 + row_in_tile;
            dst[i] = (uint32_t)src[row * BLOCK_K / 32 + row_group];
        }
    };



    //PROLOGUE
    load_tile<HALF_BLOCK_SIZE_ROW, memcpy_per_tile_A, BLOCK_K>(As[tic][0], src_a, swizzled_offsets_A, block_row, 0, 0, K);      // vmcnt += 1
    load_tile<HALF_BLOCK_SIZE_COL, memcpy_per_tile_B, BLOCK_K>(Bs[tic][0], src_b, swizzled_offsets_B, block_col, 0, 0, K);      // vmcnt += 1
    load_tile<HALF_BLOCK_SIZE_ROW, memcpy_per_tile_A, BLOCK_K>(As[tic][1], src_a, swizzled_offsets_A, block_row, 1, 0, K);      // vmcnt += 1       
    load_tile<HALF_BLOCK_SIZE_COL, memcpy_per_tile_B, BLOCK_K>(Bs[tic][1], src_b, swizzled_offsets_B, block_col, 1, 0, K);      // vmcnt += 1

    if (warp_m == 1) {
        __builtin_amdgcn_s_barrier();
    }

    load_scale_tiles(tic, 0, 0);
    asm volatile("s_waitcnt vmcnt(2)");   // wait until As[tic][0] and Bc[tic][0] are done
    __builtin_amdgcn_s_barrier();

    load_tile<HALF_BLOCK_SIZE_ROW, memcpy_per_tile_A, BLOCK_K>(As[toc][0], src_a, swizzled_offsets_A, block_row, 0, 1, K);      // vmcnt += 1
    load_tile<HALF_BLOCK_SIZE_COL, memcpy_per_tile_B, BLOCK_K>(Bs[toc][0], src_b, swizzled_offsets_B, block_col, 0, 1, K);      // vmcnt += 1
    load_tile<HALF_BLOCK_SIZE_ROW, memcpy_per_tile_A, BLOCK_K>(As[toc][1], src_a, swizzled_offsets_A, block_row, 0, 0, K);      // vmcnt += 1
    load_tile<HALF_BLOCK_SIZE_ROW, memcpy_per_tile_A, BLOCK_K>(Bs[toc][1], src_a, swizzled_offsets_A, block_row, 1, 1, K);      // vmcnt += 1
    load_scale_tiles(tic, 1, 0);        // lgkmcnt += 2
    // load_scale_regs(scale_b1, b_scale[tic][1], warp_n * REG_BLOCK_N, 2);
    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();
    // END PROLOGUE

    //HOT LOOP
    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic^=1, toc^=1) {

        const fp4x2_t* bs_subtile0 = &Bs[tic][0][warp_n * 2 * 16 * (BLOCK_K/2)];
        load_st_to_rt<2>(b_reg0 , bs_subtile0, row_in_tile, row_group);
        const fp4x2_t* as_subtile0 = &As[tic][0][warp_m * 4 * 16 * (BLOCK_K/2)];
        load_st_to_rt<4>(a_reg,  as_subtile0, row_in_tile, row_group);
        load_scale_regs(scale_a,  a_scale[tic][0], warp_m * REG_BLOCK_M, 4);
        load_scale_regs(scale_b0, b_scale[tic][0], warp_n * REG_BLOCK_N, 2);

        
        // load_scale_tiles(toc, 1, 0);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_fp4_ABt<4, 2>(c_reg0, a_reg, b_reg0, scale_a, scale_b0);        // cA


    }
    // write-back
    // int out_col = blockIdx.y * BLOCK_SIZE_COL + (threadIdx.x % 16);
    // for(int i=0; i < 4; i++) {
    //     int out_row = blockIdx.x * BLOCK_SIZE_ROW + (threadIdx.x / 16) * 4 + i;
    //     if (out_row < M && out_col < N) {
    //         C[out_row * N + out_col] = __float2bfloat16(c_reg[i]);
    //     }
    // }

}

void mxfp4_gemm(
    torch::Tensor A_q, torch::Tensor B_q,
    torch::Tensor A_scale, torch::Tensor B_scale,
    torch::Tensor C, int M, int N, int K
) { 
    int BLOCK_SIZE = 256;
    constexpr int threads_per_warp = 64;
    constexpr int warps_per_cu = 8;
    constexpr int threads_per_block = threads_per_warp * warps_per_cu;
    dim3 grid(M*N / BLOCK_SIZE*BLOCK_SIZE);
    mxfp4_gemm_kernel<<<grid, threds_per_block>>>(
        reinterpret_cast<const __amd_fp4x2_storage_t*>(A_q.data_ptr()),
        reinterpret_cast<const __amd_fp4x2_storage_t*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(A_scale.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
        M, N, K
    );
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}