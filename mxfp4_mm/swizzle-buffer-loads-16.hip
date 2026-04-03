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

constexpr int NUM_WARPS = 1;

using fp4x2_t = __amd_fp4x2_storage_t;
using fp4x8_t = __amd_fp4x8_storage_t;
using fp4x64_t = fp4x2_t __attribute__((ext_vector_type(32)));  // 32 elements = 32 bytes = 64 FP4
using fp32x4_t = float __attribute__((ext_vector_type(4)));     // 4 floats
using mfma_input_t = int __attribute__((ext_vector_type(8)));   // 8 ints = 32 bytes
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;
using as3_uint8_ptr = uint8_t __attribute__((address_space(3)))*;
using as3_fp4x2_ptr = const fp4x2_t __attribute__((address_space(3)))*;


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

__device__ void buffer_fence(int32_t cnt)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
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
    constexpr int WARPS_COL = 1;
    constexpr int WARPS_ROW = 1;
    constexpr int BLOCK_SIZE_ROW = 16;
    constexpr int BLOCK_SIZE_COL = 16;
    constexpr int BLOCKSIZE = 16;
    constexpr int BLOCK_K = 128;
    const int blocks_per_row = M / BLOCK_SIZE_ROW; // Number of blocks per matrix row
    const int blocks_per_col = N / BLOCK_SIZE_COL; // Number of blocks per matrix col
    const int total_blocks_needed = blocks_per_row * blocks_per_col; // Total blocks needed
    const int k_iters = K / BLOCK_K; // K iterations
    const int NUM_THREADS = NUM_WARPS * 64;
    // constexpr int HALF_BLOCK_SIZE_ROW = BLOCK_SIZE_ROW / 2;
    // constexpr int HALF_BLOCK_SIZE_COL = BLOCK_SIZE_COL / 2;
    constexpr int REG_BLOCK_M = BLOCK_SIZE_ROW / WARPS_ROW;
    constexpr int REG_BLOCK_N = BLOCK_SIZE_COL / WARPS_COL;

    // Shared Memory Allocation 
    __shared__ fp4x2_t As[BLOCK_SIZE_ROW][BLOCK_K/2];
    __shared__ fp4x2_t Bs[BLOCK_SIZE_COL][BLOCK_K/2];
    __shared__ uint8_t a_scale[BLOCK_SIZE_ROW][BLOCK_K/32];
    __shared__ uint8_t b_scale[BLOCK_SIZE_COL][BLOCK_K/32];

    
    // Block Indices
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;


    const int lane = threadIdx.x & 63;          // [0, 63]
    const int row_in_tile = lane & 15;          // [0, 15]  (for MFMA register mapping)
    const int row_group = lane >> 4;            // [0, 3]   (for MFMA register mapping)
    const int k_byte_offset = row_group * 16;

    // Coalesced load mapping: 4 consecutive lanes load one row of 64 bytes
    // lane / 4 = which row (0-15), lane % 4 = which 16-byte chunk (0-3)
    const int load_row = lane / 4;          // 0-15
    const int load_col = (lane % 4) * 16;   // 0, 16, 32, 48
    // Scales: A_scale_tile is 16 rows × 4 bytes = 64 bytes
    // 64 lanes, each loads 1 byte
    const int scl_row = lane / 4;   // 0-15
    const int scl_col = lane % 4;   // 0-3


    // Buffer descriptors
    int32x4_t src_a = make_buffer_resource(A, M*K/2);
    int32x4_t src_b = make_buffer_resource(B, N*K/2);

    // lds pointers
    as3_uint32_ptr A_tile = (as3_uint32_ptr)(reinterpret_cast<uintptr_t>(As[0]));
    as3_uint32_ptr B_tile = (as3_uint32_ptr)(reinterpret_cast<uintptr_t>(Bs[0]));

    // compute swizzled offsets
    const uint32_t linear_byte_offset = load_row * BLOCK_K/2 + load_col;
    uint32_t swizzle = ((linear_byte_offset >> 8) & 3) << 4;
    uint32_t swizzled_offset = linear_byte_offset ^ swizzle;
    // swizzled_offsets_B = linear_byte_offset ^ swizzle;
    int sw_row = swizzled_offset / 64;
    int sw_col = swizzled_offset % 64;
    int voffset = sw_row * K/2 + sw_col;

    // precompute once, outside K-loop
    const int mfma_byte_off  = row_in_tile * 64 + k_byte_offset;
    const int mfma_swizzle   = ((mfma_byte_off >> 8) & 3) << 4;
    const int mfma_sw_off    = mfma_byte_off ^ mfma_swizzle;


    //acc registers
    fp32x4_t c_reg = {0.0f, 0.0f, 0.0f, 0.0f}; 

    //async loads
    // loop over the K dimension
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_K) {

        // buffer load for A_tile
        llvm_amdgcn_raw_buffer_load_lds(
            src_a, A_tile, 16, voffset,
            blockIdx.x * BLOCK_SIZE_ROW * K/2 + bkIdx/2, 0, 0);

        // buffer load for B_tile
        llvm_amdgcn_raw_buffer_load_lds(
            src_b, B_tile, 16, voffset,
            blockIdx.y * BLOCK_SIZE_COL * K/2 + bkIdx/2, 0, 0);

        {
            const int gm = cRow * BLOCKSIZE + scl_row;
            if (gm < M) {
                a_scale[scl_row][scl_col] = A_scale[gm * K/32 + (bkIdx / 32) + scl_col];
            } else {
                a_scale[scl_row][scl_col] = 0;
            }
        }
        {
            const int gn = cCol * BLOCKSIZE + scl_row;
            if (gn < N) {
                b_scale[scl_row][scl_col] = B_scale[gn * K/32 + (bkIdx / 32) + scl_col];
            } else {
                b_scale[scl_row][scl_col] = 0;
            }
        }
        
        
        buffer_fence(0);
        __builtin_amdgcn_s_barrier();

        fp4x64_t a_reg {};
        fp4x64_t b_reg {};

        // load a_fragment and b_fragment
        const fp4x2_t* ldg_a = reinterpret_cast<const fp4x2_t*>(As) + mfma_sw_off;
        const fp4x2_t* ldg_b = reinterpret_cast<const fp4x2_t*>(Bs) + mfma_sw_off;
        for (int i = 0; i < 16; i++) {
            a_reg[i] = *(ldg_a + i);
            b_reg[i] = *(ldg_b + i);
        }

        // Each lane loads only its own scale byte
        uint32_t scale_a = a_scale[row_in_tile][row_group];
        uint32_t scale_b = b_scale[row_in_tile][row_group];

        //mfma
        c_reg = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            reinterpret_cast<mfma_input_t&>(a_reg),
            reinterpret_cast<mfma_input_t&>(b_reg),
            c_reg, 4, 4, 0, scale_a, 0, scale_b);
        
        __syncthreads();
    }
    // write-back
    int out_col = blockIdx.y * BLOCK_SIZE_COL + (threadIdx.x % 16);
    for(int i=0; i < 4; i++) {
        int out_row = blockIdx.x * BLOCK_SIZE_ROW + (threadIdx.x / 16) * 4 + i;
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
    int BLOCK_SIZE = 16;
    dim3 block(64);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    mxfp4_gemm_kernel<<<grid, block>>>(
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


// ## Benchmarks:
// ```
// k: 512; m: 4; n: 2880; seed: 4565
//  ⏱ 43.8 ± 0.25 µs
//  ⚡ 36.7 µs 🐌 129 µs

// k: 7168; m: 16; n: 2112; seed: 15
//  ⏱ 70.7 ± 0.25 µs
//  ⚡ 64.3 µs 🐌 166 µs

// k: 512; m: 32; n: 4096; seed: 457
//  ⏱ 43.9 ± 0.36 µs
//  ⚡ 36.5 µs 🐌 145 µs

// k: 512; m: 32; n: 2880; seed: 54
//  ⏱ 44.5 ± 0.37 µs
//  ⚡ 36.2 µs 🐌 145 µs

// k: 2048; m: 64; n: 7168; seed: 687
//  ⏱ 71.4 ± 0.33 µs
//  ⚡ 57.8 µs 🐌 147 µs

// k: 1536; m: 256; n: 3072; seed: 7856
//  ⏱ 58.5 ± 0.18 µs
//  ⚡ 51.5 µs 🐌 114 µs
// ```