//#include "hip_utils.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

#define HIP_CHECK(cmd) do { hipError_t e = cmd; if(e != hipSuccess){ \
    std::cerr << "HIP error: " << hipGetErrorString(e) << std::endl; exit(EXIT_FAILURE);} } while(0)

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < N) {
        C[gid] = A[gid] + B[gid];
    }
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *hA = new float[N], *hB = new float[N], *hC = new float[N];
    for (int i = 0; i < N; ++i) { hA[i] = i * 0.001f; hB[i] = i * 0.002f;}

    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));

    HIP_CHECK(hipMemcpy(dA, hA, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, bytes, hipMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1)/ blockSize;
    hipLaunchKernelGGL(vectorAdd, dim3(gridSize), dim3(blockSize), 0, 0, dA, dB, dC, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hC, dC, bytes, hipMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < N; ++i) {
        float expected = hA[i] + hB[i];
        if (fabs(hC[i] - expected) > 1e-5) { correct = false; break;}
    }
    std::cout << (correct ? "Results Verified : CORRECT!\n" : "Verification FAILED!\n");

    delete[] hA; delete[] hB; delete[] hC;
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    return 0;
}

