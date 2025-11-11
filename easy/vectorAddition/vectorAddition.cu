#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
        C[idx] = A[idx] + B[idx];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

#include <vector>
#include <iostream>
int main() {
    const int N = 1 << 20;
    std::vector<float> hA(N, 1.0f), hB(N, 2.0f), hC(N);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, N * sizeof(float));
    cudaMalloc(&dB, N * sizeof(float));
    cudaMalloc(&dC, N * sizeof(float));
    cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    solve(dA, dB, dC, N);

    cudaMemcpy(hC.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    std::cout << hC[0] << std::endl;
    return 0;
}