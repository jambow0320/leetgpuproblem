#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void rmsnorm_kernel(const float *input, float gamma, float beta, float *output, int N, float eps)
{
    int tid = threadIdx.x;
    float localsum = 0;
    for(int i = tid; i < N; i += blockDim.x)
        localsum += input[i] * input[i];

    __shared__ float smem[THREADS_PER_BLOCK];
    smem[tid] = localsum;
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if(tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if(tid == 0)
    {
        smem[0] = sqrtf(smem[0] / N + eps);
    }
    __syncthreads();

    localsum = smem[0];
    for(int i = tid; i < N; i += blockDim.x)
        output[i] = input[i] * gamma / localsum + beta;
}

extern "C" void solve(const float *input, float gamma, float beta, float *output, int N, float eps)
{
    dim3 grid(1);
    dim3 block(THREADS_PER_BLOCK);

    rmsnorm_kernel<<<grid, block>>>(input, gamma, beta, output, N, eps);
}