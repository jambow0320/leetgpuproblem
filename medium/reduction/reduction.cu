#include <cuda_runtime.h>

// input, output are device pointers

#define THREADS_PER_BLOCK 256

__global__ void reduction_kernel(const float *input, float *output, int N)
{
    __shared__ float smem[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int start = blockIdx.x * blockDim.x + threadIdx.x;

    if(start >= N) return ;

    ;
}

extern "C" void solve(const float* input, float* output, int N) {  
    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((N + threads.x - 1) / threads.x);

    reduction_kernel<<<blocks, threads>>>(input, output, N);
}