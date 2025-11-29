#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void SWiGLU_kernel(const float *input, float *output, int halfN)
{
    int tid = threadIdx.x;
    int pos = blockIdx.x * blockDim.x + tid;
    float x1, x2;
    if(pos < halfN)
    {
        x1 = input[pos];
        x2 = input[pos + halfN];
        output[pos] = x1 * x2 / (1 + expf(-x1));
    }
}

extern "C" void solve(const float *input, float *output, int N)
{
    int halfN = N / 2;
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((halfN + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    SWiGLU_kernel<<<grid, block>>>(input, output, halfN);
}