/*
写个softmax，先要求max，再求和，再计算
求max因为没有atomicMaxFloat，所以要手写

手写需要用到atomicCAS，这个传入一个地址，一个旧值，一个新值，检查地址上的值是否等于旧值，是的话就写入新值
这本身是个原子操作，就是执行它的时候，其他的操作都动不了这个地址，但是在你启动它之前，旧值可能就变动了，所以要不断check
它的返回值是调用这个地址时的值，所以如果assumed!=old就说明写入失败了
由于这玩意只有int版本和unsigned long long，需要对应使用__float_as_int和__double_as_longlong

最后，对于sum和max的初始化，必须使用cudaMemset和cudaMemcpy，直接在host的主函数里搞是错的
*/

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define STRIDE_FACTOR 8
#define BLOCK_SIZE (THREADS_PER_BLOCK*STRIDE_FACTOR)

__device__ float atomicMaxFloat(float *addr, float val)
{
    int *address_as_int = (int*)addr;
    int old = *address_as_int;
    int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    }while(assumed != old);

    return __int_as_float(old);
}

__global__ void max_kernel(const float* input, float* maxVal, int N) {
    int tid = threadIdx.x;
    int start = blockIdx.x * BLOCK_SIZE;

    __shared__ float smem[THREADS_PER_BLOCK];

    float localmax = (start + tid) < N ? input[start + tid] : -__FLT_MAX__;
    for(int i = 0; i < STRIDE_FACTOR; ++i)
    {
        int idx = start + i * THREADS_PER_BLOCK + tid;
        if(idx < N)
            localmax = fmax(localmax, input[idx]);
    }

    smem[tid] = localmax;
    __syncthreads();

    for(int s = THREADS_PER_BLOCK >> 1; s > 0; s >>= 1)
    {
        if(tid < s)
            smem[tid] = fmax(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    if(tid == 0)
        atomicMaxFloat(maxVal, smem[0]);
}

__global__ void sum_kernel(const float* input, float *sum, float *max, int N) {
    int tid = threadIdx.x;
    int start = blockIdx.x * BLOCK_SIZE;

    __shared__ float smem[THREADS_PER_BLOCK];

    float localsum = 0;
    for(int i = 0; i < STRIDE_FACTOR; ++i)
    {
        int idx = start + i * THREADS_PER_BLOCK + tid;
        if(idx < N)
            localsum += expf(input[idx] - *max);
    }

    smem[tid] = localsum;
    __syncthreads();

    for(int s = THREADS_PER_BLOCK >> 1; s > 0; s >>= 1)
    {
        if(tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if(tid == 0)
        atomicAdd(sum, smem[0]);
}

__global__ void softmax_kernel(const float* input, float* output, float *sum, float *max, int N) {
    int tid = threadIdx.x;
    int start = blockIdx.x * BLOCK_SIZE;

    for(int i = 0; i < STRIDE_FACTOR; ++i)
    {
        int idx = start + i * THREADS_PER_BLOCK + tid;
        if(idx < N)
            output[idx] = expf(input[idx] - *max) / *sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *max;
    cudaMalloc(&max, sizeof(float));
    //*max = -FLT_MAX;
    float tmp = -__FLT_MAX__;
    cudaMemcpy(max, &tmp, sizeof(float), cudaMemcpyHostToDevice);

    max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, max, N);

    float *sum;
    cudaMalloc(&sum, sizeof(float));
    //*sum = 0;
    cudaMemset(sum, 0, sizeof(float));

    sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, sum, max, N);

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, sum, max, N);

    cudaDeviceSynchronize();

    cudaFree(sum);
    cudaFree(max);
}