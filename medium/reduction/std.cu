#include <cuda_runtime.h>

inline __device__ __host__ unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define STRIDE_FACTOR 8
#define BLOCK_SIZE STRIDE_FACTOR*THREADS_PER_BLOCK

__device__ void warp_reduce(volatile float* smem, unsigned int tid){
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}

__global__ void reduction_kernel(const float* input, float* output, int N) {
    __shared__ float smem[THREADS_PER_BLOCK];
    auto tid = threadIdx.x;
    
    // // Stride reduction
    // auto block_start = blockIdx.x * BLOCK_SIZE;
    // auto sum = 0.0;
    // for (int i = 0; i < STRIDE_FACTOR; ++i) {
    //     auto idx = block_start + i * THREADS_PER_BLOCK + tid;
    //     if (idx < N) {
    //         sum += input[idx];
    //     }
    // }

    auto block_start = blockIdx.x * BLOCK_SIZE;
    auto sum = (block_start + tid) < N ? input[block_start + tid] : 0.0f;
    for(int i = 1; i < STRIDE_FACTOR; i++){
        auto idx = block_start + i * THREADS_PER_BLOCK + tid;
        if (idx < N){
            sum += input[idx];
        }
    }
    // smem[tid] = tid < N? sum:0.0f;

    // SMEM reduction
    smem[tid] = sum;
    __syncthreads();
    for (int s = THREADS_PER_BLOCK >> 1; s > WARP_SIZE; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // Warp reduction
    if (tid < WARP_SIZE) {
        warp_reduce(smem, tid);
    }

    // Grid reduction
    if (tid == 0) {
        atomicAdd(output, smem[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks(cdiv(N, THREADS_PER_BLOCK * STRIDE_FACTOR));
    reduction_kernel<<<blocks, threads>>>(input, output, N);
}