#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff

__device__ double warpReduce(double val) {
    for (int offset = warpSize/2; 0 < offset; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__inline__ __device__ double blockReduceSum(double val) {
    static __shared__ double shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduce(val);

    if (0 == lane) {
        shared[wid] = val;
    }

    __syncthreads();
    // warpNum = blockDim.x/warpSize
    val = (threadIdx.x < blockDim.x/warpSize) ? shared[threadIdx.x] : 0;

    if (0 == wid) {
        val = warpReduce(val);
    }
    return val;
}

__global__ void deviceReduceKernel(const float* in, float* out, int N) {
    double sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Thread %d: sum: %d\n", tid, sum);
    sum = blockReduceSum(sum);
    //printf("Thread %d: sum: %d after blockReduceSum\n", tid, sum);

    if (tid == 0) {
        out[tid] = (float)sum;
    }
}

// input, output are device pointers
extern "C" void solve_std(const float* input, float* output, int N) {  
    int threadsPerBlock = 32*8;
    deviceReduceKernel<<<1, threadsPerBlock>>>(input, output, N);
}
