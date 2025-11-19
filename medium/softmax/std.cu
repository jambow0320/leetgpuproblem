#include <cuda_runtime.h>
#include <cfloat>

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void max_kernel(const float* input, float* maxVal, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float sf[];
    sf[tid] = idx < N ? input[idx] : FLT_MIN;
    *maxVal = sf[tid];
    __syncthreads();

    for (int gap = blockDim.x >> 1; gap > 0; gap >>= 1) {
        if (tid < gap && sf[tid] < sf[tid + gap]) {
            sf[tid] = sf[tid + gap];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(maxVal, sf[0]);
    }
}

__global__ void sum_kernel(const float* input, double* sum, float* maxVal, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ double sd[];
    sd[tid] = idx < N ? expf(static_cast<double>(input[idx]) - *maxVal) : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sd[tid] += sd[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, sd[0]);
    }
}

__global__ void softmax_kernel(const float* input, float* output, float* maxVal, double* sum, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = static_cast<float>(expf(static_cast<double>(input[idx]) - *maxVal) / *sum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *maxVal;
    cudaMalloc((void **)&maxVal, sizeof(float));
    cudaMemset(maxVal, 0, sizeof(float));
    max_kernel<<<blocksPerGrid, threadsPerBlock, sizeof(float) * threadsPerBlock>>>(input, maxVal, N);

    double *sum;
    cudaMalloc((void **)&sum, sizeof(double));
    cudaMemset(sum, 0, sizeof(double));
    sum_kernel<<<blocksPerGrid, threadsPerBlock, sizeof(double) * threadsPerBlock>>>(input, sum, maxVal, N);

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, maxVal, sum, N);
}
