#include <cuda_runtime.h>

constexpr int THREADS_PER_BLOCK = 256;

__global__ void causal_attn(const float *Q, const float *K, const float *V, float *output, int M, int d, const float inv_sqrtd)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    const float *q0 = Q + bid * d;

    extern __shared__ float smem[];
    __shared__ double smem_acc[THREADS_PER_BLOCK];

    for(int k = 0; k <= bid; ++k)
    {
        const float *k0 = K + k * d;
        double acc = 0.0;
        for(int i = tid; i < d; i += blockDim.x)
        {
            acc += q0[i] * k0[i];
        }

        smem_acc[tid] = acc;
        __syncthreads();

        for(int s = blockDim.x >> 1; s > 0; s >>= 1)
        {
            if(tid < s)
                smem_acc[tid] += smem_acc[tid + s];
            __syncthreads();
        }

        if(tid == 0) smem[k] = static_cast<float>(smem_acc[0] * inv_sqrtd);
    }
    __syncthreads();
    
    float localmax = -__FLT_MAX__;
    for(int i = tid; i <= bid; i += blockDim.x)
    {
        localmax = fmaxf(localmax, smem[i]);
    }

    __shared__ float smem_max[THREADS_PER_BLOCK];
    smem_max[tid] = localmax;
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if(tid < s)
            smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + s]);
        __syncthreads();
    }

    localmax = smem_max[0];
    double localsum = 0.0;

    for(int i = tid; i <= bid; i += blockDim.x)
    {
        smem[i] = expf(smem[i] - localmax);
        localsum += static_cast<double>(smem[i]);
    }
    smem_acc[tid] = localsum;
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if(tid < s)
            smem_acc[tid] += smem_acc[tid + s];
        __syncthreads();
    }

    float inv_sum = 1.0f / static_cast<float>(smem_acc[0]);
    for(int i = tid; i < d; i += blockDim.x)
    {
        double acc = 0.0;
        for(int k = 0; k <= bid; ++k)
        {
            float score = smem[k] * inv_sum;
            acc += score * V[k * d + i];
        }
        output[bid * d + i] = static_cast<float>(acc);
    }

}

extern "C" void solve(const float *Q, const float *K, const float *V, float *output, int M, int d)
{
    dim3 grid(M);
    dim3 block(THREADS_PER_BLOCK);

    float inv_sqrtd = 1 / sqrtf(d);

    causal_attn<<<grid, block, M * sizeof(float)>>>(Q, K, V, output, M, d, inv_sqrtd);
}
