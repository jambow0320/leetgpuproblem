#include <cuda_runtime.h>

#define THREADS_PER_BLOCKS 256

__global__ void mha_kernel(const float *Q, 
                           const float *K, 
                           const float *V, 
                           float *output, 
                           int N, 
                           int d_model, 
                           int h,
                           int head_dim, 
                           float inv_sqrtd)
{
    int tid = threadIdx.x;
    int Q_bias = blockIdx.x * head_dim;
    int KV_bias = (blockIdx.x % h) * head_dim;
    const float *startQ = Q_bias + Q;

    extern __shared__ float scores[];

    for(int i = tid; i < N; i += blockDim.x)
    {
        float sum = 0;
        for(int d = 0; d < head_dim; ++d)
            sum += startQ[d] * K[i * d_model + KV_bias + d];
        scores[i] = sum * inv_sqrtd;
    }
    __syncthreads();

    __shared__ float row_max[THREADS_PER_BLOCKS];
    float local_max = -__FLT_MAX__;
    for(int i = tid; i < N; i += blockDim.x)
    {
        local_max = fmaxf(local_max, scores[i]);
    }
    row_max[tid] = local_max;
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if(tid < s)
            row_max[tid] = fmaxf(row_max[tid], row_max[tid + s]);
        __syncthreads();
    }

    __shared__ float row_sum[THREADS_PER_BLOCKS];
    float local_sum = 0;
    local_max = row_max[0];
    for(int i = tid; i < N; i += blockDim.x)
    {
        float e = expf(scores[i] - local_max);
        scores[i] = e;
        local_sum += e;
    }
    row_sum[tid] = local_sum;
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if(tid < s)
            row_sum[tid] += row_sum[tid + s];
        __syncthreads();
    }

    local_sum = row_sum[0];
    for(int i = tid; i < N; i += blockDim.x)
    {
        scores[i] = scores[i] / local_sum;
    }
    __syncthreads();

    for(int d = tid; d < head_dim; d += blockDim.x)
    {
        float val = 0;
        for(int i = 0; i < N; ++i)
            val += scores[i] * V[i * d_model + KV_bias + d];
        output[Q_bias + d] = val;
    }
}

extern "C" void solve(const float *Q, const float *K, const float *V, float *output, int N, int d_model, int h)
{
    int head_dim = d_model / h;
    float inv_sqrtd = 1. / sqrtf((float)head_dim);

    int grid = N * h;
    int block = THREADS_PER_BLOCKS;

    mha_kernel<<<grid, block, N * sizeof(float)>>>(Q, K, V, output, N, d_model, h, head_dim, inv_sqrtd);
}