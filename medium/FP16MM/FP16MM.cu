/*
FP16的gemm，中间结果用float，最后再改回去就行
注意block的xyz顺序是z在外层，也就是类似
for(z)
    for(y)
        for(x)
            kernel(...)
所以我们一般把batch放z，行放y，列放x

然后就是具体怎么实现，还是切tile，先对最终的矩阵切16*16，一个block负责它，256中每个线程负责这个16*16中的一个元素
然后这个(row, col)元素，它应该是A中的一行以及B中的一列相乘嘛，但是我们没必要每个线程都去取一行一列，这样太浪费了
一个16*16的块里，其他的线程也需要去取A中的行B中的列，这完全可以复用
所以我们每次取A中的16行，B中的16列，具体来说也是16*16的去取，然后存进shared_mem里，这样每个线程再从shared里取并计算自己的
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

__global__ void gemm_fp16(const half *A, const half *B, half *C, int M, int N, int K)
{
    int b = blockIdx.z;
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float smemA[TILE_M][TILE_K];
    __shared__ float smemB[TILE_K][TILE_N];

    float acc = 0;
    for(int k0 = 0; k0 < K; k0 += TILE_K)
    {
        int kA = k0 + threadIdx.x;
        if(kA < K && row < M)
        {
            smemA[threadIdx.y][threadIdx.x] = __half2float(A[b * M * K + row * K + kA]);
        }
        else smemA[threadIdx.y][threadIdx.x] = 0;

        int kB = k0 + threadIdx.y;
        if(kB < K && col < N)
        {
            smemB[threadIdx.y][threadIdx.x] = __half2float(B[b * K * N + kB * N + col]);
        }
        else smemB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();
        for(int kk = 0; kk < TILE_K; ++kk)
            acc += smemA[threadIdx.y][kk] * smemB[kk][threadIdx.x];

        __syncthreads();
    }

    if(row < M && col < N)
        C[b * M * N + row * N + col] = __float2half(acc);
}

extern "C" void solve(const half *A, const half *B, half *C, int BATCH, int M, int N, int K)
{
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, BATCH);

    gemm_fp16<<<grid, block>>>(A, B, C, M, N, K);
}