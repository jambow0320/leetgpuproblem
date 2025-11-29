/*
采用tile的方式，首先要确定block的大小，这个玩意必须是个正方形，
因为我们写入的时候xy坐标不交换（保证列写是连续的），必须保证两维相等
16*16刚好等于256个线程，但是这样一个线程负责一个元素干的事情太少了，不好打满
所以使用32*32，这样一个线程要负责4个元素
我们在行上取4个元素，因为这样做我实际的block是8*32，在列上刚好是一个warp32线程，比较方便取连续元素

对于ty和tx，只需要想清楚我的threadIdx.y和x是不交换的(保证列写是连续)，然后它一开始的块位置需要交换
同时smem里也对应交换一下就好了
*/

#include <cuda_runtime.h>

constexpr int TILE = 32;
constexpr int NUM_PER_THREAD = 4;

__global__ void matrix_transpose_kernel(const float *input, float *output, int rows, int cols)
{
    int y = blockIdx.y * TILE + threadIdx.y;
    int x = blockIdx.x * TILE + threadIdx.x;

    __shared__ float smem[TILE][TILE + 1];

    for(int i = 0; i < TILE; i += blockDim.y)
    {
        if(y + i < rows && x < cols)
            smem[threadIdx.y + i][threadIdx.x] = input[(y + i) * cols + x];
    }
    __syncthreads();

    int ty = blockIdx.x * TILE + threadIdx.y;
    int tx = blockIdx.y * TILE + threadIdx.x;
    for(int i = 0; i < TILE; i += blockDim.y)
        if(ty + i < cols && tx < rows)
            output[(ty + i) * rows + tx] = smem[threadIdx.x][threadIdx.y + i];
}

extern "C" void solve(const float *input, float *output, int rows, int cols)
{
    dim3 block(TILE, TILE / NUM_PER_THREAD);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    matrix_transpose_kernel<<<grid, block>>>(input, output, rows, cols);
}