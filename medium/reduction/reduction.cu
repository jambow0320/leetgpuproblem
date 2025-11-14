/*
要实现一个一行的reduce

先确定块的大小，还是经典一个块256线程，但是如果一个线程只取一个元素也太浪费了，
所以我们一个线程取STRIDE_FACTOR=8个元素
为了使warp内访问连续，我们还是得idx、idx+256、idx+512...这样取
实际一个块负责的元素是8*256=2048的

然后我们开始规约，先逐步规约到64，也就是warpsize32的两倍
因为对于最后32个，它是一个warp里的，它是天然同步的，(代码里的同一行会在32个线程同时进行
比如smem[tid]+=smem[tid+32]对应smem[0]+=smem[32]、smem[1]+=smem[33]、...这些都是同时进行)
不需要sync，所以为了加速单独处理warp内的

在执行一个warp里的归约时，必须使用volatile，不然编译器的自动优化会导致答案错误
比如说我smem[31] += smem[31 + 32]了，
但是对于tid=31的进程，它因为没看见sync，觉得暂时不会影响到其他进程，就会为了速度不把smem[31]及时写回去，
实际上，这里的smem[31]被编译器自动优化成了31号线程内的寄存器，而不是真正共享内存上的东西
而这会影响到接下来15号进程的smem[15] += smem[15+16]，因为真正的smem[31]还在31号进程的寄存器上呢
所以我们要加volatile，表示这里禁止把smem优化成寄存器缓存，使每次读写都触及真正的shared memory
*/

#include <cuda_runtime.h>

#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define STRIDE_FACTOR 8
#define BLOCK_SIZE THREADS_PER_BLOCK *STRIDE_FACTOR

__global__ void reduction_kernel(const float *input, float *output, int N)
{
    __shared__ float smem[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int start = blockIdx.x * BLOCK_SIZE;

    float sum = 0.;
    for (int i = 0; i < STRIDE_FACTOR; ++i)
    {
        int idx = start + i * THREADS_PER_BLOCK + tid;
        if (idx < N)
            sum += input[idx];
    }

    smem[tid] = sum;
    __syncthreads();

    for (int s = THREADS_PER_BLOCK >> 1; s > WARP_SIZE; s >>= 1)
    {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid < WARP_SIZE)
    {
        volatile float *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0)
    {
        atomicAdd(output, smem[0]);
    }
}

extern "C" void solve(const float *input, float *output, int N)
{
    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    reduction_kernel<<<blocks, threads>>>(input, output, N);
}