#include <cstdio>
#include <cuda_fp16.h>

#define CHECK_CUDA(call)                                                     \
do {                                                                         \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                cudaGetErrorString(err));                                    \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

// 简单起见：统一用 16×16×16 的 tile
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

// A: [B, M, K] (row-major)
// B: [B, K, N]
// C: [B, M, N]
// half 存储，float 累加
__global__ void batched_gemm_fp16_tiled(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int Bsize, int M, int K, int N)
{
    int b   = blockIdx.z;                              // batch 维
    if (b >= Bsize) return;

    int row = blockIdx.y * TILE_M + threadIdx.y;       // 全局行索引 (0..M-1)
    int col = blockIdx.x * TILE_N + threadIdx.x;       // 全局列索引 (0..N-1)

    // shared memory tiles，直接用 float 存（已经是 FP32）
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    float acc = 0.0f;  // FP32 累加器

    // K 方向分块循环
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // ---- 加载 A 的一个 tile 到 shared memory ----
        int kA = k0 + threadIdx.x;  // 本线程负责的 K 维位置
        if (row < M && kA < K) {
            int idxA = b * (M * K) + row * K + kA; // A[b, row, kA]
            As[threadIdx.y][threadIdx.x] = __half2float(A[idxA]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ---- 加载 B 的一个 tile 到 shared memory ----
        int kB = k0 + threadIdx.y;
        if (col < N && kB < K) {
            int idxB = b * (K * N) + kB * N + col; // B[b, kB, col]
            Bs[threadIdx.y][threadIdx.x] = __half2float(B[idxB]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // ---- 当前 tile 上做小矩阵乘 ----
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }

        __syncthreads(); // 用完这个 tile 再加载下一块
    }

    // ---- 写回 C，注意边界 ----
    if (row < M && col < N) {
        int idxC = b * (M * N) + row * N + col; // C[b, row, col]
        C[idxC] = __float2half_rn(acc);
    }
}

// 一个简单的 host 端示例
int main() {
    // 你可以自己改这几个参数试
    int Bsize = 3;   // batch size
    int M = 37;
    int K = 45;
    int N = 29;

    size_t sizeA = (size_t)Bsize * M * K * sizeof(half);
    size_t sizeB = (size_t)Bsize * K * N * sizeof(half);
    size_t sizeC = (size_t)Bsize * M * N * sizeof(half);

    half* dA = nullptr;
    half* dB = nullptr;
    half* dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeB));
    CHECK_CUDA(cudaMalloc(&dC, sizeC));

    // 为了简单，用 host buffer 初始化成 1.0
    half* hA = (half*)malloc(sizeA);
    half* hB = (half*)malloc(sizeB);
    half* hC = (half*)malloc(sizeC);

    for (int b = 0; b < Bsize; ++b) {
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                hA[b*M*K + i*K + k] = __float2half(1.0f);
            }
        }
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                hB[b*K*N + k*N + j] = __float2half(1.0f);
            }
        }
    }

    CHECK_CUDA(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));

    // 配置 kernel 启动参数
    dim3 block(TILE_N, TILE_M); // (x→N方向, y→M方向)
    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M,
        Bsize
    );

    batched_gemm_fp16_tiled<<<grid, block>>>(dA, dB, dC, Bsize, M, K, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost));

    // 简单检查一下结果：如果 A/B 都是 1，
    // C[b, i, j] 理论值应该是 K（因为 sum_{t=0}^{K-1} 1*1 = K）
    printf("C[0,0,0] = %f (expected ~%d)\n",
           __half2float(hC[0]), K);

    // 你也可以多打印几个值看看
    int b = 1, i = 10, j = 5;
    int idx = b * (M * N) + i * N + j;
    printf("C[%d,%d,%d] = %f (expected ~%d)\n",
           b, i, j, __half2float(hC[idx]), K);

    free(hA);
    free(hB);
    free(hC);
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    return 0;
}
