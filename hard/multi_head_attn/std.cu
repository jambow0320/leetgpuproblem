#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// ======================================================================
// Device kernel: computes Multi-Head Self-Attention
// ======================================================================
__global__ void mha_kernel(const float* Q, const float* K, const float* V, float* output,
                           int N, int d_model, int h, int head_dim, float inv_sqrt_hd) {
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * (long long)h;
    if (gid >= total) return;

    int head_idx = gid / N;
    int q_idx = gid % N;

    int head_offset = head_idx * head_dim;
    const float* Qrow = Q + ((long long)q_idx * d_model) + head_offset;

    float acc[1024]; // local accumulator (safe since head_dim <= 1024)
    for (int i = 0; i < head_dim; ++i) acc[i] = 0.0f;

    // --- first pass: max score ---
    float max_score = -INFINITY;
    for (int m = 0; m < N; ++m) {
        const float* Krow = K + ((long long)m * d_model) + head_offset;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += Qrow[d] * Krow[d];
        dot *= inv_sqrt_hd;
        if (dot > max_score) max_score = dot;
    }

    // --- second pass: exp + weighted sum ---
    float sum_exp = 0.0f;
    for (int m = 0; m < N; ++m) {
        const float* Krow = K + ((long long)m * d_model) + head_offset;
        const float* Vrow = V + ((long long)m * d_model) + head_offset;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += Qrow[d] * Krow[d];
        dot *= inv_sqrt_hd;

        float e = expf(dot - max_score);
        sum_exp += e;

        for (int d = 0; d < head_dim; ++d)
            acc[d] += e * Vrow[d];
    }

    // --- normalize and write output ---
    float inv_sum = (sum_exp == 0.0f) ? 0.0f : (1.0f / sum_exp);
    float* OutRow = output + ((long long)q_idx * d_model) + head_offset;
    for (int d = 0; d < head_dim; ++d)
        OutRow[d] = acc[d] * inv_sum;
}

// ======================================================================
// Host entry point (called by the runner)
// ======================================================================
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int N, int d_model, int h) {
    if (!Q || !K || !V || !output) return;
    if (N <= 0 || d_model <= 0 || h <= 0) return;
    if (d_model % h != 0) return;

    int head_dim = d_model / h;
    float inv_sqrt_hd = 1.0f / sqrtf((float)head_dim);

    long long total = (long long)N * (long long)h;
    int block = 256;
    int grid = (int)((total + block - 1) / block);

    mha_kernel<<<grid, block>>>(Q, K, V, output, N, d_model, h, head_dim, inv_sqrt_hd);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // optional sync (uncomment if required by the runner)
    // cudaDeviceSynchronize();
}
