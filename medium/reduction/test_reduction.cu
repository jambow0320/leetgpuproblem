#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

extern "C" void solve(const float*, float*, int);
extern "C" void solve_std(const float*, float*, int);

int main() {
    constexpr int N = 1 << 20; // 1M elements
    std::vector<float> h_input(N);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &v : h_input) v = dist(rng);

    float *d_input = nullptr, *d_out_a = nullptr, *d_out_b = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_out_a, sizeof(float));
    cudaMalloc(&d_out_b, sizeof(float));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out_a, 0, sizeof(float));
    cudaMemset(d_out_b, 0, sizeof(float));

    solve(d_input, d_out_a, N);
    solve_std(d_input, d_out_b, N);

    float h_a = 0, h_b = 0;
    cudaMemcpy(&h_a, d_out_a, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b, d_out_b, sizeof(float), cudaMemcpyDeviceToHost);

    printf("solve    : %f\nstd      : %f\ndiff     : %e\n", h_a, h_b, fabs(h_a - h_b));

    cudaFree(d_input);
    cudaFree(d_out_a);
    cudaFree(d_out_b);
    return 0;
}
