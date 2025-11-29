#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d);
extern "C" void solve_std(const float* Q, const float* K, const float* V, float* output, int M, int d);

static void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

struct DiffStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int   idx = -1;
};

static DiffStats compare(const std::vector<float>& a, const std::vector<float>& b) {
    DiffStats stats;
    for (size_t i = 0; i < a.size(); ++i) {
        float ref = b[i];
        float diff = std::fabs(a[i] - ref);
        float rel = diff / (std::fabs(ref) + 1e-6f);
        if (diff > stats.max_abs) {
            stats.max_abs = diff;
            stats.idx = static_cast<int>(i);
        }
        if (rel > stats.max_rel) {
            stats.max_rel = rel;
        }
    }
    return stats;
}

static void run_case(int M, int d, int seed = 42, float tol_abs = 1e-4f, float tol_rel = 1e-4f) {
    const size_t size = static_cast<size_t>(M) * d;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> hQ(size), hK(size), hV(size);
    for (float& v : hQ) v = dist(rng);
    for (float& v : hK) v = dist(rng);
    for (float& v : hV) v = dist(rng);

    float *dQ = nullptr, *dK = nullptr, *dV = nullptr;
    float *dOut = nullptr, *dRef = nullptr;
    check(cudaMalloc(&dQ, size * sizeof(float)), "cudaMalloc dQ");
    check(cudaMalloc(&dK, size * sizeof(float)), "cudaMalloc dK");
    check(cudaMalloc(&dV, size * sizeof(float)), "cudaMalloc dV");
    check(cudaMalloc(&dOut, size * sizeof(float)), "cudaMalloc dOut");
    check(cudaMalloc(&dRef, size * sizeof(float)), "cudaMalloc dRef");

    check(cudaMemcpy(dQ, hQ.data(), size * sizeof(float), cudaMemcpyHostToDevice), "Memcpy Q");
    check(cudaMemcpy(dK, hK.data(), size * sizeof(float), cudaMemcpyHostToDevice), "Memcpy K");
    check(cudaMemcpy(dV, hV.data(), size * sizeof(float), cudaMemcpyHostToDevice), "Memcpy V");

    solve(dQ, dK, dV, dOut, M, d);
    solve_std(dQ, dK, dV, dRef, M, d);
    check(cudaDeviceSynchronize(), "Kernel sync");
    check(cudaGetLastError(), "Kernel launch");

    std::vector<float> hOut(size), hRef(size);
    check(cudaMemcpy(hOut.data(), dOut, size * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy out");
    check(cudaMemcpy(hRef.data(), dRef, size * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy ref");

    DiffStats stats = compare(hOut, hRef);
    bool ok = (stats.max_abs <= tol_abs) || (stats.max_rel <= tol_rel);

    std::printf("M=%d d=%d seed=%d : max_abs=%g max_rel=%g %s", M, d, seed, stats.max_abs, stats.max_rel,
                ok ? "OK" : "MISMATCH");
    if (!ok) std::printf(" (first max idx=%d out=%g ref=%g)", stats.idx, hOut[stats.idx], hRef[stats.idx]);
    std::printf("\n");

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dOut);
    cudaFree(dRef);
}

int main() {
    // 选择几个不同的 M/d 组合进行对拍，M 决定共享内存占用，注意不要超过硬件上限。
    const int seeds[] = {1, 2, 3};
    struct Shape { int M; int d; };
    const Shape shapes[] = {
        {64, 64},
        {128, 96},
        {256, 128},
    };

    for (const auto& s : shapes) {
        for (int seed : seeds) {
            run_case(s.M, s.d, seed);
        }
    }
    return 0;
}
