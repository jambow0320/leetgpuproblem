#include <vector>
#include <cuda_runtime.h>

using std::vector;

__global__ void RoPE_kernel(float *K, float *V, int B, int H, int S, int D, const float *cos, const float *sin)
{
    int y = threadIdx.y, x = threadIdx.x;
    int bias = blockIdx.y * S * D + (blockIdx.x * blockDim.y + y) * D;
    float *K0 = K + bias;
    float *V0 = V + bias;
    int pos = blockIdx.x * blockDim.y + y;

    if (pos < S)
    {
        for (int i = x; i < D / 2; i += blockDim.x)
        {
            float k2i = K0[2 * i], k2i1 = K0[2 * i + 1];
            K0[2 * i] = k2i * cos[pos * (D / 2) + i] - k2i1 * sin[pos * (D / 2) + i];
            K0[2 * i + 1] = k2i * sin[pos * (D / 2) + i] + k2i1 * cos[pos * (D / 2) + i];

            float v2i = V0[2 * i], v2i1 = V0[2 * i + 1];
            V0[2 * i] = v2i * cos[pos * (D / 2) + i] - v2i1 * sin[pos * (D / 2) + i];
            V0[2 * i + 1] = v2i * sin[pos * (D / 2) + i] + v2i1 * cos[pos * (D / 2) + i];
        }
    }
}

float *cos, *sin;
void sin_cos_table(int S, int D)
{
    int half_d = D / 2;

    vector<float> cos_table(S * half_d);
    vector<float> sin_table(S * half_d);
    vector<float> inv_pow(half_d);

    for (int i = 0; i < half_d; ++i)
        inv_pow[i] = 1 / pow(10000, (float)2 * i / D);

    for (int s = 0; s < S; ++s)
        for (int i = 0; i < half_d; ++i)
        {
            cos_table[s * half_d + i] = cosf(s * inv_pow[i]);
            sin_table[s * half_d + i] = sinf(s * inv_pow[i]);
        }

    cudaMalloc(&cos, S * half_d * sizeof(float));
    cudaMalloc(&sin, S * half_d * sizeof(float));

    cudaMemcpy(cos, cos_table.data(), S * half_d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sin, sin_table.data(), S * half_d * sizeof(float), cudaMemcpyHostToDevice);
}

extern "C" void solve(float *K, float *V, int B, int H, int S, int D, int pos)
{
    dim3 block(32, 8);
    dim3 grid((S + 8 - 1) / 8, B * H);

    RoPE_kernel<<<grid, block>>>(K, V, B, H, S, D, cos, sin);
}