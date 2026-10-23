// tools/thor_tcgen05_probe.cpp - run strata::kernels::tcgen05's capability probe and grouped-GEMM entry
// against a built strata_kernels library, on Jetson Thor (sm_110).  Prints what the device executes:
// on the devkit this repo was measured with (driver 595.78, CUDA 13.2) the probe's forked child finds
// every tcgen05.mma refused (cta_group::1 raises an illegal instruction, cta_group::2 commits without
// writing TMEM), the engine's context is not touched, and gemm_f16_grouped refuses so callers keep
// MMQ/cuBLAS.  docs/THOR.md has the full findings.
//
// Built out-of-tree against a Maya build (see docs/THOR.md):
//   g++ -O2 -std=c++20 -I <maya>/include -I /usr/local/cuda/include thor_tcgen05_probe.cpp \
//       -o thor_tcgen05_probe -L <maya>/build-thor -lstrata_kernels -L /usr/local/cuda/lib64 -lcudart
#include "strata/kernels/tcgen05_moe.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>

int main() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { std::fprintf(stderr, "no CUDA device\n"); return 1; }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::printf("device: %s cc %d.%d\n", prop.name, prop.major, prop.minor);
    if (prop.major != 11) std::printf("(not sm_11x: the probe will report the device incapable)\n");

    const unsigned caps = strata::kernels::tcgen05::probe(dev);
    std::printf("probe(dev %d) = %u: probed=%d cta_group1_mma=%d cta_group2_mma=%d\n", dev, caps, caps & 1,
                (caps >> 1) & 1, (caps >> 2) & 1);

    // gemm_f16_grouped must either run and be exactly right, or refuse so the caller keeps its path.
    constexpr int E = 2, TOK = 40, NOUT = 128, K = 32;
    __half *X, *W;
    float *Y;
    int32_t* bounds;
    if (cudaMallocManaged(&X, TOK * K * 2) != cudaSuccess ||
        cudaMallocManaged(&W, (size_t) E * NOUT * K * 2) != cudaSuccess ||
        cudaMallocManaged(&Y, TOK * NOUT * 4) != cudaSuccess ||
        cudaMallocManaged(&bounds, (E + 1) * 4) != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocManaged failed\n");
        return 1;
    }
    for (int i = 0; i < TOK * K; i++) X[i] = __float2half((float) (i % 9) - 4.f);
    for (int i = 0; i < E * NOUT * K; i++) W[i] = __float2half((float) (i % 7) - 3.f);
    bounds[0] = 0;
    bounds[1] = 17;                      // expert 0: rows 0-16
    bounds[2] = TOK;                     // expert 1: rows 17-39

    const bool ran = strata::kernels::tcgen05::gemm_f16_grouped(X, K, W, (size_t) NOUT * K * 2, bounds, E, TOK,
                                                                NOUT, K, Y, NOUT, 0);
    std::printf("gemm_f16_grouped -> %s\n", ran ? "launched" : "refused (the caller keeps its fallback)");
    if (!ran) return 0;
    if (cudaDeviceSynchronize() != cudaSuccess) { std::fprintf(stderr, "the kernel faulted\n"); return 1; }
    int bad = 0;
    for (int t = 0; t < TOK && bad < 3; t++)
        for (int n = 0; n < NOUT && bad < 3; n++) {
            const int e = t >= bounds[1] ? 1 : 0;
            float r = 0;
            for (int k = 0; k < K; k++)
                r += __half2float(X[t * K + k]) * __half2float(W[((size_t) e * NOUT + n) * K + k]);
            if (Y[t * NOUT + n] != r) {
                std::printf("MISMATCH t=%d n=%d got %g want %g\n", t, n, Y[t * NOUT + n], r);
                bad++;
            }
        }
    std::printf(bad ? "numerics FAIL\n" : "numerics PASS: the UMMA path multiplies correctly\n");
    return bad != 0;
}
