#pragma once

// strata::kernels::tcgen05 - Jetson Thor (sm_110a) UMMA ("tcgen05") kernels.
//
// Thor's 5th-generation tensor cores take the Blackwell tcgen05 instructions, which the toolchain only
// assembles for the arch-specific compute_110a target (this source is given its own -gencode in
// CMakeLists.txt, guarded by MAYA_TCGEN05 / STRATA_TCGEN05).  Whether a given device + driver actually
// EXECUTES them is a separate question - on the Jetson Thor devkit this repo was measured on
// (driver 595.78, CUDA 13.2) every cta_group::1 MMA raises an illegal instruction and the cta_group::2
// form commits without writing TMEM - so everything here is gated at run time:
//
//   probe(dev) forks a child that multiplies a known 64x8x16 product on the tensor cores and checks it
//   against the CPU before the engine's context ever issues one (an illegal instruction would poison the
//   whole CUDA context).  The caps it returns decide whether gemm_f16_grouped runs or the caller keeps its
//   existing path.
//
// The intended production integration point is the GLM prompt experts (src/core/glm_prefill.cu's run_set,
// beside the Volta grouped-WMMA branch): dequantize each expert set to f16 (strata::kernels::dequant_f16,
// the same scratch Gemm::native uses) and run gemm_f16_grouped over the set's bounds, instead of the
// per-expert cuBLAS calls or MMQ.
#include <cstddef>
#include <cstdint>

namespace strata::kernels::tcgen05 {

// Capability bits: 1 = probed, 2 = a cta_group::1 MMA multiplies correctly, 4 = a cta_group::2 MMA does.
// Cached per device; the probe runs once per process (in the forked child).
unsigned probe(int dev);
inline bool available(int dev) { return (probe(dev) & 2u) != 0; }

// Grouped GEMM over f16 activations and f16 weights in fp32 accumulation: for expert e, the token rows
// [bounds[e], bounds[e+1]) of X [tokens, K] (row-major) multiply W_e [n_out, K] (row-major, expert e's
// matrix at W + e*expert_bytes) into Y [tokens, n_out] (row-major, fp32).  Requires n_out % 128 == 0,
// K % 16 == 0, ldx % 8 == 0 and ldy % 4 == 0 (vector loads/stores); max_rows bounds the largest expert's
// row count (as in gemm_iq_f16_grouped).  Returns false when the shapes do not fit or the device's probe
// did not pass - the caller keeps its fallback path.
bool gemm_f16_grouped(const void* X, int ldx, const void* W, size_t expert_bytes, const int32_t* bounds,
                      int n_experts, int max_rows, int n_out, int K, float* Y, int ldy, void* stream);

} // namespace strata::kernels::tcgen05
