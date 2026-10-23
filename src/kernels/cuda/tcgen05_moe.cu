// src/kernels/cuda/tcgen05_moe.cu - Jetson Thor (sm_110a) UMMA ("tcgen05") kernels; see tcgen05_moe.hpp.
//
// Everything device-side here is guarded by __CUDA_ARCH_FEAT_SM110_ALL, which the compiler only defines for
// the arch-specific compute_110a pass this translation unit gets through its own -gencode (CMakeLists.txt,
// STRATA_TCGEN05): in every other pass - plain sm_110 included - the file compiles to a host-side stub whose
// probe reports the device as incapable, which is exactly what the plain-sm_110 build of the same engine
// wants.
//
// Spelling notes for the CUDA 13.2 (PTX ISA 8.6) toolchain, all measured on Thor:
//  * tcgen05.mma's destination is a b32 TMEM address, and the predicate form of enable-input-d is
//    `[%d], a_desc, b_desc, idesc, p` (the {disable_output_lane} vector form also exists);
//  * there is no tcgen05.wait::mma - an MMA's completion is observed through tcgen05.commit, which arrives
//    on an mbarrier once every prior MMA of the same cta_group has written its TMEM result;
//  * ptxas refuses a function that mixes .cta_group::1 and .cta_group::2 tcgen05 instructions, so each
//    kernel body uses exactly one group;
//  * a K-major f16 operand tile lives in shared memory in the canonical interleaved layout: 8-row x 16-byte
//    core matrices, each a contiguous 128-byte block; the descriptor's leading byte offset steps core
//    matrices along K (128), the stride byte offset along M/N (core_columns * 128).  This layout round-trips
//    through tcgen05.ld/st on Thor (bit-exact, verified).
#include "strata/kernels/tcgen05_moe.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#if defined(STRATA_TCGEN05) && defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace strata::kernels::tcgen05 {
namespace {

// ---- descriptors and layouts (host + device) ----
// Field layouts from the CUTLASS that ships inside flashinfer 0.7.0 (mma_sm100_desc.hpp) - the same
// sources of the fused-MoE kernels validated on this Thor - and every value here verified on the device:

// The instruction descriptor for a dense f16 UMMA: D in f32 (c_format 1 at bit 4), both operands f16
// (a_format/b_format 0 at bits 7-12), K-major (a_major bit 15 / b_major bit 16 clear), N>>3 at bits
// 17-23, M>>2 at bits 24-29 (4 = M64, 8 = M128).  NOTE the two guards: N < 8 is invalid, and the m_dim
// field sits at bit 24 - an idesc built with the early sm_100a spellings decodes to N=0 and the device
// raises an illegal instruction.
__host__ __device__ inline uint32_t mma_idesc_f16(int m, int n) {
    return (1u << 4) | ((uint32_t) (n / 8) << 17) | ((uint32_t) (m / 64 * 4) << 24);
}

// The shared-memory matrix descriptor: start address, leading/stride byte offsets (all /4-encoded),
// no swizzle, no base offset, and version 1 - bit 46 is REQUIRED on Blackwell (zero means the pre-
// Blackwell encoding and the tensor pipe refuses the whole descriptor).
__host__ __device__ inline uint64_t smem_desc(uint32_t smem_addr, uint32_t lbo, uint32_t sbo) {
    return (uint64_t) ((smem_addr >> 4) & 0x3FFFu) | ((uint64_t) ((lbo >> 4) & 0x3FFFu) << 16) |
           ((uint64_t) ((sbo >> 4) & 0x3FFFu) << 32) | ((uint64_t) 1 << 46);
}

// Byte offset of the 16-byte chunk at (row, k16) of a BK/8-core-column tile in the canonical layout.
__host__ __device__ inline unsigned interleave_offset(int row, int k16, int core_cols) {
    return (unsigned) (((row / 8) * core_cols + k16) * 8 + (row % 8)) * 16;
}

// ---- device helpers (each kernel body uses exactly one cta_group) ----
// Compiled only in the arch-specific compute_110a pass: in every other pass of this translation unit
// (plain sm_110 included) these spellings do not assemble.
#if defined(__CUDA_ARCH_FEAT_SM110_ALL)

__device__ inline void tmem_alloc(uint32_t* dst_smem, int cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 ::"r"((uint32_t) __cvta_generic_to_shared(dst_smem)), "r"(cols));
}
__device__ inline void tmem_dealloc(uint32_t taddr, int cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(cols));
}
__device__ inline void mma_f16_cg1(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
                                   int accumulate) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        ::"r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate));
}
__device__ inline void mma_commit(uint32_t mbar_smem) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 ::"r"(mbar_smem));
}
__device__ inline void mbar_wait(void* mbar, uint32_t parity) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra WAIT;\n\t"
        "}\n"
        ::"r"((uint32_t) __cvta_generic_to_shared(mbar)), "r"(parity));
}
// One warp reads 32 TMEM lanes x count f32 columns into count registers (the warp's own sub-partition).
template <int COUNT>
__device__ inline void tmem_ld(void* dst, uint32_t taddr) {
    static_assert(COUNT == 8 || COUNT == 32, "the shapes this file uses");
    if constexpr (COUNT == 8) {
        uint32_t* r = (uint32_t*) dst;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                     : "r"(taddr));
    } else {
        uint32_t* r = (uint32_t*) dst;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,"
                     "%25,%26,%27,%28,%29,%30,%31}, [%32];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                       "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]),
                       "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]),
                       "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]),
                       "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
                     : "r"(taddr));
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}
#endif  // __CUDA_ARCH_FEAT_SM110_ALL

constexpr int BM = 64;     // token rows per tile (one TMEM sub-partition pair: lanes 0-63)
constexpr int BN = 128;    // weight rows per tile (TMEM f32 columns)
constexpr int BK = 16;     // K per UMMA (f16); the K loop steps one tile at a time
constexpr int CORE = BK / 8;
constexpr int TMEM_COLS = BN;   // power of two >= 32

#if defined(__CUDA_ARCH_FEAT_SM110_ALL)
// Stage one BK-deep tile of X rows [c0, c0+64) and W rows [n0, n0+128) into the interleaved layout.
__device__ inline void stage_tile(__half* sa, __half* sb, const __half* X, int ldx, const __half* W,
                                  size_t w_row_bytes, int c0, int crow, int n0, int K, int k0, int tid, int nthreads) {
    const int chunks_a = BM * CORE, chunks_b = BN * CORE;
    for (int i = tid; i < chunks_a + chunks_b; i += nthreads) {
        if (i < chunks_a) {
            const int r = i / CORE, kc = i % CORE, k = k0 + kc * 8;
            uint4 zero = make_uint4(0, 0, 0, 0);
            uint4 v = r < crow ? *reinterpret_cast<const uint4*>(X + (size_t) (c0 + r) * ldx + k) : zero;
            *reinterpret_cast<uint4*>(sa + interleave_offset(r, kc, CORE) / 2) = v;
        } else {
            const int j = i - chunks_a, r = j / CORE, kc = j % CORE, k = k0 + kc * 8;
            *reinterpret_cast<uint4*>(sb + interleave_offset(r, kc, CORE) / 2) =
                *reinterpret_cast<const uint4*>(W + (size_t) r * w_row_bytes / 2 + k);
        }
    }
}
#endif  // __CUDA_ARCH_FEAT_SM110_ALL

template <int IDN>   // instruction-descriptor N (the tile's BN, or 8 for the self-test)
__global__ void __launch_bounds__(256) gemm_f16_grouped_kernel(
        const __half* __restrict__ X, int ldx, const __half* __restrict__ W, size_t expert_bytes,
        const int32_t* __restrict__ bounds, int chunks_per_e, int K, float* __restrict__ Y, int ldy) {
#if defined(__CUDA_ARCH_FEAT_SM110_ALL)
    __shared__ __align__(1024) __half sa[2][BM * BK];
    __shared__ __align__(1024) __half sb[2][IDN * BK];
    __shared__ __align__(8) uint64_t mbar;
    __shared__ uint32_t tmem_base;
    const int tid = threadIdx.x;
    const int e = blockIdx.y / chunks_per_e, ch = blockIdx.y % chunks_per_e;
    const int c0 = bounds[e] + ch * BM, r1 = bounds[e + 1];
    if (c0 >= r1) return;
    const int crow = min(BM, r1 - c0);
    const int n0 = blockIdx.x * IDN;
    const auto* we = W + (size_t) e * (expert_bytes / 2);
    const size_t w_row_bytes = expert_bytes / (size_t) IDN;

    if (tid == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t) __cvta_generic_to_shared(&mbar)));
    __syncthreads();
    if (tid < 32) tmem_alloc(&tmem_base, IDN == 8 ? 32 : TMEM_COLS);
    __syncthreads();
    const uint32_t d_tmem = tmem_base;

    const int nk = K / BK;
    stage_tile(sa[0], sb[0], X, ldx, we, w_row_bytes, c0, crow, n0, K, 0, tid, blockDim.x);
    __syncthreads();
    for (int k = 0; k < nk; ++k) {
        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            mma_f16_cg1(d_tmem,
                        smem_desc((uint32_t) __cvta_generic_to_shared(sa[k & 1]), 128, CORE * 128),
                        smem_desc((uint32_t) __cvta_generic_to_shared(sb[k & 1]), 128, CORE * 128),
                        mma_idesc_f16(BM, IDN), k != 0);
            mma_commit((uint32_t) __cvta_generic_to_shared(&mbar));
        }
        mbar_wait(&mbar, (uint32_t) (k & 1));    // the MMA read both tiles and wrote TMEM
        __syncthreads();
        if (k + 1 < nk) {
            stage_tile(sa[(k + 1) & 1], sb[(k + 1) & 1], X, ldx, we, w_row_bytes, c0, crow, n0, K, (k + 1) * BK,
                       tid, blockDim.x);
        }
        __syncthreads();
    }

    // epilogue: the M=64 D layout (measured on Thor) puts output row r at TMEM lane (r/16)*32 + r%16 -
    // warp w's sub-partition holds rows w*16..w*16+15 in its lanes 0-15, lanes 16-31 read as zero.
    // tcgen05.ld is warp-collective: all 32 lanes issue it, the lane<16 half stores.  v[] lives INSIDE the
    // chunk loop: declared outside, -O3 sank the float4 stores across the async ld iterations and columns
    // landed on other columns' values (found with a one-hot B; the mma itself was never wrong).
    if (tid < 128) {
        const int warp = tid / 32, lane = tid % 32;
        for (int c = 0; c < IDN; c += 32) {
            float v[32];
            tmem_ld<32>(v, d_tmem | ((uint32_t) warp * 32u << 16) | (uint32_t) c);
            const int r = warp * 16 + lane;
            if (lane < 16 && r < crow) {   // rows past the expert's last token row stay unwritten
                float* y = Y + (size_t) (c0 + r) * ldy + n0 + c;
                const int cols = min(32, IDN - c);
                for (int j = 0; j < cols; j += 4)
                    *reinterpret_cast<float4*>(y + j) = *reinterpret_cast<const float4*>(v + j);
            }
        }
    }
    __syncthreads();
    if (tid < 32) tmem_dealloc(d_tmem, IDN == 8 ? 32 : TMEM_COLS);
#endif
}

// The probe runs in a SEPARATE PROCESS: tools/thor_tcgen05_probe.cpp --caps, which drives the real
// production kernel (a known product checked against the CPU) and exits with the capability bits.  A
// fork alone is not enough - the fork-inherited CUDA runtime (its worker threads do not survive the
// fork) dies at the first TMEM operation on this driver - so the child EXECs the helper: a fresh
// process, a fresh CUDA state, and an illegal instruction or a fault can only ever kill the helper,
// never the engine's context.  Found next to the engine (under its /proc/self/exe directory) or via
// STRATA_TCGEN05_PROBE; without it the device reports incapable and every caller keeps its fallback.
#if defined(STRATA_TCGEN05) && defined(__linux__)

unsigned probe_by_helper() {
    char exe[4096];
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) return 1;
    exe[n] = 0;
    char path[4200];
    std::snprintf(path, sizeof(path), "%s", exe);
    char* slash = strrchr(path, '/');
    const char* env = getenv("STRATA_TCGEN05_PROBE");
    if (env && *env) {
        std::snprintf(path, sizeof(path), "%s", env);
    } else {
        char dir[4200];
        if (!slash) std::snprintf(path, sizeof(path), "strata-tcgen05-probe");
        else {
            std::snprintf(dir, sizeof(dir), "%.*s", (int) (slash - path), path);
            std::snprintf(path, sizeof(path), "%s/strata-tcgen05-probe", dir);
        }
    }
    const pid_t p = fork();
    if (p == 0) {
        execl(path, path, "--caps", (char*) nullptr);
        if (getenv("STRATA_TCGEN05_DEBUG"))
            std::fprintf(stderr, "probe: exec %s failed: %s\n", path, strerror(errno));
        _exit(127);   // exec failed: the helper is not there, the device stays "incapable"
    }
    int st = 0;
    waitpid(p, &st, 0);
    if (!WIFEXITED(st)) return 1;            // the helper crashed: probed, incapable
    const unsigned code = (unsigned) WEXITSTATUS(st);
    if (code == 127) return 1;               // the exec itself failed (helper not found)
    return 1u | (code & 6u);                 // bits 2 = cg1, 4 = cg2 (probed bit added here)
}
#else

unsigned probe_by_helper() { return 1; }

#endif

constexpr int ST_ROWS = 64, ST_N = 128, ST_K = 16;

bool selftest_run() {
    __half *A, *B;
    float* D;
    int32_t* bounds;
    if (cudaMallocManaged(&A, (size_t) ST_ROWS * ST_K * 2) != cudaSuccess) return false;
    if (cudaMallocManaged(&B, (size_t) ST_N * ST_K * 2) != cudaSuccess) return false;
    if (cudaMallocManaged(&D, (size_t) ST_ROWS * ST_N * 4) != cudaSuccess) return false;
    if (cudaMallocManaged(&bounds, 2 * 4) != cudaSuccess) return false;
    for (int i = 0; i < ST_ROWS * ST_K; i++) A[i] = __float2half((float) ((i * 7) % 13) - 6.f);
    for (int i = 0; i < ST_N * ST_K; i++) B[i] = __float2half((float) ((i * 5) % 7) - 3.f);
    bounds[0] = 0;
    bounds[1] = ST_ROWS;
    gemm_f16_grouped_kernel<BN><<<dim3(1, 1), 256>>>(A, ST_K, B, (size_t) ST_N * ST_K * 2, bounds, 1, ST_K, D,
                                                     ST_N);
    bool ok = cudaDeviceSynchronize() == cudaSuccess;
    if (ok)
        for (int m = 0; m < ST_ROWS && ok; m++)
            for (int n = 0; n < ST_N && ok; n++) {
                float ref = 0;
                for (int k = 0; k < ST_K; k++)
                    ref += __half2float(A[m * ST_K + k]) * __half2float(B[n * ST_K + k]);
                ok = D[m * ST_N + n] == ref;
                if (!ok && getenv("STRATA_TCGEN05_DEBUG"))
                    std::fprintf(stderr, "selftest: m=%d n=%d got %g want %g\n", m, n, D[m * ST_N + n], ref);
            }
    cudaFree(A);
    cudaFree(B);
    cudaFree(D);
    cudaFree(bounds);
    return ok;
}

}  // namespace

bool selftest(int) { return selftest_run(); }

unsigned probe(int dev) {
#if defined(STRATA_TCGEN05) && defined(__linux__)
    static thread_local unsigned per_dev[64];
    if (dev < 0 || dev >= 64) return 0;
    if (!(per_dev[dev] & 1u)) {
        int major = 0;
        unsigned caps = 1u;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major == 11)
            caps = probe_by_helper();
        per_dev[dev] = caps;
    }
    return per_dev[dev];
#else
    (void) dev;
    return 0;
#endif
}

bool gemm_f16_grouped(const void* X, int ldx, const void* W, size_t expert_bytes, const int32_t* bounds,
                      int n_experts, int max_rows, int n_out, int K, float* Y, int ldy, void* stream) {
#if defined(STRATA_TCGEN05)
    if (n_out <= 0 || n_out % BN != 0 || K <= 0 || K % BK != 0 || n_experts <= 0 || max_rows <= 0) return false;
    if (ldx < K || ldx % 8 != 0 || ldy < n_out || ldy % 4 != 0) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (!(probe(dev) & 2u)) return false;   // only the cta_group::1 form is implemented here
    const int chunks = (max_rows + BM - 1) / BM;
    const dim3 grid((unsigned) (n_out / BN), (unsigned) (n_experts * chunks));
    gemm_f16_grouped_kernel<BN><<<grid, 256, 0, (cudaStream_t) stream>>>(
        (const __half*) X, ldx, (const __half*) W, expert_bytes, bounds, chunks, K, Y, ldy);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) X; (void) ldx; (void) W; (void) expert_bytes; (void) bounds; (void) n_experts; (void) max_rows;
    (void) n_out; (void) K; (void) Y; (void) ldy; (void) stream;
    return false;
#endif
}

}  // namespace strata::kernels::tcgen05
