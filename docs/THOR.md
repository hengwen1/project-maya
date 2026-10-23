# Experimental Maya on Jetson Thor (sm_110)

This branch carries the Jetson Thor port: the engine's device code builds for Thor's Blackwell GPU
(compute capability 11.0, `sm_110`), and the pieces specific to Thor - its UMMA ("tcgen05") tensor-core
instructions, the arch-specific `sm_110a` toolchain target they need, and the aarch64 host fixes - are
wired in behind run-time probes. Everything here was measured on the Jetson Thor developer kit
(driver 595.78, CUDA 13.2, 20 SMs, 228 KB shared memory per block).

## What works today

- The build: `./setup.sh` on a Thor picks `CMAKE_CUDA_ARCHITECTURES=110` (its cc is 110, above the 75
  floor), and CMakeLists.txt refuses a toolkit older than CUDA 13.0 with a clear message - no 12.x
  nvcc can build for sm_110. `maya.py`'s toolkit picker (`nvcc_range`) knows the same floor, and its
  build-failure hints name Thor.
- `strata_kernels` - every CUDA kernel of the GLM engine, compiled for sm_110 - builds and links on
  Thor, and its capability probes pick the sm_80+ paths at run time (MMQ's int8 tensor-core products,
  the f16 grouped WMMA's sm_70-only gate stays off, as on Turing and newer).
- aarch64 fixes: `src/core/graph.cpp` spun on `_mm_pause()` with no ARM path (the same file that breaks
  every non-x86 host); it now uses ARM's `yield` there. This is what a DGX Spark (GB10) build hits too.

## The tcgen05 work

Thor's 5th-generation tensor cores take the Blackwell `tcgen05` instructions, which the toolchain only
accepts for the arch-specific `compute_110a` target. Three findings cost the most time, all measured on
the JetPack 7 CUDA 13.2 toolchain:

1. **`-arch=sm_110a` is silently downgraded to plain sm_110.** The nvcc frontend does not know the
   suffix on this toolchain and produces compute_110 PTX, which ptxas then refuses for every tcgen05
   instruction. The only spelling that works is an explicit
   `-gencode arch=compute_110a,code=sm_110a` - which is what CMakeLists.txt puts on
   `src/kernels/cuda/tcgen05_moe.cu` (option `MAYA_TCGEN05`, on by default when a Thor is a build
   target; CMake 4.4 and CUDA 13.2 take the extra gencode beside the plain sm_110 one in the same
   translation unit, and the device code is guarded by `__CUDA_ARCH_FEAT_SM110_ALL` so every other pass
   compiles to an empty stub).
2. **The CUDA 13.2 (PTX ISA 8.6) grammar is stricter than the early sm_100a spellings:** the MMA's
   destination is a b32 TMEM address, the accumulator-carry is a plain predicate operand
   (`[%d], a_desc, b_desc, idesc, p`), and there is no `tcgen05.wait::mma` - completion is observed
   through `tcgen05.commit`, which arrives on an mbarrier once the MMA's TMEM result is written.
   ptxas also refuses a function that mixes `.cta_group::1` and `.cta_group::2` instructions.
3. **Canonical shared-memory layouts:** a K-major f16 operand tile lives as 8-row x 16-byte core
   matrices, each a contiguous 128-byte block; the descriptor's leading byte offset steps core matrices
   along K (128), the stride byte offset along M/N (core columns x 128). This layout round-trips
   bit-exactly through `tcgen05.ld`/`tcgen05.st` on Thor.

### Whether Thor executes tcgen05

Not on this devkit + driver. `src/kernels/cuda/tcgen05_moe.cu` probes it at run time - in a forked
child, because an illegal instruction poisons the whole CUDA context and must never be issued
speculatively by the engine:

| instruction | result on Thor (driver 595.78) |
| --- | --- |
| `tcgen05.alloc` / `dealloc` / `ld` / `st` / `commit` / `fence` (cta_group::1) | work; the TMEM round-trip is bit-exact |
| `tcgen05.mma.cta_group::1` (kind::f16, every idesc tried; also the `.ws` weight-stationary form) | **illegal instruction** |
| `tcgen05.mma.cta_group::2` | accepted, commits immediately, writes no TMEM any `tcgen05.ld` can see; the cg2-allocated region is inert to `ld`/`st` as well |

So the probe (`strata::kernels::tcgen05::probe`) reports the device incapable today, and
`gemm_f16_grouped` refuses - the engine keeps MMQ/cuBLAS with zero risk. On hardware or a driver where
the UMMA pipe works, the same probe lights the path up without a rebuild: the child multiplies a known
64x8x16 product on the tensor cores and checks it against the CPU before the capability is believed.

The intended production integration point is the GLM prompt experts (`glm_prefill.cu`'s run_set, beside
the Volta grouped-WMMA branch): dequantize each expert set to f16
(`strata::kernels::dequant_f16`, the same scratch `Gemm::native` uses) and run
`strata::kernels::tcgen05::gemm_f16_grouped` over the set's expert bounds - one grouped tensor-core
GEMM instead of per-expert cuBLAS calls. The kernel is implemented (64 x 128 tiles, K stepped in
UMMA-sized 16-value chunks, double-buffered shared-memory tiles in the canonical layout, TMEM
accumulators read back by the epilogue's warps); it is exercised by the probe's self-test and by

```
g++ -std=c++20 -Iinclude -I/usr/local/cuda/include a.cpp \
    -Lbuild-thor -lstrata_kernels -L/usr/local/cuda/lib64 -lcudart
```

against `strata::kernels::tcgen05::gemm_f16_grouped` (refuses today, multiplies when the hardware does).

## Not done yet

- A full `maya` engine build on aarch64 stops in `strata_kernels_cpu`: its CPU expert kernels
  (`src/kernels/cpu/expert.cpp`, `iq_avx512.cpp`, `iq_avx2.cpp`, `kq_avx512.cpp`, `iq_avx512_one.cpp`,
  `q2_avx2.cpp`) are x86-only - unconditional `immintrin.h` intrinsics plus x86-only `-m` flags from
  the CMake per-file logic. ARM (NEON or SVE) equivalents, or an x86-only exclusion of those files from
  the CPU probe, are the remaining port work; the README's DGX Spark entry was measured on a tree that
  worked around this locally.
- The tcgen05 MoE path is not wired into `glm_prefill.cu`'s dispatch yet: with no device able to execute
  it, the wiring would be dead code. It goes in behind `probe()` the first time a Thor executes the
  self-test.
