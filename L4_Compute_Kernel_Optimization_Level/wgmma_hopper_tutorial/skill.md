---
skill_name: WGMMA Instruction Programming on Hopper via CUTLASS
description: Configuring and using warpgroup-level asynchronous MMA instructions (WGMMA) with descriptor-based shared memory operands on Hopper GPUs
level: L4 - Compute Kernel Optimization Level
target_hardware: NVIDIA Hopper H100/H200 (SM90)
relevance: When implementing FlashAttention microkernels on Hopper that need to directly configure WGMMA tile shapes, shared memory layouts, swizzle modes, and accumulator fragment handling
---

# WGMMA Instruction Programming on Hopper via CUTLASS

## What It Is
WGMMA (Warpgroup Matrix-Multiply-Accumulate) is Hopper's asynchronous tensor core instruction that operates at the warpgroup level (128 threads = 4 warps). Unlike prior MMA instructions, WGMMA reads operands directly from shared memory via 64-bit matrix descriptors rather than requiring register copies. This tutorial covers configuring WGMMA through CUTLASS's TiledMMA abstraction, including atom selection, shared memory layout requirements, swizzle modes, fragment shapes, and synchronization primitives.

## Key Concepts
- **Warpgroup = 128 threads (4 warps)**: All threads must participate collectively; warp rank must be a multiple of 4
- **Tile shape m64nNk16**: M is always 64, N ranges from 8 to 256 in multiples of 8, K is 16 for FP16 (32 bytes)
- **Operand B always in SMEM**: Operand A can be SMEM (SS variant) or registers (RS variant); accumulator C always in registers
- **Matrix descriptors**: 64-bit descriptors in registers point to SMEM data; no register copy needed for operands
- **Swizzle modes**: Layout_MN_SW128_Atom is most common for bank conflict mitigation; 4 MN-major and 4 K-major variants available
- **Asynchronous execution**: WGMMA issues to hardware queue; warpgroup_arrive/commit_batch/wait primitives manage dependencies
- **Accumulator Z-pattern**: 128 threads x 32 values each in a replicated Z-pattern across the output tile

## Tile Configuration / Code Pattern
```cpp
// 1. Select MMA atom (SS = both operands from shared memory)
using MmaAtom = SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>;

// 2. Build TiledMMA (optionally tile across multiple warpgroups)
TiledMMA tiled_mma = cute::make_tiled_mma(MmaAtom{});
// For 2 warpgroups (256 threads):
// TiledMMA tiled_mma = make_tiled_mma(MmaAtom{}, Layout<Shape<_2,_1,_1>>{});

// 3. Define shared memory layouts with swizzling
auto bM = Int<128>{};  // Must be multiple of 64
auto bN = Int<128>{};  // Must be multiple of 64
auto bK = Int<64>{};   // Must be multiple of 16
auto bP = Int<3>{};    // Pipeline stages

auto sA = cute::tile_to_shape(GMMA::Layout_MN_SW128_Atom<half_t>{},
                               make_shape(bM, bK, bP));
auto sB = cute::tile_to_shape(GMMA::Layout_MN_SW128_Atom<half_t>{},
                               make_shape(bN, bK, bP));

// 4. Partition across threads
ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K, PIPE)
Tensor tCsB = thr_mma.partition_B(sB);  // (MMA, MMA_N, MMA_K, PIPE)
Tensor tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(gC));

// 5. Execute with async synchronization
cute::warpgroup_arrive();
cute::gemm(tiled_mma, tCsA(_, _, _, pipe), tCsB(_, _, _, pipe), tCrC);
cute::warpgroup_commit_batch();
cute::warpgroup_wait<0>();  // Wait for all outstanding MMAs

// RS variant for GEMM-II (operand A from registers, e.g., softmax output P)
using MmaAtom_RS = SM90_64x64x16_F16F16F16_RS<GMMA::Major::K, GMMA::Major::MN>;
```

## Performance Impact
- WGMMA descriptor-based access eliminates register-to-register copy overhead for operands
- Asynchronous issue allows overlapping MMA with TMA loads and softmax computation
- 128-byte swizzle mode is critical for avoiding shared memory bank conflicts at full throughput
- H100 theoretical peak: 756 TFLOPS FP16 (1513 with sparsity); well-tuned WGMMA kernels reach 65-80% of peak
- The SS variant is preferred for GEMM-I (QK^T) where both operands come from shared memory
- The RS variant is needed for GEMM-II (PV) where the softmax output P comes from registers

## When to Use
- Implementing FlashAttention forward/backward passes on Hopper (H100/H200)
- Any fused kernel requiring tensor core MMA on Hopper that needs precise control over operand sources
- When shared memory layout and swizzle mode selection are critical for performance
- Building GEMM microkernels that will be composed with non-GEMM operations (softmax, masking)

## When NOT to Use
- On Blackwell GPUs where WGMMA is deprecated in favor of UMMA/tcgen05
- On pre-Hopper hardware (Ampere uses HMMA/WMMA instead)
- When using high-level APIs (torch.compile, Triton) that abstract away instruction selection
- For non-MMA operations (element-wise, reductions) that do not benefit from tensor cores

## Key Takeaways
- WGMMA's M=64 constraint means FlashAttention tile height is always a multiple of 64 (64, 128, 192, 256)
- The SS vs RS variant choice maps directly to FlashAttention's two GEMMs: SS for QK^T (both from SMEM), RS for PV (P from registers, V from SMEM)
- Matrix descriptors are a fundamental innovation over WMMA: they eliminate the register copy bottleneck for operands
- Swizzle mode selection (SW32, SW64, SW128) must match the tile dimensions and data type to avoid bank conflicts
- The accumulator's Z-pattern distribution affects how softmax row reductions are implemented (shuffle-based within quads)
- warpgroup_arrive/commit_batch/wait form the essential synchronization protocol; incorrect use causes hangs or data races

## References
- [CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on Hopper (Colfax Research)](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- [CUTLASS WGMMA SM90 Tutorial Source](https://github.com/NVIDIA/cutlass/tree/main/examples)
- [PTX ISA: wgmma.mma_async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions)
