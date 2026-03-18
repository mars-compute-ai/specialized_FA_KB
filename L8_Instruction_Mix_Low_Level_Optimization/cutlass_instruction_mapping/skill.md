---
skill_name: CUTLASS & CuTe Instruction Mapping
description: Use CuTe's layout algebra and MMA atoms to precisely control the mapping from tile/warp shapes to hardware MMA and load/store instruction sequences.
level: L8 - Instruction-Mix/Low-Level Optimisation
target_hardware: NVIDIA Volta SM70 through Blackwell SM100 (architecture-specific atoms)
relevance: When an AI agent needs to select instruction shapes, tiling patterns, and unrolling factors for custom GEMM or attention kernels, or needs to understand how high-level tile parameters translate to specific instruction sequences.
---

# CUTLASS & CuTe Instruction Mapping

## What It Is
CuTe (CUDA Templates) within CUTLASS provides a hierarchical abstraction that maps high-level tensor tile operations to specific hardware instructions through composable Layout, Atom, and Tiled operation primitives. Developers specify tile shapes, warp shapes, instruction shapes (e.g., 16x8x16), and unrolling factors as template parameters. CuTe's layout algebra then deterministically translates these parameters into sequences of MMA instructions, load/store operations, and register assignments, giving precise control over the instruction mix generated for a kernel.

## Key Concepts
- **Layout<Shape, Stride>**: Unified abstraction mapping logical coordinates to physical memory indices; composable via `composition()` and partitionable across thread groups
- **MMA Atoms**: Smallest instruction units representing a single hardware MMA operation (e.g., `SM80_16x8x16_F16F16F32`), encoding thread-to-value mappings
- **MMA_Traits**: Metadata capturing instruction shape (MxNxK), data types, and thread/value layouts for each architecture-specific MMA instruction
- **Tiled MMA**: Composition of multiple atoms via tiling patterns (e.g., 2x2 tiling of 8x8 atoms = 16x16 operation), determining how many MMA instructions execute per warp
- **Tiled Copy**: Analogous composition for load/store operations, mapping thread layouts to memory access patterns
- **Instruction shape selection**: Trade-off between instruction throughput (larger shapes = more compute per instruction) and register pressure (larger shapes = more registers per thread)
- **Unrolling factor**: Controls the number of K-tiles processed per loop iteration, affecting instruction-level parallelism vs. register pressure

## Instruction Patterns / Code
```cpp
// === Selecting an MMA Atom for your target architecture ===

// Ampere (SM80): 16x8x16 MMA instruction
using MMA_Atom_SM80 = MMA_Atom<SM80_16x8x16_F16F16F32_TN>;

// Hopper (SM90): Warpgroup MMA, both operands from shared memory
using MMA_Atom_SM90 = MMA_Atom<SM90_64x128x16_F16F16F32_SS>;

// === Composing atoms into tiled operations ===

// 2x2 tiling of SM80 atom: produces 32x16x16 per warp
auto tiled_mma = make_tiled_mma(
    SM80_16x8x16_F16F16F32_TN{},
    Layout<Shape<_2, _2>>{},     // 2x2 M,N tiling
    Layout<Shape<_1, _1>>{}      // No permutation
);

// === Partitioning tensors for MMA ===

ThrMMA thr_mma = tiled_mma.get_slice(thread_idx);
Tensor tCsA = thr_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K)
Tensor tCsB = thr_mma.partition_B(sB);  // (MMA, MMA_N, MMA_K)
Tensor tCrC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

// === Main compute loop with explicit instruction mapping ===

// Each cute::gemm call generates:
//   num_mma = (TILE_M/INST_M) * (TILE_N/INST_N) MMA instructions per K-step
for (int k = 0; k < num_k_tiles; ++k) {
    cute::copy(tCsA(_, _, k), tCrA);   // Load from smem to registers
    cute::copy(tCsB(_, _, k), tCrB);   // Load from smem to registers
    cute::gemm(tiled_mma, tCrA, tCrB, tCrC);  // -> N hardware MMA instructions
}

// === Instruction count calculation ===
// Tile: 128x128, Instruction: 16x8x16, K-tile: 32
// MMA per K-step: (128/16) * (128/8) = 8 * 16 = 128 instructions
// K-steps per tile: 32/16 = 2
// Total MMA per tile: 128 * 2 = 256 MMA instructions
```

```cpp
// === Tiled Copy for load/store instruction mapping ===

auto tiled_copy = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_16, _8>>{},   // 16x8 thread layout
    Layout<Shape<_1, _8>>{}     // Each thread copies 8 values
);
// Maps to: cp.async.cg.shared.global [smem], [gmem], 16

// === Hopper TMA copy ===
auto tma_load = make_tma_copy(
    SM90_TMA_LOAD{},
    sA,                         // Shared memory tensor
    SmemLayout{}                // Swizzled layout
);
// Maps to: cp.async.bulk.tensor.2d instruction (single-thread issue)
```

## Performance Impact
- Instruction shape selection directly determines MMA instruction count: 16x8x16 generates 4x fewer instructions than 8x8x4 for the same tile
- Proper tiling reduces loop control overhead and enables better instruction-level parallelism
- Unrolling K-loop by 2x halves loop overhead but doubles register usage
- WGMMA (SM90) reads operands from shared memory, eliminating register load instructions for one operand
- Architecture progression: SM70 (8x8x4) -> SM80 (16x8x16) -> SM90 (64x128x16) represents 32x+ instruction-level efficiency improvement

## When to Use
- When writing custom GEMM or attention kernels with CUTLASS 3.x and need to control instruction-level behavior
- When profiling shows suboptimal instruction mix (too many loads vs. MMA, or vice versa)
- When tuning tile shapes and need to understand how they map to instruction counts
- When porting kernels across GPU architectures and need to select appropriate instruction atoms
- When register pressure is the limiting factor and instruction shape choice determines occupancy

## When NOT to Use
- When using high-level libraries (PyTorch, cuDNN) that handle instruction selection automatically
- For simple kernels where default CUTLASS configurations are sufficient
- When the kernel is memory-bandwidth-bound and instruction mix is not the bottleneck
- For non-GEMM workloads where MMA instructions are not applicable

## Key Takeaways
- CuTe's Layout algebra provides a mathematically rigorous way to map tile parameters to instruction sequences: `num_mma = (TILE_M/INST_M) * (TILE_N/INST_N) * (TILE_K/INST_K)`
- MMA Atoms are the bridge between abstract tiling and hardware instructions; choosing the right atom determines both performance and register pressure
- Tiled MMA composition allows building larger operations from smaller atoms while maintaining precise control over the instruction sequence
- The hierarchy (Layout -> Atom -> Tiled -> Kernel) gives developers knobs at every level of abstraction for tuning the instruction mix
- On Hopper, WGMMA atoms fundamentally change the instruction mix by eliminating register-staging loads for one MMA operand

## References
- [CUTLASS Principled Abstractions Blog (NVIDIA)](https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/)
- [CUTLASS 3.x GitHub Repository](https://github.com/NVIDIA/cutlass)
- [CuTe Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- NVIDIA PTX ISA: MMA instruction specifications per architecture
