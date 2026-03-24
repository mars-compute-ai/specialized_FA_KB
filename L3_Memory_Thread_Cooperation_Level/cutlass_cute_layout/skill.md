---
skill_name: CUTLASS 3.x & CuTe Layout Abstractions
description: Unified tensor layout algebra for composable thread-data mapping, supporting WGMMA, TMA, and multi-architecture GPU kernels
level: L3 - Memory & Thread Cooperation Level
target_hardware: NVIDIA Volta (SM70), Ampere A100 (SM80), Hopper H100 (SM90), Blackwell B200 (SM100+)
relevance: When an AI agent needs to design custom GPU kernels with complex memory tiling, understand CUTLASS-based attention or GEMM implementations, or reason about thread-to-data mapping across memory hierarchy levels
---

# CUTLASS 3.x & CuTe Layout Abstractions

## What It Is
CuTe (CUDA Tensors and Spatial Microkernels) is the foundational library in CUTLASS 3.x that represents both threads and data using a unified `Layout<Shape,Stride>` abstraction. By composing layouts functionally, developers describe how thousands of threads map to multi-dimensional data without hand-coding index arithmetic. CuTe integrates with hardware-accelerated operations (WGMMA on Hopper, UMMA on Blackwell, TMA) and provides statically-checkable correctness — if it compiles, the indexing is correct.

## Key Concepts
- **Layout<Shape,Stride>**: Maps logical coordinates to memory indices; supports hierarchical shapes for tiled/swizzled patterns
- **Tensor<Engine,Layout>**: Combines a memory pointer (global, shared, or register) with a Layout into a single composable type
- **Layout algebra / functional composition**: Compose data layouts with thread layouts to automatically generate correct access patterns
- **TV (Thread-Value) layouts**: Record how threads (T) map to values (V) within an atom, enabling partitioning of arbitrary data
- **Atoms**: Smallest hardware MMA unit (e.g., 8x8x4 on Volta, WGMMA on Hopper); described as TV layouts + PTX instructions
- **Tiled MMA/Copy**: Scale atoms across threads and data dimensions by tiling with optional permutations
- **Spatial vs temporal separation**: CuTe handles spatial (thread-data mapping); CUTLASS handles temporal (pipelining, async staging)
- **Static checking**: Layout composition errors are caught at compile time, not runtime

## Memory Layout / Data Flow
```
CuTe abstractions map across the full memory hierarchy:

Global Memory (HBM)           Shared Memory (SMEM)         Registers
┌──────────────┐              ┌──────────────┐             ┌──────────────┐
│ Tensor<GMem, │   TMA Copy   │ Tensor<SMem, │  Tiled Copy │ Tensor<RMem, │
│   Layout>    │──────────────→│   Layout>    │────────────→│   Layout>    │
└──────────────┘              └──────────────┘             └──────────────┘
                                                                  │
                                                           Tiled MMA (atom)
                                                                  │
                                                                  v
                                                           Accumulator
                                                           Tensor<RMem>

Thread-Data Mapping via Layout Composition:

  data_layout = Layout<(128,64), (64,1)>     # 128×64 row-major tile
  thread_layout = Layout<(32,4), (4,1)>      # 32 threads, 4 values each
  ───── compose ─────
  access_pattern: Each thread automatically mapped to its 4 data elements
  No manual index arithmetic needed!
```

## Performance Impact
- **Comparable to hand-tuned CUTLASS 2.x**: No abstraction penalty; layout resolution is compile-time
- **Dozens of matrix math functions consolidated** into single vocabulary type calls
- **Architecture-agnostic**: Same kernel code targets Volta, Ampere, Hopper, Blackwell with different atoms
- **Reduced development time**: Complex kernels (FlashAttention, GEMM variants) can be built by composing existing atoms and layouts
- **Correctness guarantee**: "If it compiles, it will run correctly" — eliminates subtle indexing bugs that plague hand-tuned kernels

## When to Use
- Building custom attention kernels (FlashAttention variants) using CUTLASS 3.x
- Implementing GEMM kernels targeting multiple GPU architectures
- When you need to reason about how threads map to shared memory and register tiles
- Designing kernels that use TMA (the TMA descriptor integrates directly with CuTe layouts)
- When thread-data mapping complexity becomes the primary source of bugs
- Building reusable kernel components that work across Hopper and Blackwell

## When NOT to Use
- Using high-level frameworks (PyTorch, JAX) where kernel internals are abstracted
- Simple element-wise or reduction kernels where thread mapping is trivial
- Legacy CUTLASS 2.x codebases where migration cost exceeds benefit
- When targeting non-NVIDIA hardware (CuTe is NVIDIA-specific)
- Prototyping where Triton or similar DSLs offer faster iteration cycles

## Source Code Examples

### Layout Basics: Row-Major and Column-Major

```cpp
// A 4×8 row-major layout
Layout<Shape<_4, _8>, Stride<_8, _1>> layout_4x8;
// Logical coordinate (2, 3) → memory index: 2*8 + 3*1 = 19

// A 4×8 column-major layout
Layout<Shape<_4, _8>, Stride<_1, _4>> layout_4x8_col;
// Logical coordinate (2, 3) → memory index: 2*1 + 3*4 = 14
```

### Tensor Creation in Different Memory Spaces

```cpp
// A tensor in global memory
Tensor gmem_tensor = make_tensor(
    make_gmem_ptr(data_ptr),      // Engine: global memory pointer
    make_layout(Shape<_128, _64>{}, Stride<_64, _1>{})  // Layout
);

// A tensor in shared memory
Tensor smem_tensor = make_tensor(
    make_smem_ptr(smem_ptr),      // Engine: shared memory pointer
    smem_layout                    // Layout (possibly swizzled)
);

// A tensor in registers
Tensor reg_tensor = make_tensor(
    make_rmem_ptr(reg_ptr),       // Engine: register reference
    reg_layout                     // Layout
);
```

### CuTe GEMM Outer Loop with Partitioning

```cpp
// Pseudocode for a CuTe GEMM kernel inner loop
// 1. Partition tensors across threads according to TiledMMA
auto thr_mma = tiled_mma.get_slice(thread_idx);
auto tCrA = thr_mma.partition_A(sA);  // Shared memory A partition
auto tCrB = thr_mma.partition_B(sB);  // Shared memory B partition
auto tCrC = thr_mma.partition_C(gC);  // Register accumulator partition

// 2. Clear accumulators
clear(tCrC);

// 3. Execute GEMM: iterate over K dimension
for (int k = 0; k < K_TILES; ++k) {
    // Copy from shared memory to register memory
    copy(tCrA(_, _, k), rA);
    copy(tCrB(_, _, k), rB);

    // Execute MMA: (V,M,K) × (V,N,K) => (V,M,N)
    gemm(tiled_mma, rA, rB, tCrC);
}

// 4. Write back results
copy(tCrC, gC_partition);
```

## Key Takeaways
- CuTe's core insight: **threads and data should be described with the same vocabulary** (Layout), and their interaction should be expressed as **functional composition**
- Layout algebra replaces hundreds of lines of manual index arithmetic with a single compose operation
- Atoms encapsulate hardware-specific MMA instructions as reusable TV layouts, enabling architecture-agnostic kernel code
- The spatial (CuTe) vs temporal (CUTLASS) separation cleanly divides thread mapping from pipelining concerns
- CuTe is the abstraction layer that makes TMA, WGMMA, and threadblock clusters accessible without requiring PTX-level programming
- Understanding CuTe layouts is essential for reading or modifying any CUTLASS 3.x kernel, including FlashAttention-3

## References
- [CUTLASS: Principled Abstractions for Multidimensional Data (NVIDIA Developer Blog)](https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/)
- [CUTLASS 3.x GitHub Repository](https://github.com/NVIDIA/cutlass)
