# CUTLASS 3.x & CuTe Layout Abstractions

## Overview

CUTLASS 3.x introduces **CuTe** (CUDA Tensors and Spatial Microkernels), a foundational library that provides a unified abstraction for tensor layouts and thread-data mapping. CuTe treats threads and data with identical vocabulary types — `Layout<Shape,Stride>` and `Tensor<Engine,Layout>` — enabling developers to describe complex memory tiling, hardware-accelerated operations (WGMMA, TMA), and thread partitioning through composable, statically-checkable abstractions.

## Core Abstractions

### Layout<Shape, Stride>

The fundamental building block. A Layout maps **logical coordinates** within a Shape to **memory indices** using Stride values:

```cpp
// A 4×8 row-major layout
Layout<Shape<_4, _8>, Stride<_8, _1>> layout_4x8;
// Logical coordinate (2, 3) → memory index: 2*8 + 3*1 = 19

// A 4×8 column-major layout
Layout<Shape<_4, _8>, Stride<_1, _4>> layout_4x8_col;
// Logical coordinate (2, 3) → memory index: 2*1 + 3*4 = 14
```

Layouts go far beyond simple row-major or column-major — they can describe:
- Tiled memory layouts with hierarchical shapes
- Swizzled shared memory patterns
- Interleaved data arrangements for bank conflict avoidance
- Thread-value (TV) partitioning patterns

### Tensor<Engine, Layout>

Combines a Layout with an **iterator** (pointer to global, shared, or register memory):

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

The Tensor type packages **type information, shape, memory space, and layout** into a single composable unit. This enables generic algorithms that work across all memory levels.

## Layout Algebra: Functional Composition

CuTe's most powerful feature is **layout algebra** — transforming data organization by **composing layouts together**:

```
Given:
  data_layout:   Describes how data is arranged in memory
  thread_layout:  Describes how threads are organized

Composition:
  access_pattern = compose(data_layout, thread_layout)
  → Automatically generates the mapping from each thread to its data elements
```

This eliminates "one of the most complex hurdles in GPU programming" — consistently mapping thousands of threads to data elements. The composition **mechanically handles** coordinate transformation and index calculations.

### Example: TV Layout Partitioning

Consider partitioning a 4×8 data layout across threads:

1. Define a **TV (Thread-Value) Layout** that records how to assign threads (T) and their values (V) to each coordinate
2. **Functional composition** permutes and reshapes the data so each thread's values align in contiguous rows
3. **Simple slicing** by thread index completes partitioning — no hand-coded iteration schemes

```
Data Layout (4×8):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │  ← Thread 0's values
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 8 │ 9 │10 │11 │12 │13 │14 │15 │  ← Thread 1's values
├───┼───┼───┼───┼───┼───┼───┼───┤
│16 │17 │18 │19 │20 │21 │22 │23 │  ← Thread 2's values
├───┼───┼───┼───┼───┼───┼───┼───┤
│24 │25 │26 │27 │28 │29 │30 │31 │  ← Thread 3's values
└───┴───┴───┴───┴───┴───┴───┴───┘

After TV composition, each thread slices its row:
  thread(0) → [0, 1, 2, 3, 4, 5, 6, 7]
  thread(1) → [8, 9, 10, 11, 12, 13, 14, 15]
  ...
```

## Hardware Acceleration: Atoms

An **atom** represents the smallest hardware unit requiring cooperative thread participation. Each atom combines:

1. **A PTX instruction** (e.g., `SM70_8x8x4_F32F16F16F32_NT` for Volta, or WGMMA for Hopper)
2. **Metadata** describing participating threads and their value arrangements as CuTe TV layouts
3. **Reusable TV layout representations** applicable to arbitrary data layouts

### Supported Hardware Operations

| Feature | Architecture | Description |
|---------|-------------|-------------|
| **WGMMA** | Hopper H100 | Warp-group matrix multiply-accumulate (128 threads cooperate) |
| **UMMA** | Blackwell B200 | Unified matrix multiply-accumulate (next-gen MMA) |
| **TMA** | Hopper H100 | Tensor Memory Accelerator for async data movement |
| **Threadblock Clusters** | Hopper H100 | Cooperative multi-block execution with distributed shared memory |
| **Legacy MMA** | Volta/Ampere | Per-warp matrix multiply instructions |

Users do not need to extend the atom layer — NVIDIA provides implementations for each new architecture.

## Tiled MMA and Copy Operations

**Tiled MMA** and **Tiled Copy** scale atoms to larger operations by tiling them across threads and data:

```
Single Atom (8×8×4):          Tiled 2×2 (16×16×4):
┌────────┐                    ┌────────┬────────┐
│ atom   │                    │ atom_0 │ atom_1 │
│ 8×8×4  │        ───→       ├────────┼────────┤
└────────┘                    │ atom_2 │ atom_3 │
                              └────────┴────────┘
```

These operations:
- Reproduce atoms with possible permutations and interleaving
- Present hardware-accelerated operations with consistent APIs
- Enable partitioning arbitrary data layouts across thread hierarchies
- A single atom can be tiled in row-major, column-major, or custom permuted order

## GEMM Implementation with CuTe

CuTe enables writing **generic GEMM outer loops** with inner loops derived from atoms:

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

Decisions about **temporal micro-kernels** — pipelining, async copy staging, and SMEM layout optimization — remain at the CUTLASS level rather than CuTe's spatial microkernel focus.

## Separation of Concerns

CuTe maintains a clean separation:

| Layer | Responsibility | Example |
|-------|---------------|---------|
| **CuTe (Spatial)** | Thread-data mapping, layout algebra, partitioning | TV layouts, atoms, tiled MMA |
| **CUTLASS (Temporal)** | Pipelining, async staging, SMEM management, kernel scheduling | Pipeline APIs, TMA integration, warp specialization |
| **User (Algorithmic)** | Problem decomposition, tiling strategy, numerical algorithm | FlashAttention, GEMM variants |

## Key Simplifications over CUTLASS 2.x

CUTLASS 3.x + CuTe consolidates **dozens of matrix math functions into a single vocabulary type call**:
- Hierarchical shapes and strides handle sophisticated layouts
- Tensors remain "accessible just like normal tensors" regardless of complexity
- The API surface area is dramatically reduced
- **"If it compiles, it will run correctly"** — static checking catches layout mismatches at compile time

## Benefits Summary

- **Statically checkable correctness**: Layout composition errors caught at compile time
- **Layer-by-layer customization**: Replace any layer while preserving composability
- **Architecture-agnostic programming**: Same abstractions work for Hopper, Blackwell, and future hardware
- **Cleaner separation of concerns**: Spatial (CuTe) vs temporal (CUTLASS) optimizations
- **High performance**: Performance comparable to hand-tuned CUTLASS 2.x kernels
- **Reduced cognitive load**: Developers reason about logical layouts, not physical index arithmetic

## Sources

- [CUTLASS: Principled Abstractions for Handling Multidimensional Data (NVIDIA Developer Blog)](https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/)
- [CUTLASS 3.x GitHub Repository](https://github.com/NVIDIA/cutlass)
