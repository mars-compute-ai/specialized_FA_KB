# CUTLASS & CuTe Instruction Mapping

Source: https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/

## Overview

CuTe (CUDA Templates) provides a hierarchical layout algebra that maps high-level tensor operations to specific hardware instructions. The framework uses composable abstractions -- Layouts, Atoms, and Tiled operations -- to expose template parameters for tile shapes, warp shapes, instruction shapes, and pipeline stages. This enables developers to precisely control the sequence of MMA and load/store instructions generated for custom GPU kernels.

## Layout Algebra

### Core Abstraction: Layout<Shape, Stride>

CuTe introduces `Layout<Shape, Stride>` objects that map logical coordinates to physical memory indices. This is the "unified vocabulary type" across data tensors and thread tensors.

```cpp
// A 2D layout: 4 rows x 8 columns, column-major
Layout layout = make_layout(make_shape(4, 8), make_stride(1, 4));

// Mapping: logical coordinate (row, col) -> physical index
// (0,0) -> 0, (1,0) -> 1, (2,0) -> 2, (3,0) -> 3
// (0,1) -> 4, (1,1) -> 5, ...
```

Key properties:
- Layouts can be composed: `composition(layout_A, layout_B)` chains the mappings
- Layouts can be partitioned: divide one layout across another
- "Build complicated layouts from simple known layouts or partition one layout across another"

### Hierarchical Layout Composition

```
Logical Tensor Shape: (M, N, K)
         |
    Tile Partitioning: (BLK_M, BLK_N, BLK_K)
         |
    Warp Partitioning: (WARP_M, WARP_N)
         |
    Instruction Shape: (INST_M, INST_N, INST_K)
         |
    Thread-Value Layout: per-thread register mapping
```

## MMA Atoms: Minimal Instruction Units

MMA Atoms represent the smallest cooperative thread groups executing a single hardware MMA instruction.

### Example: SM70 (Volta) Atom

```cpp
// SM70_8x8x4_F32F16F16F32_NT
// - Shape: 8x8x4 (M=8, N=8, K=4)
// - Types: F32 accumulator, F16 inputs
// - Layout: N-transpose for A, T-transpose for B

struct SM70_8x8x4_F32F16F16F32_NT {
    // Thread layout: how 32 threads in a warp map to the 8x8 output
    using ThrID = Layout<_32>;

    // Value layouts: how each thread's registers map to matrix elements
    using ALayout = Layout<Shape<_8, _4>, Stride<_4, _1>>;  // 8x4 A tile
    using BLayout = Layout<Shape<_8, _4>, Stride<_4, _1>>;  // 8x4 B tile
    using CLayout = Layout<Shape<_8, _8>, Stride<_8, _1>>;  // 8x8 C tile
};
```

### MMA_Traits Metadata

`MMA_Traits` encode thread-to-value mappings that can be visualized and composed:

```cpp
template <class MMA_Op>
struct MMA_Traits {
    using ValTypeA = ...;  // Input A data type
    using ValTypeB = ...;  // Input B data type
    using ValTypeC = ...;  // Accumulator data type

    using Shape_MNK = Shape<_M, _N, _K>;  // Instruction shape

    using ThrID  = ...;  // Thread identification layout
    using ALayout = ...;  // A operand thread-value layout
    using BLayout = ...;  // B operand thread-value layout
    using CLayout = ...;  // C accumulator thread-value layout
};
```

## Tiled Operations: Instruction Composition

### Tiled MMA

CuTe extends atoms through tiled MMAs, which "tile together individual operations as if fitting together tiles to build a reusable component of a mosaic."

```cpp
// 2x2 row-major tiling of SM70_8x8x4 atom
// Produces a 16x16x4 operation across one warp
auto tiled_mma = make_tiled_mma(
    SM70_8x8x4_F32F16F16F32_NT{},
    Layout<Shape<_2, _2>>{},     // 2x2 tiling pattern
    Layout<Shape<_1, _1>>{}      // No permutation
);
// Result: each warp computes 16x16 output using 4 individual 8x8 MMA instructions
```

### Instruction Shape Selection

Different architectures support different MMA instruction shapes:

| Architecture | Instruction | Shape (MxNxK) | Throughput |
|-------------|-------------|---------------|------------|
| SM70 (Volta) | `mma.sync` | 8x8x4 | Baseline |
| SM75 (Turing) | `mma.sync` | 16x8x8 | 2x |
| SM80 (Ampere) | `mma.sync` | 16x8x16 | 4x |
| SM90 (Hopper) | `wgmma.mma_async` | 64x128x16+ | 16x+ |
| SM100 (Blackwell) | `tcgen05.mma` | Variable | 32x+ |

The choice of instruction shape affects:
- Register pressure (larger shapes use more registers per thread)
- Instruction count (larger shapes compute more output per instruction)
- Occupancy (more registers per thread = fewer concurrent warps)

### Tiled Copy

Similar to tiled MMA, tiled copy composes individual load/store operations:

```cpp
// Construct a tiled copy from a copy atom
auto tiled_copy = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_16, _8>>{},   // Thread layout
    Layout<Shape<_1, _8>>{}     // Value layout per thread
);
```

## GEMM Mainloop: Instruction Mapping in Practice

```cpp
// Tile of global memory for this threadblock
Tensor gA = ...;  // Tile of 64x16 gmem for A
Tensor gB = ...;  // Tile of 96x16 gmem for B
Tensor gC = ...;  // Tile of 64x96 gmem for C

// Shared memory tiles
Tensor sA = make_tensor(sA_ptr, SmemLayoutA{});
Tensor sB = make_tensor(sB_ptr, SmemLayoutB{});

// Partition across threads for MMA
ThrMMA thr_mma = tiled_mma.get_slice(thread_idx);
Tensor tCsA = thr_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K) smem
Tensor tCsB = thr_mma.partition_B(sB);  // (MMA, MMA_N, MMA_K) smem
Tensor tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N) gmem

// Register fragments
Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // Register tile for A
Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // Register tile for B
Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // Accumulator registers

// Main compute loop
for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    // Copy from shared memory to registers
    cute::copy(tCsA(_, _, k_tile), tCrA);
    cute::copy(tCsB(_, _, k_tile), tCrB);

    // Execute tiled MMA (maps to hardware MMA instructions)
    cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
}
```

The `cute::gemm` call maps to a sequence of hardware MMA instructions determined by:
1. The atom's instruction shape (e.g., 16x8x16)
2. The tiling pattern (e.g., 2x2)
3. The tile dimensions (e.g., 64x96)
4. Unrolling factor along K dimension

## Architecture Support

CUTLASS 3.x supports:
- **Hopper H100** (SM90): WGMMA with Tensor Memory Accelerator, threadblock clusters
- **Blackwell B200** (SM100): UMMA (Unified MMA), `tcgen05.mma`
- **Ampere A100** (SM80): `mma.sync` with cp.async
- **Turing** (SM75) and **Volta** (SM70): Basic `mma.sync`

### Hopper-Specific Features

```cpp
// WGMMA: Warpgroup MMA (4 warps = 128 threads)
// Can read one operand directly from shared memory
auto tiled_mma = make_tiled_mma(
    SM90_64x128x16_F16F16F32_SS{},  // SS = both operands from smem
    Layout<Shape<_1, _1>>{}
);

// TMA copy atom
auto tma_load = make_tma_copy(
    SM90_TMA_LOAD{},
    sA,           // Shared memory tensor
    SmemLayout{}  // Swizzled layout for bank-conflict-free access
);
```

## Unrolling and Instruction Count

The number of MMA instructions per mainloop iteration is:

```
num_mma_per_iter = (TILE_M / INST_M) * (TILE_N / INST_N) * (TILE_K / INST_K)
```

Example: Tile 128x128x32 with instruction 16x8x16:
```
(128/16) * (128/8) * (32/16) = 8 * 16 * 2 = 256 MMA instructions per iteration
```

Unrolling the K-loop by factor U reduces loop overhead:
```
Total instructions per unrolled iteration = 256 * U MMA + load/store + sync
```

The compiler and programmer must balance:
- More unrolling = fewer loop control instructions, better instruction-level parallelism
- More unrolling = higher register pressure, potentially lower occupancy
