# CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA Hopper GPUs

Source: https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/

## Overview

This tutorial explains efficient GEMM kernel implementation on NVIDIA Hopper GPUs using CUTLASS, focusing on the Warpgroup Matrix-Multiply-Accumulate (WGMMA) instruction -- the primitive operation targeting Tensor Cores on Hopper architecture.

## WGMMA Instruction Fundamentals

### Definition

WGMMA is an asynchronous warpgroup-level matrix operation where a "warpgroup" consists of 128 contiguous threads (4 warps, warp-rank must be a multiple of 4). The instruction executes collectively across all participating threads.

**Basic operation forms:**
- `C = A * B + C` (accumulate mode)
- `C = A * B` (accumulator input disabled)

### Operand Memory Constraints

- **Operand B**: Always stored in shared memory (SMEM)
- **Operand A**: Can reside in either SMEM or register memory (RMEM)
- **Accumulator C**: Always held in register memory

### Supported Tile Shapes

- M dimension: fixed at **64**
- N dimension: multiples of 8 (range: **8 to 256**)
- K dimension: fixed at **16** for 16-bit operand datatypes (32 bytes total)

This produces the notation `m64nNk16` where N varies according to requirements.

## TiledMMA Object Configuration

### MMA Atom Specification

```cpp
// FP16 with MN-major operands (NT GEMM), both operands from shared memory
TiledMMA tiled_mma = cute::make_tiled_mma(
    SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

// Two warpgroups with independent WGMMA operations
TiledMMA tiled_mma = make_tiled_mma(
    SM90_64x64x16_F16F16F16_SS{},
    Layout<Shape<_2, _1, _1>>{});  // 256 threads total
```

### Naming Convention: SM90_MxNxK_XYZ_SS

- **X, Y**: operand datatypes
- **Z**: accumulator datatype
- **MxNxK**: tile dimensions
- **First S/R**: operand A source (S=shared, R=register)
- **Second S**: operand B always from shared memory
- **Template parameters**: memory contiguity mode (MN-major or K-major)

**Important constraint:** For non-16-bit datatypes, layouts must be K-major.

## Shared Memory Layout Requirements

### Tile Size Constraints

Based on MMA atom selection:
- bM must be a multiple of 64
- bN must be a multiple of 64
- bK must be a multiple of 16

### Layout Atoms and Swizzling

Eight predefined layout atom options:

**MN-major variants:**
- `GMMA::Layout_MN_INTER_Atom<T>` (no swizzle, 16-byte boundary)
- `GMMA::Layout_MN_SW32_Atom<T>` (32-byte swizzle)
- `GMMA::Layout_MN_SW64_Atom<T>` (64-byte swizzle)
- `GMMA::Layout_MN_SW128_Atom<T>` (128-byte swizzle)

**K-major variants:** Same naming with `_K_` instead of `_MN_`

### Layout Construction

```cpp
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int< 64>{};
auto bP = Int<  3>{};  // Pipeline stages

auto sA = cute::tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<T>{},
    cute::make_shape(bM, bK, bP));
auto sB = cute::tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<T>{},
    cute::make_shape(bN, bK, bP));
```

## Fragments and Matrix Descriptors

### Thread-Level Partitioning

```cpp
ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K, PIPE)
Tensor tCsB = thr_mma.partition_B(sB);  // (MMA, MMA_N, MMA_K, PIPE)
Tensor tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)
```

### Matrix Descriptors (Not Register Copies)

"Instead of copying SMEM values into registers, CUTLASS constructs a 64-bit matrix descriptor held in registers that describes shared memory in a manner suitable for WGMMA instruction usage."

This is a key difference from WMMA: WGMMA does not copy operands to registers. Instead, descriptors point into shared memory, and the tensor core hardware reads directly.

### Accumulator Layout

The accumulator follows a replicated Z-pattern distribution across threads. For a 64x64 output tile:
- 128 threads each hold 32 values
- Values distribute in a specific spatial pattern repeated across columns
- Shape factorization: `(MMA, MMA_M, MMA_N)` following the atom tiling

## GEMM Invocation and Synchronization

### The cute::gemm Call with Async Primitives

```cpp
cute::warpgroup_arrive();
cute::gemm(tiled_mma, tCrA(_, _, _, read_pipe),
           tCrB(_, _, _, read_pipe), tCrC);
cute::warpgroup_commit_batch();
cute::warpgroup_wait<0>();
```

### Synchronization Primitives

- `cute::warpgroup_arrive()`: Signals warpgroup entry to hardware scheduler
- `cute::warpgroup_commit_batch()`: Commits MMA batch to hardware queue
- `cute::warpgroup_wait<N>()`: Waits until at most N outstanding MMA batches remain

These primitives ensure correct data dependencies and enable asynchronous instruction completion.

## Kernel Architecture

### Three-Phase Structure

1. **Prologue**: Initialization, SMEM buffer setup, initial data loads
2. **Main Loop**: Pipelined iterations over K dimension
   - Load A and B tiles from global to shared memory (circular buffers)
   - Execute WGMMA operations on loaded tiles
   - Accumulate results into register-backed accumulators
3. **Epilogue**: Write accumulated results from registers to global memory

### Pipeline Strategy

Circular SMEM buffers with compile-time stage counts (typically 2-3), allowing overlapping of computation and memory transfers. The last dimension of SMEM shapes corresponds to the stage count.

## PTX Assembly Integration

The WGMMA instruction maps to inline PTX:

```
wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16
```

This instruction:
- Takes two 64-bit matrix descriptors (for A and B)
- Operates on 16 output registers (32 values per thread)
- Includes predicate control for conditional execution
- Supports scale factors for output

## Key Design Principles

1. **Asynchronous execution**: WGMMA instructions issue asynchronously, enabling computation/communication overlap
2. **Descriptor-based SMEM access**: Eliminates copying SMEM data to registers for operands A and B
3. **Warpgroup collective execution**: All 128 threads must participate in synchronized manner
4. **Layout flexibility**: MN-major or K-major options (K-major required for non-16-bit types)
5. **Swizzle optimization**: Multiple swizzle modes (32B, 64B, 128B) reduce bank conflicts during SMEM access
6. **Fixed M=64**: The M dimension is always 64 in the atom; larger M tiles require tiling multiple atoms
