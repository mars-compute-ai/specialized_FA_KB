---
skill_name: ThunderKittens Tile-Based CUDA Abstractions
description: Tile-based CUDA framework from Hazy Research (Flash Attention creators) that simplifies writing high-performance attention kernels through 4 core abstractions, achieving #1 on H100 attention benchmarks at release.
level: L0 - Model/Invocation Level
target_hardware: NVIDIA Hopper H100, NVIDIA Blackwell B200
relevance: When building custom attention variants, prototyping new Flash Attention algorithms, or needing a higher-level abstraction over raw CUDA/CUTLASS for attention-class kernels
---

# ThunderKittens Tile-Based CUDA Abstractions

## What It Is
ThunderKittens is a header-only CUDA framework from Hazy Research at Stanford (the group behind Flash Attention) that provides tile-level primitives for writing high-performance GPU kernels. It is built around the principle that modern GPUs are fundamentally 16x16 matrix multiply machines, and kernel development should work with tiles of this granularity rather than individual threads. ThunderKittens achieved #1 on H100 attention benchmarks at release and is used in production by Together AI, Jump Trading, and Cursor.

## 4 Core Abstractions

### 1. Register Tiles (`rt`)
- Tiles stored in warp registers, distributed across 32 threads
- Parameterized by data type (bf16, fp16, fp32), layout (row/column), and dimensions
- Example: `kittens::rt_bf<32, 64>` declares a 32x64 BF16 register tile
- Support WGMMA (Hopper) and TCGEN05 (Blackwell) tensor core instructions
- Operations: element-wise multiply, add, transpose, type conversion

### 2. Shared Tiles (`st`)
- Tiles in shared memory at the block scope
- Example: `kittens::st_bf<64, 64>` declares a 64x64 BF16 shared memory tile
- Automatic bank conflict avoidance via internal swizzling
- Support TMA (Tensor Memory Accelerator) async loads/stores

### 3. Register Vectors
- Column or row vectors associated with register tiles
- Three flavors: naive (compute-heavy like layernorm), aligned (column operations), orthogonal (row operations)
- Critical for softmax: row-max and row-sum reductions use column vectors
- Example: `kittens::rt_bf<32, 16>::col_vec` holds row-wise reduction results

### 4. Shared Vectors
- Vectors in shared memory for inter-warp communication
- Used to exchange reduction results (e.g., softmax statistics) between warps

## How It Simplifies Attention Kernels

ThunderKittens provides the Load-Store-Compute-Finish (LSCF) template for warp-specialized kernels:

```
Producer warps: TMA loads (Q, K, V tiles from HBM to shared memory)
Consumer warps: WGMMA matrix multiplies + softmax computation
Finish phase:   TMA stores (output tiles from shared to HBM)
```

An FA3-class attention kernel can be written in under 100 lines of ThunderKittens code. The framework handles:
- Asynchronous TMA data movement
- Warpgroup-level WGMMA instruction issuance
- Register/shared memory layout conversions
- Bank conflict-free shared memory access patterns
- Producer-consumer synchronization via barriers

## Performance

- **H100 attention**: #1 on benchmarks at release (740+ TFLOPS in FP16)
- **H100 GEMM**: 855 TFLOPS (86% of theoretical peak) in ~100 lines of code
- **FA3 implementation**: Matches or exceeds the reference FA3 implementation
- **Blackwell support**: ThunderKittens 2.0 adds B200 support with TCGEN05, MXFP8, NVFP4
- **Production**: Used by Together AI (training), Jump Trading, Cursor (inference)

## When to Use
- Building custom attention variants (linear attention, chunked attention, MLA, LoLCATS)
- Prototyping new Flash Attention algorithms before committing to hand-tuned CUDA
- Need production-quality attention kernels with less engineering effort than raw CUTLASS
- Targeting H100 or B200 GPUs where ThunderKittens has first-class support
- Want to contribute attention kernels to an active open-source ecosystem

## When NOT to Use
- Deploying standard Flash Attention without modifications (use official FA3/FA4 implementations)
- Targeting AMD GPUs (use HipKittens or Composable Kernel instead)
- Targeting Ampere (A100) or older GPUs (ThunderKittens no longer actively supports Ampere)
- Need a stable, versioned API (ThunderKittens is research-grade; the repo structure changes between releases)
- Writing non-attention kernels where existing libraries (cuBLAS, CUTLASS) are sufficient

## Code Snippets / Pseudo-code

```cpp
// Declaring types for a Flash Attention kernel
#include "kittens.cuh"
using namespace kittens;

// Register tiles for Q*K^T and P*V matmuls
rt_bf<16, 64> q_tile;         // Q tile in registers
st_bf<64, 64> k_tile, v_tile; // K, V tiles in shared memory

// Softmax statistics as column vectors
rt_bf<16, 64>::col_vec row_max, row_sum;

// Compute S = Q * K^T using WGMMA
warpgroup::mma_AB(s_acc, q_tile, k_tile);
warpgroup::mma_async_wait();

// Softmax reduction using register vectors
row_reduce(row_max, s_acc, kittens::base_ops::max);
```

## Source Code Examples

### Register Tile Declarations with Layout Semantics

```cpp
// kittens::rt_<type><rows, cols>
kittens::rt_bf<32, 64> tile;    // 32x64 BF16 register tile, row-layout (default)
kittens::rt_fl<16, 16> acc;     // 16x16 FP32 register tile (accumulator)
kittens::rt_hf<64, 64> big;    // 64x64 FP16 register tile
```

```cpp
// Shared tile declarations
__shared__ kittens::st_bf<64, 64> k_tile;  // 64x64 BF16 in shared memory
__shared__ kittens::st_hf<32, 64> v_tile;  // 32x64 FP16 in shared memory
```

Layout matters for correctness: ThunderKittens enforces layout compatibility at compile time via C++20 concepts. If you try to pass a row-layout tile as the B operand to `mma_AB`, you get a compile error, not a silent wrong answer.

### TMA Async Operations

On Hopper and Blackwell, shared tiles support asynchronous TMA (Tensor Memory Accelerator) operations:

```cpp
// Asynchronous load from global memory to shared tile
tma::load_async(shared_tile, global_layout, {row_idx, col_idx}, barrier);

// Asynchronous store from shared tile to global memory
tma::store_async(global_layout, shared_tile, {row_idx, col_idx});
```

TMA operations are issued by a single thread (typically `laneid() == 0`) and complete asynchronously, freeing the warp for compute.

### Register Vector Operations (Critical for Softmax)

```cpp
rt_fl<16, 64> s_acc;              // Attention scores tile
rt_fl<16, 64>::col_vec row_max;   // Max per row (16 elements)
rt_fl<16, 64>::col_vec row_sum;   // Sum per row (16 elements)

// Row-wise max reduction
row_reduce(row_max, s_acc, kittens::base_ops::max);

// Subtract max and exponentiate (for numerical stability)
sub_row(s_acc, s_acc, row_max);  // S[i][j] -= max[i]
exp(s_acc, s_acc);                // S[i][j] = exp(S[i][j])

// Row-wise sum
row_reduce(row_sum, s_acc, kittens::base_ops::sum);

// Normalize
div_row(s_acc, s_acc, row_sum);  // S[i][j] /= sum[i]
```

## Key Takeaways
- ThunderKittens bridges the gap between raw CUDA complexity and library-level opacity by providing tile-granularity primitives
- The 4 abstractions (register tiles, shared tiles, register vectors, shared vectors) map directly to GPU hardware resources
- Achieves near-peak performance (86%+ of theoretical) while dramatically reducing code complexity
- The LSCF (Load-Store-Compute-Finish) template encodes the warp specialization pattern that FA3 pioneered
- Active ecosystem with variants for AMD (HipKittens) and Apple Silicon (ThunderMittens)

## References
- [ThunderKittens GitHub Repository](https://github.com/HazyResearch/ThunderKittens)
- [GPUs Go Brrr (May 2024)](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
- [ThunderKittens Paper - Single GPU (arXiv 2410.20399)](https://arxiv.org/abs/2410.20399)
- [ThunderKittens Paper - Multi-GPU (arXiv 2511.13940)](https://arxiv.org/abs/2511.13940)
- [ThunderKittens 2.0 Blog (Feb 2026)](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)
- [HipKittens for AMD GPUs](https://github.com/HazyResearch/HipKittens)
- [ThunderMittens for Apple Silicon](https://github.com/HazyResearch/ThunderMittens)
