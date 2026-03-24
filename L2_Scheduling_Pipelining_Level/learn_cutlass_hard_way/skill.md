---
skill_name: GEMM Pipelining and Tiling Optimization Progression
description: Systematic progression from naive GEMM to pipelined tensor core kernels, covering tiling hierarchies, double buffering, WMMA, swizzling, persistent kernels, and autotuning.
level: L2 - Scheduling & Pipelining Level
target_hardware: NVIDIA Ada Lovelace RTX 4090 (demonstrated), applicable to Ampere A100, Hopper H100/H200
relevance: When an AI agent needs to understand the full optimization stack for GEMM/attention kernels—from memory coalescing through multi-stage pipelining—to make informed decisions about tile sizes, pipeline depth, and register budgets in FlashAttention implementations.
---

# GEMM Pipelining and Tiling Optimization Progression

## What It Is
A systematic, incremental approach to optimizing GPU matrix multiplication (GEMM) that progresses from a naive one-thread-per-element implementation (0.8% of peak) through seven optimization layers—memory coalescing, shared memory tiling, 1D/2D block tiling, vectorized loads, warp-level tiling, tensor cores (WMMA), and double-buffered pipelining—ultimately achieving 54%+ of cuBLAS performance. Each layer maps to a level of the GPU memory/compute hierarchy and builds the foundational concepts used in production CUTLASS kernels and FlashAttention implementations.

## Key Concepts
- **Memory coalescing**: Ensuring consecutive threads access consecutive memory addresses for maximum bandwidth utilization (7.8x speedup over naive)
- **Multi-level tiling hierarchy**: Block tiles (thread block level) -> Warp tiles (warp level) -> Thread tiles (register level), mirroring GPU execution structure
- **Register-level reuse**: Each thread computes a TM x TN output tile, reusing loaded values across multiple multiply-accumulate operations to increase arithmetic intensity
- **Vectorized memory access**: Using `float4` (128-bit) loads to transfer 4 elements per transaction, reducing instruction count
- **WMMA (Warp Matrix Multiply-Accumulate)**: Hardware tensor core API providing warp-collective MMA operations (e.g., `mma.sync.aligned.m16n8k8`)
- **Double buffering / software pipelining**: Two shared memory buffers alternating between load and compute phases, overlapping data movement with computation
- **Persistent kernels**: One CTA per SM computing multiple output tiles, enabling deeper pipelines and up to 2x speedup on small matrices
- **Swizzling**: Rearranging shared memory data layout to eliminate bank conflicts (32/64/128-byte swizzle modes)
- **Autotuning**: Systematic exploration of tile shapes (BM, BN, BK), pipeline stages, thread counts, and register budgets

## Pipeline Architecture / Pseudo-code
```
# Double-Buffered Pipelining (foundation for multi-stage)

__shared__ half tile_a[2][BM * BK]   # 2 buffers for A
__shared__ half tile_b[2][BK * BN]   # 2 buffers for B

# Prologue: load first tile into buffer 0
load_tile_async(A_global, tile_a[0])
load_tile_async(B_global, tile_b[0])
__syncthreads()

for k = 0 to K/BK:
    read_buf  = k % 2
    write_buf = (k + 1) % 2

    # Stage 1: Prefetch next tile into write_buf (overlapped)
    if k + 1 < K/BK:
        load_tile_async(A_global[k+1], tile_a[write_buf])
        load_tile_async(B_global[k+1], tile_b[write_buf])

    # Stage 2: Compute on current read_buf
    for bk = 0 to BK:
        # Register-level: each thread loads TM values from tile_a
        #                  and TN values from tile_b into registers
        for tm = 0 to TM:
            for tn = 0 to TN:
                result[tm][tn] += reg_a[tm] * reg_b[tn]

    __syncthreads()  # Ensure both load and compute complete

# Multi-stage generalization (N stages):
# - N shared memory buffer sets
# - Pipeline state tracks (index mod N, phase_bit)
# - producer_acquire / producer_commit / consumer_wait / consumer_release
# - Deeper pipeline (3-7 stages) hides more latency
```

## Performance Impact
- **Naive to coalesced**: 7.8x speedup (0.65 -> 5.10 TFLOPS)
- **Coalesced to shared memory tiling**: 1.2x (5.10 -> 6.06 TFLOPS)
- **Shared memory to 1D block tiling**: 3.0x (6.06 -> 18.40 TFLOPS)
- **1D to 2D block tiling**: 1.7x (18.40 -> 30.55 TFLOPS)
- **2D tiling to vectorized**: 1.3x (30.55 -> 39.00 TFLOPS)
- **Vectorized to warp tiling**: 1.2x (39.00 -> 45.82 TFLOPS)
- **Adding tensor cores + double buffering (BF16)**: ~58 TFLOPS
- **Double buffering alone**: 30%+ improvement over non-pipelined tensor core kernel
- **Persistent kernels with increased pipeline stages**: Up to 2x speedup on small matrices
- **Overall progression**: 0.65 TFLOPS (naive) -> 45.82 TFLOPS (warp tiling FP32) = **70x speedup**

## When to Use
- Designing custom GEMM or attention kernels that need to approach peak hardware throughput
- Choosing tile shapes (BM, BN, BK) and pipeline depth for FlashAttention microkernels
- Debugging performance—understanding which optimization layer is underperforming
- Deciding between FP32 manual kernels vs. BF16/FP8 tensor core kernels (tensor cores required for competitive low-precision performance)
- Determining appropriate pipeline depth: 2 stages (double buffer) for simple cases, 3-7 stages for production Hopper kernels
- Autotuning kernel configurations across the tile shape / pipeline stage / register budget space

## When NOT to Use
- Using high-level frameworks (PyTorch, JAX) where GEMM is already library-optimized and custom kernels are unnecessary
- Bandwidth-bound operations (e.g., elementwise, reduction) where tiling provides no compute reuse benefit
- Very small matrices where kernel launch overhead dominates and batching/fusion is a better strategy
- When Triton or other compiler-based approaches can automatically handle tiling and pipelining decisions

## Source Code Examples

### Double Buffer Declaration (C++)

```cpp
__shared__ InputType tile_a[2][BM * BK];  // Double buffer
__shared__ InputType tile_b[2][BK * BN];
```

### K-Loop with Buffer Toggling (C++)

```cpp
// K-loop with buffer toggling
for (int k = 0; k < K; k += BK) {
    int read_buf = k_iter % 2;
    int write_buf = (k_iter + 1) % 2;

    // Load next tile into write_buf (async/prefetch)
    load_tile(A, tile_a[write_buf], ...);
    load_tile(B, tile_b[write_buf], ...);

    __syncthreads();

    // Compute using read_buf
    compute(tile_a[read_buf], tile_b[read_buf], result);

    __syncthreads();
}
```

### Global Memory Coalescing (C++)

```cpp
output_row = blockIdx.x * block_size + (threadIdx.x / block_size);
output_col = blockIdx.y * block_size + (threadIdx.x % block_size);
```

### 1D Block Tiling with Register Reuse (C++)

```cpp
float b_tmp = tile_b[dot_idx * BN + thread_col];
for (uint res_idx = 0; res_idx < TM; ++res_idx) {
    thread_results[res_idx] += tile_a[...] * b_tmp;
}
```

### WMMA Tensor Core Usage

WMMA provides warp-level collective matrix multiply operations. The hardware executes instructions like `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` for 16x8x8 operations. With WMMA and double buffering on BF16 matrices, performance reaches ~58 TFLOPS (~40% of PyTorch), demonstrating that tensor cores are mandatory for competitive low-precision performance (BF16 without WMMA achieves only ~25%).

## Key Takeaways
- The biggest single optimization jump comes from **register-level tiling** (1D/2D block tiling): going from 7.7% to 38.7% of peak by increasing arithmetic intensity
- **Memory coalescing** is the prerequisite: without it, all other optimizations are bottlenecked by inefficient global memory access
- **Double buffering is the entry point to pipelining**: it overlaps load[k+1] with compute[k], and generalizes to N-stage pipelines
- **Tensor cores are mandatory for competitive low-precision performance**: BF16 without WMMA achieves only 25% of PyTorch; with WMMA, it reaches 40%
- **Persistent kernels** eliminate launch overhead and enable deeper pipelines, critical for small-matrix workloads common in attention (small sequence lengths)
- The **tiling hierarchy** (block -> warp -> thread) maps directly to GPU hardware (thread block -> warp -> registers) and is the fundamental design pattern in CUTLASS and FlashAttention
- **Nsight Compute (ncu)** profiling is essential: instruction mix analysis, occupancy, bank conflicts, and memory access patterns guide optimization decisions

## References
- Kapil Sharma, "Learn CUTLASS the Hard Way": https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/
- NVIDIA CUTLASS library: https://github.com/NVIDIA/cutlass
- NVIDIA Nsight Compute documentation
- Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance" (related work)
