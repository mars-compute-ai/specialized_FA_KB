# Learn CUTLASS the Hard Way

**Source**: https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/
**Author**: Kapil Sharma

## Overview

This comprehensive tutorial walks through GEMM (General Matrix Multiply) optimization from naive implementations to production-ready CUTLASS kernels on RTX 4090 hardware. The progression demonstrates how each optimization layer—coalescing, tiling, vectorization, tensor cores, and pipelining—contributes to closing the gap with library-optimized kernels.

## Core GEMM Formula

**C = alpha * A * B + beta * C**

Where A is M x K, B is K x N, and C is M x N with scalar coefficients alpha and beta.

## Hardware Context (RTX 4090)

- **FP32 Performance**: 82.6 TFLOPS
- **Memory Bandwidth**: 1,008 GB/s
- **Tensor Cores**: 512 (4th Gen)
- **Shared Memory per SM**: 128 KB
- **Streaming Multiprocessors**: 128

## Optimization Progression

### 1. Naive Implementation
Basic approach assigning one thread per output element achieved only **0.76% of PyTorch performance** (0.65 TFLOPS).

### 2. Global Memory Coalescing
Restructuring thread-to-output mapping for consecutive memory access improved performance to **5.8% of baseline** (5.10 TFLOPS). Key change—swapping modulo operations ensures consecutive threads access adjacent memory:
```cpp
output_row = blockIdx.x * block_size + (threadIdx.x / block_size);
output_col = blockIdx.y * block_size + (threadIdx.x % block_size);
```

### 3. Shared Memory Caching
Loading tile chunks into fast on-chip shared memory provided modest gains, reaching **7.8% of PyTorch performance** (6.06 TFLOPS) through reduced global memory traffic.

### 4. 1D Block Tiling
Each thread computed multiple outputs (TM parameter) instead of single elements, increasing arithmetic intensity to **21.8% of baseline** (18.40 TFLOPS) through register-level caching.
```cpp
float b_tmp = tile_b[dot_idx * BN + thread_col];
for (uint res_idx = 0; res_idx < TM; ++res_idx) {
    thread_results[res_idx] += tile_a[...] * b_tmp;
}
```

### 5. 2D Block Tiling
Extending to TM x TN output computation per thread achieved **38.7% of PyTorch performance** (30.55 TFLOPS) through bidirectional register reuse patterns.

### 6. Vectorized Memory Access
Using `float4` loads for coalesced 128-bit transfers improved large matrix performance to **46.3% of baseline** (39.00 TFLOPS). Single 128-bit transaction replaces four separate 32-bit operations.

### 7. Warp-Level Tiling
Exploiting the fundamental 32-thread warp execution unit reached **54.4% of PyTorch performance** (45.82 TFLOPS) through an additional tiling hierarchy level.

## Lower Precision Support

### FP16/BF16 Baseline
Initial ports without tensor core support achieved only ~25% of PyTorch performance, demonstrating the necessity of specialized hardware instructions.

### WMMA and Tensor Cores
Leveraging NVIDIA's Warp Matrix Multiply-Accumulate (WMMA) API:
- Provides warp-level collective operations
- Exposes hardware matrix multiply instructions
- Example instruction: `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` for 16x8x8 operations
- **Performance with WMMA**: 34-40% of PyTorch for BF16 matrices

## Advanced Optimization Techniques

### Double Buffering / Software Pipelining
Producer-consumer pattern overlapping memory loads with computation. Dedicates separate shared memory buffers allowing simultaneous prefetch of next tile while computing current buffer. **Achieved 30%+ performance improvement**.

Key implementation:
```cpp
__shared__ InputType tile_a[2][BM * BK];  // Double buffer
__shared__ InputType tile_b[2][BK * BN];

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

### Occupancy Considerations
Balances register usage, shared memory allocation, and thread block size:
- Higher occupancy enables latency hiding but risks reduced per-thread resources
- Formula: `Max Threads = min(registers_available / registers_per_thread, hardware_limit)`
- Trade-off between more threads (hiding latency) and more registers per thread (reducing spills)

### Swizzling
Memory access pattern optimization to prevent shared memory bank conflicts. CUTLASS provides swizzle modes (32-byte, 64-byte, 128-byte) that rearrange data layout in SMEM for conflict-free access.

### Persistent Kernels
A single CTA per SM that computes multiple output tiles sequentially, avoiding kernel launch overhead and enabling better pipeline utilization. Increasing pipeline stages provides up to **2x speedup on small matrices**.

### Autotuning
Systematically exploring the configuration space:
- Tile shapes (BM, BN, BK)
- Thread block sizes
- Pipeline stages
- Register budget allocation
- Warp tiling dimensions

## Key Performance Metrics

| Kernel | TFLOPS | vs PyTorch | Speedup vs Naive |
|--------|--------|-----------|------------------|
| Naive | 0.65 | 0.8% | 1.0x |
| Coalesced | 5.10 | 6.5% | 7.8x |
| Shared Memory | 6.06 | 7.7% | 9.3x |
| 1D Block Tiling | 18.40 | 23.3% | 28.2x |
| 2D Block Tiling | 30.55 | 38.7% | 46.9x |
| Vectorized | 39.00 | 46.3% | 59.7x |
| Warp Tiling | 45.82 | 54.4% | 70.1x |
| TC + Double Buffer (BF16) | ~58 | ~40% | - |

## CUTLASS Framework Motivation

The tutorial motivates CUTLASS as essential abstraction for modern GEMM development:
- **Hierarchical tile loading**: global -> shared -> registers
- **Multi-level tiling strategy** matching GPU memory hierarchy
- **C++ template-based abstractions** reducing manual optimization complexity
- Particularly valuable for advanced architectures (Hopper, Blackwell) with TMA, WGMMA, and warp specialization

## Critical Insights

1. **Memory bandwidth** fundamentally limits unoptimized kernels before compute becomes the bottleneck
2. **Arithmetic intensity** (FLOPS per byte) drives optimization focus—register-level reuse is essential
3. **Multiple tiling levels** (block, warp, thread) mirror GPU execution hierarchy
4. **Lower precision** (BF16, FP8) requires hardware tensor core support for competitive performance
5. **Synchronization points** (`__syncthreads()`) are necessary between stages but incur latency costs
6. **Double buffering** provides the foundation for deeper multi-stage pipelines used in production kernels
7. **Persistent kernels** with increased pipeline stages can provide up to 2x speedup on small matrices

## Profiling Tools

The tutorial emphasizes NVIDIA Nsight Compute (ncu) for:
- Instruction mix analysis (identifying LDS saturation)
- Occupancy verification
- Bank conflict detection
- Memory access pattern optimization

## Conclusion

This tutorial demonstrates how "doing GEMM the hard way" through incremental optimization builds deep understanding necessary for modern GPU programming, ultimately validating why production frameworks rely on sophisticated abstractions like CUTLASS rather than hand-written kernels. The concepts—tile shapes, swizzling, pipelining, register budgets, and autotuning—apply directly to FlashAttention microkernels.
