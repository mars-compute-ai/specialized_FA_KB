# Flash Attention 4: Reverse Engineering Summary

**Source:** https://modal.com/blog/reverse-engineer-flash-attention-4
**Context:** Modal's engineering team analyzed Flash Attention 4 (FA4), presented at Hot Chips by Tri Dao

## Overview

FA4 is a CUDA kernel achieving approximately 20% speedup over previous state-of-the-art attention implementations in NVIDIA's cuDNN library, optimized for the Blackwell GPU architecture.

## Architecture: The "Life of a Tile"

FA4 processes input tensors as tiles. A single kernel instance produces two output tiles by reading two query tiles while streaming all key-value tiles sequentially. This creates a "vectorized sequential scan" model analogous to database operations.

The kernel implements **asynchronous pipeline concurrency within each program instance** through warp specialization -- mapping pipeline steps onto 32-thread groups (warps). The warp scheduler switches between pipeline stages on each clock cycle, enabling simultaneous multithreading behavior.

## Five Warp Specializations

### 1. Load Warp
Loads query, key, and value tiles from global memory into shared memory using the Tensor Memory Accelerator (TMA) for asynchronous copies. Can buffer up to three K and V blocks concurrently.

### 2. MMA Warp
Executes matrix multiplications producing unnormalized attention scores (S) and accumulating score-weighted values into outputs (O). Uses inline PTX assembly with `tcgen05.mma.cta_group::1` instructions for Blackwell Tensor Cores.

### 3. Softmax Warps (8 total)
Compute normalized attention scores (P) and track running statistics. Two warpgroups handle the two query/output tile workstreams.

### 4. Correction Warps (4 total)
Rescale accumulated outputs when the numerical stability scaling factor changes.

### 5. Epilogue Warps (1-2)
Store completed output tiles from shared memory to global memory using TMA when possible.

## Novel Math Optimizations

### Fast Approximate Exponential for bf16

Instead of exclusively using Special Function Units (SFUs), FA4 employs a **cubic polynomial approximation** for smaller attention head sizes. The algorithm:

1. **Split exponentiation** into integer and fractional parts: `2^x = 2^floor(x) * 2^(x-floor(x))`
2. **Cubic polynomial** on unit interval: `0.07711909*r^3 + 0.22756439*r^2 + 0.69514614*r + 1.0`
3. **Evaluate via Horner's method** with three fused multiply-add (FMA) operations on `f32x2` vector lanes
4. Applied selectively on configurable iterations to avoid SFU bottlenecks

This technique stems from a 1999 Schraudolph paper but with quite different implementation targeting bf16 precision.

### Intelligent Softmax Rescaling

Previous Flash Attention versions updated the scaling factor whenever a new maximum appeared in attention scores. FA4 implements **conditional updates**: only when the maximum changes enough to impact numerical stability.

The logic checks whether new maxima warrant rescaling operations. Tri Dao reported this reduced correction operations by **approximately 10 times**, dramatically improving efficiency for long sequence attention computations.

## Synchronization Model

The kernel uses producer/consumer patterns with barrier synchronization in shared memory. The Load warp signals completion to downstream stages (MMA, Softmax, Correction) via barrier arrays indexed by offset to support variable configurations.

## Performance Characteristics

- **20% speedup** over cuDNN attention kernels
- **10x reduction** in rescaling operations (softmax optimization)
- Optimized for Blackwell Streaming Multiprocessor architecture with persistent block scheduling
- Uses `StaticPersistentTileScheduler` to launch one cooperative thread array per SM, reducing launch overhead

## GPU Programming Implications

FA4 exemplifies the swing towards tile-based, warp-specialized programming in contemporary GPU kernels. The authors note this increases complexity beyond traditional CUDA models, requiring manual event loop management similar to CPU async programming.

NVIDIA is investing in new abstractions (CuTe DSL, CUTLASS, forthcoming CuTile) to simplify warp-specialized kernel development, recognizing that manually managing asynchronous pipelines represents a major jump in complexity.

## Algorithm Pseudo-code

```
# FA4 Forward Pass (Blackwell, per-CTA view)
# Two query tiles processed simultaneously

# LOAD WARP:
for each KV block j:
    async_copy(K_j, V_j -> SMEM)  # via TMA, triple-buffered
    signal(barrier[j % 3])

# MMA WARP (for each of 2 query tiles):
for each KV block j:
    wait(barrier[j % 3])  # K_j ready
    S = Q_tile @ K_j^T    # via tcgen05.mma Blackwell Tensor Cores
    signal(softmax_barrier)
    wait(softmax_done)     # P ready
    O += P @ V_j           # accumulate

# SOFTMAX WARPS:
for each KV block j:
    wait(softmax_barrier)
    m_new = rowmax(S)
    if (m_new - m_old) > threshold:  # CONDITIONAL rescaling
        signal(correction_needed)
    P = fast_exp2(S - m_new)  # cubic polynomial approx for bf16
    l += rowsum(P)
    signal(softmax_done)

# CORRECTION WARPS:
when signaled:
    O *= exp2(m_old - m_new)  # rescale accumulated output
    # ~10x fewer corrections than unconditional approach

# EPILOGUE WARPS:
when query tile complete:
    O = O / l  # final normalization
    async_store(O -> GMEM)  # via TMA
```
