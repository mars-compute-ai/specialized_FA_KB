# Flash Attention 4: Reverse Engineering Analysis

**Source:** https://modal.com/blog/reverse-engineer-flash-attention-4
**Author:** Modal Engineering Team

## Overview

Modal's engineering team reverse-engineered NVIDIA's Flash Attention 4 kernel, which delivers approximately 20% speedup over cuDNN attention kernels on Blackwell GPUs. The analysis reveals that the architecture relies on sophisticated asynchronous pipeline management rather than novel mathematics.

## Core Architecture: Tile-Based Processing

The kernel processes attention computation by dividing inputs into tiles. One cooperative thread array (CTA) processes two query tiles while streaming all key-value tiles, similar to a vectorized sequential scan for a batch of aggregation queries against a key-value store.

## Warp Specialization Model

FA4 employs five distinct warp specializations:

### 1. Load Warp
- Transfers query, key, and value tiles from global memory to shared memory
- Uses Tensor Memory Accelerator (TMA) for asynchronous operations
- Minimal register usage since TMA handles the heavy lifting

### 2. MMA Warp
- Executes matrix multiplication via Tensor Cores
- Produces unnormalized attention scores (S = Q * K^T)
- Accumulates weighted values (O += P * V)
- Uses 5th generation Tensor Core instruction: `tcgen05.mma.cta_group::1` (single-CTA variant)

### 3. Eight Softmax Warps
- Normalize attention scores to produce probability distribution P
- Track running statistics (row max, row sum) for numerical stability
- Organized as a warpgroup (8 warps) for improved distribution across warp schedulers

### 4. Four Correction Warps
- Rescale output tiles when normalization parameters change
- Key optimization: scaling factor is only updated when the new maximum changes enough to threaten numerical stability, not every time a new maximum appears
- This optimization reduced the number of corrections by a factor of 10
- Organized as part of a warpgroup alignment

### 5. Epilogue Warp(s)
- Store completed output tiles to global memory via TMA
- Final normalization and writeback

## Key Technical Innovations

### Efficient Online Softmax

The critical optimization is that the scaling factor only needs updating when the new maximum changes enough to threaten numerical stability, not every time a new maximum appears. This dramatically reduces the work done by correction warps -- reducing corrections by a factor of 10.

### Approximate Exponentials

For smaller attention head sizes, FA4 employs cubic polynomial approximation for exponential calculations instead of relying solely on Special Function Units (SFUs). The approximation uses Horner's method with three fused multiply-adds to match bf16 precision while avoiding SFU bottlenecks from wave quantization effects.

### Memory Hierarchy Utilization

The "life of a tile" traces data through:
1. **Global memory (GPU RAM)** -- source of Q, K, V tiles
2. **Shared memory (programmer-managed L1)** -- staging area for TMA loads
3. **Tensor Memory (L1 cache for Tensor Core intermediates)** -- used by 5th gen Tensor Cores on Blackwell
4. **Registers** -- softmax computation, correction factors
5. **Back through shared memory** -- staging for output writeback
6. **Global memory** -- final output destination

### Tensor Memory (New in Blackwell)

Blackwell introduces a new level in the memory hierarchy called Tensor Memory, which sits alongside shared memory in the L1 cache but is specifically managed for Tensor Core intermediates. This reduces register pressure for MMA accumulators.

## Synchronization and Coordination

Warps synchronize via an array of barriers in shared memory referenced by offset. Producer-consumer relationships trigger barrier synchronization at each stage transition.

The architecture uses multiple buffering that increases concurrency and parallelism, allowing up to three concurrent K/V blocks to be in flight simultaneously.

### Barrier-Based Pipeline

```
Load Warp:
  For each K/V block:
    Wait for buffer slot to be released
    Issue TMA load to shared memory
    Signal barrier: data ready

MMA Warp:
  For each K/V block:
    Wait for barrier: data ready
    Execute WGMMA (S = Q * K^T)
    Signal softmax warps: scores ready
    Wait for P from softmax warps
    Execute WGMMA (O += P * V)
    Signal barrier: buffer slot released

Softmax Warps (8):
  For each block:
    Wait for scores from MMA warp
    Compute row_max, exp(S - row_max), row_sum
    Signal correction warps if rescaling needed
    Signal MMA warp: P ready

Correction Warps (4):
  Wait for signal from softmax
  Rescale O by exp(m_old - m_new) only when threshold exceeded
  (Happens ~10x less frequently than naive approach)

Epilogue Warp:
  Wait for final O to be ready
  Issue TMA store to global memory
```

## Hardware Features Leveraged

### Tensor Memory Accelerator (TMA)
- Reduces register pressure by offloading address computation
- Fires off asynchronous copies without warp involvement after issue
- Handles multi-dimensional tensor addressing in hardware

### 5th Generation Tensor Cores (Blackwell)
- Instruction: `tcgen05.mma.cta_group::1`
- Can source inputs from Tensor Memory (not just registers or SMEM)
- Single-CTA variant used for attention workload

### Warpgroups
- Eight-warp alignment used for Softmax (8 warps) and Correction (4 warps) stages
- Potentially improves distribution across the four warp schedulers per SM
- Allows coordinated execution across multiple warps

## Programming Model Insights

The architecture inverts typical CPU asynchronous models: rather than one thread managing a single datum's state transitions through a pipeline, one warp implements a single transition across the pipeline for multiple data elements. This requires manual event loop management.

Each warp role can be thought of as a stage in a dataflow pipeline, with barriers acting as the edges connecting stages. The warp scheduler serves as the runtime that decides which stage to execute next.

## Performance Context

- FA4 targets NVIDIA Blackwell Streaming Multiprocessor architecture
- Achieves ~20% speedup over cuDNN attention kernels
- Eliminates long warp stalls via TMA asynchronicity
- Sophisticated scheduling maintains approximately 15 out of 16 available execution slots filled across four cycles
- Five specialized warp roles ensure all functional units (load/store, Tensor Core, CUDA core, SFU) are continuously utilized

## Comparison with FA3

| Aspect | FA3 (Hopper) | FA4 (Blackwell) |
|---|---|---|
| Warp roles | 2-3 (producer, consumer, dQ-writer) | 5 (load, MMA, softmax, correction, epilogue) |
| Softmax overlap | Pingpong between warpgroups | Dedicated softmax warps |
| Correction | Inline with consumer warpgroup | Separate correction warps, threshold-based |
| Tensor Memory | Not available | Uses Blackwell Tensor Memory |
| Exponential | Hardware SFU | Polynomial approximation (small hdim) |
| Buffering | s-stage circular SMEM buffer | Triple-buffered K/V blocks |
