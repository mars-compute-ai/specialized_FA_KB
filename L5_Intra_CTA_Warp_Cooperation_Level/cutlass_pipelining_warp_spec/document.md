# CUTLASS Tutorial: Efficient GEMM Kernel Designs with Pipelining

**Source:** https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/
**Publisher:** Colfax Research

## Overview

This tutorial discusses techniques for optimizing GEMM kernels on NVIDIA Hopper GPUs by implementing pipelining strategies that overlap data transfer with computation. It covers the two primary approaches -- warp-specialization and multistage -- and details the CUTLASS Pipeline abstraction for barrier synchronization.

## The "Feeding the Beast" Problem

The fundamental challenge in GEMM optimization stems from asymmetric hardware capabilities: the H200 SXM GPU's Tensor Cores can deliver up to 3,958 TFLOPS, but memory bandwidth of the same H200 SXM GPU is only 4.8 TB/s. This massive disparity necessitates overlapping copy operations with math to maintain Tensor Core utilization.

## Two Primary Pipelining Strategies

### 1. Warp-Specialization

Partitions warps into dedicated producers (data transfer) and consumers (compute), running concurrently. This approach is particularly effective on Hopper due to new features like TMA async copy and warpgroup-wide register reallocation (`setmaxnreg`).

**Key characteristics:**
- Producer warps only issue TMA loads
- Consumer warps only execute WGMMA
- Different register budgets via `setmaxnreg`
- Hardware warp scheduler handles interleaving

### 2. Multistage

Uses asynchronous copy instructions (TMA on Hopper, `cp.async` on Ampere) to load subsequent data batches while computing on current buffers. All warps perform both producer and consumer roles.

**Key characteristics:**
- Simpler programming model (no role differentiation)
- Works on Ampere and Hopper
- All warps execute both load and compute code paths

## Buffer Management

### Double Buffering
- Reserves twice the necessary SMEM for two alternating buffers
- Load operations target one buffer while compute proceeds on the other
- Generalizes to N-stage circular buffers for improved overlap

### N-Stage Circular Buffer
- Global matrix tiles increment continuously
- SMEM stages alternate via modular indexing (stage = iteration % N)
- More stages increase pipeline depth and hide more latency
- Trade-off: more stages require more SMEM, reducing occupancy

## CUTLASS Pipeline Abstraction

### Barrier Synchronization

The Pipeline class manages synchronization between producers and consumers using two arrays of N barrier objects:

**Full Barrier Array** (N entries): Signals that a buffer stage contains valid data
**Empty Barrier Array** (N entries): Signals that a buffer stage has been consumed and is available for reuse

Each barrier maintains:
- A **phase bit** (0 or 1) for distinguishing consecutive passes through the circular buffer
- An **arrival count** for threshold-based phase flipping
- Visibility across all threads via SMEM mbarrier objects

### Thread-Local Pipeline State

The `PipelineState` class tracks per-thread execution:
- **Index**: position modulo N (number of stages)
- **Phase**: binary flag (0 or 1)
- Increments index modulo N with phase flipping at wraparound

### Four Core Pipeline Methods

1. **producer_acquire()**: Blocks until `empty_barrier[index]` phase matches current phase (buffer slot is free)
2. **producer_commit()**: Signals `full_barrier[index]` via arrival count increment (data is ready). For TMA operations, this is a no-op since TMA itself handles barrier signaling.
3. **consumer_wait()**: Blocks until `full_barrier[index]` phase matches current phase (data is available)
4. **consumer_release()**: Signals `empty_barrier[index]` via arrival count increment (buffer can be reused)

These methods implement symmetric acquire/release semantics.

## Execution Flow

```
PRODUCER (Load Warp):
  for each tile:
    producer_acquire(state)    // Wait for empty buffer slot
    TMA_load(smem[state.index]) // Async copy GMEM -> SMEM
    producer_commit(state)     // Signal data ready (or TMA auto-signals)
    state.advance()            // Move to next stage

CONSUMER (Compute Warp):
  for each tile:
    consumer_wait(state)       // Wait for data in buffer
    WGMMA(smem[state.index])   // Execute matrix multiply
    consumer_release(state)    // Signal buffer consumed
    state.advance()            // Move to next stage
```

## Synchronization Mechanics

- Barriers use phase bits to distinguish between consecutive passes through the circular buffer
- Prevents stale data reads (consumer reading old data) and overwrites (producer overwriting unconsumed data)
- Threads collectively manage arrival counts or elect representatives per warp
- Phase flipping coordinates when buffers transition between "full" and "empty" states
- TMA can directly signal mbarrier upon completion, eliminating explicit producer_commit

## Performance Results

Pipelining alone achieves approximately 65% utilization on Hopper GEMM kernels in half-precision. Combined with warp specialization and `setmaxnreg`, the fastest CUTLASS GEMM implementations approach 80%+ utilization.

## Key Hopper Advantages for Warp-Specialization

1. **TMA async copy**: Dedicated hardware for GMEM-SMEM transfers, freeing warps after issue
2. **`setmaxnreg`**: Dynamic register reallocation between warpgroups at kernel entry
3. **WGMMA async**: Tensor Core operations execute asynchronously with fence/commit/wait model
4. **mbarrier**: Hardware-supported barrier objects in shared memory for efficient synchronization

These features make warp-specialization the preferred approach on Hopper (over multistage), enabling the fastest CUTLASS GEMM implementations.
