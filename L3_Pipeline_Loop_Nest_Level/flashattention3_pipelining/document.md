# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

**Source**: https://arxiv.org/html/2407.08608v2
**Authors**: Tri Dao, Jay Shah

## Overview

FlashAttention-3 introduces three major optimizations for attention computation on NVIDIA Hopper GPUs:

1. **Producer-Consumer Asynchrony** through warp-specialization
2. **GEMM-Softmax Pipelining** to overlap computation
3. **Hardware-Accelerated Low-Precision (FP8)** support

## Warp-Specialization and Producer-Consumer Architecture

### Core Concept

The algorithm divides thread warps into distinct roles:

- **Producer warps**: Handle data movement via Tensor Memory Accelerator (TMA)
- **Consumer warps**: Execute matrix multiplications using asynchronous WGMMA instructions

This separation enables the compiler to generate optimal instruction schedules and allows register reallocation between warpgroups via `setmaxnreg`.

### Implementation Details

The producer warpgroup manages:
- Loading query blocks (Q_i) from global memory to shared memory
- Iteratively loading key and value blocks (K_j, V_j) using a circular shared memory buffer with s-stages

The consumer warpgroup executes:
- Score computation: S_i^(j) = Q_i * K_j^T (using WGMMA)
- Softmax operations on score blocks
- Output accumulation: O_i = P_tilde_i^(j) * V_j (using WGMMA)

Synchronization through barrier instructions prevents the producer from overwriting data still being consumed.

## GEMM-Softmax Pipelining

### Motivation

The performance bottleneck stems from throughput disparity:

- **FP16 matmul throughput**: 989 TFLOPS
- **Exponential (softmax) throughput**: 3.9 TFLOPS

This 250x difference means softmax operations can consume ~50% of execution cycles. Overlapping these with matmuls improves utilization.

### 2-Stage Pipeline Architecture

The algorithm maintains two "in-flight" blocks:

**Stage 1**: While computing K_j^(T-1) matmul, execute softmax operations on the previous iteration's scores.

**Stage 2**: While executing softmax, issue the next WGMMA for K_j^T computation.

Key modifications to break sequential dependencies:
- Store intermediate softmax results (S_cur, P_tilde_cur) in registers between iterations
- Compute S_next without waiting for P_tilde_cur completion
- Execute P_tilde_cur * V_(j-1) while computing softmax for S_next

### Implementation Challenges

**Compiler Reordering**: The NVCC compiler may rearrange instructions, disrupting carefully crafted pipelining. Analysis of SASS code confirms the compiler generates overlapped instructions as intended.

**Register Pressure**: Additional storage for S_next requires Br x Bc x 4 extra bytes per threadblock, creating trade-offs with block size selection.

### 3-Stage Extension

A more aggressive variant overlaps the second WGMMA with softmax, requiring even more registers. This offers theoretical gains but necessitates careful profiling to balance pipeline depth against tile size.

## Asynchronous Execution Model

### Hardware Features Exploited

**Tensor Memory Accelerator (TMA)**:
- Dedicated hardware for asynchronous GMEM-to-SMEM transfers
- Issues load commands that don't stall on prior operation completion
- Enables producer warps to issue all loads upfront

**Asynchronous WGMMA**:
- Warpgroup-wide matrix multiply that sources inputs from shared memory
- Executes independently of other compute units
- Supports separate register allocation for producer vs. consumer warps

### Execution Flow

```
Producer Loop:
  - Issue Q load (wait for 1st empty slot)
  - Loop K/V loads (with circular buffer wait)

Consumer Loop:
  - Wait for data availability
  - Issue WGMMA for S = Q * K^T (non-blocking)
  - Continue to softmax computation on previous S
  - Issue next WGMMA while softmax executing
  - Issue WGMMA for O += P * V
  - Overlap softmax(S_new) with V matmul
```

## Low-Precision (FP8) Attention

### Layout Conformance Challenges

FP8 WGMMA imposes stricter memory layout requirements than FP16:

- **FP16 WGMMA**: Accepts both column-major and row-major input operands in shared memory
- **FP8 WGMMA**: Requires k-major (row-major) operand format only

**Solution for matrix V**:

Since input tensors are typically head-dimension-contiguous in HBM, an in-kernel transpose is necessary:
- LDSM/STSM instructions transpose V tiles after loading into SMEM
- These instructions operate at 128-byte granularity and avoid register spilling
- Transpose scheduled during shadow of preceding WGMMA operations

### Register Layout Permutation

The FP32 accumulator layout differs from FP8 operand A layout in registers:

- **FP32 accumulator ordering**: {d0, d1, d4, d5, d2, d3, d6, d7}
- **FP8 operand A ordering**: {a0, a1, a2, a3, a4, a5, a6, a7}

**Solution**: Apply byte permutation instructions to transform the score matrix layout into FP8-compatible format. Column permutation is matched with a corresponding row permutation applied during V transpose.

### Accuracy Optimization

**Block Quantization**:
- Maintain separate scaling factors per block (Br x d or Bc x d) rather than per-tensor
- Automatically fused with preceding operations (e.g., rotary embeddings) at no cost
- Score matrix scaling adjusted per-block to account for quantization

**Incoherent Processing**:
- Multiply Q and K by random orthogonal matrix M before quantization
- Since M * M^T = I, final attention output is unchanged: (Q*M)(K*M)^T = Q*K^T
- Spreads outliers across matrix entries, reducing quantization error
- Implementation uses Hadamard matrix product with diagonal +/-1 matrices (O(d log d) complexity)
- Fuses with rotary embeddings

## Performance Results

### FP16 Performance

- **Speedup over FlashAttention-2**: 1.5-2.0x forward pass, 1.5-1.75x backward
- **Peak throughput**: Up to 740 TFLOPS/s (75% of theoretical maximum)
- **Utilization improvement**: From 35% (FlashAttention-2) to 75%

### FP8 Performance

- **Peak throughput**: ~1.2 PFLOPS/s (nearly double FP16)
- **Competitive with cuDNN**: Matches or exceeds vendor implementations for long sequences
- **Accuracy improvement**: 2.6x lower numerical error vs. per-tensor quantization baseline

### Ablation Study Results

Configuration analysis (seq_len=8448, head_dim=128, batch=4):

| Configuration | Throughput |
|---|---|
| Full FlashAttention-3 | 661 TFLOPS/s |
| Without GEMM-Softmax pipelining | 582 TFLOPS/s |
| Without warp-specialization | 570 TFLOPS/s |

GEMM-Softmax pipelining contributes ~14% throughput improvement. Warp-specialization contributes ~16% throughput improvement.

## Key Limitations and Future Work

- **Inference optimization**: Current implementation prioritizes training throughput
- **Persistent kernel for FP8**: FP16 uses persistent kernels with load balancing; FP8 does not yet
- **Scale validation**: Effects of low-precision attention in large-scale training require further study
- **Hardware generalization**: Current focus on Hopper; applicability to other architectures pending

## Technical Implementation Notes

All code uses CUTLASS abstractions (WGMMA, TMA) for portability. The implementation exploits Hopper-specific features including dynamic register reallocation and improved memory hierarchy bandwidth (3.35 TB/s global memory, 31 TB/s shared memory per GPU).
