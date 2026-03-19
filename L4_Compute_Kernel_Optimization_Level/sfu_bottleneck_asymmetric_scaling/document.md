# FlashAttention-4: SFU Bottleneck Analysis & Asymmetric Hardware Scaling

Source: https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/

## Overview

This document analyzes the fundamental hardware bottleneck driving Flash Attention 4's instruction-level optimizations: from Hopper H100 to Blackwell B200, BF16 tensor core throughput increases from 1 to 2.25 PFLOPs, while SFU count and shared memory bandwidth remain unchanged. This asymmetric scaling makes the exponential operation in softmax a first-class performance bottleneck, requiring instruction-mix co-design between tensor core and SFU workloads.

## Asymmetric Hardware Scaling: Hopper to Blackwell

| Resource | Hopper H100 | Blackwell B200 | Scaling Factor |
|----------|-------------|----------------|---------------|
| BF16 Tensor Core throughput | 1 PFLOP | 2.25 PFLOPs | **2.25x** |
| SFU count per SM | Unchanged | Unchanged | **1.0x** |
| Shared memory bandwidth | Unchanged | Unchanged | **1.0x** |

This creates a severe imbalance: tensor cores can process data 2.25x faster, but the exponential units needed for softmax cannot keep up.

## Per-SM Blackwell Resource Analysis (M=N=D=128)

### Throughput per Cycle per SM

| Resource | Ops/Cycle |
|----------|-----------|
| Tensor Cores (BF16) | 8,192 |
| Exponential unit (SFU) | 16 |
| Shared Memory | 128 bytes/cycle |

The exponential unit operates at approximately **0.2% of tensor core throughput** -- a 512:1 imbalance.

## Forward Pass Bottleneck Analysis

### Cycles per Tile (Forward Pass)

| Operation | Cycles Required |
|-----------|-----------------|
| Tensor Cores (MMA) | 1,024 |
| Exponential (SFU) | 1,024 |
| Shared Memory | 768 |

The forward pass reaches **parity between tensor cores and exponential throughput** -- both require 1,024 cycles per tile. This means:
- The softmax exponential is an equal bottleneck with the tensor core computation
- Any further increase in tensor core throughput (future architectures) will make exponential the sole bottleneck
- Optimizing the exponential pathway provides direct performance improvement

## Backward Pass Analysis

### Cycles per Tile (1-CTA Backward)

| Operation | Cycles Required |
|-----------|-----------------|
| Tensor Cores (MMA) | 2,560 |
| Exponential (SFU) | 1,024 |
| Shared Memory | 3,328 |

The backward pass is **shared-memory limited** at 3,328 cycles. The 2-CTA approach roughly halves operand B traffic, partially relieving this constraint.

## Approximate Exponentiation: Bypassing the SFU

### Range Reduction (Cody-Waite Technique)

Decompose `2^x = 2^n * 2^f` where:
- `n = floor(x)` (integer part)
- `f = x - n` (fractional part, 0 <= f < 1)

The integer part `2^n` is computed via bit manipulation of the IEEE 754 exponent field -- zero cost.

### Polynomial Approximation (Horner's Method)

```
2^f ≈ p0 + p1*f + p2*f^2 + p3*f^3
```

Coefficients optimized via Sollya:
- p0 = 1.0
- p1 ≈ 0.6951 (0.69514614)
- p2 ≈ 0.2276 (0.22756439)
- p3 ≈ 0.0771 (0.07711909)

### Horner's Evaluation (3 FMA Instructions)

```
t1 = fma(p3, f, p2)     // 0.0771 * f + 0.2276
t2 = fma(t1, f, p1)     // t1 * f + 0.6951
t3 = fma(t2, f, p0)     // t2 * f + 1.0
```

### Exponent Reconstruction

The mantissa bits of `2^f` (from the polynomial) and the shifted integer `n` combine through bit manipulation:

```
// Instead of: result = ldexp(t3, n)  // floating-point multiply
// Use: result.exponent_bits += n     // integer addition on exponent field
```

This exploits the IEEE 754 floating-point representation directly, avoiding a multiply.

### Concurrent Execution Strategy

The key innovation is distributing exponential work across **both** SFU and FMA units:
- Some iterations use hardware `MUFU.EX2` (SFU)
- Some iterations use polynomial approximation (FMA pipeline)
- The ratio is tunable and architecture-dependent
- This effectively increases exponential throughput by using otherwise idle FMA resources

## Forward Pass Pipeline Co-Design

### Ping-Pong Schedule

Two 128-token query tiles (`Q_H`, `Q_L`) per CTA alternate execution, enabling double buffering of intermediate results.

### Synchronized Softmax Warpgroups

Two 128-thread warpgroups execute softmax **sequentially** rather than concurrently:
- Explicit synchronization prevents simultaneous exponential evaluation
- This prevents SFU contention between warpgroups
- While one warpgroup computes softmax, the other can execute MMA

### Register Pressure Relief

P matrix staging stores intermediate results progressively rather than holding all 128+64 elements simultaneously, reducing peak register usage.

### Conditional Online Rescaling

Dedicated "correction" warpgroup applies rescaling only when max jump exceeds threshold tau:

```
If m_j - m_(j-1) > tau:
    // Full correction path
    O_j = exp(m_(j-1) - m_j) * O_(j-1) + exp(S_j - m_j) * V_j
Else:
    // Skip correction (use stale maximum)
    O_j = O_(j-1) + exp(S_j - m_(j-1)) * V_j
```

Final normalization using true statistics preserves output correctness while eliminating redundant rescaling computations.

## Backward Pass: MMA Overlap Strategy

### Five MMAs with Tensor Memory Accumulators

The backward pass manages five concurrent MMA operations:

- **Tile organization**: Recomputes S and P transposed relative to forward, enabling direct storage in tensor memory in operand layout
- **TMEM reuse**: S and P share one column set; dP, dS, and dQ share another
- **MMA overlap**: While computing softmax for tile j, the kernel issues dK and dQ MMAs for tile j-1, hiding exponential latency within MMA execution

### 2-CTA MMA Mode Optimization

- Partitions output accumulators across CTA pairs with M=256, N=K=128
- Reduces shared memory operand B traffic by approximately 50%
- Uses distributed shared memory (DSMEM) exchange within clusters for dQ reduction

## Performance Results

### Forward Pass (Blackwell B200, BF16)

- **Peak throughput**: 1,605 TFLOPs/s (**71% hardware utilization**)
- vs. cuDNN 9.13: **1.1-1.3x faster**
- vs. Triton: **2.1-2.7x faster**

### Backward Pass

- Consistently outperforms baselines for large sequence lengths
- Deterministic mode maintains 85-90% of nondeterministic throughput
- Uses semaphore-serialized global reductions and CTA swizzling for load balancing

## Key Insight

When tensor cores advance 2.25x but exponential units remain static, instruction-mix optimization becomes essential. By overlapping polynomial approximations on FMA units with hardware exponentials, remapping computations to tensor memory, and pipelining across heterogeneous resources, FlashAttention-4 achieves near-peak utilization despite fundamental resource imbalances.
