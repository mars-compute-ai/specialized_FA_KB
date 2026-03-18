# Tuning Flash Attention for Peak Performance in NVIDIA CUDA Tile

## Overview

This document covers the practical optimization journey of tuning Flash Attention tile sizes, math precision, and algorithmic strategies to achieve peak performance on NVIDIA GPUs (including B200/Blackwell). It reveals critical pitfalls (the "large tile trap") and the interdependent optimization stack that yields up to 1.66x speedup over baseline implementations.

## The Scale of the Problem

For sequence length N=16,384 tokens, the intermediate attention matrix QK^T contains:
- **268 million elements** — 512 MB in FP16 per head
- Standard implementations materialize this full matrix in global memory
- Flash Attention tiles the computation so the full N×N matrix is never materialized

## The Online Softmax Algorithm

The key algorithmic enabler for tiling — incremental softmax without storing full rows:

```
Maintain per-tile running statistics:
  m_i: running maximum (for numerical stability)
  l_i: running sum of exponentials (softmax denominator)

For each new K-tile:
  1. m_new = max(m_i, max(scores_new))       # Update maximum
  2. alpha = exp(m_i - m_new)                 # Correction factor
  3. l_i = l_i * alpha + sum(exp(scores_new - m_new))  # Update sum
  4. acc = acc * alpha                        # Rescale accumulator
  5. acc += softmax(scores_new) @ V_tile      # Accumulate output
```

This yields **exact softmax** without ever storing the full row of scores.

## Implementation Structure

### Kernel Organization
- Each CUDA block computes one output tile
- The query tile loads once and is reused across all K/V iterations
- Matrix multiply uses `ct.mma()` for Tensor Core acceleration
- Boundary conditions handled automatically by the framework

### Grouped-Query Attention (GQA)
Multiple query heads share K/V heads:
```
head_idx = bid_y % H
kv_head_idx = head_idx // QUERY_GROUP_SIZE
```
With 32 query heads and 4 K/V heads, this reduces K/V cache by 8x — critical for long-context inference (Llama 2, Mistral).

## The "Trap and Rescue" — Tile Size Optimization

### The Trap: Large Tiles (256x128) Hurt Performance

Initial intuition: larger tiles should amortize memory overhead and improve data reuse.

**Reality**: Increasing from 64x64 to 256x128 tiles **degraded performance by 18-43%** across sequence lengths.

Root causes:
1. **Inefficient special functions**: `exp2()` and division become bottlenecks with more elements per tile
2. **Register explosion**: Usage jumped 31% (168 vs 128 registers per thread)
3. **Occupancy collapse**: Only 18.75% occupancy with large tiles
4. **Instruction overhead**: More instructions before the next memory operation

### The Rescue: Fast Math Flags

Enabling math efficiency flags transformed the results:
- **`flush_to_zero=True`**: Denormal numbers treated as zero, avoiding slow microcode paths
- **`rounding_mode=APPROX`**: Skips iterative refinement after initial hardware approximation

Impact: Performance recovered to baseline and **exceeded it by 10-20%** for longer sequences.

**Key lesson**: Tile size cannot be evaluated in isolation from mathematical precision settings. These optimizations are interdependent.

## The Full Optimization Stack

| # | Technique | Impact | Cumulative Speedup |
|---|-----------|--------|-------------------|
| 1 | Large tiles (trap) | -18 to -43% | 0.57-0.82x |
| 2 | Fast math (rescue) | +34 to +72% | 0.91-1.05x |
| 3 | K-loop splitting | +16 to +32% | 1.15-1.31x |
| 4 | Block remapping | +1 to +2.6% | 1.16-1.34x |
| 5 | Autotuning | +10 to +45% | 1.60-1.66x |

### K-Loop Splitting (Largest Single Optimization)

For **causal attention**, roughly half the attention matrix is masked (upper triangle):

```
Full attention matrix (N×N):
┌─────────────────────┐
│ ████████            │  ← Unmasked (compute)
│ ████████████        │
│ ████████████████    │
│ ████████████████████│
└─────────────────────┘

K-loop splitting:
  - Skip fully-masked tiles entirely (no compute needed)
  - Minimize masking operations for partially-masked tiles
  - Full tiles computed without any masking overhead
```

The causal mask adds -inf to masked positions, ensuring they become zero after softmax. K-loop splitting avoids unnecessary computation for the ~50% of tiles that are fully masked.

### Block Remapping

Reorders block scheduling for better L2 cache utilization:
- Adjacent blocks access nearby K/V memory regions
- Improved L2 hit rate reduces HBM traffic
- Modest but consistent 1-2.6% improvement

### Autotuning

Rather than fixed tile sizes, the autotuner benchmarks configurations per input shape:

| Sequence Length | Optimal Tile Size |
|-----------------|-------------------|
| <= 2,048 | 64×64 (maximum parallelism) |
| 4,096-16,384+ | 128×128 or 256×128 (memory efficiency) |

The autotuner discovers optimal configurations automatically, delivering **10-45% gains** over fixed configurations.

## Performance Results (B200 GPU)

| Sequence Length | TFLOPS | Speedup vs Baseline |
|-----------------|--------|---------------------|
| 1,024 | 548 | 1.66x |
| 2,048 | 708 | 1.61x |
| 4,096 | 817 | 1.60x |
| 8,192 | 887 | 1.62x |
| 16,384 | 918 | 1.62x |

Performance scales well with sequence length, approaching **918 TFLOPS** at N=16,384.

## Key Insights

1. **Optimizations are interdependent**: Tile size, math precision, and algorithmic strategies interact non-linearly; evaluating them independently gives wrong conclusions
2. **Math matters**: `flush_to_zero` and approximate rounding unlock Tensor Core throughput by avoiding slow special-function paths
3. **Algorithmic improvements compound**: K-loop splitting + fast math + autotuning produce multiplicative (not additive) gains
4. **Larger is not always better**: The large-tile trap demonstrates that register pressure and special-function overhead can overwhelm data-reuse benefits
5. **Autotuning is essential**: Optimal tile sizes vary significantly by sequence length and hardware

## Sources

- [Tuning Flash Attention for Peak Performance in NVIDIA CUDA Tile (NVIDIA Technical Blog)](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/)
