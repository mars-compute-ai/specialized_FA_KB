# Understanding Flash Attention: Writing the Algorithm from Scratch in Triton

**Author:** Alex Dremov
**Source:** https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/

## Overview

This tutorial explains Flash Attention's efficiency improvements and demonstrates implementation using Triton. The core innovation addresses GPU memory bottlenecks through I/O-aware computation.

## What is Attention?

The scaled dot-product attention formula:

**Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V**

Naive implementations require O(N^2) memory and O(N^2) compute time. Flash Attention reduces memory to O(N) by never materializing the full attention scores matrix.

## Flash Attention Core Principles

### GPU Memory Hierarchy
- **SRAM**: Fast, on-chip, limited capacity
- **HBM**: Slower, larger capacity (standard GPU memory)

Flash Attention makes the algorithm IO-aware -- accounting for reads and writes between levels of GPU memory.

### Key Achievement
Avoids explicitly materializing the attention scores matrix. Computes results in tiles while recalculating the matrix during backpropagation if needed.

## Tiled Attention Calculation

The algorithm processes attention sequentially through tiles rather than computing the full matrix simultaneously.

### Online Softmax with Tiling

The challenge: softmax normalization requires aggregation across the entire sequence, but tiled computation prevents accessing complete data at once.

**Solution**: Implement concatenated softmax using recurrence relations. For vectors x^(1) and x^(2), compute softmax over concatenation [x^(1), x^(2)] using:

- m(x) = max(m(x^(1)), m(x^(2)))
- l(x) = e^(m(x^(1)) - m(x)) * l(x^(1)) + e^(m(x^(2)) - m(x)) * l(x^(2))

Where m(x) represents the maximum value and l(x) represents the softmax denominator. This enables computing softmax per-tile, then renormalizing results as new tiles arrive.

## Triton Implementation

### Job Assignment
Each kernel job handles:
- One Q tile (loaded once)
- All K and V tiles (iterated sequentially)
- One output tile (stored after processing)

### Algorithm Pseudocode

```python
def self_attn_fwd(...):
    # Initialize tracking variables
    m_i = zeros([TILE_Q_SIZE]) - inf   # Running QK^T maximum
    l_i = zeros([TILE_Q_SIZE])         # Softmax denominator
    acc = zeros([TILE_Q_SIZE, HEAD_DIM])  # Accumulated output

    # Load Q tile into SRAM
    q_tile = load_q_tile(...)

    # Define softmax scaling
    softmax_scale = SM_SCALE * log2(e)

    # Iterate over K,V tiles
    for kv_tile_idx in range(num_kv_tiles):
        # Load K^T and V tiles
        kt_tile = load_kt_tile(kv_tile_idx)
        v_tile = load_v_tile(kv_tile_idx)

        # Compute QK^T for this tile
        qk = dot(q_tile * softmax_scale, kt_tile)

        # Apply masking for sequence length padding
        qk = where(mask, qk, -inf)

        # Update maximum: m(x) = max(m(x^(1)), max(x^(2)))
        m_ij = maximum(m_i, max(qk, axis=1))

        # Compute softmax probabilities with numerical stability
        p = exp2(qk - m_ij[:, None])

        # Current tile softmax denominator
        l_ij = sum(p, axis=1)

        # Compute alpha for renormalization: e^(m(x^(1)) - m(x))
        alpha = exp2(m_i - m_ij)

        # Update global denominator using recurrence
        l_i = l_i * alpha + l_ij

        # Renormalize accumulator for maximum change
        acc = acc * alpha[:, None]

        # Accumulate attention output: add p @ V
        acc += dot(p, v_tile)

        # Store updated maximum
        m_i = m_ij

    # Normalize final result by softmax denominator
    acc = acc / l_i[:, None]

    # Store result tile
    save(acc, ...)
```

### Implementation Details
- **Numerical Stability**: Compute softmax(x - max(x)) rather than softmax(x)
- **Precision Handling**: Accumulate in float32 for higher precision, convert only for specific ops
- **Masking Strategy**: Set out-of-sequence values to -inf so softmax ignores them
- **exp2 vs exp**: Use exp2 (base-2) which maps directly to hardware instructions

## Performance Analysis

- **Naive Implementation**: Out-of-memory errors at large sequence lengths
- **Flash Attention (Triton)**: Maintains performance across extended sequences
- **PyTorch SDPA**: Slightly faster (well-optimized CUDA kernel) but Triton version is readable and maintainable

## Key Takeaways

1. Flash Attention achieves dramatic speedups through I/O-aware memory management
2. Tiled computation with online softmax eliminates O(N^2) memory materialization
3. Triton provides accessible syntax for implementing complex GPU algorithms
4. The approach scales to longer sequences without memory bottlenecks
