# FlashAttention: Tiling and Recomputation for IO-Aware Attention

## Overview

FlashAttention is an IO-aware exact attention algorithm that restructures the standard attention computation to minimize data movement between GPU high-bandwidth memory (HBM) and on-chip SRAM. It achieves this through two key techniques: **tiling** (computing attention in blocks that fit in SRAM) and **recomputation** (recomputing intermediate values during the backward pass instead of storing them in HBM).

## The Problem: Standard Attention Is Memory-Bound

Standard self-attention computes:
```
S = Q @ K^T        # N×N score matrix
P = softmax(S)     # N×N attention weights
O = P @ V          # N×d output
```

This naive approach:
- **Materializes the full N×N matrices** (S and P) in HBM
- Requires **Θ(Nd + N²) HBM accesses** — dominated by the quadratic N² term
- Is **memory-bound**, not compute-bound: the time is dominated by HBM access, not arithmetic
- For sequence length N=4096, d=128: the N×N matrices create **33x more memory traffic** than necessary; 97% of memory traffic comes from O(N²) intermediates

### Memory-Bound vs Compute-Bound

The key metric is **arithmetic intensity** (operations per byte accessed):
- **Memory-bound operations**: softmax, dropout, layer norm — time dominated by HBM access
- **Compute-bound operations**: large matrix multiplications — time dominated by arithmetic
- Standard attention contains many elementwise operations on N×N matrices, making it memory-bound overall

### Hardware Context (A100)

- SRAM: 192 KB per streaming multiprocessor (20 MB total), 19 TB/s bandwidth
- HBM: 40-80 GB total, 1.5-2 TB/s bandwidth
- Compute: 312 TFLOPS (FP16)
- The SRAM-to-HBM bandwidth gap is ~10x, meaning data movement is the bottleneck

## Solution: Tiling

FlashAttention divides Q, K, V matrices into smaller blocks and processes them incrementally:

```
# FlashAttention Forward Pass (simplified)
for each block Q_i of Q:                    # Outer loop over Q blocks
    Load Q_i from HBM to SRAM
    for each block K_j, V_j of K, V:        # Inner loop over K,V blocks
        Load K_j, V_j from HBM to SRAM

        # All computation happens in SRAM:
        S_ij = Q_i @ K_j^T                  # Block scores

        # Online softmax update:
        m_new = max(m_old, rowmax(S_ij))     # Running maximum
        P_ij = exp(S_ij - m_new)             # Numerically stable exp
        l_new = exp(m_old - m_new) * l_old + rowsum(P_ij)  # Running sum

        # Accumulate output with rescaling:
        O_i = diag(exp(m_old - m_new)) * O_i + P_ij @ V_j

        m_old = m_new
        l_old = l_new

    O_i = diag(1/l_new) * O_i               # Final normalization
    Write O_i to HBM
    Store (m, l) for backward pass           # Only store statistics, not N×N matrices
```

### Key Properties of Tiling
- **Block sizes** are chosen so that Q_i, K_j, V_j, S_ij all fit simultaneously in SRAM
- The **full N×N attention matrix is never materialized** in HBM
- All matmuls and softmax happen in fast SRAM
- Only the final output O and small statistics (m, l) are written back to HBM

## Solution: Online Softmax

The critical challenge is computing softmax over the full row of S while only seeing one block at a time. FlashAttention uses **online softmax**:

1. Maintain running statistics: **m** (row maximum) and **l** (sum of exponentials)
2. When processing a new block, update these statistics incrementally
3. Rescale previous partial results using the correction factor exp(m_old - m_new)
4. After processing all blocks, the result is **mathematically identical** to standard softmax

This is exact — not an approximation.

## Solution: Recomputation (Backward Pass)

Instead of storing the N×N intermediate matrices S and P for the backward pass:

1. **Store only** the output O and the softmax normalization statistics (m, l) — O(N) memory
2. During backward pass, **reload Q, K, V blocks** and recompute S and P on the fly in SRAM
3. This trades extra compute for dramatically less memory and HBM traffic

Why this works: recomputing in fast SRAM is cheaper than reading stored values from slow HBM. The extra FLOPs are "free" because the kernel was memory-bound, not compute-bound.

## Complexity Analysis

| Metric | Standard Attention | FlashAttention |
|--------|-------------------|----------------|
| HBM Accesses | Θ(Nd + N²) | Θ(N²d²M⁻¹) |
| Memory | O(N²) | O(N) |
| FLOPs | O(N²d) | O(N²d) — same |

Where M is SRAM size (~100 KB). For typical d=64-128 and M≈100KB:
- d² ≈ 4096-16384, which is much smaller than M ≈ 100,000
- So FlashAttention requires **many times fewer HBM accesses**
- The HBM access reduction is roughly **33x** for typical configurations (N=4096, d=128)

## Performance Results

- **Speed**: Up to **4x speedup** over standard attention
- **FlashAttention-2**: Achieves **70% of theoretical maximum FLOPS** on A100
- **FlashAttention-3**: Reaches **75% utilization** of H100's theoretical maximum
- **Memory**: Up to **20x more memory-efficient** than exact attention baselines
- **BERT-large** (seq 512): 15% end-to-end wall-clock speedup
- **GPT-2** (seq 1K): 3x speedup vs HuggingFace implementation
- **Long sequences**: Enables 16K-64K token contexts; 0.7 perplexity improvement on GPT-2

## Block-Sparse Extension

FlashAttention also supports block-sparse attention patterns, further reducing HBM accesses to Θ(Nd + N²d²M⁻¹s) where s is the sparsity fraction. This enables faster approximate attention that still leverages the tiling infrastructure.

## Sources

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (arXiv)](https://arxiv.org/abs/2205.14135)
- [Flash Attention: Transformer Optimization (DeepFA)](https://deepfa.ir/en/blog/flash-attention-transformer-optimization)
- [Paper Summary: FlashAttention (Shreyansh Singh)](https://shreyansh26.github.io/post/2023-03-26_flash-attention/)
- [Attention Optimizations: From Standard to FlashAttention (HuggingFace)](https://huggingface.co/blog/atharv6f/flash-attention-overview)
