# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

**Author:** Tri Dao
**Date:** July 17, 2023
**Source:** https://hazyresearch.stanford.edu/blog/2023-07-17-flash2

## Overview

FlashAttention-2 is about 2x faster than FlashAttention, reaching up to 230 TFLOPs/s on A100 GPUs (FP16/BF16). When deployed end-to-end for training GPT-style models, it reaches up to 225 TFLOPs/s (72% model FLOP utilization).

## Context and Motivation

Modern language models require longer context windows. Examples include GPT-4 (32k context), MosaicML's MPT (65k context), and Anthropic's Claude (100k context). The challenge stems from attention mechanisms having runtime and memory requirements that are quadratic in the input sequence length.

The original FlashAttention delivered 2-4x speedups but only achieved 25-40% of the theoretical maximum FLOPs/s (e.g. up to 124 TFLOPs/s on A100 GPU), leaving substantial optimization potential.

## Key Technical Improvements

### 1. Reduced Non-Matmul FLOPs

Modern GPUs possess specialized compute units (Tensor Cores) optimized for matrix multiplication. The disparity is stark: the A100 GPU has a max theoretical throughput of 312 TFLOPs/s of FP16/BF16 matmul, but only 19.5 TFLOPs/s of non-matmul FP32. This creates a **16x cost differential** -- each non-matmul operation is significantly more expensive.

FlashAttention-2 rewrites the online softmax trick to reduce the number of rescaling operations, as well as bound-checking and causal masking operations, without changing the output.

**Key algorithmic change:** In FlashAttention-1, the output accumulator O was maintained in a form that required dividing by the running softmax denominator at each step. FlashAttention-2 restructures this so the rescaling factor is applied less frequently, reducing the number of non-matmul FLOPs.

### 2. Enhanced Parallelism Strategy

The original FlashAttention parallelized across batch size and number of heads, using one thread block per attention head. This works efficiently when (batch_size * number_of_heads) is large (>= 80 streaming multiprocessors).

For long sequences with small batch sizes or few heads, FlashAttention-2 introduces additional parallelization over the **sequence length dimension**, enabling better GPU resource utilization.

### 3. Improved Work Partitioning Between Warps

This represents a fundamental architectural change:

**Original approach (Sliced-K scheme):** FlashAttention splits K and V across 4 warps while keeping Q accessible by all warps. However, all warps need to write their intermediate results to shared memory, synchronize, then add up the intermediate results.

**FlashAttention-2 approach (Sliced-Q scheme):** Instead splits Q across 4 warps while keeping K and V accessible by all warps. After each warp computes its portion of QK^T, they multiply with the shared slice of V to get their corresponding slice of the output. There is **no need for communication between warps**.

This redesign eliminates unnecessary synchronization and shared memory communication.

## Algorithm Overview

### Forward Pass (simplified)

```
# FlashAttention-2 Forward Pass
# Q is divided into blocks Q_1, ..., Q_Tr of size B_r x d
# K, V are divided into blocks K_1, ..., K_Tc of size B_c x d

for each Q block Q_i (parallelized across sequence length):
    Initialize O_i = 0, l_i = 0, m_i = -inf

    for each K, V block pair (K_j, V_j):
        # Compute attention scores
        S_ij = Q_i @ K_j^T

        # Update running max (for numerical stability)
        m_i_new = max(m_i, rowmax(S_ij))

        # Compute safe exponentials
        P_ij = exp(S_ij - m_i_new)

        # Update running sum (online softmax denominator)
        l_i = exp(m_i - m_i_new) * l_i + rowsum(P_ij)

        # Rescale previous output and accumulate new contribution
        O_i = diag(exp(m_i - m_i_new)) * O_i + P_ij @ V_j

        m_i = m_i_new

    # Final normalization
    O_i = diag(1/l_i) * O_i
```

Key improvement: The inner loop iterates over K/V blocks (not Q blocks as in FA1), and Q is partitioned across warps within a thread block, eliminating cross-warp synchronization.

## Expanded Feature Support

- **Head dimensions:** Extended from up to 128 to up to **256**, enabling support for GPT-J, CodeGen, CodeGen2, and StableDiffusion 1.x
- **Attention variants:** Supports multi-query attention (MQA) and grouped-query attention (GQA), which allow multiple heads of query to attend to the same head of key and value

## Performance Benchmarks

### A100 80GB SXM4 GPU Results
- Around 2x faster than FlashAttention (as well as xformers and Triton implementations)
- Up to 9x faster than standard PyTorch implementations

### H100 SXM5 GPU Results
- Up to 335 TFLOPs/s (without special Hopper optimization)

### End-to-End Training Results

| Model | Baseline | FlashAttention | FlashAttention-2 |
|-------|----------|----------------|------------------|
| GPT3-1.3B, 2K context | 142 TFLOPs/s | 189 TFLOPs/s | 196 TFLOPs/s |
| GPT3-1.3B, 8K context | 72 TFLOPs/s | 170 TFLOPs/s | 220 TFLOPs/s |
| GPT3-2.7B, 2K context | 149 TFLOPs/s | 189 TFLOPs/s | 205 TFLOPs/s |
| GPT3-2.7B, 8K context | 80 TFLOPs/s | 175 TFLOPs/s | 225 TFLOPs/s |

The 8K context improvements are particularly significant, representing a 1.3x end-to-end speedup over an already very optimized model with FlashAttention.

## Practical Implications

The 2x speedup means we can train models with 16k longer context for the same price as previously training an 8k context model. This enables applications in long document analysis, high-resolution image processing, and temporal data like audio and video.

## References

- Repository: https://github.com/Dao-AILab/flash-attention
- Paper: https://arxiv.org/abs/2307.08691
- Multi-Query Attention: https://arxiv.org/abs/1911.02150
- Grouped-Query Attention: https://arxiv.org/abs/2305.13245
