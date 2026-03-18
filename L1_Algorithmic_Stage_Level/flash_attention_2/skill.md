---
skill_name: FlashAttention-2 Parallelism and Work Partitioning
description: Optimized FlashAttention algorithm that reduces non-matmul FLOPs, parallelizes over sequence length, and partitions Q (not K) across warps to eliminate shared-memory synchronization.
level: L1 - Algorithmic/Mathematical Stage Level
target_hardware: NVIDIA GPUs (Ampere A100, Hopper H100)
relevance: When optimizing attention throughput on A100/H100 GPUs, especially for long-context training where batch*heads is small relative to SM count.
---

# FlashAttention-2 Parallelism and Work Partitioning

## What It Is
FlashAttention-2 (Tri Dao, 2023) improves upon FlashAttention-1 in three ways that together yield a 2x speedup: (1) it restructures the online softmax update to minimize non-matmul FLOPs, which are 16x more expensive per operation on modern GPUs; (2) it adds parallelism over the sequence-length dimension for better GPU utilization when batch*heads is small; and (3) it switches from a "Sliced-K" to a "Sliced-Q" warp partitioning scheme, allowing each warp to independently compute its output slice without inter-warp communication.

## Key Concepts
- **Non-Matmul FLOP Reduction:** On A100, matmul throughput is 312 TFLOPs/s (FP16) vs 19.5 TFLOPs/s for non-matmul FP32 -- a 16x gap. FA2 rewrites the softmax rescaling to perform fewer multiply/divide operations per tile, reducing the fraction of time spent on non-matmul work.
- **Sliced-Q Warp Partitioning:** FA1 split K and V across 4 warps (Sliced-K), requiring all warps to write partial results to shared memory, synchronize, and reduce. FA2 instead splits Q across 4 warps while K and V are shared. Each warp computes its own slice of QK^T, applies softmax, multiplies by V, and writes directly to its output slice -- no cross-warp communication needed.
- **Sequence-Length Parallelism:** FA1 parallelized only over (batch x heads), using one thread block per head. FA2 additionally parallelizes over the Q sequence dimension, assigning different Q tile blocks to different thread blocks. This is critical when batch*heads < #SMs (e.g., long sequences with few heads).
- **Inner Loop over K/V:** FA2's forward pass iterates over K/V blocks in the inner loop (not Q blocks as conceptually in FA1), which improves data reuse and cache behavior.
- **Head Dimension Support:** Extended from 128 to 256, supporting models like GPT-J and StableDiffusion.
- **MQA/GQA Support:** Multiple query heads can attend to the same K/V head by adjusting tensor indexing without duplicating K/V in HBM.

## Algorithm / Pseudo-code
```
# FlashAttention-2 Forward Pass
# Outer loop: parallelize over (batch, heads, Q_tile_index)
# Inner loop: stream over K/V tiles

# Per thread block: handles one Q tile Q_i (size B_r x d)
# Within thread block: 4 warps, each handles Q_i[warp_slice] (size B_r/4 x d)

for each Q tile Q_i (parallelized across thread blocks):
    Load Q_i to SRAM
    Initialize: O_i = 0, l_i = 0, m_i = -inf  (per-row statistics)

    for each KV tile pair (K_j, V_j):  # inner loop
        Load K_j, V_j to SRAM

        # Each warp independently (no sync needed):
        S_warp = Q_i[warp_slice] @ K_j^T     # partial scores for warp's Q rows
        m_new = max(m_i[warp_rows], rowmax(S_warp))

        # Reduced rescaling (FA2 improvement):
        # Only rescale when m changes, fewer operations than FA1
        P_warp = exp(S_warp - m_new)
        l_i[warp_rows] = exp(m_i[warp_rows] - m_new) * l_i[warp_rows] + rowsum(P_warp)

        # Accumulate output (each warp writes independently)
        O_i[warp_rows] = diag(exp(m_old - m_new)) * O_i[warp_rows] + P_warp @ V_j

        m_i[warp_rows] = m_new

    # Final normalization
    O_i = diag(1/l_i) * O_i
    Write O_i to HBM

# Key difference from FA1:
# FA1 Sliced-K: warps split K -> each produces partial O -> sync + reduce in SMEM
# FA2 Sliced-Q: warps split Q -> each produces final O slice -> NO sync needed
```

## When to Use
- Training or inference on A100 or H100 GPUs with FP16/BF16
- Long-context scenarios (4K-128K tokens) where sequence-length parallelism matters
- Models using MQA or GQA (e.g., Llama 2, Mistral) where multiple Q heads share K/V
- When batch*heads < 80 (number of A100 SMs) and you need additional parallelism
- Head dimensions up to 256

## When NOT to Use
- On Hopper H100 GPUs where FlashAttention-3 provides additional 1.5-2x gains via warp specialization and GEMM-softmax overlapping
- On non-NVIDIA hardware (AMD, TPU) -- would need architecture-specific reimplementation
- For very short sequences (< 512) where the overhead may not be worthwhile
- When approximate attention is acceptable and provides better asymptotic scaling

## Key Takeaways
- The 16x throughput gap between matmul and non-matmul FLOPs on modern GPUs means minimizing rescaling operations in the softmax loop has outsized impact on performance
- Sliced-Q partitioning is strictly better than Sliced-K: it eliminates shared-memory writes, barrier synchronization, and reduction -- the warp-level computation is fully independent
- Sequence-length parallelism is essential for the long-context regime (large N, small batch) where batch*heads alone cannot saturate all SMs
- FA2 achieves 230 TFLOPs/s on A100 (72% of peak), up from 124 TFLOPs/s for FA1 (40% of peak)
- End-to-end, FA2 delivers 1.3x training speedup over FA1 at 8K context, equivalent to training with 16K context at the cost of 8K

## References
- Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," 2023: https://arxiv.org/abs/2307.08691
- Blog: https://hazyresearch.stanford.edu/blog/2023-07-17-flash2
- Code: https://github.com/Dao-AILab/flash-attention
