---
skill_name: FlashAttention IO-Aware Tiled Attention
description: IO-aware tiled attention algorithm that fuses QK^T, softmax, and PV into one kernel using on-chip tiling and online softmax to reduce memory from O(N^2) to O(N).
level: L1 - Algorithmic/Mathematical Stage Level
target_hardware: NVIDIA GPUs (Ampere A100 and newer)
relevance: When designing or optimizing attention kernels that must handle long sequences without materializing the full N x N attention matrix in HBM.
---

# FlashAttention IO-Aware Tiled Attention

## What It Is
FlashAttention is an IO-aware exact attention algorithm that avoids materializing the N x N score and attention matrices in GPU HBM. It tiles the Q, K, V matrices into blocks that fit in fast on-chip SRAM, computes partial QK^T products, maintains running online softmax statistics (max and denominator), and accumulates the output P@V -- all within a single fused CUDA kernel. During the backward pass, it recomputes logits from Q/K tiles rather than storing them, trading cheap SRAM compute for expensive HBM I/O.

## Key Concepts
- **Tiling:** Q, K, V are divided into blocks of size B_r x d and B_c x d that fit in SRAM (~192 KB). Only 3 tiles reside on-chip at any time.
- **Online Softmax:** A surrogate denominator d'_i = d'_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i) removes the dependency on the global max m_N, enabling single-pass streaming.
- **Output Accumulation with Rescaling:** The partial output o'_i is updated as o'_{i-1} * correction_factor + new_contribution, where the correction factor adjusts for the evolving max.
- **Recomputation in Backward Pass:** Instead of storing the S = QK^T matrix (O(N^2) memory), logits are recomputed on-the-fly from Q and K tiles in SRAM during gradient calculation.
- **Fused Kernel:** The entire forward attention (QK^T -> softmax -> PV) runs in one kernel launch, eliminating intermediate HBM reads/writes.
- **Memory Complexity:** Reduced from O(N^2) to O(N) -- the working set depends only on block size B and head dimension d, not sequence length N.

## Algorithm / Pseudo-code
```
# FlashAttention Forward Pass (per row k of output)
# Inputs: Q[k,:] (1 x d), K (N x d), V (N x d)
# Block size: b

Initialize: m = -inf, d' = 0, o' = 0 (1 x d vector)

for tile_i = 1 to N/b:
    # Step 1: Load tile and compute partial scores
    x_tile = Q[k,:] @ K[(i-1)*b : i*b, :]^T    # 1 x b scores

    # Step 2: Update running max
    m_local = max(x_tile)
    m_new = max(m, m_local)

    # Step 3: Update running denominator (online softmax)
    d' = d' * exp(m - m_new) + sum(exp(x_tile - m_new))

    # Step 4: Rescale previous output and accumulate
    o' = o' * (d'_old / d') * exp(m - m_new)
       + sum_j( exp(x_tile[j] - m_new) / d' * V[j + (i-1)*b, :] )

    m = m_new

# Final output
O[k,:] = o'

# Backward Pass: Recompute S = QK^T tiles on-the-fly
# instead of loading from HBM. Compute dQ, dK, dV
# using the recomputed P and stored logsumexp L.
```

## When to Use
- Training or inference with sequence lengths >= 1K where the N x N attention matrix would be memory-prohibitive
- Any model using standard multi-head attention (GPT, BERT, ViT, etc.)
- When you need exact (not approximate) attention output
- Memory-bound scenarios where HBM bandwidth is the bottleneck
- When extending context length without increasing GPU memory

## When NOT to Use
- Very short sequences (< 256 tokens) where the overhead of tiling may not pay off
- When approximate attention (sparse, linear) is acceptable and faster for the use case
- Non-NVIDIA hardware without efficient SRAM or tensor core equivalents (needs porting)
- When the attention pattern is known to be very sparse and a sparse attention method gives better asymptotic complexity

## Key Takeaways
- FlashAttention is mathematically exact -- it produces bit-identical results to standard attention (within floating-point precision)
- The core trick is the online softmax recurrence that makes the softmax operation associative/tileable, enabling single-pass streaming through K/V blocks
- Memory scales as O(N) not O(N^2): working set = O(B * d) on SRAM, only Q, K, V, O stored in HBM
- Recomputation in backward pass is net positive because SRAM compute is ~100x cheaper per byte than HBM I/O
- FlashAttention-1 achieves 2-4x wall-clock speedup over standard attention and is the foundation for all subsequent Flash Attention versions

## References
- Original Paper: Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)
- Blog: https://deepfa.ir/en/blog/flash-attention-transformer-optimization
- Code: https://github.com/Dao-AILab/flash-attention
