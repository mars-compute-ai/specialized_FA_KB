---
skill_name: Online Softmax Algorithm
description: Single-pass softmax computation using running maxima and sums, enabling tiled attention without materializing the full attention matrix.
level: L4 - Compute Kernel Optimization Level
target_hardware: NVIDIA GPUs (general principle applies to all accelerators)
relevance: When implementing or optimizing FlashAttention kernels, designing tiled attention computations, or reducing memory bandwidth in softmax-heavy workloads.
---

# Online Softmax Algorithm

## What It Is
The online softmax algorithm computes softmax normalization in fewer memory passes than the standard approach by maintaining running statistics (maximum and sum of exponentials) as data is streamed through. The standard "safe" softmax requires three passes over the data (find max, compute exponentials, normalize), while the online version merges the first two into a single pass using a correction factor that rescales accumulated values whenever a new maximum is discovered. FlashAttention extends this to compute the full attention output O = softmax(QK^T)V in a tiled, streaming fashion without ever materializing the N x N attention matrix.

## Key Concepts
- **Standard softmax requires 3 passes**: (1) find max, (2) sum of exp(x_i - max), (3) normalize each element
- **Online softmax merges passes 1 and 2** using a recursive update with correction factor `exp(m_old - m_new)`
- **Running statistics**: `m_i = max(m_{i-1}, x_i)` and `d_i = d_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)`
- **FlashAttention extends this to output accumulation**: the output vector is also rescaled by the correction factor at each step
- **Tiled computation**: Q, K, V are split into blocks; each tile pair is processed in SRAM with O(B*d) memory instead of O(N^2)
- **SRAM footprint is independent of sequence length**: depends only on block size B and head dimension d
- **IO-aware**: minimizes HBM reads/writes by keeping intermediates in fast on-chip memory

## Algorithm / Pseudo-code
```python
# Online Softmax (Two-Pass: merged max+sum, then normalize)
def online_softmax(x):
    m = -inf   # running maximum
    d = 0.0    # running denominator (sum of exponentials)

    # Pass 1: compute max and denominator simultaneously
    for x_i in x:
        m_new = max(m, x_i)
        d = d * exp(m - m_new) + exp(x_i - m_new)  # correction + new term
        m = m_new

    # Pass 2: normalize
    return [exp(x_i - m) / d for x_i in x]


# FlashAttention: Tiled Online Attention (Single Kernel)
# Q, K, V in HBM; O, l, m as running accumulators
for each Q_tile (Br x d):
    m = -inf[Br]     # per-row running max
    l = zeros[Br]    # per-row running sum
    O = zeros[Br, d] # per-row output accumulator

    for each K_tile, V_tile (Bc x d):
        S = Q_tile @ K_tile^T                  # (Br x Bc) in SRAM
        m_new = max(m, rowmax(S))              # update running max
        alpha = exp(m - m_new)                 # correction factor
        P = exp(S - m_new)                     # softmax numerators
        l = alpha * l + rowsum(P)              # update running sum
        O = alpha * O + P @ V_tile             # corrected output accumulation

    O = O / l  # final normalization
    write O back to HBM
```

## Numerical Considerations
- **Subtracting the running max before exp()** prevents overflow; this is why it is called "safe" softmax
- **The correction factor `exp(m_old - m_new)`** is always <= 1.0 (since m_new >= m_old), so it never causes overflow
- **When m_old == m_new** (no new maximum), the correction factor is exactly 1.0 and no rescaling occurs -- this is the common case after the first few tiles
- **Catastrophic cancellation is not a concern** because all terms being summed are positive (exponentials)
- **fp16/bf16 accumulation**: The output accumulator and running statistics should be kept in fp32 to avoid precision loss over many tiles
- **Exact computation**: Online softmax produces mathematically identical results to standard softmax (no approximation involved)

## When to Use
- Implementing attention kernels where sequence length N is large and the N x N attention matrix cannot fit in memory
- Building FlashAttention-style fused kernels that need to compute softmax within a tiled inner loop
- Any reduction operation where you need to compute softmax over a dimension that is processed in streaming/tiled fashion
- When memory bandwidth is the bottleneck (not compute)
- Long-context models where O(N^2) memory is prohibitive

## When NOT to Use
- When the full softmax input fits in fast memory (SRAM/registers) -- standard softmax is simpler
- Very short sequences where tiling overhead exceeds the benefit
- When you need the full attention matrix for analysis/visualization (online softmax never materializes it)
- Non-attention use cases where softmax is over a small fixed vocabulary (e.g., classification head with <1000 classes)

## Key Takeaways
- Online softmax is the foundational algorithm behind FlashAttention, enabling O(N) memory attention computation
- The correction factor `exp(m_old - m_new)` is the key mathematical trick that makes incremental softmax possible
- The algorithm is numerically exact -- it produces bit-identical results to standard safe softmax
- SRAM memory usage depends only on tile size and head dimension, not sequence length, enabling arbitrary-length attention
- The trade-off is more FLOPs (due to correction multiplications) for fewer memory accesses -- a favorable trade on modern GPUs that are memory-bandwidth bound

## References
- [From Online Softmax to FlashAttention - Zihao Ye, CSE 599M](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [From Online Softmax to Flash Attention V3 - Chenghua Wang](https://chenghuawang.github.io/keep-moving-forward/tech/fundamental_from_online_softmax_to_flash_attentionv3/)
- [The Basic Idea Behind FlashAttention - Peter Chng](https://peterchng.com/blog/2024/06/26/the-basic-idea-behind-flashattention/)
- Milakov, M. and Gimelshein, N. "Online normalizer calculation for softmax." arXiv:1805.02867, 2018.
- Dao, T. et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS, 2022.
