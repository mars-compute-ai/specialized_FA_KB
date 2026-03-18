---
skill_name: Online Softmax Derivation for FlashAttention
description: Mathematical derivation showing how the 3-pass safe softmax is reduced to a 2-pass online softmax, then extended to a 1-pass fused self-attention algorithm (FlashAttention) via surrogate sequences.
level: L1 - Algorithmic/Mathematical Stage Level
target_hardware: General (algorithm-level, applicable to any GPU with tiled execution)
relevance: When understanding or modifying the mathematical foundations of FlashAttention, designing new attention variants, or proving correctness of tiled softmax-based algorithms.
---

# Online Softmax Derivation for FlashAttention

## What It Is
This derivation, by Zihao Ye (UW CSE 599M, 2023), shows how FlashAttention's single-pass tiled algorithm emerges naturally from the online softmax trick. It starts with the standard 3-pass safe softmax (find max, compute denominators, normalize), reduces it to a 2-pass online softmax by introducing a surrogate denominator sequence that removes the dependency on the global max, and then extends the same surrogate technique to the full self-attention output O = softmax(QK^T)V, achieving a 1-pass recurrence that is compatible with tiling.

## Key Concepts
- **3-Pass Safe Softmax:** Pass 1 finds m = max(x), Pass 2 computes denominator d = sum(exp(x_i - m)), Pass 3 normalizes a_i = exp(x_i - m)/d. Requires 3 iterations over the data (3 HBM reads of Q,K for self-attention).
- **Surrogate Denominator (d'):** d'_i = sum_{j=1}^{i} exp(x_j - m_i) uses the running max m_i instead of the global max m_N. Key recurrence: d'_i = d'_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i). Since d'_N = d_N, the final value is identical.
- **2-Pass Online Softmax:** Fuses max-finding and denominator accumulation into one pass using the surrogate. Second pass still needed for normalization. Reduces HBM reads from 3 to 2.
- **Surrogate Output (o'):** o'_i = (1/d'_i) * sum_{j=1}^{i} exp(x_j - m_i) * V[j,:]. The recurrence: o'_i = o'_{i-1} * (d'_{i-1}/d'_i) * exp(m_{i-1} - m_i) + (exp(x_i - m_i)/d'_i) * V[i,:]. Since o'_N = o_N, the final output is identical.
- **1-Pass FlashAttention:** Fuses all of QK^T computation, online softmax statistics, and output accumulation into a single streaming pass over K/V blocks.
- **Tiling Compatibility:** All operations in the single-pass algorithm are associative, so block-wise (tile-wise) execution is valid. SRAM footprint depends only on block size B and head dimension D, not sequence length L.

## Algorithm / Pseudo-code
```
# Step 1: 3-Pass Safe Softmax (baseline -- 3 reads of data)
for i = 1..N: m_i = max(m_{i-1}, x_i)           # Pass 1: find global max
for i = 1..N: d_i = d_{i-1} + exp(x_i - m_N)    # Pass 2: compute denominator
for i = 1..N: a_i = exp(x_i - m_N) / d_N         # Pass 3: normalize

# Step 2: 2-Pass Online Softmax (2 reads of data)
for i = 1..N:
    m_i = max(m_{i-1}, x_i)
    d'_i = d'_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)  # surrogate denominator
for i = 1..N:
    a_i = exp(x_i - m_N) / d'_N                              # d'_N == d_N

# Step 3: 1-Pass FlashAttention (1 read of K,V data per row of Q)
for i = 1..N:
    x_i = Q[k,:] @ K^T[:,i]                                  # compute score
    m_i = max(m_{i-1}, x_i)                                  # update max
    d'_i = d'_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)   # update denom
    o'_i = o'_{i-1} * (d'_{i-1}/d'_i) * exp(m_{i-1} - m_i)  # rescale output
          + (exp(x_i - m_i) / d'_i) * V[i,:]                 # add new contrib
O[k,:] = o'_N

# Step 4: Tiled FlashAttention (block size b, #tiles = N/b)
for tile = 1..N/b:
    x = Q[k,:] @ K^T[:, (tile-1)*b : tile*b]     # b scores at once
    m_local = max(x)
    m_new = max(m_old, m_local)
    d' = d'_old * exp(m_old - m_new) + sum(exp(x - m_new))
    o' = o'_old * (d'_old/d') * exp(m_old - m_new)
       + sum_j (exp(x[j] - m_new) / d') * V[j + (tile-1)*b, :]
    m_old = m_new; d'_old = d'
O[k,:] = o'
```

## When to Use
- When you need to understand WHY FlashAttention is mathematically correct
- When designing new attention algorithm variants that need tiled softmax
- When proving correctness of modifications to the FlashAttention algorithm
- When teaching or explaining the connection between online algorithms and GPU-efficient attention
- When adapting the FlashAttention approach to non-standard attention patterns (e.g., cross-attention, sliding window)

## When NOT to Use
- When you simply need to USE FlashAttention (use the library instead)
- For attention mechanisms that don't involve softmax (e.g., linear attention, sigmoid attention)
- When the attention matrix is explicitly sparse and you can skip computing certain tiles entirely

## Key Takeaways
- The fundamental insight is the **surrogate sequence trick**: replacing a quantity that depends on the global statistic (m_N) with one that uses only the running statistic (m_i), while preserving the final value
- The same surrogate trick that reduces softmax from 3 passes to 2 passes can be applied AGAIN to the output accumulation O = softmax(QK^T)V, reducing the full self-attention from 2 passes to 1 pass
- The 1-pass algorithm has states (x_i, m_i, d'_i, o'_i) with O(d) memory footprint per row, easily fitting in SRAM
- Tiling works because the block-level recurrence has the same mathematical structure as the element-level recurrence
- SRAM footprint: O(B * d) regardless of sequence length L, enabling arbitrarily long sequences

## References
- Zihao Ye, "From Online Softmax to FlashAttention," UW CSE 599M Notes, May 2023: https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
- Milakov & Gimelshein, "Online normalizer calculation for softmax," 2018: https://arxiv.org/abs/1805.02867
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022
