# From Online Softmax to FlashAttention

**Author:** Zihao Ye (zhye@cs.washington.edu)
**Date:** May 11, 2023
**Course:** UW CSE 599M Spring 2023: ML for ML Systems
**Source:** https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf

## Overview

The key innovation of FlashAttention is using an idea similar to Online Softmax to tile the self-attention computation, so that we can fuse the entire multi-head attention layer without accessing GPU global memory for intermediate logits and attention scores. This note explains why tiling self-attention computation is non-trivial, and how to derive FlashAttention computation from the online softmax trick.

## 1. The Self-Attention

The computation of Self-Attention (ignoring heads, batches, attention masks, and scale factor 1/sqrt(D) for simplicity):

```
O = softmax(Q K^T) V
```

where Q, K, V, O are 2D matrices with shape (L, D), where L is the sequence length and D is the dimension per head (head dimension). The softmax applies to the last dimension (columns).

The standard approach factorizes the computation into several stages:

```
X = Q K^T          (pre-softmax logits)
A = softmax(X)     (attention scores)
O = A V            (output)
```

One key fact about FlashAttention is that we don't need to materialize X and A matrices on global memory. Instead, the entire computation is fused in a single CUDA kernel, requiring careful on-chip memory management (like stream algorithms) because NVIDIA GPU's shared memory is small.

For classical algorithms such as matrix multiplication, tiling ensures on-chip memory does not exceed hardware limits. During kernel execution, only 3T^2 elements are stored on-chip, regardless of matrix shape. This works because addition is associative, allowing decomposition into tile-wise matrix multiplications.

However, Self-Attention includes a softmax operator that is not directly associative, making it hard to simply tile Self-Attention. The question becomes: Is there a way to make softmax associative?

## 2. (Safe) Softmax

The generic formula of softmax computation:

```
softmax({x_1, ..., x_N}) = { e^{x_i} / sum_{j=1}^{N} e^{x_j} } for i=1..N
```

Note that x_i might be very large and e^{x_i} can easily overflow. For float16, the maximum is 65536, meaning for x > 11, e^x would exceed the effective range.

To mitigate this, the **safe softmax** trick is used:

```
e^{x_i} / sum_{j=1}^{N} e^{x_j} = e^{x_i - m} / sum_{j=1}^{N} e^{x_j - m}
```

where m = max_{j=1}^{N}(x_j), so that each x_i - m <= 0, which is safe because the exponential is accurate for negative inputs.

### 3-Pass Safe Softmax Algorithm

Notations:
- {m_i}: max_{j=1}^{i} {x_j}, with initial value m_0 = -infinity
- {d_i}: sum_{j=1}^{i} e^{x_j - m_N}, with initial value d_0 = 0, d_N is the denominator
- {a_i}: the final softmax value

```
Pass 1: for i = 1 to N:  m_i = max(m_{i-1}, x_i)
Pass 2: for i = 1 to N:  d_i = d_{i-1} + e^{x_i - m_N}
Pass 3: for i = 1 to N:  a_i = e^{x_i - m_N} / d_N
```

This algorithm requires iterating over [1, N] three times. In self-attention context, the {x_i} are pre-softmax logits computed by QK^T. If we don't store all logits (SRAM too small), we need to access Q and K three times to re-compute logits on-the-fly, which is not I/O efficient.

## 3. Online Softmax

If we fuse equations for m and d in a single loop, we can reduce global memory access from 3 to 1. Unfortunately, we cannot directly fuse because d depends on m_N, which cannot be determined until the first loop completes.

The solution: create a surrogate sequence d'_i := sum_{j=1}^{i} e^{x_j - m_i} to remove the dependency on N. The N-th terms are identical: d_N = d'_N.

The recurrence relation for d'_i:

```
d'_i = d'_{i-1} * e^{m_{i-1} - m_i} + e^{x_i - m_i}
```

This only relies on m_i and m_{i-1}, and we can compute m_j and d'_j together:

### 2-Pass Online Softmax Algorithm

```
Pass 1: for i = 1 to N:
    m_i = max(m_{i-1}, x_i)
    d'_i = d'_{i-1} * e^{m_{i-1} - m_i} + e^{x_i - m_i}

Pass 2: for i = 1 to N:
    a_i = e^{x_i - m_N} / d'_N
```

This is the algorithm proposed in the Online Softmax paper (Milakov & Gimelshein, 2018). However, it still requires two passes.

## 4. FlashAttention

For softmax alone, we cannot reduce to 1 pass. But in Self-Attention, our final target is not the attention score matrix A, but O = A * V. Can we find a one-pass recurrence form for O instead?

### Multi-Pass Self-Attention (recurrence form for row k)

Notations:
- Q[k,:]: the k-th row vector of Q matrix
- K^T[:,i]: the i-th column vector of K^T matrix
- V[i,:]: the i-th row of V matrix
- {o_i}: partial aggregation result sum_{j=1}^{i} a_j * V[j,:]

```
Pass 1: for i = 1 to N:
    x_i = Q[k,:] * K^T[:,i]
    m_i = max(m_{i-1}, x_i)
    d'_i = d'_{i-1} * e^{m_{i-1} - m_i} + e^{x_i - m_i}

Pass 2: for i = 1 to N:
    a_i = e^{x_i - m_N} / d'_N
    o_i = o_{i-1} + a_i * V[i,:]
```

By creating a surrogate sequence o' analogous to d':

```
o'_i = (1/d'_i) * sum_{j=1}^{i} e^{x_j - m_i} * V[j,:]
```

The N-th elements are identical: o_N = o'_N. The recurrence relation:

```
o'_i = o'_{i-1} * (d'_{i-1}/d'_i) * e^{m_{i-1} - m_i} + (e^{x_i - m_i}/d'_i) * V[i,:]
```

This only depends on d'_i, d'_{i-1}, m_i, m_{i-1} and x_i, so all computations fuse into a **single loop**:

### FlashAttention Algorithm (Single-Pass)

```
for i = 1 to N:
    x_i = Q[k,:] * K^T[:,i]
    m_i = max(m_{i-1}, x_i)
    d'_i = d'_{i-1} * e^{m_{i-1} - m_i} + e^{x_i - m_i}
    o'_i = o'_{i-1} * (d'_{i-1}/d'_i) * e^{m_{i-1} - m_i} + (e^{x_i - m_i}/d'_i) * V[i,:]

O[k,:] = o'_N
```

The states x_i, m_i, d'_i, and o'_i have small footprints that easily fit into GPU shared memory.

### FlashAttention (Tiling)

Because all operations are associative, the algorithm is compatible with tiling:

```
Notations:
    b: block size of the tile
    #tiles: number of tiles in the row, N = b * #tiles
    x_i: vector storing Q[k] K^T values of the i-th tile
    m_i^(local): local maximum inside x_i

for i = 1 to #tiles:
    x_i = Q[k,:] * K^T[:, (i-1)*b : i*b]
    m_i^(local) = max_{j=1}^{b} (x_i[j])
    m_i = max(m_{i-1}, m_i^(local))
    d'_i = d'_{i-1} * e^{m_{i-1} - m_i} + sum_{j=1}^{b} e^{x_i[j] - m_i}
    o'_i = o'_{i-1} * (d'_{i-1}/d'_i) * e^{m_{i-1} - m_i}
          + sum_{j=1}^{b} (e^{x_i[j] - m_i} / d'_i) * V[j+(i-1)*b, :]

O[k,:] = o'_{N/b}
```

The overall SRAM memory footprint depends only on B (block size) and D (head dimension) and is not related to L (sequence length). This algorithm scales to long context without encountering memory issues. During computation, tiles are swept left to right for K^T and A, top to bottom for V, updating the state of m, d, and O accordingly.

## References

1. Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Re. FlashAttention: fast and memory-efficient exact attention with IO-awareness. CoRR, abs/2205.14135, 2022.
2. Andrew Kerr. GTC 2020: developing CUDA kernels to push tensor cores to the absolute limit on NVIDIA A100. May 2020.
3. Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. CoRR, abs/1805.02867, 2018.
