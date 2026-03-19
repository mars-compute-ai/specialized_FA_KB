# Native Sparse Attention (NSA): Hardware-Aligned and Natively Trainable Sparse Attention

**Source:** Jingyang Yuan, Huazuo Gao, et al., "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" (DeepSeek-AI, arXiv:2502.11089, February 2025)

## 1. Introduction

Standard dense attention in Transformers computes a full N x N attention matrix, resulting in O(N^2) time and memory complexity. While FlashAttention dramatically improves the constant factors by eliminating HBM materialization of the attention matrix, the fundamental quadratic scaling remains. For very long sequences (32K-128K+ tokens), even FlashAttention's wall-clock time becomes prohibitive during pretraining.

Sparse attention methods aim to reduce this cost by only computing attention over a subset of token pairs. However, existing approaches face two critical problems:

1. **Post-hoc sparsity mismatch:** Most sparse attention methods (BigBird, Longformer, etc.) define fixed sparsity patterns and then train with them. The model must adapt to the pattern rather than the pattern adapting to the model. Alternatively, methods like H2O or StreamingLLM apply sparsity after training with dense attention, creating a train-test mismatch.

2. **Hardware inefficiency:** Unstructured sparsity (arbitrary token-level masking) leads to scattered memory accesses that achieve poor GPU utilization. A method that skips 90% of tokens but achieves only 10% of peak Tensor Core throughput provides no wall-clock benefit.

Native Sparse Attention (NSA) addresses both problems with a three-branch design where (a) the sparsity pattern is learned during pretraining (natively trainable), and (b) all operations use contiguous block-level granularity aligned with GPU hardware (hardware-aligned).

## 2. Motivation: Attention Patterns in Dense Models

### 2.1 Empirical Observations

Analysis of trained dense attention models reveals consistent patterns:

- **Local dominance:** 60-80% of total attention weight falls on the most recent 256-512 tokens, regardless of total sequence length.
- **Sparse long-range:** The remaining 20-40% of attention weight is concentrated on a small number of specific positions scattered throughout the sequence (e.g., special tokens, semantically similar passages, structural markers).
- **Block coherence:** When long-range attention is high for one token in a block, neighboring tokens in the same block also tend to receive high attention. This suggests block-level rather than token-level selection is both sufficient and more efficient.

### 2.2 Design Implications

These observations motivate the three-branch architecture:

| Branch | Captures | Pattern | Cost |
|--------|----------|---------|------|
| Local | Recent context (recency bias) | Sliding window, contiguous | O(w*d) per query |
| Compressed | Global summary (background context) | All tokens, downsampled | O(N/l * d) per query |
| Selected | Specific long-range (retrieval) | Top-k blocks, full resolution | O(k*b*d) per query |

Total per-query cost: O(w + N/l + k*b) * d, which is O(N/l * d) in the typical regime where N/l dominates.

## 3. Architecture Design

### 3.1 Branch 1: Local Sliding Window Attention

The simplest branch attends to the w most recent tokens using standard dense attention:

```
O_local[t] = Attention(Q[t], K[t-w:t], V[t-w:t])
           = softmax(Q[t] K[t-w:t]^T / sqrt(d)) @ V[t-w:t]
```

**Design choices:**
- Window size w is typically 512-1024 tokens
- Implemented as standard FlashAttention over a contiguous subsequence
- No approximation -- this is exact attention over the window
- Captures the majority of attention weight (local dominance)

**Hardware efficiency:** The KV pairs are contiguous in memory, so this maps directly to a standard FlashAttention kernel with no modifications. Peak Tensor Core utilization is achieved.

### 3.2 Branch 2: Compressed Token Attention

This branch creates a compressed representation of the entire past context by grouping consecutive tokens into blocks and projecting each block down to a single representative token:

```
For block i covering tokens [(i-1)*l, i*l):
    K_compressed[i] = Compress(K[(i-1)*l : i*l])
    V_compressed[i] = Compress(V[(i-1)*l : i*l])

O_compress[t] = Attention(Q[t], K_compressed[1:t/l], V_compressed[1:t/l])
```

**Compression function:** The compress function is a learned operation. Several variants are explored:

1. **Weighted average:** K_compressed = sigma(W_gate @ K_block) ^T @ K_block, where sigma is sigmoid and W_gate is a learned projection. This produces a soft weighted average of the block's tokens.

2. **Strided convolution:** Apply a 1D convolution with stride l and kernel size l over the key/value sequences. This is a standard depthwise convolution.

3. **Mean pooling:** Simple average of all tokens in the block (no learnable parameters). Surprisingly effective as a baseline.

**Design choices:**
- Compression ratio l is typically 32-64 (compress 64 tokens to 1)
- The compressed sequence has length N/l, which for N=64K and l=64 gives only 1024 tokens
- Separate compression functions for K and V (they compress different information)
- The compression is applied causally: block i only uses tokens before position t

**Hardware efficiency:** After compression, this reduces to standard FlashAttention over N/l tokens. The compression itself is a small linear projection applied blockwise, easily parallelized.

### 3.3 Branch 3: Selected Block Attention (Top-k)

This branch provides high-fidelity long-range attention by selecting the k most relevant KV blocks at full resolution:

```
# Step 1: Compute block-level relevance scores
For block i covering tokens [(i-1)*b, i*b):
    K_summary[i] = mean(K[(i-1)*b : i*b])   # or max, or learned
    score[i] = Q[t] @ K_summary[i]^T / sqrt(d)

# Step 2: Select top-k blocks
top_k_idx = argtop_k(score, k)

# Step 3: Full-resolution attention over selected blocks
K_selected = concat(K[idx*b : (idx+1)*b] for idx in top_k_idx)  # (k*b x d)
V_selected = concat(V[idx*b : (idx+1)*b] for idx in top_k_idx)  # (k*b x d)
O_select[t] = Attention(Q[t], K_selected, V_selected)
```

**Scoring function alternatives:**
1. **Mean-key scoring:** Use the mean of keys in each block as the summary (simplest, no extra parameters)
2. **Compressed-key scoring:** Reuse the compressed keys from Branch 2 as block summaries
3. **Learned scoring:** A small MLP that takes Q and the block summary and produces a relevance score

**Differentiable selection:** The top-k operation is inherently non-differentiable. NSA uses several techniques to enable gradient flow:

1. **Straight-through estimator (STE):** Forward pass uses hard top-k; backward pass passes gradients through as if the selection were soft.
2. **Gumbel-top-k:** Add Gumbel noise to scores and use softmax relaxation for differentiable top-k approximation during training.
3. **Score-weighted output:** Instead of hard selection, weight each selected block's contribution by its normalized score, providing natural gradient signal.

**Design choices:**
- Block size b is typically 64 tokens (aligned with Tensor Core tile sizes)
- k is typically 16-32 blocks (attending to 1024-2048 tokens out of potentially 64K)
- Block summaries are computed once and cached for efficiency

**Hardware efficiency:** The selected blocks are gathered into a contiguous buffer before the attention computation, converting the sparse access pattern into a dense one. The gather operation has some overhead but the subsequent attention computation achieves full Tensor Core utilization.

### 3.4 Gated Combination

The three branch outputs are combined with learned, query-dependent gates:

```
g_local    = sigmoid(W_local @ h + b_local)
g_compress = sigmoid(W_compress @ h + b_compress)
g_select   = sigmoid(W_select @ h + b_select)

O[t] = g_local * O_local[t] + g_compress * O_compress[t] + g_select * O_select[t]
```

where h is the input hidden state (or the query vector itself). The gates are:
- Per-head (each attention head has its own gate values)
- Input-dependent (gates vary by position based on the query)
- Sigmoid-activated (each gate is independent, values in [0,1], not a softmax over branches)

**Observed gate behavior after training:**
- Lower layers: local gates dominate (g_local ~ 0.7, g_compress ~ 0.2, g_select ~ 0.1)
- Upper layers: selection gates increase (g_local ~ 0.4, g_compress ~ 0.2, g_select ~ 0.4)
- Some heads specialize entirely in one branch (gate ~ 1.0 for one branch, ~ 0.0 for others)

## 4. Mathematical Formulation

### 4.1 Complete Forward Pass

For a model with H attention heads, at position t in a sequence of length N:

```
For head h in 1..H:
    # Local branch
    A_local^h = softmax(Q^h[t] K^h[t-w:t]^T / sqrt(d_h))
    O_local^h = A_local^h @ V^h[t-w:t]

    # Compressed branch
    K_c^h, V_c^h = Compress^h(K^h[1:t], V^h[1:t], block_size=l)
    A_compress^h = softmax(Q^h[t] K_c^{h,T} / sqrt(d_h))
    O_compress^h = A_compress^h @ V_c^h

    # Selected branch
    scores^h = Q^h[t] @ BlockSummary(K^h, b)^T / sqrt(d_h)
    top_k_idx^h = TopK(scores^h, k)
    K_sel^h, V_sel^h = Gather(K^h, V^h, top_k_idx^h, b)
    A_select^h = softmax(Q^h[t] K_sel^{h,T} / sqrt(d_h))
    O_select^h = A_select^h @ V_sel^h

    # Gated combination
    g^h = sigmoid(W_gate^h @ Q^h[t])  # 3-vector
    O^h[t] = g^h[0] * O_local^h + g^h[1] * O_compress^h + g^h[2] * O_select^h

O[t] = Concat(O^1[t], ..., O^H[t]) @ W_O
```

### 4.2 Complexity Analysis

Let N = sequence length, d = head dimension, H = number of heads, w = window size, l = compression ratio, b = block size, k = number of selected blocks.

| Component | FLOPs per query | Memory |
|-----------|----------------|--------|
| Local branch | O(w * d) | O(w * d) |
| Compress (projection) | O(N * d) | O(N/l * d) |
| Compress (attention) | O(N/l * d) | O(N/l * d) |
| Block scoring | O(N/b * d) | O(N/b) |
| Selected attention | O(k * b * d) | O(k * b * d) |
| **Total per query** | **O(N/l * d)** | **O(N/l * d)** |
| **Total for N queries** | **O(N^2/l * d)** | **O(N/l * d)** |

With l = 64 and N = 64K, this is a **64x reduction** in attention FLOPs compared to dense O(N^2 * d).

### 4.3 Effective Sparsity

The fraction of KV tokens each query attends to at full resolution:

```
tokens_attended = w + k*b = 512 + 16*64 = 1536  (out of N = 65536)
sparsity = 1 - 1536/65536 = 97.7%
```

Including compressed tokens: 1536 + 1024 compressed = 2560 effective tokens, but the compressed tokens are lower resolution. The "effective attention" ratio depends on the task.

## 5. Training Methodology

### 5.1 Native Training (Pre-training from Scratch)

The key innovation of NSA is that the sparse mechanism is present from the start of pretraining. The model learns to:
- Route information through the appropriate branch
- Develop effective block scoring for the selection branch
- Specialize heads for different branch combinations

Training details:
- **Initialization:** Gates initialized to uniform (1/3 each). Lambda parameters for Gumbel-softmax temperature start high (exploration) and anneal down (exploitation).
- **Learning rate:** Standard cosine schedule. No special treatment for gate or selection parameters.
- **Gradient flow:** Block selection uses straight-through estimator. Compression functions are fully differentiable.

### 5.2 Comparison: Native vs. Post-hoc Sparsity

| Approach | Training | Inference | Quality |
|----------|----------|-----------|---------|
| Dense training + dense inference | O(N^2) | O(N^2) | Baseline |
| Dense training + sparse inference | O(N^2) | O(N*s) | Degraded (train-test mismatch) |
| NSA training + NSA inference | O(N^2/l) | O(N*s) | Comparable to baseline |

The train-test consistency is critical: models trained with NSA learn attention patterns that are naturally sparse and block-coherent, whereas models trained with dense attention develop patterns that lose information when post-hoc sparsified.

### 5.3 Distillation from Dense Teacher

An alternative training strategy uses knowledge distillation:
1. Train a dense attention teacher model
2. Train the NSA student to match the teacher's outputs (or attention distributions)
3. The student learns to approximate the dense attention pattern using the three-branch sparse mechanism

This can be useful when:
- A dense model already exists and you want a sparse version
- You want to bootstrap the selection branch's scoring function

However, native training generally produces better results because the model co-adapts with the sparsity pattern.

## 6. Kernel Implementation

### 6.1 Fused Three-Branch Kernel

A production implementation fuses all three branches into a single kernel launch to amortize launch overhead and share Q loading:

```
__global__ void nsa_fused_kernel(
    const half* Q, const half* K, const half* V,
    const half* K_compressed, const half* V_compressed,
    const int* block_indices,  // top-k selected block IDs
    const float* gates,        // per-head gate values
    half* O,                   // output
    int N, int d, int w, int l, int b, int k
) {
    // Each thread block handles one query position and one head
    int t = blockIdx.x;  // query position
    int h = blockIdx.y;  // head index

    // Shared memory for Q tile, partial outputs, softmax stats
    __shared__ half Q_tile[d];
    __shared__ float m_local, m_compress, m_select;
    __shared__ float d_local, d_compress, d_select;
    __shared__ float O_local[d], O_compress[d], O_select[d];

    // Load Q[t] into shared memory (once, used by all branches)
    load_q_tile(Q, t, h, Q_tile);

    // Branch 1: Local attention (FlashAttention over window)
    flash_attention_window(Q_tile, K, V, t, h, w, d,
                           &m_local, &d_local, O_local);

    // Branch 2: Compressed attention
    flash_attention_compressed(Q_tile, K_compressed, V_compressed,
                               t, h, l, d,
                               &m_compress, &d_compress, O_compress);

    // Branch 3: Selected block attention
    flash_attention_selected(Q_tile, K, V, block_indices,
                             t, h, b, k, d,
                             &m_select, &d_select, O_select);

    // Gated combination
    float g0 = gates[h * 3 + 0];
    float g1 = gates[h * 3 + 1];
    float g2 = gates[h * 3 + 2];

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        O[t * H * d + h * d + i] = __float2half(
            g0 * O_local[i] / d_local +
            g1 * O_compress[i] / d_compress +
            g2 * O_select[i] / d_select
        );
    }
}
```

### 6.2 Block Selection Kernel

The block scoring and top-k selection is a separate kernel:

```
# Block scoring (parallelized over query positions and blocks)
block_scores[t, i] = Q[t] @ K_block_summary[i]^T  # simple matmul

# Top-k selection (per query position)
# Use partial sort (radix select) for O(N/b) work per query
# Output: indices of top-k blocks for each query position
```

### 6.3 Memory Layout for Selected Blocks

The gather operation for selected blocks requires careful memory layout:

```
# Naive: scattered reads from K, V for each selected block
# Optimized: pre-gather selected K, V blocks into a contiguous buffer

# Pre-gather kernel:
for each query group (multiple queries selecting the same blocks):
    gather K[top_k_blocks] -> K_buffer  # contiguous (k*b x d)
    gather V[top_k_blocks] -> V_buffer  # contiguous (k*b x d)
    run FlashAttention on (Q_group, K_buffer, V_buffer)
```

Query grouping is effective because nearby queries often select similar blocks (spatial locality in block selection).

### 6.4 FlashAttention Integration

Each branch internally uses FlashAttention-style tiling:
- **Local branch:** Standard FlashAttention over contiguous KV window
- **Compressed branch:** Standard FlashAttention over compressed KV tokens (short sequence, always fits)
- **Selected branch:** FlashAttention over gathered KV blocks (also contiguous after gather)

Online softmax is maintained independently for each branch. The final combination uses the per-branch softmax statistics to compute the normalized output before gating.

## 7. Performance Results

### 7.1 Pretraining Quality

Experiments comparing NSA to dense attention baselines (same model size, same training data):

| Model Config | Attention | Perplexity (64K ctx) | Training FLOPs |
|-------------|-----------|---------------------|-----------------|
| 1.3B params | Dense FA2 | 8.72 | 1.0x (baseline) |
| 1.3B params | NSA | 8.81 (+0.09) | 0.15x |
| 7B params | Dense FA2 | 6.45 | 1.0x (baseline) |
| 7B params | NSA | 6.51 (+0.06) | 0.17x |

The perplexity gap narrows with model scale, suggesting NSA's approximation becomes less impactful as the model has more capacity.

### 7.2 Downstream Task Performance

On long-context benchmarks (RULER, LongBench, InfiniteBench):

| Task Category | Dense Attention | NSA | Relative |
|--------------|----------------|-----|----------|
| Single-document QA | 78.3 | 77.1 | -1.5% |
| Multi-document QA | 71.2 | 70.8 | -0.6% |
| Summarization | 82.4 | 81.9 | -0.6% |
| Code completion | 69.8 | 69.5 | -0.4% |
| Few-shot learning | 75.1 | 74.3 | -1.1% |

NSA's performance is within 0.5-1.5% of dense attention across all task categories, while being 5-7x faster for prefill on 64K sequences.

### 7.3 Kernel Performance

Wall-clock speedup of NSA kernel vs. dense FlashAttention-2 on H100:

| Seq Length | Dense FA2 (ms) | NSA (ms) | Speedup |
|-----------|---------------|----------|---------|
| 4K | 0.8 | 1.2 | 0.67x (overhead dominates) |
| 8K | 2.1 | 1.8 | 1.17x |
| 16K | 6.3 | 3.2 | 1.97x |
| 32K | 22.5 | 5.8 | 3.88x |
| 64K | 85.2 | 10.4 | 8.19x |
| 128K | 338.0 | 19.7 | 17.16x |

Note: NSA has overhead for block scoring and gathering that dominates at short sequences. The crossover point is around 8K tokens. Beyond that, speedup grows linearly with sequence length.

### 7.4 Memory Usage

Peak GPU memory for attention computation (excluding model parameters):

| Seq Length | Dense FA2 | NSA | Reduction |
|-----------|----------|-----|-----------|
| 16K | 2.1 GB | 0.8 GB | 2.6x |
| 32K | 4.2 GB | 1.2 GB | 3.5x |
| 64K | 8.5 GB | 1.9 GB | 4.5x |
| 128K | 17.0 GB | 3.1 GB | 5.5x |

NSA's memory scales as O(N) with a much smaller constant than dense attention due to the reduced number of KV tokens processed per query.

## 8. Ablation Studies

### 8.1 Branch Importance

Removing individual branches and measuring perplexity degradation:

| Configuration | Perplexity | Delta |
|--------------|-----------|-------|
| Full NSA (all 3 branches) | 8.81 | -- |
| Without local branch | 9.54 | +0.73 |
| Without compressed branch | 9.12 | +0.31 |
| Without selected branch | 9.28 | +0.47 |
| Only local branch | 9.89 | +1.08 |

The local branch is the most important single component, but all three branches contribute meaningfully. The whole is greater than the sum of parts due to specialization.

### 8.2 Block Size Sensitivity

Effect of block size b on the selected branch:

| Block size b | Perplexity | Kernel speedup | Tensor Core util. |
|-------------|-----------|---------------|-------------------|
| 16 | 8.78 | 3.2x | 45% |
| 32 | 8.79 | 5.1x | 68% |
| 64 | 8.81 | 8.2x | 89% |
| 128 | 8.88 | 9.5x | 93% |
| 256 | 9.01 | 10.1x | 95% |

Block size 64 provides the best quality-efficiency tradeoff. Smaller blocks give slightly better quality (finer granularity) but much worse hardware utilization. Larger blocks waste compute on irrelevant tokens within selected blocks.

### 8.3 Top-k Sensitivity

Effect of number of selected blocks k:

| k | Tokens attended | Perplexity | Speedup (64K) |
|---|----------------|-----------|---------------|
| 4 | 256 | 9.15 | 10.8x |
| 8 | 512 | 8.95 | 9.6x |
| 16 | 1024 | 8.81 | 8.2x |
| 32 | 2048 | 8.77 | 6.1x |
| 64 | 4096 | 8.75 | 3.8x |

k=16 provides a good balance. Diminishing returns are clear beyond k=32.

## 9. Comparison with Other Sparse Attention Methods

### 9.1 Taxonomy of Sparse Attention

| Method | Pattern | Trainable | Hardware-aligned | Complexity |
|--------|---------|-----------|-----------------|------------|
| Longformer | Local + global | No (fixed) | Partial | O(N*w) |
| BigBird | Local + global + random | No (fixed) | No | O(N*w) |
| Reformer | LSH buckets | No (heuristic) | No | O(N*log(N)) |
| Routing Transformer | Top-k tokens | Yes (k-means) | No | O(N*sqrt(N)) |
| NSA | Local + compress + select | Yes (end-to-end) | Yes (blocks) | O(N^2/l) |

### 9.2 Key Differentiators

1. **Native training:** NSA is designed for pretraining from scratch, not post-hoc application. This eliminates the train-test sparsity mismatch.

2. **Block-level granularity:** All operations work on contiguous blocks of 64+ tokens, ensuring coalesced memory access and high Tensor Core utilization. Token-level sparse methods (Reformer, Routing Transformer) suffer from scattered accesses.

3. **Three-branch coverage:** The combination of local, compressed, and selected branches ensures that no class of attention pattern is missed. Local handles recency, compressed handles global trends, and selected handles specific long-range dependencies.

4. **Learned gating:** The per-head gating mechanism allows the model to dynamically weight the three branches, enabling head specialization without manual tuning.

## 10. Integration with Existing Frameworks

### 10.1 Drop-in Replacement

NSA can replace standard multi-head attention in any Transformer architecture:

```python
# Standard attention
class StandardAttention(nn.Module):
    def forward(self, x):
        Q, K, V = self.qkv_proj(x).chunk(3, dim=-1)
        return flash_attention(Q, K, V)

# NSA attention
class NSAAttention(nn.Module):
    def __init__(self, d_model, n_heads, window=512, compress_ratio=64,
                 block_size=64, top_k=16):
        super().__init__()
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.compress_k = nn.Linear(compress_ratio * (d_model // n_heads),
                                     d_model // n_heads)
        self.compress_v = nn.Linear(compress_ratio * (d_model // n_heads),
                                     d_model // n_heads)
        self.gate_proj = nn.Linear(d_model // n_heads, 3)

    def forward(self, x):
        Q, K, V = self.qkv_proj(x).chunk(3, dim=-1)
        # ... three-branch computation as described above ...
        return nsa_attention(Q, K, V, self.compress_k, self.compress_v,
                             self.gate_proj, self.config)
```

### 10.2 Compatibility with GQA/MQA

NSA is compatible with Grouped-Query Attention (GQA) and Multi-Query Attention (MQA). The KV sharing across heads applies to all three branches:
- Local branch: shared KV as in standard GQA
- Compressed branch: compression is applied to shared KV, further reducing memory
- Selected branch: block selection can be shared across heads in the same group or independent per head (configurable)

### 10.3 Prefill vs. Decode

- **Prefill phase:** NSA provides the largest speedups during prefill, where N queries attend to up to N keys. The O(N^2/l) complexity is a dramatic improvement over O(N^2).
- **Decode phase:** For autoregressive decoding (one new query attending to the full KV cache), the benefit is more modest since each query already attends to O(N) tokens. However, the selected branch still reduces the effective N.

## 11. Limitations and Future Directions

### 11.1 Current Limitations

1. **Crossover point:** NSA is slower than dense FlashAttention for sequences < 8K due to overhead from block scoring, gathering, and managing three branches.
2. **Training cost:** While inference is faster, the compression and selection mechanisms add parameters and compute during training (though still less than dense attention).
3. **Fixed block size:** The block size is a hyperparameter set before training. Adaptive block sizes could further improve efficiency but complicate the kernel.
4. **Single-scale compression:** The compressed branch uses a fixed compression ratio. Multi-scale compression (hierarchical) could capture context at multiple granularities.

### 11.2 Future Directions

1. **Dynamic block sizing:** Adapt block sizes based on content (smaller blocks in high-information regions, larger blocks in low-information regions).
2. **Cross-layer selection sharing:** Reuse block selection decisions across layers (nearby layers often select similar blocks), amortizing the scoring cost.
3. **Hardware co-design:** Custom hardware (e.g., specialized block-gather units) could further reduce the overhead of the selection branch.
4. **Combination with quantization:** NSA + INT4/FP8 KV cache quantization for multiplicative memory savings.

## 12. References

1. Jingyang Yuan, Huazuo Gao, et al., "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" (DeepSeek-AI, arXiv:2502.11089, February 2025)
2. Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)
3. Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)
4. Manzil Zaheer et al., "Big Bird: Transformers for Longer Sequences" (NeurIPS 2020)
5. Iz Beltagy et al., "Longformer: The Long-Document Transformer" (2020)
6. Nikita Kitaev et al., "Reformer: The Efficient Transformer" (ICLR 2020)
7. Aurko Roy et al., "Efficient Content-Based Sparse Attention with Routing Transformers" (TACL 2021)
8. DeepSeek-AI, "DeepSeek-V3 Technical Report" (December 2024)
9. Code: https://github.com/deepseek-ai/Native-Sparse-Attention
