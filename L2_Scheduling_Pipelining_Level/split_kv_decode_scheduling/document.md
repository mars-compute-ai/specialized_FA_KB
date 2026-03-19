# Split-KV Decode Scheduling for Attention

## Overview

During autoregressive LLM inference, the decode phase generates one token at a time. Each new token attends to the entire KV cache accumulated from all previous tokens. This creates a fundamental scheduling problem: the query has only 1 token (or a few, with speculative decoding), but the key/value sequence can be thousands to millions of tokens long. Standard FlashAttention parallelizes over batch and heads but not over the KV sequence, leaving most GPU SMs idle during decode. Split-KV scheduling (FlashDecoding) solves this by partitioning the KV sequence into blocks, computing partial attention in parallel, and combining results via a lightweight log-sum-exp reduction.

---

## 1. The Decode Scheduling Problem

### 1.1 Why Standard FlashAttention Underutilizes GPUs During Decode

Standard FlashAttention-2 launches a grid of `(num_q_tiles, B * H)` thread blocks:

```
Grid dimensions for FlashAttention-2:
  Prefill: grid = (ceil(S_q / BLOCK_M), B * H)
    - S_q = 2048, BLOCK_M = 128: num_q_tiles = 16
    - B = 4, H = 32: B*H = 128
    - Total thread blocks: 16 * 128 = 2048 (plenty for 108-132 SMs)

  Decode: grid = (ceil(1 / BLOCK_M), B * H)
    - S_q = 1 (single token): num_q_tiles = 1
    - B = 1, H = 32: B*H = 32
    - Total thread blocks: 1 * 32 = 32 (only 30% of A100's 108 SMs used)
```

**The problem**: With batch size 1 and 32 attention heads, only 32 thread blocks launch. An A100 has 108 SMs, an H100 has 132 SMs, and an MI300X has 304 CUs. Most compute units sit idle.

### 1.2 SM Utilization Analysis

| GPU | SMs/CUs | B=1, H=32 | B=1, H=8 (GQA) | B=4, H=32 |
|-----|---------|-----------|-----------------|-----------|
| A100 | 108 SMs | 30% | 7% | >100% |
| H100 | 132 SMs | 24% | 6% | 97% |
| MI300X | 304 CUs | 11% | 3% | 42% |

For GQA models (like Llama 2/3 with 8 KV heads), the problem is even worse: only 8 thread blocks for B=1.

### 1.3 Memory Bandwidth Waste

Even when an SM is active during decode, it processes the entire KV sequence sequentially in its inner loop. The sequential access pattern means HBM bandwidth is used by only the active SMs, achieving a fraction of peak bandwidth. On MI300X with 11% SM utilization, theoretical maximum HBM utilization is also ~11%.

---

## 2. The Split-KV Algorithm (FlashDecoding)

### 2.1 Core Idea

Add a third parallelism dimension by splitting the KV sequence:

```
Standard:  grid = (1, B * H)           -- parallelize over batch, heads
Split-KV:  grid = (num_splits, B * H)  -- parallelize over batch, heads, KV splits
```

Each thread block processes a subset of the KV sequence, computing a partial attention output with local softmax statistics.

### 2.2 Algorithm Detail

```python
def split_kv_decode_attention(Q, K, V, num_splits):
    """
    Q: [B, H, 1, d]      -- single query token
    K: [B, H, S, d]      -- KV cache keys
    V: [B, H, S, d]      -- KV cache values
    num_splits: int       -- number of KV sequence splits
    """
    B, H, S, d = K.shape
    split_size = ceil(S / num_splits)

    # Phase 1: Parallel partial attention
    # Grid: (num_splits, B * H)
    O_partial = zeros(B, H, num_splits, d)   # partial outputs
    lse_partial = zeros(B, H, num_splits)     # log-sum-exp per split

    for split_id in parallel(num_splits):      # parallelized across thread blocks
        kv_start = split_id * split_size
        kv_end = min(kv_start + split_size, S)

        # Standard FlashAttention inner loop over this KV range
        m = -inf          # running max
        l = 0.0           # running sum of exp
        o = zeros(d)      # running output accumulator

        for j in range(kv_start, kv_end, BLOCK_KV):
            k_block = K[:, :, j:j+BLOCK_KV, :]
            v_block = V[:, :, j:j+BLOCK_KV, :]

            # Attention scores
            s = Q @ k_block.T * (1 / sqrt(d))    # [1, BLOCK_KV]

            # Online softmax update
            m_new = max(m, max(s))
            exp_s = exp(s - m_new)
            alpha = exp(m - m_new)               # rescale factor

            l = alpha * l + sum(exp_s)
            o = alpha * o + exp_s @ v_block
            m = m_new

        # Store partial result (unnormalized output + log-sum-exp)
        O_partial[:, :, split_id, :] = o        # NOT divided by l
        lse_partial[:, :, split_id] = m + log(l) # log-sum-exp

    # Phase 2: Reduction
    # Grid: (1, B * H)
    O_final = reduce_splits(O_partial, lse_partial)

    return O_final


def reduce_splits(O_partial, lse_partial):
    """
    Combine partial attention outputs using log-sum-exp correction.
    Mathematically exact (up to floating-point associativity).
    """
    B, H, num_splits, d = O_partial.shape

    O_final = zeros(B, H, 1, d)

    for b, h in parallel(B, H):  # parallelized across thread blocks
        # Step 1: Find global log-sum-exp
        global_lse = -inf
        for i in range(num_splits):
            global_lse = logaddexp(global_lse, lse_partial[b, h, i])

        # Step 2: Weighted combination
        o = zeros(d)
        for i in range(num_splits):
            # Weight = exp(local_lse - global_lse)
            # This rescales each partial output to the global softmax scale
            weight = exp(lse_partial[b, h, i] - global_lse)
            o += weight * O_partial[b, h, i, :]

        O_final[b, h, 0, :] = o

    return O_final
```

### 2.3 Mathematical Correctness

The split-KV reduction is exact because softmax is a monotone function that can be decomposed:

```
For the full sequence: O = softmax(Q @ K^T) @ V
                       O = sum_i(exp(s_i - m)) / sum_i(exp(s_i - m)) * V

Splitting into blocks:
  Each block i computes:
    O_i = sum_{j in block_i} exp(s_j - m_i) * V_j   (unnormalized)
    lse_i = m_i + log(sum_{j in block_i} exp(s_j - m_i))

  Global combination:
    global_lse = logsumexp(lse_0, lse_1, ..., lse_{K-1})
    O = sum_i exp(lse_i - global_lse) * O_i

  This is algebraically identical to computing softmax over the full sequence.
```

The only source of numerical difference is floating-point associativity -- the order of additions differs. In practice, the difference is negligible (within 1-2 ULP for FP32 accumulators).

---

## 3. Adaptive Split Selection

### 3.1 Choosing num_splits

The optimal number of splits depends on the GPU architecture and problem dimensions:

```python
def choose_num_splits(B, H, S, gpu_info, BLOCK_KV=256):
    """
    Heuristic for choosing num_splits for decode attention.

    Args:
        B: batch size
        H: number of attention heads (KV heads for GQA)
        S: KV sequence length
        gpu_info: GPU specifications (num_SMs, etc.)
        BLOCK_KV: KV block size per tile
    """
    num_SMs = gpu_info.num_SMs
    existing_parallelism = B * H

    # Already enough thread blocks to saturate GPU
    if existing_parallelism >= 2 * num_SMs:
        return 1

    # Target: 2-4x oversubscription for latency hiding
    target_blocks = num_SMs * 2
    num_splits = max(1, ceil(target_blocks / existing_parallelism))

    # Limit by available KV blocks
    max_splits = ceil(S / BLOCK_KV)
    num_splits = min(num_splits, max_splits)

    # Don't split too finely (reduction overhead)
    # Each split should process at least 2-4 KV blocks
    min_kv_per_split = 4 * BLOCK_KV
    max_useful_splits = max(1, S // min_kv_per_split)
    num_splits = min(num_splits, max_useful_splits)

    return num_splits

# Examples:
# B=1, H=32, S=32768, A100 (108 SMs):
#   existing = 32, target = 216
#   num_splits = ceil(216/32) = 7
#   Total blocks = 32 * 7 = 224 (good utilization)

# B=1, H=8 (GQA), S=131072, H100 (132 SMs):
#   existing = 8, target = 264
#   num_splits = ceil(264/8) = 33
#   Total blocks = 8 * 33 = 264 (good utilization)

# B=32, H=32, S=4096, A100 (108 SMs):
#   existing = 1024 >> 216
#   num_splits = 1 (no splitting needed)
```

### 3.2 Split Selection Strategies in Practice

**FlashInfer**: Uses a heuristic based on SM count and batch*heads, with configurable override.

**vLLM**: Implements `get_num_splits` function that considers both sequence length and batch size, with architecture-specific defaults.

**FlashMLA**: Uses fixed split sizes tuned for DeepSeek's MLA architecture (512 tokens per split typical).

**FlashDecoding++**: Proposes a flat GEMM approach that avoids explicit split selection by reformulating as a batched matrix multiply.

---

## 4. Implementation Variants

### 4.1 Two-Kernel Approach (Standard FlashDecoding)

```
Kernel 1: split_kv_attention
  Grid: (num_splits, B * H)
  Shared memory: Q tile + K/V tiles
  Output: O_partial [B, H, num_splits, d], lse_partial [B, H, num_splits]

Kernel 2: reduce_splits
  Grid: (1, B * H)
  Input: O_partial, lse_partial
  Output: O_final [B, H, 1, d]
```

**Pros**: Simple, clean separation of concerns, easy to tune each kernel independently.
**Cons**: Two kernel launches (minor overhead), intermediate O_partial buffer in global memory.

### 4.2 Single-Kernel with Cooperative Groups

Using CUDA cooperative groups, the partial attention and reduction can be fused into a single kernel:

```cpp
// Phase 1: Each split block computes partial attention
compute_partial_attention(Q, K, V, &local_O, &local_lse, split_id);

// Store partial results to global memory
O_partial[split_id] = local_O;
lse_partial[split_id] = local_lse;

// Grid-level sync (requires cooperative launch)
grid.sync();

// Phase 2: First thread block per (b, h) does reduction
if (split_id == 0) {
    reduce_splits(O_partial, lse_partial, O_final, num_splits);
}
```

**Pros**: Single kernel launch, no intermediate buffer.
**Cons**: Requires cooperative launch (limits grid size), grid.sync is expensive.

### 4.3 FlashDecoding++ (Flat GEMM Approach)

FlashDecoding++ reformulates the problem to avoid the explicit reduction kernel:

1. Approximate the global maximum as a constant (e.g., from a small precomputation)
2. Use this approximate max for all splits, making partial outputs directly summable
3. Combine via a simple summation (like a batched GEMM reduction)

```python
# FlashDecoding++ approach
m_approx = precompute_approximate_max(Q, K)  # lightweight, can use a subset of K

for split_id in parallel(num_splits):
    # Use shared approximate max instead of local max
    s = Q @ K_split.T * (1 / sqrt(d))
    p = exp(s - m_approx)  # all splits use same max -> outputs are on same scale
    o_split = p @ V_split

    # Partial outputs can be summed directly (no rescaling needed)
    atomicAdd(O_final, o_split)
    atomicAdd(normalizer, sum(p))

O_final /= normalizer
```

**Pros**: No explicit reduction kernel, potentially better for very large num_splits.
**Cons**: Approximate max introduces slight numerical differences, atomic additions can be a bottleneck.

### 4.4 Paged KV Cache Integration

Split-KV scheduling works naturally with paged (block) KV caches used in serving systems:

```python
# vLLM / FlashInfer paged KV cache
# KV cache is stored in non-contiguous pages: page_table[seq][page_idx] -> physical_block

def split_kv_paged_attention(Q, kv_cache, page_table, seq_lens, num_splits):
    for split_id in parallel(num_splits):
        # Determine which pages this split covers
        token_start = split_id * split_size
        token_end = min(token_start + split_size, seq_len)

        for token_idx in range(token_start, token_end, PAGE_SIZE):
            page_idx = token_idx // PAGE_SIZE
            physical_block = page_table[seq][page_idx]
            k_page = kv_cache.k[physical_block]
            v_page = kv_cache.v[physical_block]

            # Standard attention on this page
            s = Q @ k_page.T * (1 / sqrt(d))
            # ... online softmax update ...
```

The page boundaries provide natural split points. FlashInfer's `BatchDecodeWithPagedKVCacheWrapper` uses this approach.

---

## 5. Performance Analysis

### 5.1 FlashDecoding Results (A100, FP16)

From the original FlashDecoding blog post (Tri Dao et al., 2023):

| Configuration | Standard FA2 (ms) | FlashDecoding (ms) | Speedup |
|---|---|---|---|
| B=1, H=32, S=8K, d=128 | 2.1 | 0.5 | 4.2x |
| B=1, H=32, S=32K, d=128 | 8.3 | 1.1 | 7.5x |
| B=1, H=32, S=64K, d=128 | 16.5 | 2.0 | 8.3x |
| B=1, H=32, S=128K, d=128 | 33.0 | 3.8 | 8.7x |

Speedup increases with sequence length because longer sequences provide more splitting opportunity and the reduction overhead remains constant.

### 5.2 Bandwidth Analysis

For decode attention (Q has 1 token):

```
Standard FA2: Load Q once, stream K/V sequentially
  Total HBM reads: B*H*S*d * 2 * sizeof(fp16)  (K and V)
  Active SMs: B*H (e.g., 32 for B=1, H=32)
  Effective bandwidth: 32/108 * peak_bandwidth = 0.30 * 3.35 TB/s = 1.0 TB/s

Split-KV with num_splits=7:
  Total HBM reads: B*H*S*d * 2 * sizeof(fp16)  (same total K/V bytes)
  Active SMs: B*H*num_splits = 224
  Effective bandwidth: 224/108 * peak (capped at peak) ≈ 3.35 TB/s

  Plus reduction: B*H*num_splits*d * sizeof(fp32) = negligible
```

Split-KV achieves ~3.4x more effective HBM bandwidth utilization in this example by saturating all SMs.

### 5.3 Reduction Overhead

The reduction kernel processes `num_splits * d` elements per (b, h) pair:

```
Reduction data volume per (b, h):
  Read: num_splits * d * sizeof(fp32) (partial outputs)
       + num_splits * sizeof(fp32) (lse values)
  Write: d * sizeof(fp32) (final output)

Example: num_splits=7, d=128
  Read: 7 * 128 * 4 + 7 * 4 = 3612 bytes
  Write: 128 * 4 = 512 bytes
  Total: ~4 KB per (b, h) -- negligible
```

For B=1, H=32: total reduction data is ~128 KB, completing in microseconds. The reduction overhead is typically <2% of total kernel time for reasonable num_splits.

---

## 6. Interaction with Other Optimizations

### 6.1 Split-KV + GQA (Grouped-Query Attention)

GQA reduces the number of KV heads (e.g., from 32 to 8), making the scheduling problem even worse:

```
Llama 3 (70B): H_q = 64, H_kv = 8
Without split-KV, B=1: 8 thread blocks (6% of H100 SMs)
With split-KV, num_splits=17: 136 thread blocks (103% of H100 SMs)
```

Split-KV is especially critical for GQA models.

### 6.2 Split-KV + FP8/FP4 KV Cache

Quantized KV caches reduce memory bandwidth per token:

```
FP16 KV: 2 * d * 2 bytes = 512 bytes per token (d=128)
FP8 KV:  2 * d * 1 byte  = 256 bytes per token
FP4 KV:  2 * d * 0.5     = 128 bytes per token
```

With quantized KV, each split processes more tokens per unit of bandwidth, shifting the bottleneck from memory to compute earlier. Fewer splits may be optimal with quantized KV.

### 6.3 Split-KV + Speculative Decoding

Speculative decoding generates multiple candidate tokens (e.g., 4-8) simultaneously, increasing Q length:

```
Standard decode: Q_len = 1
Speculative decode: Q_len = 4-8 (k candidates)
```

With Q_len > 1, some Q-dimension parallelism returns, reducing the need for KV splitting. Adaptive split selection should account for Q_len:

```python
existing_parallelism = B * H * ceil(Q_len / BLOCK_M)
```

### 6.4 Split-KV + Multi-Head Latent Attention (MLA)

DeepSeek's MLA stores compressed latent vectors instead of full K/V. FlashMLA uses split-KV as a core component:

```
FlashMLA decode:
1. Split compressed KV cache across blocks
2. Each split: decompress K,V from latent + compute partial attention
3. Reduce with log-sum-exp correction

FlashMLA achieves 660 TFLOPS / 3000 GB/s on H800 with split-KV decode.
```

---

## 7. Source Code Patterns

### 7.1 FlashInfer Split-KV Decode

FlashInfer's batch decode implements split-KV with flexible configuration:

```cpp
// From FlashInfer attention kernel (simplified)
template <typename T, int HEAD_DIM, int BLOCK_KV>
__global__ void BatchDecodeWithPagedKVCacheSplitKV(
    T* __restrict__ q,           // [B, H, d]
    T* __restrict__ kv_cache,    // paged KV cache
    int* __restrict__ page_table, // page indirection
    float* __restrict__ o_partial, // [B, H, num_splits, d]
    float* __restrict__ lse_partial, // [B, H, num_splits]
    int seq_len,
    int num_splits)
{
    const int split_id = blockIdx.x;
    const int bh_id = blockIdx.y;  // b * H + h

    const int split_size = (seq_len + num_splits - 1) / num_splits;
    const int kv_start = split_id * split_size;
    const int kv_end = min(kv_start + split_size, seq_len);

    // Load Q to shared memory (1 row of d elements)
    __shared__ T q_smem[HEAD_DIM];
    load_q_to_smem(q, q_smem, bh_id);
    __syncthreads();

    // Online softmax over this KV range
    float m = -INFINITY;
    float l = 0.0f;
    float o[HEAD_DIM / WARP_SIZE] = {0};  // register-resident output

    for (int kv_offset = kv_start; kv_offset < kv_end; kv_offset += BLOCK_KV) {
        // Load K, V tiles from paged cache
        T k_tile[BLOCK_KV][HEAD_DIM];  // simplified -- actually in SMEM
        T v_tile[BLOCK_KV][HEAD_DIM];
        load_paged_kv(kv_cache, page_table, kv_offset, k_tile, v_tile);

        // Compute attention scores: Q @ K^T
        float s[BLOCK_KV];
        compute_qk(q_smem, k_tile, s);  // uses tensor cores or VFMA

        // Online softmax update
        float m_new = m;
        for (int i = 0; i < BLOCK_KV; i++) {
            m_new = fmaxf(m_new, s[i]);
        }
        float alpha = expf(m - m_new);
        l = alpha * l;
        m = m_new;

        float p[BLOCK_KV];
        for (int i = 0; i < BLOCK_KV; i++) {
            p[i] = expf(s[i] - m);
            l += p[i];
        }

        // O += P @ V
        accumulate_pv(p, v_tile, o, alpha);
    }

    // Store partial results
    store_partial_output(o_partial, lse_partial, o, m, l, bh_id, split_id);
}

// Reduction kernel
template <int HEAD_DIM>
__global__ void MergeSplitKVOutputs(
    float* __restrict__ o_partial,  // [B, H, num_splits, d]
    float* __restrict__ lse_partial, // [B, H, num_splits]
    half* __restrict__ o_final,      // [B, H, d]
    int num_splits)
{
    const int bh_id = blockIdx.x;

    // Find global LSE
    float global_lse = -INFINITY;
    for (int i = 0; i < num_splits; i++) {
        global_lse = logaddexpf(global_lse, lse_partial[bh_id * num_splits + i]);
    }

    // Weighted combination
    float o[HEAD_DIM] = {0};
    for (int i = 0; i < num_splits; i++) {
        float weight = expf(lse_partial[bh_id * num_splits + i] - global_lse);
        for (int j = 0; j < HEAD_DIM; j++) {
            o[j] += weight * o_partial[bh_id * num_splits * HEAD_DIM + i * HEAD_DIM + j];
        }
    }

    // Store final output
    for (int j = 0; j < HEAD_DIM; j++) {
        o_final[bh_id * HEAD_DIM + j] = __float2half(o[j]);
    }
}
```

### 7.2 PyTorch-Level API

```python
import flashinfer

# Setup decode handler with split-KV
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer,
    kv_layout="NHD",
)

# Plan: determines num_splits and workspace allocation
decode_wrapper.plan(
    indptr=kv_indptr,
    indices=kv_indices,
    last_page_len=kv_last_page_len,
    num_qo_heads=num_q_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    page_size=page_size,
)

# Run: executes split-KV attention + reduction
output = decode_wrapper.run(q, kv_data)
```

---

## 8. Architecture-Specific Considerations

### 8.1 NVIDIA A100 (Ampere)

- 108 SMs, no TMA, HMMA tensor cores
- Split-KV is critical for B=1 decode
- Each split uses standard HMMA-based FlashAttention-2
- Optimal num_splits: typically 4-8 for common configurations

### 8.2 NVIDIA H100 (Hopper)

- 132 SMs, TMA + WGMMA, higher compute throughput
- Split-KV combines with warp-specialized FA3 within each split
- TMA loads KV tiles; WGMMA computes attention
- Higher SM count means more splits needed for saturation

### 8.3 NVIDIA B200 (Blackwell)

- 192 SMs (approx), tcgen05.mma + TMEM
- Even more SMs to saturate, making split-KV more important
- FA4's 5-warp-role pipeline operates within each split

### 8.4 AMD MI300X (CDNA3)

- 304 CUs, no TMA, MFMA instructions
- Most severe underutilization without split-KV (B=1, H=32 uses only 11% of CUs)
- Larger L2 cache (256 MB) helps with KV reuse across splits on the same XCD
- Recommended: num_stages=1 within each split (no TMA for multi-stage pipelining)

---

## 9. Comparison with Prefill Scheduling

| Aspect | Prefill | Decode |
|--------|---------|--------|
| Q length | S_q (hundreds to thousands) | 1 (or few with speculative) |
| Parallelism source | Q tiles x B x H | B x H (insufficient) |
| KV sequence | Same length as Q | Much longer than Q |
| Bottleneck | Compute (GEMM) | Memory (HBM bandwidth) |
| Split-KV needed? | Rarely (Q provides parallelism) | Almost always (B=1) |
| Arithmetic intensity | High (~128 FLOP/byte) | Low (~0.5 FLOP/byte) |
| Optimal tile size | Large (128x128) | Small Q tile, large KV tile |

---

## 10. Future Directions

### 10.1 Dynamic Split Adjustment

Current implementations use a fixed num_splits determined at plan time. Future systems could dynamically adjust splits per-sequence based on actual KV cache length in a batch with mixed sequence lengths.

### 10.2 Hierarchical Splitting

For extremely long contexts (1M+ tokens), two-level splitting may be beneficial:
1. First level: coarse splits across SM clusters
2. Second level: fine splits within each SM cluster
This reduces the reduction tree depth from O(num_splits) to O(sqrt(num_splits)).

### 10.3 Fused Split-KV with Quantized Attention

Fusing KV dequantization (FP8/FP4 to FP16) directly into the split-KV kernel eliminates an extra memory pass and further improves decode throughput.

---

## References

- [FlashDecoding: Faster LLM Inference (Stanford CRFM Blog, 2023)](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- [FlashDecoding++: Faster LLM Inference on GPUs (Hong et al., 2024)](https://arxiv.org/abs/2311.01282)
- [FlashInfer: Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer)
- [FlashMLA GitHub (DeepSeek)](https://github.com/deepseek-ai/FlashMLA)
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://github.com/vllm-project/vllm)
- [FlashAttention-2 Paper (Dao, 2023)](https://arxiv.org/abs/2307.08691)
- [Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)
