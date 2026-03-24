---
skill_name: Split-KV Decode Scheduling for Attention
description: Scheduling strategy that parallelizes decode-phase attention across KV sequence blocks with log-sum-exp reduction, enabling GPU-efficient single-query attention
level: L2 - Scheduling & Pipelining Level
target_hardware: NVIDIA GPUs (Ampere A100, Hopper H100, Blackwell B200), AMD MI300X
relevance: When optimizing decode-phase (autoregressive) attention where batch_size x num_heads >> sequence_length, or when standard FlashAttention leaves GPU SMs underutilized during token generation
---

# Split-KV Decode Scheduling for Attention

## What It Is
Split-KV decode scheduling (also called FlashDecoding or split-K attention) addresses a fundamental scheduling mismatch in autoregressive LLM inference: during the decode phase, each new token attends to the entire KV cache, but standard FlashAttention parallelizes only over batch and heads -- leaving most SMs idle when batch_size x num_heads is small relative to the number of SMs. Split-KV scheduling introduces an additional parallelism dimension by partitioning the KV sequence into blocks and assigning each block to a separate thread block. Each thread block computes a partial attention output with local softmax statistics (local max and log-sum-exp), and a lightweight reduction kernel combines partial results using the log-sum-exp trick. This transforms decode attention from an SM-underutilized, memory-bandwidth-wasted operation into one that fully saturates the GPU.

## Key Concepts
- **Decode-phase scheduling problem**: During autoregressive generation, Q has shape [B, H, 1, d] (single query token) while KV has shape [B, H, S, d] (full sequence). Standard FA parallelizes over B*H thread blocks. With B=1, H=32, only 32 thread blocks launch -- wasting 75%+ of SMs on an A100 (108 SMs) or 88%+ on an H100 (132 SMs).
- **Split-KV parallelism**: Partition the KV sequence dimension S into num_splits blocks of size S/num_splits. Each split processes independently: computes local QK^T, local softmax with local max m_i and local sum l_i, and local output O_i = softmax(Q @ K_i^T) @ V_i.
- **Log-sum-exp reduction**: Partial results {(O_i, m_i, l_i)} are combined via the online softmax correction: global_max = max(m_i), rescale each O_i by exp(m_i - global_max), sum weighted outputs, normalize by sum of rescaled l_i values.
- **FlashDecoding (Tri Dao et al., 2023)**: The original formulation applying split-KV to Flash Attention decode. Introduced as a blog post showing 8x speedup for long-context decode on A100.
- **FlashDecoding++ (Ke Hong et al., 2024)**: Refinement using a flat GEMM approach and unified max approximation to avoid the two-pass reduction.
- **Adaptive num_splits**: The optimal number of splits depends on B*H (existing parallelism), S (sequence length), and GPU SM count. Too few splits underutilize SMs; too many splits increase reduction overhead and reduce per-split arithmetic intensity.
- **Paged KV cache compatibility**: Split-KV works naturally with paged/block KV caches (vLLM, FlashInfer) since KV blocks are already non-contiguous; each split handles a range of page table entries.

## Scheduling Strategy / Pseudo-code
```
# SPLIT-KV DECODE ATTENTION

# Inputs:
#   Q: [B, H, 1, d]         -- single query token per sequence
#   K: [B, H, S, d]         -- full KV cache keys
#   V: [B, H, S, d]         -- full KV cache values
#   num_splits: int          -- KV sequence split factor

# Phase 1: Parallel partial attention (one thread block per split)
# Grid: (B * H * num_splits,)
# Each thread block handles: Q[b,h,0,:] attending to K[b,h, split_start:split_end, :]

kernel split_kv_attention(Q, K, V, O_partial, lse_partial, num_splits):
    b, h, split_id = decode_block_indices(blockIdx.x, H, num_splits)

    split_size = ceil(S / num_splits)
    kv_start = split_id * split_size
    kv_end   = min(kv_start + split_size, S)

    # Load Q tile to shared memory / registers (only 1 row)
    q = load_Q(Q[b, h, 0, :])

    # Standard FlashAttention inner loop over this KV range
    m_i = -inf          # running max
    l_i = 0.0           # running sum
    o_i = zeros(d)      # running output

    for kv_block in range(kv_start, kv_end, BLOCK_KV):
        k_tile = load_K(K[b, h, kv_block:kv_block+BLOCK_KV, :])
        v_tile = load_V(V[b, h, kv_block:kv_block+BLOCK_KV, :])

        # Attention scores
        s = q @ k_tile.T / sqrt(d)      # [1, BLOCK_KV]

        # Online softmax update
        m_new = max(m_i, rowmax(s))
        p = exp(s - m_new)               # [1, BLOCK_KV]
        alpha = exp(m_i - m_new)         # rescale factor
        l_i = alpha * l_i + rowsum(p)
        o_i = alpha * o_i + p @ v_tile   # [1, d]
        m_i = m_new

    # Store partial results (NOT normalized -- keep lse for reduction)
    O_partial[b, h, split_id, :] = o_i   # unnormalized partial output
    lse_partial[b, h, split_id] = m_i + log(l_i)   # log-sum-exp

# Phase 2: Reduction kernel (one thread block per B*H)
# Grid: (B * H,)

kernel reduce_splits(O_partial, lse_partial, O_final, num_splits):
    b, h = decode_block_indices(blockIdx.x, H)

    # Find global maximum log-sum-exp
    global_lse = -inf
    for i in range(num_splits):
        global_lse = logaddexp(global_lse, lse_partial[b, h, i])

    # Weighted combination
    o_final = zeros(d)
    for i in range(num_splits):
        weight = exp(lse_partial[b, h, i] - global_lse)
        o_final += weight * O_partial[b, h, i, :]

    O_final[b, h, 0, :] = o_final

# ADAPTIVE NUM_SPLITS SELECTION:
def choose_num_splits(B, H, S, num_SMs, BLOCK_KV=256):
    existing_parallelism = B * H
    kv_blocks = ceil(S / BLOCK_KV)

    if existing_parallelism >= num_SMs:
        return 1  # Already enough parallelism, no split needed

    # Target: enough thread blocks to fill all SMs
    # Each SM can run multiple thread blocks (typically 1-4 for attention)
    target_blocks = num_SMs * 2  # 2x oversubscription for latency hiding
    num_splits = ceil(target_blocks / existing_parallelism)
    num_splits = min(num_splits, kv_blocks)  # Can't split more than KV blocks

    return num_splits

# Example: B=1, H=32, S=32768, A100 (108 SMs)
# existing_parallelism = 32 (only 30% SM utilization)
# target = 216, num_splits = ceil(216/32) = 7
# Total thread blocks = 32 * 7 = 224 (>2x SM count, good utilization)
```

## Performance Impact
- **FlashDecoding on A100**: Up to 8x speedup for long-sequence decode (S=32K+) vs standard FlashAttention-2 decode
- **SM utilization improvement**: From 30% (B=1, H=32 on 108 SMs) to >95% with adaptive splitting
- **Memory bandwidth utilization**: Approaches peak HBM bandwidth since each split is independently memory-bound
- **Reduction overhead**: Typically <5% of total kernel time (reduction kernel processes num_splits x d elements per B*H, negligible vs main attention)
- **Latency reduction**: Near-linear scaling with num_splits until SM saturation, then diminishing returns from reduction overhead
- **FlashDecoding++**: Additional 10-30% over FlashDecoding via flat GEMM and reduced synchronization

## When to Use
- Autoregressive LLM decode/generation phase where Q has 1 token per sequence
- When B * num_heads < num_SMs (GPU is underutilized by standard attention scheduling)
- Long-context inference (S > 4K) where the KV sequence provides ample splitting opportunities
- Serving scenarios with small batch sizes but long contexts (e.g., B=1-4, S=32K-128K)
- With paged KV caches (vLLM, FlashInfer) where KV blocks are already non-contiguous
- When decode latency (time-to-next-token) is the primary optimization target

## When NOT to Use
- Prefill phase where Q has many tokens -- standard FlashAttention already has sufficient parallelism from the Q sequence dimension
- Large batch decode where B * H already saturates SMs (split adds overhead without benefit)
- Very short sequences (S < 256) where splitting creates too-small tiles with poor arithmetic intensity
- When the reduction kernel's extra global memory write/read becomes a significant fraction of total time (very small d, very large num_splits)
- Training forward/backward passes that already parallelize over the Q dimension

## Source Code Examples

### FlashInfer Split-KV Decode Kernel (C++)

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

### FlashInfer Python API for Split-KV Decode

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

## Key Takeaways
- The decode-phase attention scheduling problem is fundamentally different from prefill: Q is tiny (1 token) while KV is large, so parallelism must come from splitting KV, not Q
- Split-KV is mathematically exact -- the log-sum-exp reduction produces bit-identical results to non-split attention (up to floating-point associativity)
- Adaptive num_splits selection is critical: too few leaves SMs idle, too many wastes bandwidth on the reduction kernel
- The two-kernel pattern (parallel partial attention + reduction) is a general template applicable beyond attention to any operation needing parallel online softmax
- FlashInfer, FlashMLA, and vLLM all use split-KV scheduling as a core component of their decode pipelines
- Split-KV composes naturally with other optimizations: GQA (fewer KV heads), paged KV cache, FP8 KV quantization, and speculative decoding

## References
- [FlashDecoding Blog Post (Tri Dao et al., 2023)](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- [FlashDecoding++ Paper (Hong et al., 2024)](https://arxiv.org/abs/2311.01282)
- [FlashInfer: Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer)
- [FlashMLA: Split-KV for Multi-Head Latent Attention](https://github.com/deepseek-ai/FlashMLA)
- [vLLM PagedAttention](https://github.com/vllm-project/vllm)
