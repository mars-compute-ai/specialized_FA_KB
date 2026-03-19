---
skill_name: FlashAttention-2 Warp Partitioning and Sequence-Parallel Scheduling
description: Splitting Q across warps instead of K/V to eliminate synchronization, plus parallelizing along sequence length for high occupancy
level: L2 - Scheduling & Pipelining Level
target_hardware: NVIDIA A100, H100 (Ampere, Hopper)
relevance: When designing attention kernel work distribution across warps and thread blocks, especially for long-context or low-batch scenarios
---

# FlashAttention-2 Warp Partitioning and Sequence-Parallel Scheduling

## What It Is
FlashAttention-2 redesigns how attention computation is distributed across GPU warps and thread blocks. The key insight is splitting Q (queries) across warps instead of K/V, which eliminates inter-warp synchronization and shared memory communication. Additionally, it parallelizes along the sequence length dimension (not just batch and heads), which is critical for maintaining high GPU occupancy during long-context inference with small batch sizes.

## Key Concepts
- **Sliced-Q warp partitioning**: Each of 4 warps gets a slice of Q and independently computes its portion of the output. No cross-warp reduction or shared memory writes needed for intermediate results.
- **Sliced-K eliminated**: The original FA1 split K/V across warps, requiring all warps to synchronize and accumulate partial softmax results through shared memory.
- **Sequence-length parallelism**: Grid is (batch, heads, num_q_tiles) instead of just (batch, heads), so more thread blocks are launched when batch*heads is small.
- **Loop order swap**: FA2 uses outer-Q/inner-KV loop order (vs. FA1's outer-KV/inner-Q), enabling each thread block to own one Q tile and write one output tile without cross-block communication.
- **Matmul priority**: A100 delivers 312 TFLOPs/s for matmul vs. 19.5 TFLOPs/s for non-matmul FP32 (16x gap). FA2 minimizes non-matmul operations in the critical path.
- **Shared memory reuse**: K/V tiles are loaded once into shared memory and reused across all warp-local Q computations within a thread block.

## Scheduling Strategy / Pseudo-code
```
# Grid dimensions
grid = (ceil(seq_len_q / TILE_Q),   # parallelize over Q tiles
        num_heads,                    # parallelize over heads
        batch_size)                   # parallelize over batch

# Within each thread block (4 warps):
# Each warp owns a slice of Q_tile (TILE_Q / 4 rows each)

thread_block(block_q_idx, head_idx, batch_idx):
    # Load this block's Q tile slice into registers (split across warps)
    Q_local = load_Q[batch_idx, head_idx, block_q_idx * TILE_Q : (block_q_idx+1) * TILE_Q]
    # Warp 0 gets rows 0..TILE_Q/4-1, Warp 1 gets TILE_Q/4..TILE_Q/2-1, etc.

    acc = zeros(TILE_Q / 4, head_dim)  # per-warp accumulator
    m = -inf(TILE_Q / 4)               # per-warp row-max
    l = zeros(TILE_Q / 4)              # per-warp row-sum

    for kv_tile_idx in range(num_kv_tiles):
        # ALL warps load the SAME K/V tile into shared memory
        K_shared = load_K[batch_idx, head_idx, kv_tile_idx * TILE_KV : ...]
        V_shared = load_V[batch_idx, head_idx, kv_tile_idx * TILE_KV : ...]
        __syncthreads()

        # Each warp computes its Q_slice @ K^T independently
        S_local = Q_local_warp @ K_shared.T   # no cross-warp communication

        # Online softmax update (per-warp, no sync needed)
        m_new = max(m, rowmax(S_local))
        P_local = exp(S_local - m_new)
        l = exp(m - m_new) * l + rowsum(P_local)
        acc = exp(m - m_new) * acc + P_local @ V_shared
        m = m_new

    # Final output: each warp writes its slice directly to global memory
    O[batch_idx, head_idx, warp_q_rows] = acc / l
    # NO cross-warp reduction needed
```

## Performance Impact
- **2x speedup** over FlashAttention-1 on attention microbenchmarks (A100, BF16)
- **Up to 9x** faster than standard PyTorch attention
- **230 TFLOPs/s** on A100 (73% of theoretical 312 TFLOPs/s peak for matmul)
- **335 TFLOPs/s** on H100 (without TMA or 4th-gen Tensor Core optimizations)
- **72% model FLOP utilization** in end-to-end GPT-3 training (A100, 8K context)
- Enables 2x longer context at the same training cost (16K vs 8K)
- 1.3x end-to-end training speedup over FA1

## When to Use
- Any attention kernel where occupancy matters (most production scenarios)
- Long-context models (4K-128K+) where sequence-length parallelism is essential
- Low-batch inference (batch=1) where `batch * heads` is small
- Multi-Query Attention (MQA) or Grouped-Query Attention (GQA) where head count is reduced
- When designing new attention variants and choosing warp-level work distribution

## When NOT to Use
- When using pre-built FlashAttention library (these optimizations are already applied)
- Very short sequences (< 256) where the overhead of tiling may exceed benefits
- When targeting non-NVIDIA hardware (the warp-level strategy is NVIDIA-specific)
- When memory bandwidth is the sole bottleneck (e.g., extremely large head dimensions where compute is trivial)

## Key Takeaways
- Splitting Q across warps (not K/V) is the fundamental scheduling insight: it eliminates synchronization barriers and shared memory traffic for intermediate results
- Sequence-length parallelism is critical when batch_size * num_heads < number of SMs (~108 on A100, ~132 on H100)
- The 16x throughput gap between matmul and non-matmul operations on Tensor Core GPUs means every non-matmul instruction in the inner loop is extremely costly
- Loop order matters: outer-Q/inner-KV enables each thread block to independently produce one output tile
- The forward and backward passes have different optimal parallelization strategies (forward: parallelize over Q; backward: parallelize over K/V)

## References
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Hazy Research Blog)](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)
- [FlashAttention-2 Paper (Dao, 2023)](https://arxiv.org/abs/2307.08691)
