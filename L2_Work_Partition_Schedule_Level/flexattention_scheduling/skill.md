---
skill_name: FlexAttention Block-Sparse Scheduling and Score Modification
description: Block-sparse iteration with runtime sparsity encoding and user-defined score modifications for flexible attention scheduling
level: L2 - Work-Partition/Schedule Level
target_hardware: NVIDIA Hopper H100, Blackwell GB200
relevance: When implementing custom attention patterns (sliding window, document masking, ALiBi, etc.) that need to skip masked blocks for efficiency
---

# FlexAttention Block-Sparse Scheduling and Score Modification

## What It Is
FlexAttention is a PyTorch API that combines FlashAttention's fused kernel approach with flexible, user-defined attention patterns. Its scheduling innovation is block-sparse iteration: the computation grid is organized into (query_block, kv_block) pairs, and a block mask encodes which pairs to compute. Fully masked blocks are skipped entirely in both forward and backward passes. With the new FlashAttention-4 backend, this achieves 1.2-3.2x speedup over the Triton implementation on Hopper, and near-parity with cuDNN on Blackwell.

## Key Concepts
- **Block-sparse iteration**: Queries grouped into blocks of Br, keys/values into blocks of Bc. A precomputed block mask determines which (q_block, kv_block) pairs require computation.
- **Runtime data-dependent sparsity**: Block masks can change at runtime based on input data, unlike static sparsity patterns compiled ahead of time.
- **score_mod interface**: A Python function `score_mod(score, b_idx, h_idx, q_idx, kv_idx)` modifies pre-softmax attention scores, enabling ALiBi, sliding window, document masking, soft-capping, and arbitrary combinations.
- **CuTe DSL backend**: FlashAttention-4 uses NVIDIA CuTe for hardware-aware scheduling, memory layout, and warp specialization that Triton cannot express.
- **Warp specialization on Blackwell**: FA4 employs deeply pipelined, warp-specialized kernels to keep tensor cores continuously busy.
- **JIT instantiation**: score_mod functions are JIT-compiled into the fused kernel, avoiding the overhead of separate mask computation.

## Scheduling Strategy / Pseudo-code
```
# Phase 1: Build block mask (Python, on CPU or GPU)
block_mask = create_block_mask(mask_fn, B, H, seq_len_q, seq_len_kv,
                                block_size_q=Br, block_size_kv=Bc)
# block_mask[b, h, i, j] = True if q_block i should attend to kv_block j

# Phase 2: Kernel launch -- grid over non-masked blocks only
grid = (num_active_block_pairs, num_heads, batch_size)

# Within each thread block:
thread_block(pair_idx, head_idx, batch_idx):
    q_block_idx, kv_block_idx = active_pairs[pair_idx]

    Q_tile = load_Q[batch_idx, head_idx, q_block_idx * Br : (q_block_idx+1) * Br]
    K_tile = load_K[batch_idx, head_idx, kv_block_idx * Bc : (kv_block_idx+1) * Bc]
    V_tile = load_V[batch_idx, head_idx, kv_block_idx * Bc : (kv_block_idx+1) * Bc]

    # Compute attention scores
    S = Q_tile @ K_tile.T

    # Apply user-defined score modification (fused into kernel)
    for each element (i, j) in S:
        S[i,j] = score_mod(S[i,j], batch_idx, head_idx,
                           q_block_idx*Br + i, kv_block_idx*Bc + j)

    # Online softmax and accumulate into output
    update_online_softmax(S, V_tile, acc, m, l)

# Backward pass: same block-sparse structure, skip masked blocks
```

## Performance Impact
- **1.2x to 3.2x speedup** over Triton FlexAttention on Hopper H100 (with FA4 backend)
- **Near-parity with cuDNN** on Blackwell GB200 (previously severe gap with Triton)
- Block-sparse skipping provides linear speedup proportional to sparsity (e.g., causal mask skips ~50% of blocks)
- Sliding window attention with window_size << seq_len can skip >90% of blocks
- Document masking with many short documents can achieve high sparsity ratios

## When to Use
- Implementing non-standard attention patterns (sliding window, document masking, ALiBi, prefix LM, etc.)
- When attention sparsity is high and block-level skipping provides significant savings
- Prototyping new attention variants without writing CUDA kernels
- When you need both flexibility and near-optimal performance on Hopper/Blackwell
- Combining multiple attention modifications (e.g., causal + sliding window + ALiBi)
- When sparsity patterns may change at runtime based on input data

## When NOT to Use
- Standard dense causal or bidirectional attention where FlashAttention-3/4 kernels are already optimal
- When targeting pre-Hopper GPUs (FA4 backend requires Hopper or Blackwell)
- When attention patterns do not have block-level sparsity (e.g., random per-element masking)
- Extremely latency-sensitive paths where JIT compilation overhead is unacceptable (first call compiles)
- When numerical behavior of score_mod must match a specific reference implementation exactly

## Key Takeaways
- Block-sparse scheduling is the key scheduling innovation: skip entire (q_block, kv_block) pairs that are fully masked, proportionally reducing compute
- The score_mod interface makes attention pattern customization a Python function rather than a CUDA kernel rewrite
- FlashAttention-4 backend with CuTe/warp specialization closes the performance gap between flexible and hand-tuned kernels
- Both forward and backward passes respect the same block-sparse structure, so gradients are also efficient
- `torch.compile` with `dynamic=False` and `BACKEND="FLASH"` is the recommended deployment configuration

## References
- [FlexAttention + FlashAttention-4: Fast and Flexible (PyTorch Blog)](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention (PyTorch Blog)](https://pytorch.org/blog/flexattention/)
- [FlashAttention GitHub Repository](https://github.com/dao-AILab/flash-attention)
