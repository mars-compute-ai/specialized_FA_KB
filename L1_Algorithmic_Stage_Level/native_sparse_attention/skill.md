---
skill_name: Native Sparse Attention (NSA)
description: Hardware-aligned sparse attention algorithm using a three-branch design (local + compressed + selected) with natively trainable token selection to achieve sub-quadratic complexity while preserving model quality through end-to-end differentiable sparsity.
level: L1 - Algorithmic/Mathematical Stage Level
target_hardware: NVIDIA GPUs (Hopper H100/H800 and newer with Tensor Cores)
relevance: When building or optimizing long-context LLMs (32K-64K+ tokens) that need to reduce attention compute and memory costs below O(N^2) without sacrificing training quality or requiring post-hoc sparsity patterns.
---

# Native Sparse Attention (NSA)

## What It Is
Native Sparse Attention (NSA) is a sparse attention mechanism introduced by DeepSeek (Jingyang Yuan et al., 2025) that reduces the quadratic cost of full attention by decomposing it into three hardware-friendly branches: (1) a **local sliding-window** branch for nearby tokens, (2) a **compressed-token** branch that summarizes distant context via learned compression, and (3) a **selected-block** branch that uses a trainable top-k gating mechanism to pick the most relevant coarse-grained KV blocks. Unlike post-hoc sparsity methods (e.g., applying a fixed sparse mask after training with dense attention), NSA integrates sparsity directly into pretraining so the model learns to leverage the sparse pattern natively. The block-level granularity of all three branches is deliberately aligned with GPU memory access patterns (contiguous 64-token or 128-token blocks) to maximize Tensor Core utilization and minimize scattered memory access.

## Key Concepts
- **Three-Branch Decomposition:** Attention output is a weighted combination of three branches: O = gate_local * O_local + gate_compress * O_compress + gate_select * O_select, where gates are sigmoid-activated learned scalars per head.
- **Local Branch (Sliding Window):** Attends to the most recent w tokens (e.g., w = 512-1024). This captures short-range dependencies and is trivially hardware-efficient since tokens are contiguous in memory.
- **Compressed Branch:** Groups past KV tokens into non-overlapping blocks of size l (e.g., l = 64), then compresses each block into a single representative token using a learned linear projection (MLP or weighted average). The query then attends to these ~N/l compressed tokens, reducing cost from O(N) to O(N/l) per query.
- **Selected Branch (Top-k Block Selection):** Divides the KV sequence into coarse blocks of size b (e.g., b = 64). A lightweight scoring function (e.g., dot product of query with block-level key summary) ranks all blocks, and only the top-k blocks (e.g., k = 16-32) are loaded for full-resolution attention. This provides focused long-range retrieval at O(k*b) cost.
- **Natively Trainable:** The selection mechanism uses straight-through estimators or Gumbel-softmax to remain differentiable during training, so the model learns which blocks matter during pretraining rather than relying on heuristics.
- **Hardware Alignment:** All branches operate on contiguous, power-of-2 block sizes that map directly to Tensor Core tile dimensions (e.g., 64x64 or 128x128), avoiding the irregular memory access patterns that plague unstructured sparsity.

## Algorithm / Pseudo-code
```
# NSA Forward Pass (per attention head, per query position t)
# Inputs: Q[t] (1 x d), K (N x d), V (N x d)
# Hyperparameters: window w, compress_block l, select_block b, top_k k

# ---- Branch 1: Local Sliding Window ----
K_local = K[max(0, t-w) : t]          # (w x d)
V_local = V[max(0, t-w) : t]          # (w x d)
scores_local = Q[t] @ K_local^T / sqrt(d)
O_local = softmax(scores_local) @ V_local

# ---- Branch 2: Compressed Tokens ----
num_blocks_c = t // l
for i in 1..num_blocks_c:
    # Compress block of l tokens into 1 representative token
    K_block = K[(i-1)*l : i*l]         # (l x d)
    V_block = V[(i-1)*l : i*l]         # (l x d)
    K_compressed[i] = compress_fn(K_block)  # (1 x d), learned MLP/linear
    V_compressed[i] = compress_fn(V_block)  # (1 x d)

scores_compress = Q[t] @ K_compressed^T / sqrt(d)
O_compress = softmax(scores_compress) @ V_compressed

# ---- Branch 3: Selected Blocks (Top-k) ----
num_blocks_s = t // b
# Compute coarse block scores using block-level key means
for i in 1..num_blocks_s:
    K_block_mean[i] = mean(K[(i-1)*b : i*b])  # (1 x d)
block_scores = Q[t] @ K_block_mean^T           # (num_blocks_s,)
top_k_indices = argtop_k(block_scores, k)

# Gather selected blocks and compute full-resolution attention
K_selected = gather_blocks(K, top_k_indices, b)  # (k*b x d)
V_selected = gather_blocks(V, top_k_indices, b)  # (k*b x d)
scores_select = Q[t] @ K_selected^T / sqrt(d)
O_select = softmax(scores_select) @ V_selected

# ---- Gated Combination ----
g_local   = sigmoid(W_local @ Q[t])    # scalar gate
g_compress = sigmoid(W_compress @ Q[t]) # scalar gate
g_select  = sigmoid(W_select @ Q[t])   # scalar gate

O[t] = g_local * O_local + g_compress * O_compress + g_select * O_select
```

## When to Use
- Pretraining or fine-tuning LLMs on very long contexts (32K-128K+ tokens) where full O(N^2) attention is cost-prohibitive
- When you need sparse attention that is trained end-to-end (not applied as a post-hoc optimization)
- Scenarios requiring both strong local coherence and long-range retrieval (e.g., document QA, code understanding)
- When hardware efficiency is critical: the block-aligned design maps well to modern GPU Tensor Cores and avoids the poor utilization of unstructured sparsity
- Prefill-heavy workloads where reducing the KV tokens attended per query directly reduces latency
- When you want to combine the strengths of sliding-window, compressed, and retrieval-based attention in a single unified mechanism

## When NOT to Use
- Short sequences (< 2K tokens) where full dense FlashAttention is already fast and the overhead of three branches adds unnecessary complexity
- When bit-exact compatibility with dense attention is required (NSA is an approximation; it does not produce identical outputs to full attention)
- Decode-only inference with very small batch sizes where the KV cache is small enough that PagedAttention + dense FlashDecoding is already memory-bandwidth-bound (not compute-bound)
- Models that have already been pretrained with dense attention and cannot be retrained (NSA requires training with the sparse pattern to learn effective selection)
- When using hardware without efficient block-sparse primitives (e.g., older GPUs without good Tensor Core utilization for variable block sizes)

## Key Takeaways
- NSA achieves comparable perplexity to dense attention on pretraining benchmarks while reducing attention FLOPs by 6-10x on long sequences (64K tokens)
- The three-branch design is not arbitrary: local captures recency, compressed captures global summary, and selected captures specific long-range dependencies -- together they cover the attention patterns empirically observed in trained dense models
- Hardware alignment is essential: block-sparse patterns with power-of-2 block sizes (64, 128) achieve 80-90% of dense Tensor Core throughput, while unstructured sparsity often achieves < 30%
- The gating mechanism allows each head to dynamically weight the three branches, enabling specialization (some heads become primarily local, others primarily retrieval-focused)
- NSA was validated in DeepSeek's production training pipeline, demonstrating practical scalability beyond academic benchmarks
- Kernel implementation uses FlashAttention-style tiling within each branch, so NSA benefits from all FA optimizations (online softmax, recomputation, etc.)

## References
- Original Paper: Jingyang Yuan, Huazuo Gao, et al., "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" (DeepSeek-AI, arXiv:2502.11089, February 2025)
- DeepSeek-V3 Technical Report (December 2024) -- describes the production context where NSA-style ideas were developed
- Related: BigBird (Zaheer et al., NeurIPS 2020) and Longformer (Beltagy et al., 2020) for earlier block-sparse attention approaches
- Code: https://github.com/deepseek-ai/Native-Sparse-Attention (reference implementation)
