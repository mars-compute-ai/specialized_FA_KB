---
skill_name: FlashInfer Attention Engine for LLM Serving
description: Customizable attention engine providing optimized kernels for prefill/decode/append phases with diverse KV-Cache formats, GQA, fused RoPE, and quantized attention for LLM inference.
level: L1 - Algorithmic/Mathematical Stage Level
target_hardware: NVIDIA GPUs (A100 Ampere, H100 Hopper, RTX 4090/6000 Ada)
relevance: When building or optimizing LLM inference serving systems that need specialized attention kernels for different serving phases, KV-Cache formats, or quantization strategies.
---

# FlashInfer Attention Engine for LLM Serving

## What It Is
FlashInfer (Best Paper, MLSys 2025) is an open-source attention kernel library that addresses the diverse requirements of LLM inference serving. Unlike FlashAttention which provides a single general-purpose attention kernel, FlashInfer provides specialized kernels for three serving phases (prefill, decode, append), supports multiple KV-Cache storage formats (padded tensor, ragged tensor, page table), and includes optimizations like fused rotary positional embeddings, grouped-query attention with Tensor Cores, and 4/8-bit quantized KV-Cache attention. It achieves near-100% memory bandwidth utilization for decode and outperforms FlashAttention-2 for prefill.

## Key Concepts
- **Three Attention Phases:**
  - **Prefill:** All queries attend to KV-Cache simultaneously. Compute-bound, intensity ~O(l_q). Needs high FLOPS.
  - **Decode:** Single new token attends to full KV-Cache. Memory-bound, intensity ~O(1). Needs high bandwidth.
  - **Append:** Newly appended tokens (speculative decoding). Intensity between prefill and decode.
- **KV-Cache Format Diversity:**
  - **Padded Tensor:** Fixed-size, simple but wastes memory for variable-length sequences
  - **Ragged Tensor:** Variable-length, memory-efficient, requires offset arrays
  - **Page Table:** Block-allocated pages (like virtual memory), supports dynamic growth and sharing
- **GQA Tensor Core Decode:** Uses Tensor Cores for grouped-query attention during decode (normally compute-limited ops use CUDA cores). 2-3x speedup over vLLM.
- **Fused RoPE:** Applies rotary embeddings on-the-fly inside the attention kernel instead of pre-computing and storing them in the KV-Cache. Saves memory and enables cache pruning.
- **Split-K for Decode:** Parallelizes across the KV sequence dimension when batch size is small, increasing SM utilization.
- **Block-Sparse KV-Cache:** Handles heterogeneous KV-Cache layouts efficiently for multi-tenant serving.

## Algorithm / Pseudo-code
```
# FlashInfer Decode Attention (single query per request, batched)
# Memory-bandwidth bound: optimize for minimal HBM reads

for each request in batch (parallelized):
    q = load_query(request)                    # 1 x d
    page_indices = load_page_table(request)     # prefetched to SMEM

    # Split-K: partition KV across multiple thread blocks
    for each KV chunk (parallelized via Split-K):
        for each page in chunk:
            k_page, v_page = load_kv_page(page_indices[page])  # from paged KV-Cache
            # Optionally apply fused RoPE on-the-fly:
            k_page = apply_rope(k_page, position_ids[page])

            # Standard online softmax attention
            scores = q @ k_page^T / sqrt(d)
            update m, l, acc with online softmax

    # Reduce across Split-K chunks
    final_output = reduce_split_k(partial_outputs)

# FlashInfer Prefill Attention (all queries, compute-bound)
# Similar to FlashAttention-2 but with KV-Cache format awareness

for each request in batch:
    Q_request = load_queries(request)           # l_q x d (ragged)
    for each KV tile:
        K_tile, V_tile = load_kv_tile(request, tile_idx, format=PAGE_TABLE)
        # Standard tiled attention with online softmax
        S = Q_block @ K_tile^T
        update m, l, O with rescaling

# FlashInfer Append Attention (speculative decoding)
# Handles newly verified tokens, variable append lengths

for each request with appended tokens:
    Q_append = load_appended_queries(request)   # variable length
    # Attend to existing KV-Cache + newly appended KV
    # Uses ragged tensor format for variable-length batching
```

## When to Use
- Building LLM inference serving systems (vLLM, SGLang, TensorRT-LLM)
- When you need different attention strategies for prefill vs decode phases
- Multi-tenant serving with variable-length KV-Caches and page-based memory management
- When GQA decode performance is critical (Llama 2/3, Mistral)
- When KV-Cache quantization (INT4/INT8) is needed to serve longer contexts
- Speculative decoding workflows requiring append-mode attention

## When NOT to Use
- Training workloads where FlashAttention-2/3 provides optimized forward+backward passes
- When a single general-purpose attention kernel suffices (no serving-specific requirements)
- On non-NVIDIA hardware (AMD/TPU) -- FlashInfer currently targets CUDA
- When the model does not use standard softmax attention (e.g., linear attention, state-space models)

## Code / Pseudo-code

### Python: BatchPrefillWithPagedKVCacheWrapper Usage

From FlashInfer's `prefill.py` -- the primary API for batched prefill with paged KV-Cache:

```python
import torch
import flashinfer

num_layers = 32
num_qo_heads = 64
num_kv_heads = 16
head_dim = 128
max_num_pages = 128
page_size = 16

# Allocate 128MB workspace buffer
workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)

batch_size = 7
nnz_qo = 100
qo_indptr = torch.tensor(
    [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
)
paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
paged_kv_indptr = torch.tensor(
    [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
)
# 1 <= paged_kv_last_page_len <= page_size
paged_kv_last_page_len = torch.tensor(
    [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
)
q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
kv_cache_at_layer = torch.randn(
    num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim,
    dtype=torch.float16, device="cuda:0"
)

# Create auxiliary data structures for batch prefill attention
prefill_wrapper.plan(
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    causal=True,
)

# Run prefill across layers, reusing auxiliary data structures
outputs = []
for i in range(num_layers):
    q = q_at_layer[i]
    kv_cache = kv_cache_at_layer[i]
    o = prefill_wrapper.run(q, kv_cache)
    outputs.append(o)

# outputs[0].shape => torch.Size([100, 64, 128])
```

Key API pattern: `plan()` precomputes scheduling metadata once, then `run()` executes attention across layers reusing that metadata.

## Key Takeaways
- LLM inference has fundamentally different compute characteristics across phases: prefill is compute-bound, decode is memory-bound, and they need different kernel strategies
- Page-table-based KV-Cache (like virtual memory for attention) is the standard for production serving, and kernels must handle the indirection efficiently
- Fusing RoPE into the attention kernel eliminates a separate memory-bound kernel launch and enables more flexible cache management
- GQA decode with Tensor Cores is a major optimization: standard decode kernels underutilize Tensor Cores because the QK^T matmul is too small, but batching across GQA groups enables efficient Tensor Core usage
- Split-K parallelism is essential for decode with long contexts: without it, a single thread block must sequentially process the entire KV sequence, leaving most SMs idle
- FlashInfer won Best Paper at MLSys 2025, validating the importance of serving-aware attention kernel design

## References
- Blog: https://flashinfer.ai/2024/02/02/introduce-flashinfer.html
- Paper: https://arxiv.org/abs/2501.01005
- Code: https://github.com/flashinfer-ai/flashinfer
- Integrated into: SGLang, vLLM, MLC-LLM
