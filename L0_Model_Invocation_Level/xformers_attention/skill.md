---
skill_name: xFormers Memory-Efficient Attention
description: Facebook Research's library providing multiple memory-efficient attention backends with composable bias/mask types and partial attention support.
level: L0 - Model/Invocation Level
target_hardware: NVIDIA GPUs (P100+, Ampere, Hopper via CUTLASS/Flash backends); AMD GPUs (MI200x, MI300x via Composable Kernel)
relevance: When needing memory-efficient attention with variable-length sequence support, composable mask types, or partial/chunked attention for sequence parallelism.
---

# xFormers Memory-Efficient Attention

## What It Is
xFormers is a library from Facebook Research that provides optimized, memory-efficient attention implementations through multiple backends: CUTLASS-based (broadest GPU support), FlashAttention-based (best performance on Ampere+), and Composable Kernel (AMD GPUs). It offers specialized attention bias types (causal, block-diagonal, local window) that avoid materializing N x N attention matrices, and supports partial attention computation for sequence parallelism and ring attention.

## Key Concepts
- **Multiple backends with unified API**: `xformers.ops.fmha.flash.FwOp/BwOp`, `xformers.ops.fmha.cutlass.FwOp/BwOp`, `xformers.ops.fmha.ck.FwOp/BwOp` -- select via the `op=` parameter
- **Composable attention biases**: `LowerTriangularMask` (causal), `BlockDiagonalMask` (variable-length batching without padding), `LocalAttentionFromBottomRightMask` (sliding window), and combinations thereof
- **Partial attention + merge**: `memory_efficient_attention_partial()` returns `(output, lse)` for KV chunks; `merge_attentions()` combines them exactly -- enables ring attention and sequence parallelism
- **Input format**: `(batch, seq_len, num_heads, head_dim)` -- differs from PyTorch SDPA's `(batch, num_heads, seq_len, head_dim)`
- **BlockDiagonalMask**: Eliminates padding waste by packing variable-length sequences into a single batch with `from_seqlens()` or `from_tensor_list()`

## When to Use
- You need variable-length sequence batching without padding overhead (BlockDiagonalMask)
- You need partial/chunked attention for ring attention or sequence parallelism across GPUs
- You need fp32 attention support (CUTLASS backend supports it; FlashAttention does not)
- You are on older NVIDIA GPUs (P100/V100) where PyTorch SDPA's FlashAttention backend is unavailable
- You need AMD GPU support via the Composable Kernel backend
- You want explicit control over which attention backend (flash vs cutlass) is used for forward and backward passes independently

## When NOT to Use
- You only need standard causal attention on modern NVIDIA GPUs -- PyTorch SDPA is simpler and equally performant
- You need custom score modifications (ALiBi, soft-capping) -- use FlexAttention instead
- You need FP8 attention -- use FlashAttention-3/4 directly
- You are starting a new project and want the most future-proof API -- PyTorch SDPA and FlexAttention are the official PyTorch path forward
- You need paged KV cache for production serving -- use vLLM or flash_attn's inference API

## Code Snippets / Pseudo-code
```python
import xformers.ops as xops
from xformers.ops.fmha import flash, cutlass
from xformers.ops.fmha.attn_bias import (
    LowerTriangularMask,
    BlockDiagonalMask,
)

# Basic causal attention (auto backend selection)
output = xops.memory_efficient_attention(
    query, key, value,
    attn_bias=LowerTriangularMask()
)

# Force FlashAttention backend
output = xops.memory_efficient_attention(
    query, key, value,
    attn_bias=LowerTriangularMask(),
    op=(flash.FwOp, flash.BwOp)
)

# Variable-length sequences without padding
attn_bias = BlockDiagonalMask.from_seqlens([128, 256, 64])
# Pack sequences: query shape (1, 448, num_heads, head_dim)
output = xops.memory_efficient_attention(query, key, value, attn_bias=attn_bias)

# Partial attention for sequence parallelism / ring attention
out1, lse1 = xops.memory_efficient_attention_partial(query, kv_chunk1_k, kv_chunk1_v)
out2, lse2 = xops.memory_efficient_attention_partial(query, kv_chunk2_k, kv_chunk2_v)
output = xops.merge_attentions([out1, out2], [lse1, lse2])
```

## Key Takeaways
- xFormers is the go-to library when you need composable attention biases (block-diagonal, local window) or partial attention for distributed/chunked computation
- The `BlockDiagonalMask` is uniquely powerful for eliminating padding waste in variable-length batches
- `memory_efficient_attention_partial()` + `merge_attentions()` enable ring attention and cross-GPU sequence parallelism
- CUTLASS backend provides the broadest GPU support (P100+) and fp32 compatibility
- Input tensor format is (B, S, H, D), not (B, H, S, D) -- watch for transpose bugs when switching from PyTorch SDPA
- For new projects on modern hardware, consider whether PyTorch SDPA or FlexAttention can meet your needs first, as they are the official PyTorch-maintained solutions

## References
- [xFormers Attention Ops Documentation](https://facebookresearch.github.io/xformers/components/ops.html)
- [xFormers GitHub Repository](https://github.com/facebookresearch/xformers)
- [xFormers Memory-Efficient Attention API](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
