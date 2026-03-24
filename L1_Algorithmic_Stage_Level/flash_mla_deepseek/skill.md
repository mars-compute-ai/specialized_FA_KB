---
skill_name: FlashMLA - Multi-Head Latent Attention
description: Optimized CUDA kernels for DeepSeek's MLA attention with low-rank KV cache compression, achieving 660 TFLOPS decode and 1450 TFLOPS prefill
level: L1 - Algorithmic/Mathematical Stage Level
target_hardware: NVIDIA Hopper H100/H800 (SM90), Blackwell B200 (SM100)
relevance: When optimizing attention for models using MLA (DeepSeek-V2/V3) or when KV cache memory is the bottleneck
---

# FlashMLA - Multi-Head Latent Attention

## What It Is
Multi-Head Latent Attention (MLA) replaces standard MHA/GQA with low-rank factorized projections that compress KV cache by up to 93.3%. Instead of caching full K/V heads, MLA caches a small latent vector and decompresses K/V on-the-fly during attention. FlashMLA provides optimized CUDA kernels for this on Hopper and Blackwell GPUs.

## Key Concepts
- **Low-rank KV compression**: Down-project input to latent vector (d_model → rank r), cache only the latent, decompress K/V during attention
- **Decoupled RoPE**: Split heads into position-agnostic (from latent) and RoPE-carrying (separate) sub-heads
- **Split-KV decode**: Parallelize across KV sequence blocks, combine with log-sum-exp correction
- **FP8 KV cache**: 656 bytes/token format with quantized NoPE (float8_e4m3) + unquantized RoPE (bf16) + scale factors
- **Token-level sparse attention**: Indices tensor selects which KV tokens to attend to (top-k sparsity)

## Algorithm / Pseudo-code
```
# MLA Decode (simplified)
# 1. Compress: c_kv = W_down @ x  (done during prefill, cached)
# 2. For each query token:
#    a. Split KV sequence into blocks across splits
#    b. For each split:
#       - Decompress: K_i, V_i = W_up @ c_kv[block_i]
#       - Compute: S_i = Q @ K_i^T / sqrt(d)
#       - Apply causal mask
#       - O_i, lse_i = softmax(S_i) @ V_i  (with log-sum-exp)
#    c. Combine splits: O = combine(O_splits, lse_splits)

# MLA vs Standard FA difference:
# Standard FA: cache K, V (full size), compute Q@K^T directly
# MLA: cache c_kv (compressed), decompress K,V fused into kernel
```

## Code / Pseudo-code

### Python API Usage

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

# Get tile scheduler metadata
tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens, s_q * h_q // h_kv, h_kv, h_q, is_fp8, topk
)

# Run MLA decoding
for layer in range(num_layers):
    out, lse = flash_mla_with_kvcache(
        q, kvcache, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits,
        is_causal, is_fp8_kvcache, indices
    )
```

The `indices` parameter is a 3D tensor `(batch_size, seq_len_q, topk)` encoding page block index and token offset for sparse attention. Set invalid entries to -1.

## Performance
- **Decode**: 3000 GB/s memory-bound, 660 TFLOPS compute-bound (H800)
- **Sparse FP8 decode**: 410 TFLOPS (H800), 350 TFLOPS (B200)
- **Prefill**: 640 TFLOPS sparse (H800), 1450 TFLOPS sparse / 1460 TFLOPS dense (B200)
- **KV cache reduction**: 93.3% vs MHA (DeepSeek-V2 67B equivalent)
- **Throughput**: 5.76x higher generation throughput than dense MHA predecessor

## When to Use
- Deploying DeepSeek-V2/V3 or models using MLA architecture
- KV cache memory is the inference bottleneck (long contexts, large batch sizes)
- Need token-level sparse attention for efficiency (top-k selection)
- Running on Hopper (SM90) or Blackwell (SM100) GPUs

## When NOT to Use
- Model uses standard MHA/GQA (use FlashAttention instead)
- Training (FlashMLA is primarily optimized for inference)
- Pre-Hopper GPUs (requires SM90+)
- Need backward pass support for MLA-specific operations

## Key Takeaways
- MLA is a fundamentally different attention algorithm from MHA/GQA — the kernel must fuse KV decompression into the attention computation
- The FP8 KV cache format is custom (656 bytes/token) — not a standard quantization scheme
- FlashMLA achieves near-peak GPU utilization by combining split-KV parallelism, warp specialization, and fused decompression
- Token-level sparsity (via indices tensor) provides additional speedup for decode with minimal accuracy loss

## References
- [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA)
- [Understanding Multi-Head Latent Attention](https://planetbanatt.net/articles/mla.html)
- [DeepSeek-V2 Paper (arXiv 2405.04434)](https://arxiv.org/abs/2405.04434)
