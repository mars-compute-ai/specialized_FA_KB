# FlashMLA: Efficient Multi-head Latent Attention Kernels (DeepSeek)

## Overview

Multi-Head Latent Attention (MLA) is a novel attention mechanism introduced in DeepSeek-V2 that replaces standard MHA/GQA with low-rank factorized projections, dramatically reducing KV cache memory (up to 93.3% reduction) while maintaining model quality. FlashMLA is DeepSeek's optimized CUDA kernel library for MLA inference on Hopper and Blackwell GPUs.

## How MLA Works

### Low-Rank KV Cache Compression
Instead of caching full K/V heads (standard MHA), MLA:
1. Compresses the input into a low-rank latent vector via a down-projection: `c_kv = W_down @ x` (d_model → rank r)
2. Decompresses on-the-fly during attention: `K, V = W_up @ c_kv` (rank r → 2×d_model)
3. Only the compressed latent vector `c_kv` is cached, not the full K/V heads

### Decoupled RoPE
MLA splits each head into two sub-heads:
- **Position-agnostic component**: Decompressed from latent vectors
- **RoPE-carrying component**: Extracted separately, carries positional information

This allows fine-grained control over positional encoding allocation.

### Comparison with Other Attention Variants
| Method | KV Cache per Token | Approach |
|--------|-------------------|----------|
| MHA | Full (100%) | Distinct K/V per Q head |
| MQA | ~2% | Single shared K/V head |
| GQA | ~12-25% | Grouped K/V sharing |
| MLA | ~1.4-34% | Compressed latent, on-demand decompression |

## FlashMLA Kernel Performance

### Decoding Kernels (H800 SXM5, CUDA 12.8)
- **Dense MLA Decoding**: 3000 GB/s (memory-bound), 660 TFLOPS (compute-bound)
- **Sparse FP8 Decoding**: 410 TFLOPS (compute-bound)

### Prefilling Kernels
- **Sparse Prefill MQA** (H800): 640 TFLOPS; (B200): 1450 TFLOPS
- **Dense Prefill MHA** (B200): 1460 TFLOPS fwd, 1000 TFLOPS bwd

### Hardware Support
| Kernel | Architecture | Mode | KV Format |
|--------|-------------|------|-----------|
| Dense Decoding | SM90 (Hopper) | MQA | BF16 |
| Sparse Decoding | SM90 & SM100 (Blackwell) | MQA | FP8 |
| Dense Prefill | SM100 (Blackwell) | MHA | — |
| Sparse Prefill | SM90 & SM100 | MQA | — |

## FP8 KV Cache Format
656 bytes per token:
- 512 bytes: Quantized "NoPE" section in float8_e4m3
- 16 bytes: Four float32 scale factors (one per 128 values)
- 128 bytes: Unquantized "RoPE" in bfloat16

## API Usage

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

### Sparse Attention via indices
A 3D tensor `(batch_size, seq_len_q, topk)` encoding page block index and token offset. Set invalid entries to -1.

## Kernel Design

FlashMLA builds on FlashAttention 2 & 3 and CUTLASS:
- Uses split-KV parallelism for decode (each split handles a subset of KV blocks)
- Warp-specialized pipelines on Hopper (SM90)
- Persistent kernels on Blackwell (SM100)
- FP8 dequantization fused into the attention kernel

## References
- [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA)
- [Understanding MLA](https://planetbanatt.net/articles/mla.html)
- [DeepSeek-V2 Paper (arXiv 2405.04434)](https://arxiv.org/abs/2405.04434)
- [FlashMLA Technical Analysis (YUV.AI)](https://yuv.ai/blog/flashmla)
