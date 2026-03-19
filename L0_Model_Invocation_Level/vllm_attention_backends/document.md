# vLLM Attention Backend Selection

Source: https://github.com/vllm-project/vllm
Additional: https://docs.vllm.ai/en/latest/
Additional: https://arxiv.org/abs/2309.06180 (PagedAttention / vLLM paper, SOSP 2023)

## Overview

vLLM is the dominant open-source framework for LLM inference serving, processing millions of requests daily across production deployments. At the core of vLLM's performance is its attention computation layer, which supports multiple backend implementations. The choice of attention backend significantly impacts throughput (tokens/second), latency (time-to-first-token and inter-token latency), memory efficiency, and feature availability. This document provides a comprehensive guide to vLLM's attention backends, their selection logic, performance characteristics, and configuration best practices.

## PagedAttention: The Foundation

### Concept

PagedAttention (Kwon et al., SOSP 2023) is the key innovation that makes vLLM possible. Inspired by virtual memory in operating systems, PagedAttention manages the KV cache using fixed-size memory pages (blocks):

```
Traditional KV Cache:
  Request 1: [||||||||||||........]  (contiguous, wasted space from padding)
  Request 2: [||||||||||||||||....]  (contiguous, wasted space)
  Request 3: [||||||..............]  (contiguous, massive waste)

PagedAttention KV Cache:
  Block Table for Request 1: [blk_3, blk_7, blk_12]
  Block Table for Request 2: [blk_1, blk_5, blk_9, blk_15]
  Block Table for Request 3: [blk_2, blk_8]

  Physical blocks: [blk_1][blk_2][blk_3][blk_5][blk_7][blk_8][blk_9][blk_12][blk_15]
                   (no waste, non-contiguous, managed via block table)
```

### Key Properties

- **Near-zero memory waste**: Only the last block of each sequence may have internal fragmentation
- **Dynamic growth**: Sequences can grow by allocating new blocks, no pre-allocation needed
- **Memory sharing**: Multiple sequences can share KV cache blocks (e.g., shared system prompts, beam search candidates) via reference counting
- **Copy-on-write**: When a shared block needs modification, it is copied to a new block
- **Block size**: Configurable (default 16 tokens per block). Larger blocks reduce block table overhead but increase internal fragmentation

### Block Table Structure

```python
# Block table: maps logical block indices to physical block indices
# Shape: [max_num_seqs, max_num_blocks_per_seq]
block_table = torch.tensor([
    [3, 7, 12, 0, 0],   # Request 0: blocks 3, 7, 12 (padded with 0s)
    [1, 5, 9, 15, 0],   # Request 1: blocks 1, 5, 9, 15
    [2, 8, 0, 0, 0],    # Request 2: blocks 2, 8
], dtype=torch.int32, device="cuda")

# KV cache: physical blocks stored contiguously
# Shape: [num_blocks, 2, num_heads, block_size, head_dim]
# 2 = K and V stored together
kv_cache = torch.empty(
    num_blocks, 2, num_kv_heads, block_size, head_dim,
    dtype=torch.float16, device="cuda"
)
```

## Available Attention Backends

### 1. FlashAttention-2 Backend (`FLASH_ATTN`)

**Implementation**: Uses Tri Dao's `flash-attn` library with custom paged KV cache extensions.

**Architecture**:
```
Prefill:    flash_attn_varlen_func (standard FA2, contiguous KV)
Decode:     flash_attn_with_kvcache (paged KV cache, single query per sequence)
Append:     flash_attn_varlen_func (variable-length Q, contiguous KV)
```

**Key Features**:
- Optimized tiling for both prefill (compute-bound) and decode (memory-bound)
- FP16 and BF16 support
- Causal masking without materialization
- Sliding window attention support
- GQA/MQA via the `num_kv_heads` parameter
- Paged KV cache via block table indirection

**Requirements**:
- NVIDIA Ampere+ (SM80+)
- `flash-attn` >= 2.5.0 (recommended >= 2.6.0 for latest features)
- FP16 or BF16 dtype

**Performance Profile**:
- Excellent prefill throughput (compute-bound optimization)
- Good decode throughput (memory-bound, but not as specialized as FlashInfer for GQA)
- Broad feature support and battle-tested stability

**Code Path** (simplified):
```python
# vllm/attention/backends/flash_attn.py

class FlashAttentionBackend(AttentionBackend):
    @staticmethod
    def get_impl_cls():
        return FlashAttentionImpl

class FlashAttentionImpl(AttentionImpl):
    def forward(self, query, key, value, kv_cache, attn_metadata):
        if attn_metadata.is_prompt:
            # Prefill: use flash_attn_varlen_func
            output = flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.seq_start_loc,
                cu_seqlens_k=attn_metadata.seq_start_loc,
                max_seqlen_q=attn_metadata.max_prefill_seq_len,
                max_seqlen_k=attn_metadata.max_prefill_seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
            )
        else:
            # Decode: use flash_attn_with_kvcache (paged)
            output = flash_attn_with_kvcache(
                q=query.unsqueeze(1),  # [B, 1, H, D]
                k_cache=kv_cache[0],
                v_cache=kv_cache[1],
                block_table=attn_metadata.block_tables,
                cache_seqlens=attn_metadata.seq_lens_tensor,
                softmax_scale=self.scale,
                causal=True,
            )
        return output
```

### 2. FlashInfer Backend (`FLASHINFER`)

**Implementation**: Uses the FlashInfer library (flashinfer-ai/flashinfer), which provides serving-optimized attention kernels.

**Architecture**:
```
Prefill:    BatchPrefillWithPagedKVCacheWrapper (compute-bound, tiled)
Decode:     BatchDecodeWithPagedKVCacheWrapper (memory-bound, GQA-optimized)
Append:     BatchPrefillWithPagedKVCacheWrapper (variable-length Q)
```

**Key Features**:
- **GQA Tensor Core decode**: Uses Tensor Cores for GQA decode by batching across query groups, achieving 2-3x speedup over standard CUDA-core decode
- **CUDAGraph-friendly**: Provides wrappers that are compatible with CUDAGraph capture, eliminating kernel launch overhead during decode
- **Ragged tensor support**: Natively handles variable-length sequences without padding
- **Split-K decode**: Parallelizes across KV sequence dimension for better SM utilization with long sequences and small batch sizes
- **FP8 KV cache**: Supports E4M3 FP8 quantized KV cache for memory savings
- **Fused RoPE**: Applies rotary positional embeddings inside the attention kernel

**Requirements**:
- NVIDIA Ampere+ (SM80+)
- `flashinfer` >= 0.1.0 (recommended >= 0.2.0)
- FP16, BF16, or FP8 (KV cache)

**Performance Profile**:
- Superior decode throughput for GQA models (Llama-3, Mistral, Gemma) due to Tensor Core decode
- CUDAGraph compatibility reduces per-step latency by 10-30% for decode
- Prefill performance comparable to FA2 (uses similar algorithmic approach)
- Best for high-throughput serving scenarios with many concurrent decode requests

**Code Path** (simplified):
```python
# vllm/attention/backends/flashinfer.py

class FlashInferBackend(AttentionBackend):
    @staticmethod
    def get_impl_cls():
        return FlashInferImpl

class FlashInferImpl(AttentionImpl):
    def __init__(self, ...):
        # Pre-allocate wrappers for CUDAGraph compatibility
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer,
            use_tensor_cores=True,  # GQA Tensor Core decode
        )

    def forward(self, query, key, value, kv_cache, attn_metadata):
        if attn_metadata.is_prompt:
            # Plan the prefill operation (done once, cached)
            self.prefill_wrapper.plan(
                qo_indptr=attn_metadata.qo_indptr,
                kv_indptr=attn_metadata.kv_indptr,
                kv_indices=attn_metadata.kv_indices,
                kv_last_page_len=attn_metadata.kv_last_page_len,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
            output = self.prefill_wrapper.run(query, kv_cache)
        else:
            # Plan the decode operation
            self.decode_wrapper.plan(
                indptr=attn_metadata.paged_kv_indptr,
                indices=attn_metadata.paged_kv_indices,
                last_page_len=attn_metadata.paged_kv_last_page_len,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.block_size,
            )
            output = self.decode_wrapper.run(query, kv_cache)
        return output
```

### 3. xFormers Backend (`XFORMERS`)

**Implementation**: Uses Facebook Research's xFormers library (`xformers.ops.fmha`).

**Architecture**:
```
Prefill:    xformers.ops.fmha.memory_efficient_attention (CUTLASS or Flash backend)
Decode:     Custom PagedAttention CUDA kernel (vLLM's own implementation)
```

**Key Features**:
- Broader GPU support (works on V100/P100 via CUTLASS backend)
- FP32 attention support (CUTLASS backend)
- Variable-length sequence support via BlockDiagonalMask
- Fallback option when FlashAttention installation fails

**Requirements**:
- NVIDIA GPU (P100+)
- `xformers` >= 0.0.22

**Performance Profile**:
- Prefill: Slightly slower than FA2 on Ampere+
- Decode: Uses vLLM's custom paged attention kernel (not xFormers' own kernel)
- Primarily useful as a fallback or for older GPU architectures

### 4. Triton Flash Attention Backend (`TRITON_FLASH_ATTN`)

**Implementation**: Pure Triton implementation of Flash Attention, maintained within vLLM.

**Architecture**:
```
Prefill:    Triton Flash Attention kernel
Decode:     Triton PagedAttention kernel
```

**Key Features**:
- Pure Python/Triton -- no CUDA compilation required
- Portable across NVIDIA and AMD (ROCm) GPUs
- Easier to modify and debug than CUDA kernels
- Supports paged KV cache

**Requirements**:
- Triton >= 2.1.0
- NVIDIA or AMD GPU with Triton support

**Performance Profile**:
- 10-30% slower than FA2 on NVIDIA GPUs
- Primary backend for AMD ROCm deployment
- Useful for development, debugging, and custom attention modifications

### 5. ROCm Flash Attention Backend (`ROCM_FLASH`)

**Implementation**: AMD-optimized Flash Attention using Composable Kernel (CK) backend.

**Architecture**:
```
Prefill:    CK-based Flash Attention (optimized for MI250X/MI300X)
Decode:     CK-based PagedAttention with splitKV
```

**Key Features**:
- Optimized for AMD CDNA2 (MI250X) and CDNA3 (MI300X) architectures
- Uses Matrix Fused Multiply-Add (MFMA) instructions
- Supports GQA/MQA
- Paged KV cache support with both vLLM-style and SGLang-style block tables

**Requirements**:
- AMD MI200X or MI300X GPU
- ROCm >= 5.7

## Backend Selection Logic

### Automatic Selection

vLLM implements a hierarchical backend selection algorithm:

```python
# Simplified backend selection logic from vllm/attention/selector.py

def get_attn_backend(head_size, dtype, kv_cache_dtype, block_size,
                     is_attention_free, sliding_window, is_blocksparse):
    # 1. Check environment variable override
    backend_env = os.environ.get("VLLM_ATTENTION_BACKEND")
    if backend_env:
        return validate_and_return(backend_env)

    # 2. Special cases
    if is_attention_free:
        return AttentionBackend.NO_ATTENTION

    if is_blocksparse:
        return AttentionBackend.BLOCKSPARSE

    # 3. Platform-specific selection
    if current_platform.is_rocm():
        return select_rocm_backend(head_size, dtype)

    if current_platform.is_cpu():
        return AttentionBackend.TORCH_SDPA

    # 4. NVIDIA GPU selection
    if current_platform.is_cuda():
        # Try FlashAttention-2 first
        if flash_attn_available and head_size_supported(head_size):
            return AttentionBackend.FLASH_ATTN

        # Fall back to xFormers
        if xformers_available:
            return AttentionBackend.XFORMERS

        # Last resort: Triton
        return AttentionBackend.TRITON_FLASH_ATTN
```

### Selection Criteria

| Criterion | FLASH_ATTN | FLASHINFER | XFORMERS | TRITON |
|---|---|---|---|---|
| GPU Architecture | Ampere+ | Ampere+ | P100+ | Any (Triton-supported) |
| Data Types | FP16, BF16 | FP16, BF16, FP8 KV | FP16, BF16, FP32 | FP16, BF16 |
| Head Dimensions | 32-256 (power of 2) | 64-256 | 32-256 | 32-256 |
| Causal Masking | Yes | Yes | Yes | Yes |
| Sliding Window | Yes | Yes | Limited | Yes |
| PagedAttention | Yes | Yes | Yes | Yes |
| Chunked Prefill | Yes | Yes | No | Limited |
| CUDAGraph Decode | Workaround | Native | No | No |
| FP8 KV Cache | Yes (recent) | Yes | No | No |
| Speculative Decoding | Yes | Yes | Limited | Limited |

### Manual Override

```bash
# Force a specific backend
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ATTENTION_BACKEND=TRITON_FLASH_ATTN

# Verify which backend is being used (check vLLM logs at startup)
# INFO: Using FlashAttention-2 backend.
# INFO: Using FlashInfer backend.
```

## Prefill vs Decode: Attention Kernel Requirements

### Prefill (Context) Phase

**Characteristics**:
- All prompt tokens are processed simultaneously
- Q, K, V are all full-length (seq_len tokens)
- Compute-bound: arithmetic intensity ~ O(seq_len)
- Dominant cost: two batched GEMMs (QK^T and PV)

**Kernel Requirements**:
- Maximize Tensor Core utilization
- Efficient tiling to fit tiles in shared memory
- Online softmax to avoid materializing N x N matrix
- Variable-length sequence support (different prompts have different lengths)

**Backend Performance (Prefill, H100, Llama-3-8B)**:

| Prompt Length | FLASH_ATTN (ms) | FLASHINFER (ms) | XFORMERS (ms) |
|---|---|---|---|
| 512 | 0.85 | 0.88 | 1.02 |
| 1024 | 1.95 | 2.01 | 2.45 |
| 2048 | 5.80 | 5.95 | 7.20 |
| 4096 | 18.5 | 19.0 | 23.8 |
| 8192 | 70.2 | 71.5 | 92.0 |

FA2 and FlashInfer are nearly identical for prefill (both use similar tiled attention algorithms). xFormers is 15-30% slower.

### Decode (Generation) Phase

**Characteristics**:
- Single new token generated per sequence per step
- Q has length 1 per sequence; KV cache has length seq_len
- Memory-bandwidth-bound: arithmetic intensity ~ O(1)
- Dominant cost: reading KV cache from HBM

**Kernel Requirements**:
- Maximize memory bandwidth utilization
- Efficient paged KV cache access (handle block table indirection)
- Minimize kernel launch overhead (CUDAGraph compatibility)
- GQA optimization: batch multiple query heads accessing the same KV head

**Backend Performance (Decode, H100, Llama-3-8B, 8 KV heads)**:

| Batch Size | Context Len | FLASH_ATTN (ms) | FLASHINFER (ms) | Speedup |
|---|---|---|---|---|
| 1 | 2048 | 0.12 | 0.10 | 1.20x |
| 16 | 2048 | 0.45 | 0.32 | 1.41x |
| 64 | 2048 | 1.52 | 1.05 | 1.45x |
| 256 | 2048 | 5.80 | 4.10 | 1.41x |
| 64 | 8192 | 5.85 | 4.20 | 1.39x |
| 256 | 8192 | 22.5 | 16.2 | 1.39x |

FlashInfer's GQA Tensor Core decode provides 1.2-1.5x speedup over FA2 for decode, which translates to measurably higher serving throughput.

### Chunked Prefill

Chunked prefill splits long prompts into smaller chunks and interleaves them with decode batches:

```
Without chunked prefill:
  Step 1: [=== Prefill (4096 tokens) ===]      <- Decode requests blocked
  Step 2: [Decode batch]
  Step 3: [Decode batch]

With chunked prefill (chunk_size=512):
  Step 1: [Prefill chunk 1 (512) | Decode batch]
  Step 2: [Prefill chunk 2 (512) | Decode batch]
  ...
  Step 8: [Prefill chunk 8 (512) | Decode batch]  <- Decode requests not blocked
```

**Benefits**:
- Reduces TTFT variance (decode requests are not starved during long prefills)
- Better GPU utilization by mixing compute-bound (prefill) and memory-bound (decode) work
- Required for high-throughput serving with SLA constraints

**Backend support for chunked prefill**:
- `FLASH_ATTN`: Full support
- `FLASHINFER`: Full support with CUDAGraph-compatible wrappers
- `XFORMERS`: Not supported (disable chunked prefill or use different backend)
- `TRITON_FLASH_ATTN`: Experimental support

```bash
# Enable chunked prefill in vLLM
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048   # Controls chunk size
```

## CUDAGraph Optimization

### Why CUDAGraph Matters for Decode

During decode, each step involves:
1. CPU-side scheduling and batch preparation (~100-200 us)
2. Kernel launch overhead (~10-50 us per kernel, multiple kernels per step)
3. Actual GPU computation (~50-500 us for small batch decode)

For small batch sizes, the CPU overhead (steps 1-2) can exceed the GPU computation time. CUDAGraph captures the entire sequence of GPU operations into a replayable graph, eliminating per-step CPU overhead.

### FlashInfer CUDAGraph Integration

FlashInfer provides first-class CUDAGraph support:

```python
# FlashInfer's BatchDecodeWithPagedKVCacheWrapper is CUDAGraph-compatible
# The wrapper maintains internal state that can be updated without re-capturing

# During CUDAGraph capture:
decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer,
    use_tensor_cores=True,
)

# Plan with maximum possible batch size for graph capture
decode_wrapper.plan(
    indptr=max_batch_indptr,
    indices=max_batch_indices,
    last_page_len=max_batch_last_page_len,
    num_qo_heads=32,
    num_kv_heads=8,
    head_dim=128,
    page_size=16,
)

# Capture the CUDAGraph
with torch.cuda.graph(graph):
    output = decode_wrapper.run(query, kv_cache)

# During inference, update metadata without re-capturing:
decode_wrapper.plan(  # Update for current batch
    indptr=current_indptr,
    indices=current_indices,
    last_page_len=current_last_page_len,
    ...
)
graph.replay()  # Replay captured graph with updated metadata
```

### FlashAttention-2 CUDAGraph Workaround

FA2 requires workarounds for CUDAGraph because its paged attention kernel takes variable-length inputs:

```python
# vLLM uses padding to fixed maximum batch size for CUDAGraph capture
# This wastes some computation but enables graph replay

# Pad queries to max_batch_size
padded_query = torch.zeros(max_batch_size, 1, num_heads, head_dim, ...)
padded_query[:actual_batch_size] = actual_query

# Pad block tables to max dimensions
padded_block_table = torch.zeros(max_batch_size, max_blocks_per_seq, ...)
padded_block_table[:actual_batch_size, :actual_blocks] = actual_block_table

# Capture and replay
with torch.cuda.graph(graph):
    output = flash_attn_with_kvcache(padded_query, k_cache, v_cache,
                                      block_table=padded_block_table, ...)
```

## FP8 KV Cache

### Motivation

KV cache is the dominant memory consumer in LLM inference. For a Llama-3-70B model with 80 layers, 8 KV heads, head_dim=128:
- Per token: 80 layers x 2 (K+V) x 8 heads x 128 dim x 2 bytes (FP16) = 327,680 bytes = 320 KB
- For 4096 context length, 256 sequences: 320 KB x 4096 x 256 = 320 GB (exceeds GPU memory)

FP8 quantization halves KV cache memory:
- Per token (FP8): 160 KB
- Memory savings: 50%
- Allows 2x longer contexts or 2x more concurrent sequences

### Backend Support

```python
# vLLM FP8 KV cache configuration
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --kv-cache-dtype fp8_e4m3   # FP8 KV cache
    --dtype bfloat16             # Model weights in BF16

# Backend compatibility:
# FLASH_ATTN: Supported (flash-attn >= 2.6.0)
# FLASHINFER: Supported (native FP8 KV cache kernels)
# XFORMERS: NOT supported
# TRITON: NOT supported
```

### Quality Impact

FP8 KV cache introduces quantization noise. Typical impact:
- Perplexity increase: < 0.1% for most models
- Task accuracy: Generally within noise of FP16
- Per-channel scaling is used to minimize quality degradation
- Dynamic scaling factors are computed per attention layer

## Speculative Decoding

### Attention Requirements

Speculative decoding generates N draft tokens, then verifies them in a single forward pass:

```
Draft model: generates tokens t1, t2, ..., tN (fast, small model)
Verification: target model processes [prompt, t1, t2, ..., tN] in one pass
Accept/reject: compare draft and target distributions
```

The verification pass requires "append-mode" attention where multiple new query tokens attend to the full KV cache:

```python
# Verification attention:
# Q: [t1, t2, ..., tN] (N new tokens)
# K, V: [full KV cache] + [k1, k2, ..., kN] (existing + new)
# This is neither pure prefill (Q is short) nor pure decode (Q length > 1)
```

**Backend support**:
- `FLASH_ATTN`: Supports via `flash_attn_varlen_func` with appropriate sequence length handling
- `FLASHINFER`: Native support via `BatchPrefillWithPagedKVCacheWrapper` (treats verification as variable-length prefill)
- `XFORMERS`: Limited support
- `TRITON`: Limited support

## Production Configuration Guide

### High-Throughput Serving (Many Concurrent Users)

```bash
# Optimize for maximum tokens/second
export VLLM_ATTENTION_BACKEND=FLASHINFER  # Best decode throughput for GQA
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.92 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --kv-cache-dtype fp8_e4m3      # Double KV cache capacity
```

### Low-Latency Serving (Real-Time Applications)

```bash
# Optimize for minimum latency
export VLLM_ATTENTION_BACKEND=FLASHINFER  # CUDAGraph-friendly
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager false           # Enable CUDAGraph
    --num-scheduler-steps 1         # Minimize scheduling latency
```

### Long-Context Serving (128K+ Tokens)

```bash
# Optimize for very long sequences
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Best prefill for long contexts
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --max-num-seqs 8 \              # Fewer concurrent sequences due to memory
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8_e4m3 \     # Essential for long contexts
    --tensor-parallel-size 4        # Distribute across GPUs
```

### AMD GPU Deployment

```bash
# AMD MI300X configuration
export VLLM_ATTENTION_BACKEND=ROCM_FLASH  # Or TRITON_FLASH_ATTN
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --device rocm
```

## Benchmarking Attention Backends

### Benchmark Script

```python
import os
import time
import torch
from vllm import LLM, SamplingParams

def benchmark_backend(backend, model, prompts, sampling_params):
    os.environ["VLLM_ATTENTION_BACKEND"] = backend
    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )

    # Warmup
    llm.generate(prompts[:2], sampling_params)

    # Benchmark
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed

    print(f"{backend}: {throughput:.1f} tokens/s, {elapsed:.2f}s total")
    return throughput

# Test configurations
model = "meta-llama/Llama-3.1-8B-Instruct"
prompts = ["Write a detailed essay about..." for _ in range(64)]
sampling_params = SamplingParams(max_tokens=256, temperature=0.8)

for backend in ["FLASH_ATTN", "FLASHINFER", "XFORMERS"]:
    benchmark_backend(backend, model, prompts, sampling_params)
```

### Expected Results (H100, Llama-3.1-8B, 64 concurrent requests)

| Backend | Throughput (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) |
|---|---|---|---|---|
| FLASH_ATTN | 4,200 | 45 | 120 | 12 |
| FLASHINFER | 4,800 | 42 | 110 | 9 |
| XFORMERS | 3,600 | 52 | 145 | 14 |
| TRITON | 3,200 | 58 | 160 | 16 |

FlashInfer provides ~15% higher throughput and ~25% lower inter-token latency (ITL) compared to FA2 for this GQA model, primarily due to GQA Tensor Core decode optimization.

## Troubleshooting

### Common Issues

**1. FlashAttention not available:**
```bash
# Error: flash_attn not found
pip install flash-attn --no-build-isolation

# If compilation fails on older GPUs:
export VLLM_ATTENTION_BACKEND=XFORMERS
```

**2. FlashInfer installation issues:**
```bash
# Install FlashInfer with CUDA support
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Verify installation
python -c "import flashinfer; print(flashinfer.__version__)"
```

**3. CUDAGraph failures with FlashInfer:**
```bash
# If CUDAGraph capture fails, try:
--enforce-eager  # Disable CUDAGraph temporarily for debugging

# Common cause: incompatible batch size or sequence length changes
# FlashInfer's CUDAGraph support requires fixed maximum dimensions
```

**4. Memory issues with large KV cache:**
```bash
# Reduce memory usage:
--kv-cache-dtype fp8_e4m3     # Halve KV cache memory
--gpu-memory-utilization 0.85  # Leave headroom
--max-num-seqs 128             # Limit concurrent sequences
--block-size 32                # Larger blocks, less overhead (but more fragmentation)
```

**5. Backend mismatch errors:**
```bash
# Error: backend X does not support feature Y
# Common mismatches:
# - XFORMERS + chunked_prefill -> not supported
# - TRITON + fp8_kv_cache -> not supported
# Solution: switch to FLASH_ATTN or FLASHINFER
```

## Architecture Diagram

```
vLLM Request Flow:
                                                  ┌─────────────────────┐
  Request ──> Scheduler ──> Model Runner ──> ┌──> │ Prefill Attention   │
                  │                          │    │ (FA2/FlashInfer/    │
                  │                          │    │  xFormers/Triton)   │
                  ▼                          │    └─────────────────────┘
         ┌───────────────┐                   │
         │ Batch Manager │ ──> Attention ────┤
         │ (inflight     │     Backend       │    ┌─────────────────────┐
         │  batching)    │     Selector  ────┼──> │ Decode Attention    │
         └───────────────┘                   │    │ (FA2/FlashInfer/    │
                  │                          │    │  PagedAttn/Triton)  │
                  ▼                          │    └─────────────────────┘
         ┌───────────────┐                   │
         │ KV Cache      │ <────────────────>│    ┌─────────────────────┐
         │ Manager       │                   └──> │ Append Attention    │
         │ (PagedAttn    │                        │ (Speculative/       │
         │  blocks)      │                        │  Chunked Prefill)   │
         └───────────────┘                        └─────────────────────┘
```

## Key Notes

- vLLM's attention backend choice can make a 15-40% difference in serving throughput
- PagedAttention is not a backend -- it is the KV cache management strategy used by ALL backends
- FlashInfer is increasingly becoming the recommended backend for high-throughput NVIDIA GPU serving
- Always benchmark with your specific model, hardware, and workload pattern before committing to a backend
- Backend selection interacts with other vLLM features (chunked prefill, speculative decoding, CUDAGraph) -- verify compatibility
- The vLLM project evolves rapidly; backend defaults and capabilities change between versions
- For AMD GPUs, ROCm Flash Attention and Triton are the primary options; performance tuning is more limited than on NVIDIA
