# TensorRT-LLM Attention Kernel Dispatch

Source: https://github.com/NVIDIA/TensorRT-LLM
Additional: https://nvidia.github.io/TensorRT-LLM/
Additional: https://nvidia.github.io/TensorRT-LLM/architecture/attention.html

## Overview

TensorRT-LLM (TRT-LLM) is NVIDIA's high-performance inference framework for large language models. Its attention layer implements a sophisticated multi-kernel dispatch system that selects the optimal kernel based on the inference phase, model architecture, and hardware capabilities. Unlike simpler frameworks that use a single attention kernel for all scenarios, TRT-LLM deploys specialized kernels: FMHA (Fused Multi-Head Attention) for compute-bound prefill, XQA (Cross-Query Attention) for memory-bound GQA/MQA decode, cuDNN fused attention for FP8 and fallback scenarios, and paged KV cache attention for general decode workloads. This document provides a comprehensive technical analysis of TRT-LLM's attention architecture, kernel dispatch logic, configuration options, and optimization strategies.

## TRT-LLM Attention Architecture

### The GPT Attention Plugin

The `gpt_attention_plugin` is TRT-LLM's unified attention layer that encapsulates:
1. KV cache management (read/write, paging, quantization)
2. Positional encoding application (RoPE, ALiBi)
3. Attention kernel dispatch (FMHA, XQA, paged attention)
4. Output concatenation and projection

```
                    ┌─────────────────────────────────┐
                    │      GPT Attention Plugin        │
                    │                                  │
   Q, K, V ───────>│  ┌──────────┐  ┌─────────────┐  │──────> Output
                    │  │ KV Cache │  │   Kernel     │  │
                    │  │ Manager  │  │   Dispatch   │  │
                    │  │          │  │              │  │
                    │  │ - Write  │  │ Context:     │  │
                    │  │ - Page   │  │  └─ FMHA     │  │
                    │  │ - Quant  │  │  └─ cuDNN    │  │
                    │  │          │  │              │  │
                    │  │          │  │ Generation:  │  │
                    │  │          │  │  └─ XQA      │  │
                    │  │          │  │  └─ Paged    │  │
                    │  └──────────┘  └─────────────┘  │
                    └─────────────────────────────────┘
```

### Plugin Configuration

```python
from tensorrt_llm import Builder, BuildConfig
from tensorrt_llm.plugin import PluginConfig

plugin_config = PluginConfig()

# Core attention plugin
plugin_config.gpt_attention_plugin = "float16"  # Enable with FP16 compute
# Options: "float16", "bfloat16", "float32", "disable"

# Paged KV cache
plugin_config.paged_kv_cache = True       # Enable page-based KV cache (required for production)
plugin_config.tokens_per_block = 64       # Tokens per KV cache block (default: 64)

# Context (prefill) kernel configuration
plugin_config.context_fmha_type = "enabled"
# Options:
#   "disabled"              - Use unfused attention (slow, for debugging)
#   "enabled"               - Use FMHA with FP16 accumulation (faster, slightly less precise)
#   "enabled_with_fp32_acc" - Use FMHA with FP32 accumulation (recommended for training-quality)

# FP8 context attention (Hopper+ only)
plugin_config.use_fp8_context_fmha = False  # Enable FP8 context attention via cuDNN

# XQA decode kernel
plugin_config.use_xqa = True                # Enable XQA for GQA/MQA decode (default: True)

# Multi-block mode for long sequences
plugin_config.multi_block_mode = True       # Enable split-K for decode with long contexts

build_config = BuildConfig(
    max_batch_size=64,
    max_input_len=2048,
    max_seq_len=8192,
    max_num_tokens=8192,
    plugin_config=plugin_config,
)
```

## Context Phase Kernels (Prefill)

### FMHA Plugin (Fused Multi-Head Attention)

The FMHA plugin is TRT-LLM's primary context attention kernel, providing a fused implementation of the scaled dot-product attention.

**Algorithm** (FlashAttention-style tiled attention):
```
for each batch element b, each head h:
    Initialize: O = 0, m = -inf, l = 0

    for each KV tile j (of size B_kv):
        # Load K, V tiles from HBM to shared memory
        K_j = load_smem(K[b, h, j*B_kv : (j+1)*B_kv, :])
        V_j = load_smem(V[b, h, j*B_kv : (j+1)*B_kv, :])

        for each Q tile i (of size B_q):
            Q_i = load_smem(Q[b, h, i*B_q : (i+1)*B_q, :])

            # Compute attention scores
            S_ij = Q_i @ K_j^T / sqrt(d)     # In registers via Tensor Cores

            # Apply causal mask (if applicable)
            if is_causal:
                S_ij = mask_future_tokens(S_ij, i, j)

            # Online softmax update
            m_new = max(m, rowmax(S_ij))
            P_ij = exp(S_ij - m_new)
            l_new = exp(m - m_new) * l + rowsum(P_ij)
            O = exp(m - m_new) * O + P_ij @ V_j
            m = m_new
            l = l_new

    O = O / l  # Final normalization
    store_hbm(Output[b, h, :, :], O)
```

**Supported Configurations**:

| Parameter | FMHA Support |
|---|---|
| Head dimensions | 32, 40, 64, 80, 96, 104, 128, 144, 160, 192, 224, 256 |
| Data types | FP16, BF16 |
| Accumulation | FP16 or FP32 (configurable) |
| Causal masking | Yes |
| Padding masking | Yes |
| GQA/MQA | Yes |
| Variable sequence lengths | Yes (ragged batching) |
| ALiBi positional bias | Yes |
| Sliding window | Yes |
| Cross-attention | Yes |
| Max sequence length | Memory-limited (tested up to 128K+) |

**FP32 Accumulation**:
```python
# FP16 accumulation: faster but less precise
plugin_config.context_fmha_type = "enabled"
# S_ij accumulation in FP16: ~10% faster, but may have numerical issues
# for very long sequences (softmax underflow/overflow)

# FP32 accumulation: recommended for production
plugin_config.context_fmha_type = "enabled_with_fp32_acc"
# S_ij accumulation in FP32: training-quality precision
# Slight performance cost (~5-10%) but eliminates numerical issues
```

**Tile Sizes by Architecture**:

| Architecture | B_q | B_kv | Threads/Block | Notes |
|---|---|---|---|---|
| Ampere (SM80) | 64 | 64 | 128-256 | HMMA_16816 |
| Hopper (SM90) | 64-128 | 64-128 | 128-256 | WGMMA + TMA |
| Blackwell (SM100) | 128 | 128 | 256-512 | 5th-gen TC |

### cuDNN Fused Attention (Context)

TRT-LLM uses cuDNN's fused attention as an alternative context kernel in specific scenarios:

**When cuDNN is selected over FMHA**:
1. **FP8 context attention**: When `use_fp8_context_fmha=True` on Hopper+, TRT-LLM routes through cuDNN's FP8 SDPA implementation
2. **Unsupported head dimensions**: If the head dimension is not in FMHA's supported list, cuDNN may provide a fallback
3. **Specific problem sizes**: TRT-LLM's heuristics may prefer cuDNN for certain batch/sequence/head configurations

**FP8 Context Attention via cuDNN**:
```python
# Hopper H100 with FP8 context attention
plugin_config.use_fp8_context_fmha = True

# What happens internally:
# 1. Q, K, V are quantized to FP8 E4M3 with per-tensor scaling
# 2. cuDNN's FP8 SDPA kernel computes attention with FP32 softmax
# 3. Output is dequantized back to FP16/BF16
# 4. ~1.6-1.8x speedup over FP16 FMHA for compute-bound prefill

# Performance impact (H100, Llama-3-8B, head_dim=128):
# Prompt length 2048: FMHA FP16 = 5.8ms, cuDNN FP8 = 3.5ms (1.66x speedup)
# Prompt length 4096: FMHA FP16 = 18.5ms, cuDNN FP8 = 10.8ms (1.71x speedup)
# Prompt length 8192: FMHA FP16 = 70.2ms, cuDNN FP8 = 40.5ms (1.73x speedup)
```

### Context Kernel Selection Logic

```python
# Simplified context kernel selection (from TRT-LLM's attention plugin)

def select_context_kernel(config):
    if config.use_fp8_context_fmha and config.gpu.is_hopper_plus():
        # FP8 attention always uses cuDNN
        return ContextKernel.CUDNN_FP8

    if config.context_fmha_type == "disabled":
        return ContextKernel.UNFUSED  # Separate GEMM + softmax + GEMM

    if fmha_supports(config.head_dim, config.dtype, config.features):
        if config.context_fmha_type == "enabled_with_fp32_acc":
            return ContextKernel.FMHA_FP32_ACC
        else:
            return ContextKernel.FMHA_FP16_ACC

    # Fallback to cuDNN for unsupported FMHA configs
    if cudnn_supports(config.head_dim, config.dtype):
        return ContextKernel.CUDNN_FP16

    # Last resort: unfused attention
    return ContextKernel.UNFUSED
```

## Generation Phase Kernels (Decode)

### XQA Kernel (Cross-Query Attention)

The XQA kernel is TRT-LLM's specialized decode kernel for models using Grouped-Query Attention (GQA) or Multi-Query Attention (MQA). It is the single most important kernel for decode throughput in modern LLMs.

**Motivation**:
Standard MHA decode: 32 query heads, 32 KV heads -> each thread block processes one Q-KV pair
GQA decode: 32 query heads, 8 KV heads -> 4 query heads share each KV head

Without XQA, naive GQA decode reads each KV head once per query head that uses it. XQA instead:
1. Assigns multiple thread blocks to cooperatively process one KV head
2. Each thread block handles a subset of the KV sequence (split-K)
3. Multiple query heads sharing a KV head are processed together (cross-query)
4. Achieves near-optimal memory bandwidth utilization

**Algorithm**:
```
# XQA Decode for GQA (simplified)
# Input: Q[B, H_q, 1, D], KV_cache[num_blocks, 2, H_kv, block_size, D]

# Map query heads to KV heads
group_size = H_q / H_kv  # e.g., 32/8 = 4

for each KV head kv_h in parallel:
    # Load KV data for this head (single HBM read)
    for each KV sequence chunk in parallel (split-K):
        K_chunk = load_paged_kv(block_table, kv_h, chunk_start, chunk_end)
        V_chunk = load_paged_kv(block_table, kv_h, chunk_start, chunk_end)

        # Process all query heads in this group together
        for q_h in group(kv_h):  # group_size query heads
            Q_qh = Q[:, q_h, 0, :]  # Single query vector

            # Compute scores and partial output
            scores = Q_qh @ K_chunk^T / sqrt(D)
            partial_out, partial_lse = online_softmax_partial(scores, V_chunk)

    # Reduce across split-K chunks
    for q_h in group(kv_h):
        final_out[q_h] = merge_partial_outputs(partial_outs, partial_lses)
```

**Key Optimizations in XQA**:
- **KV data reuse**: Each KV cache page is read from HBM once and used by all query heads in the group
- **Split-K parallelism**: Long KV sequences are split across multiple thread blocks for better SM utilization
- **Warp-level reduction**: Partial outputs from different warps are efficiently reduced using shared memory
- **Paged KV cache awareness**: Block table indirection is handled efficiently with prefetching

**Supported Configurations**:

| Parameter | XQA Support |
|---|---|
| Attention pattern | GQA, MQA (NOT standard MHA) |
| Head dimensions | 64, 128, 256 |
| Data types | FP16, BF16, FP8 KV cache |
| KV cache | Paged (block table) |
| Max KV length | Limited by split-K grid size (up to ~1M tokens) |
| Beam search | Supported |

**Performance**:

XQA decode throughput on H100 (Llama-3-8B, 8 KV heads, head_dim=128):

| Batch Size | KV Length | Without XQA (ms) | With XQA (ms) | Speedup |
|---|---|---|---|---|
| 1 | 2048 | 0.15 | 0.08 | 1.88x |
| 16 | 2048 | 0.82 | 0.35 | 2.34x |
| 64 | 2048 | 2.95 | 1.15 | 2.57x |
| 64 | 8192 | 10.8 | 4.50 | 2.40x |
| 256 | 4096 | 18.5 | 7.80 | 2.37x |

XQA provides 1.9-2.6x speedup over non-XQA paged attention for GQA models, making it critical for decode-limited serving scenarios.

### Paged KV Cache Attention (General Decode)

When XQA is not applicable (e.g., standard MHA models or unsupported configurations), TRT-LLM falls back to a general-purpose paged KV cache attention kernel.

**When paged attention is used instead of XQA**:
- Standard MHA models (H_q == H_kv)
- Head dimensions not supported by XQA
- XQA explicitly disabled (`use_xqa=False`)
- Configuration-specific fallbacks

**Implementation**:
```
# General paged decode attention (per query head)
for each batch element b, each query head h:
    q = Q[b, h, 0, :]  # Single query vector [D]

    # Traverse paged KV cache
    num_blocks = seq_len[b] / block_size
    m = -inf, l = 0, acc = 0

    for block_idx in range(num_blocks):
        phys_block = block_table[b, block_idx]
        K_block = kv_cache[phys_block, 0, h_kv, :, :]  # [block_size, D]
        V_block = kv_cache[phys_block, 1, h_kv, :, :]  # [block_size, D]

        scores = q @ K_block^T / sqrt(D)  # [block_size]

        # Online softmax
        m_new = max(m, max(scores))
        P = exp(scores - m_new)
        l = exp(m - m_new) * l + sum(P)
        acc = exp(m - m_new) * acc + P @ V_block
        m = m_new

    output[b, h, :] = acc / l
```

### Multi-Block Mode (Split-K for Decode)

For decode with very long KV sequences, a single thread block per head may not fully utilize the GPU. Multi-block mode splits the KV sequence across multiple thread blocks:

```python
# Enable multi-block mode
plugin_config.multi_block_mode = True

# Without multi-block: 1 thread block per (batch, head) pair
# With multi-block: N thread blocks per (batch, head), each handling KV_len/N tokens
# Partial results are reduced via a second kernel
```

**When multi-block mode helps**:
- KV sequence length > 4096 tokens
- Small batch sizes (few (batch, head) pairs to parallelize over)
- Improves decode latency by 1.3-2x for long-context scenarios

### Generation Kernel Selection Logic

```python
# Simplified generation kernel selection

def select_generation_kernel(config, batch_info):
    if config.use_xqa and is_gqa_or_mqa(config) and xqa_supports(config):
        # XQA for GQA/MQA decode
        if config.multi_block_mode and max_kv_len > threshold:
            return GenerationKernel.XQA_MULTI_BLOCK
        else:
            return GenerationKernel.XQA_SINGLE_BLOCK

    elif config.paged_kv_cache:
        # General paged attention
        if config.multi_block_mode and max_kv_len > threshold:
            return GenerationKernel.PAGED_MULTI_BLOCK
        else:
            return GenerationKernel.PAGED_SINGLE_BLOCK

    else:
        # Non-paged attention (not recommended for production)
        return GenerationKernel.UNFUSED_DECODE
```

## Paged KV Cache Implementation

### Block-Based Memory Management

TRT-LLM's KV cache uses a block-based allocation strategy:

```
KV Cache Layout:
  Total GPU memory for KV cache: determined by gpu_memory_utilization

  Physical blocks: [Block 0][Block 1][Block 2]...[Block N]
  Each block: [num_kv_heads, tokens_per_block, head_dim] x 2 (K and V)

  Block table per sequence:
    Seq 0: [phys_block_3, phys_block_7, phys_block_12, ...]
    Seq 1: [phys_block_1, phys_block_5, phys_block_9, ...]

  Token-to-block mapping:
    Token at position p -> block_table[p // tokens_per_block]
    Offset within block -> p % tokens_per_block
```

**Block size considerations**:

| Block Size | Pros | Cons |
|---|---|---|
| 16 | Low internal fragmentation | Higher block table overhead |
| 32 | Balanced | - |
| 64 (default) | Good GPU cache line utilization | Moderate fragmentation |
| 128 | Best memory bandwidth | High fragmentation for short sequences |

```python
# Configure block size
plugin_config.tokens_per_block = 64  # Default
# Larger blocks: better for long sequences (fewer block lookups)
# Smaller blocks: better for many short sequences (less waste)
```

### KV Cache Quantization

TRT-LLM supports quantized KV cache to reduce memory usage:

```python
# FP8 KV cache (Hopper+)
# Stores K, V in FP8 E4M3 format with per-channel scaling factors
# 50% memory reduction vs FP16
build_config.kv_cache_type = "fp8"

# INT8 KV cache
# 50% memory reduction vs FP16
# Slightly more quality degradation than FP8
build_config.kv_cache_type = "int8"

# INT4 KV cache (experimental)
# 75% memory reduction vs FP16
# Requires careful calibration
build_config.kv_cache_type = "int4"
```

**Quality Impact of KV Cache Quantization**:

| Quantization | Memory Savings | Perplexity Impact | Use Case |
|---|---|---|---|
| FP16 (baseline) | 0% | 0% | Accuracy-critical |
| FP8 E4M3 | 50% | < 0.1% | Recommended default on Hopper |
| INT8 | 50% | 0.1-0.5% | When FP8 unavailable |
| INT4 | 75% | 0.5-2.0% | Maximum throughput, quality-tolerant |

## Inflight Batching and Attention

### Mixed Prefill-Decode Batching

TRT-LLM's batch manager can process context (prefill) and generation (decode) requests simultaneously:

```
Inflight Batch:
  Request 0: CONTEXT phase, prompt_len=1024
  Request 1: GENERATION phase, kv_len=512
  Request 2: GENERATION phase, kv_len=2048
  Request 3: CONTEXT phase, prompt_len=256
  Request 4: GENERATION phase, kv_len=1024

Attention dispatch for this batch:
  1. Separate context and generation requests
  2. Context requests -> FMHA kernel (compute-bound)
  3. Generation requests -> XQA kernel (memory-bound)
  4. Both execute on same GPU, potentially overlapping
```

**Configuration**:
```python
from tensorrt_llm.executor import ExecutorConfig

executor_config = ExecutorConfig(
    max_batch_size=64,
    max_num_tokens=8192,       # Total tokens across all requests in a batch
    batching_type="inflight",  # Enable inflight batching
    # Alternative: "static" for fixed-batch inference
)
```

### Chunked Context in TRT-LLM

Similar to vLLM's chunked prefill, TRT-LLM can split long context requests into chunks:

```python
executor_config = ExecutorConfig(
    max_num_tokens=4096,  # Limits tokens processed per step
    # A context request with 8192 tokens will be processed in 2 chunks
    # Each chunk is processed alongside generation requests
)
```

## Performance Optimization Guide

### Profile Attention Kernels

```python
# Use NVIDIA Nsight Systems to profile TRT-LLM attention
# nsys profile -o trtllm_attention python run_inference.py

# Key kernel names to look for in the profile:
# Context kernels:
#   - "fmha_*"           -> FMHA plugin
#   - "cudnn_*_fused_*"  -> cuDNN fused attention
#
# Generation kernels:
#   - "xqa_*"            -> XQA decode
#   - "paged_attention_*" -> General paged attention
#   - "reduce_*"         -> Multi-block reduction
```

### Configuration for Different Models

**Standard MHA Model (e.g., GPT-3)**:
```python
# H_q = H_kv = 96 heads, head_dim = 128
plugin_config = PluginConfig()
plugin_config.gpt_attention_plugin = "float16"
plugin_config.paged_kv_cache = True
plugin_config.tokens_per_block = 64
plugin_config.context_fmha_type = "enabled_with_fp32_acc"
plugin_config.use_xqa = False  # XQA not beneficial for standard MHA
plugin_config.multi_block_mode = True  # Helps with long contexts
```

**GQA Model (e.g., Llama-3-70B)**:
```python
# H_q = 64 heads, H_kv = 8 heads, head_dim = 128
plugin_config = PluginConfig()
plugin_config.gpt_attention_plugin = "bfloat16"
plugin_config.paged_kv_cache = True
plugin_config.tokens_per_block = 64
plugin_config.context_fmha_type = "enabled_with_fp32_acc"
plugin_config.use_xqa = True  # Critical for GQA decode performance
plugin_config.multi_block_mode = True
```

**MQA Model (e.g., older Falcon)**:
```python
# H_q = 64 heads, H_kv = 1 head, head_dim = 64
plugin_config = PluginConfig()
plugin_config.gpt_attention_plugin = "float16"
plugin_config.paged_kv_cache = True
plugin_config.tokens_per_block = 64
plugin_config.context_fmha_type = "enabled_with_fp32_acc"
plugin_config.use_xqa = True  # Even more beneficial for MQA (64:1 ratio)
```

**FP8 Optimized (Hopper H100)**:
```python
# Maximum throughput on H100
plugin_config = PluginConfig()
plugin_config.gpt_attention_plugin = "bfloat16"
plugin_config.paged_kv_cache = True
plugin_config.tokens_per_block = 64
plugin_config.context_fmha_type = "enabled_with_fp32_acc"
plugin_config.use_fp8_context_fmha = True  # FP8 prefill via cuDNN
plugin_config.use_xqa = True
plugin_config.multi_block_mode = True

build_config = BuildConfig(
    ...,
    strongly_typed=True,
    quant_config=QuantConfig(kv_cache_quant=QuantAlgo.FP8),  # FP8 KV cache
)
```

### End-to-End Performance Comparison

**Llama-3.1-8B on H100 (single GPU)**:

| Configuration | Prefill (tok/s) | Decode (tok/s) | Total Throughput |
|---|---|---|---|
| FMHA FP16 + Paged (no XQA) | 28,000 | 1,800 | 2,400 |
| FMHA FP16 + XQA | 28,000 | 4,200 | 4,800 |
| cuDNN FP8 + XQA | 46,000 | 4,200 | 5,600 |
| cuDNN FP8 + XQA + FP8 KV | 46,000 | 4,500 | 6,200 |

**Llama-3.1-70B on 4x H100 (tensor parallel)**:

| Configuration | Prefill (tok/s) | Decode (tok/s) | Total Throughput |
|---|---|---|---|
| FMHA FP16 + XQA | 12,000 | 2,800 | 3,200 |
| cuDNN FP8 + XQA + FP8 KV | 19,500 | 3,200 | 4,100 |

## Comparison with Other Frameworks

### TRT-LLM vs vLLM Attention

| Feature | TRT-LLM | vLLM |
|---|---|---|
| Context kernel | FMHA (closed-source) or cuDNN | FlashAttention-2 (open-source) |
| Decode kernel | XQA (closed-source) | FA2 or FlashInfer (open-source) |
| GQA decode optimization | XQA (proprietary) | FlashInfer GQA Tensor Core |
| FP8 context | cuDNN FP8 SDPA | Not standard (via FP8 KV cache) |
| Paged KV cache | Block-based (similar to vLLM) | PagedAttention (original) |
| KV cache quantization | FP8, INT8, INT4 | FP8 (backend-dependent) |
| Kernel customization | Not possible (closed-source) | FlashInfer/Triton (open-source) |
| Build time | Requires TRT engine build step | Dynamic, no build step |
| Inflight batching | Built-in | Built-in (continuous batching) |

### TRT-LLM vs FlashAttention Direct

| Aspect | TRT-LLM | FlashAttention-2/3 |
|---|---|---|
| Use case | Full inference framework | Attention kernel library |
| Context attention | FMHA/cuDNN (optimized per-arch) | FA2/FA3 (open-source, hand-tuned) |
| Decode attention | XQA (GQA-specialized) | flash_attn_with_kvcache (general) |
| KV cache management | Built-in block manager | User-managed |
| Performance | Competitive, sometimes faster on Hopper | Fastest open-source option |
| Flexibility | Low (use TRT-LLM pipeline) | High (integrate into any framework) |

## Advanced Topics

### MLA (Multi-Latent Attention) Support

TRT-LLM includes support for DeepSeek-V2's Multi-Latent Attention:

```python
# DeepSeek-V2 uses compressed KV representation
# Standard attention: KV cache stores [num_kv_heads, seq_len, head_dim]
# MLA: KV cache stores compressed latent [seq_len, latent_dim] where latent_dim << num_heads * head_dim

# TRT-LLM dispatches to specialized MLA kernels:
# - Context: FMHA with latent expansion
# - Decode: XQA-MLA variant
```

### Speculative Decoding Integration

```python
# TRT-LLM speculative decoding with attention
executor_config = ExecutorConfig(
    speculative_config=SpeculativeDecodingConfig(
        num_draft_tokens=5,        # Generate 5 draft tokens
        draft_model_path="...",    # Path to draft model
    )
)

# Attention for speculative decoding:
# 1. Draft model: lightweight attention (may use smaller FMHA)
# 2. Verification: target model processes draft tokens
#    - Uses context FMHA for the batch of draft tokens
#    - KV cache is updated for accepted tokens
```

### Custom Attention Patterns

TRT-LLM supports several attention pattern variations:

```python
# Sliding window attention (e.g., Mistral)
# FMHA and XQA both support windowed attention
model_config = ModelConfig(
    sliding_window=4096,  # Only attend to last 4096 tokens
)

# ALiBi positional bias (e.g., BLOOM, MPT)
# Supported in FMHA via additive bias
model_config = ModelConfig(
    position_embedding_type="alibi",
)

# RoPE (Rotary Position Embeddings)
# Applied before attention in the GPT attention plugin
model_config = ModelConfig(
    position_embedding_type="rope",
    rope_theta=500000.0,        # Llama-3 uses 500K theta
    rope_scaling_type="dynamic", # Extended context support
)
```

## Troubleshooting

### Common Issues

**1. FMHA not being used (slow context)**:
```python
# Verify FMHA is enabled
print(f"context_fmha_type: {plugin_config.context_fmha_type}")
# Should be "enabled" or "enabled_with_fp32_acc"

# Check logs for: "Using fmha kernel for context attention"
# vs: "Falling back to unfused context attention"

# Common cause: unsupported head_dim
# Solution: Check FMHA supported head dimensions list
```

**2. XQA not being used (slow decode)**:
```python
# Verify XQA is enabled and applicable
print(f"use_xqa: {plugin_config.use_xqa}")
print(f"Model is GQA: {num_kv_heads < num_heads}")

# XQA requires:
# - GQA or MQA architecture (num_kv_heads < num_heads)
# - Supported head_dim (64, 128, 256)
# - Paged KV cache enabled
```

**3. FP8 attention quality issues**:
```python
# FP8 context attention may show quality degradation on some models
# Mitigation: use FP32 accumulation
plugin_config.context_fmha_type = "enabled_with_fp32_acc"
plugin_config.use_fp8_context_fmha = True
# This uses FP8 Tensor Cores but accumulates in FP32
```

**4. Long-context decode latency**:
```python
# Enable multi-block mode for better parallelism with long KV sequences
plugin_config.multi_block_mode = True

# Increase tokens_per_block to reduce block table overhead
plugin_config.tokens_per_block = 128  # Larger blocks for long contexts

# Use FP8 KV cache to reduce memory bandwidth requirements
build_config.quant_config = QuantConfig(kv_cache_quant=QuantAlgo.FP8)
```

## Kernel Dispatch Flow Diagram

```
                    Input Request
                         │
                         ▼
                 ┌───────────────┐
                 │  GPT Attention │
                 │    Plugin      │
                 └───────┬───────┘
                         │
                    ┌────┴────┐
                    ▼         ▼
              ┌──────────┐  ┌──────────┐
              │ Context  │  │Generation│
              │ (Prefill)│  │ (Decode) │
              └────┬─────┘  └────┬─────┘
                   │              │
             ┌─────┴─────┐  ┌────┴──────┐
             ▼           ▼  ▼           ▼
        ┌─────────┐ ┌──────┐ ┌───────┐ ┌─────────┐
        │  FMHA   │ │cuDNN │ │  XQA  │ │ Paged   │
        │(FP16/   │ │(FP8/ │ │(GQA/  │ │Attention│
        │ BF16)   │ │ FP16)│ │ MQA)  │ │(general)│
        └─────────┘ └──────┘ └───────┘ └─────────┘
             │           │       │           │
             └─────┬─────┘  ┌────┴──────┐    │
                   │        ▼           ▼    │
                   │   ┌─────────┐ ┌────────┐│
                   │   │Single-  │ │Multi-  ││
                   │   │block    │ │block   ││
                   │   └─────────┘ └────────┘│
                   │                         │
                   └────────────┬────────────┘
                                │
                                ▼
                         Output Tensor
```

## Version History

| TRT-LLM Version | Key Attention Features |
|---|---|
| 0.5.0 | Initial FMHA plugin, basic paged KV cache |
| 0.7.0 | XQA kernel for GQA decode, multi-block mode |
| 0.8.0 | FP8 context attention via cuDNN, improved FMHA |
| 0.9.0 | Enhanced XQA with FP8 KV cache, inflight batching improvements |
| 0.10.0 | Expanded head_dim support, MLA attention, Blackwell preview |
| 0.12.0 | Improved speculative decoding attention, chunked context |
| 0.15.0+ | Blackwell-optimized FMHA, FA4-competitive performance |

## Key Notes

- TRT-LLM's multi-kernel dispatch is its key architectural advantage: specialized kernels for each phase and model type
- XQA is the most impactful optimization for modern GQA/MQA models (Llama-3, Mistral, Gemma) -- always verify it is enabled
- The FMHA plugin provides training-quality attention when configured with FP32 accumulation
- FP8 context attention on Hopper provides ~1.7x speedup and is production-ready via cuDNN
- The closed-source nature of TRT-LLM's kernels means you cannot modify them -- if kernel-level customization is needed, consider FlashAttention or FlashInfer
- TRT-LLM requires a build step (TRT engine compilation) which adds deployment complexity compared to vLLM, but provides better runtime performance
- Monitor TRT-LLM releases for kernel improvements; NVIDIA continuously optimizes the attention kernels for new GPU architectures
