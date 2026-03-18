# FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

**Authors:** University of Washington, Carnegie Mellon University, OctoAI
**Source:** https://flashinfer.ai/2024/02/02/introduce-flashinfer.html
**Award:** Best Paper at MLSys 2025

## Overview

FlashInfer is an open-source library (Apache 2.0) designed to accelerate self-attention computations in LLM serving. It provides comprehensive kernel coverage for diverse attention patterns (prefill, decode, append), supports multiple KV-Cache formats (padded tensor, ragged tensor, page table), and achieves near-peak memory bandwidth utilization for decode and superior performance for prefill across A100/H100 GPUs.

## Core Architecture: Three-Stage Attention Pipeline

The system decomposes LLM serving into three distinct attention phases:

### 1. Prefill Stage
Attention computation between KV-Cache and all queries simultaneously. High operational intensity ~O(l_q), compute-bound for long query lengths.

### 2. Decode Stage
Single-token generation with attention between KV-Cache and one query. Operational intensity near O(1), making it **memory-bandwidth bound**.

### 3. Append Stage
Attention for newly appended tokens, useful in speculative decoding scenarios. Intensity between prefill and decode.

## Key Technical Features

### Comprehensive Kernel Coverage
- Single-request and batched prefill kernels
- Decode attention with various KV-Cache formats
- Append kernels for speculative decoding
- Support for Padded Tensor, Ragged Tensor, and Page Table formats

### Grouped-Query Attention (GQA) Optimization
Specialized kernels utilizing Tensor Cores for GQA decode, achieving up to 2-3x speedup compared to vLLM on A100/H100 hardware.

### Fused-RoPE Implementation
Applies rotary positional embeddings on-the-fly within attention kernels, enabling KV-Cache pruning without storing position-dependent cached values.

### Quantized Attention
Low-precision kernels supporting 4-bit and 8-bit quantization for compressed KV-Cache, enabling up to 4x compression speedups.

### Page Table Optimization
Prefetches page indices into GPU shared memory, minimizing performance degradation across different page sizes. Supports page_size=1 for novel KV-Cache management algorithms.

## Performance Optimizations

### Split-K Strategy
Parallelizes KV-Cache across the sequence dimension to increase SM utilization when batch sizes are small.

### Precision Tuning
The `allow_fp16_qk_reduction` option enables fp16 accumulation for QK matrix multiplication while maintaining fp32 for score-value operations.

## Benchmark Results

- **Prefill**: Superior performance vs FlashAttention 2.4.2 across all tested GPUs
- **Single-Request Decode**: Approaches 100% memory bandwidth utilization for extended sequences
- **Batch Decoding**: Consistent speedups over vLLM PageAttention
- **GQA Performance**: 3x speedup vs vLLM at batch_size=64
- **FP8 Quantization**: ~2x acceleration vs fp16 baseline

## Integration Ecosystem
- MLC-LLM, Punica (LoRA serving), SGLang (structured generation)
- Integrated into vLLM serving framework

## API Design
- PyTorch bindings for rapid prototyping
- Header-only C++ API for production serving systems
