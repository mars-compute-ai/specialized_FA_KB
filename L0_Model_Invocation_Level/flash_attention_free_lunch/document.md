# The Free Lunch of Flash Attention

Source: https://medium.com/better-ml/the-free-lunch-of-flash-attention-036b0040dee2
Additional: https://flashattn.dev/blog/flash-attention-vs-standard-attention

## Overview

Flash Attention is described as a "free lunch" because it achieves speedups while maintaining mathematical correctness. Despite slightly higher floating-point operations due to recomputation, the algorithm runs faster (up to 7.6x on GPT-2) and uses dramatically less memory -- linear in sequence length instead of quadratic -- thanks to massively reduced HBM (High Bandwidth Memory) access.

## Why Naive Attention is Memory-Bound

### The Memory Bandwidth Bottleneck

Modern GPUs have tremendous compute capacity but limited memory bandwidth. Standard attention must:

1. Compute Q x K^T, producing an N x N matrix (written to HBM)
2. Apply scaling and masking (read from HBM, write back)
3. Compute softmax (read from HBM, write back)
4. Apply dropout (read from HBM, write back)
5. Multiply by V (read from HBM, write back)
6. Store the final output

This requires **six separate kernel launches** and the N x N attention matrix is read/written to HBM multiple times. For a sequence length of 4096, this matrix alone is 64MB in fp16 -- and it must be stored for the backward pass.

### The Root Cause

The attention operation is **memory-bound, not compute-bound**. The ratio of memory accesses to floating-point operations is the bottleneck. Standard attention achieves only 30-50% of theoretical peak GPU performance because it spends most of its time waiting for memory transfers.

## How FlashAttention Solves This

### Key Technique 1: Tiling

FlashAttention restructures the attention computation by splitting Q, K, V into blocks and loading them from slow HBM to fast on-chip SRAM. The key insight is performing the softmax operation incrementally by making several passes over input blocks, using the online softmax trick to maintain running statistics.

The tiling approach:
1. Split Q, K, V into blocks that fit in SRAM
2. Load one block of Q and iterate over blocks of K, V
3. Compute partial attention scores in SRAM (fast)
4. Update running softmax normalization statistics (m, l)
5. Write only the final output block back to HBM

### Key Technique 2: Recomputation

In the backward pass, instead of storing the full N x N attention matrix (O(N^2) memory), FlashAttention:
1. Stores only the output O and softmax normalization statistics (m, l) from the forward pass
2. Recomputes the attention matrix S and probability matrix P on-the-fly from blocks of Q, K, V loaded into SRAM
3. This trades extra computation for dramatically reduced memory

Because the forward math is lightweight compared with HBM traffic, recomputation costs little -- especially on GPUs where unused compute cycles abound.

### IO Complexity Improvement

- **Standard attention**: O(N^2) HBM accesses (dominated by reading/writing the N x N matrix)
- **FlashAttention**: O(N^2 * d^2 / M) HBM accesses, where M is SRAM size and d is head dimension

Since for typical GPUs, SRAM size M >> d^2, this represents a **massive reduction** in HBM traffic -- up to 9x fewer accesses in practice.

## Memory Performance Comparison

### Forward Pass Memory (FP16, batch=8, heads=12, d=64)

| Sequence Length | Standard | FlashAttention | Reduction |
|----------------|----------|----------------|-----------|
| 256 tokens     | 24 MB    | 8 MB           | 3x        |
| 1024 tokens    | 384 MB   | 24 MB          | 16x       |
| 4096 tokens    | 6.1 GB   | 96 MB          | 64x       |
| 8192 tokens    | OOM      | 192 MB         | --        |

### Training (Forward + Backward)

| Sequence Length | Standard | FlashAttention | Reduction |
|----------------|----------|----------------|-----------|
| 1024 tokens    | 1.2 GB   | 0.2 GB         | 6x        |
| 2048 tokens    | 4.7 GB   | 0.4 GB         | 12x       |
| 4096 tokens    | 18.9 GB  | 0.8 GB         | 24x       |

## Speed Benchmarks (NVIDIA A100)

### Forward Pass Throughput

| Sequence Length | Speedup |
|----------------|---------|
| 256 tokens     | 1.09x   |
| 1024 tokens    | 1.49x   |
| 2048 tokens    | 2.26x   |
| 4096 tokens    | 4.21x   |

### End-to-End Training

| Model Size | Sequence Length | Speedup |
|-----------|-----------------|---------|
| 125M      | 1024            | 1.24x   |
| 6.7B      | 2048            | 1.86x   |

FlashAttention achieves 50-73% of theoretical peak GPU performance versus 30-50% for standard attention.

## When Standard Attention Wins

### Very Short Sequences (<256 tokens)
At 64 tokens: standard = 2.1ms vs FlashAttention = 2.3ms. FlashAttention's initialization overhead dominates.

### Debugging
Standard attention lets you inspect intermediate attention scores and weights. FlashAttention only returns the final output.

### Custom Sparse Patterns
For irregular sparse attention beyond causal, sliding window, or block-sparse patterns, standard attention with custom masks may be simpler.

### Non-NVIDIA Hardware
FlashAttention requires NVIDIA GPUs with CUDA (or AMD GPUs with ROCm via xFormers/Composable Kernel).

## FlashAttention Version Comparison

### FlashAttention-1 (2022)
- Original IO-aware tiling algorithm
- Introduced online softmax trick for incremental computation
- Up to 7.6x speedup on GPT-2

### FlashAttention-2 (2023)
- ~30% faster than FA1
- Better parallelism across the sequence length dimension
- Better work partitioning between warps
- Reduced non-matmul FLOPs
- Supports head dimensions up to 256
- Supports fp16, bf16

### FlashAttention-3 (2024)
- Optimized specifically for NVIDIA Hopper (H100) GPUs
- 1.5-2.0x speedup over FA2
- Achieves up to 740 TFLOPS (75% of H100 theoretical max)
- FP8 support reaching ~1.2 PFLOPS
- Three key techniques:
  - **Asynchronous operation overlapping** via warp specialization
  - **Interleaved matmul and softmax** (pingpong scheduling)
  - **Incoherent processing** for FP8 with 2.6x error reduction

### FlashAttention-4 (2025)
- Written in CuTeDSL for Hopper and Blackwell GPUs
- Integrates with FlexAttention for custom attention patterns
- 1.2-3.2x speedup over Triton implementations
- Supports JIT kernel specialization

## Decision Guide: Which Version to Pick

| Condition | Recommendation |
|-----------|---------------|
| PyTorch 2.1.1+, standard attention | Use SDPA (auto-selects FA2) |
| Custom score modifications needed | Use FlexAttention + FA4 backend |
| H100 GPU, maximum throughput | Use FA3 directly |
| H100/Blackwell + custom patterns | Use FlexAttention + FA4 |
| A100 GPU | Use FA2 via SDPA or flash_attn |
| Older GPU (V100, T4) | Use FA1 or xFormers CUTLASS |
| AMD GPU | Use xFormers Composable Kernel |
| Need fp32 | Use xFormers CUTLASS or SDPA math backend |
| Need FP8 | Use FA3 or FA4 |
| Sequence length < 256 | Standard attention may be faster |

## Why It Is a "Free Lunch"

The term "free lunch" captures three properties:
1. **Mathematically exact**: Produces identical results to standard attention (within floating-point precision)
2. **Drop-in replacement**: No model architecture changes required
3. **Simultaneously faster AND more memory-efficient**: Despite doing more FLOPs (due to recomputation), the reduced memory traffic makes it faster overall
