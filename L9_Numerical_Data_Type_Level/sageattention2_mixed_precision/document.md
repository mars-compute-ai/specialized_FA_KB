# SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

**Authors:** Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, Jianfei Chen
**Published:** November 2024
**arXiv:** [2411.10958](https://arxiv.org/abs/2411.10958)
**Code:** [https://github.com/thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)
**Venues:** ICLR 2025, ICML 2025, NeurIPS 2025 Spotlight

## Overview

SageAttention2 achieves 2-5x speedup over FlashAttention by quantizing Q, K matrices to INT4 and P, V matrices to FP8, with three key accuracy-preserving techniques: outlier smoothing for Q+K, per-thread INT4 quantization granularity, and two-level FP32 accumulation for FP8 matmuls. It matches FlashAttention-3 FP8 speed on Hopper while delivering significantly higher accuracy.

## Core Quantization Architecture

### INT4 for Q, K (Per-Thread Granularity)
- Q and K matrices are quantized to INT4 for the QK^T matmul
- Uses hardware-aligned per-thread quantization: different GPU threads handle distinct quantization groups based on MMA instruction layout
- Each thread performs dequantization using only a single quantization scale value
- This balances precision with computational efficiency, avoiding overhead of managing multiple scales per thread

### FP8 for P, V (E4M3 Format)
- Unnormalized attention weights (P_tilde) and values (V) are quantized to FP8 E4M3
- P_tilde uses a static scale of 1/448 (since attention weights after softmax are bounded)
- V is quantized per-channel to address outliers in value matrices
- FP8 preserves small values in P_tilde whose sum is non-negligible

## Outlier Smoothing Technique

### The Problem
INT4 quantization is severely affected by outlier values in Q and K matrices. Without smoothing, cosine similarity between quantized and full-precision attention can drop to 80% or lower.

### The Solution: Q+K Decomposition
Decompose matrices by subtracting means:
```
Q_smoothed = Q_i - mean(Q_i)
K_smoothed = K_j - mean(K)
```

The attention computation splits into:
```
gamma(Q_i) * gamma(K_j)^T + delta_S_ij
```

where gamma represents the smoothed (quantized) computation and delta_S represents correction terms computed in higher precision.

### Results
- With smoothing: **99.46%** cosine similarity (CogvideoX benchmark)
- Without smoothing: **80.04%** cosine similarity
- Smoothing Q+K together is critical; smoothing only Q or only K is insufficient

## Two-Level Accumulation Strategy

### The Discovery
The FP8 matmul accumulator on Ada/Hopper architectures (`mma.f32.f8.f8.f32`) actually uses **FP22** precision (1 sign bit, 8 exponent bits, 13 mantissa bits), not full FP32. This means even with nominal FP32 accumulators, FP8 matmul results have reduced precision.

### The Solution
Maintain two sets of accumulators in registers:
1. **R_ij:** Computed with the mma instruction at FP22 effective precision
2. **O_ij:** Accumulated from R_ij in true FP32 precision

This two-level strategy recovers the full FP32 accumulation precision that the hardware nominally promises but does not deliver.

## Performance Metrics

### Kernel-Level Performance
| GPU | Throughput | Speedup vs FA2 | Speedup vs xformers |
|-----|-----------|----------------|---------------------|
| RTX 4090 | 481 TOPS | 3x | 4.5x |
| H100 | Matches FA3-FP8 | ~3x | ~4x |

- On Hopper GPUs: matches FlashAttention-3(fp8) speed while delivering "much higher accuracy"
- Memory bandwidth savings: 3x over baseline FP16 kernels

### End-to-End Accuracy Results

**Language Models (Llama 3.1):**
| Metric | FlashAttention2 | SageAttention2 |
|--------|-----------------|----------------|
| WikiText Perplexity | 6.013 | 6.019 |
| MMLU Accuracy | 0.635 | 0.634 |

**Video Generation (CogvideoX 1.5-5B):**
| Metric | FlashAttention2 | SageAttention2 | FA3(fp8) |
|--------|-----------------|----------------|----------|
| VQA Score | 75.360 | 74.415 | 2.181 |
| Speedup | 1.0x | 1.8x | ~1.8x |

Key finding: FA3(fp8) suffers severe degradation on video generation tasks (VQA 2.181 vs 75.360), while SageAttention2 maintains quality (74.415).

**Image Generation (Flux):**
| Metric | FlashAttention2 | SageAttention2 |
|--------|-----------------|----------------|
| FID Score | 10.960 | 10.927 |

## Comparison with FlashAttention-3 FP8

SageAttention2 achieves comparable speed to FA3-FP8 on Hopper with substantially better accuracy through:
1. **Comprehensive outlier handling:** Smooths both Q and K, while FA3 uses simpler quantization
2. **Per-thread granularity:** Finer quantization groups aligned with MMA instructions
3. **Two-level accumulation:** Recovers full FP32 precision from nominal FP22 accumulators
4. **INT4 for QK^T:** More aggressive quantization for the largest matmul, compensated by smoothing

Videos generated using FlashAttention3(fp8) "often suffer noticeable degradation" while SageAttention2 maintains visual fidelity.

## References

- [SageAttention2 Paper (arXiv 2411.10958)](https://arxiv.org/abs/2411.10958)
- [GitHub Repository](https://github.com/thu-ml/SageAttention)
- [ICML 2025](https://icml.cc/virtual/2025/poster/44114)
