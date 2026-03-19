---
skill_name: SageAttention2 Mixed-Precision Quantized Attention
description: INT4 Q/K and FP8 P/V quantized attention with outlier smoothing and two-level accumulation for 3-5x speedup
level: L5 - Numerical Precision Level
target_hardware: NVIDIA RTX 4090 (Ada), NVIDIA H100 (Hopper), general CUDA GPUs with INT4/FP8 support
relevance: When needing faster attention than FlashAttention-2 with better accuracy than FlashAttention-3 FP8, especially for generative models (video, image) where FA3-FP8 degrades quality
---

# SageAttention2 Mixed-Precision Quantized Attention

## What It Is
SageAttention2 is a quantized attention kernel that uses INT4 for Q*K^T computation and FP8 for P*V computation, achieving 3-5x speedup over FlashAttention-2 with negligible accuracy loss. It introduces three key techniques: outlier smoothing that decomposes Q and K by subtracting means, per-thread INT4 quantization aligned with GPU MMA instruction layouts, and a two-level accumulation strategy that recovers full FP32 precision from the hardware's actual FP22 accumulators.

## Key Concepts
- **INT4 Q, K quantization:** The QK^T matmul uses INT4 with per-thread granularity, where each GPU thread handles its own quantization group matching the MMA instruction layout
- **FP8 P, V quantization:** Attention weights and values use E4M3 format; V uses per-channel scaling, P uses static 1/448 scaling
- **Outlier smoothing:** Subtracting per-row means from Q and per-matrix mean from K eliminates outliers that destroy INT4 accuracy, boosting cosine similarity from 80% to 99.5%
- **Two-level accumulation:** Hardware FP8 mma accumulators are actually FP22 (13 mantissa bits), not full FP32. Maintaining separate R (FP22) and O (FP32) accumulators recovers full precision.
- **Correction terms:** The mean-subtraction decomposition generates additive correction terms (delta_S) computed separately in higher precision

## Precision Trade-offs
- **SageAttention2 vs FlashAttention-2:** 3-5x faster, negligible accuracy loss (6.013 vs 6.019 perplexity on Llama 3.1, MMLU 0.635 vs 0.634)
- **SageAttention2 vs FlashAttention-3 FP8:** Same speed on Hopper, dramatically better accuracy on generative tasks (VQA 74.4 vs 2.2 on CogvideoX)
- **With vs without smoothing:** Cosine similarity 99.46% vs 80.04% -- smoothing is essential for INT4 accuracy
- **FP22 vs FP32 accumulation:** The hardware's FP22 accumulator silently degrades precision; two-level accumulation recovers it at minimal cost
- **INT4 QK^T vs FP8 QK^T:** INT4 is more aggressive but with smoothing achieves better accuracy than FA3's FP8 approach, since smoothing removes the outliers that cause quantization error

## Code / Configuration
```python
# SageAttention2 installation and usage
# pip install sageattention

from sageattention import sageattn

import torch

# Basic usage - drop-in replacement for attention
query = torch.randn(2, 32, 1024, 128, dtype=torch.float16, device='cuda')
key = torch.randn(2, 32, 1024, 128, dtype=torch.float16, device='cuda')
value = torch.randn(2, 32, 1024, 128, dtype=torch.float16, device='cuda')

# SageAttention2 with automatic quantization and smoothing
output = sageattn(query, key, value, is_causal=False)

# Causal attention
output_causal = sageattn(query, key, value, is_causal=True)

# === Integration with existing models ===
# Replace F.scaled_dot_product_attention calls:
#   Before: out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
#   After:  out = sageattn(q, k, v, is_causal=True)

# === Understanding the internal pipeline ===
# Conceptual breakdown (actual implementation is fused):

def sage_attention2_conceptual(Q, K, V):
    """
    Conceptual SageAttention2 pipeline.
    In practice, all steps are fused into a single kernel.
    """
    # Step 1: Outlier smoothing
    Q_mean = Q.mean(dim=-1, keepdim=True)
    K_mean = K.mean(dim=-2, keepdim=True)
    Q_smooth = Q - Q_mean
    K_smooth = K - K_mean

    # Step 2: INT4 quantization of smoothed Q, K (per-thread granularity)
    # Each thread quantizes its own elements based on MMA layout
    Q_int4, q_scale = per_thread_int4_quantize(Q_smooth)
    K_int4, k_scale = per_thread_int4_quantize(K_smooth)

    # Step 3: Compute QK^T in INT4 + correction terms
    S_approx = int4_matmul(Q_int4, K_int4, q_scale, k_scale)
    delta_S = compute_correction(Q_mean, K_mean)  # Higher precision
    S = (S_approx + delta_S) / (Q.shape[-1] ** 0.5)

    # Step 4: Softmax in FP32
    P = torch.softmax(S, dim=-1)

    # Step 5: FP8 quantization of P and V
    P_fp8 = (P / 448.0).to(torch.float8_e4m3fn)  # Static scale
    V_fp8, v_scale = per_channel_fp8_quantize(V)

    # Step 6: Two-level accumulation for P*V
    # Level 1: mma instruction (actual FP22 precision)
    R = fp8_matmul_fp22_accum(P_fp8, V_fp8)
    # Level 2: Accumulate R into true FP32
    O = fp32_accumulate(R, v_scale)

    return O
```

## When to Use
- Video and image generation models where FlashAttention-3 FP8 causes visible quality degradation
- Inference serving where 3-5x attention speedup directly reduces latency
- Models with Q/K outliers that prevent simple FP8 quantization from working well
- When you need FlashAttention-3-level speed with FlashAttention-2-level accuracy
- RTX 4090 deployments where INT4 tensor cores provide maximum throughput

## When NOT to Use
- When FlashAttention-2 speed is sufficient for your latency requirements
- On GPUs without INT4 tensor core support
- For training workloads where backward pass INT4 quantization has not been validated
- When sequence lengths are very short (quantization overhead may not be amortized)
- If your model does not have Q/K outliers and simple FP8 (FlashAttention-3) works accurately

## Key Takeaways
- SageAttention2 achieves 3-5x speedup over FlashAttention-2 with negligible accuracy loss across language, image, and video models
- It matches FlashAttention-3 FP8 speed on Hopper while delivering dramatically better accuracy (especially for generative tasks)
- The outlier smoothing technique (Q/K mean subtraction) is essential: it boosts cosine similarity from 80% to 99.5%
- Hardware FP8 accumulators are actually FP22, not FP32 -- the two-level accumulation strategy recovers full precision
- Drop-in replacement for F.scaled_dot_product_attention in most models

## References
- [SageAttention2 Paper (arXiv 2411.10958)](https://arxiv.org/abs/2411.10958)
- [GitHub Repository](https://github.com/thu-ml/SageAttention)
- [ICML 2025 Proceedings](https://icml.cc/virtual/2025/poster/44114)
