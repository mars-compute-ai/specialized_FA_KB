---
skill_name: Incoherent Processing via Randomized Hadamard Transform
description: Using Hadamard transforms to spread outlier values before quantization, reducing quantization error by ~2.6x
level: L5 - Numerical Precision Level
target_hardware: General (GPU with fast Hadamard kernel support; NVIDIA Hopper H100 for FP8 attention)
relevance: When quantizing weights or activations to low precision (FP8, INT4, INT2) and needing to mitigate accuracy loss from outlier values, especially in attention Q/K matrices or LLM weight compression
---

# Incoherent Processing via Randomized Hadamard Transform

## What It Is
Incoherent processing is a pre-quantization technique that applies a randomized Hadamard transform to "spread out" outlier values in weight or activation matrices, making them more amenable to uniform quantization. Introduced in QuIP and refined in QuIP#, this technique reduces quantization error by approximately 2.6x and has been adopted by FlashAttention-3 for FP8 attention. The transform runs in O(n log n) time with only +/-1 multiplications, making it extremely efficient.

## Key Concepts
- **The outlier problem:** LLM weights and activations often contain a small fraction (~0.1%) of values with disproportionately large magnitudes. These outliers dominate quantization error because uniform quantization must waste dynamic range to accommodate them.
- **Incoherence:** A matrix is "mu-incoherent" if its entries are bounded in magnitude. Lower mu means more evenly distributed values. The Hadamard transform achieves mu_H = sqrt(2 * log(2n^2 / delta)).
- **Randomized Hadamard Transform (RHT):** Applies H * S * x where H is a Hadamard matrix and S is a diagonal of random +/-1 signs. This preserves norms (orthogonal transform) while spreading outliers across all dimensions.
- **No floating-point multiplies:** Hadamard matrices contain only +/-1 entries, so the transform involves only additions and subtractions.
- **O(n log n) runtime:** Uses the Fast Walsh-Hadamard Transform, faster than the O(n * sqrt(n)) Kronecker product approach in the original QuIP.
- **Block LDLQ quantization:** After incoherence processing, block-wise quantization with LDL decomposition achieves error proportional to tr(H^(1/2))^2 rather than tr(H), yielding the 2-2.6x improvement factor.

## Precision Trade-offs
- **Without incoherent processing (FP8 attention):** RMSE of 2.4e-2 for standard FP8 quantization of attention matrices
- **With incoherent processing (FP8 attention):** RMSE of 9.1e-3 -- a 2.6x error reduction
- **Weight quantization (QuIP# at 2-bit):** Perplexity of 4.16 on Llama 2-70B vs. 3.32 for FP16 (within 0.84 points)
- **Weight quantization (QuIP# at 4-bit):** Perplexity of 3.38 vs. 3.32 for FP16 (within 0.06 points -- near-lossless)
- **Compute overhead:** O(d log d) per head dimension, memory-bandwidth bound. Can be fused with rotary embedding for near-zero marginal cost in attention.
- **Inference overhead:** Two Hadamard multiplies per layer (one for Q/K, one for reversal), but these are bandwidth-bound and overlap with other operations.

## Code / Configuration
```python
import torch

# === Fast Walsh-Hadamard Transform ===
def hadamard_transform(x):
    """
    In-place Fast Walsh-Hadamard Transform.
    x must have last dimension as a power of 2.
    O(n log n) with only additions/subtractions.
    """
    n = x.shape[-1]
    h = 1
    while h < n:
        # Butterfly operation: only additions and subtractions
        x_even = x[..., 0::2*h, :]  # conceptual; actual impl uses views
        x_odd = x[..., h::2*h, :]
        x[..., 0::2*h, :] = x_even + x_odd
        x[..., h::2*h, :] = x_even - x_odd
        h *= 2
    return x / (n ** 0.5)  # Normalize

# === Randomized Hadamard Transform for Incoherent Processing ===
def apply_incoherent_processing(tensor, seed=42):
    """
    Apply randomized Hadamard transform to spread outliers.
    Use same seed for paired transforms (e.g., Q and K).
    """
    d = tensor.shape[-1]
    # Generate deterministic random signs
    gen = torch.Generator(device=tensor.device).manual_seed(seed)
    signs = torch.randint(0, 2, (d,), generator=gen,
                          device=tensor.device) * 2 - 1
    # Apply: H * diag(signs) * tensor
    return hadamard_transform(tensor * signs.float())

# === Usage in FP8 Attention (FlashAttention-3 pattern) ===
def fp8_attention_with_incoherence(Q, K, V, seed=42):
    """
    FP8 attention with incoherent processing for accuracy.
    """
    # Step 1: Apply incoherent processing (fuse with rotary embedding)
    Q_inc = apply_incoherent_processing(Q, seed=seed)
    K_inc = apply_incoherent_processing(K, seed=seed)

    # Step 2: Block quantize to FP8
    Q_fp8, q_scales = block_quantize_fp8(Q_inc)
    K_fp8, k_scales = block_quantize_fp8(K_inc)
    V_fp8, v_scales = block_quantize_fp8(V)

    # Step 3: Run attention in FP8
    # (actual FA3 kernel handles this internally)
    attn_out = flash_attn_fp8(Q_fp8, K_fp8, V_fp8,
                               q_scales, k_scales, v_scales)
    return attn_out

# === Usage in Weight Quantization (QuIP# pattern) ===
def quantize_layer_quip_sharp(weight, hessian, n_bits=4):
    """
    QuIP# weight quantization with incoherent processing.
    """
    m, n = weight.shape
    # Step 1: Random signs for both dimensions
    S_U = torch.randint(0, 2, (m,), device=weight.device) * 2 - 1
    S_V = torch.randint(0, 2, (n,), device=weight.device) * 2 - 1

    # Step 2: Apply RHT to weight and Hessian
    W_hat = hadamard_transform(
        (weight * S_U.unsqueeze(1)).T * S_V.unsqueeze(1)
    ).T
    H_hat = hadamard_transform(
        (hessian * S_V.unsqueeze(1)).T * S_V.unsqueeze(1)
    ).T

    # Step 3: Block LDLQ quantization on incoherent matrices
    W_quantized = block_ldlq_quantize(W_hat, H_hat, n_bits)
    return W_quantized, S_U, S_V  # Store signs for dequantization
```

## When to Use
- Quantizing attention Q/K matrices to FP8 on Hopper GPUs (FlashAttention-3 use case)
- Extreme weight compression (2-4 bit) for LLM deployment where outliers cause accuracy degradation
- Any scenario where a small fraction of values have disproportionately large magnitudes
- When standard per-tensor or per-channel quantization produces unacceptable accuracy loss
- Combined with block quantization for compound error reduction

## When NOT to Use
- When quantizing to 8+ bits where the precision is sufficient to represent outliers directly
- When weight/activation distributions are already well-behaved (no outliers) -- the transform adds overhead without benefit
- On hardware without efficient Hadamard kernel support (the overhead may not be amortized)
- When the head dimension is not a power of 2 (Hadamard transform requires power-of-2 dimensions, though padding is possible)
- For tasks requiring exact reversibility without storing the random sign vectors

## Key Takeaways
- The randomized Hadamard transform reduces quantization error by ~2.6x by spreading outlier values across all dimensions
- It runs in O(n log n) with only additions/subtractions (no floating-point multiplies), making it extremely efficient
- FlashAttention-3 adopts this technique from QuIP# for FP8 attention, fusing it with rotary embedding at near-zero cost
- QuIP# achieves near-lossless 4-bit quantization (within 0.06 perplexity of FP16) on Llama 2-70B
- The theoretical foundation shows error scales with tr(H^(1/2))^2 rather than tr(H) after incoherence processing

## References
- [QuIP# Paper (arXiv 2402.04396)](https://arxiv.org/abs/2402.04396)
- [QuIP# GitHub](https://github.com/Cornell-RelaxML/quip-sharp)
- [FlashAttention-3 Paper (arXiv 2407.08608)](https://arxiv.org/abs/2407.08608)
- [HadaCore: Tensor Core Accelerated Hadamard Transform](https://pytorch.org/blog/hadacore/)
