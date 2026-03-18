# QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

**Authors:** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa
**Published:** February 2024 (ICML 2024)
**arXiv:** [2402.04396](https://arxiv.org/abs/2402.04396)
**Code:** [https://github.com/Cornell-RelaxML/quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp)

## Overview

QuIP# is a post-training quantization method for LLMs using three core innovations:
1. **Randomized Hadamard transforms** for incoherence processing
2. **E8 lattice codebooks** for hardware-efficient vector quantization
3. **Fine-tuning** during quantization to improve fidelity

## Randomized Hadamard Transform (RHT) for Incoherence Processing

### Core Concept
Incoherence processing suppresses outliers by transforming weight matrices to have concentrated entry magnitudes. QuIP# replaces the Kronecker product approach from prior work (QuIP) with the faster and theoretically superior RHT.

### Definition of Incoherence
A matrix H is mu-incoherent if maximum entry magnitudes in its eigenvector decomposition satisfy bounds preventing large coordinate-wise values. Lower incoherence means more evenly distributed values, which are easier to quantize.

### RHT Algorithm
The transformation applies **H * S * x** where:
- **H** is a Hadamard matrix (entries are +/-1, orthogonal)
- **S** is a diagonal matrix of random +/-1 elements

Properties:
- **Improved incoherence parameter:** mu_H = sqrt(2 * log(2n^2 / delta)) compared to prior log-squared dependence
- **Faster runtime:** O(n log n) versus O(n * sqrt(n)) for Kronecker methods
- **No floating-point multiplies needed** (entries are +/-1 only)

### Incoherence Processing Steps
1. Sample random signs S_U in {+/-1}^m, S_V in {+/-1}^n
2. Transform weights: W_hat = Had(diag(S_U) * Had(diag(S_V) * W^T)^T)
3. Transform Hessian similarly: H_hat = Had(diag(S_V) * Had(diag(S_V) * H)^T)

## Error Reduction Through Incoherence

### Theoretical Foundation
The per-layer proxy loss formulation is:

```
l(W_hat) = E_x[||(W_hat - W)x||^2] = tr((W_hat - W) * H * (W_hat - W)^T)
```

where H represents the Hessian matrix capturing importance of different quantization directions.

### Block LDLQ Theorem
When using g-block LDL decomposition with mu-incoherent Hessian:

```
E[tr((W_hat - W) * H * (W_hat - W)^T)] <= (g * m * mu^2 * sigma^2 / n) * tr(H^(1/2))^2
```

The improvement factor emerges from reducing tr(H) to tr(H^(1/2))^2 / n -- the same advantage achieved in scalar quantization. **This represents approximately 2-2.6x error reduction** depending on matrix condition.

### Empirical Confirmation
In FlashAttention-3 experiments with simulated outliers (0.1% of values with large magnitudes), incoherent processing reduced quantization error by exactly **2.6x**, consistent with the theoretical prediction.

## E8 Lattice Codebook Design

### Why E8
The E8 lattice achieves the **optimal 8-dimensional unit-ball packing** (highest kissing number). After incoherence processing, weights follow approximately ball-shaped sub-Gaussian distributions -- E8 codebooks match this shape better than hypercube alternatives.

### E8P Construction (16-bit codebook)
Encodes using:
- **8 bits:** lookup into 256-entry source codebook S (elements of D8_hat with norm <= sqrt(10))
- **7 bits:** sign flips with even/odd parity
- **1 bit:** +/-1/4 shift

This clever decomposition avoids storing all 2^16 entries explicitly while maintaining fast inference.

### Efficiency Advantage
E8-based codebooks demonstrate lower mean-squared error quantizing Gaussians compared to D4 (4D) and other alternatives due to dimension and packing density.

## Experimental Results

### WikiText2 Perplexity (Llama 2-70B, context 2048)

| Bits | FP16 Baseline | QuIP# | Gap |
|------|---------------|-------|-----|
| 2    | 3.32          | 4.16  | +0.84 |
| 3    | 3.32          | 3.56  | +0.24 |
| 4    | 3.32          | 3.38  | +0.06 |

### Key Achievements
- **3-bit models outperform theoretical lossless 4-bit quantization** -- first documented instance where intermediate bitrates surpass higher-precision variants
- Maintains **>99%** of FP16 performance at 4 bits
- Achieves **95%+** performance at 3 bits on zeroshot tasks
- Reaches **~85%** at 2 bits (compared to <50% for OmniQuant)

### Scaling Pattern
Unlike prior work claiming 4-bit is optimal, QuIP# demonstrates 2-bit performance approaching 3-bit quality, suggesting further improvements are possible at extreme compression.

### Inference Performance
Proof-of-concept CUDA implementation on RTX 4090:
- Achieves >50% peak memory bandwidth
- Generation speed: 32-170 tokens/second depending on model size and bitrate
- 4-bit generation runs ~50% faster than 2-bit (both require two Hadamard multiplies per layer)

## Comparison to Prior Methods

QuIP# outperforms AWQ, OmniQuant, and the original QuIP across extreme compression regimes:
- RHT improvement alone yields measurable perplexity gains
- Combined with lattice codebooks and fine-tuning, total improvements exceed **3 perplexity points at 2 bits** for large models

## Relevance to FlashAttention-3

FlashAttention-3 directly adopts the incoherent processing technique from QuIP# for its FP8 implementation:
- The randomized Hadamard transform spreads outlier values in Q, K matrices before FP8 quantization
- Achieves the predicted ~2.6x error reduction in practice
- The O(n log n) runtime and memory-bandwidth-bound nature allow it to be fused with rotary embedding at minimal cost

## References

- [QuIP# Paper (arXiv)](https://arxiv.org/abs/2402.04396)
- [QuIP# HTML Version](https://arxiv.org/html/2402.04396v1)
- [GitHub Repository](https://github.com/Cornell-RelaxML/quip-sharp)
- [ICML 2024 Proceedings](https://proceedings.mlr.press/v235/tseng24a.html)
