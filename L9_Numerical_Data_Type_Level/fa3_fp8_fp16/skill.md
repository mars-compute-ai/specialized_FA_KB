---
skill_name: FlashAttention-3 FP8 and FP16 Precision Optimization
description: Leveraging Hopper FP8 tensor cores with block quantization and incoherent processing for 1.2 PFLOPS attention
level: L9 - Numerical/Data-Type Level
target_hardware: NVIDIA Hopper H100 (FP8 and FP16 Tensor Cores)
relevance: When optimizing attention throughput on H100 GPUs and needing to decide between FP16 and FP8 precision, or when implementing low-precision attention with controlled accuracy loss
---

# FlashAttention-3 FP8 and FP16 Precision Optimization

## What It Is
FlashAttention-3 is the Hopper-optimized evolution of FlashAttention that exploits asynchronous tensor core execution and FP8 low-precision arithmetic to achieve up to 1.2 PFLOPS on H100 GPUs. It introduces block quantization and incoherent processing (via randomized Hadamard transforms) to mitigate the accuracy loss inherent in moving from FP16 to FP8, achieving 2.6x lower quantization error than naive FP8 approaches.

## Key Concepts
- **FP8 doubles throughput:** H100 FP8 tensor cores deliver 1978 TFLOPS vs. 989 TFLOPS for FP16 -- a 2x hardware throughput gain
- **E4M3 vs. E5M2 formats:** E4M3 (range +/-448) is used for forward pass activations; E5M2 (range +/-57,344) provides wider dynamic range for gradients
- **Block quantization:** Per-block scale factors (vs. per-tensor) preserve local dynamic range within Q, K, V matrices
- **Incoherent processing:** Randomized Hadamard transform spreads outlier values before quantization, reducing error by 2.6x
- **Asynchronous execution:** Warp-specialization overlaps GEMM and softmax computation via pingpong scheduling
- **75-85% hardware utilization:** Up from 35% in FlashAttention-2

## Precision Trade-offs
- **FP16 on H100:** 740 TFLOPS (75% utilization), 1.5-2x faster than FA2. No accuracy concerns for standard workloads.
- **BF16 on H100:** 840 TFLOPS (85% utilization). Better dynamic range than FP16 at same throughput tier.
- **FP8 on H100:** Up to 1.2-1.3 PFLOPS. Requires block quantization + incoherent processing to maintain accuracy.
- **FP8 without incoherent processing:** RMSE of 2.4e-2 (baseline) -- may be unacceptable for long-context or precision-sensitive tasks.
- **FP8 with full mitigation:** RMSE of 9.1e-3 -- acceptable for most training and inference workloads.
- **Hadamard transform cost:** O(d log d) per head dimension, memory-bandwidth bound, fusible with rotary embedding for near-zero overhead.

## Code / Configuration
```python
# FlashAttention-3 is accessed through the flash-attn package
# pip install flash-attn (requires Hopper GPU for FP8)

import torch
from flash_attn import flash_attn_func

# FP16 attention (default, works on Ampere and Hopper)
q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16, device='cuda')
k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16, device='cuda')
v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16, device='cuda')
out = flash_attn_func(q, k, v, causal=True)

# FP8 attention (Hopper H100 only)
# Requires FP8 tensors -- typically quantized with per-block scaling
q_fp8 = q.to(torch.float8_e4m3fn)  # E4M3 format for forward pass
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)

# With NVIDIA Transformer Engine for managed FP8:
import transformer_engine.pytorch as te
# Transformer Engine handles scaling and format selection automatically

# Manual block quantization pattern:
def block_quantize_fp8(tensor, block_size=32):
    """Quantize tensor to FP8 with per-block scaling."""
    B = tensor.reshape(-1, block_size)
    scales = B.abs().amax(dim=-1, keepdim=True) / 448.0  # E4M3 max
    quantized = (B / scales).to(torch.float8_e4m3fn)
    return quantized, scales

# Incoherent processing pattern (conceptual):
def apply_hadamard_incoherence(Q, K):
    """Apply randomized Hadamard transform before FP8 quantization."""
    d = Q.shape[-1]
    # Random sign vector (fixed per head for reproducibility)
    signs = torch.randint(0, 2, (d,), device=Q.device) * 2 - 1
    Q_inc = hadamard_transform(Q * signs)  # O(d log d)
    K_inc = hadamard_transform(K * signs)
    return Q_inc, K_inc
```

## When to Use
- Deploying attention on NVIDIA H100 (Hopper) GPUs where FP8 tensor cores are available
- Throughput-critical inference serving where 2x speedup from FP8 justifies implementation complexity
- Training large models where attention is the bottleneck and FP8 accuracy is acceptable
- Long-context workloads (e.g., 128K+ tokens) where the memory bandwidth savings of FP8 are most impactful
- When FlashAttention-2 achieves insufficient hardware utilization on Hopper (35% vs. 75-85%)

## When NOT to Use
- On Ampere (A100) or older GPUs that lack FP8 tensor core support -- use FA2 with FP16/BF16 instead
- Tasks requiring bit-exact reproducibility across runs (FP8 quantization introduces variability)
- Applications with extreme precision requirements (scientific computing, certain financial models)
- When sequence lengths are very short (overhead of Hadamard transform and block scaling may not be amortized)
- If the model has not been validated for FP8 attention accuracy on your specific workload

## Key Takeaways
- FlashAttention-3 achieves 1.5-2x speedup over FA2 in FP16 and up to 1.2 PFLOPS in FP8 on H100
- The combination of block quantization and incoherent processing (randomized Hadamard transform) reduces FP8 quantization error by 2.6x
- The Hadamard transform runs in O(d log d) and can be fused with rotary embedding for near-zero overhead
- FP8 attention with full mitigation achieves RMSE of 9.1e-3 vs. 2.4e-2 for naive FP8
- Hardware utilization jumps from 35% (FA2) to 75-85% (FA3) on H100

## References
- [FlashAttention-3 Paper (arXiv 2407.08608)](https://arxiv.org/abs/2407.08608)
- [FlashAttention-3 Blog Post](https://tridao.me/blog/2024/flash3/)
- [Flash-Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [NVIDIA FP8 Introduction](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
