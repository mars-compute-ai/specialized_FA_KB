---
skill_name: FP8 Data Formats and Scaling Strategies for Attention
description: Understanding E4M3/E5M2 formats and per-tensor/block/MXFP8 scaling for FP8 attention and training
level: L5 - Numerical Precision Level
target_hardware: NVIDIA Hopper H100 (FP8 Tensor Cores), NVIDIA Blackwell (MXFP8 native)
relevance: When deploying FP8 training or inference on Hopper/Blackwell GPUs and needing to select the right FP8 sub-format, scaling strategy, and precision safeguards
---

# FP8 Data Formats and Scaling Strategies for Attention

## What It Is
FP8 is a family of 8-bit floating-point formats (E4M3 and E5M2) that approximately halve memory usage and double tensor core throughput compared to FP16. Proper scaling strategies (per-tensor, delayed, or MXFP8 block scaling) are essential to maintain accuracy. MXFP8 on Blackwell GPUs divides tensors into 32-element blocks with dedicated scaling factors handled natively by Tensor Cores, providing the best balance of throughput and precision.

## Key Concepts
- **E4M3:** 4 exponent, 3 mantissa bits. Range +/-448. Optimized for forward pass (weights, activations, QKV). Higher precision.
- **E5M2:** 5 exponent, 2 mantissa bits. Range +/-57,344. Optimized for backward pass (gradients). Wider dynamic range.
- **FP8 vs INT8:** FP8 provides implicit scaling via exponents, handling transformer attention scores (near-zero to thousands) better than fixed-point INT8
- **Per-tensor scaling:** Single scale factor per tensor. Simple but coarse.
- **Delayed scaling:** Uses historical amax values. Stabilizes training but lags rapid changes.
- **MXFP8 block scaling:** Per-32-element blocks with E8M0 scale factors. Fine-grained, hardware-native on Blackwell.
- **Master weights in FP32:** Always maintain high-precision copies of weights and optimizer states

## Precision Trade-offs
- **FP8 E4M3 vs FP16:** 2x throughput, 3 mantissa bits vs 10 mantissa bits. Range +/-448 vs +/-65,504. Requires scaling to compensate.
- **FP8 E4M3 vs BF16:** 2x throughput, 3 mantissa bits vs 7 mantissa bits. BF16 has much wider range (1e38) without scaling.
- **MXFP8 vs per-tensor FP8:** Block scaling accommodates within-tensor magnitude variations at cost of additional scale factor storage (1 byte per 32 elements).
- **Training convergence:** MXFP8 validation perplexity follows BF16 closely with negligible degradation for 8B parameter models.
- **Attention accuracy:** FP8 attention with block quantization + incoherent processing achieves RMSE 9.1e-3 vs 2.4e-2 for naive FP8 (2.6x improvement).

## Code / Configuration
```python
import torch
import transformer_engine.pytorch as te

# === NVIDIA Transformer Engine: Automatic FP8 Management ===
# The recommended approach for production FP8 training

# Create a Transformer Engine linear layer (handles FP8 internally)
linear = te.Linear(hidden_size, hidden_size, bias=True)

# FP8 training context
with te.fp8_autocast(enabled=True):
    output = linear(input_tensor)

# === Manual FP8 Quantization ===
# For custom kernels or understanding the internals

def quantize_to_fp8_e4m3(tensor, per_block=False, block_size=32):
    """Quantize a tensor to FP8 E4M3 format with scaling."""
    if per_block:
        # MXFP8-style block quantization
        shape = tensor.shape
        tensor_flat = tensor.reshape(-1, block_size)
        amax = tensor_flat.abs().amax(dim=-1, keepdim=True)
        scale = amax / 448.0  # E4M3 max representable value
        scale = scale.clamp(min=1e-12)  # Avoid division by zero
        quantized = (tensor_flat / scale).to(torch.float8_e4m3fn)
        return quantized.reshape(shape), scale.reshape(-1)
    else:
        # Per-tensor quantization
        amax = tensor.abs().amax()
        scale = amax / 448.0
        quantized = (tensor / scale).to(torch.float8_e4m3fn)
        return quantized, scale

def quantize_to_fp8_e5m2(tensor):
    """Quantize gradients to FP8 E5M2 format."""
    amax = tensor.abs().amax()
    scale = amax / 57344.0  # E5M2 max representable value
    quantized = (tensor / scale).to(torch.float8_e5m2)
    return quantized, scale

# === FP8 Attention Pattern ===
def fp8_attention_forward(Q, K, V):
    """FP8 forward pass for attention (conceptual)."""
    # Quantize Q, K, V to E4M3 for forward pass
    Q_fp8, q_scale = quantize_to_fp8_e4m3(Q, per_block=True)
    K_fp8, k_scale = quantize_to_fp8_e4m3(K, per_block=True)
    V_fp8, v_scale = quantize_to_fp8_e4m3(V, per_block=True)

    # Compute attention in FP8 (actual implementation uses tensor cores)
    # Accumulator is in FP32 for precision
    scores = (Q_fp8.float() * q_scale.unsqueeze(-1)) @ \
             (K_fp8.float() * k_scale.unsqueeze(-1)).transpose(-2, -1)
    scores = scores / (Q.shape[-1] ** 0.5)

    attn_weights = torch.softmax(scores, dim=-1)  # Softmax in FP32

    # Apply attention to values
    output = attn_weights @ (V_fp8.float() * v_scale.unsqueeze(-1))
    return output

# === Controlling FP8 Behavior in PyTorch ===
# Disable reduced-precision accumulation for debugging
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# Check if FP8 is supported on current hardware
if torch.cuda.get_device_capability()[0] >= 9:  # Hopper (sm90+)
    print("FP8 Tensor Cores available")
```

## When to Use
- Training large models (7B+ parameters) on Hopper or Blackwell GPUs where FP8 tensor cores provide 2x throughput
- Inference serving where memory bandwidth is the bottleneck and FP8 halves data movement
- Long-context attention where the O(N^2) attention score computation benefits most from reduced precision
- When NVIDIA Transformer Engine is available to manage FP8 scaling automatically
- Production deployments where BF16 training has been validated and FP8 can be adopted incrementally

## When NOT to Use
- On GPUs without FP8 Tensor Core support (Ampere A100 and earlier)
- For fine-tuning tasks where precision is critical and the model is small enough that FP16 throughput is sufficient
- When the model has not been validated for FP8 convergence on your specific task
- For scientific computing or applications requiring more than 3 mantissa bits of precision
- During the early exploration phase of model architecture development where numerical debugging is frequent

## Key Takeaways
- E4M3 is for forward pass (precision-optimized), E5M2 is for backward pass (range-optimized)
- MXFP8 block scaling (32 elements per block) provides the best accuracy-throughput trade-off on Blackwell
- FP8 training converges as well as BF16 with proper scaling (validated on 8B parameter models)
- Always keep master weights and optimizer states in FP32
- NVIDIA Transformer Engine automates FP8 format selection and scaling factor management
- FP8 provides 2x Tensor Core throughput over FP16, making attention compute-bound rather than memory-bound

## References
- [NVIDIA FP8 Introduction Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [NVIDIA Transformer Engine FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [FP8 Formats for Deep Learning (arXiv 2209.05433)](https://arxiv.org/abs/2209.05433)
