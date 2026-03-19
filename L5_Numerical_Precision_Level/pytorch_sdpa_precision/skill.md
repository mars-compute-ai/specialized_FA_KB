---
skill_name: PyTorch SDPA Backend Precision and Reproducibility
description: Understanding numerical differences between SDPA backends and controlling precision via float32 intermediates and backend selection
level: L5 - Numerical Precision Level
target_hardware: General (NVIDIA GPUs with CUDA; CPU fallback)
relevance: When debugging numerical differences in attention outputs, switching between FlashAttention and math backends, or when reproducibility across backends is required
---

# PyTorch SDPA Backend Precision and Reproducibility

## What It Is
PyTorch's `scaled_dot_product_attention` (SDPA) dispatches to multiple backend implementations (FlashAttention, memory-efficient, math, cuDNN), each producing different numerical results. The math backend keeps all intermediates in float32 for FP16/BF16 inputs, providing maximum precision at the cost of speed. Developers must understand these precision differences when switching backends, debugging numerical mismatches, or validating model accuracy across deployment configurations.

## Key Concepts
- **Four backends with different precision:** Math (float32 intermediates), FlashAttention (native half-precision), Memory-Efficient (xformers-based), cuDNN (hardware-optimized, potentially nondeterministic)
- **Automatic float32 upcast:** The math backend upcasts FP16/BF16 inputs to float32, computes all intermediates in float32, then downcasts the result. This is the default precision safeguard.
- **Backend auto-selection:** By default, PyTorch selects the fastest available backend. This means numerical output changes when hardware or library versions change.
- **sdpa_kernel context manager:** Allows explicit backend selection for reproducibility or precision control
- **Nondeterministic cuDNN:** The cuDNN backend may select nondeterministic algorithms for performance
- **TF32 on Ampere+:** NVIDIA Ampere GPUs use TF32 (10-bit mantissa) for matrix multiplications, further reducing precision unless explicitly disabled
- **No bitwise reproducibility guarantee:** PyTorch cannot guarantee identical results across platforms, versions, or even commits due to floating-point non-associativity

## Precision Trade-offs
- **Math backend:** Most precise (float32 intermediates), ~87ms in benchmarks. Use for validation/debugging.
- **FlashAttention backend:** ~2.3ms (~38x faster than math), operates in native FP16/BF16. Slight precision loss but acceptable for training/inference.
- **Memory-Efficient backend:** ~4.4ms (~20x faster than math), intermediate precision characteristics.
- **Default dispatch:** ~2.3ms, auto-selects fastest. Numerical output is hardware-dependent.
- **FP16 reduced-precision reductions:** Can be re-enabled for speed via `allow_fp16_bf16_reduction_math_sdp(True)`, but increases numerical error.
- **TF32 impact:** Reduces mantissa from 23 bits to 10 bits in matrix multiplications on Ampere+ GPUs. Disabled by default for matmul, enabled for convolutions.

## Code / Configuration
```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# === Basic SDPA Usage ===
query = torch.randn(2, 8, 128, 64, dtype=torch.float16, device='cuda')
key = torch.randn(2, 8, 128, 64, dtype=torch.float16, device='cuda')
value = torch.randn(2, 8, 128, 64, dtype=torch.float16, device='cuda')

# Default: auto-selects fastest backend
out_default = F.scaled_dot_product_attention(query, key, value)

# === Explicit Backend Selection ===
# Most precise: math backend with float32 intermediates
with sdpa_kernel(SDPBackend.MATH):
    out_precise = F.scaled_dot_product_attention(query, key, value)

# Fastest for long sequences: FlashAttention
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out_fast = F.scaled_dot_product_attention(query, key, value)

# Memory-efficient attention
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    out_efficient = F.scaled_dot_product_attention(query, key, value)

# === Verifying Numerical Differences ===
print(f"Flash vs Math max diff: {(out_fast - out_precise).abs().max().item():.6f}")
print(f"Efficient vs Math max diff: {(out_efficient - out_precise).abs().max().item():.6f}")

# === Controlling Precision Globally ===
# Disable TF32 for maximum precision on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Disable reduced-precision FP16 reductions
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# Re-enable reduced precision for math SDPA backend (faster, less precise)
# torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

# === Ensuring Determinism ===
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)  # Strictest determinism

# === Causal Attention with Precision Control ===
with sdpa_kernel(SDPBackend.MATH):
    out_causal = F.scaled_dot_product_attention(
        query, key, value,
        is_causal=True,
        dropout_p=0.0  # Dropout adds nondeterminism
    )

# === torch.compile Integration ===
# Compiled SDPA with specific backend
@torch.compile
def compiled_attention(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Compilation can provide ~46% speedup on top of SDPA optimizations
out_compiled = compiled_attention(query, key, value)

# === Validating Numerical Correctness ===
def validate_attention_precision(q, k, v, rtol=1e-3, atol=1e-3):
    """Compare fused backend against math reference."""
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q, k, v)
    out = F.scaled_dot_product_attention(q, k, v)  # auto-dispatch
    match = torch.allclose(out, ref, rtol=rtol, atol=atol)
    max_diff = (out - ref).abs().max().item()
    print(f"Match (rtol={rtol}, atol={atol}): {match}, max diff: {max_diff:.6e}")
    return match
```

## When to Use
- Debugging numerical differences between attention implementations (unit tests failing due to precision)
- Validating model accuracy when switching from math backend to FlashAttention for deployment
- Building reproducible training pipelines where numerical consistency across runs is required
- Profiling which backend provides the best speed/precision trade-off for your workload
- When migrating models across GPU architectures (Ampere to Hopper) and observing numerical changes

## When NOT to Use
- When training standard models where minor numerical differences between backends are acceptable (just use default dispatch)
- When you need maximum throughput and accept the numerical characteristics of the fastest backend
- For quick prototyping where precision is not a concern
- When running on CPU only (only one backend available, no precision choices to make)

## Key Takeaways
- The math backend keeps FP16/BF16 intermediates in float32, serving as the precision reference at ~38x slower than FlashAttention
- Different SDPA backends produce different numerical results -- never use exact equality for testing
- Use `sdpa_kernel(SDPBackend.MATH)` when you need maximum precision or a reference for validation
- TF32 on Ampere+ GPUs silently reduces matmul precision; disable with `torch.backends.cuda.matmul.allow_tf32 = False`
- For reproducibility, combine `cudnn.deterministic = True`, specific backend selection, and disabled reduced-precision reductions
- Backend auto-selection means your model's numerical output can change when moving to different hardware

## References
- [SDPA Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [SDPA Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [PyTorch Numerical Accuracy Notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch Issue #119188 - Backend output differences](https://github.com/pytorch/pytorch/issues/119188)
- [PyTorch Issue #119131 - NaNs with mixed precision](https://github.com/pytorch/pytorch/issues/119131)
