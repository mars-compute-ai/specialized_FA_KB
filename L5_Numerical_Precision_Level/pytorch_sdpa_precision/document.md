# PyTorch Scaled Dot-Product Attention (SDPA) - Precision and Reproducibility

**Source:** [PyTorch Documentation - torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
**Tutorial:** [SDPA Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
**Numerical Accuracy Notes:** [PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

## Function Signature

```python
torch.nn.functional.scaled_dot_product_attention(
    query,           # (N, ..., L, E) tensor
    key,             # (N, ..., S, E) tensor
    value,           # (N, ..., S, Ev) tensor
    attn_mask=None,  # Optional mask tensor
    dropout_p=0.0,   # Dropout probability
    is_causal=False, # Whether to apply causal mask
    scale=None       # Scaling factor (default: 1/sqrt(E))
)
```

## Available Backends

PyTorch SDPA dispatches to multiple backend implementations, each with different numerical characteristics:

### 1. FlashAttention Backend (`SDPBackend.FLASH_ATTENTION`)
- Based on "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- Uses tiled computation to reduce HBM access
- Operates in reduced precision (FP16/BF16) internally
- **Fastest** for long sequences on supported hardware
- Benchmark: ~2.3ms (vs ~87ms for math backend in test cases)

### 2. Memory-Efficient Attention Backend (`SDPBackend.EFFICIENT_ATTENTION`)
- Based on xformers implementation
- Intermediate speed: ~4.4ms in benchmarks
- Uses memory-efficient computation patterns

### 3. Math Backend (`SDPBackend.MATH`)
- PyTorch C++ reference implementation
- **Key precision property:** For FP16/BF16 inputs, all intermediates are kept in torch.float32
- Slowest but most numerically precise: ~87ms in benchmarks
- Serves as the reference for numerical correctness

### 4. CuDNN Backend (`SDPBackend.CUDNN_ATTENTION`)
- NVIDIA cuDNN-backed implementation
- May select nondeterministic algorithms for performance

## Numerical Precision Behavior

### Float32 Upcast for Math Backend
The math backend implements a critical precision safeguard:

> By default, FP16/BF16 inputs are upcast to FP32 for computation, then downcast to the original dtype. This improves accuracy but increases memory usage.

This behavior can be controlled:
```python
# Re-enable reduced-precision reductions for speed (at cost of accuracy)
torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
```

### Backend-Specific Numerical Differences
Different backends produce **different numerical results** for the same inputs:
- The math backend (with float32 intermediates) is the most precise
- FlashAttention and memory-efficient backends operate in reduced precision
- Switching backends can cause test failures if exact numerical matching is expected
- A naive SDPA math backend using FP16/BF16 inputs accumulates significant numerical errors due to low-precision intermediate buffers, which is why the default upcast behavior exists

### Specific Numerical Observations
- With float32 inputs from a Uniform distribution: full match with reference implementation
- With float32 inputs from a Normal distribution: small differences appear
- With GPU tensors in lower precision: significantly larger output differences
- Batched operations may differ from equivalent sequential computations

## Backend Selection

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

# Force specific backend
with sdpa_kernel(SDPBackend.MATH):
    # Most precise - float32 intermediates
    result = F.scaled_dot_product_attention(query, key, value)

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    # Fastest for long sequences
    result = F.scaled_dot_product_attention(query, key, value)

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    # Memory-efficient implementation
    result = F.scaled_dot_product_attention(query, key, value)
```

## Reproducibility Concerns

### Nondeterministic Behavior
> In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance.

To enforce determinism:
```python
torch.backends.cudnn.deterministic = True
```

### Cross-Backend Reproducibility
- Different backends do **not** produce bitwise identical results
- Even the same backend may produce different results across:
  - Different PyTorch versions
  - Different hardware platforms
  - Different CUDA/cuDNN versions
- The math implementation is the most reproducible across configurations

### IEEE 754 Considerations
PyTorch follows IEEE 754 but:
- Cannot guarantee bitwise identical results across platforms, releases, or commits
- Floating-point operations are not associative -- computation order affects outcomes
- `(A @ B)[0]` may not equal `A[0] @ B[0]` despite mathematical equivalence

## CUDA-Specific Precision Notes

### TensorFloat-32 (TF32) on Ampere+
- NVIDIA Ampere GPUs use only the first 10 mantissa bits (TF32), reducing accuracy
- Matrix multiplications have TF32 **disabled** by default
- Convolutions have TF32 **enabled** by default
- Controllable via `torch.backends.cuda.matmul.allow_tf32`

### Half-Precision Reductions
- FP16/BF16 GEMMs typically accumulate in FP32
- Newer architectures may truncate to reduced precision for performance
- Controllable via:
```python
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

## Data Type Support

| Backend | float16 | bfloat16 | float32 |
|---------|---------|----------|---------|
| Math | Yes (upcast to fp32) | Yes (upcast to fp32) | Yes |
| FlashAttention | Yes | Yes | No (must be half-precision) |
| Memory-Efficient | Yes | Yes | Yes |
| CuDNN | Yes | Yes | Varies |

## torch.compile Integration

SDPA fully composes with `torch.compile()`:
- NanoGPT benchmark: **46% speed improvement** (6090ms to 3273ms per training step)
- Compilation can optimize backend selection and kernel fusion

## Tensor Support

- **Dense tensors:** Standard padded sequences
- **NestedTensors:** Variable-length sequences without padding waste (PyTorch 2.0+)

## Key Precision Recommendations

1. **For maximum accuracy:** Use `SDPBackend.MATH` which keeps intermediates in float32
2. **For reproducibility:** Set `torch.backends.cudnn.deterministic = True` and use the math backend
3. **For production:** Be aware that switching backends changes numerical output; validate with tolerances, not exact matching
4. **For training:** The default auto-dispatch is usually acceptable; FlashAttention's reduced precision has minimal impact on training convergence
5. **For debugging:** Compare against the math backend as the reference implementation

## References

- [SDPA Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [SDPA Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [PyTorch Numerical Accuracy Notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch Issue #119188 - Different output between backends](https://github.com/pytorch/pytorch/issues/119188)
- [PyTorch Issue #119131 - NaNs with math backend in mixed precision](https://github.com/pytorch/pytorch/issues/119131)
