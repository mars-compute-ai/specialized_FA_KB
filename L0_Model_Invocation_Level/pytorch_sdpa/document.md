# PyTorch Scaled Dot Product Attention (SDPA)

Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
Additional: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html

## Overview

PyTorch's `torch.nn.functional.scaled_dot_product_attention` (SDPA) implements the attention mechanism from "Attention is All You Need." The function automatically dispatches to optimized backends, providing significant performance improvements over naive implementations. As of PyTorch 2.1.1+, SDPA is the default attention mechanism.

## Function Signature

```python
torch.nn.functional.scaled_dot_product_attention(
    query,          # (N, ..., L, E) tensor
    key,            # (N, ..., S, E) tensor
    value,          # (N, ..., S, Ev) tensor
    attn_mask=None, # Optional mask
    dropout_p=0.0,  # Dropout probability
    is_causal=False,# Apply causal mask
    scale=None      # Scaling factor (default: 1/sqrt(E))
)
```

## Available Backends

For CUDA inputs, the function selects from these implementations:

1. **FlashAttention** (`SDPBackend.FLASH_ATTENTION`): Fast and memory-efficient exact attention with IO-awareness. Requires fp16/bf16. Controlled via `torch.backends.cuda.enable_flash_sdp()`.

2. **Memory-Efficient Attention** (`SDPBackend.EFFICIENT_ATTENTION`): From the xFormers library (Facebook Research). Supports fp32 in addition to fp16/bf16. Controlled via `torch.backends.cuda.enable_mem_efficient_sdp()`.

3. **Math Backend** (`SDPBackend.MATH`): Pure PyTorch C++ implementation. Supports all dtypes. Controlled via `torch.backends.cuda.enable_math_sdp()`.

4. **CuDNN Attention** (`SDPBackend.CUDNN_ATTENTION`): cuDNN-based backend for supported configurations.

## Backend Selection Control

### sdpa_kernel Context Manager

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

# Force FlashAttention backend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(query, key, value)

# Force memory-efficient backend
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    output = F.scaled_dot_product_attention(query, key, value)

# Force math backend (useful for debugging)
with sdpa_kernel(SDPBackend.MATH):
    output = F.scaled_dot_product_attention(query, key, value)
```

### Global Backend Control

```python
# Enable/disable specific backends globally
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
```

## Basic Usage

```python
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

query = torch.randn(2, 3, 8, device=device)
key = torch.randn(2, 3, 8, device=device)
value = torch.randn(2, 3, 8, device=device)

output = F.scaled_dot_product_attention(query, key, value)
```

## Causal Self-Attention Implementation

```python
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dimension, bias=False, is_causal=False, dropout=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        self.is_causal = is_causal

    def forward(self, x):
        query_projected = self.c_attn(x)
        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None,
            dropout_p=dropout,
            is_causal=is_causal
        )

        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)
        y = self.resid_dropout(self.c_proj(y))
        return y
```

## Benchmarking Backends

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.utils.benchmark as benchmark

def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

with sdpa_kernel(SDPBackend.MATH):
    math_time = benchmark_torch_function_in_microseconds(
        F.scaled_dot_product_attention, query, key, value
    )
    print(f"Math backend: {math_time:.3f} microseconds")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        flash_time = benchmark_torch_function_in_microseconds(
            F.scaled_dot_product_attention, query, key, value
        )
        print(f"FlashAttention: {flash_time:.3f} microseconds")
    except RuntimeError:
        print("FlashAttention not supported on this hardware")

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    try:
        efficient_time = benchmark_torch_function_in_microseconds(
            F.scaled_dot_product_attention, query, key, value
        )
        print(f"Memory-efficient: {efficient_time:.3f} microseconds")
    except RuntimeError:
        print("Efficient attention not supported")
```

## Performance Characteristics

Typical benchmark results (float16, CUDA):
- Math backend: ~87.6ms
- FlashAttention: ~2.3ms
- Memory-efficient: ~4.4ms
- Default (auto-selected): ~2.3ms

## NestedTensor Support

SDPA handles variable-length sequences without padding via `torch.nested.nested_tensor`, avoiding wasted computation on padding tokens.

## torch.compile Integration

SDPA fully composes with PyTorch's compilation:

```python
compiled_model = torch.compile(model)
```

## Causal Attention Bias Subclasses (PyTorch 2.3+)

```python
from torch.nn.attention.bias import causal_upper_left, causal_lower_right

upper_left = causal_upper_left(seq_len_q, seq_len_kv)
lower_right = causal_lower_right(seq_len_q, seq_len_kv)

out = F.scaled_dot_product_attention(query, key, value, upper_left)
```

## HuggingFace Transformers Integration

```python
from transformers import AutoModelForCausalLM

# SDPA is the default for PyTorch >= 2.1.1
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device_map="auto",
    attn_implementation="sdpa"
)

# Force specific backend during generation
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    outputs = model.generate(**inputs)

# Dynamically change attention implementation
model.set_attention_implementation("sdpa")
```

## Backend Constraints

- **FlashAttention**: Requires fp16/bf16, NVIDIA Ampere+ GPUs, no support for `output_attentions=True`
- **Memory-Efficient**: Supports fp32, broader GPU compatibility
- **Math**: Supports all dtypes, all GPUs, but significantly slower
- **CuDNN**: Specific hardware/configuration requirements

## Key Notes

- SDPA automatically selects the fastest available backend
- Each fused kernel has specific input limitations (dtype, head dimension, etc.)
- If a specific backend is required, use `sdpa_kernel()` to force it
- The `is_causal` parameter enables efficient causal masking without materializing the mask
- `output_attentions=True` forces fallback to eager/math implementation
