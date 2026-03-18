# FlexAttention + FlashAttention-4: Fast and Flexible

Source: https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/

## Overview

FlexAttention is a PyTorch API enabling custom attention variants without CUDA expertise, now with a FlashAttention-4 (FA4) backend for Hopper and Blackwell GPUs delivering 1.2x to 3.2x performance gains over the existing Triton implementation on compute-bound workloads.

## Core Concepts

### FlexAttention Fundamentals

FlexAttention extends vanilla FlashAttention with two key capabilities:

1. **Pointwise modifications to pre-softmax scores** with arbitrary global memory loads
2. **Block-sparse iteration** for both forward and backward passes with runtime data-dependent sparsity encoding

### Score Modification Functions (score_mod)

The `score_mod` function is the core interface for customizing attention behavior:

```python
def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
    """
    Args:
        score: The attention score before softmax
        b_idx: Batch index
        h_idx: Head index
        q_idx: Query position
        kv_idx: Key/Value position

    Returns:
        Modified score
    """
    return score  # Apply custom modifications
```

## API Usage

### Basic Usage with FlashAttention-4 Backend

```python
import torch
from functools import partial
from torch.nn.attention.flex_attention import flex_attention

# Compile with FA4 backend
flex_flash = torch.compile(
    partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
    dynamic=False
)

# Define custom attention variant
def local_boost(score, b_idx, h_idx, q_idx, kv_idx):
    """Boost attention scores within local window"""
    return torch.where(
        torch.abs(q_idx - kv_idx) <= 8,
        score * 2,
        score
    )

# Execute
B, H, S, D = 2, 8, 2048, 128
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

out = flex_flash(q, k, v, score_mod=local_boost)
```

## Supported Attention Variants

FlexAttention covers a comprehensive range of patterns through the `score_mod` interface:

- **ALiBi (Attention with Linear Biases)**
- **Sliding Window Attention**
- **Document Masking**
- **Soft-capping**
- **Combinations** of the above

## Block-Sparse Iteration

### Query-Key/Value Grouping Strategy

FlexAttention organizes computation into blocks:
- Queries grouped into **blocks of Br** tokens
- Keys/Values grouped into **blocks of Bc** tokens
- Each block pair (query_block, kv_block) processed in a fused kernel

### Masked Region Skipping

For data-dependent or static sparsity patterns:
- **Block masks** encode which (query_block, kv_block) pairs to compute
- The runtime skips entire blocks of masked regions
- Both forward and backward passes respect the block structure
- Sparsity can be data-dependent and change at runtime

### CuTe DSL Integration

FlexAttention leverages NVIDIA's CuTe (Collective Utilities for Tensor Expressions) DSL:
- **Automatic score/mask modification function generation** from Python definitions
- **JIT instantiation** of FlashAttention-4 for custom variants
- Hardware-aware scheduling and memory layout optimization

## Performance Metrics

### Hopper GPUs (H100)
- **Previous state**: ~80% of FlashAttention-3 performance
- **Current state**: ~60% of FlashAttention-3 (due to FA3 improvements)
- **With FA4 backend**: 1.2x to 3.2x speedup over Triton implementation
- Acceptable gap for flexibility-focused workloads

### Blackwell GPUs (GB200)
- **Previous Triton FlexAttention**: Severe performance gap vs. cuDNN
- **With FA4 backend**: Near-parity with optimized implementations
- Running at 1000W, significantly reduced gap

## Technical Improvements on Blackwell

### Warp Specialization
FlashAttention-4 employs deeply pipelined, warp-specialized kernels:
- Keeps tensor cores continuously busy
- Leverages new Blackwell hardware capabilities
- Not expressible in Triton-based implementations

### Softmax Computation Updates
New algorithms for computing softmax efficiently on Blackwell architecture with increased tensor core capabilities.

## Example Attention Variants

### Sliding Window
```python
def sliding_window(score, b_idx, h_idx, q_idx, kv_idx):
    return torch.where(
        torch.abs(q_idx - kv_idx) <= window_size,
        score,
        float('-inf')
    )
```

### Document Masking
```python
def document_mask(score, b_idx, h_idx, q_idx, kv_idx):
    q_doc = q_idx // doc_length
    kv_doc = kv_idx // doc_length
    return torch.where(
        q_doc == kv_doc,
        score,
        float('-inf')
    )
```

### Soft-capping
```python
def soft_cap(score, b_idx, h_idx, q_idx, kv_idx):
    cap_value = 30.0
    return (cap_value / torch.tanh(cap_value)) * torch.tanh(score)
```

## Compilation Strategy

```python
# Always compile for optimal performance
flex_attention_compiled = torch.compile(
    partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
    dynamic=False  # Static shapes recommended
)
```

## Requirements

- PyTorch nightly build (recent version)
- FlashAttention repository (main branch)
- CUDA 12.1+
- Hopper or Blackwell GPU

## Adoption

- Dozens of papers cite FlexAttention
- 1,000+ repositories have adopted the framework
- Democratized attention research by eliminating need for CUDA expertise

## Backward Compatibility

FlexAttention maintains full backward compatibility:
- Existing code works with both Triton and FA4 backends
- Default behavior unchanged
- Opt-in to FA4 via `BACKEND` option
