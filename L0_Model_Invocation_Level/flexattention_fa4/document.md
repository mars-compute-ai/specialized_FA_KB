# FlexAttention + FlashAttention-4: Fast and Flexible

Source: https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/

## Overview

FlexAttention is a PyTorch API that enables custom attention variants through a simple Python interface. With the FlashAttention-4 (FA4) backend, it now delivers 1.2x to 3.2x speedup over the existing Triton implementation on compute-bound workloads, specifically on Hopper and Blackwell GPUs. This eliminates the need for CUDA expertise to achieve high-performance custom attention.

## Core Concept

FlexAttention provides two core extensions to standard attention:

1. **Pointwise score modifications (`score_mod`)**: Arbitrary pre-softmax score transformations with global memory loads
2. **Block-sparse iteration (`block_mask`/`mask_mod`)**: Both forward and backward passes with runtime data-dependent sparsity encoding

## Basic API Usage

```python
import torch
from functools import partial
from torch.nn.attention.flex_attention import flex_attention

# Compile with FA4 backend
flex_flash = torch.compile(
    partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
    dynamic=False
)

def local_boost(score, b_idx, h_idx, q_idx, kv_idx):
    """Example score_mod: boost local attention within +/-8 positions"""
    return torch.where(torch.abs(q_idx - kv_idx) <= 8, score * 2, score)

# Setup
B, H, S, D = 2, 8, 2048, 128
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

# Run attention with custom score modification
out = flex_flash(q, k, v, score_mod=local_boost)
```

## score_mod Function Signature

The `score_mod` function receives five arguments:
- `score`: The raw attention score (scalar)
- `b_idx`: Batch index
- `h_idx`: Head index
- `q_idx`: Query position index
- `kv_idx`: Key/Value position index

It returns the modified score. This function is traced by the compiler and fused into the attention kernel.

## Supported Attention Variants

FlexAttention transparently handles through `score_mod` and `mask_mod`:
- **ALiBi** (Attention with Linear Biases)
- **Sliding window attention**
- **Document masking**
- **Soft-capping**
- **Causal masking**
- **Combinations of the above**

All work through the same unified interface without code changes.

### Example: Sliding Window + Causal

```python
def sliding_window_with_causal(score, b_idx, h_idx, q_idx, kv_idx):
    """Sliding window + causal masking"""
    causal_mask = q_idx >= kv_idx
    window_mask = torch.abs(q_idx - kv_idx) <= 128
    return torch.where(causal_mask & window_mask, score, float('-inf'))

out = flex_flash(q, k, v, score_mod=sliding_window_with_causal)
```

## Backend Selection: FLASH vs TRITON

Set `kernel_options={"BACKEND": "FLASH"}` to use the FlashAttention-4 backend instead of the default Triton implementation:

```python
from functools import partial
from torch.nn.attention.flex_attention import flex_attention

# Default: Triton backend
flex_triton = torch.compile(flex_attention, dynamic=False)

# FA4 backend (Hopper/Blackwell required)
flex_flash = torch.compile(
    partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
    dynamic=False
)
```

## Performance Comparison

### Hopper GPU
- FlexAttention (Triton): ~60-80% of FlashAttention-3 throughput
- FlexAttention (FA4): Significant improvement, closing the gap

### Blackwell GPU (GB200, 1000W)
- FlexAttention (FA4): 1.2x to 3.2x speedup over Triton implementation
- The gap is larger on Blackwell because:
  - Deep pipelining architecture requirements
  - Warp specialization techniques not expressible in Triton
  - Larger tensor cores demand sustained throughput

## Technical Implementation

### CuTeDSL Code Generation
PyTorch automatically generates CuTeDSL score/mask modification functions from Python `score_mod`/`mask_mod` code, eliminating manual low-level CUDA rewrites.

### JIT Instantiation
FlashAttention-4 kernels are JIT-instantiated for specific attention variants:
- Runtime customization without full recompilation
- Automatic kernel specialization
- Seamless integration with PyTorch's compilation pipeline

### Block-Sparse Iteration
- Forward pass: Block-sparse token iteration
- Backward pass: Corresponding gradient computations
- Runtime encoding: Data-dependent sparsity patterns encoded at execution time
- Efficient representation via simple data structures

## The Researcher-to-Engineer Pipeline Problem (Solved)

Previously:
1. Researchers prototype with FlexAttention (flexible, easy)
2. At scale, hit performance ceiling (60-80% of optimized kernels)
3. Must hire ML engineers to port to CUDA
4. Each new variant requires manual engineering effort

Now:
- Researchers can prototype AND deploy at scale
- Performance stays within competitive range of handwritten kernels
- No CUDA expertise required

## Hardware Requirements

- **Hopper**: H100, H200 GPUs
- **Blackwell**: B100, GB100, GB200 GPUs
- Requires recent PyTorch nightly build
- Requires recent flash-attention checkout (dao-AILab/flash-attention main branch)

**Note**: This is actively developed code; expect breaking changes during stabilization.

## Community Adoption

- Cited by dozens of research papers
- Over 1,000 GitHub repositories use FlexAttention
- Democratizes attention research and experimentation

## FlashAttention-4 (CuTeDSL) Installation

```bash
pip install flash-attn-4
```

```python
from flash_attn.cute import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)
```
