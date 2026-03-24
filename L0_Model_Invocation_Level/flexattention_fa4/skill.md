---
skill_name: FlexAttention with FlashAttention-4 Backend
description: PyTorch API for custom attention variants that compiles to high-performance FlashAttention-4 kernels on Hopper/Blackwell GPUs.
level: L0 - Model/Invocation Level
target_hardware: NVIDIA Hopper (H100, H200) and Blackwell (B100, GB200) GPUs
relevance: When implementing custom attention patterns (ALiBi, sliding window, soft-capping, document masking) and needing near-handwritten-kernel performance without CUDA expertise.
---

# FlexAttention with FlashAttention-4 Backend

## What It Is
FlexAttention is a PyTorch-native API that lets researchers implement custom attention score modifications and block-sparse masks using simple Python functions (`score_mod`, `mask_mod`). When compiled with `kernel_options={"BACKEND": "FLASH"}`, it generates CuTeDSL code that calls the FlashAttention-4 kernel, achieving 1.2x-3.2x speedup over the Triton backend on Hopper and Blackwell GPUs. It bridges the gap between researcher-friendly flexibility and production-grade performance.

## Key Concepts
- **`score_mod` function**: A Python function `(score, b_idx, h_idx, q_idx, kv_idx) -> modified_score` that is traced and fused into the attention kernel for pre-softmax score transformations
- **`mask_mod` / `block_mask`**: Defines block-sparse iteration patterns with runtime data-dependent sparsity encoding, avoiding computation on masked-out blocks
- **Two backends**: `"TRITON"` (default, broader hardware support) and `"FLASH"` (FA4, Hopper/Blackwell only, significantly faster)
- **CuTeDSL code generation**: Python score_mod is automatically compiled to CuTeDSL, then JIT-instantiated as a specialized FlashAttention-4 kernel
- **Composable patterns**: Multiple attention modifications (causal + sliding window + ALiBi) combine naturally in a single score_mod function

## When to Use
- You need custom attention patterns (ALiBi, sliding window, soft-capping, document masking, relative position biases) with high performance
- You have Hopper or Blackwell GPUs and want to close the performance gap between flexible Python code and handwritten CUDA kernels
- You are prototyping novel attention variants and want to iterate quickly without writing CUDA
- You need block-sparse attention with data-dependent sparsity patterns
- Standard SDPA does not support your attention modification (e.g., score biases, custom masks beyond simple causal)

## When NOT to Use
- You only need standard causal or full attention -- use `torch.nn.functional.scaled_dot_product_attention` (SDPA) for simpler dispatch
- You are on pre-Hopper hardware (Ampere, Ada) -- the FA4 backend is not supported; use the Triton backend or SDPA instead
- You need stable production APIs -- FlexAttention + FA4 is under active development with potential breaking changes
- You need FP8 precision -- check current FA4 support status (evolving)
- Your workload is memory-bound rather than compute-bound -- FA4 speedups are most pronounced on compute-bound workloads

## Code Snippets / Pseudo-code
```python
import torch
from functools import partial
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# --- Option 1: Score modification (e.g., ALiBi + causal) ---
def alibi_causal(score, b_idx, h_idx, q_idx, kv_idx):
    causal = q_idx >= kv_idx
    bias = -torch.abs(q_idx - kv_idx) * slopes[h_idx]
    return torch.where(causal, score + bias, float('-inf'))

# Compile with FA4 backend
flex_fa4 = torch.compile(
    partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
    dynamic=False
)

out = flex_fa4(q, k, v, score_mod=alibi_causal)

# --- Option 2: Block-sparse mask (e.g., sliding window) ---
def sliding_window_mask(b_idx, h_idx, q_idx, kv_idx):
    return torch.abs(q_idx - kv_idx) <= 512

block_mask = create_block_mask(sliding_window_mask, B, H, S, S)
out = flex_fa4(q, k, v, block_mask=block_mask)
```

## Source Code Examples

### Sliding Window + Causal Combination

```python
def sliding_window_with_causal(score, b_idx, h_idx, q_idx, kv_idx):
    """Sliding window + causal masking"""
    causal_mask = q_idx >= kv_idx
    window_mask = torch.abs(q_idx - kv_idx) <= 128
    return torch.where(causal_mask & window_mask, score, float('-inf'))

out = flex_flash(q, k, v, score_mod=sliding_window_with_causal)
```

## Key Takeaways
- FlexAttention + FA4 is the preferred path for custom attention on Hopper/Blackwell GPUs, offering 1.2-3.2x speedup over the Triton backend
- The `score_mod` API is simple: write a Python function with 5 arguments, the compiler handles the rest
- On Blackwell, the speedup advantage is even larger because deep pipelining and warp specialization cannot be expressed in Triton
- For standard attention patterns without modifications, SDPA remains the simpler and more portable choice
- FA4 backend is JIT-compiled and specialized per attention variant -- first call has compilation overhead
- This is the solution to the "researcher prototypes in Python, engineer rewrites in CUDA" problem

## References
- [FlexAttention + FlashAttention-4 Blog Post](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- [FlashAttention Repository (Dao-AILab)](https://github.com/Dao-AILab/flash-attention)
- [FlexAttention API Documentation](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)
