---
skill_name: PyTorch Scaled Dot Product Attention (SDPA)
description: PyTorch's unified API that automatically dispatches attention to the optimal backend (FlashAttention, memory-efficient, math, or CuDNN).
level: L0 - Model/Invocation Level
target_hardware: NVIDIA GPUs (Ampere, Ada, Hopper, Blackwell); CPU fallback available
relevance: When integrating attention into a PyTorch model and wanting automatic backend selection with the option to force specific implementations.
---

# PyTorch Scaled Dot Product Attention (SDPA)

## What It Is
`torch.nn.functional.scaled_dot_product_attention` is PyTorch's built-in, fused attention function that automatically selects among FlashAttention, memory-efficient attention (xFormers), CuDNN attention, and a math fallback based on input properties and hardware. It is the recommended default entry point for attention in PyTorch 2.1.1+ and serves as the backbone of HuggingFace Transformers' SDPA mode.

## Key Concepts
- **Automatic backend dispatch**: SDPA evaluates input dtype, shape, hardware, and parameters to select FlashAttention, memory-efficient attention, CuDNN, or the math kernel
- **`sdpa_kernel` context manager**: Allows forcing or restricting backends via `SDPBackend.FLASH_ATTENTION`, `SDPBackend.EFFICIENT_ATTENTION`, `SDPBackend.MATH`, `SDPBackend.CUDNN_ATTENTION`
- **Global toggles**: `torch.backends.cuda.enable_flash_sdp()`, `enable_mem_efficient_sdp()`, `enable_math_sdp()` for persistent backend control
- **`is_causal` parameter**: Enables efficient causal masking without materializing the N x N mask matrix
- **NestedTensor support**: Handles variable-length sequences without padding overhead
- **torch.compile compatible**: Fully composable with PyTorch 2.0+ compilation

## When to Use
- You are building or modifying a Transformer model in PyTorch and want the fastest available attention without vendor lock-in
- You want portability across hardware (NVIDIA, CPU) with automatic fallback
- You need a simple API that handles causal masking, dropout, and scaling out of the box
- You are using HuggingFace Transformers and want to set `attn_implementation="sdpa"`
- You want to benchmark different attention backends without changing model code

## When NOT to Use
- You need custom score modifications (ALiBi, soft-capping, relative position biases) that SDPA does not natively support -- use FlexAttention or the flash_attn library directly
- You require `output_attentions=True` for interpretability (forces slow math fallback)
- You need FP8 attention (use FlashAttention-3/4 directly)
- You need paged KV cache for inference serving (use flash_attn's KV cache API or vLLM)
- You need sliding window attention with custom window sizes (use flash_attn directly or FlexAttention)
- Your sequence lengths are very short (<256 tokens) and the kernel launch overhead matters

## Code Snippets / Pseudo-code
```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Basic usage - automatic backend selection
query = torch.randn(B, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
key = torch.randn(B, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
value = torch.randn(B, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

output = F.scaled_dot_product_attention(query, key, value, is_causal=True)

# Force a specific backend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(query, key, value, is_causal=True)

# HuggingFace integration
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

## Key Takeaways
- SDPA is the default and recommended starting point for attention in PyTorch 2.1.1+
- It provides 10-40x speedup over the math backend by automatically selecting FlashAttention or memory-efficient kernels
- Use `sdpa_kernel()` to force a specific backend for benchmarking or when you know which is optimal
- FlashAttention backend requires fp16/bf16; memory-efficient supports fp32 as well
- For custom attention patterns, score modifications, or advanced features (ALiBi, sliding window, paged KV cache), you must use FlexAttention or the standalone flash_attn library
- Always profile with `sdpa_kernel()` to verify which backend is actually being dispatched for your specific input configuration

## References
- [PyTorch SDPA Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch SDPA Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [torch.nn.attention.sdpa_kernel Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
- [HuggingFace GPU Inference Optimization](https://huggingface.co/docs/transformers/main/perf_infer_gpu_one)
