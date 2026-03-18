---
skill_name: FlashAttention-3 for Hopper GPUs
description: Hopper-optimized attention achieving 75% of H100 peak throughput via warp specialization, asynchronous pipelining, and FP8 support.
level: L0 - Model/Invocation Level
target_hardware: NVIDIA Hopper GPUs (H100, H200)
relevance: When running on H100/H200 GPUs and needing maximum attention throughput, or when FP8 attention precision is acceptable for inference.
---

# FlashAttention-3 for Hopper GPUs

## What It Is
FlashAttention-3 is a Hopper-specific reimplementation of FlashAttention that achieves 1.5-2.0x speedup over FlashAttention-2 by exploiting H100 hardware features: asynchronous TMA data movement, warp-specialized scheduling (pingpong and pipelining), and FP8 tensor cores with incoherent processing. It reaches up to 740 TFLOPS in FP16 (75% of H100 peak) and ~1.2 PFLOPS in FP8, compared to FA2's ~350 TFLOPS (~35% of peak) on the same hardware.

## Key Concepts
- **Softmax is the bottleneck on H100**: Exponential operations run at 3.9 TFLOPS vs 989 TFLOPS for matmul -- a 256x gap. FA3 hides this latency by overlapping softmax with matmul
- **Pingpong scheduling**: Two warpgroups alternate between GEMM and softmax phases, keeping tensor cores and multi-function units simultaneously busy
- **Intra-warpgroup pipelining**: Fine-grained interleaving of softmax and GEMM within a single warpgroup for even higher utilization
- **TMA (Tensor Memory Accelerator)**: Hardware unit that handles async memory transfers, freeing warps for computation
- **FP8 with incoherent processing**: Hadamard transforms spread outlier values to reduce FP8 quantization error by 2.6x, enabling viable low-precision attention
- **WGMMA instructions**: Hopper's warpgroup matrix multiply-accumulate for higher matmul throughput

## When to Use
- You have H100 or H200 GPUs and want maximum attention throughput (2x over FA2)
- You need FP8 attention for inference throughput (~1.2 PFLOPS)
- You are running standard attention patterns (causal, full) on Hopper hardware
- FA2 via SDPA is leaving significant H100 performance on the table (~35% utilization)
- You are training large models where attention is a significant fraction of total compute

## When NOT to Use
- You are on Ampere (A100) or older GPUs -- FA3 is Hopper-only; use FA2 instead
- You need custom attention patterns (score modifications, complex masks) -- use FlexAttention + FA4 instead
- You need maximum stability and mature API -- FA3 is in beta; FA2 via SDPA is more battle-tested
- You are on Blackwell GPUs -- FA4 is the preferred path for Blackwell
- Your workload is not attention-bound (e.g., attention is a small fraction of total model time)

## Code Snippets / Pseudo-code
```python
# Installation (from flash-attention repo)
# cd flash-attention/hopper && python setup.py install

import flash_attn_interface
import torch

# Basic FA3 usage on H100
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

output = flash_attn_interface.flash_attn_func(q, k, v, causal=True)

# FP8 mode for maximum throughput (inference)
q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)
output = flash_attn_interface.flash_attn_func(q_fp8, k_fp8, v_fp8, causal=True)
```

## Key Takeaways
- FA3 more than doubles H100 utilization compared to FA2 (75% vs 35% of peak)
- The core innovation is hiding softmax latency by overlapping it with matmul using warp specialization -- this is Hopper-specific and cannot be done on Ampere
- FP8 mode provides ~1.2 PFLOPS with acceptable accuracy (2.6x better than naive FP8) -- valuable for inference
- FA3 is the right choice for production H100 workloads with standard attention patterns
- For custom attention patterns on Hopper, FlexAttention + FA4 is the better path
- Requires CUDA >= 12.3 (recommend 12.8+) and the Hopper-specific installation from the flash-attention repo

## References
- [FlashAttention-3 Blog Post (Tri Dao)](https://tridao.me/blog/2024/flash3/)
- [FlashAttention-3 Paper (NeurIPS 2024)](https://arxiv.org/abs/2407.08691)
- [FlashAttention GitHub Repository - Hopper Directory](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)
- [NVIDIA H100 Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
