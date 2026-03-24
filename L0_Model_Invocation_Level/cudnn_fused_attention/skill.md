---
skill_name: cuDNN Fused Attention Backend
description: NVIDIA cuDNN's fused multi-head attention implementation, accessible via the cudnn-frontend graph API and as a PyTorch SDPA backend, providing vendor-optimized attention kernels for Ampere, Hopper, and Blackwell GPUs.
level: L0 - Implementation Selection Level
target_hardware: NVIDIA GPUs (Ampere A100, Hopper H100/H200, Blackwell B100/B200); requires cuDNN 8.9.5+ (recommended 9.0+)
relevance: When deciding whether to use NVIDIA's vendor-optimized cuDNN attention backend vs FlashAttention or other community kernels for training and inference workloads.
---

# cuDNN Fused Attention Backend

## What It Is
cuDNN Fused Attention is NVIDIA's vendor-optimized implementation of scaled dot-product attention (SDPA), available through two interfaces: (1) the `cudnn-frontend` graph API for direct programmatic access with maximum control, and (2) as the `SDPBackend.CUDNN_ATTENTION` backend in PyTorch's `torch.nn.functional.scaled_dot_product_attention`. The cuDNN backend fuses the entire QKV attention computation (matmul-softmax-matmul) into a single GPU kernel, eliminating intermediate memory traffic. As a vendor library, cuDNN attention is tuned per-GPU architecture and per-problem-size, often providing strong out-of-the-box performance without manual kernel tuning.

## Key Concepts
- **Fused attention graph**: cuDNN represents the attention computation as a dataflow graph (Q, K, V inputs -> scaled BMM -> softmax -> BMM -> output) and compiles it into a single fused kernel, avoiding materialization of the N x N attention matrix in HBM
- **cudnn-frontend graph API**: The C++/Python `cudnn-frontend` library provides a high-level graph builder (`cudnn.pygraph()`) where you define tensors, operations (matmul, softmax, pointwise), and let cuDNN auto-tune the execution plan
- **PyTorch SDPBackend.CUDNN_ATTENTION**: PyTorch 2.1+ includes cuDNN as a fourth SDPA backend; it is attempted when the input configuration matches cuDNN's supported parameter space (dtype, head dimension, sequence length)
- **Supported configurations**: FP16 and BF16 data types; head dimensions up to 256 (commonly 64, 128, 256); causal and non-causal masking; padding masks; variable sequence lengths; dropout support; GQA/MQA via broadcast dimensions
- **cuDNN 9.x enhancements**: cuDNN 9.0+ introduced the graph API v2, FP8 attention support (E4M3/E5M2), improved heuristics for kernel selection, and Hopper-specific optimizations leveraging TMA and warp-group MMA (WGMMA)
- **Auto-tuning and heuristics**: cuDNN maintains internal heuristic tables mapping problem sizes to optimal kernel configurations; `cudnn.heuristics_mode` controls whether to use the fast heuristic (mode A) or exhaustive search (mode B)
- **Backward pass support**: Full backward pass (dQ, dK, dV computation) is supported, making cuDNN attention viable for training as well as inference
- **Deterministic mode**: cuDNN offers a deterministic execution option (`torch.backends.cudnn.deterministic = True` in PyTorch), though it may sacrifice some performance

## When to Use
- You want a stable, vendor-supported attention implementation with guaranteed correctness and ongoing optimization for each new GPU architecture
- You are on Hopper/Blackwell GPUs where cuDNN receives day-one optimizations from NVIDIA engineers
- You need FP8 attention support (cuDNN 9.x provides production-quality FP8 SDPA)
- You want a drop-in backend via PyTorch SDPA without managing third-party library versions
- Your model uses standard attention patterns (causal, non-causal, padding masks) without exotic score modifications
- You need deterministic execution for reproducibility and validation workflows
- You are building a custom attention variant using the cudnn-frontend graph API and want NVIDIA's optimized kernels as building blocks

## When NOT to Use
- You need custom score modifications (ALiBi, RoPE fusion, soft-capping, relative position biases) that cuDNN's fused graph does not support -- use FlexAttention or flash_attn with custom masks
- You need the absolute fastest attention on Hopper and are willing to use open-source kernels -- FlashAttention-3 matches or exceeds cuDNN for medium-to-long sequences (1K+), and FlashAttention-4 achieves ~20% speedup over cuDNN on Blackwell
- You need paged KV cache for inference serving -- cuDNN does not natively support page-table-based KV cache layouts; use FlashInfer, vLLM, or TensorRT-LLM
- You need sliding window / local attention patterns not expressible via cuDNN's mask options
- You are on AMD GPUs -- cuDNN is NVIDIA-only; use Composable Kernel or Triton-based attention
- You need INT4/INT8 KV cache quantization -- cuDNN attention operates on FP16/BF16/FP8 only
- Your head dimension exceeds cuDNN's supported range (>256) or you use non-power-of-2 head dimensions

## Code Snippets / Pseudo-code

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Use cuDNN backend via PyTorch SDPA
query = torch.randn(B, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
key = torch.randn(B, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
value = torch.randn(B, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

# Force cuDNN backend
with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    output = F.scaled_dot_product_attention(query, key, value, is_causal=True)

# Using cudnn-frontend graph API directly (Python)
import cudnn

# Build the attention graph
graph = cudnn.pygraph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

q = graph.tensor(name="Q", dim=[B, H, S_q, D], stride=[H*S_q*D, S_q*D, D, 1])
k = graph.tensor(name="K", dim=[B, H, S_kv, D], stride=[H*S_kv*D, S_kv*D, D, 1])
v = graph.tensor(name="V", dim=[B, H, S_kv, D], stride=[H*S_kv*D, S_kv*D, D, 1])

# Define SDPA operation
o, stats = graph.sdpa(
    name="sdpa",
    q=q, k=k, v=v,
    is_inference=False,
    attn_scale=1.0 / (D ** 0.5),
    use_causal_mask=True,
)
o.set_output(True).set_dim([B, H, S_q, D])

graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
graph.check_support()
graph.build_plans()

# Execute
graph.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu, stats: stats_gpu}, workspace)
```

## Source Code Examples

### FP8 Attention (cuDNN 9.x, Hopper+)

```python
# FP8 E4M3 attention for 2x compute throughput
graph = cudnn.pygraph(
    io_data_type=cudnn.data_type.FP8_E4M3,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

q_fp8 = graph.tensor(
    name="Q",
    dim=[B, H, S_q, D],
    stride=[H * S_q * D, S_q * D, D, 1],
    data_type=cudnn.data_type.FP8_E4M3,
)
k_fp8 = graph.tensor(
    name="K",
    dim=[B, H, S_kv, D],
    stride=[H * S_kv * D, S_kv * D, D, 1],
    data_type=cudnn.data_type.FP8_E4M3,
)
v_fp8 = graph.tensor(
    name="V",
    dim=[B, H, S_kv, D],
    stride=[H * S_kv * D, S_kv * D, D, 1],
    data_type=cudnn.data_type.FP8_E4M3,
)

# Scaling tensors for FP8 (per-tensor or per-head scaling)
descale_q = graph.tensor(name="descale_Q", dim=[1,1,1,1], stride=[1,1,1,1],
                          data_type=cudnn.data_type.FLOAT)
descale_k = graph.tensor(name="descale_K", dim=[1,1,1,1], stride=[1,1,1,1],
                          data_type=cudnn.data_type.FLOAT)
descale_v = graph.tensor(name="descale_V", dim=[1,1,1,1], stride=[1,1,1,1],
                          data_type=cudnn.data_type.FLOAT)
descale_s = graph.tensor(name="descale_S", dim=[1,1,1,1], stride=[1,1,1,1],
                          data_type=cudnn.data_type.FLOAT)
scale_s = graph.tensor(name="scale_S", dim=[1,1,1,1], stride=[1,1,1,1],
                        data_type=cudnn.data_type.FLOAT)
scale_o = graph.tensor(name="scale_O", dim=[1,1,1,1], stride=[1,1,1,1],
                        data_type=cudnn.data_type.FLOAT)

o, stats, amax_s, amax_o = graph.sdpa_fp8(
    name="sdpa_fp8",
    q=q_fp8, k=k_fp8, v=v_fp8,
    descale_q=descale_q, descale_k=descale_k,
    descale_v=descale_v, descale_s=descale_s,
    scale_s=scale_s, scale_o=scale_o,
    is_inference=True,
    attn_scale=1.0 / (D ** 0.5),
    use_causal_mask=True,
)
```

### GQA / MQA Support

```python
# GQA: 32 query heads, 8 KV heads (4:1 ratio)
H_q, H_kv = 32, 8

q = graph.tensor(name="Q", dim=[B, H_q, S_q, D],
                  stride=[H_q * S_q * D, S_q * D, D, 1])
# K and V use H_kv heads, cuDNN broadcasts automatically
k = graph.tensor(name="K", dim=[B, H_kv, S_kv, D],
                  stride=[H_kv * S_kv * D, S_kv * D, D, 1])
v = graph.tensor(name="V", dim=[B, H_kv, S_kv, D],
                  stride=[H_kv * S_kv * D, S_kv * D, D, 1])

# cuDNN handles the Q-head to KV-head mapping internally
o, stats = graph.sdpa(
    name="sdpa_gqa",
    q=q, k=k, v=v,
    is_inference=True,
    attn_scale=1.0 / (D ** 0.5),
)
```

## Key Takeaways
- cuDNN Fused Attention is NVIDIA's first-party optimized SDPA, providing strong baseline performance and production stability across Ampere, Hopper, and Blackwell GPUs
- It is accessible as a PyTorch SDPA backend (`SDPBackend.CUDNN_ATTENTION`) for zero-effort integration, or via the `cudnn-frontend` graph API for advanced customization
- cuDNN 9.x adds FP8 support and Hopper-optimized kernels, making it competitive for both training and inference
- For standard attention patterns (causal, non-causal, padding masks), cuDNN provides reliable performance without the need to track open-source kernel releases
- FlashAttention-3 matches or exceeds cuDNN on Hopper for medium-to-long sequences, and FlashAttention-4 surpasses it on Blackwell by ~20% -- cuDNN is the baseline to beat, not always the fastest option
- The cudnn-frontend graph API enables custom fused attention variants (e.g., attention + bias + dropout) compiled into a single kernel, which is powerful for research experimentation
- Always benchmark cuDNN against FlashAttention for your specific workload: the winner depends on sequence length, head dimension, batch size, and GPU architecture

## References
- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/)
- [cudnn-frontend GitHub Repository](https://github.com/NVIDIA/cudnn-frontend)
- [cudnn-frontend SDPA Samples](https://github.com/NVIDIA/cudnn-frontend/tree/main/samples/python)
- [PyTorch SDPA Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [cuDNN 9.0 Release Notes](https://docs.nvidia.com/deeplearning/cudnn/release-notes/)
- [FlashAttention-3 Paper](https://arxiv.org/abs/2407.08691) (benchmark comparisons vs cuDNN)
