# cuDNN Fused Attention Backend

Source: https://github.com/NVIDIA/cudnn-frontend
Additional: https://docs.nvidia.com/deeplearning/cudnn/latest/
Additional: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

## Overview

NVIDIA cuDNN (CUDA Deep Neural Network library) provides a vendor-optimized implementation of fused multi-head attention (FMHA) that serves as both a standalone acceleration library and as a backend for PyTorch's `scaled_dot_product_attention`. The cuDNN attention implementation fuses the entire scaled dot-product attention computation -- two batched matrix multiplications separated by a softmax -- into a single GPU kernel, eliminating the need to materialize the O(N^2) attention matrix in HBM. This document covers the cuDNN attention architecture, the cudnn-frontend graph API for direct usage, PyTorch SDPA integration, supported configurations, performance characteristics, and decision criteria for when to use cuDNN vs FlashAttention.

## Architecture of cuDNN Fused Attention

### Fusion Strategy

The standard attention computation involves three separate kernel launches:

```
S = Q @ K^T / sqrt(d)         # Batched GEMM 1
P = softmax(S)                 # Softmax kernel
O = P @ V                     # Batched GEMM 2
```

Without fusion, the intermediate tensor S (shape B x H x N x N) must be written to HBM after GEMM1 and read back for softmax, and P must be written again and read back for GEMM2. For a sequence length N=4096 with FP16, this intermediate storage is 4096 x 4096 x 2 bytes = 32MB per head per batch -- a massive memory bandwidth bottleneck.

cuDNN's fused attention compiles these three operations into a single kernel where:
1. GEMM1 output tiles remain in shared memory / registers
2. Softmax is computed on-chip using the online softmax algorithm (Milakov & Gimelshein, 2018)
3. GEMM2 consumes the softmax output directly from on-chip storage
4. Only the final output O is written to HBM

This fusion provides:
- **O(N) memory complexity** instead of O(N^2) for intermediates
- **~2-4x bandwidth savings** compared to unfused implementations
- **Single kernel launch** reducing host-side overhead

### Kernel Variants by GPU Architecture

cuDNN maintains separate optimized kernel implementations for each GPU architecture:

| GPU Architecture | SM Version | Key Features |
|---|---|---|
| Ampere (A100) | SM80 | HMMA (FP16 Tensor Cores), basic tiling |
| Hopper (H100/H200) | SM90 | WGMMA, TMA, warp-group specialization, FP8 |
| Blackwell (B100/B200) | SM100 | 5th-gen Tensor Cores, enhanced FP8, larger shared memory |

On Hopper, cuDNN leverages:
- **TMA (Tensor Memory Accelerator)**: Hardware-assisted async global->shared memory copies that free up warps for computation
- **WGMMA (Warp-Group Matrix Multiply Accumulate)**: 128-thread warp-group operations that directly consume shared memory operands
- **Warp specialization**: Separate producer warps (data loading) and consumer warps (computation) for fully overlapped execution

## cudnn-frontend Graph API

### Overview

The `cudnn-frontend` library (available in C++ and Python) provides a high-level graph API for defining and executing cuDNN operations. For attention, you construct a dataflow graph specifying the input tensors, operations, and output tensors, then let cuDNN compile and optimize the execution plan.

### Installation

```bash
pip install nvidia-cudnn-frontend
# Or build from source:
git clone https://github.com/NVIDIA/cudnn-frontend.git
cd cudnn-frontend
pip install .
```

### Python API: Basic Forward Pass

```python
import cudnn
import torch

# Problem dimensions
B, H, S_q, S_kv, D = 4, 32, 2048, 2048, 128

# Create the cuDNN graph
graph = cudnn.pygraph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

# Define input tensors with strides
# cuDNN uses BHSD layout (batch, head, seq, dim)
q = graph.tensor(
    name="Q",
    dim=[B, H, S_q, D],
    stride=[H * S_q * D, S_q * D, D, 1],
    data_type=cudnn.data_type.HALF,
)
k = graph.tensor(
    name="K",
    dim=[B, H, S_kv, D],
    stride=[H * S_kv * D, S_kv * D, D, 1],
    data_type=cudnn.data_type.HALF,
)
v = graph.tensor(
    name="V",
    dim=[B, H, S_kv, D],
    stride=[H * S_kv * D, S_kv * D, D, 1],
    data_type=cudnn.data_type.HALF,
)

# Attention scale
attn_scale = 1.0 / (D ** 0.5)

# Define the SDPA forward operation
o, stats = graph.sdpa(
    name="sdpa_forward",
    q=q, k=k, v=v,
    is_inference=True,
    attn_scale=attn_scale,
    use_causal_mask=True,
)

# Mark output tensor
o.set_output(True).set_dim([B, H, S_q, D]).set_stride([H * S_q * D, S_q * D, D, 1])

# Build the graph
graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
graph.check_support()
graph.build_plans()

# Allocate GPU tensors
q_gpu = torch.randn(B, H, S_q, D, dtype=torch.float16, device="cuda")
k_gpu = torch.randn(B, H, S_kv, D, dtype=torch.float16, device="cuda")
v_gpu = torch.randn(B, H, S_kv, D, dtype=torch.float16, device="cuda")
o_gpu = torch.empty(B, H, S_q, D, dtype=torch.float16, device="cuda")

# Allocate workspace
workspace_size = graph.get_workspace_size()
workspace = torch.empty(workspace_size, dtype=torch.uint8, device="cuda")

# Execute
graph.execute(
    {q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu},
    workspace,
    handle=cudnn.create_handle(),
)
```

### Python API: Forward + Backward (Training)

```python
# For training, set is_inference=False to get softmax statistics for backward
graph = cudnn.pygraph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

q = graph.tensor(name="Q", dim=[B, H, S_q, D], stride=[H*S_q*D, S_q*D, D, 1])
k = graph.tensor(name="K", dim=[B, H, S_kv, D], stride=[H*S_kv*D, S_kv*D, D, 1])
v = graph.tensor(name="V", dim=[B, H, S_kv, D], stride=[H*S_kv*D, S_kv*D, D, 1])

o, stats = graph.sdpa(
    name="sdpa_fwd",
    q=q, k=k, v=v,
    is_inference=False,  # Keep stats for backward
    attn_scale=1.0 / (D ** 0.5),
    use_causal_mask=True,
)
o.set_output(True).set_dim([B, H, S_q, D])
stats.set_output(True).set_dim([B, H, S_q, 1])  # log-sum-exp per row

# Build and execute forward graph...
# Then build backward graph:

bwd_graph = cudnn.pygraph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

# Backward inputs
q_bwd = bwd_graph.tensor(name="Q", dim=[B, H, S_q, D], stride=[H*S_q*D, S_q*D, D, 1])
k_bwd = bwd_graph.tensor(name="K", dim=[B, H, S_kv, D], stride=[H*S_kv*D, S_kv*D, D, 1])
v_bwd = bwd_graph.tensor(name="V", dim=[B, H, S_kv, D], stride=[H*S_kv*D, S_kv*D, D, 1])
o_bwd = bwd_graph.tensor(name="O", dim=[B, H, S_q, D], stride=[H*S_q*D, S_q*D, D, 1])
do = bwd_graph.tensor(name="dO", dim=[B, H, S_q, D], stride=[H*S_q*D, S_q*D, D, 1])
stats_bwd = bwd_graph.tensor(name="stats", dim=[B, H, S_q, 1], stride=[H*S_q, S_q, 1, 1])

dq, dk, dv = bwd_graph.sdpa_backward(
    name="sdpa_bwd",
    q=q_bwd, k=k_bwd, v=v_bwd,
    o=o_bwd, dO=do, stats=stats_bwd,
    attn_scale=1.0 / (D ** 0.5),
    use_causal_mask=True,
)

dq.set_output(True).set_dim([B, H, S_q, D])
dk.set_output(True).set_dim([B, H, S_kv, D])
dv.set_output(True).set_dim([B, H, S_kv, D])

# Build and execute backward graph...
```

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

cuDNN supports grouped-query attention (GQA) and multi-query attention (MQA) through broadcasting dimensions:

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

### Variable Sequence Length Support

```python
# Variable-length sequences with sequence length arrays
seq_len_q = graph.tensor(
    name="seq_len_q",
    dim=[B, 1, 1, 1],
    stride=[1, 1, 1, 1],
    data_type=cudnn.data_type.INT32,
)
seq_len_kv = graph.tensor(
    name="seq_len_kv",
    dim=[B, 1, 1, 1],
    stride=[1, 1, 1, 1],
    data_type=cudnn.data_type.INT32,
)

o, stats = graph.sdpa(
    name="sdpa_varlen",
    q=q, k=k, v=v,
    is_inference=True,
    attn_scale=attn_scale,
    use_causal_mask=True,
    use_padding_mask=True,
    seq_len_q=seq_len_q,
    seq_len_kv=seq_len_kv,
)
```

## PyTorch SDPA Integration

### How PyTorch Dispatches to cuDNN

PyTorch's `scaled_dot_product_attention` includes cuDNN as one of four backend options (since PyTorch 2.1). The dispatch logic is:

```
1. Check if CUDNN_ATTENTION backend is enabled (default: True on supported hardware)
2. Validate constraints:
   - CUDA device with cuDNN available
   - dtype is fp16 or bf16 (fp8 requires explicit cuDNN graph API)
   - head_dim is in {64, 128, 256} (expanded in newer cuDNN versions)
   - No unsupported attention mask types
   - cuDNN version >= 8.9.5 (recommended >= 9.0)
3. If all checks pass, dispatch to cuDNN backend
4. If cuDNN fails validation, fall through to FlashAttention, then memory-efficient, then math
```

### Backend Priority and Control

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Default dispatch order (PyTorch 2.5+):
# 1. cuDNN attention (if supported)
# 2. FlashAttention
# 3. Memory-efficient attention
# 4. Math fallback

# Force cuDNN only
with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    output = F.scaled_dot_product_attention(query, key, value, is_causal=True)

# Disable cuDNN, prefer FlashAttention
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
    output = F.scaled_dot_product_attention(query, key, value, is_causal=True)

# Global enable/disable
torch.backends.cuda.enable_cudnn_sdp(True)   # Enable cuDNN backend
torch.backends.cuda.enable_cudnn_sdp(False)  # Disable cuDNN backend

# Check which backend was selected (debugging)
from torch.nn.attention import _get_flash_attention_flag_and_target
# Or use torch profiler to see which kernel was launched
```

### cuDNN Backend Constraints in PyTorch

The cuDNN SDPA backend has specific requirements that determine when it is eligible:

| Parameter | Requirement |
|---|---|
| Data type | FP16 or BF16 |
| Head dimension | 64, 128, or 256 (cuDNN 8.9.5); up to 256 with strides (cuDNN 9.x) |
| Sequence length | No strict upper limit (memory-dependent) |
| Batch size | >= 1 |
| Causal mask | Supported |
| Arbitrary mask | Supported (via `attn_mask` parameter) |
| Dropout | Supported |
| Nested tensors | Supported (PyTorch 2.4+) |
| `scale` parameter | Supported |
| `enable_gqa` | Supported (PyTorch 2.5+) |
| Return attention weights | NOT supported (forces math fallback) |
| Non-contiguous tensors | May fall back to other backends |

### torch.compile Integration

cuDNN attention works seamlessly with `torch.compile`:

```python
@torch.compile
def attention_fn(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

# torch.compile can fuse surrounding operations with the cuDNN attention call
output = attention_fn(query, key, value)
```

When `torch.compile` is used, the compiler may additionally fuse pre/post-attention operations (e.g., RoPE, projection) with the cuDNN attention kernel through graph-level optimization.

## Heuristics and Auto-Tuning

### cuDNN Heuristic Modes

cuDNN provides multiple heuristic strategies for selecting the optimal kernel configuration:

```python
# Mode A: Fast heuristic (default) -- uses precomputed lookup tables
graph.create_execution_plans([cudnn.heur_mode.A])

# Mode B: More exhaustive search -- slower plan creation, potentially faster execution
graph.create_execution_plans([cudnn.heur_mode.B])

# Fallback: Try mode A first, then fall back to mode B if A fails
graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
```

### Execution Plan Selection

cuDNN may generate multiple valid execution plans for a given attention configuration. You can enumerate and benchmark them:

```python
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])

# Get number of plans
num_plans = graph.get_execution_plan_count()
print(f"Number of execution plans: {num_plans}")

# Benchmark each plan (pseudo-code)
for plan_idx in range(num_plans):
    graph.build_plans(plan_idx)
    time = benchmark(graph.execute, ...)
    print(f"Plan {plan_idx}: {time:.3f} ms")
```

## Supported Configuration Matrix

### Data Types

| Data Type | Forward | Backward | GPU Requirement | cuDNN Version |
|---|---|---|---|---|
| FP16 | Yes | Yes | Ampere+ (SM80+) | 8.9.5+ |
| BF16 | Yes | Yes | Ampere+ (SM80+) | 8.9.5+ |
| FP8 E4M3 | Yes | Yes | Hopper+ (SM90+) | 9.0+ |
| FP8 E5M2 | dO only | Backward only | Hopper+ (SM90+) | 9.0+ |
| FP32 | No | No | -- | -- |

### Head Dimensions

| Head Dim | cuDNN 8.9.x | cuDNN 9.0+ |
|---|---|---|
| 32 | Limited | Supported |
| 64 | Supported | Supported |
| 96 | Limited | Supported |
| 128 | Supported | Supported |
| 160 | Not supported | Supported (9.1+) |
| 192 | Not supported | Supported (9.1+) |
| 256 | Supported | Supported |

### Masking Types

| Mask Type | Supported | Notes |
|---|---|---|
| No mask | Yes | Standard attention |
| Causal (lower triangular) | Yes | Via `use_causal_mask=True` |
| Padding mask | Yes | Via `seq_len_q` / `seq_len_kv` tensors |
| Arbitrary additive bias | Yes | Via bias tensor input |
| Sliding window | No | Not natively supported |
| Block-sparse | No | Not supported |
| Custom score modification | Limited | Only pre-defined operations |

## Performance Characteristics

### cuDNN vs FlashAttention-2 (Ampere A100)

Benchmark conditions: FP16, head_dim=128, 32 heads, batch_size=4

| Sequence Length | cuDNN 9.0 (ms) | FlashAttention-2 (ms) | Speedup |
|---|---|---|---|
| 512 | 0.42 | 0.38 | FA2 1.11x faster |
| 1024 | 1.15 | 1.08 | FA2 1.06x faster |
| 2048 | 3.82 | 3.65 | FA2 1.05x faster |
| 4096 | 14.1 | 13.5 | FA2 1.04x faster |
| 8192 | 54.8 | 52.1 | FA2 1.05x faster |

On Ampere, FlashAttention-2 is generally slightly faster than cuDNN due to hand-tuned tiling optimized for A100's SM80 architecture.

### cuDNN vs FlashAttention-3 (Hopper H100)

Benchmark conditions: FP16, head_dim=128, 32 heads, batch_size=4

| Sequence Length | cuDNN 9.1 (ms) | FlashAttention-3 (ms) | Winner |
|---|---|---|---|
| 512 | 0.18 | 0.21 | cuDNN 1.17x |
| 1024 | 0.52 | 0.48 | FA3 1.08x |
| 2048 | 1.72 | 1.55 | FA3 1.11x |
| 4096 | 6.35 | 5.80 | FA3 1.09x |
| 8192 | 24.5 | 22.1 | FA3 1.11x |
| 16384 | 96.2 | 85.8 | FA3 1.12x |

On Hopper, cuDNN is competitive for short sequences but FlashAttention-3 pulls ahead for medium-to-long sequences (1K+) due to more aggressive pipelining and warp specialization.

### cuDNN FP8 vs FP16 (Hopper H100)

| Sequence Length | cuDNN FP16 (ms) | cuDNN FP8 (ms) | FP8 Speedup |
|---|---|---|---|
| 2048 | 1.72 | 1.05 | 1.64x |
| 4096 | 6.35 | 3.80 | 1.67x |
| 8192 | 24.5 | 14.2 | 1.73x |

FP8 attention provides ~1.6-1.7x speedup due to doubled Tensor Core throughput, though the actual speedup is less than the theoretical 2x because softmax remains in FP32 and memory bandwidth (not compute) becomes the bottleneck for shorter sequences.

### cuDNN vs FlashAttention-4 (Blackwell B200)

FlashAttention-4 achieves approximately 1.2-1.3x speedup over cuDNN 9.13 on Blackwell B200 GPUs. The FA4 advantage comes from:
- Asymmetric scaling-aware pipeline design
- Better utilization of Blackwell's 5th-gen Tensor Cores
- Custom SFU (Special Function Unit) scheduling to avoid softmax bottlenecks
- Hand-tuned PTX for maximum instruction-level parallelism

## Integration with Frameworks

### FlashInfer cuDNN Backend

FlashInfer (the attention engine used in vLLM and SGLang) includes cuDNN as a backend option for prefill attention:

```python
import flashinfer

# Use cuDNN backend for batch prefill
handler = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer,
    backend="cudnn",  # Use cuDNN instead of FA2/FA3
)
```

This is particularly useful when cuDNN provides better performance for specific problem sizes or when FP8 attention is needed.

### TensorRT-LLM cuDNN Integration

TensorRT-LLM uses cuDNN fused attention as one of its context (prefill) kernel options:

```python
# TRT-LLM automatically selects cuDNN when:
# 1. FMHA plugin does not support the configuration
# 2. FP8 context attention is enabled (always uses cuDNN on Hopper)
# 3. cuDNN provides better performance per heuristic
build_config = BuildConfig(
    plugin_config={
        "use_fp8_context_fmha": True,  # Forces cuDNN FP8 path on Hopper
    }
)
```

## Debugging and Profiling

### Verifying Backend Selection in PyTorch

```python
import torch
from torch.profiler import profile, ProfilerActivity

query = torch.randn(2, 32, 2048, 128, device="cuda", dtype=torch.float16)
key = torch.randn(2, 32, 2048, 128, device="cuda", dtype=torch.float16)
value = torch.randn(2, 32, 2048, 128, device="cuda", dtype=torch.float16)

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )

# Look for kernel names in the trace
for event in prof.key_averages():
    if "sdp" in event.key.lower() or "attention" in event.key.lower():
        print(f"{event.key}: {event.cuda_time_total:.1f} us")
# cuDNN kernel names typically contain "cudnn" or "fused_attn"
```

### Common Issues

**1. cuDNN backend not selected:**
```python
# Check if cuDNN SDPA is enabled
print(torch.backends.cuda.cudnn_sdp_enabled())  # Should be True

# Check cuDNN version
print(torch.backends.cudnn.version())  # Should be >= 8905

# Verify head_dim is supported
# Common issue: head_dim=80 (GPT-NeoX) not supported by older cuDNN
```

**2. Performance regression after cuDNN update:**
```python
# cuDNN heuristics change between versions
# Force a specific backend to isolate:
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output_fa = F.scaled_dot_product_attention(q, k, v, is_causal=True)

with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    output_cudnn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**3. Numerical differences between backends:**
```python
# cuDNN and FlashAttention may produce slightly different results
# due to different accumulation orders and reduced-precision intermediates
with sdpa_kernel(SDPBackend.MATH):
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)

with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    cudnn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

max_diff = (reference.float() - cudnn_out.float()).abs().max()
print(f"Max absolute difference: {max_diff:.6f}")
# Typical: 1e-3 to 1e-2 for FP16, 1e-2 to 5e-2 for BF16
```

## cuDNN Attention Internals

### Memory Layout

cuDNN attention expects tensors in BHSD (batch, head, sequence, dimension) layout, matching PyTorch's default. Strides must be specified explicitly in the graph API, allowing non-contiguous layouts.

The internal tiling strategy varies by GPU architecture:
- **Ampere**: Typically tiles with Bq=64-128, Bkv=64, using HMMA_16816 instructions
- **Hopper**: Tiles with Bq=64-128, Bkv=64-128, using WGMMA instructions with TMA loads
- **Blackwell**: Larger tiles enabled by increased shared memory

### Workspace Memory

cuDNN requires a workspace buffer for temporary storage during execution:

```python
workspace_size = graph.get_workspace_size()
# Typical sizes:
# FP16, seq_len=2048, batch=4, 32 heads: ~16-64 MB
# FP8 with scaling: additional ~1-4 MB for scaling tensors
workspace = torch.empty(workspace_size, dtype=torch.uint8, device="cuda")
```

The workspace is reusable across calls with the same graph configuration.

### Thread Block Configuration

cuDNN's attention kernels use architecture-specific thread block configurations:
- **Ampere**: 128-256 threads per block, 1 warp-group per block
- **Hopper**: 128-256 threads per block with warp specialization (producer + consumer warps)
- **Blackwell**: Up to 512 threads per block leveraging larger register files

## Decision Framework: cuDNN vs FlashAttention vs Others

### Use cuDNN when:
1. **Stability is paramount**: cuDNN is a vendor library with rigorous testing and backward compatibility
2. **You need FP8 attention**: cuDNN 9.x has the most mature FP8 SDPA implementation
3. **Short sequences (< 1K)**: cuDNN's heuristic-tuned kernels often win for short sequences on Hopper
4. **You want zero maintenance**: cuDNN updates automatically with CUDA toolkit upgrades
5. **Deterministic execution**: cuDNN's deterministic mode is well-tested for reproducibility

### Use FlashAttention-2/3 when:
1. **Medium-to-long sequences (1K+)**: FA2/FA3 consistently outperform cuDNN for longer sequences
2. **You need paged KV cache**: FA2/FA3 have built-in paged KV cache support for inference
3. **You need sliding window attention**: FA2/FA3 support window-based local attention
4. **Maximum performance on Hopper**: FA3 with warp specialization and pingpong scheduling is 1.05-1.15x faster than cuDNN for typical workloads
5. **Open-source requirement**: FA2/FA3 are fully open-source; cuDNN kernels are closed-source

### Use FlexAttention when:
1. **Custom score modifications**: ALiBi, RoPE, soft-capping, or any user-defined score function
2. **Portability**: FlexAttention (via `torch.compile`) generates kernels for any supported hardware
3. **Rapid prototyping**: Define attention variants in pure Python

## Version History

| cuDNN Version | Key Attention Features |
|---|---|
| 8.6.0 | Initial fused attention support (limited configs) |
| 8.9.0 | Expanded head_dim support, padding masks |
| 8.9.5 | PyTorch SDPA integration, GQA support |
| 9.0.0 | Graph API v2, FP8 E4M3 attention, Hopper optimizations |
| 9.1.0 | Expanded head_dim (32-256), improved heuristics |
| 9.1.1 | FA3 paper benchmark baseline, TMA optimizations |
| 9.5.0 | Blackwell support, enhanced FP8 backward pass |
| 9.13.0 | FA4 paper benchmark baseline, improved Blackwell kernels |

## Key Notes

- cuDNN Fused Attention is NVIDIA's official, vendor-optimized attention kernel -- it is the baseline against which all open-source attention implementations are benchmarked
- The cudnn-frontend graph API provides maximum flexibility for custom attention variants, but the PyTorch SDPA integration is simpler for standard use cases
- cuDNN receives architecture-specific optimizations with each new GPU generation, often providing day-one support
- Performance is highly dependent on the specific cuDNN version, GPU architecture, and problem size -- always benchmark for your configuration
- The cuDNN backend is closed-source, which means you cannot inspect or modify the kernel internals -- this is a trade-off for production stability
- For the latest performance numbers, consult the FlashAttention-3 and FlashAttention-4 papers, which include detailed cuDNN comparisons
