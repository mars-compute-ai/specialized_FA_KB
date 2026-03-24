---
skill_name: vLLM Attention Backend Selection
description: Understanding vLLM's attention backend dispatch system, which selects among FlashAttention-2, FlashInfer, xFormers, and Triton backends based on hardware, model architecture, and serving configuration.
level: L0 - Implementation Selection Level
target_hardware: NVIDIA GPUs (Ampere A100, Hopper H100/H200, Ada L40S, consumer RTX); AMD GPUs (MI250X, MI300X); CPU (limited)
relevance: When configuring vLLM for optimal LLM inference performance and choosing the right attention backend for a specific hardware/model/workload combination.
---

# vLLM Attention Backend Selection

## What It Is
vLLM is the most widely-used open-source LLM inference serving framework, and its performance is heavily determined by which attention kernel backend is used. vLLM supports multiple attention backends -- FlashAttention-2, FlashInfer, xFormers, Triton Flash Attention, and platform-specific backends (e.g., ROCm Flash Attention for AMD). The backend selection can be automatic (based on hardware detection, model architecture, and feature requirements) or manually configured via the `VLLM_ATTENTION_BACKEND` environment variable. The choice of backend affects throughput, latency, memory usage, and which features (PagedAttention, chunked prefill, speculative decoding, sliding window, FP8 KV cache) are available.

## Key Concepts
- **PagedAttention**: vLLM's foundational innovation -- manages KV cache in fixed-size pages (blocks) like virtual memory, enabling near-zero waste and dynamic memory sharing across requests. All backends must support paged KV cache access.
- **Backend options**:
  - **FlashAttention-2** (`FLASH_ATTN`): Default on NVIDIA Ampere+. High performance for both prefill and decode. Supports PagedAttention via custom paged KV cache kernels.
  - **FlashInfer** (`FLASHINFER`): Optimized for decode-heavy workloads with GQA Tensor Core decode, ragged tensors, and CUDAGraph-friendly wrappers. Often faster for high-throughput decode.
  - **xFormers** (`XFORMERS`): Memory-efficient attention from Facebook Research. Used as fallback on older NVIDIA GPUs or when FlashAttention is unavailable.
  - **Triton Flash Attention** (`TRITON_FLASH_ATTN`): Pure Triton implementation. Portable, used for AMD ROCm or when CUDA-specific backends are unavailable.
  - **ROCm Flash Attention** (`ROCM_FLASH`): AMD-optimized attention kernel for MI200X/MI300X GPUs.
- **Prefill vs Decode**: Prefill (prompt processing) is compute-bound; decode (token generation) is memory-bandwidth-bound. Some backends excel at one or the other.
- **Chunked prefill**: Splits long prefill sequences into chunks to overlap with decode batches, reducing time-to-first-token (TTFT) variation. Backend must support this feature.
- **CUDAGraph compatibility**: For decode, CUDAGraph capture eliminates kernel launch overhead. FlashInfer provides CUDAGraph-compatible wrappers; FlashAttention-2 requires workarounds.
- **KV cache quantization**: FP8 and INT8 KV cache quantization reduces memory usage. Backend support varies -- FlashInfer and FlashAttention-2 (recent versions) support FP8 KV.

## When to Use
- **FlashAttention-2 backend**: Default choice for NVIDIA Ampere/Hopper GPUs; best all-around performance for mixed prefill+decode workloads; broadest feature support in vLLM
- **FlashInfer backend**: When decode throughput is critical (high QPS serving), when using GQA models (Llama-3, Mistral), when CUDAGraph decode is needed, or when using speculative decoding
- **xFormers backend**: On older NVIDIA GPUs (V100, P100) or when FlashAttention installation fails; when fp32 attention is needed
- **Triton backend**: On AMD GPUs via ROCm, or for debugging/development when a pure-Python-defined kernel is preferred
- **ROCm Flash Attention**: On AMD MI250X/MI300X for production inference

## When NOT to Use
- Do not use xFormers when FlashAttention-2 or FlashInfer is available -- it is strictly slower on modern hardware
- Do not use FlashInfer if your workload is prefill-dominated (long prompt, single completion) -- FlashAttention-2 is typically faster for prefill
- Do not manually select a backend unless benchmarking shows a clear advantage -- vLLM's auto-selection is reasonable for most cases
- Do not use Triton Flash Attention on NVIDIA GPUs unless the CUDA-native backends are unavailable

## Code Snippets / Pseudo-code

```bash
# Set attention backend via environment variable
export VLLM_ATTENTION_BACKEND=FLASH_ATTN       # FlashAttention-2 (default on NVIDIA)
export VLLM_ATTENTION_BACKEND=FLASHINFER        # FlashInfer backend
export VLLM_ATTENTION_BACKEND=XFORMERS          # xFormers backend
export VLLM_ATTENTION_BACKEND=TRITON_FLASH_ATTN # Triton backend

# Launch vLLM with specific backend
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --enable-chunked-prefill        # Enables chunked prefill (works with FA2 and FlashInfer)

# FlashInfer with CUDAGraph for decode optimization
export VLLM_ATTENTION_BACKEND=FLASHINFER
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --enforce-eager false           # Enable CUDAGraph (default)
    --max-num-seqs 256
```

```python
# Programmatic backend selection in vLLM
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.9,
    enable_chunked_prefill=True,
)

# vLLM's internal backend selection logic (simplified):
# 1. Check VLLM_ATTENTION_BACKEND env var
# 2. If not set, detect hardware (NVIDIA vs AMD vs CPU)
# 3. On NVIDIA: prefer FlashAttention-2 if available, else xFormers
# 4. Check model requirements (sliding window, head_dim, dtype)
# 5. Validate backend supports required features
```

## Source Code Examples

### Block Table and KV Cache Structure (PagedAttention)

```python
# Block table: maps logical block indices to physical block indices
# Shape: [max_num_seqs, max_num_blocks_per_seq]
block_table = torch.tensor([
    [3, 7, 12, 0, 0],   # Request 0: blocks 3, 7, 12 (padded with 0s)
    [1, 5, 9, 15, 0],   # Request 1: blocks 1, 5, 9, 15
    [2, 8, 0, 0, 0],    # Request 2: blocks 2, 8
], dtype=torch.int32, device="cuda")

# KV cache: physical blocks stored contiguously
# Shape: [num_blocks, 2, num_heads, block_size, head_dim]
# 2 = K and V stored together
kv_cache = torch.empty(
    num_blocks, 2, num_kv_heads, block_size, head_dim,
    dtype=torch.float16, device="cuda"
)
```

## Key Takeaways
- vLLM's attention backend choice has a significant impact on serving throughput and latency -- always benchmark for your specific model and hardware
- FlashAttention-2 is the safe default on NVIDIA GPUs; FlashInfer can provide 10-30% higher decode throughput for GQA models at high batch sizes
- All production backends support PagedAttention -- this is a hard requirement in vLLM's architecture
- Chunked prefill is critical for latency-sensitive serving -- verify your chosen backend supports it
- FlashInfer's CUDAGraph compatibility makes it the best choice for decode-dominated workloads where kernel launch overhead matters
- For AMD GPUs, the ROCm Flash Attention or Triton backend are the only options -- performance tuning is more limited
- Monitor vLLM releases closely -- backend support and default selections change frequently across versions

## References
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [PagedAttention Paper (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [FlashInfer Documentation](https://flashinfer.ai/)
- [vLLM Attention Backend Source Code](https://github.com/vllm-project/vllm/tree/main/vllm/attention/backends)
- [Chunked Prefill Blog Post](https://docs.vllm.ai/en/latest/design/kernel/chunked_prefill.html)
