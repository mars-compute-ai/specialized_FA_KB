---
skill_name: TensorRT-LLM Attention Kernel Dispatch
description: Understanding TensorRT-LLM's multi-kernel attention dispatch system that selects among FMHA, cuDNN fused attention, XQA decode kernels, and paged KV cache attention based on inference phase and model configuration.
level: L0 - Implementation Selection Level
target_hardware: NVIDIA GPUs (Ampere A100, Hopper H100/H200, Blackwell B100/B200); requires TensorRT-LLM 0.7+
relevance: When deploying LLM inference with TensorRT-LLM and understanding which attention kernel is dispatched for prefill vs decode, how to configure kernel selection, and how to optimize for specific workloads.
---

# TensorRT-LLM Attention Kernel Dispatch

## What It Is
TensorRT-LLM is NVIDIA's high-performance inference framework for large language models, and it implements a sophisticated multi-kernel attention dispatch system. Rather than using a single attention kernel for all scenarios, TRT-LLM selects the optimal kernel from a set of specialized implementations based on the inference phase (context/prefill vs generation/decode), model architecture (MHA/GQA/MQA), sequence length, head dimension, data type, and hardware generation. The primary kernels are: (1) FMHA (Fused Multi-Head Attention) for context/prefill phase, (2) cuDNN fused attention as an alternative context kernel, (3) XQA (Cross-Query Attention) for GQA/MQA decode, and (4) paged KV cache attention for general decode. Understanding this dispatch is critical for optimizing TRT-LLM deployment performance.

## Key Concepts
- **Context phase (prefill) kernels**:
  - **FMHA plugin**: TRT-LLM's in-house fused multi-head attention kernel, optimized for compute-bound prefill workloads. Supports FP16/BF16, head dims 32-256, causal masking, and variable sequence lengths via the `gpt_attention_plugin`.
  - **cuDNN fused attention**: NVIDIA's cuDNN SDPA used as an alternative context kernel. Selected when FMHA does not support the specific configuration or when cuDNN provides better performance for the given problem size.
- **Generation phase (decode) kernels**:
  - **XQA kernel**: Specialized kernel for GQA/MQA decode that exploits the reduced number of KV heads. Achieves high memory bandwidth utilization by assigning multiple thread blocks per KV head. Critical for Llama-3, Mistral, and other GQA models.
  - **Paged KV cache attention**: General-purpose decode kernel that reads from the paged (block-based) KV cache. Used when XQA is not applicable.
- **Paged KV cache**: TRT-LLM uses a block-based KV cache (similar to vLLM's PagedAttention) where cache memory is allocated in fixed-size blocks. All decode kernels must handle the block-table indirection to locate KV entries.
- **Inflight batching**: TRT-LLM's scheduler can batch context (prefill) and generation (decode) requests together, requiring the attention layer to dispatch different kernels for different requests within the same batch.
- **GPT Attention Plugin**: The `gpt_attention_plugin` is TRT-LLM's unified attention layer that encapsulates the kernel selection logic, RoPE application, KV cache management, and output projection.
- **FP8 attention**: On Hopper+ GPUs with cuDNN 9.x, TRT-LLM supports FP8 context attention for 2x compute throughput. The XQA decode kernel also supports FP8 KV cache.

## When to Use
- You are deploying LLMs with TensorRT-LLM and need to understand why certain attention kernels are selected and how to influence the selection
- You are optimizing TRT-LLM inference latency and need to ensure the optimal kernel is dispatched for your model architecture and workload mix
- You are debugging performance regressions in TRT-LLM and need to identify which attention kernel path is being taken
- You need to understand the interaction between paged KV cache, inflight batching, and attention kernel dispatch
- You are evaluating TRT-LLM vs vLLM or other frameworks and want to understand the architectural differences in attention kernel strategies

## When NOT to Use
- You are doing training -- TRT-LLM is inference-only; use PyTorch SDPA or FlashAttention for training
- You need to modify the attention kernel internals -- TRT-LLM's kernels are closed-source CUDA; use FlashAttention or Triton if you need kernel-level customization
- You are on AMD GPUs -- TRT-LLM is NVIDIA-only
- You need exotic attention patterns (sparse attention, linear attention) not supported by TRT-LLM's attention plugin
- Your model is small enough that the overhead of TRT-LLM's build/deploy pipeline is not justified -- use vLLM or direct PyTorch inference

## Code Snippets / Pseudo-code

```python
# TRT-LLM model build with attention plugin configuration
from tensorrt_llm import Builder, BuildConfig
from tensorrt_llm.models import LLaMAForCausalLM

# Build TRT-LLM engine with attention plugin
build_config = BuildConfig(
    max_batch_size=64,
    max_input_len=2048,
    max_seq_len=8192,
    max_num_tokens=8192,
    plugin_config={
        "gpt_attention_plugin": "float16",     # Enable fused attention plugin
        "paged_kv_cache": True,                # Enable paged KV cache
        "tokens_per_block": 64,                # KV cache block size
        "use_fp8_context_fmha": False,         # Enable FP8 context attention (Hopper+)
        "use_xqa": True,                       # Enable XQA for GQA decode
        "context_fmha_type": "enabled",        # "enabled" or "enabled_with_fp32_acc"
    }
)

model = LLaMAForCausalLM.from_hugging_face(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype="float16",
)
engine = Builder().build(model, build_config)
```

```
# TRT-LLM kernel dispatch logic (simplified pseudo-code)
def select_attention_kernel(phase, model_config, hw_config):
    if phase == "context":  # Prefill
        if use_fp8 and hw_config.is_hopper_plus:
            return "cuDNN_FP8_FMHA"
        elif fmha_supports(model_config.head_dim, model_config.dtype):
            return "FMHA_plugin"
        else:
            return "cuDNN_fused_attention"

    elif phase == "generation":  # Decode
        if model_config.is_gqa_or_mqa and xqa_supports(model_config):
            return "XQA_kernel"         # Specialized GQA/MQA decode
        else:
            return "paged_kv_attention" # General decode kernel
```

## Key Takeaways
- TRT-LLM uses a multi-kernel strategy: different specialized kernels for context (prefill) vs generation (decode) phases, unlike frameworks that use a single attention implementation
- The XQA kernel is the key differentiator for GQA/MQA models in decode -- it achieves high memory bandwidth utilization by exploiting the reduced KV head count
- The `gpt_attention_plugin` encapsulates all kernel selection logic; ensure it is enabled (it is by default) for optimal performance
- For GQA models like Llama-3, verify XQA is being used during decode -- this can provide 2-3x decode throughput improvement over generic paged attention
- FP8 context attention (via cuDNN) on Hopper+ doubles compute throughput for prefill-heavy workloads
- Paged KV cache is mandatory in TRT-LLM's production configuration -- it enables efficient memory management and inflight batching
- TRT-LLM's kernel dispatch is largely automatic, but understanding the dispatch logic helps diagnose performance issues and validate that the optimal kernel path is active

## References
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [TensorRT-LLM GitHub Repository](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT-LLM Architecture: Attention](https://nvidia.github.io/TensorRT-LLM/architecture/attention.html)
- [TensorRT-LLM GPT Attention Plugin](https://nvidia.github.io/TensorRT-LLM/architecture/gpt-attention.html)
- [XQA Kernel for GQA Decode](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html)
- [Inflight Batching Documentation](https://nvidia.github.io/TensorRT-LLM/advanced/batch-manager.html)
