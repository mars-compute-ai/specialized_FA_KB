---
skill_name: AMD FMHA Kernel Internals
description: Comprehensive internals of AMD's Flash Multi-Head Attention kernel variants including V3 forward with wave-group scheduling, backward with transpose load pipeline, Split-KV decode with paged KV cache, FP8 block-scale quantized attention, and batch prefill with MFMA-aligned memory access.
level: L6 - Micro-Kernel/Tensor-Core Level
target_hardware: AMD MI300X (gfx942), MI350X (gfx950)
relevance: When optimizing or understanding AMD's Flash Attention kernel internals, debugging FMHA performance on MI300X/MI350X, or porting attention kernels to AMD CDNA architectures
---

# AMD FMHA Kernel Internals

## What It Is
AMD's Flash Multi-Head Attention (FMHA) implementation in Composable Kernel (CK) is a family of highly optimized attention kernels for CDNA3 (MI300X) and CDNA4 (MI350X) GPUs. These kernels are optimized by the GEAK AI agent and human engineers, featuring wave-group scheduling, MFMA instruction pipeline optimization, transpose load operations, paged KV cache, FP8 block-scale quantization, and batch prefill with vectorized memory layouts.

## Key Kernel Variants

### 1. FMHA V3 Forward Pass
- Wave-group based `CoreLoopScheduler` that interleaves MFMA, TRANS, VALU, and SALU instructions across phases
- Phase-aware scheduling with `__builtin_amdgcn_sched_group_barrier` for instruction ordering
- Packed FP32 operations via inline assembly (`v_pk_mul_f32`, `v_fma_f32`, `v_cvt_pk_f16_f32`)
- Dynamic memory access counts from tile window properties
- O_acc rescaling distribution across phases to avoid SIMD idle cycles

### 2. FMHA Backward Pass (dQ/dK/dV)
- Transpose load (trload) pipeline for GFX950 with IGLP scheduling
- Two-stage prefetching for K and V with transposition during load
- Decode-specific pipeline for short query sequences (seqlen_q=1)
- Reduced padding from power-of-2 to MFMA-aligned multiples of 8
- 64-bit arithmetic for deterministic mode to prevent integer overflow

### 3. Split-KV Decode
- Specialized for inference decode where seqlen_q=1 and seqlen_k is large
- Split-K parallelism: K dimension divided across multiple workgroups
- Paged KV cache support (vLLM 2D block tables, SGLang 1D page tables)
- Online softmax combine kernel for merging partial results across splits
- Reverse block index assignment for causal mask load balancing
- Attention sink support for streaming/infinite context inference

### 4. FP8 Block-Scale Quantized Attention
- FP8 (E4M3) storage with per-block descale factors for K and V
- OCP FP8 shift (8.0f) vs FNUZ FP8 shift (7.0f) for dynamic range handling
- Per-block dequantization during GEMM accumulation phase
- Dedicated kernel argument structures with stride information for descale tensors

### 5. Batch Prefill with Paged KV Cache
- 3-level K-dimension decomposition: K = K2 x K0 x K1
- Multi-dimensional page index computation in Y-space for gather operations
- Support for VECTORIZED_LAYOUT (5D swizzled) and LINEAR_LAYOUT KV cache
- MFMA-aligned memory access matching GEMM's warp-level distribution pattern

## When to Use
- Optimizing Flash Attention performance on AMD MI300X or MI350X GPUs
- Debugging attention kernel performance issues on AMD hardware
- Understanding how MFMA instruction scheduling affects attention throughput
- Implementing custom attention variants on AMD CDNA architectures
- Integrating with vLLM or SGLang inference frameworks on AMD GPUs

## When NOT to Use
- Targeting NVIDIA GPUs (use FA3/FA4, CUTLASS, or ThunderKittens)
- Need a high-level attention API (use PyTorch SDPA with CK backend)
- Working on non-attention workloads (use CK's GEMM or convolution templates)

## Key Takeaways
- AMD's FMHA kernels use wave-group scheduling with phase-aware barriers to maximize MFMA utilization
- The transpose load pipeline on GFX950 eliminates separate transpose passes in the backward kernel
- Split-KV decode with paged KV cache enables efficient inference with vLLM/SGLang on AMD
- FP8 block-scale quantization reduces memory bandwidth by 2x while maintaining accuracy via per-block descaling
- The 3-level K-dimension decomposition in batch prefill enables correct page lookup for vectorized KV cache layouts

## References
- [Composable Kernel GitHub Repository](https://github.com/ROCm/composable_kernel)
- [GEAK: Triton Kernel AI Agent (arXiv 2507.23194)](https://arxiv.org/abs/2507.23194)
- [AMD Instinct MI300X Documentation](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [ROCm Flash Attention Documentation](https://rocm.docs.amd.com/)
