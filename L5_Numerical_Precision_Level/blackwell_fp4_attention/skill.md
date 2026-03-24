---
skill_name: Blackwell FP4 (NVFP4/E2M1) Attention
description: Using 4-bit floating-point (FP4 E2M1) for attention computation on NVIDIA Blackwell, including format details, tensor core throughput characteristics, quantization challenges, calibration strategies, and accuracy analysis
level: L5 - Numerical Precision Level
target_hardware: NVIDIA Blackwell B100/B200/GB200 (SM100/SM103, native FP4 tensor cores)
relevance: When deploying inference-time attention on Blackwell with maximum throughput requirements, especially for KV cache compression and decode-phase attention where 4-bit precision may be acceptable
---

# Blackwell FP4 (NVFP4/E2M1) Attention

## What It Is
FP4 (E2M1, also called NVFP4 in NVIDIA's terminology) is a 4-bit floating-point format with 1 sign bit, 2 exponent bits, and 1 mantissa bit, providing only 15 distinct nonzero magnitudes. On NVIDIA Blackwell GPUs, FP4 tensor cores deliver approximately 2x the throughput of FP8 and 4x the throughput of FP16, making FP4 the highest-throughput numerical format available for matrix operations. When combined with MX block scaling (MXFP4: E2M1 elements + E8M0 shared exponent per 32 elements), FP4 provides 4x memory compression over FP16 for KV cache storage and attention computation. However, the extreme quantization (only 1 mantissa bit) presents significant accuracy challenges for attention, requiring careful calibration, outlier management, and selective application to maintain model quality.

## Key Concepts
- **E2M1 format:** 1 sign + 2 exponent + 1 mantissa = 4 bits per value. Representable magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}. Only 8 positive values plus their negatives.
- **Sub-byte packing:** Two FP4 values are packed into a single byte. PyTorch stores FP4 tensors as `uint8` with the inner dimension halved. The CUDA type `__nv_fp4_e2m1` provides hardware support.
- **NVFP4 = MXFP4:** NVIDIA's NVFP4 is the MX-format FP4 with E8M0 per-block (32-element) scaling. The scale factors are stored as `float8_e4m3fn` in practice (FlashInfer) or E8M0 in the OCP specification.
- **Blackwell tensor core throughput:** FP4 tensor cores deliver ~2x throughput compared to FP8, and ~4x compared to FP16/BF16. The B200 achieves approximately 9 PFLOPS in FP4 (vs ~4.5 PFLOPS FP8, ~2.25 PFLOPS FP16).
- **UMMA FP4 support:** The UMMA instruction descriptor supports E2M1 as element format (value 5 in the MXF8F6F4Format encoding). Combined with max_shift for MX scaling, the tensor core handles MXFP4 natively.
- **Accuracy challenge:** With only 1 mantissa bit, FP4 has severe quantization error. For attention, this means: Q*K^T score computation has limited precision, and the softmax input may have significant quantization noise. Practical FP4 attention requires careful selection of which tensors are quantized.
- **FP4Tensor wrapper:** Since PyTorch lacks native FP4 dtype support, frameworks like FlashInfer use a `FP4Tensor` wrapper class containing: `data` (uint8, packed), `scale` (float8_e4m3fn), `scale_start_index`, and `original_shape`.

## Precision Analysis

### E2M1 Value Table

| Bits (SEEM) | Exponent | Mantissa | Value |
|-------------|----------|----------|-------|
| 0 00 0 | 0 | 0 | 0.0 |
| 0 00 1 | 0 | 1 | 0.5 |
| 0 01 0 | 1 | 0 | 1.0 |
| 0 01 1 | 1 | 1 | 1.5 |
| 0 10 0 | 2 | 0 | 2.0 |
| 0 10 1 | 2 | 1 | 3.0 |
| 0 11 0 | 3 | 0 | 4.0 |
| 0 11 1 | 3 | 1 | 6.0 |

With sign bit, the full set is: {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}.

### Quantization Error

```
Relative quantization error for E2M1:
  Between 1.0 and 1.5: max error = 0.25 (relative: 25%)
  Between 2.0 and 3.0: max error = 0.5  (relative: 25%)
  Between 4.0 and 6.0: max error = 1.0  (relative: 25%)

Compare with E4M3 (FP8):
  Between 1.0 and 1.125: max error = 0.0625 (relative: 6.25%)

FP4 has ~4x worse relative precision than FP8.
With block scaling, effective precision depends on value distribution
within each 32-element block.
```

## Code / Configuration
```python
import torch
import math
from typing import Optional, Tuple

# === FP4 Tensor Wrapper (from FlashInfer) ===
class FP4Tensor:
    """Wrapper for FP4 tensors since PyTorch lacks native FP4 dtype.

    - data: uint8 tensor, innermost dim = ceil(original_dim / 2)
            (two E2M1 values packed per byte)
    - scale: float8_e4m3fn tensor for per-block scale factors
    """
    def __init__(self, data: torch.Tensor, scale: torch.Tensor,
                 scale_start_index: int = 0,
                 original_shape: Optional[Tuple[int, ...]] = None):
        assert data.dtype == torch.uint8
        assert scale.dtype == torch.float8_e4m3fn
        self.data = data
        self.scale = scale
        self.scale_start_index = scale_start_index
        self.original_shape = original_shape
        self.dtype = "nvfp4"


# === Software FP4 Quantization ===
def quantize_to_fp4_e2m1(tensor: torch.Tensor, block_size: int = 32):
    """Quantize tensor to MXFP4 (E2M1 + per-block scale)."""
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, block_size)

    # Compute per-block scale (E8M0: power of 2)
    amax = flat.abs().amax(dim=-1)
    log2_amax = torch.log2(amax.clamp(min=1e-38))
    scale_exp = torch.floor(log2_amax).to(torch.int32).clamp(0, 254)
    block_scale = torch.pow(2.0, scale_exp.float() - 127)

    # Scale elements to E2M1 range [0, 6]
    scaled = flat / block_scale.unsqueeze(-1)

    # Quantize to nearest E2M1 value
    # E2M1 representable: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    e2m1_values = torch.tensor([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                device=tensor.device)
    signs = scaled.sign()
    abs_vals = scaled.abs()

    # Find nearest E2M1 value for each element
    diffs = (abs_vals.unsqueeze(-1) - e2m1_values.unsqueeze(0).unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1)  # Index into e2m1_values
    quantized_abs = e2m1_values[indices]
    quantized = signs * quantized_abs

    return quantized.reshape(orig_shape), block_scale


# === FlashInfer NVFP4 Decode Attention ===
# On Blackwell (sm_100/sm_103), FlashInfer uses trtllm-gen backend
from flashinfer import batch_decode_with_kv_cache

result = batch_decode_with_kv_cache(
    query=query,                    # [num_tokens, num_heads, head_dim]
    kv_cache=kv_cache,              # Paged KV cache
    workspace_buffer=workspace,
    block_tables=block_tables,
    seq_lens=seq_lens,
    max_seq_len=max_seq_len,
    out_dtype="nvfp4",              # Request FP4 output
    o_sf_scale=1.0,                 # Output scale factor
    o_sf_vec_size=16,               # Scale factor vector size
    backend="trtllm-gen",           # Required for Blackwell FP4
)
# result is an FP4Tensor with .data and .scale attributes


# === CUDA FP4 Vector Operations (from FlashInfer vec_dtypes.cuh) ===
# FP4 values are sub-byte; two elements pack into one uint8
# (See "Source Code Examples" section below for full implementations)


# === UMMA Descriptor for FP4 Attention ===
from flash_attn.cute.mma_sm100_desc import (
    make_instr_desc, MXF8F6F4Format, MaxShift, Major
)

# FP4 QK^T GEMM descriptor (E2M1 elements with MX scaling)
# Note: E2M1 is value 5 in the MXF8F6F4Format enum
desc_fp4 = make_instr_desc(
    a_type=cutlass.FloatE4M3FN,    # Q in E4M3 (FP8, higher precision)
    b_type=cutlass.FloatE4M3FN,    # K in E4M3 (not E2M1 directly)
    c_type=cutlass.Float32,         # Accumulator
    M=128, N=128,
    a_major=Major.K, b_major=Major.K,
    max_shift=MaxShift.MaxShift8,   # MX block scaling
)
# Note: Most practical FP4 attention uses FP8 for QK^T and FP4 for KV cache/output
```

## Source Code Examples

### FP4 Sub-Byte Packing: vec_t<\_\_nv\_fp4\_e2m1, 2> (FlashInfer vec_dtypes.cuh)

Two FP4 values packed into one uint8, with fill/load/store methods:

```cpp
// Two FP4 values packed into one uint8
template <>
struct vec_t<__nv_fp4_e2m1, 2> {
    uint8_t data;  // Lower 4 bits = element 0, upper 4 bits = element 1

    void fill(__nv_fp4_e2m1 val) {
        // Pack same value into both nibbles
        data = (__nv_fp4x2_storage_t(val.__x) << 4) |
               __nv_fp4x2_storage_t(val.__x);
    }

    void load(const __nv_fp4_e2m1* ptr) { data = *((uint8_t*)ptr); }
    void store(__nv_fp4_e2m1* ptr) const { *((uint8_t*)ptr) = data; }
};
```

### Progressively Larger FP4 Vector Types (FlashInfer vec_dtypes.cuh)

Larger vector types use wider integer storage, up to 128-bit int4 for MX-block-aligned bulk operations:

```cpp
// 4 FP4 values in uint16
template <> struct vec_t<__nv_fp4_e2m1, 4> { uint16_t data; };

// 8 FP4 values in uint32
template <> struct vec_t<__nv_fp4_e2m1, 8> { uint32_t data; };

// 16 FP4 values in uint2 (64 bits)
template <> struct vec_t<__nv_fp4_e2m1, 16> { uint2 data; };

// 32+ FP4 values in int4 arrays (128-bit loads)
template <size_t vec_size>
struct vec_t<__nv_fp4_e2m1, vec_size> {
    static_assert(vec_size % 32 == 0);
    int4 data[vec_size / 32];  // One int4 per 32 elements = one MX block
};
```

### FP4 Bulk Memory Operations (FlashInfer vec_dtypes.cuh)

128-bit (int4) load/store for 32+ FP4 elements, with global memory release semantics:

```cpp
// Bulk load/store using 128-bit (int4) operations for 32+ elements
void load(const __nv_fp4_e2m1* ptr) {
    #pragma unroll
    for (size_t i = 0; i < vec_size / 32; ++i) {
        data[i] = ((int4*)ptr)[i];  // 128-bit load
    }
}

// Global memory operations with memory ordering
void store_global_release(__nv_fp4_e2m1* addr) const {
    #pragma unroll
    for (size_t i = 0; i < vec_size / 32; ++i) {
        st_global_release(*(int4*)&data[i], (int4*)(addr + i * 16));
    }
}
```

Note: 32 FP4 values = 16 bytes = 128 bits (one int4), so the MX block size of 32 aligns perfectly with 128-bit memory operations. This alignment is by design in the OCP MX specification.

## When to Use
- On Blackwell GPUs for inference-time decode attention where memory bandwidth is the primary bottleneck and FP4 KV cache provides 4x compression
- For KV cache quantization to serve longer contexts within the same memory budget (e.g., 4x sequence length in the same GPU memory vs FP16)
- When the model has been specifically calibrated or trained with FP4-aware quantization
- For output quantization (attention output in FP4) when the subsequent layers can tolerate 4-bit input
- In serving scenarios where throughput (tokens/second) matters more than per-token accuracy
- Mixed-precision attention: FP4 for KV cache storage, FP8 for Q*K^T computation, FP32 for softmax

## When NOT to Use
- For training -- FP4 has insufficient precision for gradient computation and weight updates
- On GPUs without FP4 tensor core support (everything before Blackwell)
- For the Q*K^T score computation directly in FP4 -- the 1-mantissa-bit precision causes unacceptable softmax distortion
- When attention accuracy is critical (e.g., mathematical reasoning, code generation with strict correctness requirements)
- For short sequences where memory bandwidth is not the bottleneck and FP16/FP8 throughput is sufficient
- Without proper calibration -- naive FP4 quantization of pretrained weights produces severe accuracy degradation

## Key Takeaways
- FP4 (E2M1) has only 15 distinct nonzero values, making it extremely coarse for representing attention scores; it is most practical for KV cache compression and output quantization rather than full attention computation in FP4
- Blackwell FP4 tensor cores deliver ~2x throughput over FP8 and ~4x over FP16, making FP4 the throughput ceiling for Blackwell attention kernels
- The MXFP4 format (E2M1 + E8M0 per-32 block scaling) extends the effective dynamic range enormously but cannot compensate for the fundamental 1-bit mantissa precision limit
- Sub-byte packing (2 elements per byte) requires special handling in CUDA: FlashInfer's `vec_t<__nv_fp4_e2m1, N>` templates provide efficient pack/unpack, and PyTorch wraps FP4 in uint8 tensors with halved inner dimensions
- Practical FP4 attention deployments use mixed precision: FP4 for KV cache storage, FP8 for Q and the GEMMs, FP32 for softmax -- the GEMMs may dequantize FP4 KV on-the-fly via MX block scaling
- FlashInfer's `FP4Tensor` class is the current standard for FP4 tensor management in Python, wrapping `uint8` data with `float8_e4m3fn` scale factors
- CUDA 12.8+ and the `__nv_fp4_e2m1` type are required for hardware FP4 support; software FP4 is possible on older hardware but without tensor core acceleration

## References
- [NVIDIA Blackwell Architecture Whitepaper (FP4 Tensor Cores)](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [CUTLASS SM100 MMA Descriptor (E2M1 support)](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp)
- [FlashInfer FP4Tensor and NVFP4 Decode Attention](https://github.com/flashinfer-ai/flashinfer)
- [FlashInfer vec_dtypes.cuh (__nv_fp4_e2m1 vector operations)](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/vec_dtypes.cuh)
- [OCP Microscaling Formats Specification (MXFP4)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [ThunderKittens 2.0 NVFP4 Support](https://github.com/HazyResearch/ThunderKittens)
