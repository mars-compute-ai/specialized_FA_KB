---
skill_name: MXFP Microscaling Formats for Attention (OCP MX Standard)
description: Understanding OCP Microscaling (MX) block-scaled floating-point formats -- MXFP8, MXFP6, MXFP4 -- their shared exponent mechanism, hardware support on Blackwell, and accuracy-throughput tradeoffs for attention kernels
level: L5 - Numerical Precision Level
target_hardware: NVIDIA Blackwell B100/B200 (native MX), NVIDIA Hopper H100 (software emulation), AMD MI300X (software block scaling)
relevance: When selecting between per-tensor FP8, per-block FP8, and MX-format quantization strategies for attention computation, especially on Blackwell where MX is hardware-native
---

# MXFP Microscaling Formats for Attention (OCP MX Standard)

## What It Is
The Open Compute Project (OCP) Microscaling (MX) specification defines a family of block-scaled floating-point formats where groups of 32 contiguous elements share a single E8M0 (8-bit exponent-only) scale factor. The MX family includes MXFP8 (E4M3 or E5M2 elements), MXFP6 (E2M3 or E3M2 elements), and MXFP4 (E2M1 elements). On NVIDIA Blackwell GPUs, MX formats are processed natively by tensor cores: the hardware reads element data and scale factors separately and applies dequantization during the MMA operation, eliminating software dequantization overhead. For attention kernels, MX formats offer a middle ground between per-tensor FP8 (coarse scaling, fast) and full FP16/BF16 (high precision, slow), with Blackwell's hardware support making MX the preferred low-precision strategy for production attention deployments.

## Key Concepts
- **Block size = 32 elements:** Every contiguous group of 32 values shares one E8M0 scale factor. This is fixed by the OCP specification and matches the hardware implementation.
- **E8M0 scale factor:** An 8-bit value with 8 exponent bits and no mantissa, representing powers of 2 from 2^(-127) to 2^(127). The scale is always a power of 2, enabling efficient multiplication via exponent addition.
- **MXFP8:** Each element is standard FP8 (E4M3 or E5M2), 8 bits. With the shared scale, the effective dynamic range per block is the element's range multiplied by the block scale. Storage: 8 bits/element + 1 byte per 32 elements for the scale (3.125% overhead).
- **MXFP6:** Each element is 6-bit floating-point (E2M3 or E3M2). E2M3 has higher precision (3 mantissa bits) but narrow range (4 exponents); E3M2 has wider range but lower precision. Storage: 6 bits/element + scale overhead.
- **MXFP4:** Each element is 4-bit floating-point (E2M1). Very low precision (1 mantissa bit) but 4x compression over FP16. Viable primarily for inference where weights have been calibrated. Storage: 4 bits/element + scale overhead.
- **Native hardware support:** Blackwell's UMMA instruction includes a `max_shift` field in the 32-bit descriptor that enables native MX block scaling during MMA. The tensor core reads the E8M0 scale factors alongside element data.
- **Per-tensor vs per-block scaling:** Per-tensor FP8 uses one scale for the entire tensor; MX uses one scale per 32 elements. MX accommodates within-tensor magnitude variations (e.g., outlier-heavy attention score rows) far better than per-tensor scaling.

## Precision Trade-offs

| Format | Bits/Element | Scale Granularity | Effective Range | Storage vs FP16 | Throughput vs FP16 |
|--------|-------------|-------------------|-----------------|-----------------|-------------------|
| FP16 | 16 | N/A | +/-65,504 | 1.0x | 1.0x |
| BF16 | 16 | N/A | +/-3.4e38 | 1.0x | 1.0x |
| FP8 (per-tensor) | 8 | 1 per tensor | Limited by single scale | 0.5x | 2.0x |
| MXFP8 | 8 + scale | 1 per 32 elements | E4M3 range x E8M0 scale | ~0.53x | 2.0x (native) |
| MXFP6 | 6 + scale | 1 per 32 elements | E2M3/E3M2 range x scale | ~0.41x | ~2.7x |
| MXFP4 | 4 + scale | 1 per 32 elements | E2M1 range x scale | ~0.28x | ~4.0x |

## Code / Configuration
```python
import torch
import numpy as np

# === MX Block Quantization (Software Implementation) ===
def quantize_mx_fp8(tensor: torch.Tensor, block_size: int = 32) -> tuple:
    """
    Quantize tensor to MXFP8 (E4M3 elements + E8M0 block scales).

    Args:
        tensor: Input tensor in FP32/FP16/BF16
        block_size: MX block size (always 32 per OCP spec)

    Returns:
        (quantized_elements, block_scales)
    """
    # Reshape to expose blocks of 32
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, block_size)
    num_blocks = flat.shape[0]

    # Compute per-block maximum absolute value
    block_amax = flat.abs().amax(dim=-1)  # [num_blocks]

    # E8M0 scale: round to nearest power of 2
    # E8M0 = 2^(exponent - 127), so exponent = floor(log2(amax)) + 127
    log2_amax = torch.log2(block_amax.clamp(min=1e-38))
    e8m0_exponent = torch.floor(log2_amax).to(torch.int32).clamp(0, 254)
    block_scales = torch.pow(2.0, e8m0_exponent.float() - 127)  # [num_blocks]

    # Scale elements and quantize to E4M3
    scaled = flat / block_scales.unsqueeze(-1)
    quantized = scaled.to(torch.float8_e4m3fn)

    return quantized.reshape(orig_shape), block_scales

def dequantize_mx_fp8(quantized: torch.Tensor, block_scales: torch.Tensor,
                       block_size: int = 32) -> torch.Tensor:
    """Dequantize MXFP8 back to higher precision."""
    orig_shape = quantized.shape
    flat = quantized.float().reshape(-1, block_size)
    return (flat * block_scales.unsqueeze(-1)).reshape(orig_shape)


# === MXFP8 Attention Pattern ===
def mxfp8_attention(Q, K, V, block_size=32):
    """
    Attention with MXFP8 quantized Q, K, V.
    On Blackwell, the dequantization happens inside the tensor core.
    """
    # Quantize inputs
    Q_mx, Q_scales = quantize_mx_fp8(Q, block_size)
    K_mx, K_scales = quantize_mx_fp8(K, block_size)
    V_mx, V_scales = quantize_mx_fp8(V, block_size)

    # On Blackwell: tcgen05.mma handles dequant natively via max_shift
    # On Hopper/software: explicit dequantization needed
    Q_deq = dequantize_mx_fp8(Q_mx, Q_scales, block_size)
    K_deq = dequantize_mx_fp8(K_mx, K_scales, block_size)

    # Compute attention scores (accumulator always FP32)
    scores = Q_deq @ K_deq.transpose(-2, -1) / (Q.shape[-1] ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)  # Softmax in FP32

    V_deq = dequantize_mx_fp8(V_mx, V_scales, block_size)
    output = attn_weights @ V_deq
    return output


# === Blackwell UMMA Descriptor with MX Support ===
# In the CUTLASS/FA4 framework, MX scaling is enabled via the max_shift field:
from flash_attn.cute.mma_sm100_desc import make_instr_desc, MaxShift, Major

# MXFP8 attention descriptor (hardware-native block scaling)
desc_mxfp8 = make_instr_desc(
    a_type=cutlass.FloatE4M3FN,  # Element type
    b_type=cutlass.FloatE4M3FN,
    c_type=cutlass.Float32,       # Accumulator
    M=128, N=128,
    a_major=Major.K, b_major=Major.K,
    max_shift=MaxShift.MaxShift8,  # Enable MX block scaling
)
# The max_shift field tells the tensor core to read and apply E8M0 scale factors
```

## When to Use
- On Blackwell GPUs where MXFP8 is hardware-native and provides 2x throughput over FP16 with minimal accuracy loss
- For LLM inference serving where memory bandwidth is the bottleneck and MXFP8 halves KV cache data movement
- When per-tensor FP8 scaling produces unacceptable accuracy degradation due to outlier values in attention scores
- Training large models (8B+ parameters) where MXFP8 has been validated to converge as well as BF16
- Long-context attention (32K+ tokens) where the O(N^2) score computation dominates and precision of individual scores matters less than the aggregate softmax
- When mixing precision: MXFP8 for Q*K^T and P*V GEMMs, FP32 for softmax accumulation

## When NOT to Use
- On GPUs without MX hardware support (Hopper H100, Ampere A100) unless the software dequantization overhead is acceptable
- For fine-tuning or tasks where the model has not been validated for MX-format convergence
- When the model has very few parameters (< 1B) and BF16 throughput is already sufficient
- For scientific computing requiring more than 3 mantissa bits of precision per element
- When attention head dimension is very small (d=32 or 64), where the block size of 32 may not align well with the tensor dimensions
- During early model development where numerical debugging requires full-precision introspection

## Source Code Examples

### Hopper Software MX Implementation (FlashAttention-3)

On Hopper (H100), MX block scaling is not hardware-native. FlashAttention-3 implements block quantization in software before WGMMA, with power-of-2 rounding to satisfy the E8M0 constraint:

```python
# FlashAttention-3 approach: block quantization before WGMMA
def prepare_fp8_blocks(tensor, block_size=32):
    flat = tensor.reshape(-1, block_size)
    amax = flat.abs().amax(dim=-1, keepdim=True)
    # Round to power of 2 (E8M0 constraint)
    scale = torch.pow(2, torch.floor(torch.log2(amax.clamp(min=1e-38))))
    quantized = (flat / scale).to(torch.float8_e4m3fn)
    # Scale factors applied during accumulation (not native)
    return quantized, scale

# The WGMMA instruction on Hopper does NOT have max_shift
# Dequantization must happen in software between GEMM tiles
```

### AMD Composable Kernel Block-Scale Implementation (CDNA)

AMD's Composable Kernel implements FP8 block scaling on CDNA architectures with configurable block sizes (not fixed at 32). The per-block descale is applied in software during GEMM accumulation:

```cpp
// From CK fmha_fwd kernel (gfx9)
// Block scale is applied during GEMM accumulation
struct FmhaFwdCommonBlockScaleKargs {
    index_t nhead_stride_q_descale;
    index_t nhead_stride_k_descale;
    index_t nhead_stride_v_descale;
    index_t block_scale_size_q;    // Configurable (not fixed at 32)
    index_t block_scale_size_kv;
};

// Apply descale during score accumulation
float k_descale = k_descale_ptr[kv_idx];
auto s_acc = s_acc_raw * k_descale;

// FP8 shift for softmax numerical stability
#if CK_TILE_USE_OCP_FP8
    validated_m -= 8.0f;   // OCP FP8 shift
#else
    validated_m -= 7.0f;   // FNUZ FP8 shift
#endif
```

Key difference from Blackwell: AMD's block size is configurable (not fixed at 32), and dequantization happens in software during the GEMM accumulation phase rather than natively in the tensor core.

## Key Takeaways
- MX formats fix the block size at 32 elements with E8M0 (power-of-2) scales -- this is an OCP industry standard, not an NVIDIA-specific choice
- The E8M0 scale factor has no mantissa bits, so it can only represent exact powers of 2. This constraint enables efficient hardware implementation via exponent addition.
- MXFP8 with block scaling achieves RMSE 9.1e-3 for attention (with incoherent processing), compared to 2.4e-2 for naive per-tensor FP8 -- a 2.6x accuracy improvement
- Blackwell's UMMA instruction natively applies MX scales via the `max_shift` descriptor field, meaning zero software overhead for dequantization during MMA
- MXFP8 training convergence matches BF16 for 8B parameter models (validated on Nemotron 8B), with validation perplexity tracking nearly identically
- The 3.125% storage overhead for scale factors (1 byte per 32 bytes of E4M3 data) is negligible compared to the 2x memory savings over FP16
- AMD's Composable Kernel also supports FP8 block scaling (with configurable block sizes) on CDNA architectures, using software-applied per-block descale factors

## References
- [OCP Microscaling Formats (MX) v1.0 Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [NVIDIA FP8 Introduction Blog (covers MXFP8)](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [FlashAttention-3: Block Quantization and Incoherent Processing (arXiv 2407.08608)](https://arxiv.org/abs/2407.08608)
- [CUTLASS SM100 MMA Descriptor (max_shift field)](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp)
- [NVIDIA Transformer Engine: MXFP8 Support](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [AMD Composable Kernel FP8 Block-Scale Attention](https://github.com/ROCm/composable_kernel)
