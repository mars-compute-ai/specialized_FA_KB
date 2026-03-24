# AMD CDNA Low-Precision Floating Point Types — Detailed Reference

## Overview

This document provides comprehensive coverage of all low-precision floating point formats supported by AMD CDNA3 and CDNA4 GPUs, including binary representations, value ranges, the OCP Microscaling (MXFP) block-scaled formats, and practical programming patterns for Flash Attention kernels.

## 1. Format Specifications

### FP16 — IEEE Half Precision (E5M10)

```
Bit layout: [S|EEEEE|MMMMMMMMMM]
             1  5     10          = 16 bits

Sign:     1 bit
Exponent: 5 bits, bias = 15
Mantissa: 10 bits (implicit leading 1)
Range:    ±65504
Epsilon:  9.77 × 10⁻⁴
```

Standard format for attention training. Full IEEE 754 compliance.

### BF16 — Brain Float (E8M7)

```
Bit layout: [S|EEEEEEEE|MMMMMMM]
             1  8        7        = 16 bits

Sign:     1 bit
Exponent: 8 bits, bias = 127
Mantissa: 7 bits (implicit leading 1)
Range:    ±3.39 × 10³⁸ (same as FP32)
Epsilon:  7.81 × 10⁻³
```

Same dynamic range as FP32 with reduced precision. Preferred for training on AMD (same as NVIDIA).

### FP8 E4M3FN (OCP Standard)

```
Bit layout: [S|EEEE|MMM]
             1  4    3   = 8 bits

Sign:     1 bit
Exponent: 4 bits, bias = 7
Mantissa: 3 bits (implicit leading 1)
Range:    ±448
Special:  NaN = 0b_1_1111_111 (S=1, all ones)
          No ±Inf representation
```

Primary format for inference. Higher precision than E5M2 but narrower range. **Recommended for attention K/V values** where the distribution is typically narrow.

### FP8 E5M2 (OCP Standard)

```
Bit layout: [S|EEEEE|MM]
             1  5     2   = 8 bits

Sign:     1 bit
Exponent: 5 bits, bias = 15
Mantissa: 2 bits (implicit leading 1)
Range:    ±57344
Special:  Standard IEEE NaN/Inf
```

Wider range but lower precision. **Recommended for gradients** during FP8 training.

### FP8 E4M3FNUZ (AMD-specific variant)

```
Same bit layout as E4M3FN but:
  Exponent bias = 8 (vs 7 for FN)
  NaN = 0b_1_0000_000 (negative zero)
  No negative zero representation
  Range: ±240 (vs ±448 for FN)
```

**Dequantization difference**: OCP FP8 shift = 8.0f, FNUZ shift = 7.0f. This affects per-block descale factor computation in CK FMHA kernels.

### FP6 E3M2 (CDNA4 only)

```
Bit layout: [S|EEE|MM]
             1  3   2   = 6 bits

Sign:     1 bit
Exponent: 3 bits, bias = 3
Mantissa: 2 bits
Range:    ±28
```

### FP6 E2M3 (CDNA4 only)

```
Bit layout: [S|EE|MMM]
             1  2  3   = 6 bits

Sign:     1 bit
Exponent: 2 bits, bias = 1
Mantissa: 3 bits
Range:    ±7.5
```

Higher precision than E3M2 but much narrower range. Use with block scaling.

### FP4 E2M1 (CDNA4 only)

```
Bit layout: [S|EE|M]
             1  2  1  = 4 bits

Sign:     1 bit
Exponent: 2 bits, bias = 1
Mantissa: 1 bit
Range:    ±6
Values:   {0, 0.5, 1, 1.5, 2, 3, 4, 6} (and negatives)
```

Only 15 unique values (excluding ±0). **Block scaling is mandatory** — raw FP4 cannot represent meaningful attention scores.

### E8M0 Scale Factor (CDNA4 only)

```
Bit layout: [EEEEEEEE]
             8           = 8 bits

Exponent only, no sign or mantissa
Actual scale = 2^(value - 127)
Range: 2⁻¹²⁷ to 2¹²⁷
```

Used exclusively as the per-block exponent in MXFP formats. One E8M0 value per 32 data elements.

## 2. OCP Microscaling (MXFP) Block Scaling

### Concept

Instead of one scale factor per tensor (per-tensor quantization) or per-channel, MXFP applies one scale factor per **32-element block**. This provides fine-grained dynamic range recovery with minimal overhead (1 byte per 32 elements = 3.1% overhead for FP8, 6.25% for FP4).

### Block Structure

```
MXFP8 block (32 elements + 1 scale):
┌─────────────────────────────────────┬──────────┐
│  32 × FP8 data elements (32 bytes) │ 1× E8M0  │
│  e₀ e₁ e₂ ... e₃₁                 │ scale    │
└─────────────────────────────────────┴──────────┘
Actual value of element i = fp8_decode(eᵢ) × 2^(scale - 127)

MXFP4 block (32 elements + 1 scale):
┌─────────────────────────────────────┬──────────┐
│  32 × FP4 data elements (16 bytes) │ 1× E8M0  │
│  (packed 2 per byte)               │ scale    │
└─────────────────────────────────────┴──────────┘
```

### Scale Distribution in MFMA

For a 32×32×64 block-scaled MFMA:
- **A matrix** (32 rows × 64 cols): divided into 2 column blocks of 32 → **32×2 = 64 scale values**
- **B matrix** (64 rows × 32 cols): divided into 2 row blocks of 32 → **2×32 = 64 scale values**
- Each of the 64 threads holds 1 scale value for A and 1 for B

### Quantization Workflow for Attention

```
Training/Calibration:
1. Compute per-block statistics: max_abs per 32 elements
2. Derive E8M0 scale: scale = ceil(log2(max_abs)) + bias_offset
3. Quantize: fp8_value = round(float_value / 2^(scale - 127))

Inference (inside MFMA):
1. Hardware reads FP8/FP4 data and E8M0 scales
2. MFMA automatically applies: result += decode(A[i]) × 2^(scale_a - 127)
                                        × decode(B[j]) × 2^(scale_b - 127)
3. Accumulation in FP32 — no intermediate rounding
```

## 3. Performance Characteristics

### Throughput by Format (per GPU)

| Format | CDNA3 (MI325X) | CDNA4 (MI355X) | Memory BW Savings |
|:-------|:---------------|:---------------|:------------------|
| FP32 | 163.4 TF | 157.3 TF | 1× (baseline) |
| FP16/BF16 | 1307.4 TF | 2.5 PF | 2× |
| FP8 | 2614.9 TF | 5.0 PF | 4× |
| FP6 | — | 10.0 PF | 5.3× |
| FP4 | — | 10.0 PF | 8× |

**Note**: FP6 and FP4 achieve the same 10 PF peak on MI355X because both use the same 32×32×64 MFMA tile — the hardware processes both at the same rate.

### Arithmetic Intensity Shift

As precision decreases, attention kernels shift from memory-bound to compute-bound:
- **BF16 attention**: Often memory-bound on MI300X (5.3 TB/s is generous)
- **FP8 attention**: Balanced on MI300X — compute and memory roughly equal
- **FP4 attention on MI355X**: Likely compute-bound — 10 PF compute vs 8 TB/s memory

### Accuracy Considerations for Attention

| Format | Typical Attention Error | Notes |
|:-------|:----------------------|:------|
| FP16 | Baseline | Full precision attention |
| BF16 | ~1e-3 relative error | Acceptable for most models |
| FP8 (per-tensor) | ~2.4e-2 RMSE | May need incoherent processing |
| MXFP8 (per-block) | ~1e-2 RMSE | Block scaling recovers accuracy |
| MXFP4 (per-block) | ~5e-2 RMSE | Only for tolerant workloads |

## 4. Programming Examples

### FP8 Attention with Per-Block Descaling (CDNA3)

```cpp
// In CK FMHA kernel: FP8 K/V with per-block descale
// OCP FP8: shift = 8.0f
// FNUZ FP8: shift = 7.0f
constexpr float fp8_shift = 8.0f;  // OCP standard

// Per-block descale during GEMM accumulation
for (int k = 0; k < K_tiles; k++) {
    // Load FP8 K tile
    fp8_t k_tile[...] = load_fp8_tile(K_ptr, k);
    float descale_k = descale_K[block_k_idx];

    // MFMA: S_acc += Q_tile * K_tile (in FP8)
    S_acc = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
        q_packed, k_packed, S_acc, 0, 0, 0);

    // Apply per-block descale (multiply into accumulator)
    for (int i = 0; i < 16; i++)
        S_acc[i] *= descale_k;
}
```

### FP4 Packing/Unpacking (CDNA4)

```cpp
#include <hip/amd_detail/hip_ext_ocp.h>

// Pack two float values into one FP4x2 byte
__amd_fp4x2_storage_t pack_fp4(float a, float b) {
    return __amd_create_fp4x2(a, b);
}

// Unpack FP4x2 byte into two float values
void unpack_fp4(__amd_fp4x2_storage_t packed, float& a, float& b) {
    a = __amd_extract_fp4(packed, 0);  // lower nibble
    b = __amd_extract_fp4(packed, 1);  // upper nibble
}

// Block-scaled FP4 MFMA
void mfma_fp4_block_scaled(
    __amd_fp4x2_storage_t* a_data,
    __amd_fp4x2_storage_t* b_data,
    uint8_t* scale_a,  // E8M0 scales
    uint8_t* scale_b,
    float* c_acc
) {
    c_acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_packed, b_packed, c_acc,
        /*Atype=*/4,     // FP4
        /*Btype=*/4,     // FP4
        /*OPSEL_A=*/0,
        /*OPSEL_B=*/0,
        scale_a_reg, scale_b_reg
    );
}
```

## 5. Format Selection Guide for Flash Attention

### Decision Tree

```
Is this training?
├── Yes → BF16 (safest) or FP8 E5M2 for gradients
└── No (inference)
    ├── Accuracy-critical (medical, scientific)?
    │   └── FP16 or BF16
    ├── Standard inference?
    │   ├── CDNA3 → FP8 E4M3 with per-block descale
    │   └── CDNA4 → MXFP8 (block-scaled, best accuracy/throughput)
    └── Throughput-critical (large KV cache decode)?
        ├── CDNA4 with moderate accuracy needs → MXFP6
        └── CDNA4 with relaxed accuracy → MXFP4
```

### Asymmetric Precision Strategy

CDNA4's `mfma_scale` supports mixing A and B types:
- **Q/K in MXFP8, V in MXFP4**: Attention scores need more precision than value accumulation
- **Q in FP16, K/V in MXFP8**: Preserve query precision, compress KV cache

## References

- [Matrix Core Programming on AMD CDNA3 & CDNA4 — ROCm Blog (2025-09-30)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores-cdna/README.html)
- [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-white-paper.pdf)
- [OCP Microscaling (MX) Formats Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [ROCm Low-Precision Floating Point Types](https://rocm.docs.amd.com/)
- [AMD CDNA3 ISA Manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
