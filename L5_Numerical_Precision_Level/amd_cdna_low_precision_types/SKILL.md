---
skill_name: AMD CDNA Low-Precision Floating Point Types
description: AMD-specific low-precision floating point formats (FP4/FP6/FP8/BF8) and OCP Microscaling (MXFP) block-scaled quantization on CDNA3/CDNA4, with performance scaling, format details, and implications for Flash Attention.
level: L5 - Numerical Precision Level
target_hardware: AMD MI300X (CDNA3), MI325X (CDNA3), MI355X (CDNA4)
relevance: When choosing numerical precision for Flash Attention on AMD GPUs, implementing FP8/FP4 quantized attention, or understanding the accuracy-throughput tradeoffs of AMD-specific formats.
---

# AMD CDNA Low-Precision Floating Point Types

## What It Is

AMD CDNA3 and CDNA4 GPUs support a range of low-precision floating point formats that trade numerical range/accuracy for throughput in matrix operations. CDNA3 introduced hardware FP8 support (E4M3/E5M2). CDNA4 extends this with FP6 (E3M2/E2M3), FP4 (E2M1), and block-scaled MXFP formats that apply per-32-element E8M0 exponent scaling factors — the OCP Microscaling standard. Each precision step roughly doubles matrix throughput, enabling faster Flash Attention kernels when the accuracy budget permits.

## Key Concepts

### Supported Formats by Generation

| Format          | Bits | Exponent | Mantissa | Range          | CDNA3 | CDNA4 |
|:----------------|:----:|:--------:|:--------:|:---------------|:-----:|:-----:|
| FP16 (E5M10)   | 16   | 5        | 10       | ±65504         | ✓     | ✓     |
| BF16 (E8M7)    | 16   | 8        | 7        | ±3.39×10³⁸    | ✓     | ✓     |
| FP8 (E4M3FN)   | 8    | 4        | 3        | ±448           | ✓     | ✓     |
| BF8 (E5M2)     | 8    | 5        | 2        | ±57344         | ✓     | ✓     |
| FP8 (E4M3FNUZ) | 8    | 4        | 3        | ±240           | ✓     | ✓     |
| BF8 (E5M2FNUZ) | 8    | 5        | 2        | ±57344         | ✓     | ✓     |
| FP6 (E3M2)     | 6    | 3        | 2        | ±28            |       | ✓     |
| FP6 (E2M3)     | 6    | 2        | 3        | ±7.5           |       | ✓     |
| FP4 (E2M1)     | 4    | 2        | 1        | ±6             |       | ✓     |
| E8M0 (scale)   | 8    | 8        | 0        | 2⁻¹²⁷ to 2¹²⁷ | —     | ✓     |

### FNUZ vs FN Variants
- **FN (Finite, No NaN sign)**: NaN represented as -0. Used by NVIDIA and OCP standard.
- **FNUZ (Finite, No NaN, Unsigned Zero)**: NaN is negative zero; no negative zero representation. Different exponent bias. AMD-specific variant supported alongside FN.
- **Practical difference**: OCP FP8 shift = 8.0f; FNUZ FP8 shift = 7.0f in dequantization code.

### Block Scaling (MXFP — CDNA4 Only)
- **Concept**: Instead of a single scale factor per tensor, apply one E8M0 exponent per 32-element block.
- **Scale factor**: E8M0 format — 8-bit unsigned exponent, actual scale = 2^(value − 127).
- **MXFP8**: 32 elements of E4M3/E5M2 + 1 byte E8M0 scale per block.
- **MXFP6**: 32 elements of E3M2/E2M3 + 1 byte E8M0 scale per block.
- **MXFP4**: 32 elements of E2M1 + 1 byte E8M0 scale per block. Packed 2 per byte.
- **Advantage**: Fine-grained dynamic range recovery — each 32-element block gets its own exponent, dramatically reducing quantization error vs per-tensor scaling.

### Peak Throughput (per GPU)

| Format    | CDNA3 (MI325X) | CDNA4 (MI355X) | Speedup vs FP32 |
|:----------|:--------------:|:--------------:|:----------------:|
| FP32      | 163.4 TF       | 157.3 TF       | 1×               |
| FP16/BF16 | 1307.4 TF      | 2.5 PF         | 8–16×            |
| FP8       | 2614.9 TF      | 5.0 PF         | 16–32×           |
| FP6       | —               | 10.0 PF        | 64×              |
| FP4       | —               | 10.0 PF        | 64×              |

## When to Use

- **FP16/BF16**: Default for training and high-accuracy inference attention. No quantization overhead.
- **FP8 (E4M3)**: Inference attention where 2× bandwidth savings are needed. Works well with per-tensor or per-block scaling. Available on both CDNA3 and CDNA4.
- **MXFP8**: When per-tensor FP8 quantization error is too high — block scaling recovers dynamic range at minimal cost (1 byte per 32 elements).
- **FP6/MXFP6**: CDNA4 inference where model accuracy tolerates 6-bit weights/activations. 2× throughput over FP8.
- **FP4/MXFP4**: CDNA4 extreme throughput scenarios (decode with large KV cache). Block scaling is essential — raw FP4 has only ±6 range.

## When NOT to Use

- FP4/FP6 on CDNA3 — no hardware support
- Training with FP4/FP6 — insufficient precision for gradient computation
- When accuracy requirements demand FP16+ (e.g., scientific computing attention)
- On NVIDIA GPUs — use nvidia_fp8_formats or mxfp_microscaling_formats topics instead

## Code / Pseudo-code

### FP8 MFMA (CDNA3/CDNA4)

```cpp
#include <hip/hip_fp8.h>

// Each thread holds 8 FP8 elements of A and B, 16 FP32 of C
__hip_fp8_storage_t a_fp8[8];
__hip_fp8_storage_t b_fp8[8];
float c[16] = {0};

// Pack 8×FP8 into a long (64 bits) for the intrinsic
long a_packed = *reinterpret_cast<long*>(a_fp8);
long b_packed = *reinterpret_cast<long*>(b_fp8);

c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a_packed, b_packed, c, 0, 0, 0);
```

### Block-Scaled MXFP8 (CDNA4)

```cpp
#include <hip/amd_detail/hip_ext_ocp.h>

// Block-scaled FP8: 32x32x64 tile with E8M0 exponent scaling
__amd_fp8_storage_t a_data[...];  // FP8 elements
__amd_fp8_storage_t b_data[...];
uint8_t scale_a[...];  // E8M0: actual_scale = 2^(value - 127)
uint8_t scale_b[...];

c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    a_reg, b_reg, c_reg,
    /*Atype=*/0, /*Btype=*/0,  // 0=FP8
    /*OPSEL_A=*/0, /*OPSEL_B=*/0,
    scale_a_reg, scale_b_reg
);
```

### FP4 Packed Format (CDNA4)

```cpp
#include <hip/amd_detail/hip_ext_ocp.h>

// FP4 elements are packed 2 per byte
__amd_fp4x2_storage_t packed;  // uint8_t storing 2 FP4 values

// Extract individual FP4 values
float val0 = __amd_extract_fp4(packed, 0);  // lower nibble
float val1 = __amd_extract_fp4(packed, 1);  // upper nibble

// Create packed FP4 pair
packed = __amd_create_fp4x2(float_val0, float_val1);
```

## Source Code Examples

### FP8 Attention with Per-Block Descaling (CDNA3)

From the CK FMHA kernel: FP8 K/V tiles are loaded and multiplied via MFMA, then a per-block descale factor is applied to the FP32 accumulator. The FP8 shift constant differs between OCP and FNUZ variants:

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

## Key Takeaways

- CDNA4 extends AMD's precision hierarchy to FP6 and FP4 with OCP-standard block scaling — each step roughly doubles throughput
- Block scaling (MXFP) uses one E8M0 exponent per 32 elements, providing fine-grained dynamic range recovery that enables FP4 attention with acceptable accuracy
- AMD supports both FN (OCP-compatible) and FNUZ (AMD-native) FP8 variants — the dequantization shift differs (8.0 vs 7.0)
- FP4 E2M1 has only ±6 range — block scaling is mandatory, not optional, for meaningful attention computation
- For Flash Attention on CDNA4: MXFP8 is the sweet spot for inference (5 PF throughput with good accuracy); MXFP4 is viable for decode-heavy workloads with large KV caches

## References

- [Matrix Core Programming on AMD CDNA3 & CDNA4 — ROCm Blog](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores-cdna/README.html)
- [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-white-paper.pdf)
- [OCP Microscaling (MX) Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [ROCm Low-Precision Floating Point Documentation](https://rocm.docs.amd.com/)
