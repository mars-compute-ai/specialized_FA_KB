# AMD MFMA Matrix Core Programming — Detailed Reference

## Overview

Matrix Fused-Multiply-Add (MFMA) is AMD's hardware matrix multiplication primitive on CDNA GPUs. Each MFMA instruction computes D := A × B + C on a tile distributed across all 64 threads of a wavefront. This document provides the complete programming reference: available instruction variants, data distribution patterns, compiler intrinsic syntax, and worked examples across all supported precisions.

## 1. MFMA Instruction Catalog

### CDNA3 (MI300X/MI325X, gfx942)

| Instruction | Output Type | Input Type | Tile Shape | Elements/Thread (A, B, C) | Cycles |
|:------------|:------------|:-----------|:-----------|:--------------------------|:-------|
| `mfma_f64_16x16x4_f64` | FP64 | FP64 | 16×16×4 | (1, 1, 4) | — |
| `mfma_f32_32x32x2_f32` | FP32 | FP32 | 32×32×2 | (1, 1, 16) | — |
| `mfma_f32_16x16x4_f32` | FP32 | FP32 | 16×16×4 | (1, 1, 4) | — |
| `mfma_f32_32x32x8_f16` | FP32 | FP16 | 32×32×8 | (4, 4, 16) | — |
| `mfma_f32_16x16x16_f16` | FP32 | FP16 | 16×16×16 | (4, 4, 4) | — |
| `mfma_f32_32x32x8_bf16` | FP32 | BF16 | 32×32×8 | (4, 4, 16) | — |
| `mfma_f32_16x16x16_bf16` | FP32 | BF16 | 16×16×16 | (4, 4, 4) | — |
| `mfma_f32_32x32x16_fp8` | FP32 | FP8 | 32×32×16 | (8, 8, 16) | — |
| `mfma_f32_16x16x32_fp8` | FP32 | FP8 | 16×16×32 | (8, 8, 4) | — |

### CDNA4 Additions (MI355X, gfx950)

| Instruction | Output Type | Input Type | Tile Shape | Notes |
|:------------|:------------|:-----------|:-----------|:------|
| `mfma_f32_32x32x16_f16` | FP32 | FP16 | 32×32×16 | 2× K over CDNA3 |
| `mfma_f32_16x16x32_f16` | FP32 | FP16 | 16×16×32 | 2× K over CDNA3 |
| `mfma_f32_32x32x16_bf16` | FP32 | BF16 | 32×32×16 | 2× K over CDNA3 |
| `mfma_scale_f32_32x32x64_f8f6f4` | FP32 | FP8/FP6/FP4 | 32×32×64 | Block-scaled, 4× K |
| `mfma_scale_f32_16x16x128_f8f6f4` | FP32 | FP8/FP6/FP4 | 16×16×128 | Block-scaled, 8× K |

## 2. Compiler Intrinsic Interface

### Standard MFMA Intrinsic

```cpp
d_reg = __builtin_amdgcn_mfma_<ODType>_<M>x<N>x<K><InDType>(
    a_reg,    // Input A (packed into VGPR-width type)
    b_reg,    // Input B (packed into VGPR-width type)
    c_reg,    // Input C / accumulator (FP32 array)
    cbsz,     // Broadcast control for C (0 = no broadcast)
    abid,     // Accumulator block ID (0 = full tile)
    blgp      // B-lane group pattern (0 = no permutation)
);
```

For standard GEMM, set `cbsz = 0`, `abid = 0`, `blgp = 0`.

### Block-Scaled MFMA Intrinsic (CDNA4)

```cpp
d_reg = __builtin_amdgcn_mfma_scale_f32_<M>x<N>x<K>_f8f6f4(
    a_reg,     // Input A data
    b_reg,     // Input B data
    c_reg,     // Accumulator
    Atype,     // 0=FP8, 1=BF8, 2=FP6(E3M2), 3=FP6(E2M3), 4=FP4
    Btype,     // Same encoding as Atype
    OPSEL_A,   // Selects which 64-element block of A
    OPSEL_B,   // Selects which 64-element block of B
    scale_a,   // E8M0 scale factors for A blocks
    scale_b    // E8M0 scale factors for B blocks
);
```

## 3. Data Distribution Patterns

### 32×32×2 FP32 Example

Each of the 64 threads holds:
- **A**: 1 element (row = thread_id % 32, col = thread_id / 32)
- **B**: 1 element (row = thread_id / 32, col = thread_id % 32)
- **C/D**: 16 elements arranged as follows:

```
Thread ID → Row, Column mapping for 32×32 output:
  Threads  0-15:  rows 0-15,  columns determined by output index
  Threads 16-31:  rows 16-31, columns determined by output index
  Threads 32-47:  rows 0-15,  second column group
  Threads 48-63:  rows 16-31, second column group

Each thread's 16 output elements span non-contiguous positions
in the 32×32 tile, interleaved across column groups.
```

### 32×32×8 FP16 Example

Each of the 64 threads holds:
- **A**: 4 FP16 elements (half4 vector)
- **B**: 4 FP16 elements (half4 vector)
- **C/D**: 16 FP32 elements

### 32×32×16 FP8 Example

Each of the 64 threads holds:
- **A**: 8 FP8 elements packed into 64 bits (long)
- **B**: 8 FP8 elements packed into 64 bits (long)
- **C/D**: 16 FP32 elements

## 4. Theoretical Peak Performance Calculation

```
Peak TFLOPS = 2 × M × N × K × num_matrix_cores × (max_engine_clock / cycle_count) / 10^6
```

Where:
- `M × N × K` = tile dimensions of the MFMA instruction
- `num_matrix_cores` = total matrix core units across all CUs
- `max_engine_clock` = boost clock in MHz
- `cycle_count` = instruction latency in cycles
- Factor of 2 accounts for multiply + add

### Example: MI325X FP16 Peak

- MFMA shape: 32×32×8 = 16384 FLOPs per instruction
- 304 CUs, each with matrix cores
- Boost clock: ~2100 MHz
- Peak: ~1307.4 TFLOPS

## 5. MFMA in Flash Attention Context

### Mapping to Attention GEMMs

Flash Attention has two core GEMMs per tile:
1. **S = Q × Kᵀ**: Score computation (M×N×d where d = head_dim)
2. **O = P × V**: Output accumulation (M×d×N)

For head_dim = 128 with 32×32 MFMA tiles:
- GEMM1: Accumulate K = head_dim/K_mfma iterations (e.g., 128/8 = 16 MFMA calls for FP16)
- GEMM2: Accumulate K = seqlen_tile/K_mfma iterations

### Tile Shape Selection

| Head Dim | Recommended MFMA | Rationale |
|:---------|:-----------------|:----------|
| 64       | 32×32 or 16×16   | 16×16 gives higher occupancy |
| 128      | 32×32            | Matches typical block_M/block_N |
| 256      | 32×32            | Standard choice, multiple K iterations |

### Register Pressure Considerations

For 32×32 MFMA with FP32 accumulator:
- Each thread holds 16 FP32 accumulator values = 16 AGPRs
- Two GEMMs (S and O) active simultaneously: 32 AGPRs minimum
- Plus VGPRs for Q/K/V tile data and intermediate values
- Typical total: 128-192 VGPRs + 32-64 AGPRs

## 6. Block Scaling Deep Dive (CDNA4)

### Scale Factor Layout

For `mfma_scale_f32_32x32x64`:
- **A scale**: 32×2 matrix of E8M0 values (one scale per 32-element column block)
- **B scale**: 2×32 matrix of E8M0 values (one scale per 32-element row block)
- Each E8M0 value represents an exponent: actual_scale = 2^(value − 127)

### Thread Distribution for Scale Factors

```
For 32x32x64 block-scaled MFMA:
- A matrix: 32 rows × 64 cols, divided into 2 blocks of 32 columns
  → 32×2 = 64 scale values, distributed across 64 threads (1 per thread)
- B matrix: 64 rows × 32 cols, divided into 2 blocks of 32 rows
  → 2×32 = 64 scale values, distributed across 64 threads (1 per thread)
```

### Mixed Precision with Block Scaling

The type selector in `mfma_scale` supports mixing A and B precisions:

```cpp
// A in FP8, B in FP4 — asymmetric precision
c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    a_reg, b_reg, c_reg,
    /*Atype=*/0,   // FP8
    /*Btype=*/4,   // FP4
    0, 0,
    scale_a, scale_b
);
```

This enables asymmetric quantization strategies where Q/K matrices use different precision than V.

## References

- [Matrix Core Programming on AMD CDNA3 & CDNA4 — ROCm Blog (2025-09-30)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores-cdna/README.html)
- [AMD CDNA3 ISA Manual — Section 7: Matrix Arithmetic Instructions](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD CDNA4 ISA Manual — Section 7: Matrix Arithmetic Instructions](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi350-cdna4-instruction-set-architecture.pdf)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
- [OCP Microscaling (MX) Formats Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
