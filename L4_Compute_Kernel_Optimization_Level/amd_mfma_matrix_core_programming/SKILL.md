---
skill_name: AMD MFMA Matrix Core Programming
description: Programming AMD Matrix Fused-Multiply-Add (MFMA) instructions on CDNA3/CDNA4 using compiler intrinsics, with data layout patterns and worked examples for FP32, FP16, FP8, FP4, and block-scaled MXFP formats.
level: L4 - Compute Kernel Optimization Level
target_hardware: AMD MI300X (CDNA3, gfx942), MI325X (CDNA3, gfx942), MI355X (CDNA4, gfx950)
relevance: When writing or optimizing MFMA-based GEMM tiles inside Flash Attention kernels on AMD GPUs, choosing instruction shapes, mapping data to threads, or using low-precision block-scaled formats.
---

# AMD MFMA Matrix Core Programming

## What It Is

Matrix Fused-Multiply-Add (MFMA) instructions are AMD's equivalent of NVIDIA's Tensor Core / WGMMA operations. Each MFMA instruction computes `D := A × B + C` on a tile of data distributed across all 64 threads of a wavefront. Unlike NVIDIA where warpgroups (128 threads) drive WGMMA, AMD MFMA operates at wavefront granularity (64 threads), and each thread holds a portion of the A, B, C, and D operands in VGPRs/AGPRs. Understanding MFMA data distribution, available tile shapes, and compiler intrinsic syntax is essential for writing high-performance attention kernels on CDNA hardware.

## Key Concepts

- **Wavefront-level operation**: All 64 threads collectively execute one MFMA instruction. There is no sub-wavefront variant.
- **Accumulator registers (AGPRs)**: CDNA2+ use dedicated Accumulator GPRs for the C/D matrix, sharing a 512-register file with VGPRs per thread.
- **Tile shapes**: Each MFMA variant specifies an `MxNxK` shape (e.g., 32×32×8 for FP16, 32×32×16 for FP8). Larger K dimensions process more elements per instruction.
- **Compiler intrinsics**: `__builtin_amdgcn_mfma_<OutType>_<M>x<N>x<K><InType>(a, b, c, cbsz, abid, blgp)` — the primary interface for emitting MFMA in HIP kernels.
- **Block scaling (CDNA4)**: New `mfma_scale` intrinsics apply per-32-element E8M0 exponent scaling factors, enabling MXFP8/MXFP6/MXFP4 formats with fine-grained quantization.
- **Peak throughput scaling**: Moving from FP32 → FP16 → FP8 → FP4 roughly doubles throughput at each step. CDNA4 MI355X peaks at 10 PFLOPS for FP4/FP6.

## When to Use

- Writing custom MFMA-based GEMM tiles for attention S=Q×Kᵀ and O=P×V on AMD GPUs
- Choosing between MFMA tile shapes (16×16 vs 32×32) for different head dimensions
- Implementing FP8 or FP4 quantized attention kernels on CDNA3/CDNA4
- Understanding thread-to-element data distribution for debugging or manual assembly
- Porting NVIDIA WGMMA-based kernels to AMD (mapping warpgroup operations to wavefront MFMA)

## When NOT to Use

- Targeting NVIDIA GPUs (use WGMMA on Hopper, UMMA on Blackwell)
- Using high-level frameworks that abstract MFMA (e.g., PyTorch SDPA with CK backend) — unless debugging performance
- Non-matrix workloads (use VALU instructions instead)

## Code / Pseudo-code

### Compiler Intrinsic Syntax

```cpp
// General form:
d_reg = __builtin_amdgcn_mfma_<ODType>_<M>x<N>x<K><InDType>(a_reg, b_reg, c_reg, cbsz, abid, blgp);

// FP16 input, FP32 accumulator, 32x32x8 tile:
float16_t a[4];  // each thread holds 4 FP16 elements of A
float16_t b[4];  // each thread holds 4 FP16 elements of B
float       c[16]; // each thread holds 16 FP32 elements of C/D
c = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);

// FP8 input, FP32 accumulator, 32x32x16 tile:
long a_fp8;  // 8 FP8 elements packed into 64 bits
long b_fp8;
float c[16];
c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a_fp8, b_fp8, c, 0, 0, 0);
```

### Block-Scaled MFMA (CDNA4 only)

```cpp
// MXFP8 with E8M0 scale factors, 32x32x64 tile:
// scale_a: 32x2 matrix of E8M0 exponents for A
// scale_b: 2x32 matrix of E8M0 exponents for B
c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    a_reg, b_reg, c_reg,
    /*Atype=*/0,     // 0=FP8, 1=BF8, 2=FP6, 3=BF6, 4=FP4
    /*Btype=*/0,
    /*OPSEL_A=*/0,   // which 64-element block of A
    /*OPSEL_B=*/0,   // which 64-element block of B
    scale_a, scale_b
);
// Actual scale applied: 2^(scale_value - 127)
```

### Data Distribution Example (32×32×2 FP32)

```
Thread layout for 32x32 output tile (64 threads):
- Threads 0-15:  rows 0-15  (1 element per thread per row)
- Threads 16-31: rows 16-31
- Threads 32-47: rows 0-15  (second column group)
- Threads 48-63: rows 16-31 (second column group)
Each thread holds 16 elements of the 32x32 output.
```

## Key Takeaways

- MFMA operates at wavefront (64-thread) granularity — the fundamental difference from NVIDIA's warpgroup WGMMA (128 threads)
- Each precision step (FP32 → FP16 → FP8 → FP4) roughly doubles peak throughput; CDNA4 adds FP6 and block-scaled variants
- The `cbsz`, `abid`, `blgp` parameters control broadcast modes for accumulator sub-tiles — set all to 0 for standard GEMM
- CDNA4 block scaling uses E8M0 exponent format with actual scale = 2^(value − 127), enabling per-32-element quantization granularity
- For Flash Attention: the 32×32 tile shapes are the workhorse; 16×16 variants are useful for small head dimensions or higher occupancy

## References

- [Matrix Core Programming on AMD CDNA3 & CDNA4 — ROCm Blog (2025-09-30)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores-cdna/README.html)
- [AMD CDNA3 ISA Manual — Matrix Arithmetic Instructions (Section 7)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD CDNA4 ISA Manual — Matrix Arithmetic Instructions](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi350-cdna4-instruction-set-architecture.pdf)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
