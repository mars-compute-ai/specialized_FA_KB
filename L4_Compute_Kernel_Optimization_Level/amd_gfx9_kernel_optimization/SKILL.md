---
skill_name: AMD GFX9 Kernel Optimization Guide
description: Register usage, LDS bank conflict avoidance, global memory access patterns, and cross-lane data parallel primitives for optimizing GPU kernels on AMD GFX9 (CDNA2/CDNA3/CDNA4) architectures.
level: L4 - Compute Kernel Optimization Level
target_hardware: AMD MI250X (CDNA2, gfx90a), MI300X (CDNA3, gfx942), MI355X (CDNA4, gfx950)
relevance: When optimizing Flash Attention kernel performance on AMD GPUs — tuning register pressure, eliminating LDS bank conflicts, optimizing global memory access patterns, or choosing cross-lane communication primitives.
---

# AMD GFX9 Kernel Optimization Guide

## What It Is

A practical guide to low-level kernel optimization techniques on AMD GFX9-family GPUs (CDNA2/CDNA3/CDNA4). These techniques — register allocation, LDS bank conflict avoidance, global memory coalescing, and cross-lane data movement — are the building blocks for achieving high utilization in Flash Attention kernels. This topic covers the AMD-specific details that differ from NVIDIA: the 64-wide wavefront, the VGPR/AGPR split register file, 32-bank (CDNA3) or 64-bank (CDNA4) LDS, and the hierarchy of cross-lane instructions (DPP, permute, swizzle).

## Key Concepts

### Register Usage and Occupancy
- **VGPRs**: Up to 256 per thread. Used for general computation and MFMA input operands.
- **AGPRs**: Up to 256 per thread. Dedicated accumulator registers for MFMA C/D operands.
- **Shared file**: VGPRs + AGPRs share a pool of 512 registers × 64 threads per SIMD on CDNA2+.
- **Occupancy impact**: More registers per thread → fewer concurrent wavefronts per SIMD (max 10 waveslots).
- **Spilling**: Register spills go to scratch memory (global memory) — catastrophic for performance. Target ≤256 VGPRs + AGPRs combined.
- **Checking usage**: Inspect ISA dump for `.vgpr_count` and `.agpr_count` fields, or use `amdgpu-waves-per-eu` attribute.

### LDS (Workgroup Memory) Optimization
- **CDNA3**: 64 KB per CU, 32 banks × 4-byte entries. Bank conflicts when multiple threads hit the same bank in one cycle.
- **CDNA4**: 160 KB per CU, 64 banks × 4-byte entries. 256 bytes/clock read bandwidth.
- **Access phases**: `ds_read_b32` accesses one bank phase per cycle (32 threads → 32 banks); `ds_read_b64` requires 2 phases; `ds_read_b128` requires 4 phases.
- **Bank conflict avoidance**: Apply XOR-based swizzling to LDS addressing (same principle as NVIDIA shared memory swizzling).
- **CDNA4 Direct L1 Load**: New path loads directly from LDS to L1 cache, reducing register pressure for matrix operand staging.

### Global Memory Access Patterns
- **Optimal unit**: 16 bytes (128 bits) per thread using `global_load_dwordx4` / `global_store_dwordx4`.
- **Coalescing**: 64 threads × 16 bytes = 1024 bytes per wavefront. Subgroup-contiguous access enables 512-byte aligned transactions.
- **Clause formation**: Up to 4 adjacent load instructions form a clause = single fabric transaction. Maximizes L1 cache engagement.
- **Non-temporal hints**: Use `global_load_nt` / `global_store_nt` to bypass L1/L2 for streaming data.
- **Predicated loads**: Use buffer instructions for conditional loads to avoid divergent memory access.

### Cross-Lane Data Parallel Primitives
Performance ranking (fastest to slowest on MI300X):
| Instruction     | Latency   | Hardware    | Use Case                        |
|:----------------|:----------|:------------|:--------------------------------|
| `v_permlane`    | 4–8 cy    | VALU        | Arbitrary permutation within wavefront |
| DPP             | 4–12 cy   | VALU        | Row/bank shifts, broadcasts, reductions |
| `ds_swizzle`    | ~50 cy    | LDS         | Complex permutation patterns    |
| `ds_permute`    | ~50 cy    | LDS         | General indexed permutation     |
| `ds_bpermute`   | ~50 cy    | LDS         | Inverse permutation (broadcast-like) |

- For softmax reductions in attention: use DPP or `v_permlane` for wave-level max/sum reductions (4–12 cycles vs 50 cycles for LDS-based alternatives).

## When to Use

- Tuning occupancy by managing VGPR/AGPR allocation in attention kernels
- Eliminating LDS bank conflicts when loading Q/K/V tiles to workgroup memory
- Optimizing global memory loads for K/V streaming (coalescing, clause formation)
- Implementing fast wave-level reductions for online softmax (max, sum)
- Diagnosing performance issues via ISA dump analysis

## When NOT to Use

- Targeting NVIDIA GPUs (use CUDA-specific optimization techniques)
- Working at the algorithm level (see L1 topics for attention algorithm variants)
- Need architecture-level understanding (see amd_cdna_architecture_guide)

## Code / Pseudo-code

### Checking Register Usage in ISA Dump

```bash
# Compile and dump ISA
hipcc --offload-arch=gfx942 -save-temps kernel.cpp

# Inspect register counts in ISA
grep -E '(vgpr_count|agpr_count|sgpr_count)' kernel.s
# .vgpr_count: 128
# .agpr_count: 64
# .sgpr_count: 48
```

### Controlling Occupancy

```cpp
// Request minimum 4 waves per SIMD for latency hiding
__attribute__((amdgpu_waves_per_eu(4, 10)))
__global__ void flash_attention_kernel(...) {
    // Compiler will try to keep register usage low enough for 4 waves
}
```

### Optimal Global Load Pattern

```cpp
// Each thread loads 16 bytes (4 × float32) — forms a clause with adjacent loads
float4 q_tile = *reinterpret_cast<float4*>(&Q[offset]);
float4 k_tile = *reinterpret_cast<float4*>(&K[offset]);
// 64 threads × 16 bytes = 1024 bytes coalesced per wavefront
```

### DPP-Based Wave Reduction (for softmax max)

```cpp
// Wave-level max reduction using DPP (4-12 cycles per step)
float local_max = row_max;
local_max = __hip_ds_fmaxf(local_max, __shfl_xor(local_max, 1));
local_max = __hip_ds_fmaxf(local_max, __shfl_xor(local_max, 2));
local_max = __hip_ds_fmaxf(local_max, __shfl_xor(local_max, 4));
local_max = __hip_ds_fmaxf(local_max, __shfl_xor(local_max, 8));
local_max = __hip_ds_fmaxf(local_max, __shfl_xor(local_max, 16));
local_max = __hip_ds_fmaxf(local_max, __shfl_xor(local_max, 32));
// 6 steps for wave-64 (vs 5 for NVIDIA warp-32)
```

## Key Takeaways

- VGPRs and AGPRs share a 512-register pool on CDNA2+ — heavy MFMA usage (AGPRs) directly reduces VGPR availability and occupancy
- LDS bank conflicts follow the same 4-byte bank width as NVIDIA SMEM — XOR swizzling techniques transfer directly
- Global loads should be 16 bytes/thread with contiguous addressing to form 4-instruction clauses for maximum fabric efficiency
- For wave-level reductions (softmax max/sum), DPP and `v_permlane` are 5–12× faster than LDS-based `ds_permute`/`ds_swizzle`
- Wave-64 requires 6 reduction steps vs NVIDIA's 5 for warp-32, but the wider wavefront processes 2× more data per step

## References

- [AMDGPU Kernel Optimization Guide — Jakub Kuderski, nod-ai/amd-shark-ai (2025-08-14)](https://github.com/nod-ai/amd-shark-ai)
- [AMD CDNA3 ISA Manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [ROCm Optimization Guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides.html)
