---
skill_name: AMD CDNA3/CDNA4 Architecture Guide
description: Chiplet architecture, compute unit microarchitecture, cache hierarchy, and HBM memory system details for AMD CDNA3 (MI300X/MI325X) and CDNA4 (MI355X) GPUs relevant to Flash Attention kernel optimization.
level: L3 - Memory and Thread Cooperation Level
target_hardware: AMD MI300X (CDNA3), MI325X (CDNA3), MI355X (CDNA4)
relevance: When tuning Flash Attention tile sizes, pipeline stages, memory access patterns, or LDS usage for AMD CDNA GPUs and needing to understand the hardware constraints.
---

# AMD CDNA3/CDNA4 Architecture Guide

## What It Is

AMD's CDNA3 and CDNA4 are datacenter GPU architectures optimized for AI/ML workloads. Unlike NVIDIA's monolithic die designs (until Blackwell), AMD uses a chiplet-based architecture where multiple Accelerator Complex Dies (XCDs) are 3D-stacked on I/O Dies (IODs). Understanding the chiplet topology, compute unit (CU) microarchitecture, cache hierarchy, and memory bandwidth is essential for tuning Flash Attention kernels — particularly for choosing tile sizes, pipeline depth, LDS allocation, and workgroup scheduling strategies.

## Key Concepts

### Chiplet Architecture
- **CDNA3 (MI300X/MI325X)**: 8 XCDs on 4 IODs (TSMC 5nm XCDs, 6nm IODs). 304 active CUs total (38 per XCD).
- **CDNA4 (MI355X)**: 8 XCDs on 2 IODs (TSMC N3P XCDs). 256 active CUs total (32 per XCD).
- Each XCD has its own L2 cache (4 MB). XCDs on the same IOD share Infinity Cache and HBM controllers.
- Inter-XCD communication goes through Infinity Fabric — higher latency than intra-XCD.

### Compute Unit (CU) Microarchitecture
- **SIMD units**: 4 × 16-lane SIMDs per CU. A wavefront (64 threads) takes 4 cycles across one SIMD.
- **Matrix cores**: Dedicated MFMA execution units per CU.
  - CDNA3: 2048 FP16 FLOPS/clock/CU, 4096 FP8/INT8 FLOPS/clock/CU
  - CDNA4: 4096 FP16 FLOPS/clock/CU, 8192 FP8 FLOPS/clock/CU, 16384 FP4/FP6 FLOPS/clock/CU
- **Registers**: 512 registers × 64 threads shared between VGPRs and AGPRs per SIMD.
- **Waveslots**: Up to 10 wavefronts per SIMD for latency hiding.

### Local Data Share (LDS)
- **CDNA3**: 64 KB per CU, 32 banks × 4 bytes, same as NVIDIA shared memory role
- **CDNA4**: 160 KB per CU (2.5× increase), 64 banks, 256 bytes/clock read bandwidth (2×)
- CDNA4 adds **Direct L1 Load** from LDS — reduces register usage and latency for matrix operand staging

### Cache Hierarchy
- **L1 Data Cache**: 32 KB per CU (128-byte cache lines on CDNA3+)
- **L2 Cache**: 4 MB per XCD (32 MB total across 8 XCDs). ~34.4 TB/s aggregate read bandwidth.
- **Infinity Cache (LLC)**: 256 MB total across IODs. Memory-side cache (does not participate in coherency). ~17.2 TB/s peak bandwidth.
- **HBM**:
  - MI300X: 192 GB HBM3 @ 5.3 TB/s
  - MI325X: 256 GB HBM3E @ 6.0 TB/s
  - MI355X: 288 GB HBM3E @ 8.0 TB/s

### Key CDNA3 → CDNA4 Changes for Attention
- **2× matrix core throughput** at every precision level
- **2.5× LDS capacity** (64 KB → 160 KB) — enables larger tiles or more pipeline stages
- **2× transcendental rate** — directly benefits softmax exp/log operations
- **FP6 and FP4 support** with block scaling (MXFP standard)
- **8.0 TB/s HBM bandwidth** (vs 5.3–6.0 TB/s) — shifts the compute/memory bound crossover

## When to Use

- Choosing tile sizes (block_M, block_N) constrained by LDS capacity (64 KB vs 160 KB)
- Deciding pipeline depth — CDNA3's smaller LDS favors num_stages=1 for fused dual-GEMM
- Understanding XCD-aware workgroup scheduling to minimize inter-chiplet traffic
- Comparing AMD vs NVIDIA hardware constraints (no TMA, Wave-64 vs Warp-32, larger HBM BW)
- Planning memory access patterns that exploit the L2 → Infinity Cache → HBM hierarchy

## When NOT to Use

- Targeting NVIDIA GPUs (see gpu_memory_hierarchy and thread_block_clusters topics)
- Need software-level optimization techniques (see amd_gfx9_kernel_optimization)
- Need MFMA instruction-level programming details (see amd_mfma_matrix_core_programming)

## Source Code Examples

### Querying AMD GPU Architecture Properties

```cpp
#include <hip/hip_runtime.h>

hipDeviceProp_t props;
hipGetDeviceProperties(&props, 0);

printf("GPU: %s\n", props.name);                    // e.g., "AMD Instinct MI300X"
printf("CUs: %d\n", props.multiProcessorCount);      // 304 for MI300X
printf("Max threads/CU: %d\n", props.maxThreadsPerMultiProcessor);
printf("Shared mem/CU: %zu KB\n", props.sharedMemPerBlock / 1024);  // 64 KB (CDNA3)
printf("Total HBM: %.1f GB\n", props.totalGlobalMem / 1e9);        // 192 GB
printf("Memory bus width: %d bits\n", props.memoryBusWidth);
printf("GCN arch: gfx%d\n", props.gcnArch);          // 942 for MI300X
```

### XCD-Aware Workgroup Scheduling

```cpp
// Transform workgroup ID to improve L2 cache locality within each XCD
// Round-robin across XCDs ensures related workgroups share the same 4MB L2
__device__ inline int chiplet_aware_wgid(int wgid, int num_wgs, int num_xcds = 8) {
    int xcd_id = wgid % num_xcds;
    int local_wg = wgid / num_xcds;
    return xcd_id * (num_wgs / num_xcds) + local_wg;
}
```

## Key Takeaways

- AMD's chiplet design means workgroup placement across XCDs matters — L2 is per-XCD (4 MB), and cross-XCD traffic goes through Infinity Fabric
- CDNA3 has 64 KB LDS (same as NVIDIA Hopper SMEM) but no TMA — all data movement is explicit global loads to LDS
- CDNA4's 160 KB LDS is a game-changer for attention: enables larger tiles or multi-stage pipelines that were impossible on CDNA3
- HBM bandwidth advantage (5.3–8.0 TB/s vs NVIDIA's 3.35 TB/s on H100) means AMD attention kernels are more likely compute-bound at equivalent precision
- The 2× transcendental rate improvement in CDNA4 directly addresses softmax as a bottleneck (similar motivation to FA4's polynomial exp on Blackwell)

## References

- [AMD CDNA3 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-white-paper.pdf)
- [AMD Instinct MI300X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [AMD Instinct MI355X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)
