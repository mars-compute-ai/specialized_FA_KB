---
skill_name: AMD Composable Kernel (CK-Tile) for Flash Attention
description: AMD's high-performance kernel library and tile abstraction for implementing FlashAttention on MI250X/MI300X GPUs
level: L0 - Model/Invocation Level
target_hardware: AMD GPUs (CDNA2 MI250X, CDNA3 MI300X, CDNA4 MI350X)
relevance: When deploying Flash Attention on AMD GPUs, choosing between CK and Triton backends, or porting NVIDIA attention kernels to AMD
---

# AMD Composable Kernel (CK-Tile) for Flash Attention

## What It Is
Composable Kernel (CK) is AMD's GPU kernel library, equivalent to NVIDIA's CUTLASS. CK-Tile provides a tile-based abstraction for building fused kernels like FlashAttention on AMD hardware using MFMA matrix instructions. The ROCm flash-attention fork uses CK as its default backend, implementing FA2-equivalent kernels in ~100 lines of CK-Tile code.

## Key Concepts
- **CK-Tile abstraction**: Tile-based programming model mapping wavefronts to data, abstracting LDS/VGPR management
- **MFMA instructions**: AMD's matrix multiply-accumulate (like NVIDIA's MMA/WGMMA), operating on 64-thread wavefronts
- **LDS (Local Data Share)**: AMD's shared memory equivalent — K/V tiles buffered here for MFMA access
- **Two backends**: CK (default, C++/HIP, highest performance) or Triton (Python, easier to modify)
- **Wavefront = 64 threads** (vs NVIDIA warp = 32 threads) — affects tile shapes and scheduling

## AMD vs NVIDIA Differences
- **Wavefront size**: 64 threads (AMD) vs 32 threads (NVIDIA warp) — affects reduction patterns and tile decomposition
- **LDS vs Shared Memory**: Similar concept but different bank structure (32 banks, 4-byte width on AMD)
- **MFMA vs WGMMA**: MFMA operates per-wavefront; WGMMA operates per-warpgroup (128 threads). MFMA shapes differ (16×16×16, 32×32×8)
- **No TMA equivalent**: AMD lacks hardware TMA — data movement uses explicit wavefront-cooperative loads
- **Chiplet architecture**: MI300X has 8 XCDs with 304 CUs total — kernel launch and occupancy differ from monolithic NVIDIA dies
- **HBM3 bandwidth**: MI300X delivers 5.3 TB/s (vs H100's 3.35 TB/s) — more bandwidth-friendly for memory-bound attention

## When to Use
- Deploying models with attention on AMD MI250X/MI300X/MI350X GPUs
- Using ROCm flash-attention (defaults to CK backend automatically)
- Need maximum performance on AMD hardware (CK > Triton on AMD)
- Porting custom NVIDIA attention kernels to AMD

## When NOT to Use
- Running on NVIDIA GPUs (use CUTLASS/FlashAttention instead)
- Rapid prototyping where Triton backend is easier to iterate on
- Pre-CDNA2 AMD GPUs (require MFMA support)

## Code Snippets
```python
# Using ROCm flash-attention (CK backend is default)
from flash_attn import flash_attn_func

# Same API as NVIDIA flash-attention
output = flash_attn_func(q, k, v, causal=True)

# Force Triton backend if needed
# Set environment: FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
```

## Key Takeaways
- AMD's CK is the recommended backend for production FlashAttention on AMD GPUs
- The API is compatible with the NVIDIA flash-attention Python interface
- MI300X's 5.3 TB/s HBM3 bandwidth advantages attention's memory-bound phases
- Wavefront size (64 vs 32) requires different tile decomposition than NVIDIA
- CK-Tile allows implementing FA2 in ~100 lines — practical for custom attention variants

## References
- [CK-Tile FlashAttention Blog](https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html)
- [ROCm/composable_kernel GitHub](https://github.com/ROCm/composable_kernel)
- [ROCm/flash-attention GitHub](https://github.com/ROCm/flash-attention)
