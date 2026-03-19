---
skill_name: Flash Attention Optimization on AMD MI300X (CDNA3)
description: Architecture-specific optimization strategies for Flash Attention on AMD MI300X, covering memory hierarchy, MFMA instructions, and tuning parameters
level: L3 - Memory & Thread Cooperation Level
target_hardware: AMD MI300X (CDNA3), MI250X (CDNA2)
relevance: When optimizing Flash Attention kernels for AMD GPUs, porting NVIDIA-optimized kernels to AMD, or tuning attention performance on MI300X
---

# Flash Attention Optimization on AMD MI300X (CDNA3)

## What It Is
AMD MI300X requires fundamentally different optimization strategies than NVIDIA GPUs for Flash Attention due to its chiplet architecture (8 XCDs, 304 CUs), 64-thread wavefronts, MFMA matrix instructions, and lack of hardware TMA. However, its 5.3 TB/s HBM3 bandwidth (1.58x H100) provides an advantage for attention's memory-bound phases.

## Key Concepts
- **8-XCD chiplet design**: 304 CUs with 64 KB LDS each — different load balancing than monolithic NVIDIA dies
- **5.3 TB/s HBM3**: Higher memory bandwidth shifts the compute/memory-bound crossover point
- **MFMA per-wavefront**: Matrix instructions operate on 64-thread wavefronts (not warpgroups like WGMMA)
- **No TMA**: Data loading uses explicit wavefront-cooperative copies, not hardware DMA
- **num_stages=1**: Recommended for fused dual-GEMM kernels (FA), unlike NVIDIA's multi-stage approach
- **256 MB L2 cache**: Larger than H100's 50 MB — better K/V cache reuse for multi-query patterns

## AMD vs NVIDIA Differences
- **Wavefront 64 vs Warp 32**: Reductions cover more threads natively; different tile decomposition needed
- **LDS 64 KB vs SMEM 228 KB**: Smaller shared memory per CU — may need smaller tiles or more register usage
- **No TMA**: Producer-consumer warp specialization (FA3/FA4 pattern) needs adaptation to explicit loads
- **Higher HBM BW**: Memory-bound kernels perform relatively better; compute-bound optimizations matter less
- **More CUs (304 vs 132 SMs)**: More parallelism — different occupancy and scheduling strategies

## Tuning Parameters
```python
# Key tuning knobs for MI300X Flash Attention
config = {
    "block_M": 128,           # Q tile rows (MI300X optimal)
    "block_N": 32,            # K/V tile columns
    "threads": 512,           # Threads per block
    "num_split_q": "auto",    # Q-dimension parallelism across CUs
    "num_stages": 1,          # Pipeline stages (1 for fused dual-GEMM)
    "qk_coalesced_width": 16, # Memory coalescing for QK access
    "v_coalesced_width": 16,  # Memory coalescing for V access
    "use_swizzle": True,      # LDS bank conflict avoidance
}
```

## When to Use
- Deploying attention on AMD MI300X or MI250X GPUs
- Tuning Flash Attention kernel parameters for AMD architecture
- Porting NVIDIA Flash Attention optimizations to AMD (understanding what transfers and what doesn't)
- Evaluating AMD vs NVIDIA for attention-heavy workloads

## When NOT to Use
- Running on NVIDIA GPUs (different optimization strategies apply)
- Pre-CDNA2 AMD GPUs (lack MFMA support)

## Key Takeaways
- MI300X's 5.3 TB/s HBM3 is its biggest advantage for attention — exploit it, don't fight the memory-bound phases
- **num_stages=1** for Flash Attention on AMD (vs multi-stage on NVIDIA) — the pipeline model is different without TMA
- LDS is smaller (64 KB vs 228 KB SMEM) — use smaller K/V tiles or lean more on VGPR registers
- 64-thread wavefronts mean warp-level reductions are "free" over twice as many elements
- The FA3/FA4 warp-specialization pattern needs significant rethinking without TMA hardware
- TileLang achieves 2.69x over PyTorch on MI300X with just 80 lines — competitive with hand-tuned CK

## References
- [TileLang Flash Attention on MI300X](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html)
- [AMD MI300X Workload Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
- [AMD CDNA3 ISA Reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
