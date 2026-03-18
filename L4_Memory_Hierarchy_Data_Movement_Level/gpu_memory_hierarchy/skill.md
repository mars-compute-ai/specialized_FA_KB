---
skill_name: GPU Memory Hierarchy Understanding
description: Maps GPU memory tiers (registers, L1/shared, L2, HBM) with latencies and bandwidths to guide data placement decisions
level: L4 - Memory-Hierarchy/Data-Movement Level
target_hardware: NVIDIA Ampere A100, Hopper H100, and modern NVIDIA GPUs generally
relevance: When an AI agent needs to decide where tensors should reside, estimate data movement costs, or understand why a kernel is memory-bound
---

# GPU Memory Hierarchy Understanding

## What It Is
Modern GPUs have a multi-tiered memory system: registers (~1 cycle), L1/shared memory (~30 cycles), L2 cache (~150 cycles), and HBM (~400-800 cycles). Each tier trades capacity for speed. Understanding these tiers is the foundation for all memory-hierarchy optimizations — it explains why on-chip tiling, data reuse, and software pipelining are critical for achieving peak performance.

## Key Concepts
- **Registers** (~256 KB/SM, ~1 cycle): Fastest storage, private per thread; excessive use causes spilling and reduces occupancy
- **L1/Shared Memory** (~128-192 KB/SM, ~30 cycles): Programmer-controllable on-chip memory; enables cooperative data sharing within a thread block
- **L2 Cache** (~40-50 MB total, ~150 cycles): Shared across all SMs; acts as a global buffer to reduce redundant HBM fetches
- **HBM** (40-80 GB, ~400-800 cycles): High-capacity off-chip memory; bandwidth is 2-3.35 TB/s but latency is 200x+ worse than registers
- **Bandwidth hierarchy**: Registers (~8 TB/s) > L1 (~15-20 TB/s) > L2 (~3 TB/s) > HBM (~2-3.35 TB/s)
- **Energy hierarchy**: Register access is ~200x cheaper in energy than HBM access
- **Occupancy trade-off**: More registers per thread = fewer concurrent warps = potentially lower latency hiding

## Memory Layout / Data Flow
```
CPU RAM (TB, ~μs latency)
    |  PCIe / NVLink (~50-100 GB/s)
    v
HBM (40-80 GB, ~400-800 cycles, ~2-3.35 TB/s)
    |  Memory controllers
    v
L2 Cache (40-50 MB shared, ~150 cycles, ~3 TB/s)
    |  Crossbar / NoC
    v
L1 / Shared Memory (128-192 KB per SM, ~30 cycles, ~15-20 TB/s)
    |  Register file interface
    v
Registers (256 KB per SM, ~1 cycle, ~8 TB/s)
    |
    v
Tensor Cores / CUDA Cores (computation)
```

## Performance Impact
- L1 hit vs HBM miss: ~15x latency difference (30 cycles vs 500 cycles)
- Algorithms that keep working sets in shared memory can achieve 5-10x bandwidth improvement over HBM-bound versions
- FlashAttention achieves ~4x speedup primarily by keeping attention tiles in SRAM instead of materializing N×N matrices in HBM
- Register spilling can degrade performance by 2-3x due to implicit L1 traffic

## When to Use
- When designing any custom CUDA kernel for attention, GEMM, or reduction operations
- When profiling reveals a kernel is memory-bandwidth-bound (low arithmetic intensity)
- When deciding tile sizes for blocked algorithms (tiles must fit in shared memory)
- When analyzing whether to trade compute (recomputation) for memory (storing intermediates)

## When NOT to Use
- When working at a high-level framework level where memory management is abstracted (e.g., PyTorch eager mode)
- When the operation is purely compute-bound and already saturates tensor cores
- When the working set is small enough to fit entirely in L2 cache without explicit tiling

## Key Takeaways
- The GPU memory hierarchy has 4 main levels with ~500x latency spread from registers to HBM
- Most AI kernel optimizations reduce to: "keep data in the fastest memory tier possible for as long as possible"
- Tiling is the primary technique — break work into pieces that fit in shared memory, reuse data before evicting
- The bandwidth gap between on-chip (L1: 15-20 TB/s) and off-chip (HBM: 2-3 TB/s) memory is the root cause of memory-boundedness
- Understanding this hierarchy is prerequisite knowledge for FlashAttention, TMA, and all L4-level optimizations

## References
- [The GPU Memory Hierarchy: L2, L1, and Registers](https://www.nikhilkuniyil.com/blog/gpu-memory-hierarchy)
- [GPU Cache Hierarchy: Understanding L1, L2, and VRAM](https://charlesgrassi.dev/blog/gpu-cache-hierarchy/)
- [GPU Memory Hierarchy - How AI Training Actually Works](https://medium.com/@indiai/gpu-memory-hierarchy-how-ai-training-actually-works-24f00cc13050)
- [Nvidia's H100: Funny L2, and Tons of Bandwidth](https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/)
