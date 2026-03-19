---
skill_name: Flash Attention Tile Size Tuning and Optimization
description: Practical guide to tuning tile sizes, math precision, and K-loop splitting for peak Flash Attention performance, including the "large tile trap"
level: L3 - Memory & Thread Cooperation Level
target_hardware: NVIDIA Hopper H100, Blackwell B200, and modern NVIDIA GPUs with Tensor Cores
relevance: When an AI agent needs to tune Flash Attention tile sizes, understand why larger tiles can hurt performance, or optimize causal attention with K-loop splitting and autotuning
---

# Flash Attention Tile Size Tuning and Optimization

## What It Is
A practical optimization guide demonstrating how to tune Flash Attention from a baseline implementation to peak performance through an interdependent stack of optimizations: tile size selection, fast math flags, K-loop splitting for causal attention, block remapping for L2 cache utilization, and autotuning. It reveals the counter-intuitive "large tile trap" (18-43% regression from naively increasing tile size) and shows how to recover with fast math settings.

## Key Concepts
- **Large tile trap**: Increasing tiles from 64x64 to 256x128 degrades performance 18-43% due to register explosion (168 vs 128 regs/thread), occupancy collapse (18.75%), and slow special functions (exp2, division)
- **Fast math rescue**: `flush_to_zero=True` + approximate rounding eliminates slow denormal handling and iterative refinement, recovering and exceeding baseline performance
- **K-loop splitting**: For causal attention, skip fully-masked tiles entirely and minimize masking for partial tiles — saves ~50% of wasted compute
- **Block remapping**: Reorder block scheduling so adjacent blocks access nearby K/V memory, improving L2 cache hit rate
- **Autotuning**: Optimal tile sizes vary dramatically by sequence length (64x64 for short, 128x128+ for long); autotuner discovers optimal configs per input shape
- **Interdependent optimizations**: Tile size, math precision, and algorithmic strategy interact non-linearly — must be evaluated together

## Memory Layout / Data Flow
```
Optimization Stack (each builds on previous):

Baseline (64×64 tiles, standard math)
   │
   ├── [TRAP] Large tiles (256×128)
   │     └── Register pressure ↑31%, occupancy ↓, exp2/div bottleneck
   │           → Performance: -18 to -43% ✗
   │
   ├── [RESCUE] Fast math flags
   │     └── flush_to_zero, approximate rounding
   │           → Recovers + exceeds baseline: +34 to +72% ✓
   │
   ├── K-loop splitting (causal attention)
   │     ├── Skip fully-masked tiles (zero compute)
   │     ├── Partial tiles: minimal masking
   │     └── Full tiles: no masking overhead
   │           → +16 to +32% ✓
   │
   ├── Block remapping (L2 cache)
   │     └── Adjacent blocks → nearby K/V regions
   │           → +1 to +2.6% ✓
   │
   └── Autotuning (per input shape)
         ├── N ≤ 2048: 64×64 (max parallelism)
         └── N ≥ 4096: 128×128+ (memory efficiency)
               → +10 to +45% ✓

Final: 1.60-1.66x speedup, up to 918 TFLOPS on B200
```

## Performance Impact
- **Cumulative speedup**: 1.60-1.66x over baseline across all sequence lengths
- **Peak throughput**: 918 TFLOPS at N=16,384 on B200 GPU
- **K-loop splitting alone**: +16 to +32% (largest single optimization)
- **Autotuning alone**: +10 to +45% over fixed tile configurations
- **Fast math flags**: +34 to +72% recovery from the large-tile trap
- The optimizations are **multiplicative, not additive** — each builds on the gains of previous ones

## When to Use
- Implementing or tuning Flash Attention kernels for production deployment
- When profiling reveals register pressure or low occupancy in attention kernels
- When targeting causal/autoregressive attention (K-loop splitting provides large gains)
- When serving models with variable sequence lengths (autotuning provides per-shape optimization)
- When investigating unexpected performance regressions from "obvious" improvements like larger tiles
- Optimizing for specific hardware (B200, H100) where Tensor Core utilization matters

## When NOT to Use
- Using pre-built Flash Attention libraries (PyTorch, xformers) that already incorporate these optimizations
- Non-causal attention where K-loop splitting provides no benefit (bidirectional models)
- Very short sequences (N < 256) where tiling overhead dominates
- When the attention kernel is already compute-bound (rare, but possible with very small head dimensions)

## Key Takeaways
- **Never evaluate tile sizes in isolation** — math precision, register pressure, and occupancy interact non-linearly
- **Larger tiles are not always better** — the "large tile trap" is a real pitfall that causes 18-43% regressions
- **Fast math flags are essential**: `flush_to_zero` and approximate rounding unlock Tensor Core throughput for special functions
- **K-loop splitting** provides the largest single gain for causal attention by avoiding ~50% of masked computation
- **Autotuning is necessary**: Optimal tile sizes vary from 64x64 (short sequences) to 256x128 (long sequences) depending on the hardware and sequence length
- Algorithmic improvements (K-loop split) compound with low-level optimizations (fast math) for multiplicative gains

## References
- [Tuning Flash Attention for Peak Performance (NVIDIA Technical Blog)](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/)
