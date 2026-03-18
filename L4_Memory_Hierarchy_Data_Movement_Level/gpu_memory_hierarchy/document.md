# GPU Memory Hierarchy: How AI Training Actually Works

## Overview

Modern GPUs employ a layered memory hierarchy where each level trades capacity for speed, keeping frequently used data as close to compute units as possible. Understanding these tiers is essential for optimizing AI workloads.

## Memory Tiers

### Registers (~256 KB per SM, ~19 MB total on A100)
- **Latency**: ~1 cycle (fastest memory available)
- **Bandwidth**: ~8 TB/s
- The fastest memory in the GPU; each thread receives its own private set (32-256 registers per thread)
- Stores arithmetic operands, loop counters, addresses, and intermediate computations
- Excessive register usage causes "register spilling" into slower cache layers and reduces thread occupancy (fewer warps can run concurrently)
- On A100, total register file across all SMs is approximately 19 MB

### L1 Cache / Shared Memory (~128-192 KB per SM)
- **Latency**: ~28-35 cycles (~few nanoseconds)
- **Bandwidth**: ~15-20 TB/s
- Resides within each streaming multiprocessor (SM), private to that SM
- Serves dual purposes:
  - **Automatic L1 cache**: Recently loaded data cached transparently by hardware
  - **Programmer-managed shared memory**: Enables explicit thread cooperation within a thread block
- Foundational for **tiled matrix multiplication**, where entire thread blocks load sub-tiles once and reuse them repeatedly
- The configurable split between L1 cache and shared memory can be adjusted (e.g., 48 KB shared / 80 KB L1 or vice versa)

### L2 Cache (~40-50 MB on A100/H100)
- **Latency**: ~150-200 cycles (~tens of nanoseconds)
- **Bandwidth**: ~3 TB/s
- Sits on the GPU die outside SMs, shared across all compute units
- A100 contains 40 MB; H100 provides 50 MB
- Acts as a "global buffer," preventing repeated HBM trips when multiple SMs access identical data (e.g., shared model weights)
- L1 hit = ~30 cycles vs VRAM miss = ~500 cycles — that is a 15x difference

### HBM (High Bandwidth Memory) — 40-80 GB
- **Latency**: ~400-800 cycles (~hundreds of nanoseconds)
- **Bandwidth**: ~2-3.35 TB/s (A100: 2 TB/s, H100: 3.35 TB/s)
- Provides massive throughput but introduces significant latency compared to on-chip alternatives
- Stores model weights, activations, optimizer states, and large intermediate tensors
- Connected via stacked DRAM dies with wide buses for high throughput

### CPU RAM (Host Memory) — Hundreds of GB to TB
- **Latency**: Microseconds (orders of magnitude slower than HBM)
- **Bandwidth**: ~50-100 GB/s over PCIe/NVLink
- Used for data staging, checkpointing, and overflow
- Transfer between CPU and GPU is the slowest data movement path

## Why the Hierarchy Matters for AI

Training repeatedly accesses weights, activations, and intermediate results. Fetching everything directly from HBM would:
1. Overwhelm HBM bandwidth — even at 2+ TB/s, tensor cores can consume data faster
2. Introduce unacceptable latency — hundreds of nanoseconds per access vs single-cycle registers
3. Waste energy — off-chip memory access consumes 200x more energy than register access

The hierarchical approach stages data through progressively faster and smaller memories, reducing both latency and energy consumption while keeping tensor cores consistently supplied with operands.

## Data Flow in Training

1. **Model parameters** load from HBM into L2, accessible to all SMs
2. Within each SM, **thread groups** copy matrix sub-tiles into shared L1 memory, eliminating redundant fetches
3. **Threads** load specific operands into registers for computation by tensor cores
4. Results flow back through the hierarchy: registers → shared memory → L2 → HBM

## Why On-Chip Tiling Is Critical

The bandwidth gap between memory levels creates a fundamental bottleneck:
- Register bandwidth (~8 TB/s) vs HBM bandwidth (~2-3 TB/s) = 3-4x gap
- L1 bandwidth (~15-20 TB/s) vs HBM bandwidth = 5-10x gap

By breaking large matrices into tiles that fit in shared memory or registers, algorithms can:
- **Reuse data** multiple times before evicting it (high arithmetic intensity)
- **Avoid materializing** large intermediate results in HBM
- **Overlap** computation with data movement (software pipelining)

This is exactly what FlashAttention exploits: instead of materializing the full N×N attention matrix in HBM, it processes tiles in SRAM, dramatically reducing memory traffic.

## Approximate Memory Hierarchy Summary Table

| Level | Capacity (per SM) | Latency | Bandwidth | Scope |
|-------|-------------------|---------|-----------|-------|
| Registers | ~256 KB | ~1 cycle | ~8 TB/s | Per thread |
| L1/Shared | ~128-192 KB | ~28-35 cycles | ~15-20 TB/s | Per SM |
| L2 Cache | 40-50 MB (total) | ~150-200 cycles | ~3 TB/s | All SMs |
| HBM | 40-80 GB (total) | ~400-800 cycles | ~2-3.35 TB/s | All SMs |
| CPU RAM | 100s GB - TBs | ~μs | ~50-100 GB/s | Host |

## Sources

- [The GPU Memory Hierarchy: L2, L1, and Registers](https://www.nikhilkuniyil.com/blog/gpu-memory-hierarchy)
- [GPU Cache Hierarchy: Understanding L1, L2, and VRAM](https://charlesgrassi.dev/blog/gpu-cache-hierarchy/)
- [GPU Memory Hierarchy - How AI Training Actually Works (Medium)](https://medium.com/@indiai/gpu-memory-hierarchy-how-ai-training-actually-works-24f00cc13050)
- [Nvidia's H100: Funny L2, and Tons of Bandwidth](https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/)
