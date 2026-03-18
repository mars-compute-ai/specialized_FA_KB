---
skill_name: Shared Memory Swizzling and Bank Conflict Avoidance
description: XOR-based address remapping to eliminate shared memory bank conflicts in attention and GEMM kernels without wasting memory
level: L4 - Memory-Hierarchy/Data-Movement Level
target_hardware: All NVIDIA GPUs with shared memory (Volta, Ampere A100, Hopper H100, Blackwell B200)
relevance: When an AI agent needs to diagnose or fix shared memory bank conflicts in CUDA kernels, understand CuTe Swizzle layouts, or optimize Flash Attention shared memory access patterns
---

# Shared Memory Swizzling and Bank Conflict Avoidance

## What It Is
Shared memory in CUDA is organized into 32 banks; when multiple threads in a warp access different addresses in the same bank, accesses serialize, reducing effective bandwidth by up to 32x. Swizzling applies a bitwise XOR between row and column indices to remap logical addresses to physical banks, ensuring that column-wise access patterns (common in transposed matrix reads) hit different banks. Unlike padding, swizzling wastes no memory and preserves alignment for vectorized loads.

## Key Concepts
- **32 banks, 4 bytes each**: Successive 32-bit words map to successive banks; pattern repeats every 128 bytes
- **N-way bank conflict**: N threads hit same bank → N serial accesses (8-way conflicts = 8x bandwidth loss)
- **XOR swizzle**: `physical_bank = column_chunk XOR row_chunk` — guarantees different rows map same column to different banks
- **Bijective mapping**: XOR is one-to-one; no data loss, no wasted memory
- **CuTe integration**: `Swizzle<B,M,S>` template encodes swizzle as a layout composition, applied at compile time
- **TMA hardware swizzle**: On Hopper, swizzle patterns are encoded in TMA descriptors — zero software overhead
- **Column access is the problem**: Row-wise access naturally hits different banks; column-wise (transpose) access causes conflicts
- **Flash Attention impact**: K^T access during QK^T computation is column-wise — swizzling is critical

## Memory Layout / Data Flow
```
WITHOUT SWIZZLE (column access causes bank conflicts):

SMEM Banks:  B0   B1   B2   B3   ...  B31
Row 0:      [0,0] [0,1] [0,2] [0,3] ... [0,31]
Row 1:      [1,0] [1,1] [1,2] [1,3] ... [1,31]
Row 2:      [2,0] [2,1] [2,2] [2,3] ... [2,31]

Column 0 access: Thread 0→B0, Thread 1→B0, Thread 2→B0
                  → 32-way bank conflict! (serialized)

WITH XOR SWIZZLE (column access hits different banks):

SMEM Banks:  B0   B1   B2   B3   ...
Row 0:      [0,0] [0,1] [0,2] [0,3] ...  (bank = col XOR 0)
Row 1:      [1,1] [1,0] [1,3] [1,2] ...  (bank = col XOR 1)
Row 2:      [2,2] [2,3] [2,0] [2,1] ...  (bank = col XOR 2)

Column 0 access: Thread 0→B0, Thread 1→B1, Thread 2→B2
                  → No conflicts! (fully parallel)
```

## Performance Impact
- **Bank conflict elimination**: Up to 8x recovery of effective shared memory bandwidth
- **Matrix transpose benchmark**: 20% speedup on RTX 3090 (1.10 ms → 0.92 ms for 8192x8192)
- **Flash Attention**: Up to **2x improvement** when profiling identifies bank conflicts as the bottleneck
- **No memory waste**: Unlike padding (which wastes ~3% of shared memory), swizzling uses 100% of allocated SMEM
- **Preserved alignment**: Vectorized loads (128-bit, 4 floats) remain aligned, enabling efficient `LDS.128` instructions

## When to Use
- Any CUDA kernel that reads shared memory in a transposed (column-wise) pattern
- Flash Attention kernels during K^T access in the QK^T computation
- GEMM kernels loading matrix tiles from shared memory
- When profiling (Nsight Compute) shows high bank conflict counts in shared memory operations
- When using CuTe/CUTLASS 3.x — express swizzle as `Swizzle<B,M,S>` in layout composition
- When using TMA on Hopper — encode swizzle in the TMA descriptor for hardware-applied remapping

## When NOT to Use
- When shared memory access is purely row-wise (no bank conflicts to begin with)
- When padding is simpler and the small memory waste is acceptable (prototyping)
- When shared memory is not the performance bottleneck (check profiler first)
- Very small tiles where the indexing overhead of swizzling exceeds the bank conflict cost
- Non-NVIDIA hardware where shared memory bank organization differs

## Key Takeaways
- Bank conflicts are a **silent performance killer** — they do not cause errors, only slowdowns (up to 32x)
- XOR-based swizzling is the **standard solution** in all production CUDA kernels (CUTLASS, Flash Attention, cuBLAS)
- The technique is lossless (bijective), memory-efficient (no padding waste), and alignment-preserving
- On Hopper, TMA encodes swizzle in hardware descriptors — eliminating any software overhead
- In CuTe, `Swizzle<B,M,S>` composes with any layout at compile time — developers do not manually compute XOR indices
- **Always profile first**: Use Nsight Compute to measure bank conflict counts before applying swizzling

## References
- [CUDA Shared Memory Swizzling (Lei Mao)](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/)
- [Flash Attention from Scratch Part 4: Bank Conflicts & Swizzling](https://lubits.ch/flash/Part-4)
- [Using Shared Memory in CUDA C/C++ (NVIDIA)](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [CUTLASS: Principled Abstractions (NVIDIA)](https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/)
