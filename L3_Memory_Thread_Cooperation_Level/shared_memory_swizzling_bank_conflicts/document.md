# CUDA Shared Memory Swizzling and Bank Conflict Avoidance

## Overview

Shared memory bank conflicts are a critical performance bottleneck in CUDA kernels, including Flash Attention and GEMM implementations. When multiple threads in a warp access the same shared memory bank simultaneously, accesses are serialized, degrading performance by up to 8x. **Swizzling** is the standard hardware-friendly technique to eliminate bank conflicts by rearranging the logical-to-physical memory mapping using XOR operations, without wasting any shared memory.

## What Are Bank Conflicts?

### Shared Memory Bank Organization

CUDA shared memory is organized into **32 banks** (matching warp size of 32 threads):
- Each bank is 4 bytes (32 bits) wide
- Successive 32-bit words are assigned to successive banks
- Each bank can serve one address per clock cycle
- All 32 banks can be accessed simultaneously if each thread hits a different bank

```
Bank layout for float (4-byte) shared memory:
Address:  0    4    8    12   ...  124   128  132  ...
Bank:     0    1    2    3    ...  31    0    1    ...
          ↑    ↑    ↑    ↑         ↑    ↑    ↑
          Cycle repeats every 32 × 4 = 128 bytes
```

### When Conflicts Occur

A **bank conflict** occurs when two or more threads in the same warp access different addresses in the same bank:

```
Example: Column access in a 32×32 float array

smem[32][32]:  Bank 0  Bank 1  Bank 2  ... Bank 31
Row 0:         [0,0]   [0,1]   [0,2]   ... [0,31]
Row 1:         [1,0]   [1,1]   [1,2]   ... [1,31]
Row 2:         [2,0]   [2,1]   [2,2]   ... [2,31]
...
Row 31:        [31,0]  [31,1]  [31,2]  ... [31,31]

Thread 0 reads smem[0][0]  → Bank 0
Thread 1 reads smem[1][0]  → Bank 0  ← CONFLICT!
Thread 2 reads smem[2][0]  → Bank 0  ← CONFLICT!
...
Thread 31 reads smem[31][0] → Bank 0 ← 32-way bank conflict!

Result: 32 serial accesses instead of 1 parallel access = 32x slowdown
```

### Impact on Performance

- **N-way bank conflict**: N threads hit the same bank → serialized to N sequential accesses
- **8-way conflicts** (common in practice): Reduce effective shared memory bandwidth by **8x**
- In Flash Attention kernels, bank conflicts during QK^T and PV matmuls can be the primary performance bottleneck

## Swizzling: The Solution

### Core Idea

Swizzling rearranges the mapping from logical indices to physical addresses so that threads accessing the same logical column (or row) hit **different physical banks**:

```
Standard layout:           Swizzled layout:
Col 0 → Bank 0 always     Col 0 → Bank varies by row
Col 1 → Bank 1 always     Col 1 → Bank varies by row
...                        ...
```

### XOR-Based Swizzling

The standard swizzling technique uses bitwise XOR between row and column indices:

```
Physical bank = column_chunk XOR row_chunk

Where:
  column_chunk = (byte_offset_within_row) >> log2(bank_width)
  row_chunk = row_index (or subset of row index bits)
```

### Three-Step Transformation

1. **Convert 2D index to byte-level chunk position**:
   ```
   x_chunk = (x * sizeof(element)) / bank_width_bytes
   y_chunk = y  (or relevant bits of y)
   ```

2. **Apply XOR operation**:
   ```
   x_chunk_swizzled = y_chunk XOR x_chunk
   ```

3. **Convert back to address**:
   ```
   swizzled_address = y * row_stride + x_chunk_swizzled * bank_width_bytes + (x * sizeof(element)) % bank_width_bytes
   ```

### Verification: Why XOR Works

For any two different rows y1 and y2, the swizzled column indices differ:

```
If y1 != y2, then:
  (y1 XOR x) != (y2 XOR x)  for the same x

Therefore: threads in different rows accessing the same logical column
will hit different physical banks.
```

The XOR mapping is **one-to-one** (bijective) — no data is lost or duplicated.

### Visual Example

```
Standard 4×4 bank mapping:          Swizzled 4×4 bank mapping:
        Col 0  Col 1  Col 2  Col 3          Col 0  Col 1  Col 2  Col 3
Row 0:  Bank0  Bank1  Bank2  Bank3  Row 0:  Bank0  Bank1  Bank2  Bank3
Row 1:  Bank0  Bank1  Bank2  Bank3  Row 1:  Bank1  Bank0  Bank3  Bank2
Row 2:  Bank0  Bank1  Bank2  Bank3  Row 2:  Bank2  Bank3  Bank0  Bank1
Row 3:  Bank0  Bank1  Bank2  Bank3  Row 3:  Bank3  Bank2  Bank1  Bank0

Standard: Column access = all same bank (conflict!)
Swizzled: Column access = all different banks (no conflict!)
```

## CuTe/CUTLASS Integration

In CUTLASS 3.x and CuTe, swizzling is expressed as a **layout composition**:

```cpp
// CuTe swizzle layout for shared memory
using SmemLayoutAtom = composition(
    Swizzle<3, 3, 3>{},           // XOR-based swizzle pattern
    Layout<Shape<_8, _64>, Stride<_64, _1>>{}  // Base row-major layout
);
```

The `Swizzle<B, M, S>` template parameters control:
- **B** (bits): Number of XOR bits
- **M** (mask): Which column bits participate
- **S** (shift): Offset of the row bits used for XOR

TMA descriptors can also encode swizzle patterns, meaning the hardware applies swizzling automatically during data transfer — no software overhead at all.

## Swizzling vs Padding

### Padding Approach
The simpler alternative is to add one column of padding:
```cpp
__shared__ float smem[32][33];  // 33 instead of 32
```
This shifts each row's bank alignment by 1, eliminating conflicts.

### Comparison

| Aspect | Swizzling | Padding |
|--------|-----------|---------|
| Memory waste | None | 1 column per tile (~3%) |
| Implementation | Complex XOR mapping | Trivial (+1 to dimension) |
| Vectorized access | Preserved (aligned) | May break alignment |
| Performance | ~20% faster than conflicts | ~20% faster than conflicts |
| TMA support | Hardware-encoded in descriptor | Not applicable |
| CUTLASS/CuTe | Native Swizzle<B,M,S> type | Manual |

**For production kernels (FlashAttention, GEMM)**: Swizzling is preferred because it preserves memory alignment for vectorized loads and is supported natively by TMA and CuTe.

## Performance Impact

Testing on RTX 3090 with 8192x8192 matrix transpose:
- **With bank conflicts**: 1.10 ms
- **With swizzling**: 0.92 ms
- **Speedup**: ~20%

In Flash Attention kernels specifically:
- Profiling identifies bank conflicts as the **main performance bottleneck**
- Swizzling achieves **up to 2x performance improvement** in attention-specific patterns
- 8-way bank conflicts reduce effective SMEM bandwidth by 8x, directly impacting Q*K^T and P*V matmuls

## Application in Flash Attention

Flash Attention stores Q, K, V tiles in shared memory with specific layouts:

```
SMEM Layout for Q tile (Br × d):
┌────────────────────────────────┐
│ Q[0,0..d]  → accessed by warp │  Without swizzle: column reads
│ Q[1,0..d]  → accessed by warp │  cause bank conflicts in K^T access
│ ...                            │
│ Q[Br,0..d] → accessed by warp │  With swizzle: column reads hit
└────────────────────────────────┘  different banks → no conflicts
```

During the Q @ K^T computation:
- **Q access pattern**: Row-wise (no conflicts)
- **K^T access pattern**: Column-wise (bank conflicts without swizzling)
- **Swizzled K layout**: Column reads distributed across different banks

## Sources

- [CUDA Shared Memory Swizzling (Lei Mao)](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/)
- [Flash Attention from Scratch Part 4: Bank Conflicts & Swizzling](https://lubits.ch/flash/Part-4)
- [Using Shared Memory in CUDA C/C++ (NVIDIA)](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [CUTLASS: Principled Abstractions (NVIDIA)](https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/)
