# Reimplementing FlashAttention for Performance and Giggles: A Profiling Deep Dive

Source: https://aminediro.com/posts/flash_attn/

## Overview

This analysis documents a from-scratch Flash Attention implementation in Triton on an RTX 2070, with detailed GPU profiling using NVIDIA Nsight Compute to understand performance bottlenecks iteratively. It demonstrates profiler-driven optimization methodology for attention kernels.

## Hardware Configuration

- **Device**: NVIDIA GeForce RTX 2070
- Compute Capability: 7.5 (Turing)
- Total Memory: 8 GB VRAM
- Streaming Multiprocessors (SMs): 36
- Max Threads per SM: 1024
- Warp Size: 32
- Shared Memory per Block: 48 KB
- L2 Cache: 4 MB

## Profiling Methodology

### Tools Used
1. **Nsight Compute (ncu)**: Heavy-duty profiler providing occupancy, memory throughput, warp stalls, instruction mix, and bank conflicts
2. **Nsight Systems (nsys)**: System-wide CPU/GPU timeline analysis
3. **torch.profiler**: Quick wall-clock timing and basic GPU utilization

### Profiling Command
```bash
sudo ncu --set full --kernel-name "attn_kernel" -o profile_output -f python script.py
```

## Flash Attention v1: Initial Implementation (Outer-KV Loop)

### Algorithm Structure
- **Outer loop**: Over K/V blocks (sequence dimension)
- **Inner loop**: Over Q blocks (rows)
- **Grid configuration**: `(B, N_h, 1)` with B x N_h thread blocks
- **Block size**: 128 threads (4 warps)

### Occupancy Analysis

**Finding**: Theoretical occupancy limited to **25%** by shared memory pressure.

Occupancy calculator revealed:
- Shared memory requirement: ~7,232 floats = ~28 KB per block
- Maximum active blocks per SM: 2
- Warp occupancy: 2.0 theoretical warps per scheduler (vs 8.0 hardware maximum)

**Impact**: Low occupancy reduces GPU's ability to hide memory latency through thread scheduling.

### Memory Throughput Issues

**Main memory traffic**: 11.58 GB reads + 5.54 GB writes

**Root cause**: The loop structure forces repeated HBM accesses:
- Q reload: 64 iterations x (Q matrix size) = ~10.6 GB
- O reload/write: 64 iterations x (O matrix size) = ~5.3 GB

> "We are treating HBM like a register, which explains the massive bandwidth usage."

### SRAM Allocation Breakdown (Bc=32, D_h=32)
- Query block: 32 x 32 floats
- Key block: 32 x 32 floats
- Value block: 32 x 32 floats
- Attention scores: 32 x 32 floats
- Statistics (m, l): 2 x 32 floats
- **Total**: ~7.2 KB active allocation

### Instruction-Level Bottleneck

Division operation in online softmax flagged as problematic:
```
oi_new = (alpha * prev_li * prev_oi + beta * pij @ vj) / li_new
```
Division requires MUFU (reciprocal) + refinement multiplies, adding pipeline pressure.

## Flash Attention v2: Loop Restructuring (Outer-Q Loop)

### Key Changes
1. **Inverted loop structure**: Outer loop parallelized over Q blocks via grid; inner loop only over K/V blocks
2. **New grid**: `(S/Bc, B x N_h)` enabling ~20,480 independent thread blocks
3. **SRAM residency**: Load Q once at kernel start, keep O accumulator and statistics in registers
4. **Deferred normalization**: Single division at kernel end

### Memory Throughput Improvement

Main memory reads reduced to **412.18 MB** (-92.98% vs v1), with 80 MB writes matching output size.

### Critical Issue: Shared Memory Bank Conflicts

Profiler identified severe bottlenecks:

> "6.3-way average bank conflict across 293.6M shared load requests producing 1.17B total bank conflicts (63.64% of 1.85B wavefronts)"

### Bank Conflict Root Cause

Shared memory organization:
- 32 physical banks, one per thread in warp
- Bank mapping: `(byte_address / 4) mod 32`
- K stored with row-stride of 256 bytes (= 64 floats x 4 bytes)

**Access pattern problem**: Due to stride alignment, threads accessing K column data suffered from **16-way conflicts on identical banks**:

```
Lane 0-15: base + {0, 512, 1024, 1536, ...}  (unique addresses)
Lane 16-31: base + {0, 512, 1024, 1536, ...}  (duplicates)
```

**Efficiency**: 1/16 = 6.25% useful work, 93.75% wasted.

### MIO Pipeline Throttling

> "Warp spends 18.4 cycles stalled waiting for MIO instruction queue to be not full, representing 50.7% of total 36.2 cycles between instruction issue"

MIO handles both shared memory access and special math (exp, max, log), creating pipeline congestion.

## Flash Attention v2 with K Transpose: Bank Conflict Fix

### Solution
```python
k_trans = k.transpose(-1, -2).contiguous()  # Force contiguous layout
```

Storing K in column-major (D x Bc layout) converts problematic column reads into sequential row accesses, distributing bank usage naturally.

### Results
- **Execution time**: 34 ms (145% improvement vs v1's 166 ms)
- **Bank conflicts reduced**: 6.3-way to ~3.4-way average
- **Eligible warps per cycle**: +153% increase

### Remaining MIO Bottleneck

After fixing bank conflicts, MIO stalls still represent ~44% of warp idle time:
1. **Special math operations**: `tl.exp()` and `tl.max()` called every iteration
2. **Block size constraints**: Bc=32 limits loop iterations before softmax operations accumulate

## Tensor Core Utilization Gap

Instruction analysis showed dominance of FFMA (Fused Floating-point Multiply-Add) on regular CUDA cores, NOT Tensor Cores.

**Root cause**: Triton compiler on SM 7.5 (Turing) fails to generate Tensor Core code, falling back to FFMA instructions (16x slower).

## Occupancy vs Performance Trade-offs

The relationship is non-monotonic:

> "Theoretical occupancy (25% for v1) is limited by shared memory, but increasing occupancy beyond latency-hiding threshold degrades performance through reduced per-thread resources"

- **v1 occupancy**: 25% (limited by 28 KB SMEM per block)
- **v2 occupancy**: 63% (12 KB SMEM per block)

## Tile Size Selection

### Bc = Br = 32 Justification

**Constraints**:
- Shared memory per block: 48 KB maximum
- Required: `2*Bc + 3*Bc*D + Bc^2` floats
- With D=32: 4,192 floats = ~16.8 KB

**Trade-offs**:
- Larger Bc: Fewer loop iterations (fewer exp/max ops), but higher SRAM/register pressure
- Smaller Bc: More iterations (more special math), but better occupancy

## Profiler-Driven Optimization Summary

Each ncu report identified the next bottleneck:

| Version | Time (ms) | Speedup | Bottleneck Identified |
|---------|-----------|---------|----------------------|
| v1 (naive, outer-KV) | 166.47 | 1.0x | HBM traffic (11.6 GB reads) |
| v2 (outer-Q) | 177.25 | 0.94x | Bank conflicts (6.3-way) |
| v2-transpose | 34.00 | 4.9x | MIO pipeline / No tensor cores |

## Practical Lessons Learned

1. **Memory layout matters enormously**: Bank conflicts causing 93.75% efficiency loss show that physical memory layout directly impacts performance by orders of magnitude
2. **Loop order is fundamental**: Inverting loops transformed the problem from "treat HBM as register" to "stream K/V with Q resident"
3. **Deferred computation saves pipeline**: Deferring division to kernel end eliminates expensive MUFU operations from hot loop
4. **Architecture constraints are real**: Lack of tensor core code generation on SM 7.5 created an insurmountable 16x performance ceiling
5. **Profiler-driven optimization**: Each ncu report identified the specific next bottleneck to fix
6. **Occupancy is necessary but not sufficient**: Going from 25% to 63% occupancy improved latency hiding, but beyond that threshold, more occupancy did not help
