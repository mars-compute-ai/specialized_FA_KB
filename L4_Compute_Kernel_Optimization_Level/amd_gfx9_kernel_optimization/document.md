# AMD GFX9 Kernel Optimization Guide — Detailed Reference

## Overview

This document provides practical optimization techniques for GPU kernels on AMD GFX9-family architectures (CDNA2/CDNA3/CDNA4), derived from the AMDGPU kernel optimization guide by Jakub Kuderski (nod-ai/amd-shark-ai). It covers the four pillars of kernel optimization: register management, LDS usage, global memory access, and cross-lane communication.

## 1. GFX9 Architecture Overview

### MI300X (CDNA3, gfx942)
- **8 XCDs** on 4 IOD pairs, each XCD has 4 Shader Engines with 38 active CUs
- **304 total CUs**, each with 4 × 16-lane SIMDs
- **Memory**: 8 HBM3 stacks, 5.3 TB/s peak bandwidth
- Memory bandwidth calculation: 1300 MHz × 1024 bytes × 4 (QDR) = 5.3 TB/s

### MI355X (CDNA4, gfx950)
- **8 XCDs** on 2 IODs, 256 active CUs total
- **8 TB/s** HBM3E bandwidth
- Same GFX9 ISA family with extensions (block-scaled MFMA, larger LDS)

## 2. Execution Model

### Wavefront Execution
- A **wavefront** is 64 threads executing in lockstep on one SIMD unit
- Each 16-lane SIMD processes a wavefront over **4 clock cycles**
- Each SIMD has **10 waveslots** — up to 10 wavefronts can be context-switched for latency hiding

### Workgroup (Thread Block) Constraints
- A workgroup executes on a **single CU** and is never migrated
- Maximum **16 subgroups (wavefronts)** per workgroup
- Maximum subgroup size: 64 (fixed, unlike NVIDIA's 32)
- **Recommended**: 256 threads per workgroup (4 wavefronts) for good occupancy
  - 128 threads (2 wavefronts) for power conservation

### Occupancy
- Occupancy = number of active wavefronts / maximum possible wavefronts
- Limited by: registers per thread, LDS per workgroup, waveslots per SIMD
- **Target**: ≥4 wavefronts per SIMD for adequate latency hiding

## 3. Register Usage

### Register Types

| Register | Count/Thread | Purpose | Notes |
|:---------|:-------------|:--------|:------|
| SGPRs | Up to 104 | Scalar values (addresses, loop counters) | Shared across wavefront |
| VGPRs | Up to 256 | Per-thread vector data, MFMA inputs | |
| AGPRs | Up to 256 | MFMA accumulators (C/D operands) | CDNA2+ only |

### VGPR/AGPR Shared Pool

On CDNA2+, VGPRs and AGPRs share a single pool of **512 registers × 64 threads per SIMD**.

Example allocation scenarios:
| VGPRs | AGPRs | Total | Max Waves/SIMD |
|:------|:------|:------|:---------------|
| 128   | 0     | 128   | 4              |
| 128   | 64    | 192   | 2              |
| 256   | 128   | 384   | 1              |
| 256   | 256   | 512   | 1              |

### Checking Register Usage

```bash
# Method 1: ISA dump
hipcc --offload-arch=gfx942 -save-temps my_kernel.cpp
grep -E '(vgpr_count|agpr_count|sgpr_count)' my_kernel-hip-amdgcn-amd-amdhsa-gfx942.s

# Method 2: rocprof
rocprof --stats my_executable
# Look for ScratchSize (should be 0 — non-zero means spilling)
```

### Reducing Register Pressure

```cpp
// 1. Use amdgpu_waves_per_eu attribute to set minimum occupancy
__attribute__((amdgpu_waves_per_eu(4, 10)))
__global__ void my_kernel(...) { ... }

// 2. Use __builtin_amdgcn_readfirstlane to move uniform values to SGPRs
int uniform_val = __builtin_amdgcn_readfirstlane(some_value);

// 3. Minimize live variable ranges — reuse registers
// BAD: all values live simultaneously
float a = load_a(), b = load_b(), c = load_c(), d = load_d();
float result = a * b + c * d;

// GOOD: sequential reuse
float acc = load_a() * load_b();
acc += load_c() * load_d();
```

### Register Spilling

When the compiler runs out of registers, it **spills** to scratch memory (global memory).
- Spilling is catastrophic for performance — global memory latency vs register access
- Check for spilling: look for `ScratchSize > 0` in profiler output
- Fix: reduce live variables, use `amdgpu_waves_per_eu` to allow more registers (lower occupancy target)

## 4. LDS (Workgroup Memory) Optimization

### LDS Specifications

| Feature | CDNA3 (MI300X) | CDNA4 (MI355X) |
|:--------|:---------------|:---------------|
| Capacity | 64 KB per CU | 160 KB per CU |
| Banks | 32 | 64 |
| Bank width | 4 bytes | 4 bytes |
| Read bandwidth | 128 B/clock | 256 B/clock |

### Bank Conflict Analysis

Bank conflicts occur when multiple threads access the same bank in the same cycle.

**CDNA3 (32 banks)**:
```
Bank = (byte_address / 4) % 32

Example: stride-1 access (no conflict)
  Thread 0 → Bank 0, Thread 1 → Bank 1, ..., Thread 31 → Bank 31

Example: stride-2 access (2-way conflict)
  Thread 0 → Bank 0, Thread 1 → Bank 2, ..., Thread 16 → Bank 0 (conflict!)
```

**CDNA4 (64 banks)**:
```
Bank = (byte_address / 4) % 64

64 banks match the 64-thread wavefront — stride-1 access is naturally conflict-free.
```

### LDS Access Phases

Different LDS instructions have different access phases:

| Instruction | Bytes/Thread | Phases (CDNA3) | Phases (CDNA4) |
|:------------|:-------------|:---------------|:---------------|
| `ds_read_b32` | 4 | 1 | 1 |
| `ds_read_b64` | 8 | 2 | 1 |
| `ds_read_b128` | 16 | 4 | 2 |
| `ds_write_b128` | 16 | 4 | 2 |

**CDNA4 advantage**: Doubled bandwidth means `ds_read_b64` completes in 1 phase instead of 2.

### Bank Conflict Avoidance (XOR Swizzling)

```cpp
// Same technique as NVIDIA shared memory swizzling
// XOR the row index into the column index to distribute accesses across banks
int swizzled_col = col ^ (row % 32);
int lds_offset = row * (TILE_N + PADDING) + swizzled_col;

// Or: use padding to shift bank alignment
// Add 4 bytes of padding per row to break stride-N patterns
#define TILE_N_PADDED (TILE_N + 1)  // +1 element padding
```

## 5. Global Memory Access Patterns

### Optimal Load/Store Width

The optimal access width is **16 bytes (128 bits)** per thread:

```cpp
// Best: 16-byte vector load
float4 data = *reinterpret_cast<float4*>(&global_ptr[offset]);
// Emits: global_load_dwordx4

// Also good: 8-byte load
float2 data = *reinterpret_cast<float2*>(&global_ptr[offset]);
// Emits: global_load_dwordx2
```

### Coalescing Requirements

```
Wavefront coalescing:
  64 threads × 16 bytes = 1024 bytes per wavefront
  Requires: thread i accesses address base + i * 16 (contiguous)

Subgroup-contiguous access → 512-byte aligned transactions
```

### Clause Formation

Up to 4 adjacent load instructions can form a **clause** — a single fabric transaction:

```cpp
// These 4 loads may form a clause (single fabric transaction)
float4 a = *reinterpret_cast<float4*>(&ptr[offset + 0]);
float4 b = *reinterpret_cast<float4*>(&ptr[offset + 4]);
float4 c = *reinterpret_cast<float4*>(&ptr[offset + 8]);
float4 d = *reinterpret_cast<float4*>(&ptr[offset + 12]);
// Total: 256 bytes per thread in one clause
```

**Requirements for clause formation**:
- Loads must be adjacent in instruction stream (no intervening non-memory ops)
- Addresses must be from the same base pointer
- No dependency between loads within the clause

### Non-Temporal and Predicated Access

```cpp
// Non-temporal load: bypass L1/L2 cache (streaming data)
// Use when data will only be accessed once
__builtin_nontemporal_load(&ptr[offset]);

// Predicated load using buffer instructions
// Avoids divergent access — inactive lanes generate no memory traffic
// Compiler generates these for masked loads automatically
```

### L1 Cache Engagement

- L1 data cache is 32 KB per CU with 128-byte lines
- Repeated access to the same 128-byte region within a workgroup benefits from L1
- For Flash Attention: Q tile is typically reused across K/V iterations — benefits from L1 caching

## 6. Cross-Lane Data Parallel Primitives

### Instruction Hierarchy (fastest to slowest)

#### v_permlane (4-8 cycles, VALU)

```cpp
// Arbitrary permutation within wavefront
float result = __builtin_amdgcn_ds_permute(index * 4, value);
// Or: __builtin_amdgcn_permlane16(...)
```

Fastest option for arbitrary data exchange. Limited to specific permutation patterns.

#### DPP — Data Parallel Primitives (4-12 cycles, VALU)

```cpp
// Row shift, bank shift, broadcast patterns
// Example: shift right by 1 within rows of 16
float shifted = __builtin_amdgcn_mov_dpp(value,
    DPP_ROW_SR(1),    // shift right by 1
    0xF,               // row mask (all rows)
    0xF,               // bank mask (all banks)
    false);            // no bound control

// Example: butterfly reduction
float sum = value;
sum += __builtin_amdgcn_mov_dpp(sum, DPP_ROW_XOR(1), 0xF, 0xF, false);
sum += __builtin_amdgcn_mov_dpp(sum, DPP_ROW_XOR(2), 0xF, 0xF, false);
sum += __builtin_amdgcn_mov_dpp(sum, DPP_ROW_XOR(4), 0xF, 0xF, false);
sum += __builtin_amdgcn_mov_dpp(sum, DPP_ROW_XOR(8), 0xF, 0xF, false);
// 4 operations for 16-element reduction
```

Best for structured permutation patterns (shifts, broadcasts, reductions within rows/banks).

#### ds_swizzle (~50 cycles, LDS hardware)

```cpp
// Complex permutation patterns using LDS swizzle hardware
float result = __builtin_amdgcn_ds_swizzle(value, pattern);
// Does NOT actually access LDS memory — uses LDS crossbar hardware
```

More general than DPP but ~5-12× slower.

#### ds_permute / ds_bpermute (~50 cycles, LDS hardware)

```cpp
// General indexed permutation
float result = __builtin_amdgcn_ds_permute(src_lane * 4, value);

// Inverse permutation (each thread specifies where to READ from)
float result = __builtin_amdgcn_ds_bpermute(src_lane * 4, value);
```

Most general — supports arbitrary permutation patterns. Slowest option.

### Choosing the Right Primitive for Flash Attention

| Operation | Best Primitive | Latency | Notes |
|:----------|:--------------|:--------|:------|
| Softmax row max | DPP butterfly | 4-12 cy/step | 6 steps for wave-64 |
| Softmax row sum | DPP butterfly | 4-12 cy/step | Same pattern as max |
| Broadcast max/sum | `v_readfirstlane` | 4 cy | Scalar broadcast |
| Cross-wave exchange | LDS read/write | ~50 cy | Between wavefronts in workgroup |
| MMA operand shuffle | DPP row shift | 4-12 cy | Rearrange for MFMA layout |

## 7. Putting It All Together: Attention Kernel Optimization Checklist

1. **Register budget**: Target ≤128 VGPRs + 64 AGPRs for ≥2 waves/SIMD occupancy
2. **LDS allocation**: Q tile + K/V double buffer + softmax scratch ≤ 64 KB (CDNA3) or 160 KB (CDNA4)
3. **Global loads**: Use `global_load_dwordx4` (16B/thread), ensure coalesced access, form 4-instruction clauses
4. **LDS banking**: Apply XOR swizzling for Q/K/V tile layouts to avoid bank conflicts
5. **Reductions**: Use DPP butterfly for wave-level softmax max/sum (6 steps, 4-12 cy each)
6. **Cross-wave**: Use LDS for workgroup-level reductions (softmax normalization across waves)
7. **Prefetching**: Issue async global loads early; overlap with MFMA compute

## References

- [AMDGPU Kernel Optimization Guide — Jakub Kuderski, nod-ai/amd-shark-ai (2025-08-14)](https://github.com/nod-ai/amd-shark-ai)
- [AMD CDNA3 ISA Manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD CDNA4 ISA Manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi350-cdna4-instruction-set-architecture.pdf)
- [ROCm Optimization Guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides.html)
