# AMD Wavefront Cooperation: HIP Advanced Optimization, Native Examples, and Thread Synchronization

## Overview

This document synthesizes AMD's HIP advanced optimization techniques, native programming examples, and thread synchronization patterns relevant to Flash Attention kernel development on CDNA3/CDNA4 GPUs. It covers the Wave-64 execution model, MFMA instruction scheduling, memory operations, synchronization primitives, and cooperation patterns.

---

## 1. AMD GPU Architecture Fundamentals

### Wave-64 Execution Model

AMD CDNA GPUs execute 64 threads per wavefront in SIMT (Single Instruction, Multiple Thread) mode. This is the single most important difference from NVIDIA's 32-thread warp model.

**Thread identification:**
```cpp
constexpr int WAVE_SIZE = 64;
__device__ int laneid() { return threadIdx.x & 0x3F; }  // Mask with 63
__device__ int waveid() { return threadIdx.x >> 6; }    // Divide by 64
```

**Implications for Flash Attention:**
- Softmax reductions require 6 shuffle steps (log2(64) = 6) instead of 5 (log2(32) = 5)
- Each wavefront can process more elements per reduction, potentially requiring fewer waves per workgroup
- Ballot returns `unsigned long long` (64-bit mask) instead of `unsigned int` (32-bit)
- Register allocation is per-wave: 64 threads share the wave's register budget

### Dual Register Files

AMD CDNA GPUs have two separate register files per thread:

| Register File | Range | Purpose | Flash Attention Usage |
|--------------|-------|---------|----------------------|
| VGPR (Vector) | v[0:255] | General purpose, addressing | Q/K/V tile data, address computation |
| AGPR (Accumulator) | a[0:255] | MFMA accumulation | S = Q*K^T accumulator, O = P*V accumulator |

Total: 512 registers per thread (256 VGPR + 256 AGPR). The split register file means MFMA outputs go to AGPRs without consuming VGPR budget.

### Memory Hierarchy

```
L1 Vector Cache:    16 KB per CU (write-through)
LDS:                64 KB per CU (32 banks x 4 bytes)
L2 Cache:           ~400 KB per XCD (shared across CUs in same chiplet)
L3 Infinity Cache:  256-384 MB (shared across all XCDs)
HBM3/HBM3e:         5.3-6.9 TB/s bandwidth
```

**LDS bank structure:**
- 32 banks, 4 bytes per bank per cycle
- Bank ID = `(address / 4) % 32`
- Bank conflicts serialize access -- critical for attention tile operations

### XCD (Chiplet) Architecture

```
MI300X: 8 XCDs, 38 CUs each = 304 CUs total
MI350X: 8 XCDs, 40 CUs each = 320 CUs total
```

Workgroups on the same XCD share L2 cache. For Flash Attention, scheduling Q/K/V tiles to workgroups on the same XCD improves L2 hit rates.

---

## 2. MFMA Instructions for Attention

### MFMA (Matrix Fused Multiply-Add) Basics

The core tensor core instruction on AMD CDNA GPUs:

```asm
v_mfma_f32_16x16x32_bf16 D[0:3], A[0:3], B[0:3], C[0:3]
```

**Semantics:** D = A x B + C
- A: 16x32 BF16 matrix (4 registers)
- B: 32x16 BF16 matrix (4 registers)
- C/D: 16x16 FP32 matrix (4 registers, accumulator)
- All 64 threads in the wave participate, each holding a portion of the result

**Register layout per thread:**
- A: 2 rows x 32 cols of input (64 BF16 values = 4 registers)
- B: 2 rows x 32 cols (64 BF16 values = 4 registers)
- C/D: 4 FP32 output elements (4 registers)

### MFMA Variants for Attention

| Instruction | Tile Size | Input Type | Registers | Use Case |
|------------|----------|-----------|-----------|----------|
| `v_mfma_f32_16x16x32_bf16` | 16x16x32 | BF16 | A:4, B:4, C:4 | Standard FA tiles |
| `v_mfma_f32_16x16x16_f16` | 16x16x16 | FP16 | A:4, B:4, C:4 | FP16 FA tiles |
| `v_mfma_f32_32x32x8_bf16` | 32x32x8 | BF16 | A:4, B:4, C:16 | Larger FA tiles |
| `v_mfma_f32_16x16x32_fp8_fp8` | 16x16x32 | FP8 | A:2, B:2, C:4 | FP8 FA (CDNA4) |

### C++ Wrapper Pattern

```cpp
template<int GPR_D, int GPR_A, int GPR_B, int GPR_C>
__device__ __forceinline__ void mfma_f32_16x16x32_bf16() {
    if constexpr (GPR_D >= 256) {
        // All in AGPRs -- preferred for accumulation
        asm volatile(
            "v_mfma_f32_16x16x32_bf16 a[%0:%1], a[%2:%3], a[%4:%5], a[%6:%7]"
            : : "n"(GPR_D-256), "n"(GPR_D+3-256),
                "n"(GPR_A-256), "n"(GPR_A+3-256),
                "n"(GPR_B-256), "n"(GPR_B+3-256),
                "n"(GPR_C-256), "n"(GPR_C+3-256)
        );
    } else {
        // All in VGPRs
        asm volatile(
            "v_mfma_f32_16x16x32_bf16 v[%0:%1], v[%2:%3], v[%4:%5], v[%6:%7]"
            : : "n"(GPR_D), "n"(GPR_D+3),
                "n"(GPR_A), "n"(GPR_A+3),
                "n"(GPR_B), "n"(GPR_B+3),
                "n"(GPR_C), "n"(GPR_C+3)
        );
    }
}
```

---

## 3. Wavefront Shuffle Operations for Softmax

### Butterfly Reduction Pattern (Wave-64)

The fundamental softmax reduction across a 64-lane wavefront:

```cpp
// Max reduction across wave (for softmax numerical stability)
__device__ inline float wave_reduce_max(float value) {
    float result = value;
    for (int offset = 32; offset > 0; offset >>= 1) {
        float other = __shfl_xor(result, offset, 64);
        result = fmaxf(result, other);
    }
    return result;  // All 64 lanes have the same max
}

// Sum reduction across wave (for softmax normalization)
__device__ inline float wave_reduce_sum(float value) {
    float result = value;
    for (int offset = 32; offset > 0; offset >>= 1) {
        float other = __shfl_xor(result, offset, 64);
        result += other;
    }
    return result;  // All 64 lanes have the same sum
}
```

**Step-by-step for 64 lanes (offset = 32, 16, 8, 4, 2, 1):**
```
Step 1 (offset=32): Lane 0 <-> Lane 32, Lane 1 <-> Lane 33, ...
Step 2 (offset=16): Lane 0 <-> Lane 16, Lane 1 <-> Lane 17, ...
Step 3 (offset=8):  Lane 0 <-> Lane 8,  Lane 1 <-> Lane 9,  ...
Step 4 (offset=4):  Lane 0 <-> Lane 4,  Lane 1 <-> Lane 5,  ...
Step 5 (offset=2):  Lane 0 <-> Lane 2,  Lane 1 <-> Lane 3,  ...
Step 6 (offset=1):  Lane 0 <-> Lane 1,  Lane 2 <-> Lane 3,  ...
```

After 6 steps, all lanes hold the global max/sum of the original 64 values.

### Multi-Wave Softmax Reduction

When a workgroup has multiple wavefronts, partial results must be combined via LDS:

```cpp
__global__ void multi_wave_softmax_reduction(const float* input, float* output, int N) {
    __shared__ float wave_results[BLOCK_SIZE / 64];  // One slot per wavefront

    int lane = threadIdx.x % 64;
    int waveId = threadIdx.x / 64;

    // Step 1: Load and reduce within wavefront
    float val = input[blockIdx.x * blockDim.x + threadIdx.x];
    val = wave_reduce_max(val);

    // Step 2: First lane writes wavefront result to LDS
    if (lane == 0) {
        wave_results[waveId] = val;
    }
    __syncthreads();

    // Step 3: First wavefront reduces across all wavefronts
    if (waveId == 0) {
        val = (lane < blockDim.x / 64) ? wave_results[lane] : -INFINITY;
        val = wave_reduce_max(val);
        if (lane == 0) {
            output[blockIdx.x] = val;
        }
    }
}
```

### Shuffle Variants

**`__shfl_down` -- for prefix-style reductions:**
```cpp
// Lane 0 gets the final sum (other lanes have partial results)
float wavefront_sum_down(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, 64);
    }
    return val;  // Only lane 0 has the correct total
}
```

**`__shfl_up` -- for inclusive prefix sum (scan):**
```cpp
float wavefront_scan(float val) {
    int lane = threadIdx.x % 64;
    for (int offset = 1; offset < 64; offset <<= 1) {
        float temp = __shfl_up(val, offset, 64);
        if (lane >= offset) val += temp;
    }
    return val;  // Each lane has prefix sum up to its position
}
```

**`__shfl` -- for broadcasting:**
```cpp
float broadcast_from_lane0 = __shfl(val, 0, 64);  // All lanes get lane 0's value
```

### Voting Primitives

```cpp
// Returns 64-bit mask (unsigned long long on AMD, not unsigned int)
unsigned long long mask = __ballot(predicate);

// Count set bits
int count = __popcll(mask);

// Check if any/all lanes satisfy predicate
int any = __any(predicate);
int all = __all(predicate);
```

---

## 4. HIP Synchronization Primitives

### Block-Level Synchronization

```cpp
__syncthreads();  // Synchronize all threads in workgroup
// Equivalent to __builtin_amdgcn_s_barrier()
```

**Critical rules:**
1. ALL threads in the workgroup must reach the barrier (no conditional barriers)
2. Synchronizes within workgroup only (not across workgroups)
3. Does not guarantee memory visibility across CUs (use atomics for that)

### Wait Counters

AMD GPUs have separate counters for different operation types:

```cpp
// Wait for ALL outstanding operations
__builtin_amdgcn_s_waitcnt(0);

// Wait for LDS/GDS operations
asm volatile("s_waitcnt lgkmcnt(0)");     // Wait for all LDS ops
asm volatile("s_waitcnt lgkmcnt(4)");     // Wait until 4 or fewer remain

// Wait for vector memory (global) operations
asm volatile("s_waitcnt vmcnt(0)");       // Wait for all VMEM ops
asm volatile("s_waitcnt vmcnt(2)");       // Wait until 2 or fewer remain
```

**Use in Flash Attention:**
- `lgkmcnt` is used after LDS reads/writes for K/V tile data
- `vmcnt` is used after global memory loads of Q/K/V tiles
- Partial waits (`lgkmcnt(N)`, `vmcnt(N)`) enable software pipelining by allowing some operations to complete while others are still in flight

### Schedule Barriers

```cpp
// Prevent scheduler from reordering instructions across this point
__builtin_amdgcn_sched_barrier(0);

// Group barriers for instruction type control
__builtin_amdgcn_sched_group_barrier(mask, count, order);
// mask:  instruction type (0x008=MFMA, 0x200=TRANS, 0x002=VALU, 0x004=SALU)
// count: number of instructions of this type to allow
// order: 0 = apply to all waves
```

**FMHA V3 scheduling pattern:**
```cpp
// Phase 0, Wave Group 0: Interleave MFMA with TRANS and VALU
for (int i = 0; i < 8; i++) {
    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // 1 MFMA
    __builtin_amdgcn_sched_group_barrier(0x200, 2, 0);  // 2 TRANS
    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0);  // 2 VALU
}
```

### Priority Control

```cpp
__builtin_amdgcn_s_setprio(1);  // Elevate priority (compute gets preference)
// Critical MFMA instructions here
__builtin_amdgcn_s_setprio(0);  // Return to normal priority
```

Used in Flash Attention to ensure tensor core instructions are not starved by memory operations.

---

## 5. LDS (Shared Memory) Operations

### Basic LDS Read/Write

```cpp
// Read 64 bits (8 bytes) from LDS
template<int GPR_START>
__device__ __forceinline__ void ds_read_b64(uint32_t lds_addr, int offset = 0) {
    asm volatile(
        "ds_read_b64 v[%0:%1], %2 offset:%3"
        : : "n"(GPR_START), "n"(GPR_START+1), "v"(lds_addr), "i"(offset)
        : "memory"
    );
}

// Write 64 bits to LDS
template<int GPR_START>
__device__ __forceinline__ void ds_write_b64(uint32_t lds_addr, int offset = 0) {
    asm volatile(
        "ds_write_b64 %0, v[%1:%2] offset:%3"
        : : "v"(lds_addr), "n"(GPR_START), "n"(GPR_START+1), "i"(offset)
        : "memory"
    );
}
```

### Transposing LDS Read

A special AMD instruction that reads and transposes 16-bit elements during the load:

```cpp
template<int GPR_START>
__device__ __forceinline__ void ds_read_b64_tr_b16(uint32_t lds_addr, int offset = 0) {
    asm volatile(
        "ds_read_b64_tr_b16 v[%0:%1], %2 offset:%3"
        : : "n"(GPR_START), "n"(GPR_START+1), "v"(lds_addr), "i"(offset)
        : "memory"
    );
}
```

Used in the FMHA backward pass to convert between row-major and column-major layouts on the fly, avoiding separate transpose kernels.

### Bank Conflict Avoidance via Swizzling

```cpp
__device__ inline uint32_t swizzle_16x16_bf16(int row, int col) {
    // XOR swizzle to distribute across 32 banks
    int row_swizzle = row ^ ((col >> 2) & 0x7);
    return (row_swizzle * 16 + col) * sizeof(__hip_bfloat16);
}
```

For Flash Attention tiles, the 32 banks at 4 bytes each give 128 bytes per cycle. Without swizzling, accessing a column of a 16-wide BF16 tile would cause 8-way bank conflicts.

---

## 6. Direct Buffer-to-LDS Transfers

### The Optimization

Standard path consumes VGPR budget:
```
Global Memory -> VGPRs -> LDS
```

Optimized path bypasses VGPRs entirely:
```
Global Memory -> LDS (direct)
```

### LLVM Intrinsic

```cpp
extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc,           // Buffer resource descriptor
    uint32_t* lds_ptr,    // LDS destination (address space 3)
    uint32_t size,        // Size in bytes (4, 8, 12, or 16)
    uint32_t voffset,     // Vector offset (per-lane)
    uint32_t soffset,     // Scalar offset (uniform)
    uint32_t inst_offset, // Instruction offset (immediate)
    int coherency         // Cache coherency mode
) __asm("llvm.amdgcn.raw.buffer.load.lds");
```

### Buffer Resource Descriptor (SRD)

```cpp
using i32x4 = int32_t __attribute__((ext_vector_type(4)));

__device__ inline i32x4 make_buffer_resource(const void* ptr, uint32_t range_bytes) {
    buffer_resource br;
    br.ptr = reinterpret_cast<uint64_t>(ptr);
    br.range = range_bytes;
    br.config = 0x00020000;  // Standard config
    return *reinterpret_cast<i32x4*>(&br);
}
```

### Readfirstlane Hoisting

The LDS pointer must be converted to a uniform (scalar) value. Hoisting `readfirstlane` out of loops provides 10-20% speedup:

```cpp
// Hoist: compute ONCE before loop
uint32_t lds_base = __builtin_amdgcn_readfirstlane(
    static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&shared_buffers[0][0]))
);

// Main loop: no repeated readfirstlane
for (int i = 0; i < num_tiles; ++i) {
    llvm_amdgcn_raw_buffer_load_lds(
        buf_rsrc, (lds_ptr_t)lds_base, 16,
        i * tile_stride, 0, 0, 0
    );
}
```

---

## 7. Cooperation Patterns for Flash Attention

### Pattern 1: Ping-Pong Buffering (8-Wave)

Used for large attention tiles with 8 wavefronts per workgroup:

```cpp
__global__ void ping_pong_attention() {
    __shared__ float K[2][TILE_N][HEAD_DIM];  // Double buffer for K tiles
    __shared__ float V[2][TILE_N][HEAD_DIM];  // Double buffer for V tiles

    int tic = 0, toc = 1;

    // Prologue: load first K/V tile
    load_kv_tile(K[tic], V[tic], /*iter=*/0);
    __builtin_amdgcn_s_barrier();

    // Main loop
    for (int k = 0; k < num_kv_blocks - 1; ++k) {
        // Async load next tile
        load_kv_tile(K[toc], V[toc], k+1);

        // Compute S = Q * K^T on current tile
        compute_qk(Q_reg, K[tic], S_acc);

        // Softmax
        softmax_update(S_acc, m_prev, l_prev);

        // Compute O += P * V on current tile
        compute_pv(P_reg, V[tic], O_acc);

        // Swap buffers
        tic ^= 1; toc ^= 1;
        __builtin_amdgcn_s_barrier();
    }

    // Epilogue: process last tile
    compute_qk(Q_reg, K[tic], S_acc);
    softmax_update(S_acc, m_prev, l_prev);
    compute_pv(P_reg, V[tic], O_acc);
}
```

### Pattern 2: Fine-Grained Interleaving (4-Wave)

For smaller tiles, interleave individual MFMA and memory operations:

```cpp
__device__ void interleaved_attention_step() {
    // Cluster 0: MFMA (S = Q * K^T partial)
    mfma_f32_16x16x32_bf16<0, 16, 32, 32>();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 1: Load next K tile (overlaps with MFMA pipeline)
    buffer_load_dwordx4<48>(k_buf_rsrc, k_offset);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2: MFMA (continue Q * K^T)
    mfma_f32_16x16x32_bf16<4, 20, 36, 36>();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3: Load next V tile
    buffer_load_dwordx4<52>(v_buf_rsrc, v_offset);
    __builtin_amdgcn_sched_barrier(0);
}
```

### Pattern 3: Chiplet-Aware Scheduling

Remap workgroup IDs to improve L2 cache locality on MI300X (8 XCDs):

```cpp
__device__ inline int chiplet_transform(int wgid, int num_wgs, int num_xcds) {
    int xcd_id = wgid % num_xcds;
    int local_wg = wgid / num_xcds;
    return xcd_id * (num_wgs / num_xcds) + local_wg;
}
```

This ensures that workgroups processing adjacent Q tiles are scheduled on the same XCD, maximizing L2 cache reuse for shared K/V tiles.

---

## 8. Atomic Operations and Contention Reduction

### Wavefront-Aggregated Atomics

For Flash Attention split-K combine or multi-head reduction:

```cpp
// BAD: All 64 lanes do atomic (high contention)
atomicAdd(output, partial_result);

// BETTER: Reduce within wave first, then single atomic per wave
float wave_sum = wave_reduce_sum(partial_result);
if (laneid() == 0) {
    atomicAdd(output, wave_sum);
}

// BEST: Reduce within wave, then across waves via LDS, then single atomic per block
// (See multi-wave reduction pattern above)
```

### Memory Fences

```cpp
__threadfence();         // Visible to all threads on device
__threadfence_block();   // Visible to all threads in workgroup
__threadfence_system();  // Visible to host and device
```

Used in Flash Attention for producer-consumer patterns between data loading and computation phases.

---

## 9. Element-Wise Operations for Softmax

### Fast Math Intrinsics

```cpp
// Exponential (for softmax exp)
__device__ inline float fast_exp(float x) {
    float result;
    asm volatile("v_exp_f32 %0, %1" : "=v"(result) : "v"(x));
    return result;
}

// Reciprocal (for softmax 1/sum)
__device__ inline float fast_rcp(float x) {
    float result;
    asm volatile("v_rcp_f32 %0, %1" : "=v"(result) : "v"(x));
    return result;
}

// Reciprocal square root (for layer norm)
__device__ inline float fast_rsqrt(float x) {
    float result;
    asm volatile("v_rsq_f32 %0, %1" : "=v"(result) : "v"(x));
    return result;
}
```

### Type Conversions

```cpp
// BF16 to Float
float val = __bfloat162float(bf16_val);

// Float to BF16 (fast truncation)
__hip_bfloat16 bf16 = std::bit_cast<__hip_bfloat16>(
    static_cast<uint16_t>(std::bit_cast<uint32_t>(float_val) >> 16)
);
```

---

## 10. Profiling and Debugging

### Key Metrics for Flash Attention

```bash
# Basic kernel stats
rocprofv3 --stats ./flash_attention_kernel

# Detailed performance counters
rocprofv3 --pmc SQ_INSTS_MFMA,SQ_LDS_BANK_CONFLICT,TCC_HIT,TCC_MISS ./kernel

# Resource usage at compile time
hipcc --resource-usage flash_attention.cpp
```

**Critical metrics:**
- `SQ_INSTS_MFMA`: MFMA instruction count (should be high for compute-bound attention)
- `SQ_LDS_BANK_CONFLICT`: LDS bank conflicts (should be near zero with proper swizzling)
- `TCC_HIT` / `TCC_MISS`: L2 cache hit/miss (affected by chiplet-aware scheduling)
- VGPR/SGPR usage: Register pressure (affects occupancy)

---

## References

- [HIP Programming Guide - Warp Cross-Lane Functions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/programming_manual.html#warp-cross-lane-functions)
- [HIP Cooperative Groups](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html)
- [AMD CDNA3 ISA Reference (gfx942)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [GEAK: Triton Kernel AI Agent (arXiv 2507.23194)](https://arxiv.org/abs/2507.23194)
- [GPU Programming 101 - Module 3: Thread Synchronization](https://github.com/AIComputing101/gpu-programming-101)
