---
skill_name: AMD Wavefront Cooperation for Flash Attention
description: Low-level AMD wavefront cooperation techniques for Flash Attention including Wave-64 execution, shuffle operations, wave-level softmax reduction, HIP synchronization primitives, MFMA scheduling with phase-aware barriers, direct buffer-to-LDS transfers, and ping-pong/fine-grained interleaving patterns.
level: L3 - Memory & Thread Cooperation Level
target_hardware: AMD MI250X (gfx90a), MI300X (gfx942), MI350X (gfx950)
relevance: When implementing or optimizing Flash Attention kernels on AMD CDNA GPUs, understanding wavefront-level cooperation patterns, or debugging synchronization issues in HIP attention kernels
---

# AMD Wavefront Cooperation for Flash Attention

## What It Is
A collection of low-level HIP/ROCm wavefront cooperation techniques used in AMD's Flash Attention implementations. These techniques span the Wave-64 execution model, wavefront shuffle operations for softmax reduction, HIP synchronization primitives (barriers, wait counters, schedule barriers), MFMA instruction scheduling with phase-aware barriers, direct buffer-to-LDS asynchronous transfers, and cooperation patterns (ping-pong buffering, fine-grained interleaving).

## Key Concepts

### Wave-64 Execution Model (vs NVIDIA Warp-32)
- AMD CDNA GPUs execute 64 threads per wavefront (NVIDIA uses 32 per warp)
- Lane ID: `threadIdx.x & 0x3F` (mask with 63, not 31)
- Wave ID: `threadIdx.x >> 6` (shift by 6, not 5)
- All collective operations, bank conflicts, and register allocation scale with wave size 64
- Implications for softmax: reductions must cover 64 lanes, not 32

### Wavefront Shuffle Operations
- **`__shfl_xor(val, offset, 64)`**: Butterfly pattern exchange. Each lane exchanges data with lane `(id XOR offset)`. Used for wave-level max and sum reductions in softmax.
- **`__shfl_down(val, offset, 64)`**: Each lane gets value from lane `(id + offset)`. Used for prefix-sum style reductions.
- **`__shfl_up(val, offset, 64)`**: Each lane gets value from lane `(id - offset)`. Used for inclusive prefix sums (scan).
- **`__shfl(val, lane_id, 64)`**: Broadcast from specific lane. Used for broadcasting softmax statistics.
- Width parameter is 64 (not 32 as on NVIDIA), affecting the reduction tree depth (6 steps vs 5).

### Wave-Level Reduction for Softmax
```
Butterfly reduction (max or sum) across 64 lanes:
  offset=32: exchange with lane XOR 32
  offset=16: exchange with lane XOR 16
  offset=8:  exchange with lane XOR 8
  offset=4:  exchange with lane XOR 4
  offset=2:  exchange with lane XOR 2
  offset=1:  exchange with lane XOR 1
Result: all lanes hold the reduced value (for __shfl_xor)
```
For multi-wave blocks, partial results are written to LDS (shared memory) and a final reduction is performed by the first wavefront.

### HIP Synchronization Primitives
- **`__builtin_amdgcn_s_barrier()`**: Synchronize all threads in the workgroup (equivalent to `__syncthreads`)
- **`s_waitcnt lgkmcnt(N)`**: Wait until only N LDS/GDS operations remain outstanding
- **`s_waitcnt vmcnt(N)`**: Wait until only N vector memory operations remain outstanding
- **`__builtin_amdgcn_s_waitcnt(0)`**: Wait for ALL outstanding operations
- **`__builtin_amdgcn_sched_barrier(0)`**: Prevent instruction reordering across this point

### MFMA Instruction Scheduling with Phase-Aware Barriers
The `__builtin_amdgcn_sched_group_barrier(mask, count, order)` intrinsic controls instruction interleaving:
- `0x008, 1, 0`: Allow 1 MFMA instruction
- `0x200, 2, 0`: Allow 2 TRANS instructions
- `0x002, 4, 0`: Allow 4 VALU instructions
- `0x004, 4, 0`: Allow 4 SALU instructions

Used in the FMHA V3 forward kernel to create interleaved MFMA + TRANS + VALU patterns across 4 phases and 2 wave groups.

### Direct Buffer-to-LDS Async Transfers
- Standard path: Global Memory -> VGPRs -> LDS (consumes VGPR budget)
- Optimized path: Global Memory -> LDS (direct, via `llvm_amdgcn_raw_buffer_load_lds`)
- Requires buffer resource descriptors (SRD): base pointer + range + config
- **Readfirstlane hoisting**: Convert LDS pointer to scalar ONCE before loop (10-20% speedup)

### Priority Control
```cpp
__builtin_amdgcn_s_setprio(1);  // High priority for compute
mfma_f32_16x16x32_bf16<...>();   // Critical MFMA instruction
__builtin_amdgcn_s_setprio(0);  // Back to normal
```

## Cooperation Patterns

### Ping-Pong Buffering (8-Wave)
Double buffering for large attention tiles with 8 wavefronts per workgroup:
- Two LDS buffers (tic/toc) for K/V tiles
- Load next iteration's data while computing on current
- Swap buffers after barrier synchronization

### Fine-Grained Interleaving (4-Wave)
For smaller tiles, alternate individual MFMA and memory operations:
- Cluster 0: MFMA instruction
- Schedule barrier
- Cluster 1: Buffer load (overlaps with MFMA pipeline)
- Schedule barrier
- Repeat

### Chiplet-Aware Workgroup Scheduling
MI300X has 8 XCDs (chiplets), each with its own L2 cache:
- Remap workgroup IDs so consecutive workgroups go to the same XCD
- Improves L2 locality for attention tiles
- `wgid = chiplet_transform(wgid, gridDim.x, 8)`

## When to Use
- Implementing custom attention kernels on AMD CDNA GPUs
- Optimizing existing HIP attention kernels for better wavefront utilization
- Debugging synchronization or scheduling issues in AMD Flash Attention
- Porting attention kernels from CUDA to HIP (need to adapt warp-32 patterns to wave-64)
- Understanding performance differences between AMD and NVIDIA attention implementations

## When NOT to Use
- Working exclusively on NVIDIA GPUs (use CUDA warp primitives instead)
- Using high-level APIs (PyTorch SDPA, Composable Kernel API) that abstract these details
- Writing non-performance-critical GPU code where synchronization overhead is negligible

## Source Code Examples

### Wave-64 Thread Identification

```cpp
constexpr int WAVE_SIZE = 64;
__device__ int laneid() { return threadIdx.x & 0x3F; }  // Mask with 63
__device__ int waveid() { return threadIdx.x >> 6; }    // Divide by 64
```

### Butterfly Reduction for Softmax Max (wave_reduce_max)

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

### Multi-Wave Softmax Reduction with LDS

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

### Bank Conflict Swizzling for 16x16 BF16 Tiles

```cpp
__device__ inline uint32_t swizzle_16x16_bf16(int row, int col) {
    // XOR swizzle to distribute across 32 banks
    int row_swizzle = row ^ ((col >> 2) & 0x7);
    return (row_swizzle * 16 + col) * sizeof(__hip_bfloat16);
}
```

### Direct Buffer-to-LDS Transfers with Readfirstlane Hoisting

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

### Ping-Pong Buffering Pattern

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

### Chiplet-Aware Scheduling

```cpp
__device__ inline int chiplet_transform(int wgid, int num_wgs, int num_xcds) {
    int xcd_id = wgid % num_xcds;
    int local_wg = wgid / num_xcds;
    return xcd_id * (num_wgs / num_xcds) + local_wg;
}
```

This ensures that workgroups processing adjacent Q tiles are scheduled on the same XCD, maximizing L2 cache reuse for shared K/V tiles.

## Key Takeaways
- The Wave-64 model fundamentally changes reduction patterns: 6 steps instead of 5, affecting softmax latency
- Phase-aware scheduling barriers (`sched_group_barrier`) are the primary tool for MFMA pipeline optimization on AMD
- Direct buffer-to-LDS transfers with readfirstlane hoisting provide 10-20% speedup in memory-intensive kernels
- Chiplet-aware scheduling is essential on MI300X (8 XCDs) for L2 cache locality
- The voting primitives (`__ballot`, `__any`, `__all`) return 64-bit masks on AMD vs 32-bit on NVIDIA

## References
- [HIP Programming Guide - Warp Cross-Lane Functions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/programming_manual.html)
- [AMD CDNA3 ISA Reference (gfx942)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [GEAK: Triton Kernel AI Agent (arXiv 2507.23194)](https://arxiv.org/abs/2507.23194)
