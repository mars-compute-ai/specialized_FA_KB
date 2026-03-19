# Blackwell Tensor Memory (TMEM) for Attention Kernels

## Overview

Tensor Memory (TMEM) is a new level in the GPU memory hierarchy introduced with NVIDIA Blackwell (SM100) architecture. It provides a dedicated, explicitly managed buffer within the SM's L1 cache area that serves as operand and accumulator storage for 5th-generation Tensor Cores (tcgen05). Unlike registers, TMEM supports cross-warpgroup access; unlike shared memory (SMEM), it is optimized for tensor core data flow with warpgroup-aligned addressing. TMEM is the critical hardware enabler for FlashAttention-4's five-warp-role pipeline architecture, where separate warpgroups for MMA computation, softmax normalization, and output correction must all access the same accumulator data.

This document covers TMEM's architecture, programming model, how FA4 uses it, and design implications for custom attention kernels on Blackwell.

---

## 1. Blackwell SM100 Memory Hierarchy

### 1.1 Full Memory Hierarchy

```
+--------------------------------------------------------------------+
|                      Blackwell SM100 SM                             |
|                                                                     |
|  Registers (RMEM)    | 256 per thread, private to each thread      |
|  ~256 KB per SM      | Fast: 1 cycle latency                       |
|                      | Private: no cross-warpgroup access           |
|                                                                     |
|  Tensor Memory (TMEM)| 128 rows x 256 cols x 32-bit = 32 KB       |
|  32 KB per SM        | Medium: ~2-4 cycle latency for MMA ops      |
|                      | Shared: cross-warpgroup accessible          |
|                      | Purpose: MMA accumulators + operands         |
|                                                                     |
|  Shared Memory (SMEM)| 228+ KB per SM (configurable L1/SMEM split) |
|                      | Medium: ~20 cycle latency                    |
|                      | Shared: all threads in CTA can access        |
|                      | Purpose: tile staging, communication          |
|                                                                     |
|  L2 Cache            | 96 MB total (shared across all SMs)         |
|                      | High: ~100 cycle latency                     |
|                                                                     |
|  HBM3e               | 192-288 GB, up to 8 TB/s bandwidth          |
|                      | Very high: ~300+ cycle latency               |
+--------------------------------------------------------------------+
```

### 1.2 TMEM vs Other Memory Levels

| Property | RMEM (Registers) | TMEM | SMEM | HBM |
|----------|------------------|------|------|-----|
| Capacity per SM | ~256 KB | 32 KB | 228+ KB | N/A (global) |
| Access latency | 1 cycle | ~2-4 cycles (MMA path) | ~20 cycles | ~300+ cycles |
| Addressable by | Owning thread only | Any warpgroup in CTA | Any thread in CTA | Any thread |
| Managed by | Compiler (automatic) | Programmer (explicit alloc/dealloc) | Programmer (static/dynamic) | Runtime + programmer |
| Primary purpose | Scalar/vector ops, control flow | MMA accumulation, tensor operands | Tile staging, inter-thread communication | Bulk data storage |
| Cross-warpgroup | No | Yes | Yes | Yes (via atomics) |
| Addressing model | Per-thread register number | (row, col) = (thread_in_warpgroup, col_offset) | Flat byte address | Virtual address |

### 1.3 Why TMEM is Needed

On Hopper (SM90), MMA accumulators live in registers (RMEM). This creates two fundamental problems for deep attention pipelining:

**Problem 1: Register pressure**
```
FA3 register budget per consumer warpgroup thread (FP32, d=128):
  O accumulator:        128 / 4 * 4 = 128 registers (128 FP32 values across 4 warps)
  S scores (pipelined): 32-64 registers (one or two score tiles)
  m, l statistics:      2-4 registers
  P probabilities:      32-64 registers (for RS-WGMMA input)
  Control/addressing:   8-16 registers
  Total:                ~200-270 registers per thread

  Maximum registers per thread on Hopper: 256
  Result: Barely fits, limits occupancy, prevents deeper pipelining
```

**Problem 2: No cross-warpgroup access to accumulators**
```
On Hopper:
  Warpgroup 0 computes O += P @ V  (O in warpgroup 0's registers)
  Warpgroup 1 wants to rescale O    (cannot access warpgroup 0's registers!)

  Workaround: rescaling must happen in the SAME warpgroup that owns O
  Consequence: cannot decouple rescaling into a separate pipeline stage
```

TMEM solves both problems:
1. Accumulators move to TMEM, freeing ~128-200 registers per thread
2. Any warpgroup can read/write TMEM, enabling decoupled rescaling

---

## 2. TMEM Architecture Details

### 2.1 Physical Organization

TMEM is organized as a 2D array with warpgroup-aligned row addressing:

```
TMEM Physical Layout:
  128 rows x 256 columns x 32 bits

  Row index = thread index within warpgroup (0-127)
  Column index = programmer-specified offset (0-255)

  Each thread "owns" one row and can access columns 0..255
  But other warpgroups can also access any (row, col) via TMEM load/store

  Total capacity: 128 * 256 * 4 bytes = 128 KB (theoretical)
  Effective capacity: 32 KB per SM (hardware limit, shared across warpgroups)
```

### 2.2 TMEM Addressing

```
Thread 0 of warpgroup -> TMEM row 0
Thread 1 of warpgroup -> TMEM row 1
...
Thread 127 of warpgroup -> TMEM row 127

Within each row, columns are explicitly addressed:
  tmem_addr = base + col_offset

Warpgroup-aligned: the 128-row structure matches exactly one warpgroup (4 warps x 32 threads)
```

### 2.3 Allocation and Deallocation

TMEM is explicitly managed via PTX instructions:

```
// Allocate TMEM region
tcgen05.alloc tmem_base, num_columns;
// Returns tmem_base: descriptor for the allocated region
// num_columns: number of 32-bit columns to allocate (must be multiple of 16)

// Deallocate TMEM region
tcgen05.dealloc tmem_base, num_columns;
// Frees the allocated region

// Important constraints:
// - Allocation is per-SM, shared across all CTAs on that SM
// - Multiple allocations can coexist (if total fits in 32 KB)
// - Deallocation must match the allocation exactly
// - No garbage collection -- programmer must manage explicitly
```

### 2.4 TMEM Access Instructions

```
// MMA instruction: read from SMEM, accumulate in TMEM
tcgen05.mma.cta_group::1 tmem_acc_addr, smem_desc_A, smem_desc_B;
// Performs: TMEM[acc] += A (from SMEM) @ B (from SMEM)
// cta_group::1 means one warpgroup participates

// MMA with TMEM operand (P @ V where P is in TMEM)
tcgen05.mma.cta_group::1 tmem_O_addr, tmem_P_addr, smem_desc_V;
// Performs: O (TMEM) += P (TMEM) @ V (SMEM)

// Explicit TMEM load (for cross-warpgroup access)
tcgen05.ld.16x64b result_regs, [tmem_addr + offset];
// Loads a 16x64-byte (16 rows x 16 columns) tile from TMEM to registers
// Used by correction warpgroup to read MMA warpgroup's accumulators

// Explicit TMEM store
tcgen05.st.16x64b [tmem_addr + offset], source_regs;
// Stores from registers to TMEM
// Used by correction warpgroup to write back rescaled accumulators
```

---

## 3. TMEM Usage in FlashAttention-4

### 3.1 TMEM Allocation Map

FA4 allocates TMEM for the following data structures:

```
FA4 TMEM Budget (d=128, tile_M=128, tile_N=128):

+------------------------------------------------------------------+
| Data Structure     | Size (elements)  | Size (bytes) | Purpose    |
+------------------------------------------------------------------+
| O_tile[0]          | 128 x 128 FP32  | 64 KB*      | Output acc 0|
| O_tile[1]          | 128 x 128 FP32  | 64 KB*      | Output acc 1|
| S_tile[0]          | 128 x 128 FP32  | 64 KB*      | Scores 0    |
| S_tile[1]          | 128 x 128 FP32  | 64 KB*      | Scores 1    |
| P_tile[0a]         | 128 x 128 BF16  | 32 KB*      | Probs 0a    |
| P_tile[0b]         | 128 x 128 BF16  | 32 KB*      | Probs 0b    |
| P_tile[1a]         | 128 x 128 BF16  | 32 KB*      | Probs 1a    |
| P_tile[1b]         | 128 x 128 BF16  | 32 KB*      | Probs 1b    |
+------------------------------------------------------------------+

* Note: These are logical sizes. The actual TMEM allocation is organized
  by (row, column) addressing and may overlap temporally through
  the pipeline schedule. Not all tiles are live simultaneously.

The 32 KB physical TMEM is managed through careful lifetime analysis:
  - At any given time, only a subset of tiles are live
  - S_tile is consumed (read by softmax) before the next S_tile is produced
  - P_tile is consumed (read by MMA for O update) before its buffer is reused
  - Double-buffering ensures producer and consumer never access the same tile
```

### 3.2 FA4 Pipeline Through TMEM

```
Timeline of TMEM usage in FA4 forward pass:

Iteration j:
  MMA warp:     Q(SMEM) x K_j^T(SMEM) -> S_tile[j%2] (TMEM)
                       ^                       |
                       |                       v
  Softmax warps:       |              Read S_tile[j%2] (TMEM)
                       |              Compute exp(S - m) -> P_tile
                       |              Write P_tile[j%2][a/b] (TMEM)
                       |                       |
                       |                       v
  MMA warp:            |              P_tile[j%2](TMEM) x V_j(SMEM) -> O_tile[wg] (TMEM)
                       |                       |
                       |                       v
  Correction warps:    |              If rescale needed:
                       |                Read O_tile[wg] (TMEM)
                       |                O = O * exp(m_old - m_new)
                       |                Write O_tile[wg] (TMEM)

Key observation: Steps 2, 3, and 4 involve DIFFERENT warpgroups
accessing the SAME TMEM locations. This is impossible with registers.

TMEM Access Pattern Per Pipeline Stage:
  +----------+--------+--------+----------+--------+
  | Stage    | Writes | Reads  | Warpgroup| Notes  |
  +----------+--------+--------+----------+--------+
  | QK^T MMA | S_tile | Q,K    | MMA WG   | tcgen05.mma |
  | Softmax  | P_tile | S_tile | SM warps | ld/st TMEM  |
  | PV MMA   | O_tile | P_tile,V| MMA WG  | tcgen05.mma |
  | Correct  | O_tile | O_tile | Corr WG  | ld/st TMEM  |
  | Epilogue | --     | O_tile | Epi WG   | ld TMEM     |
  +----------+--------+--------+----------+--------+
```

### 3.3 How TMEM Enables Decoupled Correction

The key FA4 innovation enabled by TMEM is threshold-based correction in a separate warpgroup:

```
Without TMEM (Hopper FA3):
  Consumer warpgroup must do both MMA and rescaling:
    O_new = exp(m_old - m_new) * O_old + P @ V
  Rescaling is on the critical path of every iteration.

With TMEM (Blackwell FA4):
  MMA warpgroup:  O += P @ V           (O in TMEM)
  Correction WG:  if |m_new - m_old| > threshold:
                    O = O * correction_factor  (reads/writes TMEM)

  The correction warpgroup runs asynchronously, only when needed (~10% of iterations).
  Rescaling is OFF the critical path for 90% of iterations.
```

This decoupling reduces the effective cost of rescaling by ~10x, which is a major contributor to FA4's 20% speedup over cuDNN.

---

## 4. TMEM Programming Patterns

### 4.1 Basic TMEM Accumulation

```cuda
// Pseudo-CUDA with PTX intrinsics for TMEM usage

__device__ void attention_with_tmem() {
    // Allocate TMEM for output accumulator
    uint32_t tmem_O;
    asm volatile("tcgen05.alloc %0, 128;" : "=r"(tmem_O));  // 128 columns

    // Allocate TMEM for score matrix
    uint32_t tmem_S;
    asm volatile("tcgen05.alloc %0, 128;" : "=r"(tmem_S));  // 128 columns

    // Zero-initialize O accumulator in TMEM
    // (use tcgen05.st to write zeros)

    // Main loop over K/V tiles
    for (int j = 0; j < num_kv_tiles; j++) {
        // TMA load K_j, V_j to SMEM (same as FA3)
        // ...

        // MMA: S = Q @ K_j^T, result in TMEM
        asm volatile(
            "tcgen05.mma.cta_group::1 %0, %1, %2;"
            : "+r"(tmem_S)
            : "l"(smem_desc_Q), "l"(smem_desc_K)
        );

        // Signal softmax warps: "S is ready in TMEM"
        // Softmax warps read S from TMEM, compute P, write P to TMEM
        // ...

        // MMA: O += P @ V, accumulate in TMEM
        asm volatile(
            "tcgen05.mma.cta_group::1 %0, %1, %2;"
            : "+r"(tmem_O)
            : "r"(tmem_P), "l"(smem_desc_V)
        );
    }

    // Read final O from TMEM to registers for epilogue
    float O_regs[HEAD_DIM_PER_THREAD];
    asm volatile(
        "tcgen05.ld.16x64b {%0, %1, ...}, [%2];"
        : "=f"(O_regs[0]), "=f"(O_regs[1]) /* ... */
        : "r"(tmem_O)
    );

    // Normalize and store to HBM
    // ...

    // Deallocate TMEM
    asm volatile("tcgen05.dealloc %0, 128;" :: "r"(tmem_O));
    asm volatile("tcgen05.dealloc %0, 128;" :: "r"(tmem_S));
}
```

### 4.2 Cross-Warpgroup Correction Pattern

```cuda
// MMA warpgroup: accumulates O in TMEM
__device__ void mma_warp_role(uint32_t tmem_O, uint32_t tmem_S, uint32_t tmem_P) {
    for (int j = 0; j < num_kv_tiles; j++) {
        // S = Q @ K_j^T -> TMEM
        tcgen05_mma(tmem_S, smem_Q, smem_K[j]);

        // Signal softmax
        barrier_arrive(softmax_ready_barrier[j]);
        barrier_wait(P_ready_barrier[j]);

        // O += P @ V -> TMEM
        tcgen05_mma(tmem_O, tmem_P, smem_V[j]);

        // Signal correction (if softmax detected large max change)
        if (needs_correction[j]) {
            barrier_arrive(correction_needed_barrier[j]);
        }
    }
}

// Correction warpgroup: rescales O when needed
__device__ void correction_warp_role(uint32_t tmem_O, float* m_history) {
    for (int j = 0; j < num_kv_tiles; j++) {
        // Only activate when correction is needed (~10% of iterations)
        if (barrier_try_wait(correction_needed_barrier[j])) {
            float m_old = m_history[j-1];
            float m_new = m_history[j];
            float alpha = exp2f(m_old - m_new);

            // Read O from TMEM (cross-warpgroup access!)
            float O_local[TILE_SIZE];
            tcgen05_ld(O_local, tmem_O, offset);

            // Rescale
            for (int i = 0; i < TILE_SIZE; i++) {
                O_local[i] *= alpha;
            }

            // Write back to TMEM
            tcgen05_st(tmem_O, offset, O_local);

            // Signal that correction is complete
            barrier_arrive(correction_done_barrier[j]);
        }
    }
}
```

### 4.3 CUTLASS SM100 Abstractions

CUTLASS 3.x provides higher-level abstractions for TMEM:

```cpp
// CUTLASS TmemAllocation manages TMEM lifetime
using TmemAccumulator = cutlass::TmemAllocation<
    ElementAccumulator,   // float (FP32)
    TileShape_M,          // 128
    TileShape_N           // 128 (head_dim or tile_N)
>;

// In kernel:
TmemAccumulator tmem_O;
tmem_O.allocate();  // wraps tcgen05.alloc

// MMA accumulates directly to TMEM
TiledMma mma;
mma.accumulate(tmem_O, smem_A, smem_B);  // wraps tcgen05.mma

// Cross-warpgroup read
auto O_fragment = tmem_O.load<CorrectionWarpgroup>();

// Modify and write back
element_wise_scale(O_fragment, correction_factor);
tmem_O.store<CorrectionWarpgroup>(O_fragment);

tmem_O.deallocate();  // wraps tcgen05.dealloc
```

---

## 5. TMEM Design Implications for Custom Kernels

### 5.1 When to Use TMEM vs Registers

| Scenario | Use TMEM | Use Registers |
|----------|----------|---------------|
| MMA accumulators with cross-WG access | Yes | No (impossible) |
| MMA accumulators, single WG kernel | Either | Simpler |
| Softmax statistics (m, l) | No | Yes (per-thread, no cross-WG need) |
| Control flow variables | No | Yes (scalar, per-thread) |
| Small intermediate values | No | Yes (faster access) |
| Large accumulator tiles (d >= 128) | Yes (frees registers) | Possible but limits occupancy |
| Double-buffered score/prob tiles | Yes | Possible but doubles register pressure |

### 5.2 TMEM Capacity Planning

```
Available TMEM: 32 KB per SM = 8192 FP32 values
                             = 128 rows x 64 FP32 columns

Planning for attention kernel (d=128):

Option A: Minimal TMEM (single-buffered)
  O accumulator: 128 x 128 FP32 = 64 KB -> EXCEEDS 32 KB!

  Solution: Only part of O is in TMEM at once. Process O in column chunks:
    O_chunk: 128 x 32 FP32 = 16 KB  (4 chunks to cover d=128)
    S_tile:  128 x 32 FP32 = 16 KB  (partial score matrix)
    Total: 32 KB -- fits exactly

Option B: FA4 approach (partial overlap, careful scheduling)
  Leverage pipeline schedule so not all tiles are live simultaneously:
    Phase 1: S_tile + P_tile active
    Phase 2: P_tile + O_tile active
    Overlap: S_tile freed before O_tile needed -> share TMEM slots

Option C: Smaller tile sizes
  tile_M=64:
    O: 64 x 128 FP32 = 32 KB
    S: 64 x 64 FP32 = 16 KB
    P: 64 x 64 BF16 = 8 KB
    Total: 56 KB -> still too much for single-buffered
    With overlap: feasible
```

### 5.3 TMEM and Occupancy

Unlike registers, TMEM does not directly limit occupancy (warps per SM). However:
- TMEM allocation is per-SM: if one CTA allocates all 32 KB, another CTA on the same SM cannot allocate TMEM
- In practice, FA4 uses one CTA per SM (large CTA with multiple warpgroups), so TMEM is not shared across CTAs
- The register savings from using TMEM can improve occupancy by reducing RMEM pressure

### 5.4 Transitioning from Hopper to Blackwell

```
FA3 (Hopper) -> FA4 (Blackwell) changes enabled by TMEM:

1. Accumulators: RMEM -> TMEM
   - Frees 128+ registers per thread
   - Enables deeper pipeline (more in-flight tiles)

2. Rescaling: Same warpgroup (inline) -> Separate correction warpgroup
   - Decouples correction from MMA critical path
   - Enables threshold-based lazy correction (10x fewer corrections)

3. Pipeline depth: 2-stage -> 5-stage
   - More pipeline stages feasible without register spilling
   - MMA, softmax, correction, load, epilogue all overlap

4. Score/probability buffering: Register-resident -> TMEM-resident
   - S and P tiles in TMEM, accessible by both MMA and softmax warpgroups
   - Double-buffering in TMEM replaces register double-buffering
```

---

## 6. Comparison with Alternative Approaches

### 6.1 SMEM-Resident Accumulators (No TMEM)

Some kernels use shared memory for accumulators instead of registers or TMEM:

```
Approach: Store O accumulator in SMEM
  Pro: Cross-thread/cross-warpgroup access (like TMEM)
  Pro: Available on all architectures (Ampere, Hopper, Blackwell)
  Con: Much higher latency (~20 cycles vs ~2-4 cycles)
  Con: Consumes SMEM budget needed for K/V tiles
  Con: Bank conflicts on accumulator access patterns
  Con: Not integrated with tensor core pipeline
```

TMEM is strictly better than SMEM for accumulator storage because it has lower latency and is directly on the tensor core data path.

### 6.2 Register-Resident with spill/reload

On Hopper, when register pressure is too high, the compiler spills to SMEM:

```
Approach: Compiler-managed spill to local memory or SMEM
  Pro: Transparent to programmer
  Con: Unpredictable performance (compiler decides what to spill)
  Con: Spill/reload adds latency to critical path
  Con: Cannot control cross-warpgroup access pattern
```

TMEM provides explicit, programmer-controlled placement with guaranteed access patterns.

---

## 7. Known Limitations and Caveats

### 7.1 Programming Complexity

TMEM currently requires PTX-level programming or CUTLASS SM100 abstractions. It is not yet exposed in:
- Triton (as of March 2026)
- PyTorch custom CUDA extensions (without inline PTX)
- Standard CUDA C++ without PTX intrinsics

### 7.2 Capacity Constraints

32 KB per SM is relatively small. For head dimensions d > 128, TMEM cannot hold a full accumulator row, requiring chunked processing or mixed TMEM/register accumulation.

### 7.3 Single-CTA Assumption

FA4's TMEM usage assumes one CTA per SM (the CTA is large enough to fill the SM). If multiple smaller CTAs share an SM, TMEM partitioning becomes more complex.

### 7.4 No Persistence Across Kernels

TMEM is allocated and deallocated within a kernel. Unlike persistent L2 cache partitions, TMEM state cannot be preserved across kernel launches.

---

## 8. Summary: TMEM Decision Framework

```
Decision tree for attention kernel accumulator storage:

Q: Targeting Blackwell (SM100)?
├── No -> Use registers (RMEM) for accumulators
│         Apply FA3-style pipelining with register reallocation
│
└── Yes -> Q: Need cross-warpgroup access to accumulators?
           ├── No -> Either TMEM or registers works
           │         Use TMEM if register pressure is high
           │         Use registers if kernel is simple (single warpgroup)
           │
           └── Yes -> Must use TMEM
                      Design patterns:
                      1. Decoupled correction (FA4 pattern)
                      2. Multi-stage producer-consumer pipeline
                      3. Warpgroup-specialized softmax + MMA overlap
```

---

## References

- [FlashAttention-4 Paper (Dao et al., 2026)](https://arxiv.org/abs/2603.05451)
- [Tri Dao's FA4 Blog Post](https://tridao.me/blog/2026/flash4/)
- [Reverse Engineering Flash Attention 4 (Modal Blog)](https://modal.com/blog/reverse-engineer-flash-attention-4)
- [NVIDIA Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA PTX ISA: tcgen05 Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUTLASS SM100 Source](https://github.com/NVIDIA/cutlass)
- [Together AI Blog: FlashAttention-4](https://www.together.ai/blog/flashattention-4)
- [Princeton AI Lab: FlashAttention-4](https://blog.ai.princeton.edu/2026/03/12/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
