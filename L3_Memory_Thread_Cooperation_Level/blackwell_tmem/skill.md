---
skill_name: Blackwell Tensor Memory (TMEM) for Attention Kernels
description: Understanding and using NVIDIA Blackwell's dedicated Tensor Memory (TMEM) for MMA accumulators and pipeline state in Flash Attention kernels
level: L3 - Memory & Thread Cooperation Level
target_hardware: NVIDIA Blackwell B200/B100 (SM100)
relevance: When designing attention kernels on Blackwell GPUs, understanding FA4's pipeline architecture, or deciding between TMEM and registers for accumulator storage
---

# Blackwell Tensor Memory (TMEM) for Attention Kernels

## What It Is
Tensor Memory (TMEM) is a new memory level introduced in NVIDIA Blackwell (SM100) GPUs, physically located in the L1 cache area but dedicated to serving as operand storage for 5th-generation Tensor Cores. TMEM provides a 128-row x 256-column (32 KB per SM) addressable buffer that holds MMA accumulators, intermediate score matrices, and probability tiles without consuming the general-purpose register file (RMEM). In Flash Attention 4, TMEM is the critical enabler for the five-warp-role pipeline architecture: it allows the MMA warpgroup's accumulators to be accessed by the separate correction warpgroup for rescaling, and it supports double-buffering of score/probability matrices between the MMA and softmax stages. Without TMEM, FA4's decoupled rescaling and deep pipelining would be impossible due to register pressure constraints that plagued FA3 on Hopper.

## Key Concepts
- **Physical location**: TMEM occupies a dedicated partition of the L1 cache/register file area on each SM. It is not part of shared memory (SMEM) and not part of the general-purpose register file (RMEM). It has its own addressing and access hardware.
- **Capacity**: 128 rows x 256 columns of 32-bit values = 32 KB per SM. Each row corresponds to a thread in a warpgroup (128 threads = 1 warpgroup), and each thread can address up to 256 columns.
- **Allocation/deallocation**: TMEM is allocated and freed explicitly via `tcgen05.alloc` and `tcgen05.dealloc` PTX instructions. Allocation returns a base descriptor (TMEM address) that subsequent MMA instructions reference.
- **MMA integration**: Blackwell's `tcgen05.mma` instructions read inputs from SMEM or TMEM and write outputs directly to TMEM. This replaces Hopper's pattern where WGMMA accumulates to registers (RMEM).
- **Cross-warpgroup access**: Unlike registers which are private to a warpgroup, TMEM can be accessed by other warpgroups in the same CTA through explicit TMEM load/store instructions. This is what enables FA4's correction warpgroup to rescale the MMA warpgroup's accumulators.
- **Register pressure relief**: On Hopper, FA3 needed FP32 accumulators (O, S, m, l) plus pipelined state in registers, consuming 128-200+ registers per thread and limiting occupancy. TMEM moves accumulators off the register file entirely.
- **Double-buffering in TMEM**: FA4 allocates two S (score) tiles and four P (probability) tiles in TMEM, enabling one tile to be computed while the other is consumed. This overlaps MMA and softmax pipeline stages.
- **TMEM vs SMEM vs RMEM**: TMEM is purpose-built for tensor core data with warpgroup-aligned addressing; SMEM is for general thread-block communication; RMEM is for per-thread scalar/vector data. TMEM access latency is similar to registers for MMA operations but requires explicit management.

## Memory Layout / Data Flow
```
BLACKWELL SM100 MEMORY HIERARCHY FOR ATTENTION:

HBM3e (global memory)
  |
  TMA (Tensor Memory Accelerator)
  |
SMEM (shared memory, 228+ KB per SM)
  |-- K, V tiles loaded by TMA
  |-- Q tile loaded once
  |
TMEM (tensor memory, 32 KB per SM)           RMEM (registers, 256 per thread)
  |-- S tiles [2]: QK^T scores               |-- Softmax state (m, l) per row
  |-- P tiles [4]: exp(S-m) probabilities     |-- Control flow variables
  |-- O tiles [2]: output accumulators        |-- Address computation
  |
  5th-gen Tensor Cores (tcgen05.mma)

TMEM ALLOCATION IN FA4 FORWARD:

TMEM Layout (128 rows x 256 cols, 32-bit):
+------------------------------------------+
| O_tile_0  (128 x d)   | O_tile_1 (128 x d) |   Output accumulators
+------------------------------------------+
| S_tile_0  (128 x 128) | S_tile_1 (128 x 128)|  Score matrices
+------------------------------------------+
| P_tile_0a (128 x 128) | P_tile_0b (128x128) |  Probability tiles
| P_tile_1a (128 x 128) | P_tile_1b (128x128) |  (double-buffered x2 WG)
+------------------------------------------+

DATAFLOW THROUGH TMEM:

1. MMA warp: Q (SMEM) x K^T (SMEM) -> S (TMEM)   [tcgen05.mma]
2. Softmax warps: read S (TMEM) -> compute P -> write P (TMEM)
3. MMA warp: P (TMEM) x V (SMEM) -> O (TMEM)      [tcgen05.mma]
4. Correction warps: read O (TMEM) -> rescale -> write O (TMEM)
5. Epilogue: read O (TMEM) -> normalize -> write to HBM via TMA

KEY ADVANTAGE: Steps 2 and 4 are performed by DIFFERENT warpgroups
than step 1/3. This is only possible because TMEM supports
cross-warpgroup access, unlike registers.

PTX EXAMPLE:
    // Allocate TMEM for accumulator
    tcgen05.alloc tmem_addr, num_cols;

    // MMA: sources from SMEM, accumulates in TMEM
    tcgen05.mma.cta_group::1 tmem_addr, smem_desc_A, smem_desc_B;

    // Cross-warpgroup access: correction warp reads/writes TMEM
    tcgen05.ld.16x64b result, [tmem_addr + offset];
    // ... rescale ...
    tcgen05.st.16x64b [tmem_addr + offset], rescaled;

    // Deallocate when done
    tcgen05.dealloc tmem_addr, num_cols;
```

## Performance Impact
- **FA4 achieves 1613 TFLOPs/s (71% utilization)** on B200 -- TMEM enables the deep pipeline that makes this possible
- **Register pressure reduction**: Moves 128+ FP32 accumulator registers per thread to TMEM, freeing registers for softmax state and control flow
- **Enables 5-warp-role architecture**: Without TMEM, the correction warpgroup cannot access MMA accumulators (they would be in private registers)
- **Double-buffering of S/P tiles**: Enables MMA and softmax to overlap without register contention
- **~20% speedup over cuDNN**: Attributed partly to TMEM-enabled pipeline depth that cuDNN's more conservative approach does not exploit
- **Immediate pipeline startup**: Two S tiles can be computed before any softmax begins, hiding pipeline fill latency

## When to Use
- Designing attention kernels targeting NVIDIA Blackwell B200/B100 GPUs
- When register pressure limits occupancy or pipeline depth on attention kernels
- When the kernel design requires cross-warpgroup access to MMA accumulators (e.g., decoupled rescaling, correction warps)
- Implementing multi-stage pipelines where MMA output must be consumed by a different warpgroup
- When porting FA3-style kernels to Blackwell and wanting to exploit the deeper pipeline opportunities TMEM enables
- Building custom CUTLASS SM100 kernels that use `tcgen05.mma`

## When NOT to Use
- Targeting Hopper (SM90) or earlier GPUs -- TMEM does not exist; use register-resident accumulators with WGMMA
- Simple kernels where a single warpgroup handles all stages (no cross-warpgroup access needed)
- When SMEM-resident accumulators with explicit loads/stores would suffice (e.g., small tile sizes where register pressure is not an issue)
- Prototyping in Triton or high-level frameworks that abstract away memory management (TMEM is only accessible via PTX/CUTLASS)
- Kernels that do not use tensor core MMA (TMEM is specifically for tensor core operands)

## Key Takeaways
- TMEM is Blackwell's answer to the register pressure problem that plagued attention kernel pipelining on Hopper -- it provides dedicated storage for MMA accumulators outside the general register file
- The critical capability is cross-warpgroup access: TMEM accumulators can be read and modified by warpgroups other than the one that computed them, enabling FA4's decoupled correction warpgroup
- TMEM requires explicit allocation/deallocation via PTX instructions -- it is not automatically managed like registers
- The 32 KB capacity per SM is carefully budgeted in FA4: two O tiles, two S tiles, and four P tiles fill the available space
- TMEM fundamentally changes kernel design patterns: on Hopper, all pipeline stages accessing an accumulator must be in the same warpgroup; on Blackwell with TMEM, stages can be split across warpgroups
- Using TMEM currently requires PTX-level programming or CUTLASS SM100 abstractions -- it is not yet exposed in Triton or other high-level kernel DSLs

## References
- [FlashAttention-4 Paper (Dao et al., 2026)](https://arxiv.org/abs/2603.05451)
- [Reverse Engineering Flash Attention 4 - Modal Blog](https://modal.com/blog/reverse-engineer-flash-attention-4)
- [NVIDIA Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA PTX ISA: tcgen05 Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUTLASS SM100 Source Code](https://github.com/NVIDIA/cutlass)
