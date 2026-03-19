---
skill_name: UMMA (Unified MMA) Instruction on Blackwell via tcgen05
description: Configuring and using the Blackwell UMMA instruction (tcgen05.mma) with Tensor Memory (TMEM)-based operand sourcing, CTA-group operations, and CUTLASS SM100 abstractions for attention kernels
level: L4 - Compute Kernel Optimization Level
target_hardware: NVIDIA Blackwell B100/B200/GB200 (SM100/SM103)
relevance: When implementing FlashAttention microkernels on Blackwell that need to directly configure tcgen05.mma tile shapes, TMEM allocation, CTA-group (1-CTA or 2-CTA) modes, and instruction descriptor bit-fields for attention GEMMs
---

# UMMA (Unified MMA) Instruction on Blackwell via tcgen05

## What It Is
UMMA (Unified Matrix Multiply-Accumulate) is Blackwell's next-generation tensor core instruction, exposed in PTX as `tcgen05.mma`. It supersedes Hopper's WGMMA instruction with several architectural advances: operands and accumulators reside in dedicated Tensor Memory (TMEM) rather than shared memory or registers, the M dimension scales to 64/128/256 (versus WGMMA's fixed M=64), and CTA-group operations allow two CTAs in a cluster to cooperatively execute a single large MMA spanning both CTAs' TMEM. UMMA supports all Blackwell data types including FP16, BF16, TF32, FP8 (E4M3/E5M2), FP6 (E2M3/E3M2), and FP4 (E2M1) with MX block scaling, making it the compute primitive for FA4 and all Blackwell-optimized attention kernels.

## Key Concepts
- **Tensor Memory (TMEM):** A new addressable memory space private to each SM's tensor core unit. Unlike WGMMA where accumulators live in registers, UMMA stores both accumulators and one operand (A or the intermediate P matrix) in TMEM. Each SM has 512 columns of TMEM; allocation is managed via `tcgen05.alloc` and freed with `tcgen05.dealloc`.
- **Instruction descriptor (32-bit):** A packed bit-field encoding the MMA configuration: A/B element formats (3 bits each), accumulator format (2 bits), M dimension (5 bits: M>>4), N dimension (6 bits: N>>3), operand major mode (MN vs K), negate flags, saturation, sparsity, and max-shift for MX scaling.
- **M dimension flexibility:** M can be 64, 128, or 256 (encoded as M>>4 in the descriptor). This allows a single instruction to cover 128 or 256 query rows without tiling multiple atoms, reducing instruction overhead versus WGMMA's fixed M=64.
- **CTA-group operations:** `tcgen05.mma.cta_group::2` allows two CTAs in a 2x1 cluster to cooperatively execute one MMA. Each CTA contributes its TMEM portion; the instruction spans both. This doubles the effective M dimension per instruction (e.g., 2 CTAs x 128 rows = 256-row GEMM).
- **Operand sourcing:** Operand B comes from shared memory (SMEM) via 64-bit descriptors (similar to WGMMA). Operand A can come from either SMEM or TMEM. The intermediate P matrix (softmax output) is sourced from TMEM for the PV GEMM, eliminating the register-to-SMEM copy that WGMMA required.
- **SMEM descriptor for UMMA:** A 64-bit descriptor encoding swizzle mode, stride, leading byte offset, and layout type. Five swizzle modes are legal: SWIZZLE_NONE, SWIZZLE_32B, SWIZZLE_64B, SWIZZLE_128B, and SWIZZLE_128B_BASE32B (new for Blackwell).
- **MX format support:** The instruction descriptor includes a 2-bit `max_shift` field for MX (microscaling) block-scaled formats. When nonzero, the tensor core applies per-block E8M0 scale factors natively during the MMA, supporting MXFP8, MXFP6, and MXFP4 without software dequantization.
- **Warp-level (not warpgroup-level):** Unlike WGMMA which required a full warpgroup (128 threads, 4 warps), UMMA is dispatched by a single warp (32 threads) -- specifically the designated MMA warp in FA4's warp-specialized design.

## Tile Configuration / Code Pattern
```python
# FA4 UMMA configuration (from flash_fwd_sm100.py)
import cutlass.cute.nvgpu.tcgen05 as tcgen05

# 1. Select CTA-group mode (1-CTA or 2-CTA cooperative)
cta_group = tcgen05.CtaGroup.TWO  # 2-CTA for larger tiles
# cta_group = tcgen05.CtaGroup.ONE  # 1-CTA for standard

# 2. Specify operand major modes
q_major_mode = tcgen05.OperandMajorMode.K     # Q is K-major
k_major_mode = tcgen05.OperandMajorMode.K     # K is K-major
v_major_mode = tcgen05.OperandMajorMode.MN    # V is MN-major
p_source     = tcgen05.OperandSource.TMEM     # P comes from TMEM

# 3. Build TiledMMA for QK^T (both operands from SMEM)
m_block, n_block, head_dim = 128, 128, 128
mma_tiler_qk = (cta_group_size * m_block, n_block, head_dim)
tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
    dtype, q_major_mode, k_major_mode,
    Float32,        # accumulator in FP32
    cta_group,
    mma_tiler_qk[:2],
)

# 4. Build TiledMMA for PV (P from TMEM, V from SMEM)
mma_tiler_pv = (cta_group_size * m_block, head_dim, n_block)
tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
    dtype, p_major_mode, v_major_mode,
    Float32,
    cta_group,
    mma_tiler_pv[:2],
    p_source,  # operand A sourced from TMEM
)

# 5. TMEM allocation (in kernel, MMA warp only)
tmem = cutlass.utils.TmemAllocator(...)
tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))  # 512 cols
tmem.wait_for_alloc()
tmem_ptr = tmem.retrieve_ptr(Float32)

# 6. Build 32-bit instruction descriptor (low-level)
from flash_attn.cute.mma_sm100_desc import make_instr_desc, Major
desc = make_instr_desc(
    a_type=cutlass.BFloat16,
    b_type=cutlass.BFloat16,
    c_type=cutlass.Float32,
    M=128,                    # 64, 128, or 256
    N=128,                    # 8..256, multiple of 8
    a_major=Major.K,
    b_major=Major.K,
)
# desc is a 32-bit integer encoding all MMA parameters

# 7. SMEM layout with swizzle for UMMA
sK_layout = sm100_utils.make_smem_layout_b(
    tiled_mma_qk, mma_tiler_qk, dtype, kv_stages
)
```

## TMEM Layout for Attention
```
TMEM Column Allocation (512 columns per SM):
|-- S0 (n_block) --|-- S1 (n_block) --|-- O0 (head_dim_v) --|-- O1 (head_dim_v) --|
   0..127             128..255            256..383               384..511

S stage 0: attention scores for query tile 0
S stage 1: attention scores for query tile 1
O stage 0: accumulated output for query tile 0
O stage 1: accumulated output for query tile 1

P (softmax output) reuses S columns with offset:
  P0 at S0 + n_block/2, P1 at S1 + n_block/2
  (P is written into the upper half of S after softmax consumes the lower half)
```

## Performance Impact
- TMEM eliminates the register pressure bottleneck: accumulators no longer consume register file entries, freeing registers for softmax computation and other non-MMA work
- CTA-group::2 mode doubles effective M per instruction, reducing instruction count for large query tiles
- Single-warp dispatch (vs WGMMA's warpgroup) enables finer-grained warp specialization with more warps available for softmax, correction, and epilogue roles
- Operand A from TMEM for PV GEMM removes the SMEM write + descriptor setup path that WGMMA required for the P matrix
- FA4 achieves ~20% speedup over cuDNN on Blackwell, with UMMA as the core compute primitive
- B200 delivers ~2x tensor core throughput over H100 in BF16/FP16, amplified further with FP8/FP4

## When to Use
- Implementing FlashAttention forward/backward passes on Blackwell (B100/B200/GB200)
- Building fused attention kernels that need fine-grained control over MMA tile shapes and operand sourcing
- When the intermediate P matrix should stay in TMEM to avoid register or SMEM round-trips
- Targeting CTA-group::2 mode for large query tile sizes (256+ rows across 2 CTAs)
- When using MX-scaled low-precision formats (MXFP8, MXFP4) where the hardware applies block scaling natively

## When NOT to Use
- On Hopper GPUs (H100/H200) -- use WGMMA instead; tcgen05 instructions are not available
- On Ampere or older GPUs -- use HMMA/WMMA
- When using high-level APIs (torch.compile, Triton) that abstract away instruction selection
- For non-MMA operations (element-wise, reductions) that do not use tensor cores
- When kernel complexity is not justified (e.g., short sequences where cuDNN or torch SDPA suffice)

## Key Takeaways
- UMMA's M dimension flexibility (64/128/256) is a major upgrade from WGMMA's fixed M=64, reducing tiling overhead for large attention tiles
- The 32-bit instruction descriptor encodes all MMA parameters in a single integer, including data types, dimensions, layout modes, and MX scaling -- understanding its bit-fields is essential for debugging and tuning
- TMEM is the defining architectural innovation: it provides a dedicated memory space for MMA operands and accumulators that is neither registers nor shared memory, enabling the warp specialization design of FA4
- CTA-group::2 mode requires a 2x1 cluster configuration and doubles the M dimension per instruction, but adds synchronization complexity between the cooperating CTAs
- The SMEM descriptor format for UMMA adds the SWIZZLE_128B_BASE32B mode (not present in WGMMA) for finer-grained bank conflict avoidance
- FA4's architecture (1 MMA warp, 8 softmax warps, 4 correction warps, load + epilogue warps) is specifically designed around UMMA's single-warp dispatch model

## References
- [NVIDIA PTX ISA: tcgen05.mma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUTLASS SM100 MMA Descriptor (mma_sm100_desc.hpp)](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp)
- [CUTLASS Blackwell FMHA Example](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha)
- [Flash Attention 4 Source (flash_fwd_sm100.py)](https://github.com/Dao-AILab/flash-attention)
- [Modal Blog: Reverse Engineering Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4)
- [ThunderKittens 2.0 TCGEN05 Support](https://github.com/HazyResearch/ThunderKittens)
