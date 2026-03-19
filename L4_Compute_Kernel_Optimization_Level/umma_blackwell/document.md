# UMMA (Unified MMA) on NVIDIA Blackwell: Architecture, Instruction Format, and Attention Kernel Integration

**Sources:** NVIDIA PTX ISA documentation, CUTLASS SM100 MMA descriptor headers, Flash Attention 4 source code (flash_fwd_sm100.py, mma_sm100_desc.py), CUTLASS Blackwell FMHA example (example 77)

## Overview

UMMA (Unified Matrix Multiply-Accumulate) is the tensor core instruction for NVIDIA's Blackwell architecture (SM100/SM103), exposed in the PTX ISA as `tcgen05.mma`. It replaces Hopper's WGMMA instruction with fundamental architectural changes: a dedicated Tensor Memory (TMEM) for operands and accumulators, flexible M dimensions (64/128/256 vs. WGMMA's fixed 64), CTA-group cooperative execution across cluster members, single-warp dispatch, and native support for MX block-scaled sub-byte formats. UMMA is the compute backbone of Flash Attention 4 (FA4) and all Blackwell-optimized attention kernels.

## Architectural Evolution: WGMMA to UMMA

### WGMMA on Hopper (SM90) -- The Predecessor

On Hopper, WGMMA operates at the warpgroup level (128 threads = 4 warps). Key characteristics:
- **Operand A:** From shared memory (SS variant) or registers (RS variant)
- **Operand B:** Always from shared memory via 64-bit matrix descriptors
- **Accumulator C:** Always in registers, distributed across all 128 threads in a Z-pattern
- **Tile shape:** Fixed M=64, N=8..256 (multiples of 8), K=16 for FP16
- **Dispatch:** Requires full warpgroup participation; warp rank must be multiple of 4

### UMMA on Blackwell (SM100) -- The Successor

UMMA introduces several paradigm shifts:

| Aspect | WGMMA (Hopper) | UMMA (Blackwell) |
|--------|----------------|-------------------|
| Dispatch unit | Warpgroup (128 threads, 4 warps) | Single warp (32 threads) |
| Accumulator location | Registers (Z-pattern across 128 threads) | Tensor Memory (TMEM) |
| Operand A options | SMEM (SS) or Registers (RS) | SMEM or TMEM |
| M dimension | Fixed at 64 | 64, 128, or 256 |
| CTA cooperation | N/A | CTA-group::1 or CTA-group::2 |
| MX block scaling | Not supported | Native (max_shift field) |
| Sub-byte formats | FP8 only (E4M3, E5M2) | FP8, FP6 (E2M3, E3M2), FP4 (E2M1) |
| Instruction encoding | Per-variant PTX mnemonics | 32-bit descriptor + PTX |
| SMEM swizzle modes | 4 modes (none, 32B, 64B, 128B) | 5 modes (+128B_BASE32B) |

## Tensor Memory (TMEM)

### What Is TMEM

Tensor Memory is a new addressable memory space introduced on Blackwell, private to each SM's tensor core unit. It provides dedicated storage for MMA accumulators and operands without consuming register file entries or shared memory capacity.

### TMEM Characteristics

- **Capacity:** 512 columns per SM (each column holds M elements of the accumulator)
- **Allocation granularity:** Columns are allocated in contiguous blocks via `tcgen05.alloc`
- **Addressing:** Byte-addressed within TMEM space; offsets are relative to the allocated base
- **Lifetime:** Allocated by the MMA warp, accessible by softmax and correction warps after pointer broadcast
- **Deallocation:** Explicit via `tcgen05.dealloc`, coordinated through cluster barriers for 2-CTA mode

### TMEM Allocation in FA4

The FA4 kernel allocates TMEM as follows:

```python
# In the MMA warp (warp_idx == self.mma_warp_id)
tmem = cutlass.utils.TmemAllocator(
    storage.tmem_holding_buf,           # SMEM buffer for ptr exchange
    barrier_for_retrieve=tmem_alloc_barrier,  # Named barrier for sync
    allocator_warp_id=self.mma_warp_id,
    is_two_cta=self.use_2cta_instrs,
    two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
)

# Request maximum TMEM (512 columns)
tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
tmem.wait_for_alloc()
tmem_ptr = tmem.retrieve_ptr(Float32)  # Base pointer in TMEM

# After kernel completes:
tmem.relinquish_alloc_permit()
tmem_alloc_barrier.arrive_and_wait()
tmem.free(tmem_ptr)
```

### TMEM Column Layout for Attention

FA4 partitions the 512 TMEM columns across the attention computation stages:

```
Column offset:  0         128       256       384       512
                |-- S0 --|-- S1 --|-- O0 ---|-- O1 ---|

S0: Attention scores for query tile 0 (128 columns for n_block=128)
S1: Attention scores for query tile 1 (128 columns)
O0: Output accumulator for query tile 0 (128 columns for head_dim_v=128)
O1: Output accumulator for query tile 1 (128 columns)

P (softmax output) shares S columns with an offset:
  P0 = S0 + n_block/2 = column 64
  P1 = S1 + n_block/2 = column 192
```

The column layout is configured in the kernel constructor:

```python
self.tmem_s_offset = [0, self.n_block_size]                    # e.g., [0, 128]
self.tmem_o_offset = [
    self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
    for i in range(self.q_stage)
]                                                               # e.g., [256, 384]
self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded  # e.g., 512
assert self.tmem_total <= self.tmem_alloc_cols                  # Must fit in 512 cols
```

## Instruction Descriptor Format

### The 32-Bit Descriptor

UMMA uses a 32-bit instruction descriptor that encodes all MMA configuration parameters. This is a key difference from WGMMA, which used separate PTX instruction variants (e.g., `SM90_64x64x16_F16F16F16_SS`).

```
Bit field layout (32 bits total):
[31:30] max_shift      (2 bits) - MX block scaling: 0=none, 1=shift8, 2=shift16, 3=shift32
[29]    reserved
[28:24] m_dim          (5 bits) - M dimension >> 4 (e.g., 128 >> 4 = 8)
[22:17] n_dim          (6 bits) - N dimension >> 3 (e.g., 128 >> 3 = 16)
[16]    b_major        (1 bit)  - Operand B layout: 0=K-major, 1=MN-major
[15]    a_major        (1 bit)  - Operand A layout: 0=K-major, 1=MN-major
[14]    b_negate       (1 bit)  - Negate operand B: 0=no, 1=yes
[13]    a_negate       (1 bit)  - Negate operand A: 0=no, 1=yes
[12:10] b_format       (3 bits) - Operand B element type
[9:7]   a_format       (3 bits) - Operand A element type
[5:4]   c_format       (2 bits) - Accumulator type: 0=F16, 1=F32, 2=S32
[3]     saturate       (1 bit)  - Output saturation
[2]     sparse_flag    (1 bit)  - Structured sparsity
[1:0]   sparse_id      (2 bits) - Sparsity metadata
```

### Element Format Encodings

**A/B format (3 bits) -- F16/F32 family:**

| Value | Format | Description |
|-------|--------|-------------|
| 0 | F16 | IEEE FP16 (1-5-10) |
| 1 | BF16 | Brain Float 16 (1-8-7) |
| 2 | TF32 | TensorFloat-32 (1-8-10, 19 bits) |

**A/B format (3 bits) -- MX F8/F6/F4 family:**

| Value | Format | Description |
|-------|--------|-------------|
| 0 | E4M3 | FP8 (4 exponent, 3 mantissa) |
| 1 | E5M2 | FP8 (5 exponent, 2 mantissa) |
| 3 | E2M3 | FP6 (2 exponent, 3 mantissa) |
| 4 | E3M2 | FP6 (3 exponent, 2 mantissa) |
| 5 | E2M1 | FP4 (2 exponent, 1 mantissa) |

**Integer format:**

| Value | Format |
|-------|--------|
| 0 | UINT8 |
| 1 | INT8 |

**Accumulator format (2 bits):**

| Value | Format |
|-------|--------|
| 0 | F16 |
| 1 | F32 |
| 2 | S32 |

### Building the Descriptor in Python

From the FA4 codebase (mma_sm100_desc.py):

```python
def make_instr_desc(
    a_type, b_type, c_type,
    M: int, N: int,
    a_major: Major, b_major: Major,
    a_neg=ScaleIn.One, b_neg=ScaleIn.One,
    c_sat=Saturate.False_,
    is_sparse=False,
    max_shift=MaxShift.NoShift,
) -> int:
    """Build the 32-bit instruction descriptor for Blackwell MMA."""
    a_fmt = int(to_UMMA_format(a_type))
    b_fmt = int(to_UMMA_format(b_type))
    c_fmt = int(to_C_format(c_type))

    # Range checks
    assert M in (64, 128, 256), "M must be 64, 128 or 256"
    assert 8 <= N <= 256 and (N & 7) == 0, "N must be 8..256, multiple of 8"

    m_dim = M >> 4   # 5-bit field
    n_dim = N >> 3   # 6-bit field

    # Pack bit-fields
    desc = 0
    desc |= (int(is_sparse)  & 0x1) << 2
    desc |= (int(c_sat)      & 0x1) << 3
    desc |= (c_fmt           & 0x3) << 4
    desc |= (a_fmt           & 0x7) << 7
    desc |= (b_fmt           & 0x7) << 10
    desc |= (int(a_neg)      & 0x1) << 13
    desc |= (int(b_neg)      & 0x1) << 14
    desc |= (int(a_major)    & 0x1) << 15
    desc |= (int(b_major)    & 0x1) << 16
    desc |= (n_dim           & 0x3F) << 17
    desc |= (m_dim           & 0x1F) << 24
    desc |= (int(max_shift)  & 0x3) << 30

    return desc & 0xFFFF_FFFF
```

### Descriptor Examples for Attention

```python
# QK^T GEMM: BF16 x BF16 -> FP32, M=128, N=128, both K-major
# a_fmt=1(BF16), b_fmt=1(BF16), c_fmt=1(F32), m_dim=8, n_dim=16
# Result: 0x08_02_00_90  (approximate)

# PV GEMM with MX-scaled FP8: E4M3 x E4M3 -> FP32, M=128, N=128
# a_fmt=0(E4M3), b_fmt=0(E4M3), c_fmt=1(F32), max_shift=MaxShift8
# This enables native block scaling during the MMA
```

## SMEM Descriptor for UMMA

### 64-Bit SMEM Descriptor Format

Similar to WGMMA, UMMA reads operand B (and optionally operand A) from shared memory through 64-bit descriptors. The Blackwell descriptor format extends the Hopper format:

```
Bit field layout (64 bits):
[63:61] layout_type      (3 bits) - Swizzle mode family
[52]    lbo_mode         (1 bit)  - Leading byte offset mode
[51:49] base_offset      (3 bits) - Base offset (typically 0)
[47:46] version          (2 bits) - Descriptor version (always 1)
[45:32] stride_byte_off  (14 bits) - Stride byte offset
[29:16] leading_byte_off (14 bits) - Leading dimension byte offset
[13:0]  start_addr       (14 bits) - SMEM start address >> 4
```

### Legal Swizzle Modes

| LayoutType Value | Name | Swizzle Parameters | Notes |
|-----------------|------|-------------------|-------|
| 0 | SWIZZLE_NONE | Swizzle<0,4,3> | No swizzle (interleaved) |
| 1 | SWIZZLE_128B_BASE32B | Swizzle<2,5,2> | New for Blackwell |
| 2 | SWIZZLE_128B | Swizzle<3,4,3> | Most common for attention |
| 4 | SWIZZLE_64B | Swizzle<2,4,3> | Smaller tiles |
| 6 | SWIZZLE_32B | Swizzle<1,4,3> | Smallest swizzle |

The SWIZZLE_128B_BASE32B mode is new on Blackwell and provides a different base alignment for bank conflict avoidance with certain data types and tile shapes.

### SMEM Layout Construction for Attention

```python
# CUTLASS helper creates swizzled SMEM layouts
sQ_layout = sm100_utils.make_smem_layout_a(
    tiled_mma_qk, mma_tiler_qk, dtype, q_stages=2
)
sK_layout = sm100_utils.make_smem_layout_b(
    tiled_mma_qk, mma_tiler_qk, dtype, kv_stages=3
)
sV_layout = sm100_utils.make_smem_layout_b(
    tiled_mma_pv, mma_tiler_pv, dtype, kv_stages=3
)

# When head_dim != head_dim_v, K and V share physical SMEM with adjusted strides
if not same_hdim_kv_padded:
    stage_stride = max(stride_sK, stride_sV)
    sK_layout = make_composed_layout(sK_layout.inner, 0,
        make_layout((*sK_layout.outer.shape[:-1], kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride)))
```

## CTA-Group Operations

### CTA-Group::1 (Single CTA)

Standard mode where one CTA independently executes UMMA instructions using its own TMEM and SMEM:

```python
cta_group = tcgen05.CtaGroup.ONE
cluster_shape_mn = (1, 1)
mma_tiler_qk = (m_block_size, n_block_size, head_dim)  # e.g., (128, 128, 128)
```

### CTA-Group::2 (Cooperative Two-CTA)

Two CTAs in a 2x1 cluster cooperatively execute a single UMMA instruction. Each CTA contributes its TMEM, effectively doubling the M dimension:

```python
cta_group = tcgen05.CtaGroup.TWO
cluster_shape_mn = (2, 1)
# MMA tiler M covers BOTH CTAs
mma_tiler_qk = (2 * m_block_size, n_block_size, head_dim)  # e.g., (256, 128, 128)
# But each CTA only owns m_block_size rows
cta_tiler = (q_stage * m_block_size, n_block_size, head_dim)
```

In 2-CTA mode:
- The MMA instruction spans both CTAs' TMEM (256 rows if m_block_size=128)
- SMEM for K/V is divided between the two CTAs: `smem_size_kv_per_stage // cta_group_size`
- Pipeline barriers must account for both CTAs in the cluster
- TMEM deallocation requires cluster-level barrier synchronization

### When to Choose 2-CTA

FA4 uses 2-CTA mode selectively:

```python
# 2-CTA is beneficial for:
# - Large head dimensions (hdim=192) where SMEM per CTA is tight
# - When the M dimension naturally aligns with 2x the block size
# - When amortizing K/V loading across more query rows

# FA4 configuration (from flash_fwd_sm100.py):
use_2cta_instrs = True  # For larger configurations
cta_group_size = 2 if use_2cta_instrs else 1
```

## Pipeline Integration with UMMA

### FA4's UMMA Pipeline Types

FA4 uses specialized pipeline types that bridge UMMA's TMEM-based operation with the producer-consumer synchronization model:

```python
# PipelineTmaUmma: TMA loads -> UMMA consumes from SMEM
pipeline_q = pipeline_custom.PipelineTmaUmma.create(
    barrier_storage=storage.mbar_load_Q.data_ptr(),
    num_stages=q_stage,
    producer_group=tma_warp,      # Load warp produces
    consumer_group=mma_warp,      # MMA warp consumes
    tx_count=tma_copy_bytes["Q"],
    cta_layout_vmnk=cta_layout_vmnk,
)

# PipelineUmmaAsync: UMMA produces (writes S to TMEM) -> Softmax consumes
pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
    barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
    num_stages=q_stage,
    producer_group=mma_warp,
    consumer_group=softmax_correction_threads_cluster,
    cta_layout_vmnk=cta_layout_vmnk,
)

# PipelineAsyncUmma: Softmax produces (writes P to TMEM) -> UMMA consumes
pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
    barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
    num_stages=q_stage,
    producer_group=softmax_warps_cluster,
    consumer_group=mma_warp,
    cta_layout_vmnk=cta_layout_vmnk,
)
```

### Pipeline Flow for One KV Block

```
Load Warp          MMA Warp              Softmax Warps        Correction Warps
    |                  |                      |                     |
    | TMA K->SMEM      |                      |                     |
    |----signal-------->|                      |                     |
    |                  | tcgen05.mma(Q,K)->S   |                     |
    |                  |     (S in TMEM)       |                     |
    |                  |----signal------------>|                     |
    |                  |                      | read S from TMEM    |
    |                  |                      | compute P=softmax(S)|
    |                  |                      | write P to TMEM     |
    |                  |                      |----signal---------->|
    |                  |                      |                     | rescale O if needed
    | TMA V->SMEM      |<------signal---------|<------signal--------|
    |----signal-------->|                      |                     |
    |                  | tcgen05.mma(P,V)->O   |                     |
    |                  |     (P from TMEM,     |                     |
    |                  |      O in TMEM)       |                     |
```

## Warp Specialization with UMMA

### FA4 Warp Assignment

FA4 assigns 16 warps per CTA with specific roles:

| Warp IDs | Count | Role | Register Budget |
|----------|-------|------|-----------------|
| 0-3 | 4 | Softmax warpgroup 0 | 192-200 |
| 4-7 | 4 | Softmax warpgroup 1 | 192-200 |
| 8-11 | 4 | Correction warps | 64-80 |
| 12 | 1 | MMA warp (UMMA dispatch) | 48 |
| 13 | 1 | Epilogue warp | 48 |
| 14 | 1 | Load warp (TMA) | 48 |
| 15 | 1 | Empty (padding) | 48 |

Key insight: The MMA warp needs very few registers (48) because UMMA stores its accumulators in TMEM, not registers. This is the fundamental enabler of FA4's 5-way warp specialization -- the register file budget freed by TMEM is redistributed to softmax warps (192-200 registers each) which perform complex row-wise computations.

### Register Pressure Comparison

```
WGMMA (Hopper, FA3):
  - 128 threads in MMA warpgroup each hold 32 accumulator values in registers
  - Total: 128 * 32 * 4 bytes = 16 KB of register file per warpgroup for accumulators alone
  - Limits available registers for softmax computation

UMMA (Blackwell, FA4):
  - 32 threads in MMA warp need ~0 accumulator registers (accumulators in TMEM)
  - 512 TMEM columns * M rows * 4 bytes for accumulators
  - Registers freed for softmax warps: enables 200 regs/warp for softmax computation
  - Enables simultaneous operation of 8 softmax warps + 4 correction warps
```

## Supported Tile Shapes for Attention

### QK^T GEMM (Score Computation)

| Configuration | M | N | K | CTA-group | Typical Use |
|--------------|---|---|---|-----------|-------------|
| Standard | 128 | 128 | 128 | 1 | hdim=128, standard block size |
| Large M | 256 | 128 | 128 | 2 | 2-CTA cooperative, more Q rows |
| Small hdim | 128 | 128 | 64 | 1 | hdim=64 |
| Large hdim | 128 | 128 | 192 | 1 | hdim=192 (padded to 192) |

### PV GEMM (Output Accumulation)

| Configuration | M | N (hdim_v) | K (n_block) | A Source | Typical Use |
|--------------|---|------------|-------------|----------|-------------|
| Standard | 128 | 128 | 128 | TMEM | P from TMEM, V from SMEM |
| Different hdim_v | 128 | 64 | 128 | TMEM | When head_dim_v < head_dim |
| 2-CTA | 256 | 128 | 128 | TMEM | Cooperative mode |

## Comparison with ThunderKittens TCGEN05 Interface

ThunderKittens 2.0 provides a higher-level abstraction over the same UMMA instructions:

```cpp
// ThunderKittens approach (high-level)
rt_bf<32, 64> q_tile;
st_bf<64, 64> k_tile;
warpgroup::mma_AB(s_acc, q_tile, k_tile);  // Internally uses tcgen05.mma

// Direct CUTLASS/FA4 approach (low-level)
tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
    dtype, q_major, k_major, Float32, cta_group, tile_shape
)
# ... configure descriptors, TMEM layout, pipeline barriers ...
```

ThunderKittens abstracts away descriptor construction, TMEM management, and pipeline synchronization, making it suitable for prototyping. FA4's direct CUTLASS approach provides full control over all UMMA parameters for maximum performance.

## Debugging UMMA Kernels

### Common Issues

1. **TMEM allocation failure:** Requesting more than 512 columns, or failing to free TMEM before another kernel tries to allocate.

2. **Descriptor mismatch:** The instruction descriptor's M/N dimensions must match the actual tile being processed. A mismatch causes silent data corruption.

3. **CTA-group synchronization:** In 2-CTA mode, both CTAs must arrive at barriers before either can proceed. A deadlock occurs if one CTA's warp takes a divergent path.

4. **Swizzle mode incompatibility:** The SMEM layout swizzle must match what the UMMA descriptor expects. An invalid combination produces `"Not a canonical UMMA_MN Layout"` or `"Not a canonical UMMA_K Layout"` errors.

5. **TMEM pointer exchange:** Non-MMA warps (softmax, correction) must wait for the MMA warp to broadcast the TMEM base pointer via shared memory and a named barrier before accessing TMEM.

### Validation Approach

```python
# Verify TMEM fits
assert tmem_total <= cute.arch.get_max_tmem_alloc_cols("sm_100")

# Verify descriptor dimensions
assert M in (64, 128, 256)
assert 8 <= N <= 256 and N % 8 == 0

# Verify CTA-group consistency
if use_2cta_instrs:
    assert cluster_shape_mn == (2, 1)
    assert mma_tiler[0] == 2 * m_block_size
```

## Performance Tuning Guidelines

### Tile Size Selection

- **m_block_size=128, n_block_size=128** is the default for most head dimensions
- **2-CTA mode** is beneficial for hdim=192 where SMEM is tight per CTA
- **q_stage=2** (dual query tiles) is standard; q_stage=1 for memory-constrained configurations

### KV Pipeline Staging

```python
# FA4 determines KV stages based on available SMEM
kv_stage = (224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage
# Typically 3 stages for standard configurations
# Can reach 3 stages even for hdim=192/128 with uneven_kv_smem trick
```

### SMEM Budget

Blackwell SMs have 228 KB of configurable shared memory. FA4 budgets approximately:
- Q tiles: 2 stages x 128 x 128 x 2 bytes = 64 KB
- KV tiles: 3 stages x 128 x 128 x 2 bytes = 96 KB (shared between K and V)
- O tiles: 2 stages x 128 x 128 x 2 bytes = 64 KB (may overlap with Q)
- Barriers, scale factors, metadata: ~4 KB

## Key Design Principles

1. **TMEM as the central innovation:** By moving accumulators and the P matrix to TMEM, UMMA decouples MMA computation from register pressure, enabling more warp specialization roles.

2. **Single-warp MMA dispatch:** One warp controls all MMA instructions, while 12+ warps handle softmax, correction, and I/O -- a much finer division of labor than Hopper's 2-way warpgroup split.

3. **Descriptor-driven configurability:** The 32-bit instruction descriptor allows runtime selection of data types, tile shapes, and MX scaling without different instruction variants.

4. **CTA-group scalability:** The same kernel code works for 1-CTA and 2-CTA modes with configuration changes, not code changes.

5. **Pipeline type hierarchy:** PipelineTmaUmma, PipelineUmmaAsync, and PipelineAsyncUmma encode the three producer-consumer patterns in FA4: TMA->UMMA, UMMA->async, and async->UMMA.

## References

- [NVIDIA PTX ISA: tcgen05.mma instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUTLASS SM100 MMA Descriptor Header](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp)
- [CUTLASS SM100 MMA Traits](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm100.hpp)
- [CUTLASS Example 77: Blackwell FMHA](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha)
- [Flash Attention 4 SM100 Kernel (flash_fwd_sm100.py)](https://github.com/Dao-AILab/flash-attention)
- [Modal Blog: Reverse Engineering Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4)
- [ThunderKittens 2.0: Blackwell TCGEN05 Support](https://github.com/HazyResearch/ThunderKittens)
- [NVIDIA Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
