# AMD FMHA Kernel Internals: Comprehensive Technical Reference

## Overview

This document synthesizes the technical details from AMD's Flash Multi-Head Attention (FMHA) kernel implementations in Composable Kernel (CK), covering the V3 forward pass, backward pass, Split-KV decode, FP8 block-scale quantization, and batch prefill kernels. These kernels target the CDNA3 (gfx942, MI300X) and CDNA4 (gfx950, MI350X) architectures.

---

## 1. V3 Forward Kernel Architecture

### Wave-Group Based Core Loop Scheduler

The FAv3 forward pipeline uses a `CoreLoopScheduler` template that controls instruction ordering based on wave group assignment (0 or 1), computation phase (0-3), and masking mode (masked vs. non-masked).

**Instruction categories controlled by the scheduler:**
| Barrier Mask | Instruction Type | Purpose |
|-------------|-----------------|---------|
| `0x008` | MFMA | Matrix Fused Multiply-Add (tensor core) |
| `0x200` | TRANS | Transpose operations |
| `0x002` | VALU | Vector ALU operations |
| `0x004` | SALU | Scalar ALU operations |

**Phase scheduling for masked attention (Wave Group 0):**

- **Phase 0**: Interleaves MFMA with TRANS and VALU
  ```
  Repeat 8 times:
    1 MFMA instruction
    2 TRANS instructions
    2 VALU instructions
  ```
  This pattern keeps tensor cores busy while transpose and vector ALU operations fill pipeline bubbles.

- **Phase 1**: VALU + SALU scheduling for address calculations and loop control. Previously empty phases were filled to prevent wave stalls.

- **Phase 2**: MFMA + VALU interleaving with packed FP32 preamble
  ```
  4 VALU instructions (packed FP32 setup, when enabled)
  Repeat 8 times:
    1 MFMA instruction
    4 VALU instructions
  ```

- **Phase 3**: VALU + SALU for post-processing (matching Phase 1 pattern).

### Packed FP32 Operations

The V3 pipeline uses inline assembly for critical FP32 operations to ensure optimal instruction encoding:

- **`v_fma_f32`**: Fused multiply-add with scalar operand. Using a scalar register for the scale factor (`"s"(b)`) reduces VGPR pressure.
- **`v_pk_mul_f32`**: Packed FP32 multiply -- executes 2 multiplications per instruction. Used for softmax rescaling.
- **`v_cvt_pk_f16_f32` / `v_cvt_pk_bf16_f32`**: Packed conversion for output. Converts 2 FP32 values to FP16/BF16 in one instruction.

These can be disabled with `CK_TILE_DISABLE_PACKED_FP32` for debugging or compatibility.

### MFMA Instruction Ordering

The scheduler uses `__builtin_amdgcn_sched_group_barrier(mask, count, order)` to control instruction ordering:

```cpp
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // Allow 1 MFMA
__builtin_amdgcn_sched_group_barrier(0x200, 2, 0); // Allow 2 TRANS
__builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // Allow 2 VALU
```

The third parameter (0) specifies that these barriers apply to all waves, not just specific ones. The scheduler ensures that MFMA instructions are interleaved with data movement to hide pipeline latency.

### Dynamic Memory Access Count

Instead of hardcoding memory load instruction counts, the V3 pipeline queries tile window properties:

```cpp
constexpr int K_mem_su_ld_insts = k_dram_window.get_num_of_access();
constexpr int V_mem_su_ld_insts = v_dram_window.get_num_of_access();
```

This ensures correct scheduling regardless of tile configuration (different head dimensions, tile sizes).

### O_acc Rescaling Distribution

The output accumulator rescaling (for online softmax) is distributed across phases:

```cpp
constexpr index_t fmha_alu_D_reg_cnt = 6;
```

This moves 6 rescaling instructions from the epilogue to the `fmha_alu1()` phase, distributing work more evenly and avoiding SIMD idle cycles at the end of each loop iteration.

---

## 2. Backward Pass (dQ/dK/dV Gradients)

### Transpose Load Pipeline

The backward pass requires accessing Q, K, V tensors in different layouts for computing dQ, dK, and dV. The transpose load (trload) pipeline performs data transposition during memory load, avoiding separate transpose passes.

**Pipeline variant:** `BlockFmhaBwdDqDkDvPipelineTrloadKrKtrVr`
- `Kr`: K loaded in regular layout
- `Ktr`: K loaded with transpose
- `Vr`: V loaded in regular layout

**Two-stage prefetching:**
```
Stage 1: async_load_tile_transpose(k_lds_window, k_window)  -- K with transpose
Stage 2: async_load_tile(v_lds_window, v_window)             -- V regular
Move windows: advance by kK0/kK1 for next iteration
```

This double-buffered approach hides memory latency by overlapping current computation with next iteration's data loads.

### IGLP Scheduling

The backward pass uses the same `sched_group_barrier` mechanism as the forward pass, with MFMA, TRANS, and VALU interleaving optimized for the different compute pattern of gradient computation.

### Padding Reduction

The backward kernel reduces unnecessary padding from power-of-2 alignment to MFMA-minimum alignment:

```
Before: kHDimPadded = next_power_of_2(kHDim)     // e.g., 192 -> 256
After:  kHDimPadded = (kHDim + 7) / 8 * 8         // e.g., 192 -> 192
```

For head_dim=192 (Llama-3), this eliminates 64 elements of wasted padding per head, reducing memory footprint and improving cache utilization.

### Deterministic Mode Integer Overflow Fix

For very long sequences, the product of batch, head, and sequence indices can overflow 32-bit integers. The backward kernel uses 64-bit arithmetic:

```cpp
int64_t offset = static_cast<int64_t>(batch) * batch_stride +
                 static_cast<int64_t>(head) * head_stride +
                 static_cast<int64_t>(seq_q) * seq_q_stride +
                 static_cast<int64_t>(seq_k);
```

This is critical for deterministic mode where exact reproducibility is required.

---

## 3. Split-KV Decode

### Single-Query Specialization

During LLM inference decode, each new token generates a query of length 1 (seqlen_q=1) attending to the entire KV cache (seqlen_k can be 100K+). The Split-KV kernel is specialized for this pattern:

**Tile sizes for decode:**
- `kM0Decode = 1` (single query row)
- `kN0Decode = 128` (large K tile for throughput)
- `kBlockSizeDecode = 64` (reduced from 256 for better occupancy)

### Split-K Parallelism

The K dimension is divided across multiple workgroups, each computing partial attention results:

```
Workgroup 0: K[0 : K/N]           -> (m_partial_0, l_partial_0, o_partial_0)
Workgroup 1: K[K/N : 2K/N]       -> (m_partial_1, l_partial_1, o_partial_1)
...
Workgroup N: K[(N-1)K/N : K]     -> (m_partial_N, l_partial_N, o_partial_N)
```

Each workgroup stores its partial `(m, l, o)` tuple -- the running max, running sum, and unnormalized output.

### Combine Kernel (Online Softmax Reduction)

A separate combine kernel merges partial results using online softmax reduction:

```
For each split s:
    m_new = max(m_combined, m_s)
    scale_old = exp(m_combined - m_new)
    scale_new = exp(m_s - m_new)
    l_combined = l_combined * scale_old + l_s * scale_new
    o_combined = o_combined * scale_old + o_partial_s * scale_new
    m_combined = m_new

Final: output = o_combined / l_combined
```

This is numerically equivalent to computing attention over the full K dimension.

### Paged KV Cache

The kernel supports two page table formats for KV cache:

1. **vLLM-style 2D block table**: `page_table[batch_size][max_num_blocks_per_seq]`
   - Selected via `VLLM_BLOCK_TABLE_2D` trait

2. **SGLang-style 1D page table**: Flat array with separate page indices
   - Selected via `SGLANG_PAGE_TABLE_1D` trait

**Page address computation:**
```cpp
index_t page_idx = seq_idx / kPageSize;
index_t page_offset = seq_idx % kPageSize;
index_t physical_page = page_table[page_idx];
address = kv_cache + physical_page * kPageSize * head_dim + page_offset * head_dim;
```

### Reverse Block Index Assignment

With causal masking, later query positions attend to more keys than earlier positions. The kernel reverses block assignment so workgroups processing later queries (more work) start first:

```cpp
if (use_mask) {
    block_idx = num_blocks - 1 - block_id;  // Reverse for load balancing
}
```

### Attention Sink Support

For streaming/infinite context inference, the kernel supports "sink tokens" -- special tokens at the beginning of the sequence that are always attended to, even with sliding window attention:

```
Attend if: k_idx < sink_size  OR  (q_idx - k_idx) < window_size
```

---

## 4. FP8 Block-Scale Quantized Attention

### Block-Scale Quantization Scheme

Q, K, V tensors are stored in FP8 (E4M3) format with per-block descale factors. This reduces memory bandwidth by 2x compared to FP16/BF16.

**Descale factor lookup:**
```cpp
const index_t kv_idx = (kv_load_start + i_total_loops * kN0) / block_scale_size_kv;
float k_descale = k_descale_ptr[kv_idx];
float v_descale = v_descale_ptr[kv_idx];
```

The descale factor is looked up based on the current K/V block position, with `block_scale_size_kv` determining the granularity.

### OCP vs FNUZ FP8 Formats

Two FP8 formats are supported with different dynamic range handling:

| Format | Shift Value | Usage |
|--------|------------|-------|
| OCP FP8 (E4M3) | 8.0f | Standard Open Compute Platform format |
| FNUZ FP8 | 7.0f | Finite, No Unsigned Zero format |

The shift is subtracted from the row maximum before `exp2` computation to keep values in the representable FP8 range:

```cpp
#if CK_TILE_USE_OCP_FP8
    validated_m -= 8.0f;   // OCP shift
#else
    validated_m -= 7.0f;   // FNUZ shift
#endif
```

### Per-Block Dequantization During GEMM

The descale factor is applied during the GEMM accumulation phase, not as a separate pass:

```cpp
// S = Q * K^T: multiply accumulated S by k_descale
auto s_scaled = s_acc_element_func * k_descale;

// O = P * V: multiply accumulated O by v_descale
auto o_scaled = o_acc_element * v_descale;
```

This fuses dequantization with computation, avoiding extra memory accesses.

### Kernel Arguments for Block Scale

Dedicated structures carry stride information for efficient strided access to descale tensors:

```cpp
struct FmhaFwdCommonBlockScaleKargs {
    index_t nhead_stride_q_descale;
    index_t nhead_stride_k_descale;
    index_t nhead_stride_v_descale;
    index_t block_scale_size_q;
    index_t block_scale_size_kv;
};
```

---

## 5. Batch Prefill with Paged KV Cache

### 3-Level K-Dimension Decomposition

The batch prefill kernel decomposes the K dimension (seqlen_k) into 3 sub-dimensions for VECTORIZED_LAYOUT KV cache:

```
K = K2 x K0 x K1
    |     |     |
    |     |     +-- V_KIterInner: Vector load size (matches GEMM kKPerThread)
    |     +-------- V_KLanes: Lanes for K dimension (matches GEMM kABKLane)
    +-------------- V_KIterOuter: Outer iteration count
```

This decomposition ensures that memory access patterns align with the MFMA instruction's data layout requirements.

**For VECTORIZED_LAYOUT (3D decomposition):**
- K2 controls the outer loop over K blocks
- K0 maps to wave lanes accessing different K positions simultaneously
- K1 is the vector load width (contiguous elements per thread)

**For LINEAR_LAYOUT (simpler):**
- No outer iteration for page lookup
- K0 and K1 handle standard 2D distribution

### Multi-Dimensional Page Index Lookup

For VECTORIZED_LAYOUT with outer iteration, page indices require 2D lookup in Y-space:

```
VPageIndexYDims = sequence<Y_K1, Y_K2>
gather_index = y_k1 + y_k2 * len(Y_K1)
```

This linearizes the 2D page lookup into a 1D gather index for efficient memory access.

### Vectorized KV Cache Layout (5D)

The VECTORIZED_LAYOUT stores KV cache in a 5D format with x=8 swizzling:

```
K: [num_blocks, num_kv_heads, head_size/8, block_size, 8]
V: [num_blocks, num_kv_heads, block_size/8, head_size, 8]
```

Benefits:
- Coalesced 128-bit (8 x FP16) memory accesses
- Better cache line utilization
- Reduced shared memory bank conflicts

### Unified V Offset Update

A single lambda function handles both 2D and 3D K decomposition cases:

```cpp
auto update_v_offsets = [&](auto k_loop_start) {
    if constexpr(V_KIterOuter > 1) {
        // 3D: iterate over K2, computing offsets for each slice
        for each k2 in [0, V_KIterOuter):
            compute v_offsets with page lookup
            offset = k_loop_start + k2 * V_KLanes * V_KIterInner
    } else {
        // 2D: simple offset computation
        compute v_offsets directly
    }
};
```

### MFMA-Aligned Memory Access

The V tensor loading matches the GEMM's warp-level distribution pattern:
- `V_KLanes` matches `kABKLane` (number of lanes participating in K dimension of MFMA)
- `V_KIterInner` matches `kKPerThread` (elements per thread in K dimension)

This alignment ensures that the data loaded from paged KV cache is in the exact format needed by the MFMA instruction, avoiding costly register shuffles.

---

## Cross-Cutting Optimizations

### Instruction Alignment for Specific Head Dimensions

The kernels include optimized binary variants for common head dimensions:
- **HD64**: Standard configuration
- **HD128**: Default for most models
- **HD192**: Optimized for Llama-3 (requires special MFMA alignment in causal mode)
- **HD256**: For models with larger head dimensions

Each variant has pre-compiled binaries with different rounding modes:
- `rtna`: Round-to-nearest-away
- `rtne`: Round-to-nearest-even
- `rtz`: Round-toward-zero

### Vectorized KV Cache with Framework Integration

The kernel supports both vLLM and SGLang KV cache formats:

| Framework | Block Table Format | Trait |
|-----------|-------------------|-------|
| vLLM | 2D `[batch_size, max_blocks]` | `VLLM_BLOCK_TABLE_2D` |
| SGLang | 1D flat with page indices | `SGLANG_PAGE_TABLE_1D` |

Selection is automatic based on input arguments at kernel launch.

---

## References

- [Composable Kernel GitHub Repository](https://github.com/ROCm/composable_kernel)
- [GEAK: Triton Kernel AI Agent (arXiv 2507.23194)](https://arxiv.org/abs/2507.23194)
- [AMD Instinct MI300X Architecture](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
