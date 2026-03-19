# Blackwell FP4 (NVFP4/E2M1) for Attention: Format, Hardware, Quantization, and Deployment

**Sources:** NVIDIA Blackwell Architecture, CUTLASS SM100 MMA descriptor, FlashInfer source code (vec_dtypes.cuh, decode.py, prefill.py, utils.py), OCP Microscaling Formats Specification, ThunderKittens 2.0

## Overview

FP4 (E2M1) is the most aggressively compressed floating-point format supported by NVIDIA Blackwell's tensor cores. With only 4 bits per element -- 1 sign, 2 exponent, 1 mantissa -- FP4 provides extreme memory compression (4x over FP16) and the highest tensor core throughput on Blackwell (~2x FP8, ~4x FP16). For attention kernels, FP4 is primarily viable for KV cache compression and output quantization during inference, where the memory bandwidth savings outweigh the severe precision limitations. This document covers the E2M1 format in detail, Blackwell's hardware support through UMMA, practical FP4 attention patterns in FlashInfer and ThunderKittens, quantization strategies, accuracy analysis, and deployment considerations.

## The E2M1 Format

### Bit Layout

```
E2M1 (4 bits): [S | E1 E0 | M0]

S:     Sign bit (0 = positive, 1 = negative)
E1 E0: 2 exponent bits with bias = 1
M0:    1 mantissa bit (implicit leading 1 for normal numbers)
```

### Complete Value Table

```
Binary  | Sign | Exp | Mantissa | Value     | Category
--------|------|-----|----------|-----------|----------
0 00 0  |  +   |  0  |    0     |  0.0      | Zero
0 00 1  |  +   |  0  |    1     |  0.5      | Subnormal
0 01 0  |  +   |  1  |    0     |  1.0      | Normal
0 01 1  |  +   |  1  |    1     |  1.5      | Normal
0 10 0  |  +   |  2  |    0     |  2.0      | Normal
0 10 1  |  +   |  2  |    1     |  3.0      | Normal
0 11 0  |  +   |  3  |    0     |  4.0      | Normal
0 11 1  |  +   |  3  |    1     |  6.0      | Normal (max)
1 00 0  |  -   |  0  |    0     | -0.0      | Zero
1 00 1  |  -   |  0  |    1     | -0.5      | Subnormal
1 01 0  |  -   |  1  |    0     | -1.0      | Normal
1 01 1  |  -   |  1  |    1     | -1.5      | Normal
1 10 0  |  -   |  2  |    0     | -2.0      | Normal
1 10 1  |  -   |  2  |    1     | -3.0      | Normal
1 11 0  |  -   |  3  |    0     | -4.0      | Normal
1 11 1  |  -   |  3  |    1     | -6.0      | Normal (min)

Total: 16 distinct values (including +0 and -0)
       15 distinct magnitudes (including zero)
       7 distinct positive nonzero magnitudes
```

### Value Distribution

```
Number line showing E2M1 representable values:

-6  -4  -3  -2  -1.5  -1  -0.5  0  0.5  1  1.5  2  3  4  6
 |   |   |   |    |    |    |    |   |   |   |   |  |  |  |
 *   *   *   *    *    *    *    *   *   *   *   *  *  *  *

Note the non-uniform spacing:
  [0, 1]:   spacing = 0.5 (2 intervals)
  [1, 2]:   spacing = 0.5 (2 intervals)
  [2, 4]:   spacing = 1.0 (2 intervals)
  [4, 6]:   spacing = 2.0 (1 interval)

This geometric spacing means relative precision is roughly constant
at ~25-50% of the value, but absolute precision is very coarse.
```

### Comparison with Other Formats

| Format | Bits | Distinct Values | Max Magnitude | Min Positive | Relative Precision |
|--------|------|----------------|---------------|--------------|-------------------|
| FP4 E2M1 | 4 | 15 | 6.0 | 0.5 | ~25-50% |
| FP6 E2M3 | 6 | 62 | 7.5 | 0.0625 | ~6-12% |
| FP6 E3M2 | 6 | 62 | 28.0 | 0.25 | ~12-25% |
| FP8 E4M3 | 8 | 238 | 448.0 | 0.001953 | ~6.25% |
| FP8 E5M2 | 8 | 238 | 57,344 | 1.5e-5 | ~12.5% |
| FP16 | 16 | 65,504 | 65,504 | 5.96e-8 | ~0.1% |

## MXFP4: FP4 with MX Block Scaling

### Format Structure

MXFP4 pairs E2M1 elements with E8M0 per-block (32-element) scaling:

```
Block of 32 MXFP4 elements:
  ┌──────────────────────────────────────────────┐
  │ 32 x E2M1 values = 32 x 4 bits = 16 bytes   │
  │ (packed: 2 values per byte = 16 uint8 values) │
  ├──────────────────────────────────────────────┤
  │ 1 x E8M0 scale factor = 1 byte               │
  │ scale = 2^(exponent - 127)                    │
  └──────────────────────────────────────────────┘

Effective value: e2m1_value * scale
Effective range: [-6 * 2^127, +6 * 2^127] (astronomically large)
Storage: 16.5 bytes per 32 elements = 4.125 bits per element
Overhead: 6.25% for scale factors
```

### Effective Dynamic Range

```
Without scaling:
  E2M1 range: [-6, +6]
  Only 15 distinct values

With E8M0 block scaling:
  Effective range per block: [-6 * 2^127, +6 * 2^127]
  Within-block precision: still only 15 values in the scaled range
  Key insight: MXFP4 expands RANGE but not PRECISION

Example: if block_scale = 2^10 (= 1024):
  Representable values in this block:
  {0, 512, 1024, 1536, 2048, 3072, 4096, 6144} and negatives
  Spacing between values: 512 to 2048 (very coarse!)
```

## Blackwell Hardware Support

### Tensor Core Throughput

```
NVIDIA B200 Approximate Peak Throughput:
  FP4 (E2M1):  ~9,000 TFLOPS  (4x FP16)
  FP8 (E4M3):  ~4,500 TFLOPS  (2x FP16)
  FP16/BF16:   ~2,250 TFLOPS  (baseline)
  TF32:        ~1,125 TFLOPS

Throughput ratio: FP4 : FP8 : FP16 = 4 : 2 : 1
```

### UMMA Instruction Support

The UMMA instruction (tcgen05.mma) on Blackwell supports FP4 natively:

```python
# From mma_sm100_desc.py -- E2M1 is a valid element format
class MXF8F6F4Format(IntEnum):
    E4M3 = 0   # FP8
    E5M2 = 1   # FP8
    E2M3 = 3   # FP6
    E3M2 = 4   # FP6
    E2M1 = 5   # FP4

# Building an instruction descriptor with FP4 elements:
desc = make_instr_desc(
    a_type=...,           # Element type for operand A
    b_type=...,           # Element type for operand B (can be E2M1)
    c_type=Float32,       # Accumulator always FP32
    M=128, N=128,
    a_major=Major.K, b_major=Major.MN,
    max_shift=MaxShift.MaxShift8,  # Enable MX block scaling
)
```

When the element format is E2M1 and max_shift is nonzero, the tensor core:
1. Reads 32 packed FP4 elements from SMEM (16 bytes)
2. Reads the corresponding E8M0 scale factor (1 byte)
3. Unpacks each 4-bit element to internal precision
4. Multiplies by the block scale (exponent addition)
5. Performs the multiply-accumulate in FP32

### CUDA Type Support

CUDA 12.8+ provides the `__nv_fp4_e2m1` type:

```cpp
// From CUDA headers
typedef struct __nv_fp4_e2m1 {
    __nv_fp4_storage_t __x;  // 4-bit storage (packed in uint8)
} __nv_fp4_e2m1;

// Packed types for efficient operations
typedef uint8_t __nv_fp4x2_storage_t;  // Two FP4 values in one byte

// Requires CUDA 12.8+ and FLASHINFER_ENABLE_FP4_E2M1 define
#if defined(FLASHINFER_ENABLE_FP4_E2M1) && CUDA_VERSION >= 12080
```

## FP4 Vector Operations in FlashInfer

FlashInfer provides comprehensive FP4 vector types for efficient CUDA operations:

### Sub-Byte Packing

```cpp
// Two FP4 values packed into one uint8
template <>
struct vec_t<__nv_fp4_e2m1, 2> {
    uint8_t data;  // Lower 4 bits = element 0, upper 4 bits = element 1

    void fill(__nv_fp4_e2m1 val) {
        // Pack same value into both nibbles
        data = (__nv_fp4x2_storage_t(val.__x) << 4) |
               __nv_fp4x2_storage_t(val.__x);
    }

    void load(const __nv_fp4_e2m1* ptr) { data = *((uint8_t*)ptr); }
    void store(__nv_fp4_e2m1* ptr) const { *((uint8_t*)ptr) = data; }
};
```

### Progressively Larger Vector Types

```cpp
// 4 FP4 values in uint16
template <> struct vec_t<__nv_fp4_e2m1, 4> { uint16_t data; };

// 8 FP4 values in uint32
template <> struct vec_t<__nv_fp4_e2m1, 8> { uint32_t data; };

// 16 FP4 values in uint2 (64 bits)
template <> struct vec_t<__nv_fp4_e2m1, 16> { uint2 data; };

// 32+ FP4 values in int4 arrays (128-bit loads)
template <size_t vec_size>
struct vec_t<__nv_fp4_e2m1, vec_size> {
    static_assert(vec_size % 32 == 0);
    int4 data[vec_size / 32];  // One int4 per 32 elements = one MX block
};
```

### Memory Operations

```cpp
// Bulk load/store using 128-bit (int4) operations for 32+ elements
void load(const __nv_fp4_e2m1* ptr) {
    #pragma unroll
    for (size_t i = 0; i < vec_size / 32; ++i) {
        data[i] = ((int4*)ptr)[i];  // 128-bit load
    }
}

// Global memory operations with memory ordering
void store_global_release(__nv_fp4_e2m1* addr) const {
    #pragma unroll
    for (size_t i = 0; i < vec_size / 32; ++i) {
        st_global_release(*(int4*)&data[i], (int4*)(addr + i * 16));
    }
}
```

Key observation: The 32-element MX block aligns perfectly with int4 (128-bit) loads and stores for FP4 data, since 32 FP4 values = 16 bytes = 128 bits. This alignment is by design in the OCP MX specification.

## FP4 Attention Patterns

### Pattern 1: FP4 KV Cache with FP8/FP16 Compute

The most practical FP4 attention pattern uses FP4 only for KV cache storage, with higher-precision computation:

```
Storage:
  Q:  FP8 (E4M3) or FP16        -- queries need precision for scoring
  KV: MXFP4 (E2M1 + E8M0)      -- KV cache compressed for memory savings

Computation:
  Q * K^T: Q (FP8) x K_dequant (FP8/FP16) -> FP32 accumulator
         K is dequantized from FP4 on-the-fly during MMA via MX scaling
  Softmax: FP32 (always)
  P * V:   P (FP16) x V_dequant (FP8/FP16) -> FP32 accumulator
         V is dequantized from FP4 on-the-fly

Advantage: 4x KV cache compression, moderate accuracy impact
```

### Pattern 2: FP4 Output Quantization

FlashInfer supports quantizing the attention output to FP4:

```python
# FlashInfer decode with FP4 output
result = batch_decode_with_kv_cache(
    query=query,                        # FP8 or FP16
    kv_cache=kv_cache,                  # Standard or FP8 cache
    out_dtype="nvfp4",                  # Quantize output to FP4
    o_sf_scale=1.0,                     # Output scale factor
    o_sf_vec_size=16,                   # 16 elements per scale group
    backend="trtllm-gen",               # Blackwell backend
)

# Constraints (from FlashInfer source):
# - Query must be FP8 (E4M3) when output is NVFP4
# - Only o_sf_vec_size=16 is supported
# - Only available on sm_100/sm_103 (Blackwell)
```

The output FP4Tensor has:
- `data`: uint8 tensor with shape `[..., ceil(head_dim / 2)]`
- `scale`: float8_e4m3fn tensor for per-group scale factors
- `scale.shape[0]` must be a multiple of 128

### Pattern 3: FP4 KV Cache with NVFP4 Scale Factors

```python
# FlashInfer supports NVFP4 KV cache with separate scale factors
result = batch_decode_with_kv_cache(
    query=query,
    kv_cache=kv_cache,                  # FP4 KV cache
    kv_cache_sf=(k_scale, v_scale),     # Per-block scale factors
    out_dtype="nvfp4",
    o_sf_scale=o_scale,
    backend="trtllm-gen",
)
```

## Quantization Strategies for FP4 Attention

### Challenge: 1-Bit Mantissa

With only 1 mantissa bit, FP4 quantization error is fundamentally severe:

```
Quantization error analysis for uniform random values in [0, 1]:
  FP16: max error = 2^-11 ≈ 4.88e-4 (0.049%)
  FP8:  max error = 2^-4  ≈ 6.25e-2 (6.25%)
  FP4:  max error = 0.25            (25%)

For attention scores with softmax:
  - Softmax amplifies quantization errors exponentially
  - Small errors in scores -> moderate errors in attention weights
  - FP4 quantization noise on scores -> significant softmax distortion

This is why FP4 for Q*K^T scores is generally not recommended.
```

### Strategy 1: Calibration-Based Quantization

For KV cache quantization, calibration using representative data helps select optimal scale factors:

```python
def calibrate_fp4_kv(model, calibration_data, block_size=32):
    """
    Run calibration forward passes to determine optimal
    per-block scale factors for FP4 KV cache.
    """
    all_amax = []
    for batch in calibration_data:
        with torch.no_grad():
            k, v = model.compute_kv(batch)
            # Record per-block amax statistics
            k_flat = k.reshape(-1, block_size)
            v_flat = v.reshape(-1, block_size)
            all_amax.append(k_flat.abs().amax(dim=-1))
            all_amax.append(v_flat.abs().amax(dim=-1))

    # Use percentile-based clipping rather than absolute max
    # to reduce the impact of rare outliers
    combined_amax = torch.cat(all_amax)
    clip_value = torch.quantile(combined_amax, 0.999)
    return clip_value
```

### Strategy 2: Outlier-Aware Quantization

LLM activations contain outlier channels that dominate quantization:

```
Approach: Handle outliers separately
  1. Identify outlier channels (>3 sigma from mean)
  2. Store outlier channels in FP8 or FP16
  3. Store remaining channels in FP4
  4. Recombine during dequantization

Alternative: Incoherent processing (Hadamard transform)
  1. Apply randomized Hadamard transform to spread outliers
  2. Quantize the transformed values to FP4 (more uniform distribution)
  3. Apply inverse transform after dequantization
  Cost: O(d log d) for head dimension d
```

### Strategy 3: Mixed-Precision KV Cache

Different parts of the KV cache may tolerate different precision:

```
Layer-wise mixed precision:
  Lower layers (0-8):    FP8 (early features need precision)
  Middle layers (8-24):  FP4 (tolerant to quantization)
  Upper layers (24-32):  FP4 or FP8 (depends on task)

Head-wise mixed precision:
  Attention heads with high entropy:   FP4 (diffuse attention is tolerant)
  Attention heads with low entropy:    FP8 (sharp attention needs precision)

Time-wise mixed precision:
  Recent KV entries (last 256 tokens):  FP8 (active attention)
  Older KV entries:                     FP4 (sparse attention)
```

## Accuracy Analysis

### Impact on Attention Quality

```
Attention accuracy degradation by format (approximate, model-dependent):

Format     | RMSE vs FP16 | Perplexity Impact | Notes
-----------|-------------|-------------------|------
FP16       | baseline    | 0                 | Reference
FP8 (naive)| 2.4e-2      | ~0.01-0.05        | Per-tensor scaling
MXFP8      | 9.1e-3      | ~0.001-0.01       | Block scaling + incoherent
MXFP4 (KV) | ~5-10e-2    | ~0.05-0.2         | KV cache only, calibrated
MXFP4 (all)| ~0.1-0.3    | ~0.5-2.0          | Full FP4, generally unacceptable
```

### Critical Observation

The softmax function amplifies quantization errors nonlinearly:

```
Consider two attention scores:
  True values: s1 = 5.0, s2 = 4.5
  FP4 quantized: s1_q = 4.0 or 6.0, s2_q = 4.0 or 6.0

  If s1_q = 6.0, s2_q = 4.0:
    softmax([6, 4]) = [0.881, 0.119]
    softmax([5, 4.5]) = [0.622, 0.378]
    Error in attention weight: 0.259 (42% relative error!)

  This is why FP4 for Q*K^T scores is problematic:
  the softmax exponentially amplifies the already-large quantization errors.
```

### When FP4 KV Cache Works

FP4 KV cache is viable when:
1. The model has been calibrated/fine-tuned with FP4-aware quantization
2. The attention pattern is diffuse (high entropy) rather than sharp
3. The downstream task is tolerant (e.g., chat, summarization vs. math, code)
4. Combined with techniques like incoherent processing to smooth outliers
5. Head dimensions are large enough (128+) that block averaging helps

## Memory and Bandwidth Analysis

### KV Cache Compression

For a 70B parameter model with:
- 80 layers, 64 heads, d=128
- Sequence length N = 128K

```
Per-token KV storage (all layers):
  FP16:  2 * 80 * 64 * 128 * 2 bytes = 2.5 MB/token
  FP8:   2 * 80 * 64 * 128 * 1 byte  = 1.25 MB/token
  MXFP4: 2 * 80 * 64 * 128 * 0.53 bytes = 0.66 MB/token
    (0.5 bytes/element + scale overhead)

Total KV cache for 128K tokens:
  FP16:  320 GB  (far exceeds B200's 192 GB)
  FP8:   160 GB  (tight fit on B200)
  MXFP4:  84 GB  (comfortable fit on B200)
  FP4 enables serving 128K context that FP16/FP8 cannot!
```

### Memory Bandwidth Savings

```
B200 HBM bandwidth: ~8 TB/s

Decode attention (1 token, batch=1, seq=32K):
  Reading full KV cache per layer:
    FP16: 32K * 128 * 2 * 2 bytes = 16 MB  -> 2 us
    MXFP4: 32K * 128 * 2 * 0.53 bytes = 4.3 MB -> 0.54 us

  Per-layer speedup: ~3.7x from reduced data movement
  This is the primary value proposition of FP4 for decode attention.
```

## Integration with Attention Frameworks

### FlashInfer (Production-Ready NVFP4)

FlashInfer provides the most complete FP4 attention support as of early 2026:

```python
# Backend selection for Blackwell
# sm_100, sm_103: trtllm-gen (supports NVFP4)
# sm_90: xqa (no NVFP4 support)

if backend == "auto":
    backend = "trtllm-gen" if compute_capability[0] == 10 else "xqa"

# FP4 output constraints
if out_dtype == "nvfp4":
    assert query.dtype == torch.float8_e4m3fn  # Query must be FP8
    assert o_sf_scale is not None               # Scale required
    assert o_sf_vec_size in [None, 16]          # Only 16 supported
```

### ThunderKittens 2.0

ThunderKittens 2.0 added NVFP4 support for Blackwell:

```cpp
// ThunderKittens FP4 type
// Uses the standard __nv_fp4_e2m1 packed types
// Requires special shared tile types for sub-byte data:
static_assert(!std::is_same_v<_T, fp4e2m1>,
    "For FP4 types, you must use a packed type "
    "(i.e., fp4e2m1_2 or fp4e2m1_4).");
```

### CUTLASS SM100

CUTLASS provides the low-level building blocks for FP4 MMA:

```python
# MXF8F6F4Format.E2M1 = 5 in the instruction descriptor
# The make_instr_desc function accepts E2M1 as a valid format
# when targeting SM100/SM103 architectures
```

## FP4 vs FP8 Decision Framework

```
Should you use FP4?

┌─ Is this training? ─── YES ──► Use FP8 or BF16 (FP4 lacks gradient precision)
│
├─ Is this inference? ── YES ──┐
│                              │
│  ┌─ Blackwell GPU? ─── NO ──► Use FP8 (no FP4 tensor core support)
│  │
│  ├─ YES ─────────────────────┐
│  │                           │
│  │  ┌─ Memory-bound? ── NO ─► Use FP8 (sufficient throughput)
│  │  │
│  │  ├─ YES ──────────────────┐
│  │  │                        │
│  │  │  ┌─ Accuracy-critical? ─ YES ─► FP4 KV + FP8 compute (mixed)
│  │  │  │
│  │  │  ├─ NO ───────────────── Full FP4 pipeline (max throughput)
│  │  │  │                       BUT: validate accuracy on your task
```

## Practical Deployment Checklist

1. **Validate accuracy** on representative evaluation sets before deploying FP4
2. **Calibrate scale factors** using calibration data that covers the expected input distribution
3. **Use mixed precision**: FP4 for KV cache, FP8 for query/compute, FP32 for softmax
4. **Monitor quality metrics** in production (perplexity, task accuracy, user satisfaction)
5. **Start with FP8** and graduate to FP4 only when memory constraints demand it
6. **Consider per-layer mixed precision**: some layers are more tolerant than others
7. **Keep FP16/FP8 fallback** for accuracy-sensitive requests or tasks
8. **Ensure CUDA 12.8+** and the correct CUTLASS/FlashInfer version for FP4 support
9. **Test with edge cases**: very long sequences, rare tokens, multilingual input
10. **Profile memory vs compute**: FP4 only helps when memory-bound

## References

- [NVIDIA Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [CUTLASS SM100 MMA Descriptor (mma_sm100_desc.hpp)](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp)
- [Flash Attention 4 SM100 Source (E2M1 support)](https://github.com/Dao-AILab/flash-attention)
- [FlashInfer NVFP4 Decode Attention](https://github.com/flashinfer-ai/flashinfer)
- [FlashInfer vec_dtypes.cuh (FP4 vector types)](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/vec_dtypes.cuh)
- [FlashInfer FP4Tensor Wrapper (utils.py)](https://github.com/flashinfer-ai/flashinfer)
- [OCP Microscaling Formats v1.0 (MXFP4 specification)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [ThunderKittens 2.0 (NVFP4 support)](https://github.com/HazyResearch/ThunderKittens)
- [NVIDIA CUDA Toolkit 12.8: __nv_fp4_e2m1 type](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
