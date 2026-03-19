# OCP Microscaling (MX) Formats: MXFP8, MXFP6, MXFP4 for Attention Kernels

**Sources:** OCP Microscaling Formats Specification v1.0, NVIDIA FP8 Blog, FlashAttention-3 paper (arXiv 2407.08608), CUTLASS SM100 MMA descriptor headers, AMD Composable Kernel FP8 block-scale implementation

## Overview

The Open Compute Project (OCP) Microscaling (MX) specification defines a family of block-scaled numerical formats designed for AI training and inference. The core idea is simple: instead of one scale factor per entire tensor (per-tensor scaling) or one scale factor per element (full floating-point), MX uses one shared exponent per block of 32 contiguous elements. This provides fine-grained dynamic range adaptation with minimal storage overhead. The MX family encompasses MXFP8, MXFP6, and MXFP4 formats, each offering different precision-throughput tradeoffs. NVIDIA Blackwell GPUs implement MX natively in their tensor cores, while software implementations enable MX on Hopper and AMD GPUs.

## The MX Format Family

### Core Architecture: Block of 32 + E8M0 Scale

Every MX format shares the same fundamental structure:

```
Block Structure (invariant across all MX formats):
  ┌─────────────────────────────────────────────┐
  │  32 contiguous elements (in MXFP8/6/4)      │
  │  [e0, e1, e2, ..., e31]                     │
  └─────────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────┐
  │  1 shared E8M0 scale factor (8 bits)        │
  │  scale = 2^(exponent - 127)                 │
  │  Range: 2^(-127) to 2^(127)                 │
  └─────────────────────────────────────────────┘

Dequantized value: element_value * scale
```

### E8M0 Scale Factor Format

The E8M0 format is unique -- it has 8 exponent bits and zero mantissa bits:

```
E8M0 bit layout: [e7 e6 e5 e4 e3 e2 e1 e0]

Value = 2^(E - 127)    where E = e7*128 + e6*64 + ... + e0*1

Special cases:
  E = 0:   represents 2^(-127) (smallest positive)
  E = 254: represents 2^(127)  (largest)
  E = 255: represents NaN

Properties:
  - Only exact powers of 2 can be represented
  - Multiplication by E8M0 is equivalent to integer exponent addition
  - No rounding error in scale application (exact operation)
  - 255 distinct scale values spanning 254 orders of magnitude
```

The power-of-2 constraint is critical for hardware efficiency: applying the scale factor requires only adding the E8M0 exponent to the element's exponent, which is a simple integer operation in the tensor core.

### MXFP8: 8-Bit Elements with Block Scaling

**Element formats:** E4M3 (OCP standard) or E5M2

```
E4M3 element (8 bits): [S | E3 E2 E1 E0 | M2 M1 M0]
  Sign:     1 bit
  Exponent: 4 bits (bias = 7)
  Mantissa: 3 bits (precision: 4 significant bits including implicit 1)
  Range:    +/- 448  (without scale)
  With E8M0 scale: +/- 448 * 2^127 = astronomically large effective range

E5M2 element (8 bits): [S | E4 E3 E2 E1 E0 | M1 M0]
  Sign:     1 bit
  Exponent: 5 bits (bias = 15)
  Mantissa: 2 bits (precision: 3 significant bits)
  Range:    +/- 57,344  (without scale)
```

**Storage analysis for attention (sequence length N, head dim d):**

```
Q tensor: N * d elements
  FP16:  N * d * 2 bytes
  MXFP8: N * d * 1 byte + ceil(N * d / 32) * 1 byte
       = N * d * 1.03125 bytes
  Savings: 1.94x over FP16

KV cache per layer (2 * N * d):
  FP16:  4 * N * d bytes
  MXFP8: 2 * N * d * 1.03125 bytes = 2.0625 * N * d bytes
  Savings: 1.94x over FP16
```

### MXFP6: 6-Bit Elements with Block Scaling

Two sub-formats optimized for different use cases:

```
E2M3 element (6 bits): [S | E1 E0 | M2 M1 M0]
  Sign:     1 bit
  Exponent: 2 bits (bias = 1)
  Mantissa: 3 bits
  Range:    +/- 7.5 (without scale)
  Precision: 4 significant bits -- same as E4M3 but much narrower range
  Best for: Forward pass where precision matters more than element-level range

E3M2 element (6 bits): [S | E2 E1 E0 | M1 M0]
  Sign:     1 bit
  Exponent: 3 bits (bias = 3)
  Mantissa: 2 bits
  Range:    +/- 28 (without scale)
  Precision: 3 significant bits
  Best for: When slightly wider element range is needed
```

**Storage:** 6 bits per element is sub-byte, requiring packing logic. Two elements occupy 12 bits (1.5 bytes), so elements are typically packed 4 per 3 bytes.

### MXFP4: 4-Bit Elements with Block Scaling

```
E2M1 element (4 bits): [S | E1 E0 | M0]
  Sign:     1 bit
  Exponent: 2 bits (bias = 1)
  Mantissa: 1 bit
  Representable values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} and negatives
  Range:    +/- 6.0 (without scale)
  With E8M0 scale: +/- 6.0 * 2^127

Properties:
  - Only 15 distinct magnitudes (plus zero and sign)
  - 4x compression over FP16
  - 2x compression over FP8
  - Requires careful calibration for attention
  - Sub-byte: two elements packed per byte
```

## Comparison: Per-Tensor vs Per-Block vs MX Scaling

### Scaling Strategy Spectrum

```
Coarsest                                              Finest
  ├──────────┼──────────┼──────────┼──────────┤
  Per-Tensor   Per-Channel  Per-Block    Per-Element
  (1 scale     (1 scale     (1 scale     (full FP)
   for all)     per row/     per 32
                column)      elements)

MX formats use per-block (per-32-element) scaling.
```

### Why Per-Block Matters for Attention

Attention scores have extreme within-row variation:

```
Example: attention score row for token at position 100:
  positions 0-30:   scores near -50 (irrelevant tokens)
  positions 31-60:  scores near 0 (slightly relevant)
  positions 95-100: scores near +20 (highly relevant, recent context)
  positions 101+:   masked to -inf (causal mask)

Per-tensor FP8: single scale for the entire [N, N] score matrix
  - Dominated by the largest value across ALL rows
  - Most values quantized to near-zero
  - RMSE: 2.4e-2

MX per-block FP8: one scale per 32 elements in each row
  - Block [0:32]: scale matches the -50 range -> good precision
  - Block [32:64]: scale matches the 0 range -> good precision
  - Block [96:128]: scale matches the +20 range -> good precision
  - RMSE: ~1.5e-2

MX + incoherent processing: same block scaling but outliers are smoothed
  - RMSE: 9.1e-3 (2.6x better than per-tensor)
```

### Accuracy Results from FlashAttention-3

| Method | RMSE | Relative to Baseline |
|--------|------|---------------------|
| Per-tensor FP8 (naive) | 2.4e-2 | 1.0x |
| Per-block FP8 (no incoherent processing) | ~1.5e-2 | ~1.6x better |
| Per-block FP8 + incoherent processing (MXFP8-style) | 9.1e-3 | 2.6x better |
| FP16 reference | ~1e-4 | ~240x better |

### Training Convergence

Experiments with 8B parameter models (Nemotron 8B) demonstrate:

```
Validation perplexity comparison:
  BF16 baseline:    ~1.02-1.05 (across training steps)
  MXFP8 (per-block): Tracks BF16 within noise margin
  Per-tensor FP8:    Slight degradation in later training stages

Key finding: "MXFP8 follows closely that of BF16, indicating that
MXFP8 converges as well as BF16" for pretraining.
```

## Hardware Implementation

### Blackwell Native MX Support

NVIDIA Blackwell's tensor cores handle MX formats natively through the UMMA instruction:

```
UMMA Instruction Descriptor (32 bits):
  Bits [31:30]: max_shift field (2 bits)
    0 = NoShift    (standard FP8/FP16, no MX scaling)
    1 = MaxShift8  (apply MX scaling for 8-bit elements)
    2 = MaxShift16 (apply MX scaling for 16-bit elements)
    3 = MaxShift32 (apply MX scaling for 32-bit elements)

  Bits [12:7]: a_format / b_format (3 bits each)
    For MX formats:
      0 = E4M3 (MXFP8)
      1 = E5M2 (MXFP8)
      3 = E2M3 (MXFP6)
      4 = E3M2 (MXFP6)
      5 = E2M1 (MXFP4)
```

When max_shift is nonzero, the tensor core:
1. Reads the 32-element block data from SMEM
2. Reads the corresponding E8M0 scale factor
3. Applies the scale during the multiply-accumulate (exponent addition)
4. Accumulates in FP32

This happens entirely in hardware with zero software overhead.

### Memory Layout for MX on Blackwell

```
SMEM layout for MXFP8 operand B (K or V matrix):
  ┌──────────────────────────────────────────────────┐
  │ Element data: N x d bytes (E4M3, 1 byte each)   │
  │ Contiguous, swizzled as per UMMA descriptor      │
  ├──────────────────────────────────────────────────┤
  │ Scale factors: ceil(N*d/32) bytes (E8M0)         │
  │ One byte per block of 32 elements                │
  └──────────────────────────────────────────────────┘

The tensor core interleaves access to elements and scales automatically.
```

### Hopper Software Emulation

On Hopper (H100), MX block scaling is implemented in software:

```python
# FlashAttention-3 approach: block quantization before WGMMA
def prepare_fp8_blocks(tensor, block_size=32):
    flat = tensor.reshape(-1, block_size)
    amax = flat.abs().amax(dim=-1, keepdim=True)
    # Round to power of 2 (E8M0 constraint)
    scale = torch.pow(2, torch.floor(torch.log2(amax.clamp(min=1e-38))))
    quantized = (flat / scale).to(torch.float8_e4m3fn)
    # Scale factors applied during accumulation (not native)
    return quantized, scale

# The WGMMA instruction on Hopper does NOT have max_shift
# Dequantization must happen in software between GEMM tiles
```

### AMD Block-Scale Implementation

AMD's Composable Kernel implements FP8 block scaling on CDNA architectures with configurable block sizes:

```cpp
// From CK fmha_fwd kernel (gfx9)
// Block scale is applied during GEMM accumulation
struct FmhaFwdCommonBlockScaleKargs {
    index_t nhead_stride_q_descale;
    index_t nhead_stride_k_descale;
    index_t nhead_stride_v_descale;
    index_t block_scale_size_q;    // Configurable (not fixed at 32)
    index_t block_scale_size_kv;
};

// Apply descale during score accumulation
float k_descale = k_descale_ptr[kv_idx];
auto s_acc = s_acc_raw * k_descale;

// FP8 shift for softmax numerical stability
#if CK_TILE_USE_OCP_FP8
    validated_m -= 8.0f;   // OCP FP8 shift
#else
    validated_m -= 7.0f;   // FNUZ FP8 shift
#endif
```

Key difference: AMD's block size is configurable (not fixed at 32), and dequantization happens in software during the GEMM accumulation phase.

## MX Format Selection Guide for Attention

### Decision Matrix

```
                        Precision Needed
                     High ◄───────────► Low
                   ┌──────────────────────────┐
    Throughput   H │  BF16/FP16              │
    Needed       i │  (no quantization)       │
                 g │                          │
                 h │  MXFP8 (E4M3)           │  MXFP6 (E2M3)
                   │  Best general-purpose    │  Higher compression
                 ◄─┤                          │
                   │  MXFP8 (E5M2)           │  MXFP6 (E3M2)
    Throughput   L │  (gradients only)        │
    Needed       o │                          │  MXFP4 (E2M1)
                 w │  Per-tensor FP8          │  Maximum compression
                   │  (simplest, least        │  (inference only,
                   │   accurate)              │   requires calibration)
                   └──────────────────────────┘
```

### Recommendation by Use Case

| Use Case | Recommended Format | Rationale |
|----------|-------------------|-----------|
| LLM inference, Blackwell | MXFP8 (E4M3) | Native HW support, 2x throughput, minimal accuracy loss |
| LLM training, Blackwell | MXFP8 (E4M3 fwd, E5M2 bwd) | Matches BF16 convergence, doubles TFLOPS |
| Long-context inference | MXFP8 | 2x KV cache compression, block scaling handles score variation |
| KV cache compression | MXFP4 or MXFP8 | 4x or 2x KV cache size reduction |
| Inference, extreme compression | MXFP4 (E2M1) | 4x throughput on Blackwell, requires calibration |
| Training on Hopper | Per-block FP8 (software) | No native MX, but block scaling improves accuracy over per-tensor |
| AMD MI300X inference | FP8 with block descale | CK supports configurable block sizes |

## Incoherent Processing for MX Accuracy

FlashAttention-3 introduces incoherent processing to improve MX/block-scaled FP8 accuracy:

### The Problem: Outliers

LLM activations contain outlier values -- a small fraction of elements with disproportionately large magnitudes. Even with per-block scaling, these outliers dominate their block's scale factor, forcing the remaining 31 elements into low-precision representation.

### The Solution: Randomized Hadamard Transform

```
Before quantization:
  x = [0.1, 0.2, ..., 100.0, ..., 0.3]  (outlier at position k)
  Block amax dominated by 100.0
  Other values lose precision

After Hadamard transform:
  x' = H * S * x   (H = Hadamard matrix, S = random sign diagonal)
  x' = [2.5, -1.8, 3.1, ..., 2.9, ...]  (outlier energy spread)
  Block amax ~3.1, all values well-represented
  O(d log d) computation cost
```

The Hadamard transform:
1. Spreads outlier energy across all dimensions
2. Makes the distribution more uniform (better for block quantization)
3. Is orthogonal (preserves dot products after inverse transform)
4. Can be fused with rotary embedding for near-zero overhead

### Accuracy Impact

```
FP8 Attention RMSE (Q, K, V from N(0,1) + 0.1% outliers):
  Without incoherent processing:
    Per-tensor FP8:    RMSE = 2.4e-2
    Per-block FP8:     RMSE = 1.5e-2

  With incoherent processing:
    Per-block FP8:     RMSE = 9.1e-3  (2.6x improvement over baseline)
```

## Storage and Memory Bandwidth Analysis

### MXFP8 vs FP16 for Attention KV Cache

For a model with:
- 32 layers, 32 heads, d=128
- Sequence length N = 32,768

```
Per-token KV storage:
  FP16:  2 * 32 * 32 * 128 * 2 bytes = 512 KB
  MXFP8: 2 * 32 * 32 * 128 * (1 + 1/32) bytes = 264 KB
  MXFP4: 2 * 32 * 32 * 128 * (0.5 + 1/32) bytes = 140 KB

Total KV cache for N=32768:
  FP16:  16 GB
  MXFP8: 8.25 GB
  MXFP4: 4.38 GB
```

The memory bandwidth savings directly translate to throughput improvements in memory-bound attention inference.

### Scale Factor Storage Overhead

```
MXFP8: 1 E8M0 byte per 32 E4M3 bytes  = 3.125% overhead
MXFP6: 1 E8M0 byte per 24 bytes (32 * 6/8)  = 4.167% overhead
MXFP4: 1 E8M0 byte per 16 bytes (32 * 4/8)  = 6.25% overhead
```

The overhead is small enough that it does not significantly impact the compression ratio or memory bandwidth savings.

## Implementation Considerations

### Block Alignment in Attention Tensors

The MX block size of 32 interacts with attention tile dimensions:

```
Head dimension d:
  d=64:  2 MX blocks per row (aligned)
  d=128: 4 MX blocks per row (aligned)
  d=256: 8 MX blocks per row (aligned)
  d=96:  3 MX blocks per row (aligned)

Sequence dimension (within attention tile of size n_block):
  n_block=128: 4 MX blocks (aligned)
  n_block=64:  2 MX blocks (aligned)
  n_block=32:  1 MX block (aligned)

All standard attention configurations align naturally with block_size=32.
```

### Scale Factor Computation

The E8M0 constraint (power-of-2 only) means the scale is computed as:

```python
# Correct: round amax to nearest power of 2
scale = 2 ** floor(log2(amax))

# This is NOT the same as:
scale = amax  # This would require a general float, not E8M0

# The power-of-2 constraint introduces up to 2x quantization loss
# compared to optimal (non-power-of-2) per-block scaling.
# In practice, this loss is negligible compared to the element
# quantization error.
```

### Softmax Numerical Stability with MX

When attention scores are in MX format, the softmax computation requires care:

```python
# Scores are in MXFP8: S[i] = element[i] * block_scale[i // 32]
# For softmax: exp(S[i] - max(S))

# The block scale shifts the effective range of scores
# AMD CK applies an explicit shift before exp2:
#   For OCP FP8: shift = 8.0 (accounts for E4M3 bias)
#   For FNUZ FP8: shift = 7.0
# This keeps the exp2 argument in a safe range

# On Blackwell, the dequantized scores in TMEM are already in FP32,
# so standard softmax applies after the QK^T MMA.
```

## Integration with Attention Frameworks

### FlashAttention-3 (Hopper, Software MX)

FA3 implements block quantization as a preprocessing step:

```
1. Apply Hadamard transform to Q, K, V (optional, for incoherent processing)
2. Quantize Q, K, V to FP8 E4M3 with per-block scale factors
3. Pass FP8 data + scale factors to the WGMMA-based attention kernel
4. Inside kernel: dequantize during GEMM tile accumulation
5. Accumulate in FP32, compute softmax in FP32
```

### FlashAttention-4 (Blackwell, Native MX)

FA4 leverages Blackwell's native MX support:

```
1. Store Q, K, V in MXFP8 format (elements + E8M0 scales)
2. Load via TMA to SMEM (elements and scales together)
3. UMMA instruction reads both and applies scaling in hardware
   - max_shift=MaxShift8 in the instruction descriptor
4. Accumulate in TMEM (FP32)
5. Softmax operates on FP32 values in TMEM
```

### FlashInfer (Blackwell, NVFP4 support)

FlashInfer supports NVFP4 (MXFP4-compatible) output:

```python
# FlashInfer decode attention with NVFP4 output
from flashinfer.utils import FP4Tensor

result = batch_decode_with_kv_cache(
    query=query,
    kv_cache=kv_cache,
    out_dtype="nvfp4",        # Request MXFP4 output
    o_sf_scale=1.0,           # Output scale factor scale
    o_sf_vec_size=16,         # Scale factor vector size
)
# Returns FP4Tensor with .data (uint8, packed) and .scale (float8_e4m3fn)
```

## Future Directions

### MX v2 and Beyond

The OCP MX specification is versioned, with v1.0 as the current standard. Expected evolution:

- **Larger block sizes** (64, 128 elements) for even lower scale overhead
- **Multi-level scaling** (block + tensor scale) for wider effective range
- **FP3/FP2 elements** for extreme compression in inference-only scenarios
- **Training-aware quantization** calibration integrated into optimizer steps

### Cross-Vendor Convergence

The OCP standard ensures MX compatibility across vendors:
- NVIDIA: Native in Blackwell tensor cores (UMMA max_shift)
- AMD: Software support in Composable Kernel (CDNA2/3); potential hardware support in future CDNA4
- Intel: MX support planned for Gaudi3 and future data center GPUs
- The standardization on block_size=32 and E8M0 ensures model portability

## References

- [OCP Microscaling Formats (MX) v1.0 Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [NVIDIA FP8 Introduction Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [NVIDIA Transformer Engine: FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [FlashAttention-3 Paper (arXiv 2407.08608)](https://arxiv.org/abs/2407.08608)
- [FP8 Formats for Deep Learning (arXiv 2209.05433)](https://arxiv.org/abs/2209.05433)
- [CUTLASS SM100 MMA Descriptor (mma_sm100_desc.hpp)](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp)
- [AMD Composable Kernel FP8 Block-Scale FMHA](https://github.com/ROCm/composable_kernel)
- [NVIDIA Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
