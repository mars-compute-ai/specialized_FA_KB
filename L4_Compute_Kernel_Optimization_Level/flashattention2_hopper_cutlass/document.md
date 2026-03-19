# A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using CUTLASS

Source: https://arxiv.org/html/2312.11918v1

## Executive Summary

This case study documents an optimized implementation of FlashAttention-2's forward pass targeting NVIDIA Hopper (SM90) architecture using the CUTLASS library. The implementation achieves 20-50% higher FLOPS/s over FlashAttention-2 optimized for NVIDIA Ampere architecture on H100 PCIe GPUs.

## Core Algorithm

The implementation fuses attention computation into a single kernel by restructuring the algorithm to handle softmax computation during the inner loop. The algorithm processes Q, K, V matrices in tiles and maintains running statistics for row-wise softmax.

**Key algorithmic components:**
- Tiles Q matrix along the row dimension (bM = QBLK)
- Tiles K, V matrices with block size bN = KBLK
- Maintains accumulator O, row maximum m, and row sum l in registers
- Applies "online softmax" technique to avoid materializing intermediate attention matrices

## Tile Shape Selection and Register Pressure

### Configurations Explored

Four tile configurations for (QBLK x KBLK):
- 64x64
- 64x128
- 128x64
- 128x128

### Performance by Head Dimension (TFLOPS on H100 PCIe)

| Head Dim | 64x64 | 64x128 | 128x64 | 128x128 |
|----------|-------|--------|--------|---------|
| 64       | 230.1 | 259.5  | 247.9  | 251.4   |
| 128      | 292.6 | 289.3  | 295.7  | 208.7   |
| 256      | 308.1 | 276.1  | 39.3   | 36.7    |

### Critical Finding: Register Pressure

The 128x128 configuration suffers from severe register pressure and register spills. GEMM-II (PV multiplication) requires both operand A (the softmax output P) and accumulator C (the output O) in registers simultaneously. When insufficient register space exists, the compiler serializes operations, causing massive performance degradation.

At head dimension 256, the 128x64 and 128x128 configurations collapse to 39.3 and 36.7 TFLOPS respectively -- an 8-10x degradation from the 64x64 configuration's 308.1 TFLOPS.

### Optimal Configuration

The best-performing implementation uses **one warpgroup (128 threads) per CTA**, leaving register space for another warpgroup wasted. This is suboptimal but necessary given the register pressure constraints of fused attention kernels.

## CUTLASS/CuTe Implementation Details

### WGMMA Configuration

Two GMMA variants are used within the same kernel:
- **SS variant for GEMM-I (QK^T)**: Both Q and K loaded from shared memory

```cpp
using TiledMma0 = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<MmaA, MmaB, MmaC, Shape<bM, bN, bK>>(),
    MmaTileShape{}));
```

- **RS variant for GEMM-II (PV)**: P from registers (softmax output), V from shared memory

```cpp
using TiledMma1 = decltype(cute::make_tiled_mma(
    cute::GMMA::rs_op_selector<MmaA, MmaB, MmaC, Shape<bM, bK, bN>,
                               GMMA::Major::K, GMMA::Major::MN>(),
    MmaTileShape{}));
```

### Accumulator Layout

The WGMMA 64x64 accumulator layout distributes across 128 threads:

```cpp
CLayout_64x64 = Layout<Shape <Shape <_4, _8, _4>, Shape <_2, _2, _8>>,
                        Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>>
```

This encodes a (Thread, Value) -> (M, N) mapping where 128 threads each hold 32 values in a replicated Z-pattern.

### Layout Transformation: Accumulator to Operand

GEMM-I produces a 64x128 accumulator (S matrix) that must be reshaped into a 64x16 operand for GEMM-II input. The `ReshapeTStoTP()` method handles this:

```
tSrS: ptr[32b] o ((2,2,16),2,1):((1,2,4),64,0)     // Accumulator shape
tOrS: ptr[16b] o ((2,2,2),2,8):((1,2,4),8,16)       // Operand shape
tOrPLayout: ((2,2,2),2,8):((1,2,4),64,8)             // Final P layout
```

The transformation preserves physical data location while reinterpreting the stride pattern.

### Transposed Layout for GEMM-II

Standard CUTLASS GEMM computes C = AB^T. Since GEMM-II requires PV (not PV^T), a layout transformation is applied:

```cpp
auto smemLayoutVt = composition(smemLayoutV,
                                make_layout(tileShapeVt, GenRowMajor{}));
```

### TMA Configuration

TMA provides asynchronous global-to-shared memory copying:

```cpp
auto tmaQ = make_tma_copy(SM90_TMA_LOAD{}, gQ, smemLayoutQ,
                          tileShapeQ, Int<1>{});
```

Key TMA properties:
- Single-threaded programming model (one thread issues copy, others sync via barriers)
- K-major 128-byte swizzling for bank conflict mitigation
- Hardware-managed address generation and bounds checking

## Online Softmax Implementation

### Row-wise Reduction with Z-Pattern Awareness

The S matrix is distributed across threads in a replicated Z-pattern. Threads maintain **two separate row maximum values** to handle this:

```cpp
for (int k = 0; k < NT * size<2>(VT); ++k) {
    data[n] = FragValType(AccumType(data[n]) * scaleFactor);
    max0 = cutlass::fast_max(max0, AccumType(data[n])); n++;
    data[n] = FragValType(AccumType(data[n]) * scaleFactor);
    max0 = cutlass::fast_max(max0, AccumType(data[n])); n++;
    data[n] = FragValType(AccumType(data[n]) * scaleFactor);
    max1 = cutlass::fast_max(max1, AccumType(data[n])); n++;
    data[n] = FragValType(AccumType(data[n]) * scaleFactor);
    max1 = cutlass::fast_max(max1, AccumType(data[n])); n++;
}
```

### Shuffle-Based Reduction

Aggregation uses shuffle instructions within 4-thread quads (not atomics):

```cpp
auto max_quad_0 = ShflReduce<4>::run(max0, maxOp);
auto max_quad_1 = ShflReduce<4>::run(max1, maxOp);
mi(rowId) = max_quad_0;
mi(rowId + 1) = max_quad_1;
```

## COPY-GEMM Pipelining

Rather than full software pipelining, the design exploits FlashAttention's natural dual-GEMM structure:

1. Issue GEMM-I using current K tile (from SMEM)
2. Simultaneously load V tile via TMA for GEMM-II
3. Issue GEMM-II with loaded V tile
4. Simultaneously load next K tile for next iteration

```cpp
cfk::copy_nobar(tVgV(_, 0), tVsV(_, 0), tmaLoadV, tma_load_mbar[1]);
cfk::gemm_bar_wait(tiledMma0, tSrQ, tSrK, tSrS, tma_load_mbar[0]);
if (blockIdxY != (nTilesOfK - 1)) {
    cfk::copy_nobar(tKgK(_, 0), tKsK(_, 0), tmaLoadK, tma_load_mbar[0]);
}
cfk::gemm_bar_wait(tiledMma1, convert_type<PrecType, AccumType>(tOrP),
                   tOrV, tOrO, tma_load_mbar[1]);
```

This natural interleaving hides memory latency without substantial code restructuring.

## Performance Results (H100 PCIe, FP16, seq_len=4096, batch=4)

- **2.5-3x speedup** over CUTLASS 3.3 FMHA kernel
- **20-50% improvement** over FlashAttention-2 from Dao AI Lab
- Best configuration: 64x64 or 64x128 for head_dim <= 128; 64x64 for head_dim = 256
- H100 theoretical maximum: 756 TFLOPS FP16

## Lessons Learned

### 1. Layout Mastery is Essential
"A working understanding of Layouts and Tensors is essential for developing a fused kernel with CUTLASS." Layout transformations directly impact performance.

### 2. Register Pressure Dominates Tile Selection
Larger tiles do not guarantee better performance. The 128x128 tile collapses at head_dim=256 because GEMM-II needs both P (operand A) and O (accumulator) in registers simultaneously.

### 3. Warpgroup Utilization Tradeoffs
One warpgroup per CTA wastes register space but outperforms two-warpgroup configurations. The performance frontier involves careful register allocation rather than maximizing thread-level parallelism.

### 4. Swizzling is Non-Negotiable
K-major 128-byte swizzling for shared memory is required to mitigate bank conflicts at full throughput.

### 5. The Two-GEMM Structure Enables Natural Pipelining
FlashAttention's alternating QK^T and PV multiplications create natural overlap points where one GEMM's data loading hides behind the other GEMM's computation.

## Future Directions

1. **Two-warpgroup CTA with warp specialization** (ping-pong pattern)
2. **Enhanced software pipelining** beyond natural dual-GEMM structure
3. **Threadblock clusters and distributed shared memory** for K/V copying
