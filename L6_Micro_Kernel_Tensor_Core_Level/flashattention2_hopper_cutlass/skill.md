---
skill_name: FlashAttention-2 Microkernel Design on Hopper with CUTLASS
description: Tile shape selection, register pressure analysis, and dual-GEMM pipelining for FlashAttention-2 on H100 using WGMMA and TMA
level: L6 - Micro-Kernel/Tensor-Core Level
target_hardware: NVIDIA Hopper H100/H200 (SM90)
relevance: When implementing or tuning FlashAttention forward pass kernels on Hopper and needing to choose tile shapes, manage register pressure between dual GEMMs, and configure WGMMA SS/RS variants
---

# FlashAttention-2 Microkernel Design on Hopper with CUTLASS

## What It Is
A detailed case study implementing FlashAttention-2's forward pass on Hopper using CUTLASS, achieving 20-50% higher FLOPS/s over the Ampere-optimized version. The key contribution is a systematic analysis of how tile shape selection interacts with register pressure in fused attention kernels, where two GEMMs (QK^T and PV) share register space with online softmax state. The study reveals that the 128x128 tile -- optimal for standalone GEMM -- catastrophically fails for fused attention at large head dimensions due to register spills.

## Key Concepts
- **Dual-GEMM Fusion**: GEMM-I (QK^T) uses SS variant (both operands from SMEM); GEMM-II (PV) uses RS variant (P from registers, V from SMEM)
- **Register Pressure Crisis**: At head_dim=256, the 128x128 tile collapses to 36.7 TFLOPS (vs 308.1 for 64x64) because GEMM-II needs P + O accumulators simultaneously in registers
- **Accumulator Z-Pattern**: WGMMA distributes 128 threads x 32 values in a Z-pattern; softmax reductions must account for this layout using shuffle-based quad reductions
- **ReshapeTStoTP**: Layout transformation converting GEMM-I's accumulator shape into GEMM-II's operand shape without data movement
- **Natural COPY-GEMM Pipelining**: Load V during GEMM-I execution, load next K during GEMM-II execution
- **One Warpgroup per CTA**: Wastes register space but avoids the complexity of warp-specialized multi-warpgroup designs

## Tile Configuration / Code Pattern
```cpp
// GEMM-I: QK^T (both Q and K from shared memory)
using TiledMma0 = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<half_t, half_t, float,
        Shape<Int<bM>, Int<bN>, Int<bK>>>(),  // e.g., 64x64x16
    Layout<Shape<_1, _1, _1>>{}));  // 1 warpgroup

// GEMM-II: PV (P from registers after softmax, V from shared memory)
using TiledMma1 = decltype(cute::make_tiled_mma(
    cute::GMMA::rs_op_selector<half_t, half_t, float,
        Shape<Int<bM>, Int<bK>, Int<bN>>,
        GMMA::Major::K, GMMA::Major::MN>(),
    Layout<Shape<_1, _1, _1>>{}));

// V transposition for PV (CUTLASS computes AB^T, we need PV not PV^T)
auto smemLayoutVt = composition(smemLayoutV,
                                make_layout(tileShapeVt, GenRowMajor{}));

// Natural pipelining: overlap load with compute
cfk::copy_nobar(tVgV, tVsV, tmaLoadV, tma_load_mbar[1]);  // Load V
cfk::gemm_bar_wait(tiledMma0, tSrQ, tSrK, tSrS, tma_load_mbar[0]);  // GEMM-I
cfk::copy_nobar(tKgK, tKsK, tmaLoadK, tma_load_mbar[0]);  // Load next K
// ... softmax on S ...
cfk::gemm_bar_wait(tiledMma1, tOrP, tOrV, tOrO, tma_load_mbar[1]);  // GEMM-II

// Softmax reduction using shuffle (quad-level, no atomics)
auto max_quad_0 = ShflReduce<4>::run(max0, maxOp);  // Row 0
auto max_quad_1 = ShflReduce<4>::run(max1, maxOp);  // Row 1
```

## Performance Impact
- **64x64 tiles**: 230-308 TFLOPS across head dimensions (most consistent)
- **64x128 tiles**: 259-289 TFLOPS (best for head_dim=64, degrades at 256)
- **128x64 tiles**: 248-296 TFLOPS (good for head_dim=128, collapses at 256 to 39 TFLOPS)
- **128x128 tiles**: 251 TFLOPS at head_dim=64, collapses to 36.7 at head_dim=256
- **Overall**: 20-50% faster than Ampere-optimized FlashAttention-2; 2.5-3x faster than CUTLASS 3.3 FMHA
- **H100 PCIe peak**: 756 TFLOPS FP16; best result ~308 TFLOPS (~41% utilization)

## When to Use
- Implementing FlashAttention forward pass on Hopper with direct CUTLASS kernel development
- Choosing tile shapes for attention kernels where head dimension varies (64, 128, 256)
- Diagnosing register spill issues in fused attention kernels (NCU register count analysis)
- Understanding the SS vs RS WGMMA variant selection for attention's two GEMMs
- Building the foundation before advancing to warp-specialized ping-pong designs (FlashAttention-3)

## When NOT to Use
- For backward pass implementation (different register pressure profile, not covered here)
- On Blackwell GPUs where UMMA/tcgen05 replaces WGMMA with different constraints
- When using pre-built FlashAttention libraries that already incorporate these optimizations
- For non-attention GEMM workloads where register pressure from softmax state is not a factor

## Key Takeaways
- **Tile shape selection for attention is fundamentally different from standalone GEMM**: The fused softmax state (m, l, O accumulator) creates register pressure that does not exist in pure GEMM
- **64x64 is the safest default** for FlashAttention on Hopper across all head dimensions; 64x128 can be better for small head dimensions
- **128x128 is dangerous** for fused attention: it works at head_dim=64 but catastrophically fails at head_dim=256 (8x slowdown from register spills)
- **SS for GEMM-I, RS for GEMM-II** is the canonical WGMMA variant assignment for FlashAttention
- **The accumulator Z-pattern** directly impacts softmax implementation: row reductions need two separate max/sum tracks and shuffle-based quad reductions
- **One warpgroup per CTA** is simpler but wastes half the available register file; warp specialization (ping-pong) is the path to closing this gap
- **Natural dual-GEMM pipelining** (load V during GEMM-I, load K during GEMM-II) provides latency hiding without complex software pipelining infrastructure

## References
- [A Case Study in CUDA Kernel Fusion: FlashAttention-2 on Hopper (arXiv)](https://arxiv.org/html/2312.11918v1)
- [CUTLASS GitHub Repository](https://github.com/NVIDIA/cutlass)
- [Colfax WGMMA Tutorial](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- [Colfax Pipelining Tutorial](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)
