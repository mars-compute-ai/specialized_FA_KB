---
skill_name: Fast Softmax CUDA Kernel Implementation
description: Warp-level reductions, vectorized memory access, and block-level strategies for writing high-performance softmax CUDA kernels.
level: L4 - Compute Kernel Optimization Level
target_hardware: NVIDIA GPUs (A100, H100, B200 and similar)
relevance: When writing custom softmax kernels in CUDA, optimizing standalone softmax layers, or understanding the low-level building blocks used inside fused attention kernels.
---

# Fast Softmax CUDA Kernel Implementation

## What It Is
This guide from Facebook's AITemplate project covers the practical engineering of high-performance softmax CUDA kernels. It details two primary strategies: warp-level reductions using `__shfl_xor_sync()` for small reduction dimensions (K < ~2000) where all data fits in registers, and block-level reductions using shared memory for larger K. Combined with vectorized memory operations (half4/float4 packing), these techniques achieve up to 50% speedup over state-of-the-art implementations on A100 GPUs. The guide provides complete kernel code, thread organization strategies, and performance crossover thresholds.

## Key Concepts
- **Warp-level reduction via `__shfl_xor_sync()`**: 32 threads exchange register values directly without shared memory; reduces max/sum in 5 iterations (log2(32))
- **Vectorized loads**: Reading `half4` or `float4` packs reduces memory transaction count by 4-8x
- **Block-level reduction**: Two-stage approach -- reduce within warps, write to shared memory, reduce across warp results
- **Register-resident data**: The `buf[]` array holding input data lives in registers for fast reuse across max, exp, and sum phases
- **Thread organization**: Warp mode uses `<32, 4>` blocks; block mode uses {128, 256, 512, 1024} threads with occupancy-based selection
- **fp32 intermediates**: All max, exp, sum computed in fp32 even for fp16/bf16 inputs to maintain numerical stability
- **Pack size determines crossover**: Larger packs shift the warp-to-block threshold upward (1408 for pack=1, 3840 for pack=8)

## Algorithm / Pseudo-code
```cuda
// Warp-level softmax kernel (small K, all data in registers)
// Grid: <M/pack_rows>, Block: <32, pack_rows>

__global__ void softmax_warp(half4* input, half4* output, int M, int N) {
    float4 buf[NUM_PACKS];          // register-resident data
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;           // lane within warp

    // Step 1: Load data into registers, find local max
    float local_max = -INF;
    for (int p = 0; p < NUM_PACKS; p++) {
        int col = p * 32 + tid;
        buf[p] = load_half4_as_float4(input[row * N/4 + col]);
        local_max = max(local_max, max_of_float4(buf[p]));
    }
    // Reduce max across 32 threads via warp shuffle
    warpReduceMax(&local_max);       // 5 iterations of __shfl_xor_sync

    // Step 2: Compute exp(x - max) and sum, reusing register data
    float local_sum = 0.0f;
    for (int p = 0; p < NUM_PACKS; p++) {
        buf[p].x = expf(buf[p].x - local_max);
        buf[p].y = expf(buf[p].y - local_max);
        buf[p].z = expf(buf[p].z - local_max);
        buf[p].w = expf(buf[p].w - local_max);
        local_sum += buf[p].x + buf[p].y + buf[p].z + buf[p].w;
    }
    warpReduceSum(&local_sum);       // 5 iterations of __shfl_xor_sync

    // Step 3: Normalize and write output
    float inv_sum = 1.0f / local_sum;
    for (int p = 0; p < NUM_PACKS; p++) {
        int col = p * 32 + tid;
        output[row * N/4 + col] = float4_to_half4(buf[p] * inv_sum);
    }
}

// Block-level softmax kernel (large K, requires shared memory)
// Grid: <M>, Block: <block_size>
// Same algorithm but warpReduceMax/Sum replaced with blockReduceMax/Sum
// which uses __shared__ memory for inter-warp communication
```

## Numerical Considerations
- **Always subtract max before exp()**: Prevents overflow in the exponential; this is "safe softmax"
- **fp32 accumulation is essential**: Even with fp16/bf16 inputs, max/sum/exp must use fp32 to avoid catastrophic precision loss
- **Warp shuffle preserves precision**: Register-to-register transfer has no precision loss
- **Shared memory bank conflicts**: Block reduction uses padding (`shared[NUM][33]` instead of `[32]`) to avoid bank conflicts
- **Large K values**: When K > ~4000, data cannot fit in registers; block reduction adds shared memory latency but is unavoidable
- **Output precision**: Final division by sum is done in fp32 before converting back to fp16/bf16

## When to Use
- Writing standalone softmax kernels for layers outside of attention (e.g., classification heads, mixture-of-experts routing)
- Optimizing softmax within custom fused kernels
- Reduction dimensions K < 4000 (the common case for vocabulary-free softmax, attention heads)
- When PyTorch/framework softmax is a bottleneck identified by profiling
- Learning CUDA kernel optimization techniques applicable beyond softmax

## When NOT to Use
- Softmax within attention: use FlashAttention's fused online softmax instead (avoids materializing the attention matrix)
- Very large reduction dimensions (K > 32K): consider multi-block approaches with global memory atomics
- When Triton or framework-provided fused kernels already achieve satisfactory performance
- Prototyping where development speed matters more than runtime performance

## Key Takeaways
- **Warp shuffles are the key primitive**: `__shfl_xor_sync()` enables register-only reductions that avoid shared memory latency entirely
- **Vectorized loads (half4/float4) provide up to 50% speedup** by reducing memory transaction count
- **The warp-to-block crossover is around K=1400-3800** depending on pack size; choose the right strategy based on reduction dimension
- **Register-resident data enables single-load, multi-use patterns**: load once, reuse for max, exp, sum, and normalize
- **fp32 intermediates are non-negotiable** for numerical stability with fp16/bf16 data types
- **Memory bandwidth, not compute, is the bottleneck** for standalone softmax kernels

## References
- [AITemplate Wiki: How to write a fast Softmax CUDA kernel](https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F)
- [OneFlow: How to Implement an Efficient Softmax CUDA Kernel](https://oneflow2020.medium.com/how-to-implement-an-efficient-softmax-cuda-kernel-oneflow-performance-optimization-sharing-405ad56e9031)
- [Triton: Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [FastSoftmax GitHub Repository](https://github.com/SzymonOzog/FastSoftmax)
