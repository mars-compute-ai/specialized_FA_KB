# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

**Authors:** Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
**Published:** July 11, 2024
**arXiv:** [2407.08608](https://arxiv.org/abs/2407.08608)
**Code:** [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

## Abstract

Attention, as a core layer of the ubiquitous Transformer architecture, is the bottleneck for large language models and long-context applications. FlashAttention elaborated an approach to speed up attention on GPUs through minimizing memory reads/writes. However, it has yet to take advantage of new capabilities present in recent hardware, with FlashAttention-2 achieving only 35% utilization on the H100 GPU. FlashAttention-3 develops three main techniques for Hopper GPU optimization: exploiting asynchrony of Tensor Cores and TMA to overlap computation and data movement via warp-specialization, interleaving block-wise matmul and softmax operations, and block quantization and incoherent processing that leverages hardware support for FP8 low-precision.

## Performance Results

### FP16 Performance
- FlashAttention-3 achieves up to **740 TFLOPS** on H100, representing **75% utilization** of theoretical max FLOPs
- This is a **1.5-2.0x speedup** over FlashAttention-2 with FP16
- Forward pass improvements from ~570 TFLOPS (with WGMMA alone) to 620-660 TFLOPS through overlapping techniques
- With BF16, reaches up to **840 TFLOPS (85% utilization)**

### FP8 Performance
- Reaches close to **1.2 PFLOPS** (1,200 TFLOPS)
- Updated results report up to **1.3 PFLOPS** with BF16
- FP8 Tensor Cores deliver **1978 TFLOPS** theoretical throughput vs. **989 TFLOPS** for FP16 (2x hardware throughput)

### Hardware Utilization Improvement
- FlashAttention-2: **35%** utilization on H100
- FlashAttention-3 FP16: **75%** utilization on H100
- FlashAttention-3 BF16: **85%** utilization on H100

## Key Hardware Features Exploited on H100 Hopper

### WGMMA (Warpgroup Matrix Multiply-Accumulate)
Delivers significantly higher throughput than older mma.sync instructions on Ampere GPUs, enabling faster matrix operations. This is the fundamental compute primitive.

### TMA (Tensor Memory Accelerator)
Specialized hardware unit that accelerates data transfers between global memory and shared memory, freeing up registers for larger tile sizes and decoupling data movement from computation.

### FP8 Tensor Cores
Hopper's FP8 support doubles Tensor Core throughput compared to FP16:
- **FP8:** 1978 TFLOPS theoretical peak
- **FP16:** 989 TFLOPS theoretical peak

## Asynchrony Techniques

### Inter-Warpgroup Overlapping (Pingpong Scheduling)
Separate warpgroups (groups of 4 warps) synchronize with barriers (bar.sync). One warpgroup executes GEMMs while another performs softmax computations in parallel. This improves throughput from 570 to 620 TFLOPS.

### Intra-Warpgroup Overlapping
Within a single warpgroup, softmax operations partially execute during GEMM computation through pipelining. This advances performance to 640-660 TFLOPS despite higher register pressure.

## Block Quantization and Incoherent Processing for FP8

### The Problem
Moving from FP16 to FP8 introduces quantization error that can degrade attention accuracy. LLM activations often contain outliers (a small fraction of entries with disproportionately large magnitudes) which exacerbate quantization error.

### Block Quantization
Rather than quantizing entire tensors with a single scale factor, FlashAttention-3 divides Q, K, V matrices into blocks and applies per-block scale factors. This better preserves the dynamic range within each block.

NVIDIA Blackwell's MXFP8 further refines this with block-level scaling where each contiguous block of 32 values receives a distinct scaling factor, executed natively by the GPU's Tensor Cores.

### Incoherent Processing via Randomized Hadamard Transform
The key technique to maintain FP8 accuracy:

1. **Concept:** Multiply Q and K matrices by a random orthogonal matrix (Hadamard matrix with random signs) to "spread out" outlier values before quantization
2. **Algorithm:** Apply H * S * x where H is a Hadamard matrix and S contains random +/-1 diagonal elements
3. **Runtime:** O(d log d) where d = head dimension, versus O(d^2) for dense matrix multiplication
4. **Fusion:** The Hadamard transform is memory-bandwidth bound, so it can be fused with other bandwidth-bound operations like rotary embedding "for free"
5. **Implementation:** Includes in-kernel transpose operations for FP8 processing

### FP8 Accuracy Results

In experiments with Q, K, V generated from a standard normal distribution with 0.1% of entries having large magnitudes (simulating outliers):

| Method | RMSE |
|--------|------|
| Baseline FP8 (per-tensor quantization) | 2.4e-2 |
| FP8 with block quantization (no incoherent processing) | ~1.5e-2 |
| FP8 with block quantization + incoherent processing | 9.1e-3 |

**Incoherent processing reduces quantization error by 2.6x** compared to the baseline FP8 approach.

## Algorithmic Foundation

FlashAttention-3 builds upon the original FlashAttention approach of reordering the attention computation and leveraging tiling and recomputation. The core mechanism uses block-wise operations to minimize HBM (GPU memory) reads/writes, reducing memory overhead from quadratic to linear in sequence length.

The key innovations are hardware-specific:
- Leverages NVIDIA's **CUTLASS** library abstractions for hardware-specific optimizations
- Integrates **warp-specialization** patterns with separate producer (TMA) and consumer (WGMMA) warps
- Supports **variable length sequences** and persistent kernels
- Uses **in-kernel transpose** operations for FP8 processing

## FP8 Format Details

FlashAttention-3 uses the two FP8 sub-formats:
- **E4M3:** 4 exponent bits, 3 mantissa bits, range +/-448. Used for forward pass (Q, K, V storage)
- **E5M2:** 5 exponent bits, 2 mantissa bits, range +/-57,344. Used where wider dynamic range is needed (backward pass gradients)

## References

- [FlashAttention-3 Paper (arXiv)](https://arxiv.org/abs/2407.08608)
- [FlashAttention-3 Blog Post](https://tridao.me/blog/2024/flash3/)
- [GitHub Repository](https://github.com/Dao-AILab/flash-attention)
- [OpenReview](https://openreview.net/forum?id=tVConYid20)
