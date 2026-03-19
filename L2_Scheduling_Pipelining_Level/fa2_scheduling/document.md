# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

Source: https://hazyresearch.stanford.edu/blog/2023-07-17-flash2

## Overview

FlashAttention-2 is a GPU-optimized attention algorithm achieving approximately 2x speedup over the original FlashAttention by improving parallelism and work partitioning. The implementation reaches 230 TFLOPs/s on A100 GPUs and 335 TFLOPs/s on H100 GPUs.

## Core Technical Improvements

### 1. Warp-Level Work Partitioning: Sliced-Q vs. Sliced-K

**Original FlashAttention ("Sliced-K" approach):**
- Splits K and V across 4 warps while keeping Q accessible to all warps
- Requires all warps to write intermediate results to shared memory
- Necessitates synchronization (`__syncthreads()`) and accumulation steps
- Generates significant shared memory read/write overhead

**FlashAttention-2 ("Sliced-Q" approach):**
- Splits Q across 4 warps while keeping K and V accessible to all warps
- Each warp computes its slice of QK^T independently
- Multiplies result by shared V to obtain output slice
- Eliminates inter-warp communication and synchronization bottlenecks
- Each warp owns its output accumulator entirely -- no cross-warp reduction needed

### 2. GPU Occupancy and Sequence-Length Parallelism

The original version parallelized across batch size and head count only:
- Grid dimensions: (batch_size, num_heads)
- For scenarios with long sequences but small batch/head dimensions, occupancy suffered
- Fewer thread blocks could not fill the GPU's 108 streaming multiprocessors on A100

**FlashAttention-2 enhancement:**
- Additionally parallelizes along the sequence length dimension
- Grid dimensions: (batch_size, num_heads, num_q_tiles)
- Improves multiprocessor utilization when `batch_size * num_heads < 80`
- Critical for long-context inference where batch=1 and heads may be few (e.g., MQA)

### 3. Algorithmic Optimizations -- Prioritize Matmul FLOPs

The design prioritizes matrix-multiply operations over other floating-point work:
- A100 GPU theoretical throughput: **312 TFLOPs/s** (FP16/BF16 matmul) vs. **19.5 TFLOPs/s** (non-matmul FP32)
- That is a **16x** gap -- every non-matmul operation is 16x more expensive in opportunity cost
- Reengineered online softmax to reduce rescaling operations
- Streamlined bound-checking and causal masking logic
- Maintains algorithmic correctness without approximation

### 4. Loop Ordering

FlashAttention-2 swaps the loop order from the original:
- **FA1**: Outer loop over K/V tiles, inner loop over Q tiles
- **FA2**: Outer loop over Q tiles, inner loop over K/V tiles
- This change enables the sliced-Q warp partitioning and improves shared memory reuse for K/V

### 5. Implementation Foundation

Completely rewritten using NVIDIA CUTLASS 3.x and its CuTe core library, providing "clean abstractions and powerful building blocks" for optimization.

## Performance Results

### Attention Microbenchmarks (A100, BF16)

FlashAttention-2 demonstrates approximately 2x speedup versus FlashAttention and up to 9x improvement over standard PyTorch implementations across various configurations (with/without causal masking, head dimensions 64-128).

### End-to-End Model Training (A100)

| Model Configuration       | Baseline     | FlashAttention | FlashAttention-2 |
|---------------------------|-------------|----------------|-------------------|
| GPT3-1.3B, 2K context     | 142 TFLOPs/s | 189 TFLOPs/s   | 196 TFLOPs/s      |
| GPT3-1.3B, 8K context     | 72 TFLOPs/s  | 170 TFLOPs/s   | 220 TFLOPs/s      |
| GPT3-2.7B, 2K context     | 149 TFLOPs/s | 189 TFLOPs/s   | 205 TFLOPs/s      |
| GPT3-2.7B, 8K context     | 80 TFLOPs/s  | 175 TFLOPs/s   | 225 TFLOPs/s      |

Achieves **72% model FLOP utilization** on A100 (225 TFLOPs/s out of 312 TFLOPs/s theoretical) and 1.3x speedup over optimized FlashAttention implementations.

### Hardware Performance
- **A100 SXM:** 230 TFLOPs/s attention throughput
- **H100 SXM5:** 335 TFLOPs/s (without specialized features like TMA or 4th-gen Tensor Cores)
- Enables training with "16k longer context for the same price as previously training an 8k context model"

## Expanded Feature Support

- Head dimensions up to 256 (previous limit: 128)
- Multi-Query Attention (MQA) support
- Grouped-Query Attention (GQA) support
- Enables optimization for models like GPT-J, CodeGen, StableDiffusion 1.x

## Forward vs. Backward Pass

### Forward Pass
- Outer loop: Q tiles (each thread block owns one Q tile)
- Inner loop: iterate over all K/V tiles
- Each thread block writes one output tile -- no cross-block communication

### Backward Pass
- More complex: requires access to the full attention matrix (recomputed on-the-fly)
- Parallelizes across K/V tiles in the outer loop
- Inner loop iterates over Q tiles
- Requires atomic additions for gradient accumulation across blocks

## Future Directions

Planned work includes H100-specific optimizations leveraging TMA and 4th-gen Tensor Cores, FP8 support, broader device compatibility (AMD GPUs), and integration of high-level algorithmic variants (local/dilated/block-sparse attention).
