# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

Source: https://tridao.me/blog/2024/flash3/
Paper: https://arxiv.org/abs/2407.08691

## Overview

FlashAttention-3 is optimized specifically for NVIDIA Hopper (H100/H200) GPUs, achieving 1.5-2.0x speedup over FlashAttention-2 through three key techniques: exploiting hardware asynchrony for overlapping computation and data movement, interleaving block-wise matmul and softmax operations, and leveraging FP8 low-precision with incoherent processing for accuracy. It achieves up to 740 TFLOPS in FP16 (75% of H100 theoretical maximum) and ~1.2 PFLOPS in FP8.

## The Bottleneck: Special Functions vs Tensor Cores

A critical insight motivating FA3: on H100, exponential operations (used in softmax) run at **3.9 TFLOPS** versus **989 TFLOPS** for FP16 matrix multiplication -- a **256x difference**. Previous FlashAttention-2 only utilized 35% of H100 capabilities because softmax became the bottleneck that stalled the tensor cores.

## Three Key Techniques

### 1. Asynchronous Operation Overlapping (Warp Specialization)

FA3 exploits Hopper's asynchrony to overlap computation and data movement. Two scheduling strategies:

**Pingpong Scheduling:**
- Coordinates multiple warpgroups so one executes matrix multiplications while another performs softmax operations
- Improves FP16 forward pass from ~570 TFLOPS to ~620 TFLOPS
- Different warpgroups alternate between GEMM and softmax phases

**Intra-Warpgroup Pipelining:**
- Allows softmax computations within a single warpgroup to overlap with GEMM operations
- Pushes performance to 640-660 TFLOPS
- Fine-grained interleaving within a single warpgroup

### 2. Interleaved Block-wise Matmul and Softmax

Instead of completing all matmuls before starting softmax (or vice versa), FA3 interleaves these operations:
- While tensor cores compute Q x K^T for block i+1, the multi-function units compute softmax for block i
- This keeps both execution units busy simultaneously
- Eliminates the sequential bottleneck that limited FA2

### 3. Low-Precision (FP8) with Incoherent Processing

FA3 implements a novel approach to FP8 attention:

**The Problem:** Direct FP8 quantization of attention activations causes large errors due to outlier values.

**The Solution: Incoherent Processing**
- Applies Hadamard transforms with random signs to spread outliers across dimensions
- Reduces FP8 quantization error by **2.6x** compared to baseline FP8 approaches
- Makes lower-precision computation viable without sacrificing model quality

**FP8 Performance:** Reaches approximately 1.2 PFLOPS, nearly doubling FP16 throughput.

## Hardware Features Leveraged

FA3 fully utilizes Hopper-specific capabilities:

- **WGMMA (Warpgroup Matrix Multiply-Accumulate)**: Higher throughput matrix operations than previous SM80 instructions
- **TMA (Tensor Memory Accelerator)**: Efficient asynchronous data transfers between global memory and shared memory, freeing the warps to do computation
- **FP8 Tensor Cores**: Double the throughput of FP16 tensor cores
- **Asynchronous execution model**: Allows memory operations and compute to overlap far more aggressively than Ampere

## Performance Benchmarks

### FP16 Performance on H100

| Configuration | Throughput (TFLOPS) | % of Peak |
|--------------|--------------------:|----------:|
| FA2 baseline  | ~350               | ~35%      |
| FA3 (basic)   | ~570               | ~58%      |
| FA3 + pingpong| ~620               | ~63%      |
| FA3 + pipelining | 640-660         | ~67%      |
| FA3 (best)    | ~740               | ~75%      |
| H100 FP16 peak| 989                | 100%      |

### FP8 Performance on H100

| Configuration | Throughput |
|--------------|----------:|
| FA3 FP8      | ~1.2 PFLOPS |
| H100 FP8 peak| ~2.0 PFLOPS |
| Utilization  | ~60%       |

### Speedup over FlashAttention-2

- **Forward pass**: 1.5-2.0x faster
- **Consistent across sequence lengths**: Demonstrated at 8K and beyond
- **FP8 mode**: Further 1.5x beyond FP16 FA3

## Comparison with FA2

| Feature | FlashAttention-2 | FlashAttention-3 |
|---------|-----------------|-----------------|
| Target GPU | Ampere (A100) | Hopper (H100) |
| H100 utilization | ~35% | ~75% |
| FP16 throughput | ~350 TFLOPS | ~740 TFLOPS |
| FP8 support | No | Yes (~1.2 PFLOPS) |
| Warp specialization | No | Yes |
| TMA usage | No | Yes |
| Scheduling | Sequential | Pingpong / pipelined |

## Integration Status

- Code available at [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) (hopper/ directory)
- Being integrated into PyTorch's SDPA
- Works with the flash_attn Python package

### Installation

```bash
cd flash-attention/hopper
python setup.py install
```

### Usage

```python
import flash_attn_interface

# Basic usage
output = flash_attn_interface.flash_attn_func(q, k, v, causal=True)
```

## Requirements

- NVIDIA Hopper GPU (H100, H200)
- CUDA >= 12.3 (recommend 12.8+)
- CUTLASS library (used for hardware abstractions)
- Recent PyTorch (2.2+)

## Key Technical Insight

The fundamental advance in FA3 is recognizing that on modern GPUs, the bottleneck is not total FLOPs but the **utilization of heterogeneous execution units**. By carefully scheduling work across tensor cores (for matmul) and multi-function units (for softmax/exponential) simultaneously, FA3 keeps both units busy instead of one waiting for the other. This is the same principle that makes GPUs fast for graphics (pipeline parallelism) applied to attention.
