# Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training

**Source:** [NVIDIA Developer Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)

## Overview

FP8 is a family of 8-bit floating-point formats designed for efficient neural network training and inference. By using only 8 bits per value instead of 16 (FP16/BF16) or 32 (FP32), FP8 approximately halves memory usage and doubles tensor core throughput on supported hardware, with minimal accuracy degradation when proper scaling strategies are applied.

## FP8 Format Specifications

### E4M3 (4 exponent bits, 3 mantissa bits)
- **Purpose:** Optimized for forward pass computations (weights and activations)
- **Range:** Approximately +/-448
- **Precision:** Higher mantissa precision than E5M2 for finer-grained value representation
- **Special values:** Can represent NaN
- **Best for:** Storing weights, activations, and computing Q*K^T in attention

### E5M2 (5 exponent bits, 2 mantissa bits)
- **Purpose:** Optimized for backward pass computations (gradients)
- **Range:** +/-57,344, +/-inf, NaN
- **Precision:** Lower mantissa precision but much wider dynamic range
- **Best for:** Gradient computation where values vary significantly in magnitude

## Comparison with Other Formats

| Format | Sign | Exponent | Mantissa | Dynamic Range | Typical Use |
|--------|------|----------|----------|---------------|-------------|
| FP32 | 1 | 8 | 23 | 1e-38 to 1e38 | Master weights, accumulators |
| BF16 | 1 | 8 | 7 | 1e-38 to 1e38 | Training (wide range) |
| FP16 | 1 | 5 | 10 | 6e-5 to 65,504 | Training/inference |
| FP8 E4M3 | 1 | 4 | 3 | ~+/-448 | Forward pass |
| FP8 E5M2 | 1 | 5 | 2 | ~+/-57,344 | Backward pass |

### Why FP8 Over INT8
FP8 outperforms integer formats because floating-point provides implicit scaling through exponents. INT8 uses fixed-point scaling that struggles with the unpredictable dynamic ranges in transformers, where attention scores can span from near-zero to thousands. FP8's exponents naturally accommodate these variations.

## Training Accuracy Results

Validation perplexity experiments with 8B parameter models demonstrate that MXFP8 "follows closely that of BF16, indicating that MXFP8 converges as well as BF16" during pretraining. Validation loss comparisons with tensor-wise dynamic scaling track nearly identically with BF16 baseline training, maintaining loss values around 1.02-1.05 for Nemotron 8B.

## Scaling Strategies

### 1. Delayed Scaling
- Uses a history of maximum absolute values (amax) from several previous training steps
- Determines scaling factor from historical data rather than current values
- **Advantage:** Stabilizes training by smoothing transient spikes
- **Disadvantage:** May lag behind rapid dynamic changes in tensor statistics

### 2. Per-Tensor Current Scaling
- Determines scaling factors based on current iteration statistics
- More reactive than delayed scaling
- **Advantage:** Immediately responsive to present data ranges, improving convergence
- **Disadvantage:** Requires computing amax before quantization (additional kernel launch)

### 3. MXFP8 Block Scaling (Microscaling)
- Divides tensors into contiguous blocks of 32 elements
- Each block receives its own dedicated power-of-2 scaling factor in E8M0 format
- Executed natively by GPU Tensor Cores
- **Advantage:** Fine-grained scaling accommodates within-tensor magnitude variations; broader E4M3 adoption
- **Disadvantage:** Additional storage for per-block scale factors (though E8M0 format is compact)

## Hardware Support

### NVIDIA Hopper (H100)
- Dedicated FP8 Tensor Cores
- 2x throughput compared to FP16 Tensor Cores (1978 vs 989 TFLOPS)
- Supports E4M3 and E5M2 formats
- Per-tensor and delayed scaling via NVIDIA Transformer Engine

### NVIDIA Blackwell
- Enhanced FP8 support with native MXFP8 (block-level scaling handled by hardware)
- Extends to sub-FP8 formats: FP4 and FP6
- Finer-grained scaling natively in Tensor Cores
- E8M0 scale factor format processed directly by hardware

## NVIDIA Transformer Engine

The Transformer Engine library handles FP8/MXFP8 optimization automatically:
- Manages scaling factor computation and tensor quantization
- Selects optimal FP8 sub-format (E4M3 vs E5M2) per operation
- Minimizes quantization degradation while achieving significant speedups
- Supports latest FP8 recipes for production deployment

### Memory Efficiency
- Standard FP8: Single FP32 scaling factor per tensor
- MXFP8: E8M0 format (8-bit) scaling factors per 32-element block, further optimizing memory consumption

## Best Practices

1. **Use E4M3 for forward pass** (weights, activations, QKV in attention) and E5M2 for backward pass (gradients)
2. **Keep master weights in FP32** and optimizer states in higher precision
3. **Apply per-tensor or block scaling** rather than static scaling for better accuracy
4. **Consider early learning rate reduction** and selective use of BF16 for sensitive layers
5. **Use NVIDIA Transformer Engine** for automated FP8 management
6. **Validate convergence** by comparing training curves against BF16 baseline
7. **Monitor for gradient overflow** especially in early training stages

## References

- [NVIDIA FP8 Blog Post](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [NVIDIA Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [FP8 Formats for Deep Learning (arXiv 2209.05433)](https://arxiv.org/abs/2209.05433)
