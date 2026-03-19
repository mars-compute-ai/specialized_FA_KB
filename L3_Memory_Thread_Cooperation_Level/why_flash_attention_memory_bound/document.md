# Why FlashAttention Is Memory-Bound

## Overview

Standard attention in transformers is fundamentally **memory-bound**, meaning GPU compute units spend most of their time waiting for data rather than performing arithmetic. FlashAttention exploits this insight: by restructuring data movement to minimize HBM traffic, it achieves dramatic speedups despite performing the same (or more) FLOPs.

## The Memory-Bound Problem

### What "Memory-Bound" Means

An operation is memory-bound when its execution time is dominated by memory access rather than computation. The key metric is **arithmetic intensity** — the ratio of FLOPs to bytes transferred:

- **Compute-bound**: Large matrix multiplications (high arithmetic intensity, e.g., hundreds of FLOPs per byte)
- **Memory-bound**: Elementwise operations, reductions, softmax, dropout (low arithmetic intensity, e.g., <10 FLOPs per byte)

Standard attention contains many memory-bound operations applied to N×N matrices.

### Naive Attention Data Movement

Standard attention performs these steps, each requiring separate HBM reads and writes:

```
Step 1: S = Q @ K^T     # Read Q,K from HBM; Write N×N S to HBM
Step 2: P = softmax(S)   # Read N×N S from HBM; Write N×N P to HBM
Step 3: P = dropout(P)   # Read N×N P from HBM; Write N×N P to HBM
Step 4: O = P @ V        # Read N×N P and V from HBM; Write O to HBM
```

Each intermediate result (S, P) is a full **N×N matrix**. For N=4096, d=128 in FP16:
- Q, K, V: 4096 × 128 × 2 bytes = **1 MB each**
- S, P: 4096 × 4096 × 2 bytes = **32 MB each**
- Total N×N traffic: **~64 MB** per attention head, per layer
- The N×N intermediates represent **97% of total memory traffic**

### The Bandwidth Bottleneck

On an A100 GPU:
- HBM bandwidth: ~2 TB/s
- SRAM bandwidth: ~19 TB/s
- Tensor core throughput: 312 TFLOPS (FP16)

To keep tensor cores busy at 312 TFLOPS, the algorithm needs an arithmetic intensity of:
```
312 TFLOPS / 2 TB/s = 156 FLOPs per byte
```

Standard attention's elementwise operations (softmax, masking, dropout) on N×N matrices have arithmetic intensities of **1-10 FLOPs per byte** — far below the threshold needed to saturate compute. The GPU's tensor cores sit **idle** waiting for data.

### Concrete Example: How Bad Is It?

For a typical transformer layer with batch_size=32, num_heads=16, seq_len=4096, head_dim=128:
- Total bytes moved through HBM for attention: **~33 GB** (including all heads, all batch elements)
- Time to move data at 2 TB/s: **~16.5 ms**
- Time to compute at 312 TFLOPS: **~0.5 ms**
- The operation is **~33x slower** than it needs to be due to memory traffic

## How FlashAttention Solves This

### Fused Kernel: No Intermediate Materialization

FlashAttention fuses all attention steps into a **single GPU kernel**. The N×N intermediate matrices (S, P) are:
- Computed in tiles within SRAM
- Consumed immediately for the next operation
- **Never written to HBM**

```
# Standard: 4 separate kernels, each with HBM round-trips
HBM → [Kernel 1: QK^T] → HBM → [Kernel 2: softmax] → HBM → [Kernel 3: dropout] → HBM → [Kernel 4: PV] → HBM

# FlashAttention: 1 fused kernel, tiles stay in SRAM
HBM → [Single Kernel: QK^T → softmax → dropout → PV, all in SRAM tiles] → HBM
```

### Quantifying the Improvement

| Metric | Standard Attention | FlashAttention | Improvement |
|--------|-------------------|----------------|-------------|
| HBM accesses | Θ(Nd + N²) | Θ(N²d²M⁻¹) | ~33x fewer |
| Memory footprint | O(N²) | O(N) | Quadratic → linear |
| Wall-clock time | Memory-bound | Near compute-bound | Up to 4x faster |

For N=4096, d=128, M=100KB:
- Standard: ~64 MB of N×N traffic per head
- FlashAttention: ~2 MB of tiled traffic per head
- **~33x reduction** in HBM traffic

### Why Extra Compute Is Free

FlashAttention performs **more total FLOPs** than standard attention (due to recomputation in the backward pass and online softmax rescaling). But since standard attention is severely memory-bound, these extra FLOPs execute during time that would otherwise be spent waiting for memory. The net result is still faster because the bottleneck (memory bandwidth) is dramatically reduced.

## The Roofline Model Perspective

The **roofline model** plots achievable performance (FLOPS) against arithmetic intensity:

```
Performance (TFLOPS)
    ^
312 |                        _______________  (compute ceiling)
    |                      /
    |                    /
    |                  /    ← Memory-bound region
    |                /      (standard attention lives here)
    |              /
    |            /
    |          /  ← FlashAttention moves operations here
    |        /     by increasing effective arithmetic intensity
    |      /
    |    /
    |  /
    |/________________________> Arithmetic Intensity (FLOPs/byte)
         ~2        ~156
    (softmax)   (break-even point)
```

Standard attention operations cluster in the memory-bound region (left side). FlashAttention increases effective arithmetic intensity by reducing bytes transferred while keeping FLOPs the same (or slightly higher), moving the operation closer to the compute-bound region.

## Implications Beyond Attention

The memory-bound insight applies broadly:
- **Any operation** that materializes large intermediate tensors in HBM is a candidate for tiling
- **Kernel fusion** (combining multiple operations into one kernel) reduces HBM round-trips
- **Recomputation** can be cheaper than storage when the operation is memory-bound
- Modern GPU architectures (Hopper's TMA, async copy) provide hardware support for efficient tiling

## Sources

- [The Free Lunch of Flash Attention (Better ML / Medium)](https://medium.com/better-ml/the-free-lunch-of-flash-attention-036b0040dee2)
- [FlashAttention Paper (arXiv 2205.14135)](https://arxiv.org/abs/2205.14135)
- [Why FlashAttention? (Medium)](https://medium.com/@katherineolowookere/why-flashattention-4b0f6cca8653)
- [Attention Optimizations Overview (HuggingFace)](https://huggingface.co/blog/atharv6f/flash-attention-overview)
- [How FlashAttention Eliminates Transformer Memory Bottlenecks (Galileo)](https://galileo.ai/blog/stanford-flashattention-algorithm)
