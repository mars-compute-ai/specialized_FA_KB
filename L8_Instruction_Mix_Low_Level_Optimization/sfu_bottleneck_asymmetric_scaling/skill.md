---
skill_name: SFU Bottleneck Analysis & Asymmetric Hardware Scaling
description: Quantify the SFU-vs-tensor-core throughput imbalance and co-design instruction pipelines that distribute exponential work across both SFU and FMA units.
level: L8 - Instruction-Mix/Low-Level Optimisation
target_hardware: NVIDIA Blackwell B200/B100, Hopper H100 (asymmetry increases with each generation)
relevance: When an AI agent is analyzing attention kernel performance on modern GPUs and needs to understand or mitigate the fundamental bottleneck from softmax exponentiation competing with tensor core throughput.
---

# SFU Bottleneck Analysis & Asymmetric Hardware Scaling

## What It Is
Modern GPU architectures exhibit asymmetric hardware scaling: from Hopper to Blackwell, tensor core throughput increased 2.25x while SFU (Special Function Unit) count and shared memory bandwidth remained unchanged. This creates a fundamental instruction-mix bottleneck where the exponential operations required for softmax in attention kernels consume as many cycles as the tensor core MMA operations themselves. Understanding and quantifying this imbalance is the prerequisite for instruction-level optimizations that distribute exponential work across both SFU and FMA pipelines.

## Key Concepts
- **512:1 throughput imbalance**: Blackwell tensor cores deliver 8,192 ops/cycle while the exponential unit delivers only 16 ops/cycle per SM
- **Forward pass parity**: Both tensor cores and exponential units require 1,024 cycles per tile -- softmax is an equal bottleneck with MMA
- **Backward pass is shared-memory-limited**: 3,328 cycles for shared memory vs 2,560 for tensor cores vs 1,024 for exponential
- **Concurrent SFU + FMA execution**: Distribute exponential work across hardware MUFU.EX2 (SFU) and polynomial approximation (FMA pipeline) simultaneously
- **Cody-Waite range reduction**: Decompose 2^x = 2^n * 2^f, compute integer part via bit manipulation, fractional part via polynomial
- **Sollya-optimized coefficients**: Polynomial coefficients chosen to minimize relative error for bf16 output precision
- **Sequential softmax warpgroups**: Prevent SFU contention by synchronizing warpgroups to avoid simultaneous exponential evaluation
- **Conditional rescaling with threshold tau**: Apply online softmax corrections only when maximum changes exceed threshold, eliminating redundant vector operations

## Instruction Patterns / Code
```
# === Feeds-and-Speeds Analysis (per SM, per tile, M=N=D=128) ===

# Forward Pass:
#   Tensor Cores: 128*128*128 / 8192 ops/cycle = 1024 cycles
#   Exponential:  128*128     / 16 ops/cycle   = 1024 cycles  <-- PARITY!
#   Shared Mem:   (bytes)     / 128 bytes/cycle = 768 cycles

# Backward Pass (1-CTA):
#   Tensor Cores: (5 MMAs)    / 8192 ops/cycle = 2560 cycles
#   Exponential:  128*128     / 16 ops/cycle   = 1024 cycles
#   Shared Mem:   (bytes)     / 128 bytes/cycle = 3328 cycles  <-- BOTTLENECK

# === Concurrent SFU + FMA exponential strategy ===

# Iteration i (use hardware SFU):
MUFU.EX2 R0, R1                    # Hardware exponential on SFU

# Iteration i+1 (use FMA polynomial):
FFMA R2, R_p3, R_frac, R_p2        # p3*f + p2 (on FMA unit)
FFMA R2, R2,   R_frac, R_p1        # t1*f + p1 (on FMA unit)
FFMA R2, R2,   R_frac, R_p0        # t2*f + p0 (on FMA unit)
# + bit manipulation for 2^n

# By alternating, effective exponential throughput ≈ 2x single-path

# === Synchronized softmax to prevent SFU contention ===
# Warpgroup 0 computes softmax (uses SFU)
# barrier.sync                      # Wait for WG0 to finish
# Warpgroup 1 computes softmax (uses SFU)
# Meanwhile: WG0 issues MMA instructions (uses Tensor Cores)
```

## Performance Impact
- **Forward pass**: 1,605 TFLOPs/s peak (71% hardware utilization on Blackwell B200)
- **vs cuDNN 9.13**: 1.1-1.3x faster
- **vs Triton**: 2.1-2.7x faster
- **Conditional rescaling**: Reduces correction operations by ~10x
- **2-CTA backward mode**: Reduces shared memory operand B traffic by ~50%
- **Deterministic backward**: 85-90% of nondeterministic throughput

## When to Use
- When targeting Blackwell or newer GPUs where tensor core scaling outpaces SFU scaling
- When profiling (Nsight Compute) shows the exponential unit as a bottleneck in the instruction mix
- When designing new attention kernels and need to plan the instruction pipeline around SFU limitations
- When evaluating whether to invest in approximate exponentiation for a given kernel
- For architecture-aware performance modeling to predict kernel throughput limits

## When NOT to Use
- On older architectures (Volta, Turing) where SFU throughput is not the bottleneck relative to tensor cores
- When the kernel is memory-bandwidth-bound (instruction mix optimization has no effect)
- For workloads that do not involve softmax or exponentiation (e.g., pure GEMM)
- When exact IEEE-754 exponentiation is required for numerical correctness

## Key Takeaways
- The SFU bottleneck is a **hardware design constraint**, not a software bug -- it will worsen with each GPU generation as tensor cores scale faster
- Quantitative feeds-and-speeds analysis (cycles per tile per resource) is essential before attempting instruction-level optimization
- The forward pass attention kernel on Blackwell has reached parity between tensor core and exponential throughput at 1,024 cycles each
- Co-designing the algorithm (conditional rescaling, approximate exponentiation) with the pipeline (warpgroup synchronization, SFU+FMA distribution) is the key to recovering performance
- The backward pass has a different bottleneck (shared memory), requiring different optimization strategies (2-CTA mode, DSMEM exchange)

## References
- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling (Colfax Research)](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
- [Reverse Engineering Flash Attention 4 (Modal Blog)](https://modal.com/blog/reverse-engineer-flash-attention-4)
- NVIDIA Blackwell Architecture Whitepaper
- NVIDIA Nsight Compute documentation (instruction mix profiling)
