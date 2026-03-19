---
skill_name: FlashAttention-4 Softmax-MMA Pipelining
description: Co-designed algorithm and kernel pipelining that overlaps polynomial softmax computation with tensor core MMA on Blackwell GPUs.
level: L4 - Compute Kernel Optimization Level
target_hardware: NVIDIA Blackwell GPUs (B200/B100)
relevance: When designing attention kernels that must pipeline softmax with matrix multiplication, when exponential throughput is a first-order bottleneck, or when targeting Blackwell-specific features like tensor memory.
---

# FlashAttention-4 Softmax-MMA Pipelining

## What It Is
The official FlashAttention-4 paper presents a co-designed algorithm and kernel pipelining strategy that addresses the fundamental bottleneck of exponential computation in attention kernels. On Blackwell GPUs, the roofline analysis shows that exponential throughput (1024 cycles for M=N=d=128) matches MMA compute cycles exactly, making it a first-order bottleneck equal to matrix multiplication itself. FA4 addresses this through: (1) partial polynomial emulation of exp2 on FMA units (10-25% of entries) to supplement SFU throughput, (2) conditional rescaling with threshold tau=8.0 that skips unnecessary corrections, and (3) ping-pong warpgroup scheduling that overlaps softmax with MMA using Blackwell's tensor memory. The result is 1.3x speedup over cuDNN 9.13, reaching 1613 TFLOPs/s (71% utilization).

## Key Concepts
- **Exponential is a first-order bottleneck**: At d=128, exp throughput matches MMA cycles (1024 each); it cannot be ignored
- **Partial polynomial emulation**: Only 10-25% of exp operations use cubic polynomial on FMA units; rest use hardware SFU; ratio is tunable
- **Degree-3 polynomial suffices**: Max relative error 8.77e-5 at FP32, but BF16 quantization error (3.89e-3) dominates; within 1 BF16 ULP on 99% of inputs
- **Sollya-optimized coefficients**: Minimax polynomial fitting minimizes worst-case relative error over [0,1)
- **Conditional rescaling threshold**: tau = log2(256) = 8.0; only rescale when new max exceeds old max by 256x in linear scale
- **Final normalization absorbs errors**: `Output = (1/l_final) * O_final` corrects all accumulated deviations from skipped rescaling
- **Ping-pong warpgroup scheduling**: Two warpgroups alternate MMA and softmax on separate Q tiles
- **Tensor Memory (TMEM)**: Blackwell's 128x128 TMEM enables decoupled rescaling in a separate correction warpgroup
- **CuTe-DSL**: Python-based kernel DSL compiles 20-30x faster than C++ CUTLASS (2.5s vs 55s)

## Algorithm / Pseudo-code
```
# FA4 Forward Pass: Softmax-MMA Pipeline (Blackwell)

# Two warpgroups (WG0, WG1) per thread block, each processing a Q tile

# TMEM allocation:
#   O_tiles[2]     -- two output accumulators
#   S_tiles[2]     -- two score matrices
#   P_tiles[4]     -- four probability matrices (double-buffered per WG)

# Main loop: for each K,V tile pair
parallel:
    WG0: MMA(Q_tile_0, K_tile) -> S_tile_0    # tensor cores
    WG1: softmax(S_tile_1) -> P_tile_1         # FMA + SFU

barrier()

parallel:
    WG0: softmax(S_tile_0) -> P_tile_0         # FMA + SFU
    WG1: MMA(P_tile_1, V_tile) -> O_tile_1     # tensor cores

# Softmax with partial emulation and conditional rescaling:
def softmax_with_fa4_opts(S_tile, m_old, l_old, O_old):
    m_new = rowmax(S_tile)

    # Conditional rescaling (lazy update)
    if m_new > m_old + TAU:     # TAU = 8.0 = log2(256)
        alpha = exp2(m_old - m_new)
        O = alpha * O_old
        l = alpha * l_old
        m_old = m_new
    else:
        O = O_old
        l = l_old

    # Partial polynomial emulation for exp2
    for i in range(len(S_tile)):
        x = S_tile[i] - m_old
        if i % EMULATION_RATIO == 0:       # 10-25% polynomial
            P[i] = poly_exp2(x)             # cubic on FMA units
        else:
            P[i] = hardware_exp2(x)         # SFU (MUFU.EX2)

    l = l + rowsum(P)
    return P, m_old, l, O

# Polynomial exp2 (Horner's method, 3 FMA ops)
def poly_exp2(x):
    x_floor = floor(x)
    f = x - x_floor
    # Sollya-optimized coefficients for minimax on [0,1)
    result = ((p3 * f + p2) * f + p1) * f + 1.0
    return ldexp(result, x_floor)

# After all K,V tiles processed:
O_final = O / l_final   # absorbs all lazy rescaling deviations
```

## Numerical Considerations
- **Degree-3 polynomial vs hardware exp2**: At BF16 precision, they are indistinguishable; quantization error dominates polynomial error by 44x (3.89e-3 vs 8.77e-5)
- **Conditional rescaling correctness**: Final normalization `O/l_final` mathematically corrects all intermediate approximations; the output is functionally equivalent to standard FlashAttention
- **Threshold tau=8.0 (256x)**: Conservative enough to prevent underflow/overflow in intermediate accumulations while skipping most corrections
- **fp32 accumulators**: All internal computations (O, l, m) use fp32; only final output is converted to bf16
- **Partial emulation ratio**: Higher ratios free more SFU bandwidth but increase register pressure from polynomial temporaries
- **Deterministic backward pass**: Achieves 75% of nondeterministic speed through careful CTA swizzling

## When to Use
- Targeting Blackwell GPUs (B200/B100) where TMEM and tcgen05.mma are available
- Attention workloads where exponential throughput is a measured bottleneck (profiling shows SFU saturation)
- Head dimensions d=64 or d=128 where softmax cost is comparable to MMA cost
- Large-scale LLM training and inference requiring maximum hardware utilization
- When cuDNN attention kernels are not fast enough and custom kernel development is justified

## When NOT to Use
- Pre-Blackwell GPUs (Ampere, Hopper) that lack TMEM and tcgen05.mma instructions
- Small models or short sequences where attention is not the bottleneck
- When cuDNN or existing FlashAttention libraries provide sufficient performance
- fp64 workloads requiring full-precision exponentials
- Prototyping or research where kernel development time is the constraint (use Triton instead)

## Key Takeaways
- **Exponential throughput is a first-order bottleneck** on Blackwell, equal to MMA itself for typical head dimensions
- **Partial polynomial emulation (10-25% of entries)** effectively supplements SFU throughput without excessive register pressure
- **Degree-3 polynomial is BF16-accurate**: higher degrees waste compute since quantization error dominates
- **Conditional rescaling with tau=8.0 skips most corrections** since attention maxima stabilize quickly; final normalization guarantees correctness
- **Ping-pong warpgroup scheduling** enables overlapping softmax and MMA, hiding softmax latency behind tensor core computation
- **Tensor Memory (TMEM) on Blackwell is critical**: enables decoupled rescaling that was impossible with register-only accumulators on Hopper
- **1613 TFLOPs/s (71% utilization)** demonstrates that attention can approach the hardware roofline with careful co-design

## References
- Dao, T. et al. "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling." arXiv:2603.05451, 2026.
- [Tri Dao's Blog Post](https://tridao.me/blog/2026/flash4/)
- [Together AI Blog](https://www.together.ai/blog/flashattention-4)
- [Princeton AI Lab Blog](https://blog.ai.princeton.edu/2026/03/12/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
- [Modal Blog: Reverse Engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4)
