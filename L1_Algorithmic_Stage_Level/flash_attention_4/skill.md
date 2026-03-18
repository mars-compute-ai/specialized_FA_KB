---
skill_name: FlashAttention-4 Blackwell-Optimized Attention
description: Blackwell GPU-optimized attention kernel with 5 warp specializations, cubic polynomial exp approximation for bf16, and conditional softmax rescaling that reduces corrections by ~10x.
level: L1 - Algorithmic/Mathematical Stage Level
target_hardware: NVIDIA Blackwell GPUs (B100/B200)
relevance: When targeting maximum attention throughput on Blackwell GPUs, or when designing attention kernels that need fast approximate exponentials and conditional rescaling.
---

# FlashAttention-4 Blackwell-Optimized Attention

## What It Is
FlashAttention-4 (FA4), analyzed by Modal's reverse engineering of the released CUDA code (originally presented by Tri Dao at Hot Chips), is an attention kernel achieving ~20% speedup over cuDNN on Blackwell GPUs. It advances the warp-specialization paradigm from FA3's 2 roles (producer/consumer) to 5 specialized roles (Load, MMA, Softmax, Correction, Epilogue), introduces a fast cubic polynomial approximation for the exponential function targeting bf16 precision, and implements intelligent conditional softmax rescaling that only triggers output corrections when the running maximum changes significantly -- reducing correction operations by approximately 10x.

## Key Concepts
- **5 Warp Specializations:**
  - **Load Warp:** TMA-based async loads of Q, K, V tiles to SMEM, triple-buffered for K/V
  - **MMA Warp:** Executes QK^T and PV matrix multiplications via Blackwell Tensor Cores (`tcgen05.mma`)
  - **Softmax Warps (8):** Compute row-wise softmax P from scores S, maintain running statistics (m, l)
  - **Correction Warps (4):** Rescale accumulated O when maximum changes -- activated only when needed
  - **Epilogue Warps (1-2):** Final normalization O/l and TMA store to GMEM
- **Cubic Polynomial Exp Approximation:** Instead of using the slow Special Function Unit (SFU) for exp2, FA4 splits 2^x = 2^floor(x) * 2^frac(x) and approximates 2^r for r in [0,1) with: `0.07711909*r^3 + 0.22756439*r^2 + 0.69514614*r + 1.0`, evaluated via 3 FMA operations on f32x2 lanes. Sufficient accuracy for bf16. Applied selectively on configurable iterations to avoid SFU bottlenecks.
- **Conditional Softmax Rescaling:** Previous versions rescaled O every time a new row maximum appeared. FA4 checks if the new maximum changes enough to impact numerical stability before triggering correction warps. For typical attention score distributions, this reduces corrections by ~10x.
- **Persistent Block Scheduling:** Uses `StaticPersistentTileScheduler` to launch one CTA per SM for the kernel's lifetime, reducing launch overhead.
- **Dual Query Tiles:** Each kernel instance processes two query tiles simultaneously, amortizing K/V streaming cost.

## Algorithm / Pseudo-code
```
# FlashAttention-4 Forward Pass (Blackwell, per-CTA, persistent)
# Processes 2 query tiles Q_a, Q_b simultaneously

# === LOAD WARP ===
TMA_load(Q_a, Q_b -> SMEM)
for each KV block j = 0 to T_c-1:
    TMA_load(K_j, V_j -> SMEM[j % 3])  # triple-buffered
    signal(barrier[j % 3])

# === MMA WARP (for each query tile Q_t in {Q_a, Q_b}) ===
for each KV block j:
    wait(barrier[j % 3])                   # K_j ready
    S = tcgen05.mma(Q_t, K_j^T)           # Blackwell Tensor Cores
    signal(softmax_ready[j])
    wait(softmax_done[j])                  # P ready
    O += tcgen05.mma(P, V_j)              # accumulate

# === SOFTMAX WARPS (2 warpgroups for 2 query tiles) ===
for each KV block j:
    wait(softmax_ready[j])
    m_new = rowmax(S)

    # CONDITIONAL rescaling (key FA4 innovation)
    if abs(m_new - m_old) > threshold:
        signal(correction_needed[j])       # trigger correction warps
    else:
        # Skip correction -- saves ~10x corrections

    # Fast exp approximation for bf16:
    r = frac(S - m_new)  # fractional part, in [0, 1)
    # Horner's method: 3 FMA ops
    P = ((0.07711909 * r + 0.22756439) * r + 0.69514614) * r + 1.0
    P *= 2^floor(S - m_new)  # integer part via bit shift

    l += rowsum(P)
    m_old = m_new
    signal(softmax_done[j])

# === CORRECTION WARPS (activated only when signaled) ===
when correction_needed[j]:
    O *= exp2(m_old - m_new)               # rescale accumulated output
    # Runs ~10x less frequently than in FA1-FA3

# === EPILOGUE WARPS ===
when query tile complete:
    O = O / l                              # final normalization
    TMA_store(O -> GMEM)
```

## When to Use
- On NVIDIA Blackwell GPUs (B100, B200) for maximum attention throughput
- When bf16 precision is used and the cubic polynomial exp is sufficient (no need for full-precision exp)
- Long-sequence attention where the 10x reduction in corrections compounds into significant savings
- Inference and training workloads that can leverage persistent kernel scheduling
- When kernel launch overhead is a concern (persistent scheduling amortizes it)

## When NOT to Use
- On Hopper (H100) or older GPUs -- the `tcgen05.mma` instructions and Blackwell SM features are not available; use FA3 instead
- When FP16 (not bf16) full-precision exponential is required and the cubic approximation error is not acceptable
- For very short sequences where the overhead of 5-way warp specialization and persistent scheduling may not pay off
- When the attention pattern is highly structured (e.g., sliding window) and a specialized sparse kernel would be faster

## Key Takeaways
- Evolving from FA3's 2 roles to FA4's 5 warp specializations shows the trend towards finer-grained GPU pipeline parallelism
- The cubic polynomial exp approximation (3 FMA ops) is a practical alternative to the SFU for bf16 values, avoiding the SFU throughput bottleneck
- Conditional softmax rescaling is a simple but high-impact optimization: real attention score distributions rarely produce maxima that change frequently enough to require constant correction
- The ~10x reduction in corrections means the Correction Warps are mostly idle, freeing warp scheduler slots for productive work
- FA4 achieves ~20% speedup over cuDNN, continuing the trajectory of 1.5-2x per generation (FA1->FA2->FA3->FA4)
- The increasing complexity of warp-specialized kernels (FA4 is described as "quite gnar code") is driving investment in higher-level abstractions (CuTe DSL, CuTile)

## References
- Modal Blog (reverse engineering): https://modal.com/blog/reverse-engineer-flash-attention-4
- Hot Chips presentation by Tri Dao
- Schraudolph (1999): original cubic exp approximation idea
- NVIDIA CuTe DSL and CUTLASS for warp-specialized kernel abstractions
