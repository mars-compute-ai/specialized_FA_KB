# FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling

**Source**: [arXiv:2603.05451](https://arxiv.org/html/2603.05451v1)
**Authors**: Tri Dao et al.
**Also available**: [Tri Dao's Blog](https://tridao.me/blog/2026/flash4/), [Together AI Blog](https://www.together.ai/blog/flashattention-4), [Princeton AI Lab Blog](https://blog.ai.princeton.edu/2026/03/12/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)

---

## Overview

FlashAttention-4 (FA4) is the fourth generation of the FlashAttention algorithm, co-designed with kernel pipelining to address the asymmetric scaling of hardware resources on NVIDIA Blackwell GPUs. It achieves up to 1.3x speedup over cuDNN 9.13 and 2.7x over Triton on B200 GPUs, reaching 1613 TFLOPs/s (71% utilization) with BF16.

---

## 1. Polynomial Exponential Approximation (Detailed)

### Range Reduction

The exponential `2^x` is decomposed:

```
2^x = 2^floor(x) * 2^(x - floor(x))
```

- **Integer part** `2^floor(x)`: Computed via IEEE 754 bit manipulation (exact)
- **Fractional part** `2^x_frac`: Approximated via polynomial where `x_frac in [0, 1)`

### Polynomial Form

```
2^(x_frac) ≈ p_0 + p_1 * x_frac + p_2 * x_frac^2 + p_3 * x_frac^3
```

Where `p_0 = 1.0` and remaining coefficients minimize relative approximation error, computed using the Sollya software library for optimal minimax polynomial fitting.

### Accuracy Analysis (Table 2 from paper)

| Degree | Max Relative Error (FP32) | Max Error after BF16 Rounding |
|--------|---------------------------|-------------------------------|
| 2      | 3.44 x 10^-3              | 3.89 x 10^-3                 |
| 3      | 8.77 x 10^-5              | 3.89 x 10^-3                 |
| 4      | 1.64 x 10^-6              | 3.89 x 10^-3                 |

**Key insight**: For degree >= 3, the FP32-level polynomial error (8.77e-5) is dominated by BF16 quantization error (3.89e-3). The polynomial is "nearly indistinguishable" from hardware MUFU.EX2 (Special Function Unit) at bf16 operational precision. Degree-3 matches hardware to within 1 BF16 ULP on 99% of inputs.

### Partial Emulation Strategy

Not all exponentials use the polynomial approximation:
- **10-25% of softmax entries** use the polynomial (computed on FMA/CUDA cores)
- **Remaining 75-90%** use hardware MUFU.EX2 (SFU)
- This ratio balances register pressure against throughput gains from FMA unit parallelization
- The tunable ratio allows adapting to different head dimensions and tile sizes

---

## 2. Conditional (Lazy) Rescaling Mechanism

### Standard Approach

In standard FlashAttention, every tile that produces a new maximum triggers a full rescaling of the output accumulator:

```
alpha = exp(m_old - m_new)
O = alpha * O
l = alpha * l
```

### FA4's Conditional Rescaling

Rescaling is only applied when the maximum changes significantly:

```
if m_new > m_old + tau:    # where tau ≈ log2(256) = 8.0
    alpha = exp(m_old - m_new)
    O = alpha * O
    l = alpha * l
    m_old = m_new
```

The threshold `tau ≈ 8.0` corresponds to a 256x scaling factor, meaning rescaling is triggered only when the new maximum represents a 256x change in scale.

### Correctness Guarantee

Mathematical correctness is maintained through **final renormalization**:

```
Output = (1 / l_final) * O_final
```

Accumulated deviations from skipped rescaling steps are corrected by the true maximum and normalizer computed at the end. The final normalization absorbs all intermediate approximation errors.

---

## 3. Pipeline Architecture

### Ping-Pong Warpgroup Scheduling

Two warpgroups process separate Q tiles per thread block:
- While one tile's tensor core operations (MMA) execute, the other tile computes softmax
- This overlaps MMA computation with softmax computation

### Blackwell-Specific: Tensor Memory (TMEM)

Unlike Hopper's register-resident accumulators, Blackwell provides 128x128 tensor memory storage that enables:
- **Decoupling rescaling of output** to a separate correction warpgroup
- Removing rescaling from the critical execution path
- Separate register files per warpgroup prevent interference

### TMEM Partitioning

The tensor memory is allocated as:
- Two output tiles (O)
- Two S score matrices
- Four P probability matrices

This enables **immediate pipeline startup** by computing two S tiles without waiting.

---

## 4. Roofline Analysis: Softmax-MMA Interleaving

For M=N=d=128:
- **MMA compute**: 1024 cycles
- **Exponential throughput**: 1024 cycles (matches MMA!)
- **Shared memory**: 768 cycles (secondary bottleneck)

The exponential operation is a first-order bottleneck equal to MMA itself. This motivates:
1. Polynomial emulation to increase exponential throughput via FMA units
2. Explicit synchronization to prevent softmax warpgroups from overlapping critical sections
3. Hardware pipelining between MMA units and softmax

---

## 5. Performance Results

### Forward Pass (B200 GPU, BF16)

- Up to **1.3x speedup over cuDNN 9.13**
- Up to **2.7x over Triton**
- Peak: **1613 TFLOPs/s** (71% utilization)

### Backward Pass

- 2-CTA mode reduces shared memory traffic from 1536 to 1024 cycles
- Halves atomic reductions through decomposed dQ computation
- Deterministic mode achieves up to 75% the speed of nondeterministic backward pass

### Compilation

- CuTe-DSL (Python-based) compiles **20-30x faster** than C++ CUTLASS templates
- Forward pass: 2.5s vs 55s compile time

---

## 6. Numerical Stability Summary

| Aspect | Guarantee |
|--------|-----------|
| Polynomial exp accuracy | Within 1 BF16 ULP on 99% of inputs |
| Lazy rescaling | Corrected by final normalization |
| BF16 accumulation | fp32 accumulators used internally |
| Overall output quality | Functionally equivalent to standard FlashAttention for bf16 workloads |

---

## References

- Dao, T. et al. "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling." arXiv:2603.05451, 2026.
- [Tri Dao's Blog Post](https://tridao.me/blog/2026/flash4/)
- [Together AI Blog](https://www.together.ai/blog/flashattention-4)
- [Princeton AI Lab Blog](https://blog.ai.princeton.edu/2026/03/12/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
- [Modal Blog: Reverse Engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4)
