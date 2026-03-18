# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

**Authors:** Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao
**Affiliations:** Colfax Research, Meta, NVIDIA, Georgia Tech, Princeton University, Together AI
**Date:** July 12, 2024
**Source:** https://tridao.me/publications/flash3/flash3.pdf

## Abstract

FlashAttention-3 develops three main techniques to speed up attention on Hopper GPUs: exploiting asynchrony of the Tensor Cores and TMA to (1) overlap overall computation and data movement via warp-specialization and (2) interleave block-wise matmul and softmax operations, and (3) block quantization and incoherent processing that leverages hardware support for FP8 low-precision. Achieves speedup on H100 GPUs by 1.5-2.0x with FP16 reaching up to 740 TFLOPs/s (75% utilization), and with FP8 reaching close to 1.2 PFLOPs/s. FP8 FlashAttention-3 achieves 2.6x lower numerical error than a baseline FP8 attention.

## Background

### Multi-Head Attention

Let Q, K, V in R^{N x d} be the query, key and value input sequences, where N is the sequence length and d is the head dimension. The attention output O is computed as:

```
S = alpha * Q @ K^T    (in R^{N x N})
P = softmax(S)          (in R^{N x N}, applied row-wise)
O = P @ V               (in R^{N x d})
```

where alpha = 1/sqrt(d). In practice, rowmax(S) is subtracted from S to prevent numerical instability with the exponential function.

### Backward Pass Gradients

Given output gradient dO in R^{N x d}:
```
dV = P^T @ dO
dP = dO @ V^T
dS = dsoftmax(dP)    where ds = (diag(p) - p*p^T) * dp for p = softmax(s)
dQ = alpha * dS @ K
dK = alpha * dS^T @ Q
```

### GPU Hardware: H100 Hopper Architecture

| Hardware Level | Parallel Agent | Data Locale | Capacity @ Bandwidth |
|----------------|----------------|-------------|---------------------|
| Chip | Grid | GMEM (HBM) | 80 GiB @ 3.35 TB/s |
| GPC | Threadblock Clusters | L2 | 50 MiB @ 12 TB/s |
| SM | Threadblock (CTA) | SMEM | 228 KiB per SM, 31 TB/s per GPU |
| Thread | Thread | RMEM | 256 KiB per SM |

Key hardware features:
- **TMA (Tensor Memory Accelerator):** Dedicated hardware unit for async memory copy between GMEM and SMEM
- **WGMMA:** Warpgroup-wide asynchronous Tensor Core instruction, can source inputs directly from shared memory
- **setmaxnreg:** Dynamic register reallocation between warpgroups

## Algorithm: Three Key Techniques

### Technique 1: Producer-Consumer Asynchrony via Warp-Specialization

The CTA is divided into producer and consumer warpgroups:

**Producer warpgroup:**
1. Deallocate registers (give to consumers)
2. Issue TMA load of Q_i from HBM to SMEM
3. For each j in [0, T_c): Issue TMA loads of K_j, V_j to SMEM circular buffer
4. Signal consumers upon completion via barriers

**Consumer warpgroup:**
1. Reallocate registers
2. Initialize O_i = 0, l_i = 0, m_i = -inf on-chip
3. Wait for Q_i
4. For each j in [0, T_c):
   - Wait for K_j; Compute S_i^(j) = Q_i @ K_j^T (SS-GEMM)
   - Update m_i, compute P_tilde_i^(j) = exp(S_i^(j) - m_i), update l_i
   - Wait for V_j; Compute O_i = diag(exp(m_old - m_i))^{-1} * O_i + P_tilde_i^(j) @ V_j (RS-GEMM)
   - Release buffer stage for producer
5. Final: O_i = diag(l_i)^{-1} * O_i; L_i = m_i + log(l_i)

### Pingpong Scheduling

Two consumer warpgroups alternate GEMM and softmax phases:
- Warpgroup 1 performs GEMMs while Warpgroup 2 does softmax, then roles swap
- Hides the comparatively low-throughput softmax (3.9 TFLOPS for exponential vs 989 TFLOPS for FP16 matmul)
- Improves performance from 570 to 620-640 TFLOPS for FP16 forward with head dim 128

### Technique 2: Intra-Warpgroup GEMM-Softmax Overlapping (2-Stage Pipeline)

Breaks sequential dependencies by pipelining across iterations:

```
# 2-Stage WGMMA-Softmax Pipeline (Consumer Mainloop)
Initialize: Compute S_cur = Q_i @ K_0^T, commit and wait
Compute m_i, P_tilde_cur, l_i from S_cur, rescale O_i

for j = 1 to T_c - 1:
    # Stage A: Async GEMM for next QK^T
    Compute S_next = Q_i @ K_j^T using WGMMA. Commit but DO NOT WAIT.

    # Stage B: Async GEMM for current PV
    Compute O_i += P_tilde_cur @ V_{j-1} using WGMMA. Commit but DO NOT WAIT.

    # While WGMMAs execute asynchronously:
    Wait for S_next WGMMA
    Compute m_i, P_tilde_next, l_i from S_next  # softmax overlaps with PV WGMMA

    Wait for PV WGMMA, then rescale O_i
    Copy S_next -> S_cur, P_tilde_next -> P_tilde_cur

# Epilogue: Final PV accumulation and write
```

The key insight: softmax for iteration j+1 overlaps with the PV GEMM for iteration j. SASS analysis confirms the compiler generates overlapped code as expected.

### Technique 3: FP8 Low-Precision with Block Quantization

**Layout challenges for FP8:**
- FP8 WGMMA only supports k-major operands (vs FP16 which supports both)
- FP32 accumulator layout differs from FP8 operand A register layout
- Solution: In-kernel transpose of V tiles using LDSM/STSM instructions + byte permute for accumulator-to-operand format conversion

**Block quantization:**
- One scaling factor per block (B_r x d or B_c x d) instead of per-tensor
- Can be fused with preceding operations (e.g., rotary embedding) at no extra cost
- Naturally aligns with FlashAttention-3's block-wise operation

**Incoherent processing:**
- Multiply Q and K with random orthogonal matrix M before quantizing to FP8
- Since MM^T = I, (QM)(KM)^T = QK^T -- attention output unchanged
- M = product of random diagonal(+/-1) and Hadamard matrix, O(d log d) cost
- Spreads outliers uniformly, reducing quantization error by 2.6x

## Backward Pass with Warp Specialization

Three roles: Producer, Consumer, dQ-writer

- Producer: Loads K_j, V_j, then streams Q_i, dO_i blocks
- Consumer: Computes dK_j, dV_j accumulations; produces dQ_i^(local)
- dQ-writer: Atomically accumulates dQ_i^(local) to global dQ_i via semaphore (avoids blocking consumer)

## Performance Results

### Forward Pass (FP16, H100 80GB SXM5)
- 1.5-2.0x speedup over FlashAttention-2
- Up to 740 TFLOPs/s (75% utilization)
- Surpasses cuDNN for medium-to-long sequences (1k+)

### Backward Pass (FP16)
- 1.5-1.75x faster than FlashAttention-2
- Up to 561 TFLOPs/s for head dim 128

### FP8 Forward Pass
- Close to 1.2 PFLOPs/s for head dim 256
- Competitive with cuDNN FP8 kernels

### Ablation Study
| Configuration | Time | TFLOPs/s |
|---------------|------|----------|
| Full FlashAttention-3 | 3.538 ms | 661 |
| No GEMM-Softmax Pipelining, with Warp-Spec | 4.021 ms | 582 |
| GEMM-Softmax Pipelining, No Warp-Spec | 4.105 ms | 570 |

### Numerical Error (FP16)
| Method | RMSE |
|--------|------|
| Baseline FP16 | 3.2e-4 |
| FlashAttention-2 FP16 | 1.9e-4 |
| FlashAttention-3 FP16 | 1.9e-4 |

### Numerical Error (FP8)
| Method | RMSE |
|--------|------|
| Baseline FP8 (per-tensor scaling) | 2.4e-2 |
| FlashAttention-3 FP8 (block quant + incoherent) | 9.1e-3 |

## References

- FlashAttention-3 source: https://github.com/Dao-AILab/flash-attention
- CUTLASS library used for WGMMA and TMA abstractions
- Built on CUDA 12.3, cuDNN 9.1.1.17, CUTLASS 3.5
