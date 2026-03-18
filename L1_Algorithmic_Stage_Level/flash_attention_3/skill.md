---
skill_name: FlashAttention-3 Asynchronous Pipelined Attention
description: Warp-specialized attention algorithm that hides softmax latency under asynchronous GEMM and supports FP8 via block quantization and incoherent processing on Hopper GPUs.
level: L1 - Algorithmic/Mathematical Stage Level
target_hardware: NVIDIA Hopper H100 GPU (and architectures with async Tensor Cores + TMA)
relevance: When targeting maximum attention throughput on H100 GPUs, when FP8 attention is needed for training/inference efficiency, or when designing warp-specialized GPU kernels.
---

# FlashAttention-3 Asynchronous Pipelined Attention

## What It Is
FlashAttention-3 (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao -- 2024) redesigns the FlashAttention algorithm to exploit Hopper GPU hardware features. It introduces three techniques: (1) warp-specialized producer-consumer pipelining that overlaps TMA data movement with Tensor Core computation, (2) a 2-stage GEMM-softmax pipeline that hides the low-throughput softmax (3.9 TFLOPS for exp) under the high-throughput asynchronous WGMMA (989 TFLOPS for FP16 matmul), and (3) FP8 support using block quantization and incoherent processing (Hadamard rotation) to reduce quantization error by 2.6x.

## Key Concepts
- **Warp Specialization:** CTA is split into producer warpgroup (issues TMA loads of K_j, V_j into circular SMEM buffer) and consumer warpgroup(s) (execute WGMMA for QK^T and PV, plus softmax). Producers deallocate registers to give consumers more; `setmaxnreg` enables dynamic register redistribution.
- **Pingpong Scheduling:** Two consumer warpgroups alternate: while WG1 does GEMMs, WG2 does softmax, then they swap. Hides the 256x throughput gap between matmul and special functions.
- **2-Stage GEMM-Softmax Pipeline:** Within a single warpgroup, the QK^T GEMM for iteration j+1 is issued (commit but don't wait) while softmax for iteration j executes, and the PV GEMM for iteration j is overlapped with softmax for iteration j+1. This exploits WGMMA's asynchronous execution.
- **Circular SMEM Buffer:** s-stage buffer for K/V tiles enables producers to stay ahead of consumers, hiding TMA latency.
- **FP8 Layout Challenges:** FP8 WGMMA requires k-major operands (unlike FP16 which accepts both). V tiles need in-kernel transpose using LDSM/STSM. FP32 accumulator layout differs from FP8 operand A layout, requiring byte permute instructions.
- **Block Quantization:** One scaling factor per (B_r x d) or (B_c x d) block instead of per-tensor. Naturally aligns with tiled execution. Can be fused with rotary embedding at no cost.
- **Incoherent Processing:** Multiply Q, K by random orthogonal M = diag(+/-1) * Hadamard before FP8 quantization. Since (QM)(KM)^T = QK^T, output unchanged but outliers are spread, reducing quantization error. O(d log d) cost via fast Hadamard transform.

## Algorithm / Pseudo-code
```
# FlashAttention-3 Forward Pass (CTA-level, 2-stage pipeline)

# PRODUCER WARPGROUP:
  deallocate_registers()
  TMA_load(Q_i -> SMEM)
  for j = 0 to T_c-1:
      wait(buffer_stage[j % s] consumed)
      TMA_load(K_j, V_j -> SMEM[j % s])
      signal(consumers: K_j, V_j ready)

# CONSUMER WARPGROUP (2-stage GEMM-softmax pipeline):
  reallocate_registers()
  O_i = 0, l_i = 0, m_i = -inf

  # Prologue: first iteration
  wait(K_0 ready)
  S_cur = WGMMA(Q_i, K_0^T)  # SS-GEMM, commit+wait
  compute softmax: m_i, P_tilde_cur, l_i from S_cur

  # Main loop: overlapped iterations
  for j = 1 to T_c - 1:
      wait(K_j ready)
      S_next = WGMMA(Q_i, K_j^T)        # commit, DO NOT WAIT
      wait(V_{j-1} ready)
      O_i += WGMMA(P_tilde_cur, V_{j-1}) # commit, DO NOT WAIT

      wait(S_next)                         # softmax overlaps with PV GEMM above
      compute m_i, P_tilde_next, l_i from S_next

      wait(PV GEMM); rescale O_i
      S_cur = S_next; P_tilde_cur = P_tilde_next

  # Epilogue: final PV
  O_i += WGMMA(P_tilde_last, V_{T_c-1}), commit+wait
  O_i = diag(l_i)^{-1} * O_i
  L_i = m_i + log(l_i)
  write O_i, L_i to HBM

# BACKWARD PASS adds a dQ-writer warp role:
  Producer: loads K_j, V_j, then streams Q_i, dO_i
  Consumer: accumulates dK_j, dV_j; produces dQ_i^(local) to SMEM
  dQ-writer: atomically adds dQ_i^(local) to global dQ_i (avoids blocking consumer)

# FP8 VARIANT:
  # Pre-process: Q, K *= diag(+/-1) * Hadamard  (incoherent processing)
  # Block quantize: one scale per (B_r x d) block
  # In-kernel: transpose V tiles via LDSM/STSM for k-major layout
  # Byte permute: convert FP32 accumulator to FP8 operand A layout
```

## When to Use
- Training or inference on H100 GPUs where FA2 only achieves ~35% utilization
- When FP8 precision is acceptable and you want to approach 1.2 PFLOPS
- Long-context applications (1K+ tokens) where FA3 outperforms even cuDNN
- When you need maximum attention throughput and can target Hopper-specific instructions
- Models with outlier features (LLMs) where block quantization + incoherent processing preserves accuracy

## When NOT to Use
- On Ampere (A100) or older GPUs that lack TMA, WGMMA, and `setmaxnreg` -- use FA2 instead
- On Blackwell GPUs where FA4 provides further optimizations (conditional rescaling, cubic exp approx)
- When the model does not bottleneck on attention (e.g., very small sequence lengths)
- When you need FP8 backward pass (FA3's FP8 support is forward-pass only in initial release)

## Key Takeaways
- The throughput gap between Tensor Core matmul (989 TFLOPS) and softmax exp (3.9 TFLOPS) on H100 means softmax can take 50% of the cycle budget -- hiding it under async WGMMA is the single biggest win
- Warp specialization + pingpong scheduling together boost throughput from 570 to 661 TFLOPS (16% gain from each technique)
- FP8 block quantization + incoherent processing reduces RMSE by 2.6x vs naive per-tensor FP8, making FP8 attention practical
- The in-kernel V transpose for FP8 layout compliance is a key implementation detail that can be hidden under the GEMMs
- FA3 achieves 740 TFLOPS FP16 (75% utilization) and ~1.2 PFLOPS FP8, up from FA2's ~350 TFLOPS (35% utilization)

## References
- Paper: Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," 2024: https://tridao.me/publications/flash3/flash3.pdf
- Code: https://github.com/Dao-AILab/flash-attention
- Built on CUTLASS 3.5 primitives for WGMMA and TMA abstractions
