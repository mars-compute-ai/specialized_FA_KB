---
skill_name: FlashAttention-3 GEMM-Softmax Pipelining
description: Overlapping GEMM (Q*K^T, P*V) with softmax computation using warp-specialized asynchronous pipelines to achieve 75% Hopper utilization.
level: L2 - Scheduling & Pipelining Level
target_hardware: NVIDIA Hopper H100/H200 (SM90)
relevance: When implementing or optimizing attention kernels that need to hide softmax latency behind tensor core GEMM operations, particularly for training workloads on Hopper GPUs.
---

# FlashAttention-3 GEMM-Softmax Pipelining

## What It Is
FlashAttention-3 introduces a novel pipelining strategy that overlaps GEMM operations (score computation Q*K^T and output accumulation P*V) with softmax computation using warp-specialized asynchronous execution on Hopper GPUs. The key insight is that softmax throughput (3.9 TFLOPS for exponentials) is 250x lower than FP16 matmul throughput (989 TFLOPS), meaning softmax can consume ~50% of execution time if not overlapped. By breaking the sequential GEMM-then-softmax dependency and using a 2-stage pipeline, FlashAttention-3 achieves 75% GPU utilization (up from 35% in FlashAttention-2).

## Key Concepts
- **250x throughput gap**: FP16 GEMM at 989 TFLOPS vs. softmax exponential at 3.9 TFLOPS drives the need for overlapping
- **Warp specialization**: Producer warps (TMA loads of Q, K, V tiles) and consumer warps (WGMMA + softmax) with register reallocation via `setmaxnreg`
- **2-stage GEMM-softmax pipeline**: Compute S_next = Q * K_j^T while executing softmax on S_cur from the previous iteration; execute P*V while computing softmax on the next scores
- **Circular shared memory buffers**: s-stage pipeline for K/V tiles managed by producer warps with barrier synchronization
- **Asynchronous WGMMA**: Non-blocking warpgroup MMA that allows softmax instructions to issue while GEMM executes on tensor cores
- **Register pressure trade-off**: Keeping S_next in registers requires Br x Bc x 4 extra bytes, limiting tile sizes
- **FP8 attention**: Requires in-kernel V transpose (LDSM/STSM), register layout permutation, and block quantization for accuracy
- **Persistent kernels**: FP16 path uses persistent kernel with load balancing for reduced launch overhead

## Pipeline Architecture / Pseudo-code
```
# FlashAttention-3: 2-Stage GEMM-Softmax Pipeline

PRODUCER_WARPS:
    tma_load(Q_i -> smem_Q)                     # Load query block once
    for j in range(num_kv_blocks):
        producer_acquire(stage[j % s])           # Wait for empty buffer
        tma_load(K_j -> smem_K[j % s])
        tma_load(V_j -> smem_V[j % s])
        # TMA hardware auto-commits barrier

CONSUMER_WARPS:
    # Prologue: compute first S block
    consumer_wait(stage[0])
    S_cur = wgmma(Q_i, K_0^T)                   # Score: Q * K^T
    warpgroup_wait()

    for j in range(1, num_kv_blocks):
        consumer_wait(stage[j % s])

        # === OVERLAP ZONE: 2-stage pipeline ===

        # Stage A: Issue next score GEMM (non-blocking)
        S_next = wgmma_async(Q_i, K_j^T)        # Tensor cores busy

        # Stage B: While GEMM executes, do softmax on PREVIOUS scores
        m_new = max(m_old, rowmax(S_cur))        # Running max
        P_cur = exp(S_cur - m_new)               # Softmax numerator
        l_new = exp(m_old - m_new) * l_old + rowsum(P_cur)  # Denominator
        O_i = rescale(O_i, m_old, m_new)         # Rescale running output

        # Stage C: Issue P*V GEMM (non-blocking)
        O_i += wgmma_async(P_cur, V_(j-1))      # Output accumulation

        # Wait for S_next GEMM to complete
        warpgroup_wait()
        S_cur = S_next                           # Rotate pipeline

        consumer_release(stage[(j-1) % s])       # Free buffer

    # Epilogue: final softmax + P*V for last block
    softmax(S_cur) -> P_cur
    O_i += wgmma(P_cur, V_last)
    O_i = O_i / l_final                          # Normalize

# Timeline (per consumer iteration):
# Tensor Cores: [--- S_next GEMM ---][--- P*V GEMM ---]
# SFU/ALU:      [--- softmax(S_cur) ---]
#                ^ softmax hidden behind GEMM ^
```

## Performance Impact
- **FlashAttention-2 to FlashAttention-3**: 1.5-2.0x speedup on forward pass
- **Peak throughput**: 740 TFLOPS/s in FP16 (75% of H100 theoretical max)
- **Utilization jump**: 35% (FA-2) to 75% (FA-3) on Hopper
- **Ablation: GEMM-softmax pipelining alone**: +14% throughput (582 -> 661 TFLOPS/s)
- **Ablation: warp-specialization alone**: +16% throughput (570 -> 661 TFLOPS/s)
- **FP8 peak**: ~1.2 PFLOPS/s (nearly 2x FP16), with 2.6x lower numerical error vs. naive per-tensor quantization
- **Backward pass**: 1.5-1.75x speedup over FlashAttention-2

## When to Use
- Training or inference with attention on Hopper H100/H200 GPUs
- Sequence lengths where softmax overhead is significant (typically seq_len >= 1024)
- FP16 or FP8 precision workloads (FP8 requires additional layout handling)
- When tensor core utilization is below 50%, indicating softmax is the bottleneck
- Building custom attention variants (sliding window, causal, cross-attention) that follow the Q*K^T -> softmax -> P*V pattern
- Large head dimensions (d=128, 256) where GEMM tiles are large enough to benefit from overlap

## When NOT to Use
- Pre-Hopper GPUs (Ampere, Volta) that lack asynchronous WGMMA and TMA—use FlashAttention-2 instead
- Very short sequences (seq_len < 256) where kernel launch overhead dominates
- Sparse attention patterns that don't follow dense GEMM structure
- When using Triton or other compilers that cannot express the fine-grained 2-stage pipeline
- Inference-only workloads where simpler implementations may suffice (though FA-3 still helps)

## Key Takeaways
- The 250x throughput gap between GEMM and softmax is the fundamental bottleneck that GEMM-softmax pipelining addresses
- Breaking the sequential GEMM->softmax->GEMM dependency requires keeping two score blocks in flight simultaneously, trading register pressure for pipeline depth
- Warp specialization (producer/consumer) and GEMM-softmax pipelining are complementary: together they provide ~30% throughput improvement over baseline
- FP8 attention on Hopper requires solving three sub-problems: V transpose, register layout permutation, and block quantization for accuracy
- The compiler (NVCC) cooperates with the pipeline by generating overlapped SASS instructions, but SASS analysis is needed to verify this
- FlashAttention-3's pipeline pattern generalizes: any kernel with alternating GEMM and non-GEMM phases can benefit from similar overlap strategies

## References
- Tri Dao, Jay Shah, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision": https://arxiv.org/abs/2407.08608
- FlashAttention-3 paper HTML: https://arxiv.org/html/2407.08608v2
- Tri Dao's page: https://tridao.me/publications/flash3/flash3.pdf
- CUTLASS WGMMA and TMA abstractions used in implementation
