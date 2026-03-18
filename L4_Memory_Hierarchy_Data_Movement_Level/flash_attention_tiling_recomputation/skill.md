---
skill_name: FlashAttention Tiling and Recomputation
description: IO-aware attention algorithm that tiles computation into SRAM and recomputes intermediates to reduce HBM traffic from O(N²) to O(N)
level: L4 - Memory-Hierarchy/Data-Movement Level
target_hardware: NVIDIA Ampere A100, Hopper H100, and any GPU with programmer-managed shared memory
relevance: When an AI agent needs to understand or implement memory-efficient attention, optimize sequence length scaling, or reduce memory-bandwidth bottlenecks in transformer models
---

# FlashAttention Tiling and Recomputation

## What It Is
FlashAttention is an IO-aware exact attention algorithm that avoids materializing the full N×N attention matrix in slow HBM by processing Q, K, V in tiles that fit in fast on-chip SRAM. It uses online softmax to compute exact results incrementally and recomputes intermediate values during the backward pass instead of storing them, reducing memory complexity from O(N²) to O(N) while maintaining identical mathematical output.

## Key Concepts
- **Tiling**: Q, K, V are divided into blocks sized to fit in SRAM (~128-192 KB per SM); all matmuls and softmax happen on-chip
- **Online softmax**: Maintains running statistics (row max m, sum of exponentials l) to compute exact softmax incrementally across blocks
- **Recomputation**: Backward pass reloads Q, K, V blocks and recomputes S, P in SRAM rather than reading stored N×N matrices from HBM
- **IO-awareness**: Algorithm is designed around the memory hierarchy — trades extra FLOPs (cheap) for fewer HBM accesses (expensive)
- **No approximation**: Results are bit-for-bit identical to standard attention
- **HBM access reduction**: From Θ(Nd + N²) to Θ(N²d²M⁻¹), roughly 33x fewer for typical configs

## Memory Layout / Data Flow
```
HBM (slow, 2 TB/s)                    SRAM (fast, 19 TB/s)
┌─────────────────┐                   ┌──────────────────┐
│ Q [N × d]       │──── load Q_i ───→│ Q_i [B_r × d]    │
│ K [N × d]       │──── load K_j ───→│ K_j [B_c × d]    │
│ V [N × d]       │──── load V_j ───→│ V_j [B_c × d]    │
│                 │                   │                  │
│ O [N × d]       │←── write O_i ────│ S_ij = Q_i @ K_j^T│
│ (m, l) [N × 1]  │←── write stats ──│ P_ij = softmax(S) │
│                 │                   │ O_i += P_ij @ V_j │
│ ✗ S [N×N] NEVER │                   │ m, l (running)    │
│ ✗ P [N×N] NEVER │                   └──────────────────┘
└─────────────────┘

Forward: Only O and (m,l) written to HBM — N×N never materialized
Backward: Reload Q,K,V blocks; recompute S,P in SRAM using stored (m,l)
```

## Performance Impact
- Up to **4x wall-clock speedup** over standard attention (forward + backward)
- FlashAttention-2: **70% of theoretical peak FLOPS** on A100
- FlashAttention-3: **75% utilization** of H100 theoretical maximum
- **20x more memory-efficient** than exact attention baselines
- Enables **16K-64K token** sequence lengths that would OOM with standard attention
- BERT-large: 15% end-to-end speedup; GPT-2: 3x speedup vs HuggingFace
- **33x reduction** in HBM memory traffic for typical configurations (N=4096, d=128)

## When to Use
- Any transformer model with self-attention or cross-attention layers
- When sequence length scaling is a bottleneck (quadratic memory → linear)
- When training is memory-bandwidth-bound rather than compute-bound
- When you need exact attention (not approximate) with better performance
- Long-context models (16K+ tokens) where standard attention would OOM
- Both training (forward + backward with recomputation) and inference (forward-only)

## When NOT to Use
- Very short sequences (N < 128) where tiling overhead exceeds benefit
- Single-token autoregressive decode (only one query token; minimal N×N savings)
- When using sparse or linear attention approximations that are sufficient for the task
- When custom attention patterns require intermediate matrix access that tiling cannot support
- Hardware without programmer-managed shared memory (non-GPU accelerators)

## Key Takeaways
- FlashAttention's core insight: **attention is memory-bound, not compute-bound** — reducing HBM traffic matters more than reducing FLOPs
- Tiling + online softmax enables exact attention without ever materializing N×N matrices in HBM
- Recomputation in the backward pass is "free" because the saved HBM bandwidth more than compensates for extra FLOPs
- Memory complexity drops from O(N²) to O(N), enabling much longer sequences
- This is the foundational algorithm for all modern efficient attention implementations

## References
- [FlashAttention Paper (arXiv 2205.14135)](https://arxiv.org/abs/2205.14135)
- [Flash Attention: Transformer Optimization (DeepFA)](https://deepfa.ir/en/blog/flash-attention-transformer-optimization)
- [Paper Summary: FlashAttention](https://shreyansh26.github.io/post/2023-03-26_flash-attention/)
- [Attention Optimizations Overview (HuggingFace)](https://huggingface.co/blog/atharv6f/flash-attention-overview)
