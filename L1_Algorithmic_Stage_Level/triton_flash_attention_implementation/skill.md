---
skill_name: Flash Attention Triton Implementation
description: Practical step-by-step Triton implementation of the FlashAttention forward pass, showing how tiled QK^T, online softmax recurrence, and output accumulation map to GPU kernel code.
level: L1 - Algorithmic/Mathematical Stage Level
target_hardware: NVIDIA GPUs (any architecture supporting Triton)
relevance: When implementing custom FlashAttention variants in Triton, prototyping attention kernel modifications, or understanding how the mathematical algorithm maps to GPU code.
---

# Flash Attention Triton Implementation

## What It Is
This is a practical Triton implementation of the FlashAttention forward pass that translates the mathematical online-softmax tiled attention algorithm into readable GPU kernel code. It demonstrates how each kernel instance handles one Q tile, iterates over all K/V tiles, maintains running softmax statistics (max and denominator), accumulates the output with rescaling, and writes the final normalized result. The implementation uses exp2 (base-2 exponential) instead of exp (base-e) for direct hardware mapping, and accumulates in float32 for numerical precision.

## Key Concepts
- **Kernel Job Assignment:** Each kernel instance processes one Q tile against ALL K/V tiles, producing one output tile. This maximizes parallelism across Q tiles while ensuring each job sees the complete K/V sequence needed for correct softmax.
- **exp2 vs exp:** Hardware has native exp2 (base-2) instructions. The softmax scale factor is premultiplied by log2(e) so that exp2(x * log2(e)) = exp(x), avoiding the more expensive exp function.
- **Float32 Accumulation:** The output accumulator and softmax statistics are maintained in float32 even when inputs are float16/bf16, preventing precision loss during the many incremental updates.
- **Masking:** Out-of-sequence positions are set to -inf before softmax, ensuring they contribute zero weight.
- **Renormalization:** When a new tile produces a larger maximum, the previous accumulator and denominator are rescaled by alpha = exp2(m_old - m_new) before adding the new tile's contribution.

## Algorithm / Pseudo-code
```python
# Triton Flash Attention Forward Kernel
# Each program instance handles one Q tile

@triton.jit
def flash_attn_fwd(Q, K, V, Out, ...):
    # Determine which Q tile this instance handles
    q_tile_idx = tl.program_id(0)

    # Initialize per-row statistics
    m_i = tl.full([TILE_Q], float('-inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros([TILE_Q], dtype=tl.float32)                # running denom
    acc = tl.zeros([TILE_Q, HEAD_DIM], dtype=tl.float32)      # output accum

    # Load Q tile once (stays in registers/SRAM)
    q = load_tile(Q, q_tile_idx, TILE_Q, HEAD_DIM)

    # Pre-scale for exp2: scale * log2(e)
    softmax_scale = SM_SCALE * 1.44269504  # log2(e)

    # Stream over all K/V tiles
    for kv_idx in range(0, NUM_KV_TILES):
        # Load K^T and V tiles into SRAM
        kt = load_tile(K_transposed, kv_idx, HEAD_DIM, TILE_KV)
        v  = load_tile(V, kv_idx, TILE_KV, HEAD_DIM)

        # Compute partial scores: [TILE_Q, TILE_KV]
        qk = tl.dot(q * softmax_scale, kt)

        # Apply causal/padding mask
        qk = tl.where(valid_mask, qk, float('-inf'))

        # New per-row max across this tile
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))

        # Softmax numerators (using exp2 for hardware efficiency)
        p = tl.math.exp2(qk - m_ij[:, None])

        # This tile's contribution to denominator
        l_ij = tl.sum(p, axis=1)

        # Rescaling factor for previous accumulations
        alpha = tl.math.exp2(m_i - m_ij)

        # Update running denominator: l = l_old * alpha + l_new
        l_i = l_i * alpha + l_ij

        # Rescale previous output accumulator
        acc = acc * alpha[:, None]

        # Accumulate new contribution: acc += P @ V
        acc += tl.dot(p.to(v.dtype), v)

        # Update running max
        m_i = m_ij

    # Final normalization by softmax denominator
    acc = acc / l_i[:, None]

    # Store output tile to HBM
    store_tile(Out, q_tile_idx, acc)
```

## When to Use
- Prototyping custom attention variants (e.g., ALiBi, sliding window, prefix caching) in Triton
- Learning how FlashAttention maps to GPU code without the complexity of CUDA/CUTLASS
- When you need a readable reference implementation to validate against
- When Triton's JIT compilation and auto-tuning provide sufficient performance for the use case
- Educational contexts where understanding the algorithm-to-code mapping is the goal

## When NOT to Use
- Production inference/training on A100/H100 where the optimized CUDA FlashAttention-2/3 implementations are 10-30% faster than Triton
- When you need backward pass support (this tutorial covers only forward)
- On non-NVIDIA hardware where Triton may not be available or well-optimized

## Key Takeaways
- The Triton implementation is structurally identical to the mathematical algorithm: one Q tile, iterate K/V tiles, maintain (m, l, acc) statistics
- Using exp2 instead of exp maps directly to hardware instructions and is a common optimization in GPU kernels
- Float32 accumulation is essential: with float16 accumulators, the many incremental rescaling operations would accumulate unacceptable error
- The causal mask is applied by setting masked positions to -inf before the max/exp computation, which naturally produces zero attention weights
- Triton's tl.dot maps to Tensor Core instructions when tile sizes are compatible, giving near-CUDA performance with much simpler code

## References
- Tutorial: https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/
- Triton language: https://triton-lang.org
- OpenAI Triton Flash Attention tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
