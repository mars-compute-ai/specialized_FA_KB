# Ring Attention with Blockwise Transformers for Near-Infinite Context

## Overview

Ring Attention (Liu et al., 2023) is a distributed attention algorithm that enables near-infinite context lengths by distributing sequence processing across multiple GPUs in a ring topology. It overlaps the communication of key-value blocks with blockwise attention computation, achieving zero-overhead scaling of context size.

## Core Algorithm

### Ring Communication Pattern
1. Each device holds a contiguous chunk of the sequence (Q, K, V blocks)
2. Devices are arranged in a logical ring
3. At each step:
   - Each device computes attention between its local Q block and the current K/V block
   - Simultaneously, K/V blocks are sent to the next device in the ring (via P2P or all-gather)
4. After N steps (N = number of devices), every Q block has attended to all K/V blocks
5. Partial outputs are combined using the online softmax correction (same as FlashAttention)

### Integration with FlashAttention
Ring Attention uses blockwise attention computation identical to FlashAttention's tiling:
- Each device computes `O_partial, lse_partial = FlashAttention(Q_local, K_current, V_current)`
- Running statistics (max, sum for softmax) are maintained and corrected as new K/V blocks arrive
- This is mathematically equivalent to computing full attention — no approximation

### Pseudo-code
```python
# Ring Attention (simplified)
# Each device d has: Q_d, K_d, V_d (local chunks)
# Ring of D devices

O_d = zeros_like(Q_d)
lse_d = -inf  # log-sum-exp accumulator

for step in range(D):
    src = (d - step) % D  # which device's KV we're processing
    K_cur, V_cur = K[src], V[src]

    # Overlap: send K_cur, V_cur to next device
    async_send(K_cur, V_cur, to=(d+1) % D)

    # Compute blockwise attention (FlashAttention-style)
    O_partial, lse_partial = flash_attention(Q_d, K_cur, V_cur)

    # Online softmax correction to combine with running output
    O_d, lse_d = online_combine(O_d, lse_d, O_partial, lse_partial)

    # Receive next KV block
    K_cur, V_cur = async_recv(from=(d-1) % D)
```

## Key Properties

### Scaling
- **Context length scales linearly** with number of devices: L_total = D × L_per_device
- Communication is fully overlapped with computation when the attention compute time ≥ KV transfer time
- No additional memory overhead beyond what FlashAttention already requires per device

### Communication Analysis
- Each step transfers 2 × L_chunk × d_head × n_heads × sizeof(dtype) bytes
- With NVLink (900 GB/s bidirectional on H100), the transfer time is typically much smaller than the compute time for practical head dimensions
- Zero overhead for sufficiently large chunk sizes

### Comparison with Other Approaches
| Method | Context Scaling | Approximation | Communication |
|--------|----------------|---------------|---------------|
| Ring Attention | Linear with GPUs | Exact | Overlapped P2P |
| Sequence Parallelism (Megatron) | Limited | Exact | All-reduce |
| DeepSpeed Ulysses | Linear with GPUs | Exact | All-to-all |
| Longformer/BigBird | Fixed window | Approximate | None |

## Variants and Extensions

### Context Parallelism (PyTorch)
PyTorch's `torch.distributed` provides `context_parallel` APIs that implement Ring Attention-style distribution, integrated with FSDP and tensor parallelism.

### USP (Unified Sequence Parallelism)
Combines DeepSpeed-Ulysses (all-to-all on heads) with Ring Attention (ring on sequence), achieving better performance depending on the sequence length vs. head count ratio.

### StarTrail
Concentric ring topology that reduces the number of communication rounds for causal attention by exploiting the triangular structure of the causal mask.

## Performance
- Enables training with millions of tokens context on clusters of GPUs
- Near-linear scaling efficiency when compute dominates communication
- Used in practice by systems like Llama long-context training, Google Gemini, etc.

## References
- [Ring Attention Paper (arXiv 2310.01889)](https://arxiv.org/abs/2310.01889)
- [PyTorch Context Parallelism Tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)
- [GPU MODE Lecture 13: Ring Attention](https://christianjmills.com/posts/cuda-mode-notes/lecture-013/)
- [Ring Attention: Shedding Light (Akasa blog)](https://akasa.com/blog/ring-attention)
