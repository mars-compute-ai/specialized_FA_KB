---
skill_name: Ring Attention - Distributed Flash Attention for Long Context
description: Distributes attention computation across GPUs in a ring topology, enabling near-infinite context length with zero communication overhead
level: L1 - Algorithmic/Mathematical Stage Level
target_hardware: Multi-GPU systems (NVIDIA H100/A100 with NVLink, AMD MI300X with Infinity Fabric)
relevance: When sequence length exceeds single-GPU memory capacity or when training/serving models with >128K context
---

# Ring Attention - Distributed Flash Attention

## What It Is
Ring Attention distributes the FlashAttention computation across multiple GPUs arranged in a ring. Each device holds a chunk of the sequence and computes blockwise attention locally while simultaneously sending its K/V blocks to the next device. By fully overlapping communication with computation, it achieves near-linear context length scaling with zero overhead.

## Key Concepts
- **Ring topology**: Devices pass K/V blocks around the ring; after D steps, every Q block has seen all K/V blocks
- **Blockwise attention**: Uses FlashAttention's tiled computation on each device — exact, no approximation
- **Online softmax correction**: Combines partial attention outputs across ring steps using running max/sum statistics
- **Communication overlap**: P2P K/V transfer happens concurrently with attention computation
- **Causal mask optimization**: Skip computation for future K/V blocks that are fully masked (saves ~50% compute for causal)

## Algorithm / Pseudo-code
```
# Ring Attention across D devices
# Device d holds Q_d (local queries), K_d, V_d (local KV)

Initialize O_d = 0, lse_d = -inf
K_buf, V_buf = K_d, V_d

for step in 0..D-1:
    # Async: send K_buf, V_buf to device (d+1) % D
    handle = isend(K_buf, V_buf, dst=(d+1)%D)

    # Compute: FlashAttention(Q_d, K_buf, V_buf)
    O_step, lse_step = flash_attn_block(Q_d, K_buf, V_buf, causal=(step==0))

    # Combine with running output (online softmax)
    O_d, lse_d = rescale_combine(O_d, lse_d, O_step, lse_step)

    # Wait for receive from device (d-1) % D
    K_buf, V_buf = irecv(src=(d-1)%D)
    wait(handle)
```

## When to Use
- Sequence length exceeds single-GPU HBM capacity (e.g., >128K tokens on 80GB GPU)
- Training long-context models (1M+ tokens)
- Multi-GPU inference with very long contexts
- Combined with FlashAttention for optimal per-device performance

## When NOT to Use
- Sequence fits comfortably on a single GPU — ring overhead is unnecessary
- Small batch sizes where communication cannot be fully overlapped
- Single-GPU deployments
- When approximate/sparse attention is acceptable (cheaper alternatives exist)

## AMD vs NVIDIA Considerations
- **NVIDIA**: NVLink provides 900 GB/s bidirectional bandwidth (H100), well-suited for KV block transfers
- **AMD**: Infinity Fabric on MI300X provides high-bandwidth inter-chiplet communication; ROCm supports P2P transfers
- The algorithm is hardware-agnostic — only requires P2P or all-gather communication primitives

## Key Takeaways
- Ring Attention is **exact** — mathematically equivalent to full attention, no approximation
- Context length scales **linearly** with GPU count: L_total = D × L_per_device
- Communication overhead is **zero** when compute time ≥ transfer time (typical for large head dims)
- Combines naturally with FlashAttention (per-device) and tensor parallelism (across heads)
- Modern frameworks (PyTorch Context Parallelism, DeepSpeed) provide built-in implementations

## References
- [Ring Attention Paper (arXiv 2310.01889)](https://arxiv.org/abs/2310.01889)
- [PyTorch Context Parallelism](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)
- [GPU MODE Lecture 13: Ring Attention](https://christianjmills.com/posts/cuda-mode-notes/lecture-013/)
