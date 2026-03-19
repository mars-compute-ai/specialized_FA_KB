---
skill_name: CUTLASS Ping-Pong GEMM Pipeline
description: A warp-specialized GEMM kernel design that alternates two consumer warp groups to overlap epilogue with tensor core compute, maximizing Hopper GPU throughput.
level: L2 - Scheduling & Pipelining Level
target_hardware: NVIDIA Hopper H100/H200 (SM90), extensible to Blackwell
relevance: When designing high-throughput GEMM or attention kernels on Hopper GPUs where epilogue overhead limits tensor core utilization, and when building FlashAttention-style fused kernels that need deep producer-consumer pipelines.
---

# CUTLASS Ping-Pong GEMM Pipeline

## What It Is
The Ping-Pong GEMM kernel is a warp-specialized matrix multiplication design in CUTLASS 3.x for Hopper GPUs that uses three warp groups: one producer (issuing TMA loads) and two consumers (performing WGMMA compute + epilogue). The two consumer warp groups alternate in a "ping-pong" fashion—while one consumer executes MMA operations, the other writes its results to global memory (epilogue)—ensuring tensor cores are never idle waiting for memory writes. This represents the fully asynchronous execution paradigm that Hopper's architecture was designed to enable.

## Key Concepts
- **Three warp groups**: 1 producer (TMA loads) + 2 consumers (WGMMA compute + epilogue), each with a dedicated role
- **Ping-pong alternation**: Consumer A does MMA while Consumer B does epilogue, then they swap—overlapping epilogue latency with compute
- **Ordered Sequence Barrier**: Producer fills buffers for the two consumers sequentially in order
- **TMA (Tensor Memory Accelerator)**: Hardware unit that handles global-to-shared-memory transfers asynchronously with automatic `producer_commit()`
- **Multi-stage pipeline**: N shared memory buffers (typically 3-7) managed via `mbarrier` phase bits and arrival counts
- **WGMMA**: Warp Group MMA operating on 128 threads (4 warps), using matrix descriptors pointing to SMEM rather than data copies
- **Persistent kernel**: One CTA per SM computes multiple output tiles, with the Tile Scheduler assigning different tiles to each consumer
- **Pipeline API**: Four core methods—`producer_acquire()`, `producer_commit()`, `consumer_wait()`, `consumer_release()`

## Pipeline Architecture / Pseudo-code
```
# CUTLASS Ping-Pong GEMM: 3 Warp Groups

PRODUCER_WARP_GROUP:
    for each k_tile in K_tiles:
        # Fill Consumer A's buffer
        producer_acquire(state_A)           # Wait: buffer empty
        tma_async_load(A_tile -> smem_A[stage])
        tma_async_load(B_tile -> smem_B[stage])
        # TMA hardware auto-commits (producer_commit)
        advance(state_A)

        # Fill Consumer B's buffer (Ordered Sequence Barrier)
        producer_acquire(state_B)
        tma_async_load(A_tile -> smem_A2[stage])
        tma_async_load(B_tile -> smem_B2[stage])
        advance(state_B)

CONSUMER_WARP_GROUP_A (tile_0 from TileScheduler):
    for each k_tile:
        consumer_wait(read_state)           # Wait: data ready
        warpgroup_arrive()
        wgmma(smem_A[stage], smem_B[stage], accumulator)
        warpgroup_commit_batch()
        warpgroup_wait<0>()
        consumer_release(read_state)        # Signal: buffer free
        advance(read_state)
    # Epilogue: write accumulator -> global memory
    # (Overlaps with Consumer B's MMA phase)

CONSUMER_WARP_GROUP_B (tile_1 from TileScheduler):
    # Same structure as Consumer A, but phase-shifted
    # When A computes -> B does epilogue
    # When B computes -> A does epilogue

# Timeline:
# Time -->
# Producer:  [Load A0] [Load B0] [Load A1] [Load B1] ...
# Consumer A: -------- [MMA    ] [Epilogue] [MMA    ] ...
# Consumer B: -------- [Epilogue] [MMA    ] [Epilogue] ...
#                       ^ Tensor cores always busy ^
```

## Performance Impact
- Achieves ~65% GPU utilization in half-precision on Hopper through proper pipeline overlap
- Eliminates tensor core idle time during epilogue by alternating two consumer groups
- Hardware context: H200 delivers 3,958 TFLOPS compute vs. 4.8 TB/s bandwidth—the extreme compute-to-bandwidth ratio makes aggressive pipelining essential
- Ping-pong specifically outperforms cooperative schedule when epilogue is non-trivial (e.g., fused activation functions, bias addition)
- Multi-stage buffering (3-7 stages) provides sufficient depth to hide global memory latency (~400-800 cycles)

## When to Use
- Targeting peak GEMM throughput on Hopper H100/H200 GPUs
- Epilogue operations are non-trivial (fused activations, bias, residual connections) and would otherwise stall tensor cores
- Building FlashAttention-style kernels that need to overlap softmax/rescaling with MMA operations
- Output tiles are large enough that each consumer warp group has substantial work
- Using persistent kernel scheduling (one CTA per SM computing multiple tiles)
- Half-precision (FP16/BF16) or FP8 workloads where tensor core throughput is the target

## When NOT to Use
- Pre-Hopper architectures (Ampere, Volta) that lack TMA and WGMMA—use multistage `cp.async` pipelines instead
- Very small matrices where kernel launch overhead dominates
- Epilogue is trivial (simple store)—cooperative schedule may be simpler and equally fast
- Compute-bound kernels with minimal data movement where simple multistage suffices
- Memory-bandwidth-bound operations (e.g., very tall-skinny matrices) where pipelining compute doesn't help

## Key Takeaways
- The ping-pong design solves a specific problem: tensor core underutilization caused by epilogue stalls
- Three warp groups with strict role separation (1 producer + 2 alternating consumers) is the core pattern
- TMA hardware handles the producer side with minimal software overhead, automatically managing barrier signaling
- The pattern extends naturally to FlashAttention: replace simple epilogue with softmax rescaling and output accumulation
- CUTLASS 3.x's `MainloopSm90TmaGmmaWarpSpecialized` with `KernelTmaWarpSpecializedPingpong` schedule encapsulates this entire pattern
- Multi-stage shared memory buffering (controlled by `StageCountAuto`) determines pipeline depth based on available SMEM

## References
- PyTorch Blog, "Deep Dive on CUTLASS Ping-Pong GEMM Kernel": https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/
- Colfax Research, "CUTLASS Tutorial: Efficient GEMM Kernel Designs with Pipelining": https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/
- Colfax Research, "CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on Hopper GPUs": https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
- NVIDIA Blog, "CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design": https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design
- CUTLASS source: `cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp`
