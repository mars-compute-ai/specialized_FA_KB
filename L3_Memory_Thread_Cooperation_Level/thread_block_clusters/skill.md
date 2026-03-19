---
skill_name: Thread Block Clusters for Attention Kernels
description: Using Hopper/Blackwell thread block clusters with distributed shared memory and TMA multicast to share KV tiles across cooperating CTAs in attention kernels
level: L3 - Memory & Thread Cooperation Level
target_hardware: NVIDIA Hopper H100 (SM90), Blackwell B200 (SM100)
relevance: When designing attention kernels that can benefit from sharing KV tiles across multiple CTAs, reducing redundant HBM reads, or when exploring cluster-level parallelism for attention
---

# Thread Block Clusters for Attention Kernels

## What It Is
Thread block clusters are a cooperative thread hierarchy introduced in NVIDIA Hopper (SM90, CUDA compute capability 9.0) that group multiple CTAs (Cooperative Thread Arrays, i.e., thread blocks) into a single schedulable unit co-resident on nearby SMs within the same GPC (GPU Processing Cluster). Clusters enable three key capabilities absent from pre-Hopper GPUs: (1) distributed shared memory (DSMEM), where any CTA in the cluster can directly read/write another CTA's shared memory; (2) TMA multicast, where a single TMA load from global memory is broadcast to the shared memory of multiple CTAs simultaneously; and (3) cluster-level barriers for synchronization across CTAs without going through global memory. For attention kernels, clusters can reduce redundant KV tile loads by multicasting K/V tiles to all CTAs in a cluster that process different Q tiles but attend to the same KV block, effectively amortizing HBM bandwidth cost by the cluster size.

## Key Concepts
- **Cluster launch**: Clusters are specified at kernel launch via `cudaLaunchKernelEx` with `cudaLaunchAttributeClusterDimension` or in CUTLASS via `ClusterShape`. A cluster of shape (2, 1, 1) means 2 CTAs are grouped together. The hardware scheduler co-places cluster CTAs on adjacent SMs within the same GPC.
- **Distributed shared memory (DSMEM)**: Each CTA in a cluster can access any other CTA's shared memory using `cluster.map_shared_rank(smem_ptr, target_rank)`. This provides a larger effective shared memory pool across the cluster without global memory round-trips. Access latency is higher than local SMEM (~20-40 cycles vs ~20 cycles) but far lower than global memory (~300+ cycles).
- **TMA multicast**: When issuing a TMA load, the multicast mask specifies which CTAs in the cluster receive the data. A single HBM read distributes to N CTAs' shared memories simultaneously. For attention: one TMA read of a KV tile can populate SMEM on all cluster CTAs, reducing total HBM bandwidth by up to Nx.
- **Cluster barriers**: `cluster.sync()` synchronizes all CTAs in the cluster. Finer-grained synchronization uses `mbarrier` (async transaction barriers) with cluster scope, enabling producer-consumer patterns across CTAs.
- **GPC co-location**: The hardware guarantees cluster CTAs are on SMs within the same GPC (typically 2-8 SMs). This ensures L2 cache locality and low-latency DSMEM access.
- **Cluster size constraints**: Cluster size must be a power of 2 and is limited by GPC size (typically 2, 4, or 8 CTAs). Larger clusters increase scheduling constraints and may reduce overall occupancy since the scheduler must find contiguous SM slots.
- **Attention-specific pattern**: For attention, the natural cluster use is: cluster CTAs process different Q tiles (Q_0, Q_1, ...) but the same K/V tile stream. TMA multicast broadcasts each K_j, V_j to all CTAs in the cluster, so total HBM reads for K/V are reduced by cluster_size.

## Memory Layout / Data Flow
```
CLUSTER-BASED ATTENTION (Cluster size = 2):

Global Memory (HBM):
  Q[0..S_q-1, d], K[0..S_kv-1, d], V[0..S_kv-1, d]

                    TMA multicast
                    (single HBM read)
                         |
              +----------+----------+
              |                     |
     SM_0 (CTA 0)           SM_1 (CTA 1)
    +-----------+           +-----------+
    | SMEM:     |           | SMEM:     |
    |  Q_tile_0 |  DSMEM    |  Q_tile_1 |
    |  K_tile_j |<--------->|  K_tile_j |   (same K/V via multicast)
    |  V_tile_j |           |  V_tile_j |
    +-----------+           +-----------+
    | Compute:  |           | Compute:  |
    | S0=Q0*Kj^T|           | S1=Q1*Kj^T|
    | O0+=P0*Vj |           | O1+=P1*Vj |
    +-----------+           +-----------+

HBM Bandwidth Savings:
  Without clusters: 2 CTAs each load K_j, V_j = 2x HBM reads
  With cluster multicast: 1 TMA load of K_j, V_j -> both CTAs = 1x HBM reads
  Savings: 50% for cluster_size=2, 75% for cluster_size=4

CLUSTER LAUNCH (CUDA):
    cudaLaunchConfig_t config = {0};
    config.gridDim = {num_q_tiles, B * H, 1};
    config.blockDim = {128, 1, 1};

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};  // 2 CTAs per cluster
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, attention_kernel, Q, K, V, O, ...);

CUTLASS CLUSTER SPECIFICATION:
    using ClusterShape = Shape<_2, _1, _1>;  // 2 CTAs per cluster
    // KernelScheduler automatically handles TMA multicast

TMA MULTICAST EXAMPLE (PTX):
    // Load K_tile to SMEM, multicast to both CTAs in cluster
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
        [smem_ptr], [tma_desc, {coord_x, coord_y}], [mbar_ptr], multicast_mask;
    // multicast_mask = 0b11 -> both CTA 0 and CTA 1 receive the data

CLUSTER SYNCHRONIZATION:
    // Barrier across cluster CTAs
    cluster.sync();

    // Fine-grained: mbarrier with cluster scope
    mbarrier.init.shared::cta [mbar_ptr], expected_count;
    // Producer (CTA 0) signals:
    mbarrier.arrive.shared::cluster [mbar_ptr_on_cta1], tx_count;
    // Consumer (CTA 1) waits:
    mbarrier.try_wait.shared::cta [mbar_ptr], phase;

DSMEM ACCESS:
    // CTA 0 reads from CTA 1's shared memory
    int target_cta = 1;
    void* remote_smem = cluster.map_shared_rank(local_smem_ptr, target_cta);
    // Now remote_smem points to the equivalent address in CTA 1's SMEM
    data = *remote_smem;  // Direct load, no global memory round-trip
```

## Performance Impact
- **HBM bandwidth reduction**: TMA multicast reduces KV tile loads by cluster_size factor (2x for cluster=2, 4x for cluster=4)
- **CUTLASS GEMM benchmarks**: Cluster size 2-4 shows 5-15% speedup for large matrix sizes by reducing global memory traffic
- **Attention-specific**: Benefits depend on the ratio of KV loading cost to compute cost. For memory-bound attention (short sequences, large batch), clusters provide significant benefit. For compute-bound attention (long sequences), the benefit is smaller since KV loading is already hidden.
- **L2 cache locality**: Cluster CTAs on the same GPC share L2 cache partitions, improving hit rates for shared data
- **Scheduling overhead**: Large clusters (8 CTAs) can reduce SM occupancy due to contiguous scheduling requirements, potentially hurting short-running kernels
- **FA3 partial adoption**: FlashAttention-3 code references cluster support for KV broadcasting, though the primary speedups come from warp specialization and GEMM-softmax pipelining

## When to Use
- Attention kernels where KV loading from HBM is a significant fraction of total execution time (memory-bound regime)
- Multi-query attention (MQA) or grouped-query attention (GQA) where multiple Q heads attend to the same KV heads -- clusters can share KV across Q-head CTAs
- Long-context attention where each CTA processes a large KV range and KV tile reloading is expensive
- CUTLASS-based attention implementations where cluster support is integrated into the kernel scheduler
- When L2 cache pressure from multiple CTAs loading the same KV tiles causes cache thrashing
- Inference serving with large batch sizes where many sequences attend to similar KV patterns

## When NOT to Use
- Pre-Hopper GPUs (Ampere, Volta) -- clusters do not exist
- When attention is already compute-bound and KV loading is fully hidden behind WGMMA computation (clusters add complexity without benefit)
- Very small problem sizes where the scheduling constraint of co-locating cluster CTAs reduces overall SM utilization
- When cluster_size > GPC capacity, causing launch failures or degraded placement
- Decode-phase attention with split-KV scheduling where each split processes a different KV range (no KV sharing opportunity)
- Simple attention kernels in Triton or PyTorch that do not have low-level control over TMA and cluster launch
- When the kernel already achieves near-peak throughput without clusters (added complexity is not justified)

## Key Takeaways
- Thread block clusters introduce a new level in the CUDA thread hierarchy between CTAs and grids, enabling inter-CTA cooperation without global memory
- The primary attention benefit is TMA multicast: broadcasting KV tiles to multiple CTAs from a single HBM read, reducing memory bandwidth by the cluster size factor
- Distributed shared memory (DSMEM) provides a secondary benefit: CTAs can exchange intermediate results (e.g., partial softmax statistics) without global memory atomics
- Cluster scheduling has real costs: the hardware must co-locate CTAs on the same GPC, which can fragment SM availability and reduce occupancy for large cluster sizes
- The sweet spot for attention is cluster_size=2 -- it provides 2x KV bandwidth savings with minimal scheduling overhead
- Clusters compose with warp specialization: within each CTA, producer/consumer warps still manage TMA and WGMMA, while the cluster handles inter-CTA KV sharing
- CUTLASS 3.x provides the most accessible API for cluster-based kernels via `ClusterShape` template parameters; raw CUDA requires `cudaLaunchKernelEx` with launch attributes

## References
- [NVIDIA CUDA Programming Guide: Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [CUTLASS 3.x Documentation: Cluster Launch](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
- [FlashAttention-3 Paper (Shah et al., 2024)](https://tridao.me/publications/flash3/flash3.pdf)
- [TMA Deep-Dive](../tma_deep_dive/skill.md)
- [NVIDIA PTX ISA: Cluster Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)
