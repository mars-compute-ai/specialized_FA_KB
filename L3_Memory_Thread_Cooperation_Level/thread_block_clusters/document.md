# Thread Block Clusters for Attention Kernels

## Overview

Thread block clusters, introduced in NVIDIA Hopper (SM90, compute capability 9.0), add a new level to the CUDA thread hierarchy between CTAs and grids. A cluster is a group of CTAs (typically 2, 4, or 8) that are guaranteed to be co-scheduled on nearby SMs within the same GPC (GPU Processing Cluster). This co-scheduling guarantee enables three capabilities: distributed shared memory (DSMEM) for direct cross-CTA SMEM access, TMA multicast for broadcasting global memory loads to multiple CTAs simultaneously, and cluster-scoped barriers for efficient inter-CTA synchronization.

For Flash Attention kernels, clusters offer a specific optimization: when multiple CTAs process different Q tiles but stream the same KV tiles, TMA multicast can broadcast each KV tile to all cluster CTAs from a single HBM read, reducing total memory bandwidth consumption by the cluster size factor. This document covers the cluster architecture, programming model, attention-specific applications, and practical trade-offs.

---

## 1. CUDA Thread Hierarchy with Clusters

### 1.1 Complete Hierarchy

```
Grid
 └── Thread Block Cluster (NEW in Hopper)
      └── CTA (Thread Block)
           └── Warpgroup (4 warps = 128 threads, Hopper+)
                └── Warp (32 threads)
                     └── Thread

Pre-Hopper hierarchy (for comparison):
Grid -> CTA -> Warp -> Thread

Cluster adds: Grid -> Cluster -> CTA -> Warpgroup -> Warp -> Thread
```

### 1.2 Memory Visibility by Hierarchy Level

| Level | Private Memory | Shared Memory | Can Synchronize |
|-------|---------------|---------------|-----------------|
| Thread | Registers, local memory | -- | -- |
| Warp | -- | -- | warp-level primitives |
| Warpgroup | -- | -- | warpgroup barriers |
| CTA | -- | SMEM (local to CTA) | __syncthreads() |
| **Cluster** | -- | **DSMEM (all CTAs in cluster)** | **cluster.sync()** |
| Grid | -- | Global memory | cooperative groups |

### 1.3 Hardware Mapping

```
NVIDIA H100 SXM Physical Layout:
  8 GPCs (GPU Processing Clusters)
  x ~16 SMs per GPC = ~132 SMs total

Cluster Placement:
  All CTAs in a cluster -> same GPC
  Cluster size <= SMs per GPC (max ~16, practical limit 8)

Implications:
  - Low-latency DSMEM access (same GPC, shared L2 partition)
  - L2 cache locality (cluster CTAs share L2 slice)
  - Scheduling constraint: must find cluster_size contiguous SMs in same GPC
```

---

## 2. Cluster Programming Model

### 2.1 Cluster Launch

```cpp
// Method 1: cudaLaunchKernelEx (runtime API)
__global__ void __cluster_dims__(2, 1, 1)
attention_kernel(float* Q, float* K, float* V, float* O, int seq_len);
// __cluster_dims__ attribute sets compile-time cluster shape

// Alternative: set cluster dimensions at launch time
cudaLaunchConfig_t config;
memset(&config, 0, sizeof(config));

config.gridDim = dim3(num_q_tiles, B * H, 1);
config.blockDim = dim3(128, 1, 1);  // 128 threads per CTA

cudaLaunchAttribute attr;
attr.id = cudaLaunchAttributeClusterDimension;
attr.val.clusterDim.x = 2;  // 2 CTAs per cluster
attr.val.clusterDim.y = 1;
attr.val.clusterDim.z = 1;

config.attrs = &attr;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, attention_kernel, Q, K, V, O, seq_len);

// Method 2: CUTLASS 3.x (template-based)
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
// Passed as template parameter to CollectiveMainloop
```

### 2.2 Cluster Identification

```cpp
__device__ void attention_kernel(...) {
    // Cluster-level indexing
    uint32_t cluster_id = cyclic_cluster_id();  // which cluster am I in?
    uint32_t rank_in_cluster = block_rank_in_cluster();  // my rank within cluster
    uint32_t cluster_size = num_blocks_in_cluster();  // total CTAs in cluster

    // Example for cluster_size=2:
    // CTA 0: cluster_id=0, rank=0
    // CTA 1: cluster_id=0, rank=1
    // CTA 2: cluster_id=1, rank=0
    // CTA 3: cluster_id=1, rank=1
    // ...

    // For attention: cluster CTAs process adjacent Q tiles
    int q_tile_id = blockIdx.x;  // global Q tile index
    int bh_id = blockIdx.y;      // batch * head index

    // Within cluster: CTA rank determines which Q tile subset
    // All CTAs in cluster share the same KV tile stream
}
```

### 2.3 Distributed Shared Memory (DSMEM)

```cpp
__device__ void access_remote_smem() {
    // Local shared memory
    __shared__ float local_smem[1024];

    // Get pointer to another CTA's shared memory
    uint32_t target_rank = 1;  // access CTA rank 1's SMEM
    float* remote_smem = cluster.map_shared_rank(local_smem, target_rank);

    // Direct read from remote CTA's SMEM (no global memory!)
    float val = remote_smem[threadIdx.x];

    // Direct write to remote CTA's SMEM
    remote_smem[threadIdx.x] = my_result;
}
```

**DSMEM Access Characteristics**:

| Access Type | Latency | Bandwidth | Notes |
|------------|---------|-----------|-------|
| Local SMEM | ~20 cycles | ~100+ TB/s per SM | Standard shared memory access |
| DSMEM (same GPC) | ~40-60 cycles | Lower than local | Cross-SM interconnect within GPC |
| Global memory | ~300+ cycles | Up to 3.35 TB/s total | Via L2 cache |

DSMEM is 5-7x faster than going through global memory but 2-3x slower than local SMEM. Use it for data that would otherwise require global memory round-trips.

### 2.4 TMA Multicast

TMA multicast is the highest-value cluster feature for attention kernels:

```cpp
__device__ void tma_multicast_kv_tile(
    const CUtensorMap* tma_desc_K,
    float* smem_K,
    uint64_t* mbar,
    int kv_tile_idx)
{
    uint32_t my_rank = block_rank_in_cluster();

    // Only one CTA issues the TMA load (producer pattern)
    if (my_rank == 0 && threadIdx.x == 0) {
        // Multicast mask: 0b11 = both CTAs in a 2-CTA cluster
        uint16_t multicast_mask = 0b11;  // all cluster members receive data

        // Single TMA load -> broadcasts to all CTAs' SMEM
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1, {%2, %3}], [%4], %5;"
            :
            : "r"(smem_K),           // destination (SMEM in all cluster CTAs)
              "l"(tma_desc_K),       // TMA descriptor
              "r"(kv_tile_idx),      // coordinate x
              "r"(0),               // coordinate y
              "r"(mbar),            // mbarrier for completion tracking
              "h"(multicast_mask)    // which CTAs receive the data
        );
    }

    // All CTAs wait on their local mbarrier
    // (mbarrier tracks arrivals from TMA + all cluster CTAs)
    mbarrier_wait(mbar, phase);

    // Now smem_K contains the KV tile on ALL cluster CTAs
    // Only ONE HBM read was performed!
}
```

**Bandwidth savings from TMA multicast**:

```
Without multicast (N CTAs load the same KV tile independently):
  Total HBM reads = N * tile_size_bytes
  HBM bandwidth consumed = N * tile_size_bytes / kernel_time

With multicast (1 TMA load, N CTAs receive):
  Total HBM reads = 1 * tile_size_bytes
  HBM bandwidth consumed = 1 * tile_size_bytes / kernel_time

Savings factor = N (cluster size)

Example: cluster_size=4, KV tile = 128 x 128 x 2 bytes (FP16) = 32 KB
  Without multicast: 4 * 32 KB = 128 KB per tile load
  With multicast: 1 * 32 KB = 32 KB per tile load
  Savings: 75% bandwidth reduction for KV loading
```

### 2.5 Cluster Barriers

```cpp
// Full cluster synchronization
__device__ void full_cluster_sync() {
    // Blocks all threads in all CTAs of the cluster
    // Expensive: avoid in inner loops
    asm volatile("barrier.cluster.arrive;\n"
                 "barrier.cluster.wait;\n");
}

// Fine-grained: mbarrier with cluster scope
__device__ void cluster_mbarrier_example() {
    __shared__ uint64_t mbar;

    // Initialize barrier (CTA 0 only)
    if (block_rank_in_cluster() == 0 && threadIdx.x == 0) {
        mbarrier_init(&mbar, expected_arrivals);
    }
    cluster.sync();  // ensure mbar is visible to all cluster CTAs

    // Any CTA can arrive at another CTA's barrier
    uint32_t target_rank = 0;
    uint64_t* remote_mbar = cluster.map_shared_rank(&mbar, target_rank);
    mbarrier_arrive(remote_mbar);

    // CTA 0 waits for all arrivals
    if (block_rank_in_cluster() == 0) {
        mbarrier_wait(&mbar, phase);
    }
}
```

---

## 3. Cluster-Based Attention Kernel Design

### 3.1 Architecture: Cluster KV Broadcasting

```
Attention Kernel with Cluster Size 2:

Grid: (num_q_tiles / 2, B * H)  -- pairs of Q tiles per cluster
Block: 128 threads (1 warpgroup) or 256 threads (2 warpgroups)

Cluster: [CTA 0: Q_tile_2i] [CTA 1: Q_tile_2i+1]
         Both process the SAME stream of K/V tiles

Per cluster, for each KV tile j:
  1. CTA 0 (producer) issues TMA multicast of K_j, V_j
     -> K_j appears in SMEM of both CTA 0 and CTA 1
  2. CTA 0: S_0 = Q_0 @ K_j^T, O_0 += softmax(S_0) @ V_j
     CTA 1: S_1 = Q_1 @ K_j^T, O_1 += softmax(S_1) @ V_j
     (computed independently, no inter-CTA data dependency)
  3. CTA 0 signals CTA 1 via cluster barrier: "K_j, V_j loaded"
     (or use mbarrier with TMA for automatic signaling)

HBM Traffic Comparison:
  Non-cluster: Each CTA loads K_j, V_j independently
    Per cluster pair: 2 * (K_tile + V_tile) = 2 * 64 KB = 128 KB

  With cluster multicast: One TMA load of K_j, V_j, broadcast to both
    Per cluster pair: 1 * (K_tile + V_tile) = 1 * 64 KB = 64 KB

  Savings: 50% of KV loading bandwidth
```

### 3.2 Complete Kernel Skeleton

```cpp
template <int CLUSTER_SIZE, int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
cluster_flash_attention_fwd(
    const half* __restrict__ Q,    // [B, H, S, d]
    const half* __restrict__ K,    // [B, H, S, d]
    const half* __restrict__ V,    // [B, H, S, d]
    half* __restrict__ O,          // [B, H, S, d]
    float* __restrict__ L,         // [B, H, S] log-sum-exp
    const CUtensorMap* tma_Q,
    const CUtensorMap* tma_K,
    const CUtensorMap* tma_V,
    int seq_len)
{
    // Cluster identification
    const int rank = block_rank_in_cluster();
    const int bh_id = blockIdx.y;

    // Q tile assignment: each CTA in cluster gets a different Q tile
    // cluster processes CLUSTER_SIZE adjacent Q tiles
    const int cluster_q_base = blockIdx.x * CLUSTER_SIZE;
    const int my_q_tile = cluster_q_base + rank;

    // Shared memory layout
    extern __shared__ char smem_buf[];
    half* smem_Q = (half*)smem_buf;                          // [BLOCK_M, HEAD_DIM]
    half* smem_K = smem_Q + BLOCK_M * HEAD_DIM;              // [BLOCK_N, HEAD_DIM]
    half* smem_V = smem_K + BLOCK_N * HEAD_DIM;              // [BLOCK_N, HEAD_DIM]
    uint64_t* mbar_kv = (uint64_t*)(smem_V + BLOCK_N * HEAD_DIM);

    // Initialize mbarrier for KV loading
    if (threadIdx.x == 0) {
        mbarrier_init(mbar_kv, /* expected */ 1);  // TMA completion
    }
    __syncthreads();

    // Load Q tile (each CTA loads its own Q tile, no multicast needed)
    if (threadIdx.x == 0) {
        tma_load_2d(tma_Q, smem_Q, my_q_tile, bh_id);
    }

    // Initialize accumulators
    float O_acc[BLOCK_M * HEAD_DIM / WARP_SIZE] = {0};
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Main loop over KV tiles
    const int num_kv_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;

    for (int j = 0; j < num_kv_tiles; j++) {
        // === KV LOADING WITH TMA MULTICAST ===
        // Only rank 0 issues the TMA load; multicast delivers to all CTAs
        if (rank == 0 && threadIdx.x == 0) {
            uint16_t multicast_mask = (1 << CLUSTER_SIZE) - 1;  // all CTAs

            tma_load_multicast_2d(
                tma_K, smem_K, j, bh_id, mbar_kv, multicast_mask);
            tma_load_multicast_2d(
                tma_V, smem_V, j, bh_id, mbar_kv, multicast_mask);
        }

        // Wait for KV tile to arrive in SMEM (all CTAs wait)
        mbarrier_wait(mbar_kv, j % 2);

        // === COMPUTE (independent per CTA) ===

        // S = Q @ K^T  (WGMMA: SS variant)
        float S[BLOCK_M * BLOCK_N / WARP_SIZE];
        wgmma_ss(S, smem_Q, smem_K, BLOCK_M, BLOCK_N, HEAD_DIM);

        // Online softmax update
        float m_new = m_i;
        for (int idx = 0; idx < BLOCK_M * BLOCK_N / WARP_SIZE; idx++) {
            m_new = fmaxf(m_new, S[idx]);
        }
        // Warp-level reduction for max
        m_new = warp_reduce_max(m_new);

        float alpha = expf(m_i - m_new);
        l_i = alpha * l_i;

        float P[BLOCK_M * BLOCK_N / WARP_SIZE];
        float l_update = 0.0f;
        for (int idx = 0; idx < BLOCK_M * BLOCK_N / WARP_SIZE; idx++) {
            P[idx] = expf(S[idx] - m_new);
            l_update += P[idx];
        }
        l_update = warp_reduce_sum(l_update);
        l_i += l_update;
        m_i = m_new;

        // O = alpha * O + P @ V  (WGMMA: RS variant)
        scale_accumulator(O_acc, alpha);
        wgmma_rs(O_acc, P, smem_V, BLOCK_M, HEAD_DIM, BLOCK_N);

        // Release mbarrier for next iteration
        if (threadIdx.x == 0) {
            mbarrier_init(mbar_kv, 1);
        }
    }

    // Epilogue: normalize O by l_i, write to HBM
    for (int idx = 0; idx < BLOCK_M * HEAD_DIM / WARP_SIZE; idx++) {
        O_acc[idx] /= l_i;
    }
    store_output(O, O_acc, my_q_tile, bh_id);
    store_lse(L, m_i + logf(l_i), my_q_tile, bh_id);
}
```

### 3.3 Cluster Size Selection

```python
def choose_cluster_size(problem_config, gpu_config):
    """
    Choose cluster size for attention kernel.

    Factors:
    1. KV bandwidth savings (larger cluster = more savings)
    2. Scheduling overhead (larger cluster = harder to place)
    3. SM count per GPC (hardware limit)
    4. Whether kernel is memory-bound (clusters help more for memory-bound)
    """
    B, H, S_q, S_kv, d = problem_config
    num_SMs = gpu_config.num_SMs
    SMs_per_GPC = gpu_config.SMs_per_GPC  # ~16 for H100

    num_q_tiles = ceil(S_q / BLOCK_M)
    total_CTAs = num_q_tiles * B * H

    # Don't use clusters if already undersubscribed
    if total_CTAs < num_SMs:
        return 1  # not enough work to fill SMs

    # Estimate whether kernel is memory-bound
    # Arithmetic intensity: 2*S_kv*d FLOPS / (S_kv*d*2 bytes for K+V)
    AI = 2 * S_kv * d / (S_kv * d * 2)  # simplifies to ~1 for decode
    is_memory_bound = AI < 100  # rough threshold

    if not is_memory_bound:
        return 1  # compute-bound kernels don't benefit much

    # Memory-bound: use clusters for KV bandwidth savings
    # Sweet spot is 2 for most configurations
    cluster_size = 2

    # Can go to 4 if problem is very memory-bound and GPC has enough SMs
    if AI < 10 and SMs_per_GPC >= 8:
        cluster_size = 4

    # Ensure cluster_size divides num_q_tiles evenly
    while num_q_tiles % cluster_size != 0 and cluster_size > 1:
        cluster_size //= 2

    return cluster_size

# Examples:
# Prefill, S=4096, d=128: AI ~= 128, compute-bound -> cluster_size=1
# Decode, S=32K, d=128: AI ~= 1, memory-bound -> cluster_size=2
# Long-context prefill, S=128K, d=128: AI ~= 128, compute-bound -> cluster_size=1
```

---

## 4. Performance Analysis

### 4.1 Theoretical Bandwidth Savings

```
KV Loading Bandwidth per Attention Iteration:
  Per KV tile: BLOCK_N * d * sizeof(fp16) * 2 (K and V)
  = 128 * 128 * 2 * 2 = 64 KB per tile

  For S_kv = 32768, BLOCK_N = 128:
  Total KV tiles = 256
  Total KV bytes = 256 * 64 KB = 16 MB per (Q_tile, head, batch)

Without clusters (cluster_size=1):
  Each CTA independently loads all KV tiles
  HBM reads per cluster of 2 adjacent CTAs: 2 * 16 MB = 32 MB

With clusters (cluster_size=2):
  TMA multicast: 1 read per tile, broadcast to 2 CTAs
  HBM reads per cluster of 2 CTAs: 1 * 16 MB = 16 MB
  Savings: 16 MB (50%)

With clusters (cluster_size=4):
  HBM reads per cluster of 4 CTAs: 1 * 16 MB = 16 MB
  vs non-cluster: 4 * 16 MB = 64 MB
  Savings: 48 MB (75%)
```

### 4.2 Practical Performance Impact

The actual speedup depends on whether KV loading is on the critical path:

**Scenario 1: Memory-bound decode attention**
```
Decode: Q = [1, d], KV = [S, d], S >> d
Kernel time dominated by KV loading from HBM
Cluster multicast directly reduces the bottleneck
Expected speedup: close to cluster_size (2x for cluster=2)
Actual: 1.3-1.8x (overhead reduces theoretical maximum)
```

**Scenario 2: Compute-bound prefill attention**
```
Prefill: Q = [S, d], KV = [S, d], S is large
Kernel time dominated by QK^T and PV GEMMs
KV loading is hidden behind WGMMA computation
Cluster multicast saves bandwidth but doesn't reduce critical path
Expected speedup: 1-5% (bandwidth savings help with edge cases)
```

**Scenario 3: Mixed attention (moderate sequence lengths)**
```
S = 1024-4096, compute and memory are roughly balanced
Cluster savings help in the memory-bound portions
Expected speedup: 5-15%
```

### 4.3 CUTLASS GEMM Benchmark Data

From CUTLASS documentation, cluster-based GEMMs show:

| Matrix Size | Cluster=1 | Cluster=2 | Cluster=4 | Speedup |
|------------|-----------|-----------|-----------|---------|
| 4096x4096 | 810 TFLOPS | 845 TFLOPS | 860 TFLOPS | 1.06x |
| 8192x8192 | 830 TFLOPS | 870 TFLOPS | 890 TFLOPS | 1.07x |
| 16384x16384 | 840 TFLOPS | 880 TFLOPS | 910 TFLOPS | 1.08x |

These are compute-bound GEMMs where the benefit is modest. For memory-bound operations (like attention decode), the benefit is proportionally larger.

---

## 5. Cluster Usage in FlashAttention

### 5.1 FlashAttention-3 Cluster Support

The FA3 paper mentions cluster support:

> "Threadblock clusters and distributed shared memory for K/V copying"

The FA3 codebase includes cluster configurations:

```cpp
// From flash-attention FA3 kernel config (simplified)
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

// Kernel launch with cluster
auto kernel = flash_fwd_kernel<KernelTraits, ClusterShape>;
```

In practice, FA3's primary speedups come from warp specialization and GEMM-softmax pipelining (which provide 15-30% speedup), while clusters add an incremental 5-10% for memory-bound configurations.

### 5.2 CUTLASS Attention with Clusters

CUTLASS 3.x provides cluster support in its FMHA (Fused Multi-Head Attention) examples:

```cpp
// CUTLASS FMHA with cluster support
using CollectiveMainloop = typename cutlass::fmha::CollectiveMainloopFwd<
    cutlass::arch::Sm90,
    ElementQ, ElementK, ElementV,
    TileShape,
    ClusterShape,  // e.g., Shape<_2, _1, _1>
    StageCount,
    SmemLayoutQ, SmemLayoutK, SmemLayoutV
>;

// The CollectiveMainloop automatically:
// 1. Configures TMA multicast for KV loading
// 2. Sets up cluster barriers
// 3. Handles DSMEM access patterns
```

### 5.3 When Clusters Help Most in Attention

```
Benefit ranking for attention workloads:

1. GQA decode, small batch (highest benefit)
   B=1, H_kv=8, S=32K: only 8 CTAs without split-KV
   Clusters: multicast KV across Q-head CTAs that share KV heads
   Combined with split-KV: further improves bandwidth utilization

2. MQA decode (high benefit)
   B=1, H_kv=1, S=32K: 1 KV head, many Q heads
   All Q heads attend to same KV -> perfect multicast opportunity

3. Standard MHA decode (moderate benefit)
   B=1, H=32, S=32K: each head has its own KV
   Clusters help pairs of Q tiles within each head

4. Prefill, long sequences (low benefit)
   Already compute-bound, KV loading is hidden
   Clusters provide marginal benefit

5. Prefill, short sequences (negligible benefit)
   Not enough tiles per head for meaningful cluster grouping
```

---

## 6. DSMEM Applications Beyond KV Broadcasting

### 6.1 Partial Softmax Statistics Exchange

In some attention configurations, clusters can exchange softmax statistics:

```cpp
// After each CTA computes local softmax statistics
float local_max = compute_local_max(S);
float local_sum = compute_local_sum(S, local_max);

// Share statistics across cluster for joint normalization
// (useful for ring attention variants within a node)
__shared__ float stats[2];
stats[0] = local_max;
stats[1] = local_sum;

// Other cluster CTAs can read these stats via DSMEM
for (int r = 0; r < cluster_size; r++) {
    float* remote_stats = cluster.map_shared_rank(stats, r);
    float remote_max = remote_stats[0];
    float remote_sum = remote_stats[1];
    // ... combine into global statistics ...
}
```

### 6.2 Backward Pass dQ Accumulation

In the backward pass, dQ contributions from different KV tiles must be accumulated. With clusters, different CTAs processing different KV ranges can accumulate dQ into a shared DSMEM buffer without global memory atomics:

```cpp
// CTA 0 processes KV range [0, S/2), CTA 1 processes [S/2, S)
// Both contribute to the same dQ tile
// Use DSMEM to accumulate without global atomics

__shared__ float dQ_local[BLOCK_M * HEAD_DIM];

// Each CTA computes its partial dQ
compute_partial_dQ(dQ_local, ...);

// CTA 1 adds its contribution to CTA 0's SMEM
if (rank == 1) {
    float* remote_dQ = cluster.map_shared_rank(dQ_local, 0);
    for (int i = threadIdx.x; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        atomicAdd(&remote_dQ[i], dQ_local[i]);  // DSMEM atomic
    }
}
cluster.sync();

// CTA 0 stores the accumulated dQ
if (rank == 0) {
    store_dQ(dQ_global, dQ_local);
}
```

---

## 7. Practical Trade-offs and Limitations

### 7.1 Scheduling Overhead

Clusters impose scheduling constraints that can reduce overall SM utilization:

```
Without clusters:
  Scheduler places CTAs on any available SM
  108 SMs (A100) or 132 SMs (H100) all independently schedulable
  Grid of 100 CTAs: all placed on first 100 available SMs

With cluster_size=2:
  Scheduler must find PAIRS of SMs in the same GPC
  If 1 SM per GPC is busy and 1 is free, neither can start a new cluster
  Fragmentation reduces effective SM availability

With cluster_size=4:
  Need 4 contiguous SMs in same GPC
  Fragmentation is worse
  Can significantly reduce throughput for non-square grids
```

**Guideline**: cluster_size=2 has minimal scheduling overhead. cluster_size=4 or higher should be benchmarked carefully.

### 7.2 Grid Size Constraints

```
The grid's X dimension must be divisible by cluster_size.X
The grid's Y dimension must be divisible by cluster_size.Y

For attention: grid = (num_q_tiles, B * H)
  num_q_tiles must be divisible by cluster_size

If S_q = 512 and BLOCK_M = 128: num_q_tiles = 4
  cluster_size=2: OK (4 / 2 = 2 clusters)
  cluster_size=4: OK (4 / 4 = 1 cluster)
  cluster_size=8: FAIL (4 / 8 not integer)

If S_q = 384 and BLOCK_M = 128: num_q_tiles = 3
  cluster_size=2: FAIL (3 / 2 not integer)
  Need to pad or fall back to cluster_size=1
```

### 7.3 Shared Memory Pressure

Each CTA in a cluster still has its own SMEM allocation. DSMEM provides cross-CTA access but does not pool SMEM:

```
Cluster of 2 CTAs:
  CTA 0: 228 KB SMEM (local)
  CTA 1: 228 KB SMEM (local)
  Total accessible by any CTA: 2 * 228 KB = 456 KB (via DSMEM)
  But CTA 0 can only write to its own 228 KB

  KV tile in CTA 0's SMEM: accessible by CTA 1 via DSMEM
  KV tile in CTA 1's SMEM: accessible by CTA 0 via DSMEM
  TMA multicast: both CTAs have the same KV tile in their own SMEM
    -> redundant storage but simpler programming model
```

### 7.4 Interaction with Warp Specialization

Clusters add complexity on top of warp specialization:

```
FA3-style kernel with clusters:
  Cluster: [CTA 0] [CTA 1]

  CTA 0:
    Producer warpgroup: issues TMA loads (with multicast to CTA 1)
    Consumer warpgroup: WGMMA + softmax

  CTA 1:
    Producer warpgroup: NO TMA loads needed (receives via multicast)
    Consumer warpgroup: WGMMA + softmax

  Key insight: CTA 1's producer warpgroup is freed up since KV loading
  is handled by CTA 0's TMA multicast. This warpgroup can be used for
  other work (e.g., prefetching Q tiles or computing auxiliary outputs).
```

---

## 8. Cluster Feature Availability

| Feature | Hopper (SM90) | Blackwell (SM100) |
|---------|---------------|-------------------|
| Cluster launch | Yes | Yes |
| DSMEM | Yes | Yes |
| TMA multicast | Yes | Yes |
| Cluster barriers | Yes | Yes |
| Max cluster size | ~16 | ~16 |
| TMEM + clusters | N/A (no TMEM) | Yes (orthogonal features) |

On Blackwell, clusters compose with TMEM: each CTA in a cluster has its own TMEM allocation, and TMA multicast handles KV loading while TMEM handles accumulator storage.

---

## 9. Summary: When to Use Clusters for Attention

```
Decision framework:

1. Is the attention kernel memory-bound?
   No  -> cluster_size=1 (no significant benefit)
   Yes -> continue to step 2

2. Is there KV sharing opportunity across CTAs?
   (Multiple Q tiles attending to same KV sequence)
   No  -> cluster_size=1
   Yes -> continue to step 3

3. Is the grid large enough for cluster scheduling?
   num_q_tiles < 2 -> cluster_size=1 (cannot pair)
   num_q_tiles >= 2 and even -> cluster_size=2
   num_q_tiles >= 4 and divisible by 4 -> try cluster_size=4

4. Profile both configurations:
   Compare: cluster=1 vs cluster=2 vs cluster=4
   Choose the one with best throughput
   (do not blindly assume larger cluster is better)
```

---

## References

- [NVIDIA CUDA Programming Guide: Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [CUTLASS 3.x: Efficient GEMM with Clusters](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
- [FlashAttention-3 Paper (Shah et al., 2024)](https://tridao.me/publications/flash3/flash3.pdf)
- [NVIDIA GTC 2023: CUTLASS Hopper Tutorial](https://www.nvidia.com/en-us/on-demand/)
- [NVIDIA PTX ISA: Cluster Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [TMA Deep-Dive (knowledge base entry)](../tma_deep_dive/skill.md)
- [FA3 Warp Specialization (knowledge base entry)](../../L2_Scheduling_Pipelining_Level/fa3_warp_specialization/skill.md)
