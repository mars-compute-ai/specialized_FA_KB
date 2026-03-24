---
skill_name: MI300 Compute and Memory Partitioning
description: AMD Instinct MI300 compute partition modes (SPX/CPX/DPX/QPX) and memory partition modes (NPS1/NPS4) that control XCD scheduling, NUMA topology, and memory bandwidth allocation for Flash Attention deployment.
level: L3 - Memory and Thread Cooperation Level
target_hardware: AMD MI300X (CDNA3), MI325X (CDNA3), MI355X (CDNA4)
relevance: When deploying Flash Attention inference or training on MI300 systems and choosing partition modes for maximum bandwidth, latency, or multi-tenant isolation.
---

# MI300 Compute and Memory Partitioning

## What It Is

AMD Instinct MI300 GPUs expose their chiplet architecture through configurable compute and memory partition modes. The compute partitioning controls how the 8 XCDs (Accelerator Complex Dies) appear to the programmer — as a single monolithic device (SPX) or as multiple independent logical GPUs (CPX). The memory partitioning controls NUMA topology — whether all 8 HBM stacks appear as one unified pool (NPS1) or as 4 localized partitions (NPS4). The choice directly impacts memory bandwidth, latency, and workgroup scheduling behavior for attention kernels.

## Key Concepts

### Compute Partition Modes
- **SPX (Single Partition)**: All 8 XCDs as one logical GPU. Workgroups distributed round-robin across XCDs. Default mode. No programmer control over XCD placement.
- **CPX (Core Partitioned)**: Each XCD appears as a separate logical GPU (8 GPUs per MI300X in `amd-smi`). Explicit scheduling control per XCD.
- **DPX (Dual Partition)**: 4 XCDs per logical GPU (2 logical GPUs per MI300X).
- **QPX (Quad Partition)**: 2 XCDs per logical GPU (4 logical GPUs per MI300X).

### Memory Partition Modes
- **NPS1**: All HBM stacks as one unified pool. Compatible with SPX and CPX.
- **NPS4**: 4 separate NUMA domains, each with 2 HBM stacks. Compatible with CPX only. Higher local bandwidth but memory is partitioned.

### Compatibility Matrix

| Compute \ Memory | NPS1 | NPS4 |
|:-----------------|:----:|:----:|
| SPX              | ✓    |      |
| DPX              | ✓    |      |
| QPX              | ✓    |      |
| CPX              | ✓    | ✓    |

### Performance Impact
- **CPX/NPS4** achieves ~4210 GB/s bandwidth (highest) due to memory localization
- **SPX/NPS1** achieves ~4017 GB/s (baseline)
- **CPX/NPS4** maintains higher compute clock speeds under load
- For GEMM workloads, CPX modes achieve 10–15% higher total throughput than SPX

## When to Use

- **CPX/NPS4**: Maximum bandwidth per workgroup. Best for memory-bound attention kernels with small batch sizes or decode workloads where each XCD runs independent inference.
- **SPX/NPS1**: Simplest programming model. Best for large batch prefill where a single kernel needs all CUs. Default starting point.
- **CPX/NPS1**: Multi-tenant deployment where each XCD serves a different request but memory needs to be shared. Also useful for multi-process inference (vLLM with tensor parallelism mapped to XCDs).
- **DPX/QPX**: Intermediate granularity when 8 partitions is too fine but 1 is too coarse.

## When NOT to Use

- On non-MI300 hardware (these modes are MI300-specific)
- When the application already saturates all XCDs uniformly — partitioning adds complexity without benefit
- When memory requirements exceed a single partition's capacity in NPS4 mode (48 GB per partition on MI300X)

## Code / Pseudo-code

### Setting Partition Modes

```bash
# Set compute partition to CPX (8 logical GPUs)
amd-smi set --gpu all --compute-partition CPX

# Set memory partition to NPS4 (4 NUMA domains)
amd-smi set --gpu all --memory-partition NPS4

# Check current partition mode
amd-smi static --partition

# Reset to defaults
amd-smi reset --compute-partition
amd-smi reset --memory-partition
```

### HIP Multi-Partition Kernel Launch

```cpp
int num_devices;
hipGetDeviceCount(&num_devices);  // Returns 8 in CPX mode per MI300X

for (int i = 0; i < num_devices; i++) {
    hipSetDevice(i);
    hipStream_t stream;
    hipStreamCreate(&stream);
    // Launch attention kernel on this XCD
    flash_attention_kernel<<<grid, block, 0, stream>>>(q, k, v, o);
}
```

### GPU Isolation for Inference

```bash
# Expose only XCDs 0-3 to this process
export HIP_VISIBLE_DEVICES=0,1,2,3

# Or use ROCR_VISIBLE_DEVICES for ROCm-level isolation
export ROCR_VISIBLE_DEVICES=0,1,2,3
```

## Key Takeaways

- CPX/NPS4 delivers 5–15% higher bandwidth and throughput than SPX/NPS1 by localizing memory access to nearby HBM stacks
- SPX mode distributes workgroups round-robin across XCDs with no programmer control — fine for large kernels but suboptimal for small decode batches
- In CPX mode, each XCD appears as a separate GPU in HIP — use `hipGetDeviceCount()` and `hipSetDevice()` for explicit placement
- NPS4 requires CPX mode — you cannot use NPS4 with SPX
- For Flash Attention serving: CPX/NPS4 with one inference instance per XCD often outperforms SPX with a single instance using all XCDs

## References

- [AMD Instinct MI300 Compute and Memory Partition Modes — ROCm Blog (2025-02-09)](https://rocm.blogs.amd.com/artificial-intelligence/mi300-compute-memory-partition-modes/README.html)
- [AMD Instinct MI300X Documentation](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [ROCm AMD SMI Documentation](https://rocm.docs.amd.com/projects/amdsmi/)
