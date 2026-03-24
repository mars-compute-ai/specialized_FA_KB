# MI300 Compute and Memory Partitioning — Detailed Reference

## Overview

AMD Instinct MI300 GPUs expose configurable compute and memory partition modes that control how the 8 XCDs and 8 HBM stacks are presented to the programmer. This document covers the architecture, all partition combinations, deployment patterns, and performance benchmarks relevant to Flash Attention workloads.

## 1. Architecture Foundation

### MI300X Chiplet Layout

The MI300X contains:
- **8 XCDs (Accelerator Complex Dies)**: Each with 38 active CUs and 4 MB L2 cache
- **4 IODs (I/O Dies)**: Each connecting 2 XCDs to 2 HBM3 stacks and 64 MB Infinity Cache
- **8 HBM3 stacks**: 24 GB each, 192 GB total, 5.3 TB/s aggregate bandwidth

XCD pairs are 3D-stacked on IODs. XCDs on the same IOD share memory controllers and Infinity Cache. Cross-IOD communication traverses Infinity Fabric.

### Natural Boundaries

The chiplet design creates natural performance boundaries:
- **Intra-XCD**: L2 cache hit (4 MB, lowest latency)
- **Intra-IOD**: Infinity Cache hit (64 MB, medium latency)
- **Cross-IOD**: Infinity Fabric traversal (highest latency)

Partition modes let you align workload scheduling to these boundaries.

## 2. Compute Partition Modes

### SPX — Single Partition X-celerator (Default)

```
┌────────────────────────────────────────────┐
│               1 Logical GPU                │
│  XCD0  XCD1  XCD2  XCD3  XCD4  ...  XCD7  │
│  ← Workgroups distributed round-robin →    │
└────────────────────────────────────────────┘
```

- All 8 XCDs as one device. `hipGetDeviceCount()` returns 1 per MI300X.
- Workgroup placement is automatic (round-robin). No programmer control.
- Best for large kernels that need all 304 CUs.

### CPX — Core Partitioned X-celerator

```
┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐
│GPU 0 │ │GPU 1 │ │GPU 2 │ ... │GPU 7 │
│XCD 0 │ │XCD 1 │ │XCD 2 │     │XCD 7 │
│38 CUs│ │38 CUs│ │38 CUs│     │38 CUs│
└──────┘ └──────┘ └──────┘     └──────┘
```

- Each XCD is a separate logical GPU. `hipGetDeviceCount()` returns 8 per MI300X.
- Explicit scheduling via `hipSetDevice()`.
- Best for multi-tenant inference or decode workloads with small batch sizes.

### DPX — Dual Partition X-celerator

- 4 XCDs per logical GPU → 2 logical GPUs per MI300X
- Intermediate granularity between SPX and CPX

### QPX — Quad Partition X-celerator

- 2 XCDs per logical GPU → 4 logical GPUs per MI300X
- Useful when CPX is too fine-grained but SPX too coarse

## 3. Memory Partition Modes

### NPS1 — Single NUMA Domain

- All 8 HBM stacks appear as one unified memory pool (192 GB on MI300X)
- Any XCD can access any memory location
- Uniform memory access latency (approximately)
- Compatible with: SPX, DPX, QPX, CPX

### NPS4 — Four NUMA Domains

- Memory divided into 4 partitions (48 GB each on MI300X)
- Each partition mapped to one IOD (2 HBM stacks)
- Local memory access is faster; remote access requires Infinity Fabric traversal
- Compatible with: CPX only

### NPS2 — Two NUMA Domains (CDNA4)

- Memory divided into 2 partitions (one per IOD)
- CDNA4 addition for DPX mode
- 2.25× capacity and 2.67× bandwidth improvement per partition vs NPS4

## 4. Performance Benchmarks

### Stream Bandwidth (Triton copy_kernel, MI300X)

| Mode | Bandwidth | vs SPX/NPS1 |
|:-----|:----------|:------------|
| SPX/NPS1 | ~4017 GB/s | baseline |
| CPX/NPS1 | ~4010 GB/s | ~0% |
| CPX/NPS4 | ~4210 GB/s | **+4.8%** |

**Per-XCD bandwidth in NPS4**:
- Single XCD active: ~1 TB/s (captures full IOD bandwidth)
- Both XCDs on same IOD active: ~500 GB/s each (shared)
- NPS1 provides consistent bandwidth per XCD regardless of active count

### GEMM Throughput (16384×16384×4096 FP16, MI300X)

| Mode | Total System Throughput | vs SPX |
|:-----|:-----------------------|:-------|
| SPX/NPS1 | baseline | — |
| CPX/NPS1 | +10-12% | higher |
| CPX/NPS4 | **+12-15%** | highest |

**Key finding**: CPX/NPS4 maintains higher compute clock speeds under load than SPX/NPS1, contributing to the throughput advantage.

## 5. Deployment Patterns

### Pattern 1: Multi-Instance Inference (CPX/NPS4)

Run one inference instance per XCD for maximum per-request throughput:

```bash
# Set partition modes
amd-smi set --gpu all --compute-partition CPX
amd-smi set --gpu all --memory-partition NPS4

# Launch 8 independent inference servers
for i in $(seq 0 7); do
    HIP_VISIBLE_DEVICES=$i python serve.py --port $((8000+$i)) &
done
```

### Pattern 2: Tensor-Parallel Training (SPX/NPS1)

Use all CUs for a single large training job:

```bash
# Default mode — or explicitly set
amd-smi set --gpu all --compute-partition SPX
amd-smi set --gpu all --memory-partition NPS1

# Launch distributed training across physical GPUs
torchrun --nproc_per_node=8 train.py
```

### Pattern 3: Docker-Based XCD Isolation

```bash
# In CPX mode, each XCD is a separate render device
# Render IDs: renderD128 to renderD191 (8 per physical GPU)

# Assign XCD 0 to a container
docker run --device /dev/dri/renderD128 --device /dev/kfd my_image

# Assign all 8 XCDs of GPU 0
docker run --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
           --device /dev/dri/renderD130 --device /dev/dri/renderD131 \
           --device /dev/dri/renderD132 --device /dev/dri/renderD133 \
           --device /dev/dri/renderD134 --device /dev/dri/renderD135 \
           --device /dev/kfd my_image
```

### Pattern 4: MPI Job Scheduling

```bash
# Assign disjoint XCD sets to MPI ranks
mpirun \
  -np 1 -x ROCR_VISIBLE_DEVICES=0,8,16,32 ./inference : \
  -np 1 -x ROCR_VISIBLE_DEVICES=1,9,17,33 ./inference
```

## 6. HIP Programming with Partitions

### Multi-Partition Vector Add

```cpp
#include <hip/hip_runtime.h>

int main() {
    int num_devices;
    hipGetDeviceCount(&num_devices);  // 8 in CPX mode

    for (int dev = 0; dev < num_devices; dev++) {
        hipSetDevice(dev);

        float *d_a, *d_b, *d_c;
        hipMalloc(&d_a, N * sizeof(float));
        hipMalloc(&d_b, N * sizeof(float));
        hipMalloc(&d_c, N * sizeof(float));

        hipStream_t stream;
        hipStreamCreate(&stream);

        hipMemcpyAsync(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice, stream);

        vector_add<<<grid, block, 0, stream>>>(d_a, d_b, d_c, N);

        hipStreamSynchronize(stream);
    }
    return 0;
}
```

### PyTorch Multi-Partition

```python
import torch
import torch.multiprocessing as mp

def run_on_xcd(device_id, data):
    torch.cuda.set_device(device_id)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = torch.randn(N, N, device=f'cuda:{device_id}')
        b = torch.randn(N, N, device=f'cuda:{device_id}')
        c = torch.matmul(a, b)
    stream.synchronize()
    return c

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Required for HIP
    num_devices = torch.cuda.device_count()  # 8 in CPX
    processes = []
    for i in range(num_devices):
        p = mp.Process(target=run_on_xcd, args=(i, data))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
```

## 7. Implications for Flash Attention

### Decode Workloads (CPX/NPS4 Recommended)

- Small batch decode (seqlen_q=1) underutilizes all 304 CUs in SPX mode
- CPX/NPS4: Each XCD runs its own decode instance with localized HBM access
- Per-XCD bandwidth: ~1 TB/s in NPS4 (full IOD bandwidth when single XCD active)
- Combined throughput across 8 XCDs exceeds SPX by 10-15%

### Prefill Workloads (SPX/NPS1 or DPX/NPS1)

- Large sequence prefill benefits from all CUs working on one kernel
- SPX mode avoids the complexity of multi-GPU programming
- If sequence length is moderate, DPX or QPX may offer a sweet spot

### Mixed Workloads

- Use CPX/NPS4 during inference serving (many small decode requests)
- Switch to SPX/NPS1 for batch processing or training
- Mode switching via `amd-smi` takes a few seconds and resets GPU state

## References

- [AMD Instinct MI300 Compute and Memory Partition Modes — ROCm Blog (2025-02-09)](https://rocm.blogs.amd.com/artificial-intelligence/mi300-compute-memory-partition-modes/README.html)
- [AMD Instinct MI300X Documentation](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [ROCm AMD SMI Documentation](https://rocm.docs.amd.com/projects/amdsmi/)
- [AMD CDNA3 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
