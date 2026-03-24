# AMD CDNA3/CDNA4 Architecture Guide — Detailed Reference

## Overview

This document provides detailed architecture specifications for AMD CDNA3 (MI300X/MI325X) and CDNA4 (MI355X) GPUs, focusing on the hardware features that directly impact Flash Attention kernel design: chiplet topology, compute unit microarchitecture, memory hierarchy, and key generational improvements.

## 1. Chiplet Architecture

### CDNA3: MI300X / MI325X

```
┌──────────────────────────────────────────────────────────────┐
│                      MI300X Package                          │
│                                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │  XCD 0  │ │  XCD 1  │ │  XCD 2  │ │  XCD 3  │  ← 5nm  │
│  │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │          │
│  │ 4MB L2  │ │ 4MB L2  │ │ 4MB L2  │ │ 4MB L2  │          │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘          │
│       │3D stack    │           │           │                 │
│  ┌────┴────────────┴──┐  ┌────┴───────────┴──┐             │
│  │      IOD 0         │  │      IOD 1         │  ← 6nm     │
│  │  64MB Inf. Cache   │  │  64MB Inf. Cache   │             │
│  │  2× HBM3 stacks   │  │  2× HBM3 stacks   │             │
│  └────────────────────┘  └────────────────────┘             │
│                                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │  XCD 4  │ │  XCD 5  │ │  XCD 6  │ │  XCD 7  │          │
│  │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │ │ 38 CUs  │          │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘          │
│  ┌────┴────────────┴──┐  ┌────┴───────────┴──┐             │
│  │      IOD 2         │  │      IOD 3         │             │
│  │  64MB Inf. Cache   │  │  64MB Inf. Cache   │             │
│  │  2× HBM3 stacks   │  │  2× HBM3 stacks   │             │
│  └────────────────────┘  └────────────────────┘             │
└──────────────────────────────────────────────────────────────┘
Total: 304 CUs, 32MB L2, 256MB LLC, 192GB HBM3 @ 5.3 TB/s
```

**MI325X variant**: Same chiplet layout, upgraded to HBM3E. 256 GB @ 6.0 TB/s.

### CDNA4: MI355X

```
┌──────────────────────────────────────────────────┐
│                 MI355X Package                    │
│                                                   │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
│  │ XCD 0  │ │ XCD 1  │ │ XCD 2  │ │ XCD 3  │   │
│  │ 32 CUs │ │ 32 CUs │ │ 32 CUs │ │ 32 CUs │   │
│  │ 4MB L2 │ │ 4MB L2 │ │ 4MB L2 │ │ 4MB L2 │   │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘   │
│  ┌───┴──────────┴─────┐┌───┴──────────┴─────┐   │
│  │       IOD 0        ││       IOD 1        │   │
│  │  128MB Inf. Cache  ││  128MB Inf. Cache  │   │
│  │  4× HBM3E stacks  ││  4× HBM3E stacks  │   │
│  └────────────────────┘└────────────────────┘   │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
│  │ XCD 4  │ │ XCD 5  │ │ XCD 6  │ │ XCD 7  │   │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘   │
│      └──────────┴──────────┴──────────┘         │
│              (connected to IODs)                 │
└──────────────────────────────────────────────────┘
Total: 256 CUs, 32MB L2, 256MB LLC, 288GB HBM3E @ 8.0 TB/s
```

**Key difference**: Fewer CUs per XCD (32 vs 38) but each CU is significantly more powerful. 2 IODs instead of 4.

## 2. Compute Unit Microarchitecture

### CDNA3 Compute Unit

Each CU contains:
- **4 × 16-lane SIMD units**: Each SIMD executes one wavefront (64 threads) over 4 cycles
- **10 waveslots per SIMD**: Up to 10 concurrent wavefronts for latency hiding
- **Matrix cores**: Dedicated MFMA execution units
- **Scalar ALU**: Shared across the CU for control flow and address computation
- **64 KB instruction cache**: Shared between 2 CUs (8-way set-associative, doubled from CDNA2)
- **64 KB LDS**: Workgroup-shared memory (32 banks × 4 bytes)
- **32 KB L1 data cache**: Per-CU (128-byte cache lines, doubled line size from CDNA2)
- **512 registers × 64 threads per SIMD**: Shared between VGPRs and AGPRs

### CDNA3 Matrix Core Throughput (per CU per clock)

| Datatype | FLOPS/clock/CU | vs CDNA2 |
|:---------|:---------------|:---------|
| FP64     | 256            | 1×       |
| FP32     | 256            | 1×       |
| TF32     | 1024           | new      |
| FP16/BF16| 2048           | 3.4×     |
| FP8      | 4096           | new      |
| INT8     | 4096           | 6.8×     |

Sparse support: 4:2 structured sparsity doubles effective throughput.

### CDNA4 Compute Unit Changes

| Feature | CDNA3 | CDNA4 | Change |
|:--------|:------|:------|:-------|
| LDS capacity | 64 KB | 160 KB | 2.5× |
| LDS banks | 32 | 64 | 2× |
| LDS read BW | 128 B/clock | 256 B/clock | 2× |
| FP16/BF16 FLOPS | 2048 | 4096 | 2× |
| FP8 FLOPS | 4096 | 8192 | 2× |
| FP4/FP6 FLOPS | — | 16384 | new |
| Transcendental rate | 1× | 2× | 2× |
| Direct L1 Load | No | Yes | new |
| TF32 | Hardware | Software (via BF16) | removed |

**Direct L1 Load** (CDNA4): Enables loading data from LDS directly to L1 cache, bypassing register staging. Reduces register pressure for matrix operand delivery.

**2× Transcendental rate** (CDNA4): Directly benefits softmax exp/log computation — same motivation as FA4's polynomial exp approximation on NVIDIA Blackwell.

## 3. Cache and Memory Hierarchy

### Complete Hierarchy (CDNA3)

```
Registers (VGPR/AGPR)
  │ 0 cycles, ~19.5 TB/s per CU
  ▼
L1 Data Cache (32 KB per CU)
  │ ~10 cycles, 128B lines
  ▼
L2 Cache (4 MB per XCD, 16 channels × 256 KB)
  │ ~100 cycles, 34.4 TB/s aggregate (8 XCDs)
  ▼ (Infinity Fabric)
Infinity Cache / LLC (256 MB total, 16-way)
  │ ~200 cycles, 17.2 TB/s
  ▼
HBM3 (192 GB MI300X / 256 GB MI325X)
    ~400 cycles, 5.3 TB/s (MI300X) / 6.0 TB/s (MI325X)
```

### L2 Cache Details

- 4 MB per XCD, 32 MB total across 8 XCDs
- 16 parallel channels × 256 KB each
- Writeback, write-allocate, coherent within XCD
- Snoop filter covers multiple XCD L2 caches (cross-XCD coherence)
- **Aggregate read bandwidth: 34.4 TB/s** across all XCDs

### Infinity Cache (LLC)

- 256 MB total distributed across IODs (64 MB per IOD on CDNA3, 128 MB per IOD on CDNA4)
- Memory-side cache — does **not** participate in coherency protocol
- 16-way set-associative
- Peak bandwidth: ~17.2 TB/s
- Acts as an effective bandwidth amplifier for data that fits

### HBM Specifications

| GPU | Memory | Capacity | Bandwidth | Stacks | Interface Speed |
|:----|:-------|:---------|:----------|:-------|:----------------|
| MI300X | HBM3 | 192 GB | 5.3 TB/s | 8 | — |
| MI325X | HBM3E | 256 GB | 6.0 TB/s | 8 | 6 Gbps |
| MI355X | HBM3E | 288 GB | 8.0 TB/s | 8 | 8 Gbps |

### Comparison with NVIDIA

| Feature | MI300X | H100 SXM | B200 |
|:--------|:-------|:---------|:-----|
| HBM BW | 5.3 TB/s | 3.35 TB/s | 8.0 TB/s |
| L2 Cache | 32 MB | 50 MB | 50 MB |
| SMEM/LDS | 64 KB/CU | 228 KB/SM | 228 KB/SM |
| TMA | No | Yes | Yes |
| Matrix Unit | MFMA (wave-64) | WGMMA (warpgroup-128) | UMMA (warpgroup) |

**Key implication**: AMD's higher HBM bandwidth (1.6× over H100) means attention kernels are more likely to be compute-bound on AMD at equivalent precision. However, NVIDIA's TMA provides hardware-accelerated data movement that AMD must handle in software.

## 4. Implications for Flash Attention

### Tile Size Selection

| GPU | LDS | Recommended block_M × block_N | Rationale |
|:----|:----|:------------------------------|:----------|
| MI300X | 64 KB | 128×64 or 64×128 | Limited by LDS for Q+K+V tiles |
| MI355X | 160 KB | 128×128 or 128×192 | 2.5× LDS enables larger tiles |
| H100 | 228 KB | 128×128 or 192×128 | Larger SMEM + TMA |

### Pipeline Depth

- **CDNA3**: `num_stages=1` typically optimal for fused dual-GEMM attention. The smaller LDS doesn't support double-buffering large tiles.
- **CDNA4**: `num_stages=2` becomes viable with 160 KB LDS. Double-buffering K/V tiles while computing on current tiles.

### Data Movement Strategy

Without TMA, AMD kernels must:
1. Use explicit `global_load_dwordx4` for data movement (16 bytes per thread)
2. Stage data through LDS for coalesced access patterns
3. Use `ds_read_b128` for maximum LDS → register bandwidth
4. Rely on software prefetching via async buffer-to-LDS transfers

### XCD-Aware Scheduling

- Workgroups on the same XCD share L2 cache — schedule attention blocks that share K/V data to the same XCD when possible
- In SPX mode, round-robin distribution may scatter related workgroups across XCDs
- CPX mode gives explicit control but requires multi-GPU programming model

## References

- [AMD CDNA3 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-white-paper.pdf)
- [AMD Instinct MI300X Datasheet](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [AMD Instinct MI355X Datasheet](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)
