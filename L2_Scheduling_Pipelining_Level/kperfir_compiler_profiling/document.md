# KPerfIR: Compiler-Centric GPU Kernel Profiling — Detailed Reference

## Overview

KPerfIR (OSDI 2025, Guan et al.) is a profiling infrastructure integrated into the Triton compiler's MLIR-based pipeline. Unlike external profilers (Nsight Compute, rocprof) that operate on compiled binaries, KPerfIR instruments the intermediate representation at multiple compiler levels, maintaining semantic context from high-level tensor operations down to individual instructions. This enables profiling capabilities that are impossible with external tools: correlating pipeline bubbles with specific software pipeline stages, identifying which loop iteration is slow, and feeding profiling data back into compiler optimization passes.

## 1. Motivation: The Profiler-Compiler Gap

### Limitations of External Profilers

External profilers (Nsight Compute, Omniperf) provide:
- Hardware counter readings (cache hit rates, DRAM bandwidth, instruction counts)
- Kernel-level timing (whole-kernel execution time)
- Source-line mapping (approximate, often lost through compiler optimization)

But they **cannot** provide:
- Timing of individual software pipeline stages (load, compute, softmax)
- Correlation between loop iteration index and execution time
- Understanding of which warp role is the critical path in warp-specialized kernels
- Feedback to the compiler to improve scheduling decisions

### KPerfIR's Approach

Insert profiling markers directly into the compiler IR:
1. High-level markers at TTIR level (Triton's top-level IR) — understand tensor operations
2. GPU-specific markers at TTGIR level — understand pipeline stages and barriers
3. Low-level instrumentation at LLVM level — minimal overhead timing

The markers travel through the compilation pipeline, maintaining semantic context that would be lost if profiling only the final binary.

## 2. Architecture

### Multi-Level IR Stack

```
User Code (Python/Triton)
         │
         ▼
    ┌─────────┐
    │  TTIR   │  ← RecordOp (high-level profiling markers)
    │ (Triton │     Understands: loops, tensor ops, algorithmic phases
    │   IR)   │
    └────┬────┘
         │ Lowering
         ▼
    ┌─────────┐
    │ TTGIR   │  ← KPerfGPUIR (GPU-specific profiling ops)
    │ (GPU    │     Understands: pipeline stages, barriers, shared memory
    │  IR)    │
    └────┬────┘
         │ Lowering
         ▼
    ┌─────────┐
    │  LLVM   │  ← startInstrumentationOp / stopInstrumentationOp
    │   IR    │     Minimal overhead, cycle-accurate timing
    └────┬────┘
         │ Code generation
         ▼
    PTX / AMDGPU Assembly
```

### Core Operations

| Operation | Level | Purpose |
|:----------|:------|:--------|
| `RecordOp` | TTIR | High-level profiling marker (marks a region to profile) |
| `ReadCounterOp` | TTGIR | Read GPU cycle counter |
| `StoreCounterOp` | TTGIR | Store timing data to shared memory buffer |
| `InitOp` | TTGIR | Initialize profiling buffer |
| `FinalizeOp` | TTGIR | Flush profiling data to global memory |
| `startInstrumentationOp` | LLVM | Begin low-level timing region |
| `stopInstrumentationOp` | LLVM | End low-level timing region |

### Shared Memory Circular Buffer

Profiling data is stored in a **circular buffer in shared memory**:
- Efficient: no global memory traffic during profiling
- Bounded: fixed SMEM allocation regardless of profiling duration
- Overflow handling: circular wrap-around with sequence numbers
- Post-processing: trace replay corrects for overhead artifacts

```
Shared Memory Layout:
┌──────────────────────────────────────────┐
│ Profiling Circular Buffer (configurable) │
│ [record_0] [record_1] ... [record_N-1]  │
│ Each record: {timestamp, region_id,      │
│               iteration, metadata}       │
├──────────────────────────────────────────┤
│ Kernel Working Memory (remaining SMEM)   │
└──────────────────────────────────────────┘
```

## 3. Profiling Tools Built on KPerfIR

### Region-Based Timing Tool

Profiles intra-kernel execution at region granularity:

```python
# Triton kernel with KPerfIR region markers
@triton.jit
def flash_attention_kernel(...):
    # Region: Load Q tile
    with kperfir.region("load_q"):
        q_tile = tl.load(Q_ptr + offsets)

    for i in range(num_kv_blocks):
        # Region: Load K/V tiles
        with kperfir.region("load_kv"):
            k_tile = tl.load(K_ptr + kv_offsets)
            v_tile = tl.load(V_ptr + kv_offsets)

        # Region: Compute S = Q @ K^T
        with kperfir.region("gemm_qk"):
            s_tile = tl.dot(q_tile, tl.trans(k_tile))

        # Region: Softmax
        with kperfir.region("softmax"):
            p_tile = softmax(s_tile)

        # Region: Compute O += P @ V
        with kperfir.region("gemm_pv"):
            o_acc = tl.dot(p_tile, v_tile, o_acc)
```

**Output**: Per-region timing breakdown showing where time is spent in each pipeline stage.

### Multi-Record Technique for Async Instructions

For asynchronous instructions (TMA loads, WGMMA), a single start/stop timestamp is insufficient:

```
Timeline:
  ─────────────────────────────────────────────→ time
  TMA_start ──── (async) ──── TMA_complete
       MMA_start ── (async) ── MMA_complete

Problem: TMA_complete may overlap with MMA_start
Solution: Multiple records per region to capture
          issue time AND completion time separately
```

KPerfIR uses the multi-record technique:
1. Record timestamp at instruction issue
2. Insert a completion barrier
3. Record timestamp at barrier completion
4. Post-process to determine actual execution time

### Iteration-Based Timing

Correlates loop iteration index with execution time:

```
Output example (FA3 main loop):
Iteration 0:  load_kv=120cy  gemm_qk=80cy  softmax=45cy  gemm_pv=75cy  (cold start)
Iteration 1:  load_kv=85cy   gemm_qk=80cy  softmax=45cy  gemm_pv=75cy  (steady state)
...
Iteration N:  load_kv=85cy   gemm_qk=80cy  softmax=45cy  gemm_pv=75cy  (steady state)
Iteration N+1: load_kv=85cy  gemm_qk=60cy  softmax=30cy  gemm_pv=75cy  (causal mask, fewer elements)
```

Reveals:
- First iteration cold-start overhead (cache misses, TLB misses)
- Steady-state performance
- Last iteration behavior with causal masking

### Critical Path Analysis

For warp-specialized kernels (FA3/FA4 style with producer/consumer warps):

```
Warp Role Timeline:
  Producer (Load):  [load_K]---[load_V]---[load_K]---[load_V]---
  Consumer (MMA):   --[wait]--[GEMM_QK]---[GEMM_PV]--[GEMM_QK]---
  Softmax:          -----------[wait]---[softmax]---[wait]---

Critical path: Consumer (MMA) — it's always waiting for data
Bottleneck:    Producer (Load) — it can't supply data fast enough
```

KPerfIR identifies:
- Which warp role has the longest execution time per iteration
- Where idle bubbles occur (waiting for barriers)
- Whether the bottleneck is load latency, compute throughput, or synchronization

## 4. Case Study: Flash Attention 3 Optimization

### Problem Discovery

KPerfIR profiled a vanilla Triton FA3 kernel and found:
- **Idle bubbles** in the consumer (MMA) warp between GEMM_QK and GEMM_PV
- The V tensor TMA load was not issued early enough — consumer stalled waiting for V data
- The barrier arrival for V loading happened too late in the producer pipeline

### Optimization

Based on KPerfIR's pipeline-stage timing:
1. **Advanced the V-load barrier arrival** — issued V TMA load earlier in the producer pipeline
2. This overlapped V loading with K processing in the consumer
3. Reduced the idle bubble between GEMM_QK and GEMM_PV

### Results

| Kernel Variant | Performance | vs Vanilla Triton FA3 |
|:---------------|:-----------|:---------------------|
| Vanilla Triton FA3 | baseline | — |
| Manual FA3 (hand-optimized) | +15.3% | — |
| KPerfIR-optimized FA3 | **+24.1%** | +7.6% over manual |

The KPerfIR-guided optimization outperformed even the manually-optimized version because:
- The profiling data revealed a specific scheduling opportunity invisible to human analysis
- The barrier advancement was a precise, targeted change
- External profilers could not have identified this opportunity (no visibility into pipeline stage timing)

## 5. Overhead Analysis

### Profiling Overhead

| Metric | Value |
|:-------|:------|
| Average latency overhead | 8.2% |
| Maximum latency overhead | <15% |
| Per-record instrumentation cost | ~33 cycles |
| Performance degradation from instrumentation | <2% |
| Shared memory overhead | Configurable (typically 1-4 KB) |

### Overhead Correction

KPerfIR's trace replay post-processing:
1. Measures the instrumentation overhead itself (33 cycles per record)
2. Subtracts overhead from reported timings
3. Adjusts for circular buffer wrap-around
4. Produces corrected timing reports

## 6. Platform Portability

KPerfIR works on both NVIDIA and AMD GPUs because it operates at the IR level:
- **NVIDIA**: Uses `%clock` PTX register for cycle counting, PTX-level instrumentation
- **AMD**: Uses `s_memrealtime` for cycle counting, AMDGPU-level instrumentation
- Same high-level profiling tools work on both platforms
- Platform-specific lowering handled by the compiler backend

## 7. Comparison with External Profilers

| Capability | Nsight Compute | Omniperf | KPerfIR |
|:-----------|:--------------|:---------|:--------|
| Hardware counters | ✓ | ✓ | — |
| Kernel-level timing | ✓ | ✓ | ✓ |
| Roofline analysis | ✓ | ✓ | — |
| Pipeline stage timing | — | — | ✓ |
| Loop iteration profiling | — | — | ✓ |
| Critical path (warp roles) | — | — | ✓ |
| Compiler feedback loop | — | — | ✓ |
| Platform portable | NVIDIA only | AMD only | Both |

**Complementary usage**: Use Nsight/Omniperf for hardware counter analysis, KPerfIR for software pipeline analysis. Together they provide complete visibility.

## References

- [KPerfIR: A Compiler-Centric Infrastructure for GPU Kernel Profiling — OSDI 2025 (Guan et al.)](https://www.usenix.org/conference/osdi25/presentation/guan)
- [Triton Language and Compiler](https://triton-lang.org/)
- [MLIR: Multi-Level Intermediate Representation](https://mlir.llvm.org/)
- [Nsight Compute Documentation](https://developer.nvidia.com/nsight-compute)
- [AMD Omniperf Documentation](https://rocm.docs.amd.com/projects/omniperf/)
