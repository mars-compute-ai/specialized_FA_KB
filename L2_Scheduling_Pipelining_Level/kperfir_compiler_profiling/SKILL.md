---
skill_name: KPerfIR Compiler-Centric GPU Kernel Profiling
description: KPerfIR is a compiler-centric GPU kernel profiling infrastructure (OSDI 2025) that integrates profiling into Triton's MLIR-based compiler IR, enabling multi-level instrumentation, critical path analysis, and profile-driven optimization of attention kernels.
level: L2 - Scheduling and Pipelining Level
target_hardware: NVIDIA (Ampere/Hopper/Blackwell), AMD (CDNA3/CDNA4) — platform-portable
relevance: When profiling warp-specialized Flash Attention kernels at fine granularity, identifying pipeline bubbles and scheduling bottlenecks, or building profile-driven optimization feedback loops in Triton-based kernels.
---

# KPerfIR: Compiler-Centric GPU Kernel Profiling

## What It Is

KPerfIR (OSDI 2025) is a profiling infrastructure that integrates directly into the Triton compiler's MLIR-based IR workflow, rather than operating as an external tool like Nsight Compute or rocprof. By inserting profiling markers at multiple IR levels (TTIR, TTGIR, LLVM), KPerfIR can correlate timing data with high-level constructs like software pipeline stages, loop iterations, and tensor operations — something external profilers cannot do. Its key innovation is enabling profile-driven compiler optimization: the profiler feeds data back into the compiler to improve scheduling decisions.

## Key Concepts

### Multi-Level IR Instrumentation
- **TTIR level**: High-level Triton IR. Profile loop structures, tensor operations, and algorithmic phases.
- **TTGIR level**: GPU-specific IR. Profile software pipeline stages, barrier synchronization, and shared memory access patterns.
- **LLVM level**: Low-level. Profile individual instruction sequences and warp-level behavior.
- Each level offers different granularity vs overhead tradeoffs.

### Core Abstractions
- **RecordOp**: High-level profiling marker that gets lowered through the compiler stack.
- **KPerfGPUIR**: GPU-specific operations (ReadCounterOp, StoreCounterOp, InitOp, FinalizeOp).
- **Circular buffer**: Profiling data stored in shared memory using a circular buffer to handle limited SMEM capacity.
- **Trace replay**: Post-processing step that corrects for profiling overhead artifacts.

### Key Profiling Capabilities

1. **Region-based timing**: Profiles intra-kernel behavior at instruction-region granularity. Uses multi-record technique to handle asynchronous instructions (TMA, WGMMA).
2. **Iteration-based timing**: Correlates loop induction variables with timing records — identifies which loop iterations are slow (e.g., first iteration cold-start, last iteration causal mask overhead).
3. **Critical path analysis**: Identifies execution bottlenecks in warp-specialized pipelines — which warp role (producer, consumer, softmax) is the bottleneck.
4. **Program-correlated memory analysis**: Maps memory access patterns back to high-level tensor constructs.

### FA3 Optimization Case Study
- KPerfIR identified idle bubble regions in vanilla Flash Attention 3 kernel
- Profiling revealed that V tensor TMA loads could be overlapped by advancing arrival barriers
- Result: **24.1% improvement** over vanilla Triton FA3, **7.6% improvement** over manually-optimized FA3
- The improvement came from a targeted scheduling change that external profilers could not have guided

## When to Use

- Diagnosing pipeline bubbles in warp-specialized attention kernels (FA3/FA4 style)
- Identifying which pipeline stage (load, MMA, softmax) is the bottleneck in a multi-stage kernel
- Building profile-driven optimization passes for Triton-based attention kernels
- Profiling iteration-level behavior (e.g., first vs middle vs last tile of an attention pass)
- Understanding the overlap efficiency between TMA loads and WGMMA/MFMA compute

## When NOT to Use

- Need system-level profiling (GPU utilization, PCIe transfer, multi-kernel timeline) — use Nsight Systems or Omnitrace
- Need hardware counter analysis (cache hit rates, DRAM bandwidth) — use Nsight Compute or Omniperf
- Profiling non-Triton kernels — KPerfIR is integrated into the Triton compiler IR
- Simple kernel performance comparison — external profilers are simpler to set up

## Source Code Examples

### Triton Kernel with KPerfIR Region Markers

```python
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

Each `kperfir.region()` inserts profiling markers at the TTIR level that persist through lowering to TTGIR and LLVM, enabling per-region timing with semantic context.

## Key Takeaways

- KPerfIR bridges the gap between profilers and compilers — profiling markers travel through the compilation pipeline, maintaining semantic context that external tools lose
- The 24.1% FA3 speedup demonstrates the power of compiler-integrated profiling: the optimization (advancing V-load barriers) was invisible to external profilers
- Profiling overhead is low: 8.2% average latency, <2% from instrumentation itself
- Critical path analysis in warp-specialized kernels reveals which warp role is the bottleneck — essential for FA3/FA4 pipeline tuning
- The circular buffer design for shared memory profiling storage is reusable across different profiling tools

## References

- [KPerfIR: A Compiler-Centric Infrastructure for GPU Kernel Profiling — OSDI 2025 (Guan et al.)](https://www.usenix.org/conference/osdi25/presentation/guan)
- [Triton Language and Compiler](https://triton-lang.org/)
- [MLIR: Multi-Level Intermediate Representation](https://mlir.llvm.org/)
