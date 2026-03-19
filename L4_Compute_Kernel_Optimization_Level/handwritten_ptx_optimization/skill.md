---
skill_name: Handwritten PTX Instruction Optimization
description: Use inline PTX assembly to replace compiler-generated branch-heavy code with predicate-based branchless execution and explicit register control for 7-14% gains.
level: L4 - Compute Kernel Optimization Level
target_hardware: All NVIDIA CUDA GPUs (PTX is the portable virtual ISA)
relevance: When an AI agent has profiled a GPU kernel and identified branch divergence, register spills, or suboptimal instruction selection in performance-critical sections like softmax reductions or top-k operations.
---

# Handwritten PTX Instruction Optimization

## What It Is
Handwritten PTX (Parallel Thread Execution) assembly allows developers to bypass the CUDA C++ compiler's code generation for performance-critical kernel sections. By writing inline PTX, developers gain direct control over instruction selection (e.g., predicate-based `selp` instead of branches), register allocation (explicit `.reg` declarations to prevent spills), and instruction variant selection (e.g., `.ftz` and `.approx` modifiers). This is used in CUTLASS and other high-performance libraries as a last-resort optimization that recovers 7-14% performance.

## Key Concepts
- **Predicate-based execution**: Replace conditional branches (`if/else`) with `setp` + `selp` instruction pairs, eliminating warp divergence entirely
- **Explicit register declaration**: `.reg .f32 tmp` declarations give the compiler exact register requirements, preventing spills to local memory
- **Instruction variant selection**: Direct access to `.ftz` (flush-to-zero), `.approx` (approximate), and `.rn` (round-to-nearest) instruction modifiers
- **Branchless reductions**: Top-k and softmax reductions implemented without any conditional branches
- **Compilation hierarchy**: CUDA C++ -> PTX (virtual ISA) -> SASS (native ISA); PTX is the optimal intervention point balancing portability with control
- **Scoped temporaries**: PTX blocks `{...}` scope register declarations, enabling the compiler to reuse registers across blocks

## Instruction Patterns / Code
```
// === Pattern 1: Branchless conditional select ===
// Replaces: if (a > b) { result = x; } else { result = y; }
asm volatile(
    "{\n"
    "  .reg .pred p;\n"
    "  setp.gt.f32 p, %1, %2;\n"     // p = (a > b)
    "  selp.f32 %0, %3, %4, p;\n"    // result = p ? x : y
    "}\n"
    : "=f"(result) : "f"(a), "f"(b), "f"(x), "f"(y)
);

// === Pattern 2: Top-2 reduction without branches ===
asm volatile(
    "{\n"
    "  .reg .f32 mx;\n"
    "  .reg .pred p;\n"
    "  max.f32 mx, %3, %4;\n"         // mx = max(val1, val2)
    "  setp.gtu.f32 p, %2, %4;\n"     // p = (max1 > val2)
    "  selp.f32 %1, mx, %2, p;\n"     // max2 = p ? mx : max1
    "  selp.f32 %0, %2, %4, p;\n"     // max1 = p ? max1 : val2
    "}\n"
    : "=f"(max1), "=f"(max2)
    : "f"(max1), "f"(val1), "f"(val2)
);

// === Pattern 3: Fast exponential with flush-to-zero ===
asm volatile(
    "ex2.approx.ftz.f32 %0, %1;\n"    // 2^x, approx, flush denormals
    : "=f"(result) : "f"(input)
);

// === Pattern 4: Fast reciprocal for softmax division ===
asm volatile(
    "rcp.approx.ftz.f32 %0, %1;\n"    // 1/x, approx, flush denormals
    : "=f"(result) : "f"(input)
);
```

## Performance Impact
- **+14.1% improvement** at m=1024 (5,704 vs 4,998 GFlop/s)
- **+10.7% improvement** at m=8192 (19,794 vs 17,885 GFlop/s)
- **+7.0% improvement** at m=16384 (21,476 vs 20,066 GFlop/s)
- Improvements are for top_k and softmax operations in GEMM kernels
- Benefits are most pronounced at smaller problem sizes where instruction overhead is a larger fraction of total time

## When to Use
- When Nsight Compute profiling shows branch divergence in performance-critical sections
- When register spill counts are high despite occupancy tuning
- When the compiler selects suboptimal instruction variants (e.g., full-precision instead of approximate)
- For softmax, top-k, and reduction operations in attention kernels
- When the last 5-15% of performance matters and all algorithmic optimizations are exhausted

## When NOT to Use
- For the vast majority of kernel code -- handwritten PTX should be "a tool of last resort"
- When portability across GPU architectures is required (PTX is portable but optimization benefits may not transfer)
- When the kernel is memory-bound (instruction-level optimizations have no effect)
- When the code section is not on the critical path
- When the compiler already generates optimal code (verify with `cuobjdump --dump-sass`)

## Key Takeaways
- The `setp` + `selp` pattern is the single most important PTX optimization: it eliminates branch divergence for conditional operations
- Explicit `.reg` declarations prevent register spills by giving the `ptxas` compiler exact register requirements
- PTX sits at the right abstraction level: portable across architectures while enabling instruction-level control
- In attention kernels, the highest-value targets for PTX optimization are softmax reduction, exponential computation, and online rescaling operations
- Always validate with benchmarks -- the compiler is often better than manual optimization for non-critical code

## References
- [Advanced NVIDIA CUDA Kernel Optimization: Handwritten PTX (NVIDIA Blog)](https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx)
- [Understanding PTX (NVIDIA Blog)](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- CUTLASS source code (examples of production PTX usage)
