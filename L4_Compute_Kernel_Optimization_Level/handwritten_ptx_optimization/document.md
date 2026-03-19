# Advanced NVIDIA CUDA Kernel Optimization: Handwritten PTX

Source: https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx

## Overview

Handwritten PTX (Parallel Thread Execution) assembly is a tool of last resort for performance-critical GPU kernel sections where compiler-generated code leaves measurable performance on the table. PTX provides fine-grained control over instruction selection, register allocation, predicate-driven execution, and instruction ordering that is unavailable through CUDA C++.

## Why Handwritten PTX is Needed

The CUDA C++ compiler (`nvcc` -> `ptxas`) generally produces good code, but certain patterns are suboptimal:

1. **Branch-heavy code**: Compiler may generate conditional branches where predicate-based execution would be more efficient
2. **Register spills**: Complex expressions may cause the compiler to spill registers to local memory
3. **Suboptimal instruction selection**: The compiler may not choose the most efficient instruction variant for a given operation
4. **Instruction ordering**: The compiler's scheduling heuristics may not produce the optimal instruction sequence for a specific pipeline configuration

CUTLASS includes handwritten PTX because it is designed for the best possible performance on each GPU architecture.

## Instruction-Level Optimization Example: top_2_reduce_scalar

### CUDA C++ Version (Compiler-Generated)
```cpp
// C++ implementation with conditional branches
void top_2_reduce_scalar(float& max1, float& max2, float val1, float val2) {
    float mx = fmaxf(val1, val2);
    if (max1 > val2) {
        max2 = mx;
        max1 = max1;  // unchanged
    } else {
        max2 = max1;
        max1 = val2;
    }
}
// Problem: conditional branch causes warp divergence and register spill
```

### Handwritten PTX Version (Optimized)
```
asm volatile(
    "{\n"
    "  .reg .f32 mx;\n"           // Explicit temporary register
    "  .reg .pred p;\n"           // Predicate register
    "  max.f32 mx, %3, %4;\n"    // mx = max(val1, val2)
    "  setp.gtu.f32 p, %2, %4;\n" // p = (max1 > val2)
    "  selp.f32 %1, mx, %2, p;\n" // max2 = p ? mx : max1
    "  selp.f32 %0, %2, %4, p;\n" // max1 = p ? max1 : val2
    "}\n"
    : "=f"(max1), "=f"(max2)
    : "f"(max1), "f"(val1), "f"(val2)
);
```

Key advantages:
- **No branch divergence**: `setp` + `selp` pattern replaces conditional branches
- **Explicit register control**: `.reg .f32 mx` and `.reg .pred p` prevent spills
- **4 instructions** instead of branch + multiple conditional moves
- **Deterministic execution**: All threads execute the same instructions regardless of data

## Performance Results

Benchmarks on GEMM with top_k and softmax operations:

| Matrix Size (m) | With PTX (GFlop/s) | Without PTX (GFlop/s) | Improvement |
|-----------------|--------------------|-----------------------|-------------|
| 1,024           | 5,704              | 4,998                 | **+14.1%**  |
| 8,192           | 19,794             | 17,885                | **+10.7%**  |
| 16,384          | 21,476             | 20,066                | **+7.0%**   |

Performance improved between 7% to 14% when handwritten PTX replaced CUDA C++ fallback implementations for top_k and softmax operations.

## Key Optimization Techniques

### 1. Avoiding Branching with Predicates
```
// Instead of: if (cond) { a = x; } else { a = y; }
setp.gt.f32 p, cond_val, threshold;   // Set predicate
selp.f32 a, x, y, p;                  // Select based on predicate
// No branch, no divergence, no pipeline flush
```

### 2. Register Pressure Control
```
asm volatile(
    "{\n"
    "  .reg .f32 tmp0, tmp1;\n"    // Declare exactly the temporaries needed
    "  .reg .pred p0, p1;\n"       // Predicate registers
    // ... use tmp0, tmp1, p0, p1 ...
    "}\n"                           // Temporaries go out of scope
);
// Compiler knows exact register requirements, avoids spilling
```

### 3. Specialized Instruction Selection
```
// Direct access to optimal instructions:
max.f32     // Hardware max (single cycle)
min.f32     // Hardware min (single cycle)
setp.gtu.f32  // Set predicate, greater-than-unsigned
selp.f32    // Select on predicate (branchless conditional)
fma.rn.f32  // Fused multiply-add with round-to-nearest
ex2.approx.ftz.f32  // Fast exponential with flush-to-zero
rcp.approx.ftz.f32  // Fast reciprocal with flush-to-zero
```

### 4. Compilation Levels

```
CUDA C++ source
    |
    v  (nvcc frontend)
PTX assembly (virtual ISA, portable across architectures)
    |
    v  (ptxas compiler)
SASS assembly (native ISA, architecture-specific)
    |
    v
Binary (cubin)
```

Handwritten PTX sits at the intermediate level -- more portable than SASS but with finer control than CUDA C++. The `ptxas` compiler still performs instruction scheduling and register allocation on PTX, but respects explicit `.reg` declarations and instruction ordering hints.

## Tradeoffs

### Advantages
- 7-14% performance improvement on critical code sections
- Eliminates branch divergence via predicate-based execution
- Controls register pressure explicitly
- Enables use of specific instruction variants (`.ftz`, `.approx`)

### Disadvantages
- Reduced portability across GPU architectures
- Increased maintenance complexity
- Architecture-specific performance gains may not generalize
- Should be "a tool of last resort" -- most code should remain in CUDA C++

## Relevance to Attention Kernels

In Flash Attention and similar kernels, handwritten PTX is used for:
- **Softmax reduction**: Predicate-based max/sum reduction without branch divergence
- **Exponential computation**: Direct control over `ex2.approx.ftz.f32` vs full-precision variants
- **Score accumulation**: FMA instruction selection for online softmax updates
- **Warp-level operations**: Shuffle instructions for cross-thread communication
