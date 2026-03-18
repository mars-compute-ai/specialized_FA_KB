# ThunderKittens: Tile Abstraction Design, GPU Mapping, and Attention Kernels

## Overview

ThunderKittens is a header-only CUDA framework from Hazy Research at Stanford that provides tile-level primitives for writing high-performance GPU kernels. This document covers how ThunderKittens' abstractions map to GPU hardware, the design philosophy behind the tile primitives, and how they enable compact, high-performance attention kernel implementations.

## Design Philosophy: Hardware-Up Abstraction

ThunderKittens is designed from the hardware up, not from the algorithm down. The key insight is that modern GPUs are not general-purpose 1000x1000 matrix multiply machines -- they are manycore processors where each core (warp) efficiently runs ~16x16 matrix multiplies. All ThunderKittens abstractions start at this 16x16 tile granularity.

### Why Tiles?

The 16x16 minimum tile size is dictated by hardware:
- **Tensor cores** operate on 16x16 (or 8x8, 32x32) matrix fragments
- **Shared memory** bank structure maps naturally to 16-wide access patterns
- **Warp registers** (32 threads x 4-8 registers) can hold exactly one 16x16 tile with appropriate distribution

By making the tile the fundamental unit, ThunderKittens eliminates an entire class of performance bugs: incorrect thread-to-data mappings, bank conflicts, and misaligned tensor core operands.

## The 4 Core Abstractions in Detail

### 1. Register Tiles (`rt`)

Register tiles are the primary compute abstraction. Their contents are distributed across the threads of a single warp (32 threads on NVIDIA).

**Declaration syntax:**
```cpp
// kittens::rt_<type><rows, cols>
kittens::rt_bf<32, 64> tile;    // 32x64 BF16 register tile, row-layout (default)
kittens::rt_fl<16, 16> acc;     // 16x16 FP32 register tile (accumulator)
kittens::rt_hf<64, 64> big;    // 64x64 FP16 register tile
```

**How data is distributed across threads:**
For a 16x16 FP32 tile, each of 32 threads holds 8 elements (256 total elements / 32 threads). The distribution follows tensor core requirements -- for WGMMA, the B operand must be in column layout, while A can be in row layout.

**Layout matters for correctness:**
ThunderKittens enforces layout compatibility at compile time via C++20 concepts. If you try to pass a row-layout tile as the B operand to `mma_AB`, you get a compile error, not a silent wrong answer. This is one of ThunderKittens' most important safety features.

**Key operations:**
```cpp
// Assembly-like signature: destination first
kittens::mul(c, a, b);        // c = a * b (element-wise)
kittens::add(c, a, b);        // c = a + b
kittens::mma_AB(d, a, b, c);  // d = a @ b + c (tensor core matmul)
kittens::transpose_sep(b, a);  // b = a^T (requires separate src/dst)
kittens::copy(dst, src);       // dst = src (with type conversion if needed)
```

### 2. Shared Tiles (`st`)

Shared tiles reside in shared memory and are accessible by all warps in a block.

**Declaration syntax:**
```cpp
__shared__ kittens::st_bf<64, 64> k_tile;  // 64x64 BF16 in shared memory
__shared__ kittens::st_hf<32, 64> v_tile;  // 32x64 FP16 in shared memory
```

**Bank conflict avoidance:**
ThunderKittens handles shared memory bank conflicts internally. When you store a register tile to shared memory or load from shared memory, ThunderKittens applies the appropriate swizzling pattern so that all 32 threads access different banks simultaneously. You never need to manually compute swizzle offsets.

**TMA integration:**
On Hopper and Blackwell, shared tiles support asynchronous TMA (Tensor Memory Accelerator) operations:
```cpp
// Asynchronous load from global memory to shared tile
tma::load_async(shared_tile, global_layout, {row_idx, col_idx}, barrier);

// Asynchronous store from shared tile to global memory
tma::store_async(global_layout, shared_tile, {row_idx, col_idx});
```

TMA operations are issued by a single thread (typically `laneid() == 0`) and complete asynchronously, freeing the warp for compute.

### 3. Register Vectors

Register vectors are 1D arrays associated with register tiles, used for reductions and broadcasts.

**Three flavors:**
1. **Column vectors (`::col_vec`)**: One element per row of the associated tile. Used for row-wise reductions (e.g., softmax row-max, row-sum).
2. **Row vectors (`::row_vec`)**: One element per column. Used for column-wise reductions.
3. **Naive vectors**: Flat layout optimized for compute-heavy operations like layernorm.

**Critical for softmax:**
```cpp
rt_fl<16, 64> s_acc;              // Attention scores tile
rt_fl<16, 64>::col_vec row_max;   // Max per row (16 elements)
rt_fl<16, 64>::col_vec row_sum;   // Sum per row (16 elements)

// Row-wise max reduction
row_reduce(row_max, s_acc, kittens::base_ops::max);

// Subtract max and exponentiate (for numerical stability)
sub_row(s_acc, s_acc, row_max);  // S[i][j] -= max[i]
exp(s_acc, s_acc);                // S[i][j] = exp(S[i][j])

// Row-wise sum
row_reduce(row_sum, s_acc, kittens::base_ops::sum);

// Normalize
div_row(s_acc, s_acc, row_sum);  // S[i][j] /= sum[i]
```

### 4. Shared Vectors

Shared vectors reside in shared memory and are used for inter-warp communication of reduction results.

**Use case:** In multi-warp attention kernels, each warp computes partial softmax statistics. These must be combined across warps via shared memory:

```cpp
__shared__ kittens::sv_fl<16> shared_max;  // Shared max vector

// Each warp writes its partial max
if (warp_is_representative) {
    store(shared_max, local_row_max);
}
__syncthreads();

// Each warp reads the combined max
load(combined_max, shared_max);
```

## Mapping to GPU Hardware Hierarchy

### NVIDIA's Scope Hierarchy and ThunderKittens

| GPU Scope | ThunderKittens Scope | Hardware Resources | TK Abstraction |
|-----------|---------------------|-------------------|----------------|
| Thread | (Not directly used) | Up to 256 registers | - |
| Warp (32 threads) | Default scope | Register file | Register tiles, register vectors |
| Warpgroup (4 warps) | `kittens::warpgroup::` | WGMMA unit | Warpgroup MMA operations |
| Block (N warps) | Block scope | Shared memory (up to 228 KB on H100) | Shared tiles, shared vectors |
| Grid | (Minimal support) | Global memory, TMA descriptors | Global layouts for TMA |

### Register Budget

On H100, each thread has up to 256 32-bit registers. For a warp of 32 threads, that is 8192 registers total. A 16x16 FP32 register tile uses 256 elements = 256 registers across 32 threads = 8 registers per thread. This means a warp can hold ~32 register tiles simultaneously.

ThunderKittens provides explicit register management:
```cpp
warpgroup::decrease_registers<40>();  // Producer warps: fewer registers
warpgroup::increase_registers<232>(); // Consumer warps: more registers
```

This maps to Hopper's dynamic register allocation between producer and consumer warpgroups.

## Attention Kernel Architecture: The LSCF Template

ThunderKittens' `prototype.cuh` provides the Load-Store-Compute-Finish (LSCF) template for warp-specialized kernels. This encodes the same warp specialization pattern that FA3 pioneered.

### Kernel Structure

```
Block of 8+ warps
├── Producer warpgroup (1 warpgroup = 4 warps)
│   └── load(): TMA loads from global to shared memory (ping-pong buffered)
├── Consumer warpgroup(s) (1-2 warpgroups = 4-8 warps)
│   ├── compute(): WGMMA matmuls + softmax on shared memory tiles
│   └── finish(): TMA stores from shared to global memory
└── Synchronization via named barriers (inputs_arrived, inputs_finished, finish_finished)
```

### Layout Definition

```cpp
template<int M_BLOCK, int N_BLOCK>
struct attention_layout {
    using base_tile      = st_bf<64, 64>;           // 64x64 shared tile
    using global_layout  = gl<bf16, 1, 1, -1, -1>;  // Global memory layout
    struct globals       { global_layout Q, K, V, O; };
    struct input_block   { base_tile q, k[N_BLOCK]; };  // Shared memory inputs
    struct finish_block  { base_tile o[M_BLOCK]; };      // Shared memory outputs
    struct common_state  { int2 coord; };                 // Shared state
    struct consumer_state {
        rt_fl<16, 64> o_acc;           // Output accumulator
        rt_fl<16, 64>::col_vec m, l;   // Softmax statistics
    };
};
```

### Producer: Asynchronous Data Loading

```cpp
struct producer {
    __device__ static void setup(producer_setup_args<layout> args) {
        warpgroup::decrease_registers<40>();
    }
    __device__ static void load(producer_load_args<layout> args) {
        if (warpgroup::laneid() == 0) {
            tma::expect(args.inputs_arrived, args.input);
            tma::load_async(args.input.q, args.globals.Q,
                           {args.common.coord.x, args.iter}, args.inputs_arrived);
            tma::load_async(args.input.k[0], args.globals.K,
                           {args.iter, args.common.coord.y}, args.inputs_arrived);
        }
    }
};
```

### Consumer: Compute + Softmax

```cpp
struct consumer {
    __device__ static void setup(consumer_setup_args<layout> args) {
        warpgroup::increase_registers<232>();
        zero(args.state.o_acc);
        fill(args.state.m, -INFINITY);
        zero(args.state.l);
    }
    __device__ static void compute(consumer_compute_args<layout> args) {
        // S = Q * K^T
        warpgroup::mma_AB(s_acc, args.input.q, args.input.k[0]);
        warpgroup::mma_async_wait();

        // Online softmax update
        row_reduce(new_max, s_acc, base_ops::max);
        // ... rescale, exp, accumulate ...

        // O += P * V
        warpgroup::mma_AB(args.state.o_acc, p_tile, args.input.v);
        warpgroup::mma_async_wait();

        arrive(args.inputs_finished);
    }
    __device__ static void finish(consumer_finish_args<layout> args) {
        // Final normalization: O /= l
        div_row(args.state.o_acc, args.state.o_acc, args.state.l);

        // Store to shared, then TMA to global
        warpgroup::store(args.finish.o[0], args.state.o_acc);
        tma::store_async(args.globals.O, args.finish.o[0], {coord});

        arrive(args.finish_finished);
    }
};
```

## Attention Kernel Examples in ThunderKittens

### Available Attention Implementations

ThunderKittens ships with several attention kernel implementations under `kernels/`:

1. **Standard Attention (Forward + Backward)**: Causal and non-causal variants for H100
2. **GQA (Grouped Query Attention)**: For Llama-3 style models
3. **Linear Attention**: For LoLCATS-style sub-quadratic models
4. **Based Attention**: Short sliding window + large-state linear attention
5. **FlashMLA**: Optimized Multi-Latent Attention for DeepSeek-style models

### Performance on H100

| Kernel | Configuration | Performance | % of Peak |
|--------|--------------|-------------|-----------|
| GEMM (BF16) | 2-block x 4-block | 855 TFLOPS | 86% |
| Attention (FP16) | Standard FA3 | 740+ TFLOPS | 75% |
| Attention (FP8) | With incoherent processing | ~1.2 PFLOPS | ~60% |
| ThunderMLA | FlashMLA | Fastest at release | - |

### LLM Inference Demos

ThunderKittens includes end-to-end LLM inference demos:
- **Llama 3 8B** with TK GQA attention
- **Qwen 2.5 7B** with TK attention
- **LoLCATS-Llama 3.1 8B** with TK linear attention

## ThunderKittens 2.0 (Blackwell Support)

Released January 2026, ThunderKittens 2.0 adds:

- **Blackwell B200 GPU support** with TCGEN05 tensor core instructions
- **MXFP8 and NVFP4 precision** support
- **Restructured repository**: Kernels are self-contained with individual Makefiles
- **Ampere support deprecated**: No longer actively maintained for A100

### Blackwell-Specific Features
- `TCGEN05` calls replace `WGMMA` for tensor core operations
- Distributed Shared Memory for inter-SM communication
- GPU networking primitives for NVLink/NVSwitch multi-GPU operations

## The Kittens Ecosystem

| Framework | Target Hardware | Repository |
|-----------|---------------|------------|
| ThunderKittens | NVIDIA (H100, B200) | [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) |
| HipKittens | AMD (MI300X, MI350X) | [HazyResearch/HipKittens](https://github.com/HazyResearch/HipKittens) |
| ThunderMittens | Apple Silicon | [HazyResearch/ThunderMittens](https://github.com/HazyResearch/ThunderMittens) |

## Building and Running

### Requirements
- CUDA 12.8+
- C++20 (gcc-11 or later)
- PyTorch 2.8+ (for Python bindings)
- H100 or B200 GPU

### Quick Start
```bash
git clone https://github.com/HazyResearch/ThunderKittens
cd ThunderKittens/kernels/attn/h100
make && make run
```

### Environment Setup
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```

## References

- [ThunderKittens GitHub Repository](https://github.com/HazyResearch/ThunderKittens)
- [GPUs Go Brrr (May 2024)](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
- [Easier, Better, Faster, Cuter (Oct 2024)](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2)
- [ThunderKittens FP8 (Nov 2024)](https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8)
- [ThunderMLA (Mar 2025)](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla)
- [ThunderKittens on Blackwell (Mar 2025)](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell)
- [No Bubbles Megakernel (May 2025)](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [One Kernel for All GPUs (Sep 2025)](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl)
- [ThunderKittens 2.0 (Feb 2026)](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)
- [Single GPU Paper (arXiv 2410.20399)](https://arxiv.org/abs/2410.20399)
- [Multi-GPU Paper (arXiv 2511.13940)](https://arxiv.org/abs/2511.13940)
