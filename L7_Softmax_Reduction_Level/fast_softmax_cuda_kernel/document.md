# How to Write a Fast Softmax CUDA Kernel

**Source**: [AITemplate Wiki - Facebook Incubator](https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F)

---

## Overview

This guide from Facebook's AITemplate project details how to write efficient softmax CUDA kernels, covering warp-level reductions, vectorized memory operations, block-level reductions, and thread organization strategies. The techniques achieve up to 50% speedup over state-of-the-art implementations on NVIDIA A100 GPUs.

---

## 1. Core Optimization Techniques

### 1.1 Warp-Level Reductions

The primary optimization leverages register caching within warps (32-thread groups). Using `__shfl_xor_sync()`, threads share values directly through registers without accessing shared memory.

```cuda
template <typename T, int NUM>
__inline__ __device__ T warpReduceMax(T* val, int thread_group_width = 32) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = thread_group_width / 2; mask > 0; mask >>= 1) {
      val[i] = max(val[i], __shfl_xor_sync(0xffffffff, val[i], mask, 32));
    }
  }
  return (T)(0.0f);
}
```

This approach is optimal when K (reduction dimension) is small because register access has minimal latency compared to shared memory. In just 5 iterations (16->8->4->2->1), 32 values are reduced to one, all happening in registers.

### 1.2 Vectorized Memory Operations

Reading data in packs (e.g., `half4`, `float4`) dramatically improves memory throughput:
- Pack sizes of 1, 2, 4, and 8 elements
- Larger vectors enable fewer memory transactions
- Up to 50% speedup with pack size of 8

### 1.3 Block Reductions for Large K

When K exceeds register capacity, block-level reductions use shared memory with a two-stage approach:

```cuda
template <typename T, int NUM>
__inline__ __device__ T blockReduceSum(T* val) {
  __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  // Stage 1: Reduce within each warp
  warpReduceSum<T, NUM>(val);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }
  __syncthreads();

  // Stage 2: Reduce warp results
  for (int i = 0; i < NUM; i++) {
    val[i] = threadIdx.x < (blockDim.x / 32.f) ? shared[i][lane] : (T)(0.0f);
  }
  if(wid==0) warpReduceSum<T, NUM>(val);
  return (T)0.0f;
}
```

---

## 2. Thread Organization Strategies

### Warp Reduction Configuration (Small K)

- **Grid**: `<M/pack_size>`, **Block**: `<32, 4>`
- When `K/pack_size >= 32`: Each thread processes `K/32` columns
- When `K/pack_size < 32`: Adjust block dimensions; warp size becomes `K/pack_size`

### Block Reduction Configuration (Large K)

- **Grid**: `<M>`, **Block**: `<block_size>` where block_size in {128, 256, 512, 1024}
- Uses `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` to optimize occupancy
- Each thread processes `K/block_size` columns

---

## 3. Warp vs. Block Reduction Thresholds

Performance crossover points on NVIDIA A100 by pack size:

| Pack Size | Threshold K (switch from warp to block) |
|-----------|----------------------------------------|
| 1         | 1408                                   |
| 2         | 1152                                   |
| 4         | 1920                                   |
| 8         | 3840                                   |

Below the threshold, warp reduction is faster. Above it, block reduction is faster.

---

## 4. Complete Warp Reduction Softmax Kernel

```cuda
template<int cols_per_thread>
__global__ void softmax_stored_locally_multi_dim(
  const half4* input, half4* output, size_t m, size_t n) {
  constexpr int num_packs = (cols_per_thread + 3) / 4;
  float4 buf[num_packs];
  const int m_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int tid = threadIdx.x;

  for (int64_t row = m_idx; row < m; row += gridDim.x * blockDim.y) {
    const int64_t row_offset = row * (n >> 2);
    const half4* row_x = input + row_offset;
    half4* row_y = output + row_offset;
    float local_max[1] = {-Inf<float>()};

    // Step 1: Read data and find local max
    #pragma unroll
    for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
      const int col = pack_id * blockDim.x + tid;
      if (col < n / 4) {
        buf[pack_id] = __half42float4(row_x[col]);
        local_max[0] = max(local_max[0], max4(buf[pack_id]));
      }
    }
    // Reduce max across warp
    warpReduceMax<float, 1>(local_max, blockDim.x);

    // Step 2: Compute exp and sum
    float local_sum[1] = {0.0f};
    #pragma unroll
    for (int i = 0; i < num_packs; ++i) {
      buf[i].x = exp(buf[i].x - local_max[0]);
      buf[i].y = exp(buf[i].y - local_max[0]);
      buf[i].z = exp(buf[i].z - local_max[0]);
      buf[i].w = exp(buf[i].w - local_max[0]);
      local_sum[0] += buf[i].x + buf[i].y + buf[i].z + buf[i].w;
    }
    // Reduce sum across warp
    warpReduceSum<float, 1>(local_sum, blockDim.x);

    // Step 3: Write normalized output
    for (int i = 0; i < num_packs; ++i) {
      const int col = i * blockDim.x + tid;
      if (col < n / 4) {
        row_y[col] = __float42half4({
          buf[i].x / local_sum[0],
          buf[i].y / local_sum[0],
          buf[i].z / local_sum[0],
          buf[i].w / local_sum[0]
        });
      }
    }
  }
}
```

---

## 5. Memory Hierarchy Optimization

Access latencies: **global memory >> shared memory > register**

The implementation exploits this by:
1. **Storing intermediate results in registers** when possible (the `buf` array lives in registers)
2. **Using shared memory only when register storage is insufficient** (block reduction case)
3. **Minimizing synchronization points** through warp-level operations (no `__syncthreads()` needed)
4. **Vectorized loads** reduce the number of memory transactions

---

## 6. Numerical Stability

The standard safe softmax formula is used throughout:

```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

The max subtraction prevents overflow before exponential computation. All intermediate computations (max, sum, exp) are performed in fp32 even when input/output is fp16/bf16.

---

## 7. Performance Results (NVIDIA A100)

- **Pack size 8**: Up to 50% speedup compared to OneFlow's state-of-the-art implementation
- **Typical improvement**: ~10% gain when K < 4000 (the most common softmax use case)
- **Warp reduction dominates** for K < ~2000: avoids shared memory overhead entirely
- **Block reduction necessary** for K > ~4000: shared memory amortizes over more elements

---

## References

- [AITemplate Wiki](https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F)
- [OneFlow Softmax Kernel](https://oneflow2020.medium.com/how-to-implement-an-efficient-softmax-cuda-kernel-oneflow-performance-optimization-sharing-405ad56e9031)
- [Triton Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
