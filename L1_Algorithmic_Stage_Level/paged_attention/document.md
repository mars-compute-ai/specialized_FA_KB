# PagedAttention: Efficient Memory Management for Large Language Model Serving

**Source:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023, arXiv:2309.06180)

## 1. Introduction

Large Language Model (LLM) inference is fundamentally a memory management problem. During autoregressive text generation, the model must store the **Key-Value (KV) cache** -- the key and value projections of all previously generated tokens -- so that each new token can attend to the full context without recomputing all past attention. For a model like LLaMA-13B with 40 layers, 40 heads, and head dimension 128, the KV cache for a single sequence of 2048 tokens consumes:

```
2 (K + V) * 40 layers * 40 heads * 128 dim * 2048 tokens * 2 bytes (FP16)
= 2 * 40 * 40 * 128 * 2048 * 2 = ~1.6 GB
```

For a batch of 256 concurrent requests, this is 410 GB -- far exceeding any single GPU's memory. In practice, the KV cache is the primary bottleneck limiting batch size and therefore serving throughput.

The critical problem is not just the total memory required but **how that memory is managed**. Existing systems pre-allocate contiguous memory for each sequence's KV cache based on the maximum possible sequence length, leading to massive **internal fragmentation** (allocated but unused memory) and **external fragmentation** (unusable gaps between allocations). Measurements show that **60-80% of allocated KV cache memory is wasted** in typical serving workloads.

PagedAttention solves this by applying the virtual memory and paging concepts from operating systems to the KV cache, achieving near-zero waste and enabling 2-4x higher serving throughput.

## 2. The Memory Management Problem

### 2.1 KV Cache Lifecycle

For each request in an LLM serving system:

1. **Prefill phase:** Process the input prompt. KV cache is allocated for all prompt tokens at once (known length).
2. **Decode phase:** Generate tokens one at a time. Each new token adds one KV entry to the cache. The final length is unknown in advance.
3. **Completion:** Request finishes (EOS token, max length, or user cancellation). All KV cache memory is freed.

The unknown output length creates a fundamental allocation dilemma:
- **Over-allocate (max length):** Wastes memory. A 2048-token maximum allocation for a response that only generates 50 tokens wastes 97.6% of memory.
- **Under-allocate (heuristic):** Risks running out of space mid-generation, requiring either recomputation (expensive) or request termination (poor UX).
- **Dynamic reallocation:** Requires copying the entire KV cache to a new, larger contiguous allocation -- expensive and fragments memory further.

### 2.2 Fragmentation Analysis

In a serving system with many concurrent requests:

**Internal fragmentation:** Each request's pre-allocated KV buffer has unused space (tokens not yet generated or never generated). Measured at 30-50% of allocated memory.

**External fragmentation:** As requests complete and free their memory, the remaining free space is split into non-contiguous chunks that may be individually too small for new allocations. Measured at 10-30% of total memory.

**Reservation overhead:** Some systems reserve memory for the maximum possible sequence length at request arrival, even though most requests will be much shorter. This further reduces effective capacity.

**Total waste:** 60-80% of KV cache memory is wasted at any given time in naive allocation schemes.

### 2.3 Impact on Throughput

Wasted memory directly limits batch size. If a system wastes 70% of KV cache memory, it can only serve 30% as many concurrent requests as it theoretically could. Since LLM decode is memory-bandwidth-bound (not compute-bound), throughput scales linearly with batch size until compute becomes the bottleneck. Thus, reducing memory waste directly translates to higher throughput.

## 3. PagedAttention Design

### 3.1 Core Concepts

PagedAttention introduces three key concepts borrowed from OS virtual memory:

#### 3.1.1 Pages (Blocks)

The KV cache is divided into fixed-size **pages** (also called blocks). Each page holds B tokens' worth of key and value vectors for one attention head in one layer.

```
Page structure:
    keys:   [B x d_head] float16 tensor
    values: [B x d_head] float16 tensor
    metadata: ref_count, layer_id, head_id

Typical B = 16 tokens per page
Page size: 2 * 16 * 128 * 2 = 8 KB (for d_head = 128, FP16)
```

The choice of B = 16 is a tradeoff:
- **Smaller B:** Less internal fragmentation (at most B-1 tokens wasted per sequence), but more pages to manage and more page table entries
- **Larger B:** Fewer pages to manage, but more internal fragmentation
- **Hardware alignment:** B should be a multiple of memory transaction size (128 bytes / 2 bytes per FP16 = 64 elements, but this applies to the d_head dimension, not the token dimension)

#### 3.1.2 Page Table

Each sequence has a **page table** that maps logical page indices to physical page addresses:

```
page_table[sequence_id] = [phys_addr_0, phys_addr_1, ..., phys_addr_k]

# Logical page i for sequence s is at physical address:
physical_address = page_table[s][i]

# Token t of sequence s is in:
logical_page = t // B
offset_in_page = t % B
physical_address = page_table[s][logical_page]
kv_location = kv_pool[physical_address][offset_in_page]
```

The page table is stored in CPU memory (small: one int32 per page per sequence) and transferred to the GPU kernel as an argument.

#### 3.1.3 Block Allocator

A GPU-side memory pool manages physical blocks:

```
class BlockAllocator:
    def __init__(self, num_blocks, block_size):
        self.free_blocks = deque(range(num_blocks))
        self.ref_counts = [0] * num_blocks

    def allocate(self):
        block_id = self.free_blocks.popleft()
        self.ref_counts[block_id] = 1
        return block_id

    def free(self, block_id):
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_blocks.append(block_id)

    def share(self, block_id):
        self.ref_counts[block_id] += 1
```

### 3.2 Memory Layout

The GPU memory pool is a pre-allocated contiguous buffer divided into physical blocks:

```
GPU Memory Layout:
+--------+--------+--------+--------+--------+--------+
| Block 0| Block 1| Block 2| Block 3| Block 4| Block 5| ...
+--------+--------+--------+--------+--------+--------+
    |        |        |        |        |        |
    v        v        v        v        v        v
  Seq A    Seq B    Seq A    Free    Seq C    Seq B
  Page 0   Page 0   Page 1           Page 0   Page 1

Page Tables:
  Seq A: [0, 2]        (logical pages 0,1 -> physical blocks 0,2)
  Seq B: [1, 5]        (logical pages 0,1 -> physical blocks 1,5)
  Seq C: [4]           (logical page 0 -> physical block 4)
  Free:  [3, 6, 7, ...]
```

Key property: **physical blocks for a single sequence are not contiguous.** This eliminates external fragmentation entirely -- any free block can be used by any sequence.

### 3.3 Attention Kernel Modification

The standard attention kernel assumes KV cache is contiguous per sequence. PagedAttention modifies the kernel to follow page table indirection:

```
# Standard attention kernel (decode, single query)
for i in range(seq_len):
    score[i] = q @ kv_cache[seq_id, i, :d].T   # contiguous access

# PagedAttention kernel (decode, single query)
for block_idx in range(num_blocks):
    phys_block = page_table[seq_id][block_idx]
    for offset in range(B):
        if block_idx * B + offset >= seq_len:
            break
        token_idx = block_idx * B + offset
        score[token_idx] = q @ kv_pool[phys_block, offset, :d].T  # indirect access
```

### 3.4 Optimized PagedAttention Kernel

The production PagedAttention kernel is carefully optimized:

```cuda
// PagedAttention v2 kernel (simplified)
// Each warp group handles one query head attending to all KV blocks

template <int HEAD_DIM, int BLOCK_SIZE, int NUM_WARPS>
__global__ void paged_attention_kernel(
    const half* __restrict__ q,           // [batch, heads, d]
    const half* __restrict__ kv_pool,     // [num_blocks, 2, block_size, d]
    const int*  __restrict__ page_tables, // [batch, max_pages]
    const int*  __restrict__ seq_lens,    // [batch]
    half*       __restrict__ output,      // [batch, heads, d]
    float*      __restrict__ exp_sums,    // [batch, heads, max_partitions]
    float*      __restrict__ max_logits   // [batch, heads, max_partitions]
) {
    const int seq_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int partition_id = blockIdx.z;  // for split-k parallelism
    const int seq_len = seq_lens[seq_id];

    // Load query into shared memory (once per block)
    __shared__ half q_smem[HEAD_DIM];
    load_query(q, seq_id, head_id, q_smem);

    // Online softmax accumulators (per thread)
    float m = -INFINITY;  // running max
    float d = 0.0f;       // running denominator
    float o[HEAD_DIM / WARP_SIZE] = {0};  // partial output (distributed)

    // Iterate over KV blocks assigned to this partition
    int start_block = partition_id * blocks_per_partition;
    int end_block = min(start_block + blocks_per_partition, num_blocks);

    for (int block_idx = start_block; block_idx < end_block; block_idx++) {
        // PAGE TABLE LOOKUP
        int phys_block = page_tables[seq_id * max_pages + block_idx];

        // Load K block from physical address (coalesced within block)
        half k_tile[BLOCK_SIZE][HEAD_DIM];  // in registers/shared mem
        load_kv_block(kv_pool, phys_block, /*is_key=*/true, k_tile);

        // Compute QK^T for this block
        float scores[BLOCK_SIZE];
        for (int t = 0; t < BLOCK_SIZE; t++) {
            scores[t] = dot_product(q_smem, k_tile[t]) * rsqrt_d;
        }

        // Mask padding tokens in last block
        int valid = min(BLOCK_SIZE, seq_len - block_idx * BLOCK_SIZE);
        for (int t = valid; t < BLOCK_SIZE; t++) {
            scores[t] = -INFINITY;
        }

        // Online softmax update
        float m_local = max_of(scores, BLOCK_SIZE);
        float m_new = fmaxf(m, m_local);
        float correction = expf(m - m_new);

        // Load V block and accumulate
        half v_tile[BLOCK_SIZE][HEAD_DIM];
        load_kv_block(kv_pool, phys_block, /*is_key=*/false, v_tile);

        d = d * correction;
        for (int t = 0; t < BLOCK_SIZE; t++) {
            float p = expf(scores[t] - m_new);
            d += p;
            for (int i = 0; i < HEAD_DIM / WARP_SIZE; i++) {
                o[i] = o[i] * correction + p * __half2float(v_tile[t][...]);
            }
        }
        m = m_new;
    }

    // Write partition results (to be reduced across partitions)
    exp_sums[seq_id * heads * partitions + head_id * partitions + partition_id] = d;
    max_logits[...] = m;
    write_partial_output(output, o, d);
}
```

### 3.5 PagedAttention v1 vs v2

**PagedAttention v1:** Each thread block handles the full sequence for one (batch, head) pair. Limited parallelism when sequence length is short relative to the number of SMs.

**PagedAttention v2 (Split-K):** Splits the KV blocks across multiple thread blocks (partitions) for the same (batch, head) pair. Each partition computes partial attention results, which are then reduced in a second kernel pass. This improves SM utilization for long sequences.

```
v1: Grid = (batch_size, num_heads, 1)
v2: Grid = (batch_size, num_heads, num_partitions)
    + Reduction kernel: Grid = (batch_size, num_heads, 1)
```

Split-K PagedAttention v2 is 2-3x faster than v1 for long sequences (> 4K tokens) due to better GPU utilization.

## 4. Memory Sharing: Copy-on-Write

### 4.1 Shared Prefix

Many LLM serving scenarios involve requests that share a common prefix:

- **System prompt:** All requests to the same chatbot share the system prompt
- **Beam search:** All beam candidates share tokens up to the divergence point
- **Speculative decoding:** Draft and verification paths share the accepted prefix
- **Document QA:** Multiple questions about the same document share the document's KV cache

Without sharing, each request independently stores the prefix's KV cache. With N concurrent requests sharing a 1K-token system prompt on LLaMA-13B:

```
Waste without sharing: N * 1024 * 2 * 40 * 40 * 128 * 2 bytes
                     = N * 838 MB per request for just the prefix
With sharing:          838 MB total (one copy, shared by all N)
```

### 4.2 Copy-on-Write (CoW) Mechanism

PagedAttention implements CoW for shared pages:

```
# Forking: create a new sequence that shares prefix pages
def fork_sequence(parent_seq_id, child_seq_id):
    child_page_table = copy(parent_page_table)  # shallow copy
    for block in child_page_table:
        block_allocator.increment_ref_count(block)

# Writing: when a shared page needs modification
def append_token_to_cache(seq_id, layer, head, new_k, new_v):
    current_block = page_table[seq_id][-1]
    if block_is_full(current_block):
        # Allocate new (unshared) block
        new_block = block_allocator.allocate()
        page_table[seq_id].append(new_block)
        current_block = new_block
    elif ref_count(current_block) > 1:
        # Copy-on-Write: block is shared, must copy before modifying
        new_block = block_allocator.allocate()
        copy_block_data(current_block, new_block)
        block_allocator.decrement_ref_count(current_block)
        page_table[seq_id][-1] = new_block
        current_block = new_block

    # Write new KV to the (now exclusively owned) block
    write_kv(current_block, offset, new_k, new_v)
```

### 4.3 Beam Search Optimization

Beam search is the canonical use case for CoW sharing. Consider beam search with beam width B over a generated sequence of length T:

**Without sharing:**
- Memory: B * T * (KV per token) -- each beam stores the full sequence
- Waste: up to (B-1)/B of KV memory is duplicate (beams share most tokens)

**With CoW sharing:**
- Memory: T * (KV per token) + B * (diverged tokens) * (KV per token)
- Typically 1.5-2x instead of Bx the single-sequence memory

```
Example: B = 4 beams, T = 512 generated tokens
- Beams diverge at tokens: 0, 128, 256, 384
- Shared pages: 512 / 16 = 32 pages (one copy)
- Unique pages per beam: ~8 pages average
- Total: 32 + 4*8 = 64 pages vs 4*32 = 128 pages (2x savings)
```

Measured beam search memory savings: **up to 55% reduction** compared to no-sharing baselines.

## 5. System Integration: vLLM

### 5.1 Architecture Overview

vLLM is the production serving system built around PagedAttention:

```
vLLM Architecture:
+------------------+
|   API Server     |  (FastAPI, OpenAI-compatible)
+------------------+
         |
+------------------+
|   Scheduler      |  (Continuous batching, preemption)
+------------------+
         |
+------------------+
|   Block Manager  |  (Page table management, CoW)
+------------------+
         |
+------------------+
|  Model Executor  |  (GPU, PagedAttention kernels)
+------------------+
```

### 5.2 Continuous Batching

vLLM uses **continuous batching** (also called iteration-level batching) where the batch composition can change at every decode step:

```
Iteration 1: [Req A (decode), Req B (decode), Req C (prefill)]
Iteration 2: [Req A (decode), Req B (done!), Req C (decode), Req D (prefill)]
Iteration 3: [Req A (decode), Req C (decode), Req D (decode)]
```

PagedAttention makes this efficient because:
- New requests only need to allocate as many pages as their current length
- Completed requests immediately free all their pages
- No memory compaction needed (non-contiguous pages eliminate fragmentation)

### 5.3 Preemption and Swapping

When GPU memory is exhausted, vLLM can preempt (pause) in-progress requests:

1. **Swap to CPU:** Move a request's KV pages to CPU memory, freeing GPU pages for other requests. When the preempted request is resumed, pages are swapped back.
2. **Recomputation:** Drop the KV cache entirely and recompute it from the prompt when the request is resumed (saves CPU memory at the cost of recomputation).

PagedAttention simplifies swapping because pages are uniform in size and can be independently swapped.

### 5.4 Scheduling Policies

vLLM implements several scheduling policies:
- **FCFS (First-Come-First-Served):** Default. Serves requests in arrival order.
- **Shortest-Job-First:** Prioritize requests expected to finish sooner (based on output length prediction).
- **Preemptive priority:** High-priority requests can preempt low-priority ones.

The scheduler's key constraint is GPU memory (managed by PagedAttention), not compute. It greedily admits requests as long as there are free pages.

## 6. Performance Analysis

### 6.1 Memory Utilization

Comparison of memory utilization across serving systems:

| System | Internal Frag. | External Frag. | Reservation Waste | Total Waste |
|--------|---------------|----------------|-------------------|-------------|
| HF Transformers | 40% | 20% | 15% | ~75% |
| FasterTransformer | 30% | 15% | 20% | ~65% |
| Orca (continuous batch) | 25% | 15% | 10% | ~50% |
| vLLM (PagedAttention) | < 4% | 0% | 0% | < 4% |

PagedAttention's waste is bounded by B-1 tokens per sequence (last page internal fragmentation). For B = 16 and average sequence length 500: waste = 15/500 = 3%.

### 6.2 Serving Throughput

Throughput comparison (requests per second, higher is better):

**OPT-13B on A100-80GB:**

| System | ShareGPT workload | Alpaca workload |
|--------|-------------------|-----------------|
| HF Transformers | 1.2 req/s | 3.8 req/s |
| FasterTransformer | 2.1 req/s | 5.5 req/s |
| Orca | 2.8 req/s | 6.2 req/s |
| vLLM | 5.2 req/s (2.4x) | 10.1 req/s (2.7x) |

**LLaMA-13B on A100-80GB:**

| System | ShareGPT workload | Alpaca workload |
|--------|-------------------|-----------------|
| HF Transformers | 1.0 req/s | 3.2 req/s |
| FasterTransformer | 1.8 req/s | 4.9 req/s |
| vLLM | 4.5 req/s (2.5x) | 9.2 req/s (2.9x) |

The throughput improvement comes almost entirely from the increased effective batch size enabled by reduced memory waste.

### 6.3 Latency Analysis

PagedAttention has a small latency overhead due to page table indirection:

| Sequence Length | Standard Attention | PagedAttention | Overhead |
|----------------|-------------------|----------------|----------|
| 128 | 0.15 ms | 0.16 ms | +6.7% |
| 512 | 0.42 ms | 0.44 ms | +4.8% |
| 2048 | 1.51 ms | 1.56 ms | +3.3% |
| 8192 | 5.83 ms | 5.97 ms | +2.4% |

Overhead decreases with sequence length because the cost of the indirection is amortized over more tokens. In all cases, the overhead is < 7% per token, which is negligible compared to the throughput gains.

### 6.4 Beam Search Performance

Beam search with CoW sharing:

| Beam Width | Without Sharing | With CoW | Memory Savings |
|------------|----------------|----------|----------------|
| 2 | 2.0x base | 1.3x base | 35% |
| 4 | 4.0x base | 2.1x base | 48% |
| 8 | 8.0x base | 3.8x base | 53% |
| 16 | 16.0x base | 7.2x base | 55% |

Maximum beam width on A100-80GB for LLaMA-13B (2048 tokens):
- Without sharing: beam 4
- With CoW sharing: beam 16 (4x improvement)

### 6.5 Prefix Sharing Performance

For a shared system prompt of 1024 tokens with multiple concurrent requests:

| Concurrent Requests | Without Sharing | With Sharing | Memory Savings |
|--------------------|----------------|--------------|----------------|
| 8 | 6.7 GB | 1.8 GB | 73% |
| 16 | 13.4 GB | 2.8 GB | 79% |
| 32 | 26.8 GB | 4.7 GB | 82% |
| 64 | 53.6 GB | 8.5 GB | 84% |

Prefix sharing becomes more valuable as the number of concurrent requests increases.

## 7. Kernel Implementation Details

### 7.1 Memory Access Patterns

The page table indirection changes the memory access pattern:

**Standard (contiguous):**
```
KV address for token t = kv_base + seq_id * max_len * stride + t * stride
```
Single arithmetic operation, perfectly predictable by hardware prefetcher.

**PagedAttention (indirect):**
```
page_idx = t / B
offset = t % B
phys_block = page_table[seq_id][page_idx]  // one extra load
KV address = kv_pool_base + phys_block * block_stride + offset * token_stride
```
Requires one extra memory load (page table lookup), but:
- Page table is small and fits in L2 cache after first access
- Within each block, access is still contiguous (offset increments sequentially)
- The hardware prefetcher can predict the within-block pattern

### 7.2 Block Size Tuning

Effect of block size B on performance:

| Block Size B | Internal Frag. | Page Table Size | Kernel Performance |
|-------------|---------------|-----------------|-------------------|
| 1 | 0% | Large (N entries) | Poor (no coalescing benefit) |
| 4 | ~1.5% | Moderate | Good |
| 8 | ~3% | Moderate | Better |
| 16 | ~6% | Small | Best (typical choice) |
| 32 | ~12% | Very small | Similar (diminishing returns) |
| 64 | ~24% | Tiny | Worse (too much frag.) |

B = 16 is the default in vLLM, providing a good balance between fragmentation and kernel efficiency. The kernel loads B tokens at once, computing B QK^T dot products in parallel within the block.

### 7.3 Integration with FlashAttention

PagedAttention is designed for the **decode phase** (single-token query attending to KV cache). For the **prefill phase** (batch of tokens with known KV), standard FlashAttention is used since:
- During prefill, the KV cache for the prompt is computed all at once (no incremental allocation needed)
- FlashAttention's tiled algorithm for batch Q is more efficient than PagedAttention's single-query design

The typical serving pipeline:
1. **Prefill:** Standard FlashAttention kernel (Q, K, V all present)
2. **Allocate KV pages:** Store the computed KV cache in pages
3. **Decode:** PagedAttention kernel (single Q, paged KV cache) for each generated token

### 7.4 Multi-GPU Considerations

For tensor-parallel serving across multiple GPUs:
- Each GPU stores a shard of the KV cache (different heads or head dimensions)
- Page tables are replicated on all GPUs (small overhead)
- Block allocation is coordinated (same logical block IDs across GPUs)
- The PagedAttention kernel runs independently on each GPU's shard

For pipeline-parallel serving:
- Each pipeline stage manages its own KV cache pages independently
- Page tables are per-stage
- Memory pressure may differ across stages (later stages generate output longer, so they accumulate more KV cache)

## 8. Extensions and Variations

### 8.1 Prefix Caching (Automatic Prefix Sharing)

vLLM's prefix caching automatically detects and shares common prefixes across requests without explicit beam search:

```
Request 1: "You are a helpful assistant. What is 2+2?"
Request 2: "You are a helpful assistant. Write a poem."

# Automatic prefix detection:
Common prefix: "You are a helpful assistant." (shared KV pages)
Unique suffixes: different KV pages per request
```

Implementation: hash-based page deduplication. When a new page's content matches an existing page (same token IDs, same position), point to the existing page instead of allocating a new one.

### 8.2 Speculative Decoding Integration

PagedAttention naturally supports speculative decoding:

1. **Draft model** generates k candidate tokens, allocating k new pages
2. **Verification model** checks all k tokens in parallel
3. If verified: keep the pages. If rejected at position i: free pages i+1 through k (instant deallocation).

### 8.3 Chunked Prefill

For very long prompts, vLLM splits the prefill into chunks that interleave with decode steps:

```
# Standard: prefill 10K tokens blocks decode for all sequences
# Chunked: prefill 512 tokens, then do decode steps, then prefill next 512, ...
```

PagedAttention makes this seamless because the partial prefill's KV cache is stored in pages that can be extended in later chunks without reallocation.

### 8.4 KV Cache Quantization

PagedAttention is compatible with KV cache quantization (INT8, INT4, FP8):
- Each page stores quantized KV data (smaller page size in bytes)
- The attention kernel dequantizes on-the-fly within the inner loop
- 2-4x more pages fit in the same GPU memory, proportionally increasing batch size

### 8.5 Multi-Modal Attention

For vision-language models with image tokens, PagedAttention treats image KV cache the same as text KV cache. Image tokens are typically many (e.g., 576 for ViT-L/14 at 224px) and shared across all queries about the same image.

## 9. Comparison with Other Memory Management Approaches

### 9.1 Static Allocation (HuggingFace Transformers)

```
# Pre-allocate max_len for every sequence
kv_cache = torch.zeros(batch, layers, 2, max_len, heads, d)
```
- Simple, no indirection overhead
- Massive waste (max_len is typically 2048-8192, average generation is 100-500)

### 9.2 Dynamic Reallocation (FasterTransformer)

```
# Start with small buffer, double when full
if kv_cache_full:
    new_cache = allocate(2 * current_size)
    copy(kv_cache, new_cache)
    free(kv_cache)
    kv_cache = new_cache
```
- Less waste than static, but copy is expensive
- External fragmentation from variable-size frees

### 9.3 Memory Pool (TensorRT-LLM)

```
# Pre-allocate pool, variable-size allocation from pool
kv_cache = pool.allocate(estimated_len)
```
- Better than static but still wastes on over-estimation
- External fragmentation within pool

### 9.4 PagedAttention (vLLM)

```
# Fixed-size pages, non-contiguous allocation
page = pool.allocate_one_page()  # always same size
page_table[seq].append(page)
```
- Near-zero fragmentation
- O(1) allocation and deallocation
- Natural sharing via reference counting

## 10. Adoption and Ecosystem Impact

PagedAttention has become the de facto standard for LLM KV cache management:

| System | PagedAttention Support | Notes |
|--------|----------------------|-------|
| vLLM | Native (original) | Reference implementation |
| TensorRT-LLM | Adopted in v0.5+ | NVIDIA's serving framework |
| SGLang | Adopted | Stanford/Berkeley |
| LMDeploy | Adopted | Alibaba/SenseTime |
| TGI (HuggingFace) | Integrated via vLLM | HuggingFace serving |
| MLC LLM | Adopted variant | Universal deployment |
| llama.cpp | Partial (block-based KV) | CPU inference |

The concept has become so fundamental that it is often assumed rather than explicitly cited in newer serving systems.

## 11. Limitations and Future Directions

### 11.1 Current Limitations

1. **Block size is fixed:** A single B value must work for all sequence lengths and workloads. Adaptive block sizes could improve efficiency but complicate the memory manager.

2. **Page table overhead:** While small, the page table requires CPU memory and CPU-GPU transfer. For very large batch sizes (1000+), the page table itself can become non-trivial.

3. **Decode-only optimization:** PagedAttention is designed for the decode phase. Prefill still uses standard contiguous KV storage with FlashAttention.

4. **Single-GPU focus:** While compatible with tensor parallelism, the cross-GPU coordination adds complexity. Distributed KV cache management (across hosts) is an active research area.

5. **No hardware support:** PagedAttention's indirection is implemented in software. Hardware-supported page table (like CPU MMU) for GPU memory could eliminate the overhead entirely.

### 11.2 Future Directions

1. **Hardware-accelerated indirection:** GPU vendors could add lightweight page-table support to the memory controller, making PagedAttention essentially free.

2. **Hierarchical paging:** Multiple page sizes (like huge pages in OS) to reduce page table size for long sequences while maintaining fine granularity for short ones.

3. **Distributed KV cache:** Extend PagedAttention across multiple hosts for disaggregated serving, where prefill and decode happen on different machines.

4. **Learned allocation policies:** Use ML to predict sequence output length and optimize page allocation accordingly.

5. **Cache eviction policies:** When all pages are exhausted, intelligent eviction (like LRU or attention-score-based) could improve quality compared to simple preemption.

## 12. References

1. Woosuk Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023, arXiv:2309.06180)
2. Gyeong-In Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022) -- continuous batching
3. Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)
4. Tri Dao, "FlashDecoding" (2023) -- split-K attention for long sequences
5. vLLM: https://github.com/vllm-project/vllm
6. vLLM Blog: https://blog.vllm.ai/2023/06/20/vllm.html
7. Yaniv Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)
8. NVIDIA TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
