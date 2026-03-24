---
skill_name: PagedAttention
description: Virtual memory-inspired attention algorithm that manages KV cache as fixed-size pages with page table indirection, eliminating memory fragmentation and enabling efficient dynamic memory sharing for LLM serving.
level: L1 - Algorithmic/Mathematical Stage Level
target_hardware: NVIDIA GPUs (Ampere A100 and newer), AMD GPUs (MI250X, MI300X)
relevance: When optimizing LLM inference serving systems that must handle variable-length sequences, batched requests, and beam search with minimal KV cache memory waste.
---

# PagedAttention

## What It Is
PagedAttention, introduced by Kwon et al. (UC Berkeley, 2023) and implemented in the vLLM serving system, applies operating system virtual memory concepts to the GPU KV cache used in autoregressive LLM inference. Instead of pre-allocating a contiguous block of GPU memory for each sequence's KV cache (which wastes memory due to fragmentation and over-allocation for unknown future lengths), PagedAttention divides the KV cache into fixed-size **pages** (blocks of, e.g., 16 tokens) that can be allocated non-contiguously in physical GPU memory and mapped via a **page table**. The attention kernel is modified to perform block-level attention computation that follows page table indirection to gather the correct KV blocks for each sequence. This eliminates internal and external fragmentation (reducing waste from 60-80% to < 4%), enables efficient memory sharing across sequences (e.g., shared prefixes in beam search), and dramatically increases the effective batch size and throughput of LLM serving.

## Key Concepts
- **KV Cache as Pages:** The KV cache for each attention layer is divided into fixed-size blocks (pages) of B tokens each (typically B = 16). Each page stores B key vectors and B value vectors for one attention head/layer.
- **Page Table:** A per-sequence mapping from logical block indices to physical block addresses in GPU memory, analogous to OS virtual-to-physical page tables. The page table is stored in CPU memory and passed to the GPU kernel.
- **Non-Contiguous Storage:** Physical pages for a single sequence need not be adjacent in GPU memory. Pages are allocated on demand as the sequence grows, and freed when the sequence completes.
- **Block-Level Attention Kernel:** The attention kernel iterates over the logical blocks of a sequence, uses the page table to look up the physical address of each KV block, loads that block, and computes partial attention scores. Online softmax accumulation proceeds across blocks.
- **Copy-on-Write (CoW) Sharing:** Multiple sequences that share a common prefix (e.g., system prompt, beam search candidates) can share physical KV pages via reference counting. When a sequence diverges, only the diverging page is copied (CoW), reducing memory usage by up to 55% for beam search.
- **Near-Zero Waste:** Memory waste is bounded by at most one page per sequence (the last, partially filled page), achieving < 4% fragmentation compared to 60-80% in naive pre-allocation schemes.

## Algorithm / Pseudo-code
```
# PagedAttention Forward Pass (single query token decode step)
# Inputs: q (1 x d) -- query for current token
#         page_table[seq_id] -- array of physical block addresses
#         kv_cache -- global pool of physical KV blocks on GPU
#         seq_len -- current sequence length for this request
# Parameters: block_size B (e.g., 16), num_heads H, head_dim d

# Initialize online softmax accumulators
m = -inf          # running max
d_acc = 0.0       # running softmax denominator
o = zeros(d)      # running output accumulator

num_blocks = ceil(seq_len / B)

for block_idx in 0 .. num_blocks - 1:
    # Step 1: Page table lookup -- translate logical to physical block
    physical_addr = page_table[seq_id][block_idx]

    # Step 2: Load KV block from (potentially non-contiguous) GPU memory
    k_block = kv_cache.keys[physical_addr]    # (B x d) or partial
    v_block = kv_cache.values[physical_addr]  # (B x d) or partial

    # Handle last block (may be partially filled)
    valid_tokens = min(B, seq_len - block_idx * B)
    k_block = k_block[:valid_tokens]
    v_block = v_block[:valid_tokens]

    # Step 3: Compute attention scores for this block
    scores = q @ k_block^T / sqrt(d)   # (1 x valid_tokens)

    # Step 4: Online softmax update
    m_block = max(scores)
    m_new = max(m, m_block)

    # Rescale previous accumulator
    correction = exp(m - m_new)
    d_acc = d_acc * correction + sum(exp(scores - m_new))
    o = o * correction + exp(scores - m_new) @ v_block

    m = m_new

# Step 5: Final normalization
O = o / d_acc

# --- Memory Management (CPU-side) ---

# Allocation: when a new token is generated
if current_block_is_full(seq_id):
    new_physical_block = allocate_free_block()
    page_table[seq_id].append(new_physical_block)
append_kv_to_current_block(seq_id, new_k, new_v)

# Sharing: for beam search fork
def fork_sequence(parent_id, child_id):
    page_table[child_id] = copy(page_table[parent_id])
    for block in page_table[child_id]:
        increment_ref_count(block)

# Copy-on-Write: when modifying a shared block
def write_kv(seq_id, block_idx, new_k, new_v):
    physical_block = page_table[seq_id][block_idx]
    if ref_count(physical_block) > 1:
        new_block = allocate_free_block()
        copy_data(physical_block, new_block)
        decrement_ref_count(physical_block)
        page_table[seq_id][block_idx] = new_block
        physical_block = new_block
    write_to_block(physical_block, new_k, new_v)
```

## When to Use
- LLM inference serving with variable-length requests where pre-allocating max-length KV cache per sequence wastes memory
- Batched inference where maximizing the number of concurrent sequences is critical for throughput
- Beam search or speculative decoding where multiple sequence candidates share common prefixes
- Systems that need to handle dynamic request arrival/completion (online serving) with efficient memory recycling
- Any scenario where KV cache memory fragmentation is limiting effective batch size or requiring over-provisioned GPU memory

## When NOT to Use
- Training workloads where the full sequence is known in advance and KV cache is not used (standard FlashAttention is more appropriate)
- Prefill phase of inference where all tokens are processed at once (PagedAttention is primarily designed for the decode phase; prefill uses standard dense attention)
- Very short sequences or small batch sizes where fragmentation is negligible and the page table indirection adds unnecessary overhead
- When using models that don't use KV cache (e.g., encoder-only models like BERT, or SSM-based models like Mamba)
- Latency-critical single-sequence inference where the indirection overhead of page table lookups may add measurable latency

## Code / Pseudo-code

### Python: Block Allocator for Page Management

```python
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

### CUDA: PagedAttention v2 Kernel (Split-K)

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

## Key Takeaways
- PagedAttention reduces KV cache memory waste from 60-80% (naive pre-allocation) to < 4% (only last-page internal fragmentation), enabling 2-4x more concurrent sequences and proportional throughput gains
- The vLLM system built on PagedAttention achieves 2-4x higher serving throughput than HuggingFace Transformers and 2.2x higher than FasterTransformer on real LLM serving workloads (OPT-13B, LLaMA-13B)
- Copy-on-Write sharing reduces beam search memory usage by up to 55%, enabling larger beam widths within the same GPU memory budget
- The block-level attention kernel integrates naturally with online softmax (FlashAttention-style), so PagedAttention benefits from all FA tiling optimizations
- Page table indirection adds minimal latency overhead (< 5%) because the lookup is a simple array index and the KV blocks are still accessed with coalesced memory patterns within each page
- PagedAttention has become the de facto standard for production LLM serving, adopted by vLLM, TensorRT-LLM, SGLang, and other major serving frameworks

## References
- Original Paper: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023, arXiv:2309.06180)
- vLLM Project: https://github.com/vllm-project/vllm
- Blog: https://blog.vllm.ai/2023/06/20/vllm.html
- Related: FlashDecoding (Dao et al., 2023), Continuous Batching (Yu et al., OSDI 2022)
