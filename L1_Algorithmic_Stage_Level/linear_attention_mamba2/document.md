# Linear Attention and Mamba-2: The Structured State Space Duality

**Source:** Tri Dao, Albert Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (ICML 2024, arXiv:2405.21060)

## 1. Introduction

The Transformer's attention mechanism and State Space Models (SSMs) have evolved as competing paradigms for sequence modeling. Attention provides powerful, content-dependent token-to-token interactions but scales quadratically. SSMs provide efficient linear-time recurrence but with fixed (or limited) data-dependent dynamics. The Mamba-2 paper establishes a deep mathematical connection between these paradigms through the framework of **structured matrices**, showing they are different computational views of the same underlying operation.

This duality is not merely theoretical. It yields a practical algorithm called **Structured State Space Duality (SSD)** that decomposes sequence computation into:
- **Intra-block:** Dense matrix multiplication (attention-like, leveraging Tensor Cores)
- **Inter-block:** Recurrent state propagation (SSM-like, compact state)

SSD achieves 2-8x speedup over Mamba-1's selective scan kernel while matching or exceeding its quality, and provides a principled framework for designing hybrid attention-SSM architectures.

## 2. Background

### 2.1 Standard Softmax Attention

The standard causal attention mechanism computes:

```
For position i:
    O[i] = sum_{j <= i} softmax_j(Q[i] K[j]^T / sqrt(d)) * V[j]
```

This can be written as a matrix equation:

```
O = (softmax(Q K^T / sqrt(d)) ⊙ L) V
```

where L is a lower-triangular causal mask (L[i,j] = 1 if j <= i, 0 otherwise) and ⊙ denotes element-wise multiplication (masking). The key matrix M = softmax(QK^T/sqrt(d)) ⊙ L is an (N x N) lower-triangular matrix.

Computation: O(N^2 d) FLOPs, O(N^2) memory (for M).

### 2.2 Linear Attention

Linear attention (Katharopoulos et al., 2020) replaces the softmax with a kernel feature map phi:

```
O[i] = sum_{j <= i} phi(Q[i])^T phi(K[j]) * V[j]
     = phi(Q[i])^T * sum_{j <= i} phi(K[j]) V[j]^T
     = phi(Q[i])^T * S[i]
```

where S[i] = sum_{j <= i} phi(K[j]) V[j]^T is a running "state matrix" of size (d x d).

The recurrent form:
```
S[0] = 0
S[t] = S[t-1] + phi(K[t]) V[t]^T    # (d x d) state update
O[t] = phi(Q[t])^T S[t]               # (1 x d) output
```

Computation: O(N d^2) FLOPs (linear in N), O(d^2) memory for state.

The corresponding attention matrix is:
```
M[i,j] = phi(Q[i])^T phi(K[j])  for j <= i, else 0
```

This is a lower-triangular matrix where each entry is a simple dot product (no softmax). The key structural property: **M has rank at most d**, since M = (phi(Q) phi(K)^T) ⊙ L and the unmasked part phi(Q) phi(K)^T has rank d.

### 2.3 Structured State Space Models (S4/S6/Mamba)

A discrete-time linear state space model:

```
x[t] = A x[t-1] + B[t] u[t]
y[t] = C[t]^T x[t]
```

where:
- x[t] is the hidden state, dimension P (state size)
- u[t] is the input, dimension D (model dimension)
- y[t] is the output, dimension D
- A is the state transition matrix, dimension (P x P) -- typically **diagonal**
- B[t] is the input projection, dimension (P x D) -- **data-dependent** in Mamba (selective)
- C[t] is the output projection, dimension (P x D) -- **data-dependent** in Mamba

In Mamba's selective scan, B[t] and C[t] are functions of the input x[t] (i.e., content-dependent), while A is a learned diagonal matrix (content-independent but with learned decay rates).

Unrolling the recurrence:
```
x[t] = A^t x[0] + sum_{j=1}^{t} A^{t-j} B[j] u[j]
y[t] = C[t]^T x[t] = C[t]^T A^t x[0] + sum_{j=1}^{t} C[t]^T A^{t-j} B[j] u[j]
```

The output can be written as:
```
y[t] = sum_{j=1}^{t} M[t,j] u[j]
where M[t,j] = C[t]^T A^{t-j} B[j]  (for t >= j)
```

This M is the **SSM kernel matrix**.

### 2.4 The Key Observation: Both Are Structured Matrices

Both linear attention and SSMs compute:
```
Y = M * U
```
where M is an (N x N) lower-triangular matrix and U is the input sequence. The difference is the **structure** of M:

| Model | M[i,j] (for i >= j) | Structure |
|-------|---------------------|-----------|
| Linear attention | Q[i]^T K[j] | Rank-d (outer product) |
| SSM (diagonal A) | C[i]^T diag(a^{i-j}) B[j] | Semiseparable |
| SSM (general A) | C[i]^T A^{i-j} B[j] | Semiseparable |
| Softmax attention | softmax(QK^T)[i,j] | Unstructured (dense) |

The crucial insight: **SSM matrices are semiseparable**, and semiseparable matrices generalize the low-rank structure of linear attention.

## 3. Semiseparable Matrices: The Unifying Framework

### 3.1 Definition

An (N x N) lower-triangular matrix M is **semiseparable of rank P** if every submatrix of its strictly lower-triangular part has rank at most P.

Equivalently, M has a representation:
```
M[i,j] = C[i]^T (A[i] A[i-1] ... A[j+1]) B[j]    for i > j
M[i,i] = C[i]^T B[i]                                 for i = i (diagonal)
M[i,j] = 0                                           for i < j (upper triangle)
```

where A[t] are (P x P) matrices, B[t] are (P x 1) vectors (or P x D for multi-input), and C[t] are (P x 1) vectors (or P x D for multi-output). This is exactly the SSM form.

### 3.2 Special Cases

**Rank-1 semiseparable (scalar A):**
```
M[i,j] = c[i] * (a[i] * a[i-1] * ... * a[j+1]) * b[j]
```
This is a scalar SSM with 1D state. The product of a's is an exponential decay.

**Low-rank without decay (A = I):**
```
M[i,j] = C[i]^T B[j] = sum_{p=1}^{P} c_p[i] * b_p[j]
```
This is exactly linear attention with P-dimensional features.

**Low-rank with data-dependent decay (diagonal A):**
```
M[i,j] = C[i]^T diag(a[i:j+1]) B[j]
```
This is Mamba's selective scan -- linear attention with multiplicative decay.

### 3.3 Hierarchy of Models

```
Softmax Attention (unstructured M, rank N)
    |
    | restriction to structured M
    v
Semiseparable (rank P, with decay)  <-- SSMs (Mamba, S4, S6)
    |
    | set A = I (no decay)
    v
Low-rank (rank P, no decay)  <-- Linear Attention
    |
    | restrict P = 1
    v
Rank-1 (scalar gating)  <-- Simple RNNs (LSTM gate)
```

This hierarchy shows that SSMs are **strictly more expressive** than linear attention (they include decay) while being **strictly less expressive** than softmax attention (they have rank-P structure).

## 4. The SSD Algorithm

### 4.1 Motivation: Limitations of Existing Algorithms

**Attention computation (parallel):** O(N^2 d) FLOPs, highly parallelizable (matrix multiplication), excellent Tensor Core utilization. But quadratic in N.

**Recurrent computation (sequential):** O(N d P) FLOPs, sequential (each step depends on the previous), poor Tensor Core utilization (small matrix operations). Linear in N but slow constant.

**Mamba-1's selective scan:** Uses a specialized CUDA kernel with memory-level parallelism (loading multiple channels simultaneously) and work decomposition. Achieves good throughput but **does not use Tensor Cores** -- the bottleneck is that each step's matrix multiplication (A @ x + B @ u) is too small to fill the Tensor Cores.

### 4.2 Block Decomposition

The SSD algorithm resolves this by decomposing the sequence into blocks of size B and using different computation strategies within and between blocks:

**Step 1: Partition the sequence**
```
Blocks: [1..B], [B+1..2B], ..., [(N/B-1)*B+1 .. N]
```

**Step 2: Intra-block computation (attention-like)**

Within each block, compute the full B x B attention-like matrix and apply it:

```
For block b (positions (b-1)*B+1 to b*B):
    Q_b = Q[(b-1)*B+1 : b*B]    # (B x d)
    K_b = K[(b-1)*B+1 : b*B]    # (B x d)
    V_b = V[(b-1)*B+1 : b*B]    # (B x d)

    # Compute intra-block attention matrix (with decay)
    # M_intra[i,j] = C[i]^T diag(A[i:j+1]) B[j]  for i >= j within block
    # For diagonal A, this simplifies to element-wise products

    # In practice (scalar A per head):
    # D[i,j] = prod(a[s] for s in j+1..i) -- cumulative decay within block
    D = compute_decay_matrix(A_block)  # (B x B), lower triangular

    M_intra = (Q_b @ K_b^T) * D       # (B x B), element-wise product
    O_intra = M_intra @ V_b            # (B x d)
```

This is a dense matrix multiplication -- **perfect for Tensor Cores**. Cost: O(B^2 d) per block, O(N B d) total.

**Step 3: Inter-block computation (recurrence)**

Between blocks, propagate a state summary:

```
h[0] = 0  # (d x P) initial state

For block b:
    # State carries information from all previous blocks
    O_inter = Q_b @ h[b-1]   # (B x d), contribution from past blocks

    # Update state with current block's contribution
    # Weighted K^T V with decay from block boundary
    Kbar_b = D_lower_b^T @ K_b   # Decay-weighted keys
    h[b] = decay_b * h[b-1] + Kbar_b^T @ V_b   # (d x P) state update
```

This involves (d x P) matrix operations -- reasonable for Tensor Cores when P (state size) is not too small. Cost: O(N d P / B) total.

**Step 4: Combine**
```
O = O_intra + O_inter  # (N x d)
```

### 4.3 Total Complexity

```
Intra-block: O(N * B * d)        -- B^2 matmul per block, N/B blocks
Inter-block: O(N * d * P / B)    -- d*P state update per block, N/B blocks
Total:       O(N * B * d + N * d * P / B)
```

Optimizing over B:
```
d(Total)/dB = N*d - N*d*P/B^2 = 0
=> B_opt = sqrt(P)
=> Total_opt = O(N * d * sqrt(P))
```

For Mamba-2 with P = d (state dimension = head dimension):
```
Total = O(N * d * sqrt(d))  (with optimal B)
```

In practice, B is chosen to match Tensor Core tile sizes (64, 128, 256) rather than strictly optimizing for minimal FLOPs. The wall-clock time improvement from Tensor Core utilization typically outweighs the theoretical FLOP count.

### 4.4 Detailed SSD Algorithm

```
# SSD Forward Pass
# Inputs: Q, K, V each (N x d), A (N,) -- per-position scalar decay
# State dimension P = d (for Mamba-2's multi-head SSM)
# Block size B

def ssd_forward(Q, K, V, A, B_size):
    N, d = Q.shape
    num_blocks = N // B_size

    # Precompute cumulative decay factors
    # For diagonal scalar A: decay[i] = exp(-delta[i] * a) where a is learned
    log_decay = compute_log_decay(A)  # (N,)

    O = zeros(N, d)
    h = zeros(d, d)  # recurrent state (using P = d)

    for b in range(num_blocks):
        start = b * B_size
        end = start + B_size

        Q_b = Q[start:end]  # (B x d)
        K_b = K[start:end]  # (B x d)
        V_b = V[start:end]  # (B x d)
        decay_b = log_decay[start:end]  # (B,)

        # ---- Intra-block: attention-like matmul ----
        # Compute decay matrix D[i,j] = exp(sum of log_decay from j+1 to i)
        # For i >= j within the block (lower triangular)
        cum_decay = cumsum(decay_b)  # (B,)
        D = exp(cum_decay.unsqueeze(0) - cum_decay.unsqueeze(1))  # (B x B)
        D = tril(D)  # lower triangular mask

        # Attention-like computation
        M = (Q_b @ K_b.T) * D  # (B x B), Tensor Core matmul + pointwise
        O_intra = M @ V_b       # (B x d), Tensor Core matmul

        # ---- Inter-block: state contribution ----
        # Contribution from previous blocks' state
        # Decay the state from block boundary to each position in this block
        decay_to_pos = exp(cum_decay)  # (B,) decay from block start
        O_inter = (Q_b * decay_to_pos.unsqueeze(1)) @ h  # (B x d)

        # ---- Update state for next block ----
        # Decay old state to end of current block
        block_total_decay = exp(cum_decay[-1])
        h = h * block_total_decay

        # Add current block's contribution
        decay_from_pos = exp(cum_decay[-1] - cum_decay)  # (B,) decay to block end
        K_weighted = K_b * decay_from_pos.unsqueeze(1)  # (B x d)
        h = h + K_weighted.T @ V_b  # (d x d) outer product sum

        # ---- Combine ----
        O[start:end] = O_intra + O_inter

    return O
```

### 4.5 Backward Pass

The backward pass uses a similar block decomposition but processes blocks in reverse:

```
# SSD Backward Pass (sketch)
# Given: dO (N x d), need: dQ, dK, dV, dA

dh = zeros(d, d)  # backward state gradient

for b in range(num_blocks - 1, -1, -1):
    start = b * B_size
    end = start + B_size

    # Recompute forward quantities (FlashAttention-style)
    M = recompute_intra_block_matrix(Q_b, K_b, decay_b)

    # Intra-block gradients (standard attention backward)
    dM = dO_b @ V_b.T          # (B x B)
    dV_b += M.T @ dO_b         # (B x d)
    dQ_b += (dM * D) @ K_b     # (B x d)
    dK_b += (dM * D).T @ Q_b   # (B x d)

    # Inter-block gradients
    dh += K_b.T @ (dO_b * decay_to_pos)  # accumulate state gradient
    dQ_b += (dO_b @ h.T) * decay_to_pos  # grad through O_inter
    # ... (additional terms for decay gradients)

    # Propagate state gradient backward
    dh = dh * block_total_decay
```

## 5. Mamba-2 Architecture

### 5.1 Multi-Head SSM

Mamba-2 introduces a **multi-head** structure analogous to multi-head attention:

```
Mamba-2 Layer:
    # Input: x (N x D)
    # Project to multi-head form
    Q = x @ W_Q  # (N x H x d)   -- "C" in SSM notation
    K = x @ W_K  # (N x H x d)   -- "B" in SSM notation
    V = x @ W_V  # (N x H x d)   -- "u" in SSM notation (input)
    A = x @ W_A  # (N x H)       -- scalar decay per position per head

    # Apply SSD per head
    for h in range(H):
        O[:, h, :] = ssd_forward(Q[:, h], K[:, h], V[:, h], A[:, h])

    # Output projection
    y = O.reshape(N, D) @ W_O
```

Design choices:
- **Head dimension d:** Typically 64-128 (matches Tensor Core tile sizes)
- **State dimension P:** Set equal to d (P = d), so the state h is (d x d)
- **Shared A:** The decay matrix A can be shared across heads (reducing parameters)
- **Separate B, C:** Input (K) and output (Q) projections are per-head (analogous to QKV in attention)

### 5.2 Comparison with Mamba-1

| Feature | Mamba-1 (S6) | Mamba-2 (SSD) |
|---------|-------------|---------------|
| Core algorithm | Selective scan (custom CUDA) | Block SSD (matmul-based) |
| Tensor Core usage | No | Yes (intra-block matmul) |
| State dimension P | 16 | 64-128 (= d) |
| Multi-head | No (single large state) | Yes (H heads of size d) |
| Head dimension | N/A | 64-128 |
| Sequence mixing | Per-channel scan | Per-head block attention |
| Speed (A100, N=2K) | 1x (baseline) | 2-8x faster |
| Quality | Strong | Equal or better |

### 5.3 Connection to GQA/MQA

The multi-head SSM structure in Mamba-2 enables GQA-like sharing:

```
# Multi-Query SSM: all heads share K, V; only Q (output proj) varies
Q = per_head_projection(x)  # H different projections
K = shared_projection(x)     # 1 shared projection
V = shared_projection(x)     # 1 shared projection
A = shared_or_per_head(x)    # flexible

# Grouped-Query SSM: G groups of heads share K, V
# Group g has H/G heads sharing one K, V projection
```

This reduces parameter count and KV memory, just as in attention.

## 6. When Linear Attention Suffices vs. Softmax

### 6.1 Theoretical Expressiveness Gap

Softmax attention can represent arbitrary (N x N) attention patterns. Linear attention (and SSMs) are restricted to rank-P patterns. This gap matters when the task requires:

1. **Sharp, sparse attention:** Softmax can produce near-one-hot attention (attending to exactly one token). Linear attention's rank constraint prevents this -- it can only approximate sharp attention with high rank.

2. **Exact copying:** Tasks like "repeat the input" require attending to exact positions. Softmax attention solves this trivially. Linear attention struggles because the state S = sum_j K[j] V[j]^T mixes all past tokens into a fixed-size (d x d) matrix.

3. **Associative recall:** "What is the value associated with key X?" requires precise retrieval from a compressed state. Softmax attention retrieves perfectly; linear attention retrieves approximately.

### 6.2 Empirical Results

| Task | Softmax Attention | Linear Attention | Mamba-2 (SSD) |
|------|-------------------|-----------------|---------------|
| Language modeling (PPL) | 10.5 (3B) | 11.8 (+12%) | 10.7 (+2%) |
| MQAR (associative recall) | 99.2% | 72.3% | 94.8% |
| Copy task (length 512) | 100% | 45.2% | 89.7% |
| Long-range arena | 86.3% | 84.1% | 87.2% |
| Code generation | 42.1% | 35.8% | 40.5% |

Key observations:
- **Mamba-2 >> vanilla linear attention:** The data-dependent decay (A matrix) is critical. Without it, linear attention significantly underperforms.
- **Mamba-2 ~ softmax attention on most tasks:** For language modeling and long-range tasks, SSD matches or approaches softmax attention quality.
- **Gap remains on retrieval-heavy tasks:** Exact copying and associative recall still favor softmax attention, though the gap narrows with larger state dimension P.

### 6.3 Design Guidelines

| Scenario | Recommended | Rationale |
|----------|------------|-----------|
| Long sequence, smooth patterns | SSD/Mamba-2 | Linear cost, sufficient expressiveness |
| Short sequence (< 2K) | Softmax FA | Quadratic cost is cheap, maximum expressiveness |
| Retrieval-heavy (RAG, QA) | Softmax FA or hybrid | Need sharp attention for precise retrieval |
| Continuous signals (audio, DNA) | SSD/Mamba-2 | Recurrent state is natural for signals |
| Latency-critical decode | SSD/Mamba-2 | O(1) per step (recurrent), no KV cache growth |
| Throughput-critical prefill | Softmax FA | Highly parallelizable, mature kernel optimizations |
| Hybrid (best of both) | Interleave layers | Some layers use attention, others use SSD |

## 7. Kernel Implementation

### 7.1 SSD Kernel Structure

The SSD kernel has three main phases, each mapping to efficient GPU operations:

```
Phase 1: Intra-block matmul (Tensor Core intensive)
    For each block b:
        M_b = Q_b @ K_b^T    # (B x B) matmul, GEMM on Tensor Cores
        M_b = M_b * D_b      # pointwise multiply with decay mask
        O_intra_b = M_b @ V_b  # (B x d) matmul

Phase 2: State update (sequential across blocks, parallel within)
    For each block b (sequentially):
        h[b] = decay * h[b-1] + K_b^T @ V_b  # (d x d) outer product

Phase 3: Inter-block output (Tensor Core, parallel across blocks)
    For each block b:
        O_inter_b = Q_b @ h[b]   # (B x d) matmul
```

### 7.2 Memory Access Pattern

```
Phase 1 (intra-block):
    Read: Q_b, K_b, V_b (3 * B * d elements per block)
    Write: O_intra_b (B * d elements per block)
    Arithmetic intensity: O(B * d) / O(B * d) = O(1) -- memory-bound for small B
    For B = 128, d = 128: ~50 FLOPS/byte -- compute-bound, good for Tensor Cores

Phase 2 (state update):
    Read: K_b, V_b (2 * B * d per block)
    Write: h[b] (d * d per block)
    This is the sequential bottleneck -- but d*d is small (128*128 = 16K elements)

Phase 3 (inter-block):
    Read: Q_b, h[b] (B*d + d*d per block)
    Write: O_inter_b (B * d per block)
    Arithmetic intensity: O(B * d * d) / O((B+d) * d) ~ O(d) when B ~ d
```

### 7.3 Fused Kernel

In practice, all three phases are fused into a single kernel:

```cuda
// SSD fused kernel (simplified)
__global__ void ssd_kernel(
    const half* Q, const half* K, const half* V,
    const float* log_decay,
    half* O,
    int N, int d, int B
) {
    // Each thread block handles one sequence block
    int block_id = blockIdx.x;
    int head_id = blockIdx.y;

    __shared__ half Q_smem[B][d];
    __shared__ half K_smem[B][d];
    __shared__ half V_smem[B][d];
    __shared__ float D_smem[B][B];  // decay matrix
    __shared__ float h[d][d];       // persistent state in shared memory

    // Load block data
    load_block(Q, block_id, Q_smem);
    load_block(K, block_id, K_smem);
    load_block(V, block_id, V_smem);

    // Compute decay matrix D
    compute_decay_matrix(log_decay, block_id, D_smem);

    // Phase 1: Intra-block (using wmma for Tensor Core matmul)
    half M[B][B];
    wmma_matmul(Q_smem, K_smem, M);     // Q @ K^T
    elementwise_multiply(M, D_smem);      // M * D (apply decay mask)
    half O_intra[B][d];
    wmma_matmul(M, V_smem, O_intra);     // M @ V

    // Phase 3: Inter-block output (using state from previous iteration)
    half O_inter[B][d];
    wmma_matmul(Q_smem, h, O_inter);     // Q @ h (with position-specific decay)

    // Phase 2: Update state (for next block)
    decay_state(h, block_total_decay);
    wmma_outer_product_accumulate(K_smem, V_smem, h);  // h += K^T @ V

    // Combine and write output
    for (int i = 0; i < B; i++)
        for (int j = 0; j < d; j++)
            O[block_id * B + i][j] = O_intra[i][j] + O_inter[i][j];
}
```

### 7.4 State Passing Between Thread Blocks

A challenge: the state h must be passed sequentially between blocks. Options:

1. **Sequential block launch:** Process blocks one at a time (simplest, but underutilizes GPU for many heads)
2. **Persistent kernel:** A single persistent kernel processes all blocks for a head, keeping h in shared memory between blocks
3. **Global memory state:** Write h to global memory after each block, read before next block (adds memory traffic)
4. **Chunk-level parallelism:** Process multiple chunks of blocks in parallel across different heads/batch elements, with sequential processing within each chunk

Mamba-2 uses approach (2) for short sequences and (4) for long sequences.

## 8. Performance Results

### 8.1 Training Speed

Wall-clock training time comparison (tokens per second, higher is better):

**A100-80GB, batch size 64:**

| Model | Seq Len | Transformer (FA2) | Mamba-1 | Mamba-2 (SSD) |
|-------|---------|-------------------|---------|---------------|
| 370M | 2K | 125K tok/s | 115K tok/s | 140K tok/s |
| 370M | 8K | 32K tok/s | 48K tok/s | 55K tok/s |
| 1.3B | 2K | 48K tok/s | 42K tok/s | 56K tok/s |
| 1.3B | 8K | 12K tok/s | 16K tok/s | 21K tok/s |
| 2.7B | 2K | 25K tok/s | 22K tok/s | 30K tok/s |
| 2.7B | 8K | 6.5K tok/s | 8.5K tok/s | 11K tok/s |

Key observations:
- Mamba-2 is **1.2-1.7x faster** than Mamba-1 across all configurations (Tensor Core utilization)
- Mamba-2 is **1.1-1.7x faster** than Transformer with FA2, with the advantage growing at longer sequences
- At 2K sequences, Mamba-2 and Transformers are similar speed; the advantage emerges at 4K+

### 8.2 Inference Speed

Single-sequence autoregressive generation (tokens per second):

| Model Size | Transformer + KV Cache | Mamba-1 | Mamba-2 |
|-----------|----------------------|---------|---------|
| 370M | 2850 tok/s | 3200 tok/s | 3400 tok/s |
| 1.3B | 980 tok/s | 1150 tok/s | 1250 tok/s |
| 2.7B | 520 tok/s | 610 tok/s | 680 tok/s |

Mamba-2 is slightly faster than Mamba-1 for inference (simpler kernel). Both are faster than Transformer at single-sequence decode because:
- **No KV cache growth:** Mamba's state is fixed-size (d x d per head)
- **O(1) per step:** Each decode step is a single state update + output, not attention over growing KV cache
- **No memory-bandwidth bottleneck:** The state fits in registers/shared memory

### 8.3 Language Modeling Quality

Perplexity on The Pile (lower is better):

| Model | Params | Transformer | Mamba-1 | Mamba-2 |
|-------|--------|------------|---------|---------|
| Small | 370M | 14.2 | 14.5 | 14.1 |
| Medium | 1.3B | 11.1 | 11.3 | 10.9 |
| Large | 2.7B | 10.0 | 10.2 | 9.8 |

Mamba-2 slightly outperforms both Mamba-1 and the Transformer baseline, likely due to the larger state dimension (P = d = 128 vs Mamba-1's P = 16) enabled by the efficient SSD algorithm.

### 8.4 Downstream Tasks

Zero-shot accuracy on common benchmarks (2.7B models):

| Benchmark | Transformer | Mamba-1 | Mamba-2 |
|-----------|------------|---------|---------|
| LAMBADA (acc) | 70.1 | 69.8 | 71.2 |
| HellaSwag (acc_norm) | 56.8 | 55.9 | 57.3 |
| PIQA (acc) | 76.2 | 75.8 | 76.5 |
| ARC-Easy (acc) | 65.4 | 64.7 | 66.1 |
| ARC-Challenge (acc_norm) | 35.8 | 35.1 | 36.4 |
| WinoGrande (acc) | 63.5 | 62.8 | 64.1 |
| Average | 61.3 | 60.7 | 61.9 |

Mamba-2 consistently outperforms both baselines across all benchmarks.

## 9. Hybrid Architectures

### 9.1 Attention + SSM Hybrids

The duality framework suggests natural hybrid architectures:

```
Hybrid Model:
    Layer 1:  SSD (efficient, handles majority of sequence mixing)
    Layer 2:  SSD
    Layer 3:  SSD
    Layer 4:  Attention (precise retrieval for critical positions)
    Layer 5:  SSD
    Layer 6:  SSD
    Layer 7:  SSD
    Layer 8:  Attention
    ...
```

Every k-th layer uses softmax attention (typically k = 4 or k = 6), providing periodic "checkpoints" of precise attention while keeping most layers efficient.

### 9.2 Hybrid Results

| Architecture | Params | Pile PPL | Training Speed | Inference Speed |
|-------------|--------|----------|----------------|-----------------|
| All Transformer | 2.7B | 10.0 | 1.0x | 1.0x |
| All Mamba-2 | 2.7B | 9.8 | 1.5x | 1.3x |
| Hybrid (1:3 attn:SSD) | 2.7B | 9.7 | 1.35x | 1.2x |
| Hybrid (1:7 attn:SSD) | 2.7B | 9.8 | 1.45x | 1.25x |

Hybrid architectures often achieve the **best of both worlds**: quality matching or exceeding both pure architectures while maintaining most of the SSM speed advantage.

### 9.3 Jamba and Other Hybrid Models

Several production models have adopted hybrid attention-SSM architectures:
- **Jamba (AI21, 2024):** Interleaves Mamba and attention layers with MoE
- **Griffin (Google, 2024):** Recurrent-attention hybrid
- **RecurrentGemma (Google, 2024):** Griffin-based model
- **Zamba (Zyphra, 2024):** Mamba-attention hybrid focused on efficiency

## 10. Connections to Other Linear-Time Models

### 10.1 RetNet (Retentive Network)

RetNet (Sun et al., 2023) uses a similar recurrence with exponential decay:
```
O[t] = sum_{j<=t} gamma^{t-j} (Q[t]^T K[j]) V[j]
```
This is a special case of SSD with **scalar, fixed decay** gamma. SSD generalizes this to learned, potentially data-dependent decay.

### 10.2 RWKV

RWKV (Peng et al., 2023) uses a channel-wise recurrence:
```
wkv[t] = sum_{j<=t} exp(-(t-j)*w + k[j]) v[j] / sum_{j<=t} exp(-(t-j)*w + k[j])
```
This includes a normalization (like softmax) on top of the decay. SSD's framework can express this as a normalized semiseparable matrix multiplication.

### 10.3 Gated Linear Attention (GLA)

GLA (Yang et al., 2024) uses data-dependent gating:
```
S[t] = G[t] ⊙ S[t-1] + K[t]^T V[t]
O[t] = Q[t] S[t]
```
where G[t] is a data-dependent gating matrix. This is equivalent to SSD with diagonal, data-dependent A[t] = diag(G[t]).

### 10.4 Unified View

| Model | A (transition) | B (input) | C (output) | Normalization |
|-------|---------------|-----------|-----------|---------------|
| Linear Attention | I (identity) | K | Q | None |
| RetNet | gamma * I (scalar) | K | Q | None |
| RWKV | exp(-w) * I | exp(k) | Q | Softmax-like |
| GLA | diag(G) (data-dep.) | K | Q | None |
| Mamba-1 | diag(A) (learned) | B(x) | C(x) | None |
| Mamba-2 (SSD) | diag(A) (learned) | K(x) | Q(x) | Optional |

All these models compute Y = M * U where M is a semiseparable matrix. They differ in the parameterization and constraints on A, B, C.

## 11. Practical Recommendations for Kernel Engineers

### 11.1 Block Size Selection

The optimal block size B depends on:
1. **State dimension P (= d for Mamba-2):** B ~ sqrt(P) minimizes total FLOPs
2. **Tensor Core tile size:** B should be a multiple of 16 (FP16 wmma) or 64 (WGMMA on Hopper)
3. **Shared memory capacity:** Must fit Q_b, K_b, V_b, M_b, h in SMEM

Practical choices:
- **A100:** B = 64 or 128 (192 KB SMEM per SM)
- **H100:** B = 128 or 256 (228 KB SMEM per SM, larger Tensor Core tiles)

### 11.2 State Dimension Tradeoff

Larger state P:
- Better quality (higher-rank attention matrix, more expressive)
- More inter-block compute: O(N d P / B)
- Larger state to pass between blocks: O(d P) elements

Practical sweet spot: P = d = 64-128 (matches head dimension, good Tensor Core tile size)

### 11.3 Memory Layout

```
# Optimal memory layout for SSD kernel
Q: [batch, num_blocks, block_size, heads, d]  -- block-contiguous for coalesced loads
K: [batch, num_blocks, block_size, heads, d]
V: [batch, num_blocks, block_size, heads, d]
h: [batch, num_blocks, heads, d, d]           -- state per block (written sequentially)
O: [batch, num_blocks, block_size, heads, d]
```

### 11.4 Mixed Precision

- **Intra-block matmul (Q @ K^T, M @ V):** FP16 or BF16 Tensor Core (high throughput)
- **Decay computation (cumsum of log_decay):** FP32 (avoid precision loss in cumulative sum)
- **State h:** FP32 (accumulates across many blocks, needs precision)
- **Output O:** FP16 or BF16 (matches model precision)

## 12. Limitations and Open Problems

### 12.1 Current Limitations

1. **Retrieval gap:** Despite improvements, SSMs still lag softmax attention on tasks requiring precise retrieval from long contexts.
2. **Sequential state dependency:** The inter-block state propagation is inherently sequential, limiting parallelism along the sequence dimension.
3. **Fixed state size:** The (d x d) state cannot grow with sequence length, creating an information bottleneck for very long sequences.
4. **Kernel maturity:** SSD kernels are newer and less optimized than FlashAttention. There is room for significant performance improvement.

### 12.2 Open Problems

1. **Adaptive state size:** Can the state dimension grow dynamically based on content complexity?
2. **Better decay mechanisms:** Can we design data-dependent decay that closes the retrieval gap with softmax attention?
3. **Hardware co-design:** Can specialized hardware make the sequential state propagation faster?
4. **Theoretical understanding:** What is the exact computational class of semiseparable matrix models? What tasks fundamentally require full-rank attention?

## 13. References

1. Tri Dao, Albert Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (ICML 2024, arXiv:2405.21060)
2. Albert Gu, Tri Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arXiv:2312.00752, 2023)
3. Angelos Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (ICML 2020)
4. Albert Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces" (ICLR 2022) -- S4
5. Yutao Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models" (arXiv:2307.08621, 2023)
6. Bo Peng et al., "RWKV: Reinventing RNNs for the Transformer Era" (EMNLP 2023)
7. Songlin Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training" (ICML 2024)
8. Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model" (AI21 Labs, 2024)
9. De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models" (Google, 2024)
10. Code: https://github.com/state-spaces/mamba
