---
skill_name: Linear Attention and Mamba-2 SSM-Attention Duality
description: Unification of linear attention and structured state space models (SSMs) through the framework of semiseparable matrices, enabling a hybrid block-decomposition algorithm (SSD) that combines the parallel efficiency of attention with the recurrent efficiency of SSMs.
level: L1 - Algorithmic/Mathematical Stage Level
target_hardware: NVIDIA GPUs (Ampere A100, Hopper H100 and newer with Tensor Cores)
relevance: When choosing between attention and SSM architectures, designing hybrid models, or implementing kernels that exploit the mathematical duality between linear attention and state space models for optimal hardware utilization.
---

# Linear Attention and Mamba-2 SSM-Attention Duality

## What It Is
Linear Attention replaces the softmax in standard attention with a kernel feature map, enabling the computation to be rearranged from O(N^2 d) to O(N d^2) via the associativity of matrix multiplication. The Mamba-2 paper ("Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" by Dao and Gu, 2024) establishes a rigorous mathematical equivalence between a broad class of structured SSMs and a form of linear attention with data-dependent (input-varying) decay, unified through the framework of **semiseparable matrices**. This duality yields a new algorithm called **Structured State Space Duality (SSD)** that decomposes the sequence into blocks and uses matrix multiplication (attention-like) within blocks while using SSM-like recurrence between blocks, achieving 2-8x speedup over Mamba-1 by leveraging Tensor Cores for the intra-block computation.

## Key Concepts
- **Linear Attention:** Replace softmax(QK^T) with phi(Q) phi(K)^T where phi is a feature map. By rewriting as phi(Q) @ (phi(K)^T @ V), the per-token cost becomes O(d^2) instead of O(Nd), making total cost O(Nd^2) -- linear in sequence length N.
- **Structured State Space Models (SSMs):** Recurrent models defined by x_t = A_t x_t-1 + B_t u_t; y_t = C_t x_t, where A_t is the state transition matrix (often diagonal), B_t and C_t are input/output projections, and the subscript t indicates data-dependent (selective) parameters.
- **Semiseparable Matrices:** The output matrix Y = M @ U where M is an (N x N) lower-triangular matrix whose (i,j) entry is C_i^T (A_i A_{i-1} ... A_{j+1}) B_j. This matrix M has semiseparable structure (low rank off-diagonal blocks), which is the mathematical link connecting SSMs to attention.
- **SSD Block Decomposition:** Partition the sequence into blocks of size B. Within each block, compute the full B x B attention-like matrix (using Tensor Cores). Between blocks, propagate state using the SSM recurrence. This gives O(NB + N d^2/B) compute, optimized when B ~ d.
- **Data-Dependent Decay:** Unlike vanilla linear attention (which has no decay), SSMs use a multiplicative A_t factor that decays older information. The SSD framework shows this is equivalent to linear attention with a causal, data-dependent exponential mask.
- **Mamba-2 Architecture:** Uses the SSD layer as its core, with multi-head structure (analogous to multi-head attention), shared A across heads, and separate B, C per head. Head dimension is typically 64-128 to match Tensor Core tile sizes.

## Algorithm / Pseudo-code
```
# ========================================
# Linear Attention (basic kernel formulation)
# ========================================
# Inputs: Q, K, V each (N x d), feature map phi
# Causal linear attention (recurrent form):

S = zeros(d, d)  # running state matrix
for t in 1..N:
    S = S + phi(K[t])^T @ phi(Q[t])  # ERROR: should be K outer V
    # Correct recurrent form:
    S = S + outer(phi(K[t]), V[t])    # (d x d) state update
    O[t] = phi(Q[t]) @ S              # (1 x d) output

# Total cost: O(N * d^2) -- linear in N

# ========================================
# SSM Recurrence (Mamba/S6 selective scan)
# ========================================
# Inputs: u (N x D), parameters A_t (D x P), B_t (P,), C_t (P,)
# State: x_t (D x P)

for t in 1..N:
    x_t = diag(A_t) @ x_{t-1} + outer(B_t, u_t)  # state update
    y_t = C_t^T @ x_t                               # output

# ========================================
# SSD: Structured State Space Duality (Mamba-2)
# Block-decomposed algorithm combining attention + recurrence
# ========================================
# Inputs: Q, K, V each (N x d), decay factors A (N,)
# Block size: B (typically 64-256)

num_blocks = N // B

# Precompute cumulative decay within each block
# D[i,j] = product(A[j+1:i+1]) for i >= j within block (lower triangular)

for block in 0 .. num_blocks-1:
    start = block * B

    # --- Intra-block: attention-like matmul (on Tensor Cores) ---
    Q_b = Q[start : start+B]          # (B x d)
    K_b = K[start : start+B]          # (B x d)
    V_b = V[start : start+B]          # (B x d)

    # Compute masked attention matrix with decay
    # M[i,j] = (Q_b[i] @ K_b[j]^T) * D[i,j] for i >= j, else 0
    M = (Q_b @ K_b^T) * D_block       # (B x B), lower triangular
    O_intra = M @ V_b                  # (B x d), via Tensor Cores

    # --- Inter-block: SSM-like state propagation ---
    # State from previous block: h_{block-1} of shape (d x d)
    # Decay state: h_block = A_cumulative * h_{block-1} + K_b^T @ V_b

    O_inter = Q_b @ h_prev             # (B x d), contribution from past

    # Update state for next block
    h_new = decay_block * h_prev + K_b^T @ (D_lower @ V_b)

    # --- Combine ---
    O[start : start+B] = O_intra + O_inter
    h_prev = h_new

# Complexity: O(N*B + N*d^2) -- B^2 for intra, d^2 for inter
# Optimal when B ~ d, giving O(N * d^2) total
```

## When to Use
- When sequence length N >> head dimension d, making linear attention's O(Nd^2) cheaper than softmax attention's O(N^2 d)
- When designing hybrid architectures that want to combine the expressiveness of attention with the efficiency of SSMs
- Language modeling tasks where recurrent state compression is acceptable and the model does not need to attend to arbitrary token pairs with full softmax precision
- When Tensor Core utilization is important: SSD's block decomposition maps the intra-block computation to dense matrix multiplications that fully utilize Tensor Cores
- Continuous signal processing or very long sequences (100K+ tokens) where quadratic attention is infeasible even with FlashAttention
- When you want a single unified kernel that seamlessly transitions between "attention mode" (short sequences, intra-block) and "recurrence mode" (long sequences, inter-block)

## When NOT to Use
- Tasks requiring precise, arbitrary-range token-to-token attention (e.g., copying, exact retrieval) where softmax attention's ability to produce sharp, peaked distributions is essential
- Short sequences (< 2K tokens) where O(N^2 d) with FlashAttention is already fast and the O(Nd^2) of linear attention may actually be slower when d is large
- When using pre-trained softmax attention models that cannot be replaced (the SSM/linear attention duality requires training the model with the new mechanism)
- Tasks with very large head dimension d where d^2 > N*d, making linear attention more expensive than softmax
- When bit-exact compatibility with standard softmax attention is required

## Key Takeaways
- The SSD algorithm in Mamba-2 achieves 2-8x wall-clock speedup over Mamba-1's selective scan by converting the bottleneck computation into matrix multiplications that leverage Tensor Cores
- The semiseparable matrix framework provides the theoretical foundation: any SSM with diagonal state transition computes an output equivalent to multiplying by a semiseparable matrix, which is also what linear attention computes
- Block decomposition is the key algorithmic insight: use O(B^2) attention within blocks (parallelizable, Tensor Core friendly) and O(d^2) recurrence between blocks (sequential but small), optimizing the compute-memory tradeoff
- Mamba-2 matches or exceeds Transformer quality on language modeling benchmarks up to 2.7B parameters while being significantly faster on long sequences
- The duality suggests a design spectrum: pure attention (B=N, no recurrence), pure SSM (B=1, fully recurrent), and hybrids (moderate B) that can be tuned based on hardware and sequence length
- Linear attention without decay (vanilla kernel attention) generally underperforms softmax attention on language tasks; the data-dependent decay in SSMs is critical for matching quality

## References
- Mamba-2 Paper: Tri Dao, Albert Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (ICML 2024, arXiv:2405.21060)
- Mamba-1 Paper: Albert Gu, Tri Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arXiv:2312.00752, December 2023)
- Linear Attention: Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (ICML 2020)
- S4: Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces" (ICLR 2022)
- Code: https://github.com/state-spaces/mamba (Mamba and Mamba-2 implementations)
- Related: RWKV (Peng et al., 2023), RetNet (Sun et al., 2023), GLA (Yang et al., 2024)
