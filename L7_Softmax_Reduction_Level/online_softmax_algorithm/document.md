# From Online Softmax to FlashAttention

**Source**: [CSE 599M Course Notes - University of Washington](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
**Author**: Zihao Ye (zhye@cs.washington.edu)
**Supplementary Sources**:
- [From Online Softmax to Flash Attention V3 - Chenghua Wang](https://chenghuawang.github.io/keep-moving-forward/tech/fundamental_from_online_softmax_to_flash_attentionv3/)
- [The Basic Idea Behind FlashAttention - Peter Chng](https://peterchng.com/blog/2024/06/26/the-basic-idea-behind-flashattention/)

## Document Outline

1. The Self-Attention
2. (Safe) Softmax
3. Online Softmax
4. FlashAttention

---

## 1. Standard (Safe) Softmax

The naive "safe" softmax requires **three passes** over an input vector of length N:

1. **First pass** - Find maximum: `m = max(x_i)` for all i (prevents overflow)
2. **Second pass** - Compute denominator: `d = sum(exp(x_i - m))` for all i
3. **Third pass** - Normalize: `softmax(x_i) = exp(x_i - m) / d` for each i

Each pass requires reading the entire input vector from memory, resulting in 3N memory reads for a vector of length N.

### Mathematical Definition

Given input vector X in R^N:

```
M = max(X)
softmax(x_i) = exp(x_i - M) / sum_{j=0}^{N} exp(x_j - M)
```

The subtraction of the maximum M ensures numerical stability by preventing overflow in the exponential computation.

---

## 2. Online Softmax Algorithm

The key innovation is a **recursive formula** that maintains running statistics. Instead of three sequential passes, online softmax merges the first two passes into one, computing both the max `m` and the denominator `d` simultaneously.

### Recursive Update Rules

For each new element x_i:

```
m_i = max(m_{i-1}, x_i)
d_i = d_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)
```

Where:
- `m_i` tracks the running maximum after seeing elements x_0 through x_i
- `d_i` accumulates the running denominator (sum of exponentials)
- `exp(m_{i-1} - m_i)` is the **correction factor** that rescales the previous partial sum when a new maximum is discovered

### Why the Correction Factor Works

When a new maximum m_i is found (m_i > m_{i-1}), all previously accumulated exponentials were computed relative to the old maximum. The factor `exp(m_{i-1} - m_i)` adjusts them:

```
d_{i-1} was sum of exp(x_j - m_{i-1}) for j < i
d_{i-1} * exp(m_{i-1} - m_i) = sum of exp(x_j - m_i) for j < i  (re-biased to new max)
```

This makes the accumulated sum consistent with the new maximum.

### Python Implementation (Two-Pass Online Softmax)

```python
import math

def online_softmax(x):
    # Pass 1: Compute max and denominator simultaneously
    m = float('-inf')
    d = 0
    for x_i in x:
        m_next = max(m, x_i)
        d = d * math.exp(m - m_next) + math.exp(x_i - m_next)
        m = m_next

    # Pass 2: Normalize
    o = []
    for x_i in x:
        o.append(math.exp(x_i - m) / d)
    return o
```

This reduces the algorithm from 3 passes to 2 passes.

---

## 3. FlashAttention: Extending to Attention Output Computation

FlashAttention extends the online softmax to compute the attention output O = softmax(QK^T) * V **without materializing the full attention matrix**.

### Key Insight

The output O = A * V (where A = softmax(S) and S = QK^T) can be computed **incrementally**. We never need to store the full N x N attention matrix.

### Recursive Output Update

For a single query row k against n_ctx key-value rows, at each step i:

1. Compute attention score: `x_i = Q[k,:] . K^T[:,i]`
2. Update running max: `m_i = max(m_{i-1}, x_i)`
3. Update running denominator: `d_i = d_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)`
4. Update output accumulator:
   ```
   o_i = o_{i-1} * (d_{i-1} / d_i) * exp(m_{i-1} - m_i)  +  (exp(x_i - m_i) / d_i) * V[i,:]
   ```

The full recursive formula:
```
o_i' = o_{i-1}' * (d_{i-1}' * exp(m_{i-1} - m_i)) / d_i' + (exp(x_i - m_i) / d_i') * V[i,:]
```

### PyTorch Implementation

```python
import torch

torch.manual_seed(1337)
d_head = 10
n_ctx = 5

q = torch.randn((d_head,))
k = torch.randn((n_ctx, d_head))
v = torch.randn((n_ctx, d_head))
o = torch.zeros_like(q)

m = torch.tensor(float('-inf'))
d = torch.tensor(0.0)

for i in range(n_ctx):
    x_i = q @ k[i:i+1, :].transpose(-2, -1)
    m_next = torch.maximum(m, x_i)
    d_next = d * torch.exp(m - m_next) + torch.exp(x_i - m_next)

    o_adjust = d * torch.exp(m - m_next) / d_next
    o_add = torch.exp(x_i - m_next) * v[i:i+1, :] / d_next
    o = o * o_adjust + o_add
    m = m_next
    d = d_next
```

---

## 4. Tiled FlashAttention

The practical FlashAttention implementation processes **tiles** (blocks) rather than individual elements:

### Memory Model
- **HBM (High Bandwidth Memory)**: Large but slow - stores Q, K, V, O matrices
- **SRAM (Shared Memory)**: Small but fast - holds tiles during computation

### Tiling Strategy
- Q, K, V are split into blocks of size B
- For each tile of Q, iterate over tiles of K and V
- Compute partial attention scores in SRAM
- Update running statistics (max, sum) and output using the online softmax correction

### Key Property
The overall SRAM memory footprint depends only on **block size B and head dimension D** and is **not related to sequence length L**. This allows the algorithm to scale to arbitrarily long sequences without running out of on-chip memory.

### Algorithm Pseudocode

```
# Initialize
O = zeros(N, d)
l = zeros(N)      # running sum of exponentials
m = -inf(N)       # running maximum

# Load Q block
for each Q tile (size Br x d):
    for each K, V tile (size Bc x d):
        # Compute attention scores for this tile
        S_tile = Q_tile @ K_tile^T          # (Br x Bc)

        # Update running max
        m_new = max(m_old, rowmax(S_tile))

        # Compute exponentials with new max
        P_tile = exp(S_tile - m_new)

        # Correction factor for previous accumulations
        alpha = exp(m_old - m_new)

        # Update running sum
        l_new = alpha * l_old + rowsum(P_tile)

        # Update output with correction
        O = alpha * O + P_tile @ V_tile

    # Final normalization
    O = O / l_new
```

---

## 5. Memory and Computational Analysis

### Standard Attention
- Memory: O(N^2) for storing the attention matrix
- HBM accesses: Multiple passes over N^2 data

### FlashAttention
- Memory: O(N) - only stores output and running statistics
- HBM accesses: O(N^2 * d / M) where M is SRAM size
- The algorithm is **IO-aware**: it minimizes HBM reads/writes by keeping intermediate data in SRAM

### Trade-off
FlashAttention performs **more FLOPs** than standard attention (due to recomputation and rescaling), but is faster in practice because it is **memory-bandwidth bound**, not compute-bound. The reduction in HBM accesses more than compensates for the extra arithmetic.

---

## References

- Milakov, M. and Gimelshein, N. "Online normalizer calculation for softmax." arXiv:1805.02867, 2018.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Re, C. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS, 2022.
- Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR, 2024.
- Zihao Ye, CSE 599M Course Notes, University of Washington, Spring 2023.
