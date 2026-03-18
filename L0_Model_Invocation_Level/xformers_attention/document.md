# xFormers Memory-Efficient Attention

Source: https://facebookresearch.github.io/xformers/components/ops.html

## Overview

xFormers provides multiple memory-efficient multi-head attention (MHA) implementations through a unified interface. It offers Cutlass-based, FlashAttention-based, and Composable Kernel (AMD) variants, each with forward (FwOp) and backward (BwOp) operator classes. The library also provides specialized bias/mask types that avoid materializing large attention matrices.

## Available Backend Implementations

### 1. CUTLASS-based (`xformers.ops.fmha.cutlass`)

- **FwOp** and **BwOp**: Forward and backward operators
- Supports a large number of settings including without TensorCores and f32
- Compatible with GPUs as old as P100 (Sm60)
- Broadest hardware compatibility

### 2. Flash-Attention (`xformers.ops.fmha.flash`)

- **FwOp** and **BwOp**: Wrap the FlashAttention implementation
- Computes memory-efficient attention using the FlashAttention algorithm
- Requires fp16/bf16
- Best performance on Ampere+ GPUs

### 3. Composable Kernel (`xformers.ops.fmha.ck`)

- xFormers' MHA kernel based on AMD's Composable Kernel library
- Forward and backward support
- Targets AMD GPUs (MI200x, MI250x, MI300x)

## Core API Functions

### `memory_efficient_attention_forward()`

Executes the forward attention pass. Optionally returns LSE (log-sum-exp) values needed for backward pass computation.

### `memory_efficient_attention_backward()`

Computes gradients. Returns a tuple `(dq, dk, dv)` from forward intermediate values.

### `memory_efficient_attention_partial()`

Returns a tuple `(output, lse)` enabling attention computation on disjoint key-value chunks. This is useful for:
- Ring attention / sequence parallelism
- Processing very long sequences in chunks
- Distributed attention across GPUs

### `merge_attentions()`

Combines partial attention outputs using the formula:
```
Out_full = (Out1 * exp(LSE1) + Out2 * exp(LSE2) + ...) / (exp(LSE1) + exp(LSE2) + ...)
```

This allows computing attention over chunks and merging results exactly.

## Attention Bias Types

xFormers provides specialized bias classes that avoid materializing large N x N masks:

### `LowerTriangularMask`

Standard causal masking: a query Q cannot attend to a key which is farther from the initial key than Q is from the initial query.

### `LowerTriangularFromBottomRightMask`

Right-aligned causal masking for unequal query/key lengths. Useful for KV-cache inference where query length < key length.

### `BlockDiagonalMask`

Queries and Keys are each divided into the same number of blocks. Queries in block i only attend to keys in block i. Useful for:
- Batching variable-length sequences without padding
- Document-level attention in multi-document batches

Provides `from_tensor_list()` for handling batches of sequences of different lengths, and `split()` for recovering individual sequences.

### `BlockDiagonalCausalMask`

Combines block-diagonal structure with causality constraints. Each block has its own causal mask.

### `LocalAttentionFromBottomRightMask`

Windowed/local attention: the query at position q can attend the key at position k if `q - window_left <= k + s <= q + window_right`.

### Variants with Padding and Paging

Support for padded keys, gappy sequences, and paged attention mechanisms for serving workloads.

## Debugging with `materialize()`

All bias implementations include a `materialize()` method that converts abstract masks to concrete tensors:

```python
mask = LowerTriangularMask()
concrete = mask.materialize(shape=(seq_len, seq_len))
```

**Warning**: This is very slow and should only be used for debugging/testing.

## Usage Patterns

### Basic Memory-Efficient Attention

```python
import xformers.ops as xops

# Basic usage
output = xops.memory_efficient_attention(query, key, value)

# With causal mask
output = xops.memory_efficient_attention(
    query, key, value,
    attn_bias=xops.LowerTriangularMask()
)
```

### Forcing a Specific Backend

```python
from xformers.ops.fmha import flash, cutlass

# Use FlashAttention backend specifically
output = xops.memory_efficient_attention(
    query, key, value,
    op=(flash.FwOp, flash.BwOp)
)

# Use CUTLASS backend
output = xops.memory_efficient_attention(
    query, key, value,
    op=(cutlass.FwOp, cutlass.BwOp)
)
```

### Variable-Length Sequences (No Padding)

```python
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

# Create mask for variable-length sequences
attn_bias = BlockDiagonalMask.from_seqlens([seq_len_1, seq_len_2, seq_len_3])

# Concatenate all sequences along sequence dimension
# query shape: (1, total_seq_len, num_heads, head_dim)
output = xops.memory_efficient_attention(
    query, key, value,
    attn_bias=attn_bias
)
```

### Partial / Chunked Attention

```python
# Compute attention on KV chunks separately
out1, lse1 = xops.memory_efficient_attention_partial(query, key_chunk1, value_chunk1)
out2, lse2 = xops.memory_efficient_attention_partial(query, key_chunk2, value_chunk2)

# Merge results exactly
output = xops.merge_attentions([out1, out2], [lse1, lse2])
```

## Input Format

xFormers expects tensors in the format `(batch, seq_len, num_heads, head_dim)` -- note this differs from PyTorch's SDPA which uses `(batch, num_heads, seq_len, head_dim)`.

## Key Characteristics

- **Memory efficiency**: O(N) memory instead of O(N^2) by not materializing the full attention matrix
- **Exact computation**: Results are mathematically identical to standard attention (within floating-point precision)
- **Automatic backend selection**: Chooses the best available backend based on inputs and hardware
- **Composable biases**: Mask types compose naturally (block-diagonal + causal, local + causal)
- **Partial attention**: Enables sequence parallelism and ring attention patterns
