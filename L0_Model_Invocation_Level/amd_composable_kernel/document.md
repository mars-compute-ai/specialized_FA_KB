# AMD Composable Kernel (CK) & CK-Tile for Flash Attention

## Overview

Composable Kernel (CK) is AMD's high-performance kernel library for GPU compute, analogous to NVIDIA's CUTLASS. CK-Tile is its tile-based programming abstraction for building fused kernels like FlashAttention on AMD GPUs. The ROCm flash-attention fork uses CK as its default backend.

## CK-Tile FlashAttention-v2 Implementation

### Architecture Mapping (AMD vs NVIDIA)
| AMD (ROCm) | NVIDIA (CUDA) | Size |
|------------|---------------|------|
| Wavefront | Warp | 64 threads (vs 32) |
| LDS (Local Data Share) | Shared Memory | Up to 64 KB per CU |
| VGPR/SGPR | Registers | Architecture-dependent |
| Compute Unit (CU) | Streaming Multiprocessor (SM) | MI300X: 304 CUs |
| MFMA instruction | MMA/WMMA/WGMMA instruction | Matrix multiply-accumulate |

### Memory Hierarchy and Data Movement
The implementation uses three memory levels:
1. **DRAM (HBM3)**: Stores Q, K, V inputs and O output — MI300X has 192GB at 5.3 TB/s
2. **LDS**: Buffers K and V tiles — 128-bit cache line aligned access
3. **VGPRs**: Hold Q tiles and intermediate computation — loaded once per output block

Data flow:
```
Q: DRAM → VGPR (loaded once per C block)
K: DRAM → LDS → VGPR (looped access)
V: DRAM → LDS (looped access, MFMA reads from LDS)
O: VGPR → DRAM (output write)
```

### Tiling Strategy
- M-dimension tiles: 128 rows of Q
- N-dimension tiles: 128 columns of K
- K-dimension tiles: 32 (shared dimension)
- Inter-threadblock parallelism across Q blocks
- Intra-threadblock loops iterate K and V

### Computation Pipeline
```python
for each output O block:
    Load Q block into VGPRs
    for each K block:
        Load K to LDS
        Compute S = Q × K^T via MFMA (GEMM0)
        Track row-wise max(m) for online softmax
    for each V block:
        Load V to LDS
        Compute softmax attention weights
        Accumulate O += P × V^T via MFMA (GEMM1)
    Write O block to DRAM
```

### Key CK-Tile APIs
- `make_tile_window()` — Defines data access patterns
- `make_tensor_view<global/lds>()` — Memory space abstractions
- `tile_distribution_encoding` — Maps wavefronts and lanes to data
- `block_tile_reduce()` — Synchronous reduction
- `gemm0_pipeline()` / `gemm1()` — MFMA wrapper functions

### ROCm Flash Attention Backends
The ROCm flash-attention fork supports two backends:
1. **Composable Kernel (CK)** — Default, highest performance, C++/HIP
2. **OpenAI Triton** — Alternative, Python-based, easier to modify

## MFMA (Matrix Fused Multiply-Add) Instructions
AMD's equivalent to NVIDIA's Tensor Cores:
- **MI250X (CDNA2)**: MFMA_F32_16x16x16_F16, MFMA_F32_32x32x8_F16
- **MI300X (CDNA3)**: Additional FP8 MFMA instructions, higher throughput
- Operate on wavefront (64 threads) — 4 threads per row in a 16×16 tile

## References
- [CK-Tile FlashAttention Blog (AMD ROCm)](https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html)
- [ROCm/composable_kernel GitHub](https://github.com/ROCm/composable_kernel)
- [ROCm/flash-attention GitHub](https://github.com/ROCm/flash-attention)
- [CK FMHA kernel examples](https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha)
