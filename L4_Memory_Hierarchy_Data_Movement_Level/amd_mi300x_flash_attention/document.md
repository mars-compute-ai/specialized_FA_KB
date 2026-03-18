# Flash Attention Optimization on AMD MI300X (CDNA3)

## MI300X Architecture for Attention

### Chiplet Design
- **8 XCDs** (Accelerator Complex Dies), each with 4 Shader Engines
- **304 Compute Units** total (38 per XCD)
- Each CU: 4 SIMD units, 64 KB LDS, 256 KB VGPR file
- **Unified Memory Architecture**: CPU and GPU share a single memory space

### Memory Hierarchy
| Level | Capacity | Bandwidth | Latency |
|-------|----------|-----------|---------|
| VGPRs | 256 KB per CU | ~100 TB/s (register-level) | 1 cycle |
| LDS | 64 KB per CU | ~25 TB/s | ~20 cycles |
| L2 Cache | 256 MB (32 MB per XCD) | ~12 TB/s | ~100 cycles |
| HBM3 | 192 GB | **5.3 TB/s** | ~300 cycles |

**Key advantage**: MI300X's 5.3 TB/s HBM3 bandwidth exceeds H100's 3.35 TB/s by 1.58x, benefiting the memory-bound phases of attention.

### MFMA Instructions (Matrix Fused Multiply-Add)
CDNA3 MFMA shapes for attention:
- `MFMA_F32_16x16x16_F16` — 16×16 output, K=16 in FP16
- `MFMA_F32_32x32x8_F16` — 32×32 output, K=8 in FP16
- `MFMA_F32_16x16x32_F8` — FP8 variant (CDNA3 new)
- Operate per-wavefront (64 threads), unlike NVIDIA's per-warpgroup WGMMA

## TileLang Flash Attention on MI300X

### Performance Results
Batch=1, heads=8, seq_len=4096, dim=128:

| Implementation | Latency (ms) | Speedup vs PyTorch |
|---|---|---|
| PyTorch Reference | 0.97 | 1.0x |
| Triton | 0.55 | 1.76x |
| **TileLang** | **0.36** | **2.69x** |

### Optimization Parameters
Autotuning searches 108 configurations:
- **block_M, block_N**: Tile sizes controlling SRAM utilization (optimal: M=128, N=32)
- **threads**: 512 per block (optimal for MI300X CU architecture)
- **num_split_q**: Parallel splits for Q dimension — improves multi-CU utilization
- **num_stages**: Pipeline stages (typically 1 for fused dual-GEMM kernels like FA)
- **qk_coalesced_width / v_coalesced_width**: Memory coalescing parameters for HBM access
- **T.use_swizzle**: Memory reordering to improve LDS bank conflict patterns

### MI300X-Specific Optimizations
- **GemmWarpPolicy.FullRow**: Warp scheduling adapted to MI300X wavefront structure
- **Shared memory allocation**: Strategic LDS placement exploiting 64 KB per CU
- **Register caching**: `acc_s_cast` reduces LDS pressure through VGPR caching
- **Rasterization**: Block scheduling across XCDs for load balancing

### Key Differences from NVIDIA Optimization
| Aspect | AMD MI300X | NVIDIA H100 |
|--------|-----------|-------------|
| Warp/Wavefront | 64 threads | 32 threads |
| Shared Memory | LDS, 64 KB/CU | SMEM, 228 KB/SM |
| Matrix Instruction | MFMA (per-wavefront) | WGMMA (per-warpgroup, 128 threads) |
| Data Load Accelerator | None (explicit copies) | TMA (hardware DMA) |
| HBM Bandwidth | 5.3 TB/s (HBM3) | 3.35 TB/s (HBM3) |
| Compute Units | 304 CUs | 132 SMs |
| FP16 Peak | 1307.4 TFLOPS | 989.4 TFLOPS |
| Pipeline stages for FA | 1 (recommended) | 2-4 (typical) |

### Implications for Flash Attention Optimization
1. **Higher HBM bandwidth** means MI300X can tolerate more memory traffic — the crossover point where attention becomes compute-bound shifts to larger dimensions
2. **No TMA** means producer-consumer warp specialization patterns from FA3/FA4 need adaptation — data loading uses explicit wavefront cooperative copies
3. **64-thread wavefronts** change reduction patterns — warp shuffles cover more threads natively
4. **num_stages=1** recommended for fused dual-GEMM kernels (different from NVIDIA's multi-stage pipelining)
5. **304 CUs** (vs 132 SMs) means more parallelism available — different occupancy calculus

## References
- [TileLang Flash Attention on MI300X (AMD ROCm Blog)](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html)
- [AMD Instinct MI300X Workload Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
- [AMD CDNA3 ISA Reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
