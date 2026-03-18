# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

**Source:** https://tridao.me/publications/flash3/flash3.pdf
**Authors:** Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao
**Affiliations:** Colfax Research, Meta, NVIDIA, Georgia Tech, Princeton University, Together AI
**Date:** July 12, 2024

## Abstract

Attention, as a core layer of the ubiquitous Transformer architecture, is the bottleneck for large language models and long-context applications. FlashAttention elaborated an approach to speed up attention on GPUs through minimizing memory reads/writes. However, it has yet to take advantage of new capabilities present in recent hardware, with FlashAttention-2 achieving only 35% utilization on the H100 GPU. FlashAttention-3 develops three main techniques to speed up attention on Hopper GPUs:

1. **Overlap overall computation and data movement** via warp-specialization and TMA
2. **Interleave block-wise matmul and softmax operations**
3. **Block quantization and incoherent processing** that leverages hardware support for FP8 low-precision

FlashAttention-3 achieves speedup on H100 GPUs by 1.5-2.0x with FP16 reaching up to 740 TFLOPs/s (75% utilization), and with FP8 reaching close to 1.2 PFLOPs/s. FP8 FlashAttention-3 achieves 2.6x lower numerical error than a baseline FP8 attention.

## 1. Introduction

FlashAttention-2 achieves poor utilization on newer GPUs relative to optimized GEMM kernels (35% vs. 80-90% on the Hopper H100 GPU). This is attributed to implementation-level differences, such as not using Hopper-specific instructions in place of Ampere ones when targeting Tensor Cores.

Asynchrony is a result of hardware specialization: specific hardware units performing matrix multiplication (Tensor Cores) or memory loading (TMA), separate from the rest of the CUDA cores performing logic, integer, and floating point computation.

### Three Key Ideas

1. **Producer-Consumer asynchrony:** A warp-specialized software pipelining scheme that exploits the asynchronous execution of data movement and Tensor Cores by splitting producers and consumers of data into separate warps, thereby extending the algorithm's ability to hide memory and instruction issue latencies.

2. **Hiding softmax under asynchronous block-wise GEMMs:** Overlap the comparatively low-throughput non-GEMM operations involved in softmax with the asynchronous WGMMA instructions for GEMM. In the 2-stage version, softmax executes on one block of the scores matrix while WGMMA executes in the asynchronous proxy to compute the next block.

3. **Hardware-accelerated low-precision GEMM:** Adapt the forward pass algorithm to target FP8 Tensor Cores for GEMM, nearly doubling the measured TFLOPs/s.

## 2. Background: GPU Hardware and Execution Model

### Memory Hierarchy (H100 SXM5)

| Hardware Level | Parallel Agent | Data Locale | Capacity @ Bandwidth |
|---|---|---|---|
| Chip | Grid | GMEM | 80 GiB @ 3.35 TB/s |
| GPC | Threadblock Clusters | L2 | 50 MiB @ 12 TB/s |
| SM | Threadblock (CTA) | SMEM | 228 KiB per SM, 31TB/s per GPU |
| Thread | Thread | RMEM | 256 KiB per SM |

### Thread Hierarchy

From finest to coarsest level: threads (individual), warps (32 threads), warpgroups (4 contiguous warps = 128 threads), threadblocks (CTAs), threadblock clusters, and grids.

- Threads in the same CTA are co-scheduled on the same SM
- SMEM is directly addressable by all threads within a CTA
- Each thread has at most 256 registers (RMEM) private to itself

### Asynchrony and Warp-Specialization

GPUs rely on concurrency and asynchrony to hide memory and execution latencies:

- **TMA (Tensor Memory Accelerator):** Dedicated hardware unit for async memory copy between GMEM and SMEM
- **WGMMA:** Unlike Ampere, the Tensor Core of Hopper is asynchronous and can source its inputs directly from shared memory
- **Warp-specialized kernels:** Warps of a CTA are divided into producer or consumer roles -- producers issue data movement, consumers issue computation
- **`setmaxnreg`:** Hopper supports dynamic reallocation of registers between warpgroups, so compute warps doing MMAs can obtain a larger share of RMEM than load warps

## 3. FlashAttention-3 Algorithm

### 3.1 Producer-Consumer Asynchrony Through Warp-Specialization and Pingpong Scheduling

#### Warp-Specialization Scheme

The forward pass operates at CTA-level, processing a tile Q_i of the query matrix to compute the corresponding output tile O_i.

**Producer warpgroup:**
- Deallocates registers (needs minimal state)
- Issues loads of Q_i from HBM to shared memory via TMA
- In main loop: loads K_j, V_j from HBM to SMEM at (j % s)th stage of circular buffer
- Signals consumers upon completion via barrier commit
- Waits for consumers to release buffer stages before reuse

**Consumer warpgroup:**
- Reallocates registers as function of number of consumer warps
- Initializes O_i = 0, running statistics l_i = 0, m_i = -infinity
- Main loop for j = 0 to T_c:
  - Wait for K_j to be loaded in shared memory
  - Compute S_i^(j) = Q_i * K_j^T (SS-GEMM via WGMMA). Commit and wait.
  - Store m_i^old, compute new m_i = max(m_i^old, rowmax(S_i^(j)))
  - Compute P_i^(j) = exp(S_i^(j) - m_i) and l_i = exp(m_i^old - m_i) * l_i + rowsum(P_i^(j))
  - Wait for V_j to be loaded in shared memory
  - Compute O_i = diag(exp(m_i^old - m_i))^-1 * O_i + P_i^(j) * V_j (RS-GEMM). Commit and wait.
  - Release the buffer stage for the producer
- Epilogue: O_i = diag(l_i)^-1 * O_i, compute L_i = m_i + log(l_i)
- Write O_i and L_i to HBM

Implementation uses `setmaxnreg` for register (de)allocation, TMA for loads of Q_i and {K_j, V_j}, and WGMMA to execute the GEMMs (SS prefix = both operands from SMEM, RS prefix = one from registers).

#### Pingpong Scheduling

The H100 SXM5 GPU has 989 TFLOPS of FP16 matmul but only 3.9 TFLOPS of special functions (exponential, necessary for softmax). For FP16 with head dimension 128, there are 512x more matmul FLOPS compared to exponential operations, but the exponential has 256x lower throughput, so exponential can take 50% of the cycle compared to matmul.

**Solution:** Use synchronization barriers (`bar.sync`) to force the GEMMs of one warpgroup to be scheduled before the GEMMs of another. As a result, the softmax of warpgroup 1 will be scheduled while warpgroup 2 is performing its GEMMs. Then the roles swap:

```
Warpgroup 1: [GEMM0 Softmax GEMM1] [GEMM0 Softmax GEMM1] ...
Warpgroup 2:     [GEMM0 Softmax GEMM1] [GEMM0 Softmax GEMM1] ...
```

The softmax of one warpgroup executes during the GEMMs of the other warpgroup (hence "pingpong" scheduling).

**Result:** Improves from 570 TFLOPS to 620-640 TFLOPS for FP16 forward with head dimension 128 and sequence length 8192.

### 3.2 Intra-Warpgroup Overlapping GEMMs and Softmax

Even within one warpgroup, softmax instructions can overlap with GEMM instructions via a 2-stage GEMM-softmax pipelining algorithm:

```
WGMMA0: [  0  ] [  1  ] [  2  ] ... [ N-1 ]
Softmax:   [ 0  ] [  1  ] [  2  ] ... [ N-1 ]
WGMMA1:      [ 0  ] [  1  ] ... [ N-2 ][ N-1 ]
```

The second WGMMA of iteration j (WGMMA1 computing O_i += P * V) is overlapped with softmax operations from iteration j+1 (computing on S_next).

**Consumer warpgroup forward pass (Algorithm 2):**
1. Initialize O_i = 0, compute initial S_cur = Q_i * K_0^T
2. Compute softmax statistics and rescale O_i
3. Main loop: compute S_next = Q_i * K_j^T (commit but do not wait), then compute O_i = O_i + P_cur * V_{j-1} (commit but do not wait), then compute softmax on S_next while waiting for WGMMA
4. Epilogue: final rescaling and write to HBM

**Practical considerations:**
- **Compiler reordering:** NVCC may disrupt the carefully crafted WGMMA and non-WGMMA pipelining sequence. SASS analysis confirms the 2-stage pipelining works as expected.
- **Register pressure:** The 2-stage pipeline requires additional registers to store S_next, leading to extra register usage of size B_r x B_c x sizeof(float) per threadblock.
- **3-stage pipelining:** Proposed but performs worse due to higher register pressure and compiler not overlapping the second WGMMA with softmax.

### 3.3 Low-Precision with FP8

- **Layout transformations:** FP8 WGMMA only supports k-major format. In-kernel transpose of V tiles uses LDSM/STSM instructions (warp collectively loading/storing SMEM to RMEM at 128-byte granularity).
- **Byte permute instructions** transform the FP32 accumulator layout to match FP8 operand A layout.
- **Block quantization:** One scalar per block of Q, K, V (size B_r x d or B_c x d) instead of per-tensor scaling.
- **Incoherent processing:** Multiply Q and K with a random orthogonal matrix M before quantizing to FP8, spreading out outliers. M = product of random diagonal matrices of +/-1 and a Hadamard matrix, computable in O(d log d).

## 4. Empirical Validation

### Benchmarking Attention (H100 80GB SXM5)

**FP16 Forward Pass:**
- FlashAttention-3 is 1.5-2.0x faster than FlashAttention-2
- Reaches up to 740 TFLOPs/s (75% of theoretical maximum)
- For medium and long sequences (1k+), surpasses cuDNN optimized for H100

**FP16 Backward Pass:**
- 1.5-1.75x faster than FlashAttention-2

**FP8 Forward Pass:**
- Reaches close to 1.2 PFLOPs/s
- Competitive with cuDNN for large sequence lengths

### Ablation Study: Pipelining (batch=4, seqlen=8448, nheads=16, hdim=128)

| Configuration | Time | TFLOPs/s |
|---|---|---|
| FlashAttention-3 (full) | 3.538 ms | 661 |
| No GEMM-Softmax Pipelining, With Warp-Specialization | 4.021 ms | 582 |
| GEMM-Softmax Pipelining, No Warp-Specialization | 4.105 ms | 570 |

Warp-specialization alone contributes ~12 TFLOPs/s; GEMM-softmax pipelining adds ~79 TFLOPs/s on top; combined they yield 91 TFLOPs/s improvement.

### Numerical Error (RMSE)

| Method | FP16 RMSE | FP8 RMSE |
|---|---|---|
| Baseline | 3.2e-4 | 2.4e-2 |
| FlashAttention-2 | 1.9e-4 | -- |
| FlashAttention-3 | 1.9e-4 | 9.1e-3 |

FP8 FlashAttention-3 is 2.6x more accurate than baseline FP8 (with block quantization and incoherent processing).

## 5. Backward Pass with Warp Specialization

Similar producer-consumer pattern as forward pass, with an additional **dQ-writer warp** role:

- **Producer warpgroup:** Loads K_j, V_j, then Q_i, dO_i tiles
- **Consumer warpgroup:** Computes all GEMMs for backward pass (5 matmuls per iteration)
- **dQ-writer warp:** Atomically accumulates local dQ contributions to global memory (avoids memory contention from multiple CTAs writing to the same dQ location)

## SASS Analysis (2-Stage Pipelining)

Analysis of compiled SASS code confirms:
1. Softmax is reordered to the very beginning, even before the first WGMMA
2. The first WGMMA is interleaved with softmax and FP32->FP16 datatype conversion of S (WGMMA and non-WGMMA execute in parallel)
3. exp2, row_sum, O rescaling and FP32->FP16 conversions are interleaved together
4. The second WGMMA is NOT overlapped with other instructions (as expected in 2-stage)

## Key Implementation Details

- Built on CUTLASS primitives for WGMMA and TMA abstractions
- Circular SMEM buffer with s stages for producer-consumer decoupling
- Pipeline object manages barrier synchronization
- `setmaxnreg` dynamically redistributes registers at kernel entry
- Open-source at https://github.com/Dao-AILab/flash-attention
