# AMD ROCm Profiling for Flash Attention Kernels

## Overview

Profiling Flash Attention kernels on AMD Instinct GPUs requires a different tool suite and mental model than NVIDIA's Nsight ecosystem. This document provides a comprehensive guide to using ROCm profiling tools -- rocprof, Omniperf, and Omnitrace -- to identify scheduling, pipelining, and memory bottlenecks in attention kernels on MI300X (CDNA3) and MI250X (CDNA2) GPUs.

The core challenge is the same as on NVIDIA: attention kernels interleave matrix multiplication (MFMA) with softmax (scalar/vector ALU), and the profiler must reveal which pipeline stage is the bottleneck. However, AMD's architecture differs in critical ways: 64-thread wavefronts, LDS instead of SMEM, MFMA instead of WGMMA, no TMA hardware, and a chiplet-based XCD topology that adds a load-balancing dimension.

---

## 1. ROCm Profiling Tool Suite

### 1.1 Omnitrace (System-Wide Tracing)

Omnitrace is the AMD equivalent of NVIDIA Nsight Systems. It provides CPU/GPU timeline tracing, HIP API tracing, and kernel-level duration measurements.

**Primary use**: Identify which kernel is the bottleneck before diving into detailed profiling.

```bash
# Instrumentation-based tracing (more detail, higher overhead)
omnitrace-instrument -- python model.py

# Sampling-based tracing (lower overhead, suitable for production)
omnitrace-sample --trace-gpu -- python model.py

# Output: perfetto trace file (.proto)
# Open in https://ui.perfetto.dev or chrome://tracing
```

**Key metrics from Omnitrace**:
- Kernel duration breakdown (which kernels dominate wall time)
- HIP API call patterns (launch overhead, memcpy timing)
- Multi-GPU synchronization patterns
- CPU-GPU overlap analysis

**Attention-specific usage**:
```bash
# Trace only attention-related kernels
export OMNITRACE_TRACE_GPU=ON
export OMNITRACE_TRACE_HIP=ON
export OMNITRACE_FILTER_INCLUDE="fmha|attention|flash"

omnitrace-sample -- python inference.py
```

### 1.2 rocprof (ROCProfiler) -- Hardware Counter Collection

rocprof is the fundamental counter collection tool, analogous to NVIDIA's `ncu` (Nsight Compute) at the counter level. It reads hardware Performance Monitor Counters (PMCs) from the GPU.

#### rocprof v1 (Text-Based)

```bash
# Create counter specification file
cat > counters.txt << 'EOF'
# Pass 1: Compute utilization
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VALU_MFMA SQ_BUSY_CYCLES GRBM_GUI_ACTIVE

# Pass 2: Memory hierarchy
pmc: TCC_EA_RDREQ_32B_sum TCC_EA_WRREQ_32B_sum TCC_HIT_sum TCC_MISS_sum

# Pass 3: LDS and stalls
pmc: SQ_LDS_BANK_CONFLICT SQ_WAIT_INST_VMEM SQ_WAIT_INST_LDS SQ_INSTS_LDS

# Pass 4: Special function units
pmc: SQ_INSTS_VALU_TRANS SQ_INSTS_SALU SQ_INSTS_SMEM
EOF

# Profile with kernel name filter
rocprof -i counters.txt --kernel-name "attention" -o results.csv python script.py
```

**Important limitations**:
- rocprof v1 can only collect ~4-8 PMCs per pass due to hardware counter multiplexing
- Multiple passes are required for comprehensive profiling (kernel must be deterministic)
- Counter names differ between CDNA2 and CDNA3

#### rocprof v2 (Programmatic API)

```bash
# rocprofv2 provides more flexible counter collection
rocprofv2 --hip-trace --kernel-trace \
    --pmc SQ_WAVES,SQ_INSTS_VALU_MFMA,SQ_LDS_BANK_CONFLICT \
    -- python script.py
```

rocprof v2 also supports:
- Callback-based profiling via C API
- Per-dispatch counter collection
- Integration with ROCm runtime events

### 1.3 Omniperf (Roofline-Guided Analysis)

Omniperf is the most important tool for attention kernel optimization. It is AMD's equivalent of Nsight Compute's guided analysis, providing roofline models, derived metrics, and bottleneck identification.

```bash
# Profile: collects counters across multiple auto-generated passes
omniperf profile -n flash_attn_profile -- python model.py

# Analyze: compute derived metrics and display results
omniperf analyze -p workloads/flash_attn_profile/MI300X/

# Alternatively, launch web dashboard
omniperf analyze -p workloads/flash_attn_profile/MI300X/ --gui
```

**Omniperf analysis sections relevant to attention**:

| Section | Key Metrics | Attention Relevance |
|---------|------------|---------------------|
| **System Speed-of-Light** | % of peak compute, % of peak bandwidth | Overall kernel efficiency |
| **Compute Unit (CU)** | MFMA utilization, VALU utilization, SALU utilization | Is MFMA saturated? Is softmax (VALU) dominating? |
| **Local Data Share (LDS)** | LDS bandwidth, bank conflicts, utilization | Are Q/K tile accesses causing bank conflicts? |
| **L2 Cache (TCC)** | Hit rate, bandwidth, read/write ratio | Is K/V reuse effective? |
| **HBM (EA)** | Achieved bandwidth, % of peak 5.3 TB/s | Is the kernel memory-bound? |
| **Wavefront** | Occupancy, stall reasons, active wavefronts/SIMD | What limits parallelism? |
| **Instruction Mix** | MFMA/VALU/SALU/LDS/VMEM counts | Compute vs memory instruction balance |

---

## 2. AMD-Specific Performance Counters for Attention

### 2.1 Compute Unit Counters

| Counter | Description | Attention Usage |
|---------|-------------|-----------------|
| `SQ_WAVES` | Total wavefronts dispatched | Overall kernel activity |
| `SQ_INSTS_VALU` | Vector ALU instructions | Softmax element-wise ops |
| `SQ_INSTS_VALU_MFMA` | MFMA instructions | QK^T and PV matrix multiplies |
| `SQ_INSTS_VALU_TRANS` | Transcendental instructions (exp, rcp, rsq) | Softmax exp and reciprocal |
| `SQ_BUSY_CYCLES` | Cycles CU is active | CU utilization |
| `SQ_WAIT_INST_VMEM` | Cycles waiting for vector memory | HBM stalls |
| `SQ_WAIT_INST_LDS` | Cycles waiting for LDS | LDS bank conflict stalls |
| `SQ_WAIT_INST_VALU` | Cycles waiting for VALU | MFMA/VALU pipeline stalls |

### 2.2 LDS Counters

| Counter | Description | Attention Usage |
|---------|-------------|-----------------|
| `SQ_INSTS_LDS` | LDS instructions issued | Tile load/store frequency |
| `SQ_LDS_BANK_CONFLICT` | LDS bank conflict events | K/V tile access pattern quality |
| `TA_LDS_WAVEFRONTS` | Wavefronts accessing LDS | LDS utilization |

### 2.3 Memory Hierarchy Counters

| Counter | Description | Attention Usage |
|---------|-------------|-----------------|
| `TCC_EA_RDREQ_32B_sum` | 32B read requests to HBM (all L2 partitions) | Total HBM read traffic |
| `TCC_EA_WRREQ_32B_sum` | 32B write requests to HBM | Total HBM write traffic |
| `TCC_HIT_sum` | L2 cache hits | K/V reuse efficiency |
| `TCC_MISS_sum` | L2 cache misses | Cold misses / thrashing |
| `TCC_READ_sum` | Total L2 reads | L2 traffic volume |

### 2.4 Derived Metrics (Computed by Omniperf)

```
MFMA Utilization (%) = SQ_INSTS_VALU_MFMA / (SQ_BUSY_CYCLES * mfma_throughput_per_cycle) * 100

HBM Bandwidth (GB/s) = (TCC_EA_RDREQ_32B_sum + TCC_EA_WRREQ_32B_sum) * 32 / kernel_time_ns

LDS Bank Conflict Rate (%) = SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS * 100

Wavefront Occupancy = SQ_WAVES / (num_CUs * max_waves_per_CU * kernel_cycles)

Arithmetic Intensity (FLOP/byte) = total_MFMA_flops / total_HBM_bytes
```

---

## 3. Profiling Workflow for Attention Kernels

### 3.1 Phase 1: Timeline Analysis with Omnitrace

```bash
# Trace the full inference pipeline
omnitrace-sample --trace-gpu --output-path ./trace_results -- python serve.py --batch 1

# Open trace and identify:
# 1. Which kernel(s) dominate execution time
# 2. Whether attention is a single fused kernel or multiple kernels
# 3. CPU-GPU synchronization gaps
# 4. Memory transfer overhead
```

**What to look for in the timeline**:
- Is the attention kernel a single long-running dispatch or many short dispatches?
- Are there unexpected gaps between kernel launches (HIP API overhead)?
- For split-KV decode: are the partial attention and reduction kernels balanced?

### 3.2 Phase 2: Roofline Positioning with Omniperf

```bash
omniperf profile -n attn_roofline -- python model.py
omniperf analyze -p workloads/attn_roofline/MI300X/ --gui
```

**Navigate to the roofline chart**:
- Plot shows kernel as a point in (arithmetic intensity, throughput) space
- MI300X has two roofline ceilings:
  - Compute ceiling: 1307.4 TFLOPS (FP16 MFMA peak)
  - Memory ceiling: 5.3 TB/s (HBM3 peak)
- Crossover point at ~246 FLOP/byte

**Attention kernel positioning**:
- **Prefill (long sequences)**: Typically compute-bound, should be near compute ceiling
- **Decode (short Q, long KV)**: Typically memory-bound, should be near memory ceiling
- **If far from both ceilings**: Scheduling/pipeline inefficiency (the focus of L2 optimization)

### 3.3 Phase 3: Detailed Counter Analysis

Based on roofline position, focus on the appropriate counters:

#### If Compute-Bound (Near Compute Ceiling)

Focus on MFMA utilization and instruction mix:

```bash
# Collect compute-focused counters
cat > compute_counters.txt << 'EOF'
pmc: SQ_INSTS_VALU_MFMA SQ_INSTS_VALU SQ_INSTS_VALU_TRANS SQ_BUSY_CYCLES
pmc: SQ_WAIT_INST_VALU SQ_INSTS_SALU SQ_WAIT_OTHER
EOF

rocprof -i compute_counters.txt --kernel-name "attention" python script.py
```

**Key questions**:
1. What fraction of VALU instructions are MFMA? (target: >50%)
2. How many transcendental instructions (exp, rcp) per MFMA? (softmax overhead)
3. What are the primary stall reasons? (VALU wait = MFMA pipeline full, good; other waits = scheduling issue)

#### If Memory-Bound (Near Memory Ceiling)

Focus on bandwidth utilization and cache efficiency:

```bash
# Collect memory-focused counters
cat > memory_counters.txt << 'EOF'
pmc: TCC_EA_RDREQ_32B_sum TCC_EA_WRREQ_32B_sum TCC_HIT_sum TCC_MISS_sum
pmc: SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT SQ_WAIT_INST_LDS SQ_WAIT_INST_VMEM
EOF

rocprof -i memory_counters.txt --kernel-name "attention" python script.py
```

**Key questions**:
1. Is achieved HBM bandwidth close to 5.3 TB/s? (if not, pipeline stalls)
2. Is L2 hit rate reasonable for the access pattern? (MHA: K/V reused across heads; GQA: shared KV heads should have high L2 hits)
3. Are LDS bank conflicts significant? (K tile layout issue)

#### If Far From Both Ceilings (Scheduling Issue)

Focus on wavefront scheduling and occupancy:

```bash
# Collect scheduling-focused counters
cat > sched_counters.txt << 'EOF'
pmc: SQ_WAVES SQ_BUSY_CYCLES GRBM_GUI_ACTIVE SQ_LEVEL_WAVES
pmc: SQ_WAIT_INST_VMEM SQ_WAIT_INST_LDS SQ_WAIT_INST_VALU SQ_WAIT_OTHER
pmc: SQ_THREAD_CYCLES_VALU SQ_ACTIVE_INST_SCA SQ_ACTIVE_INST_VALU
EOF

rocprof -i sched_counters.txt --kernel-name "attention" python script.py
```

**Key questions**:
1. Is wavefront occupancy limited by VGPRs, LDS, or workgroup size?
2. Are wavefronts stalled waiting for memory (VMEM), LDS, or barriers?
3. Is there XCD load imbalance (check per-XCD counters)?

---

## 4. AMD vs NVIDIA Profiling Comparison

### 4.1 Tool Mapping

| Task | NVIDIA | AMD |
|------|--------|-----|
| System timeline | Nsight Systems (nsys) | Omnitrace |
| Kernel profiling | Nsight Compute (ncu) | Omniperf + rocprof |
| Hardware counters | ncu `--set full` | rocprof PMC collection |
| Roofline analysis | ncu roofline section | Omniperf roofline |
| Occupancy analysis | ncu occupancy calculator | Omniperf wavefront section |
| Bank conflict detection | ncu SMEM section | rocprof `SQ_LDS_BANK_CONFLICT` |
| Warp/wavefront stalls | ncu warp stall reasons | rocprof `SQ_WAIT_*` counters |
| Memory throughput | ncu memory section | Omniperf HBM/L2 section |
| Instruction mix | ncu instruction section | rocprof `SQ_INSTS_*` counters |
| GUI analysis | Nsight Compute GUI | Omniperf web dashboard |

### 4.2 Counter Mapping

| Metric | NVIDIA Counter | AMD Counter |
|--------|---------------|-------------|
| Tensor core activity | `sm__inst_executed_pipe_tensor` | `SQ_INSTS_VALU_MFMA` |
| Shared memory conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` | `SQ_LDS_BANK_CONFLICT` |
| Global memory reads | `dram__bytes_read` | `TCC_EA_RDREQ_32B_sum * 32` |
| Global memory writes | `dram__bytes_write` | `TCC_EA_WRREQ_32B_sum * 32` |
| L2 hit rate | `lts__t_sector_hit_rate` | `TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)` |
| Active warps/wavefronts | `sm__warps_active` | `SQ_LEVEL_WAVES` |
| Special math (exp) | `sm__inst_executed_pipe_xu` (MUFU) | `SQ_INSTS_VALU_TRANS` |
| Memory stalls | `smsp__warp_stall_reasons_mio_throttle` | `SQ_WAIT_INST_VMEM` |

### 4.3 Key Differences in Interpretation

1. **Wavefront granularity**: AMD wavefronts are 64 threads vs NVIDIA warps at 32 threads. Occupancy calculations must account for this: 4 active wavefronts per SIMD provides similar latency-hiding to 8 active warps per sub-partition on NVIDIA.

2. **MFMA vs WGMMA**: MFMA operates per-wavefront (64 threads), while WGMMA operates per-warpgroup (128 threads). MFMA utilization is reported per-CU, while WGMMA utilization is reported per-SM.

3. **No TMA stalls**: AMD has no TMA hardware, so there is no equivalent of producer-warp TMA stalls. Instead, look for VMEM wait stalls which indicate global memory load latency.

4. **LDS vs SMEM capacity**: MI300X LDS is 64 KB/CU vs H100 SMEM 228 KB/SM. Attention tile sizes must be smaller on AMD, and occupancy analysis must account for different LDS/SMEM budgets.

5. **XCD-level analysis**: MI300X's 8-XCD chiplet design means per-CU metrics can show bimodal distributions if workload is unevenly distributed. Always check XCD-level breakdown in Omniperf.

6. **Counter multiplexing**: Both platforms require multiple passes for comprehensive profiling, but AMD's rocprof typically needs more passes (4-8 PMCs per pass) than NVIDIA's ncu (which can collect more counters per pass with hardware support).

---

## 5. Common Attention Bottleneck Patterns on AMD

### 5.1 Low MFMA Utilization Due to Softmax Overhead

**Symptom**: MFMA utilization <40%, high `SQ_INSTS_VALU_TRANS` (transcendental instructions).

**Root cause**: Softmax operations (exp, max, sum, reciprocal) execute on the scalar/vector ALU, not the MFMA units. The MFMA pipeline stalls waiting for softmax results before the next QK^T or PV multiply can begin.

**Diagnosis**:
```bash
# Check instruction mix
rocprof --pmc SQ_INSTS_VALU_MFMA,SQ_INSTS_VALU_TRANS,SQ_INSTS_VALU \
    --kernel-name "attention" -- python script.py
```

If `SQ_INSTS_VALU_TRANS` > 20% of total VALU instructions, softmax is a significant bottleneck.

**Mitigation**:
- Increase tile size to amortize softmax cost over more MFMA operations
- Defer normalization (divide by l) to kernel epilogue
- Consider approximate softmax for inference (polynomial exp)

### 5.2 LDS Bank Conflicts from K/V Tile Layout

**Symptom**: High `SQ_LDS_BANK_CONFLICT`, LDS bandwidth well below theoretical peak.

**Root cause**: K/V tiles stored in LDS with row-major layout can cause systematic bank conflicts when accessed column-wise for the QK^T computation. This is the exact same problem as NVIDIA SMEM bank conflicts described in the Nsight profiling entry.

**Diagnosis**:
```bash
rocprof --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
    --kernel-name "attention" -- python script.py

# Bank conflict rate = SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS
# > 20% indicates significant bank conflict overhead
```

**Mitigation**:
- Transpose K tiles before storing in LDS (column-major for QK^T access)
- Apply LDS swizzling (`T.use_swizzle` in TileLang)
- Pad LDS allocations to break conflict patterns (add 1-2 bank padding)

### 5.3 HBM Bandwidth Underutilization

**Symptom**: Kernel is memory-bound per roofline but achieved HBM bandwidth is <70% of 5.3 TB/s peak.

**Root cause**: Possible causes include insufficient wavefront occupancy to saturate memory pipeline, uncoalesced global memory accesses, or suboptimal L2 cache utilization.

**Diagnosis**:
```bash
# Check achieved bandwidth
omniperf analyze -p workloads/attn_profile/MI300X/ --block 7.1  # HBM section

# Check occupancy
omniperf analyze -p workloads/attn_profile/MI300X/ --block 2.1  # Wavefront section
```

**Mitigation**:
- Increase wavefront occupancy: reduce VGPR usage (smaller tiles or fewer accumulators)
- Ensure memory accesses are coalesced (64 consecutive threads access 64 consecutive 4-byte values)
- Use `num_split_q` to increase parallelism for decode-phase attention

### 5.4 XCD Load Imbalance

**Symptom**: Average CU utilization is moderate, but per-XCD utilization shows high variance. Some XCDs are busy while others are idle.

**Root cause**: Kernel grid dimensions do not evenly divide across 8 XCDs (38 CUs each on MI300X). If grid size is not a multiple of 304 CUs, tail effects cause imbalance.

**Diagnosis**:
```bash
# Check per-XCD metrics in Omniperf
omniperf analyze -p workloads/attn_profile/MI300X/ --block 2  # CU section

# Look for high variance in SQ_BUSY_CYCLES across CUs
```

**Mitigation**:
- Pad grid dimensions to be multiples of 8 (or 304 for full balance)
- Use CTA rasterization patterns that distribute work evenly across XCDs
- For GQA: distribute KV head groups evenly across XCDs

### 5.5 VGPR-Limited Occupancy

**Symptom**: Wavefront occupancy is low, and Omniperf identifies VGPRs as the limiter.

**Root cause**: FP32 accumulators for attention output O and intermediate scores S consume large VGPR allocations. Each MFMA output tile requires 4-16 VGPRs depending on tile shape. With dual-GEMM (QK^T + PV), both sets of accumulators may be live simultaneously.

**Diagnosis**:
```bash
# Check VGPR allocation per workgroup
omniperf analyze -p workloads/attn_profile/MI300X/ --block 2.2  # Resource section
```

On MI300X, each CU has 256 VGPRs per thread and 256 AGPRs per thread:
- Maximum wavefronts per CU depends on VGPR allocation
- VGPR > 128 per thread: max 2 wavefronts
- VGPR > 64 per thread: max 4 wavefronts
- VGPR <= 64 per thread: max 8 wavefronts

**Mitigation**:
- Use AGPR (accumulator registers) for MFMA outputs -- these do not consume VGPR budget
- Reduce tile sizes to lower accumulator requirements
- Accept lower occupancy if compute throughput is sufficient (test both configurations)

---

## 6. Example: Profiling Composable Kernel Flash Attention on MI300X

### 6.1 Baseline Profile

```bash
# Step 1: Timeline
omnitrace-sample --trace-gpu -- python benchmark_ck_fmha.py \
    --batch 4 --heads 32 --seqlen 4096 --headdim 128

# Step 2: Roofline
omniperf profile -n ck_fmha_baseline -- python benchmark_ck_fmha.py \
    --batch 4 --heads 32 --seqlen 4096 --headdim 128

omniperf analyze -p workloads/ck_fmha_baseline/MI300X/ --gui
```

### 6.2 Interpreting Results

**Expected roofline position for prefill (B=4, H=32, S=4096, d=128)**:
- Arithmetic intensity: ~128 FLOP/byte (2*S*d FLOPs per S*d bytes)
- Position: Near compute-bound region
- Target: >60% MFMA utilization

**Expected roofline position for decode (B=4, H=32, S=4096, d=128, Q_len=1)**:
- Arithmetic intensity: ~0.5 FLOP/byte (memory-bound)
- Position: Near memory bandwidth ceiling
- Target: >70% HBM bandwidth utilization

### 6.3 Optimization Loop

```
1. Profile → Identify primary bottleneck
2. If MFMA-limited:
   → Check softmax instruction fraction
   → Try larger tile sizes, deferred normalization
3. If LDS-limited:
   → Check bank conflict counter
   → Apply swizzling or layout transpose
4. If HBM-limited:
   → Check occupancy (increase if <4 wavefronts/SIMD)
   → Check L2 hit rate (improve KV reuse)
5. If XCD-imbalanced:
   → Adjust grid dimensions
   → Improve rasterization pattern
6. Re-profile → Verify improvement → Repeat
```

---

## 7. Omniperf Command Reference for Attention

```bash
# Full profile (all available counter groups)
omniperf profile -n my_kernel -- python script.py

# Analyze specific sections
omniperf analyze -p workloads/my_kernel/MI300X/ --block 0    # System SoL
omniperf analyze -p workloads/my_kernel/MI300X/ --block 2    # CU metrics
omniperf analyze -p workloads/my_kernel/MI300X/ --block 5    # LDS
omniperf analyze -p workloads/my_kernel/MI300X/ --block 7    # HBM
omniperf analyze -p workloads/my_kernel/MI300X/ --block 10   # Instruction mix
omniperf analyze -p workloads/my_kernel/MI300X/ --block 12   # Wavefront

# Filter to specific kernel
omniperf analyze -p workloads/my_kernel/MI300X/ --filter-kernel-names "fmha"

# Compare two profiles (before/after optimization)
omniperf analyze -p workloads/baseline/MI300X/ -p workloads/optimized/MI300X/

# Launch web dashboard
omniperf analyze -p workloads/my_kernel/MI300X/ --gui --port 8080
```

---

## 8. rocprof Counter Collection Quick Reference

### Most Important Counters for Attention Kernels

```bash
# Quick check: is attention compute-bound or memory-bound?
rocprof --pmc SQ_INSTS_VALU_MFMA,TCC_EA_RDREQ_32B_sum,GRBM_GUI_ACTIVE \
    --kernel-name "attention" -- python script.py

# Compute utilization breakdown
rocprof --pmc SQ_INSTS_VALU_MFMA,SQ_INSTS_VALU_TRANS,SQ_INSTS_VALU,SQ_INSTS_SALU \
    --kernel-name "attention" -- python script.py

# Memory hierarchy
rocprof --pmc TCC_EA_RDREQ_32B_sum,TCC_EA_WRREQ_32B_sum,TCC_HIT_sum,TCC_MISS_sum \
    --kernel-name "attention" -- python script.py

# LDS efficiency
rocprof --pmc SQ_INSTS_LDS,SQ_LDS_BANK_CONFLICT,SQ_WAIT_INST_LDS \
    --kernel-name "attention" -- python script.py

# Wavefront scheduling
rocprof --pmc SQ_WAVES,SQ_BUSY_CYCLES,SQ_WAIT_INST_VMEM,SQ_WAIT_INST_VALU \
    --kernel-name "attention" -- python script.py
```

### Computing Derived Metrics from Raw Counters

```python
import pandas as pd

# Read rocprof CSV output
df = pd.read_csv("results.csv")

# MFMA utilization (approximate)
# Each MFMA instruction takes N cycles depending on tile shape
# MFMA_F32_16x16x16_F16: 16 cycles
# MFMA_F32_32x32x8_F16: 32 cycles
mfma_cycles = df['SQ_INSTS_VALU_MFMA'] * 16  # assuming 16x16x16
total_cycles = df['SQ_BUSY_CYCLES']
mfma_utilization = mfma_cycles / total_cycles * 100

# HBM bandwidth (GB/s)
hbm_bytes = (df['TCC_EA_RDREQ_32B_sum'] + df['TCC_EA_WRREQ_32B_sum']) * 32
kernel_time_ns = df['DurationNs']
hbm_bandwidth_gbps = hbm_bytes / kernel_time_ns  # GB/s

# LDS bank conflict rate
lds_conflict_rate = df['SQ_LDS_BANK_CONFLICT'] / df['SQ_INSTS_LDS'] * 100

# L2 cache hit rate
l2_hit_rate = df['TCC_HIT_sum'] / (df['TCC_HIT_sum'] + df['TCC_MISS_sum']) * 100

print(f"MFMA utilization: {mfma_utilization:.1f}%")
print(f"HBM bandwidth: {hbm_bandwidth_gbps:.0f} GB/s (peak: 5300 GB/s)")
print(f"LDS bank conflict rate: {lds_conflict_rate:.1f}%")
print(f"L2 hit rate: {l2_hit_rate:.1f}%")
```

---

## 9. Practical Tips

### 9.1 Ensure Deterministic Kernels for Multi-Pass Profiling

Both rocprof and Omniperf require multiple kernel replay passes for comprehensive counter collection. The kernel must produce the same behavior on each pass:

```python
# Fix random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Disable any dynamic behavior (dynamic batching, variable-length padding)
# Use fixed input shapes and deterministic algorithms
```

### 9.2 Isolate the Attention Kernel

```bash
# Use kernel name filtering to avoid profiling unrelated kernels
rocprof --kernel-name "fmha" ...
omniperf profile -n attn -- python script.py
omniperf analyze -p workloads/attn/MI300X/ --filter-kernel-names "fmha"
```

### 9.3 Profile at Target Batch Size

AMD profiling should match the deployment batch size because:
- Batch size determines SM/CU occupancy (affects whether you need split-KV)
- L2 cache behavior changes with batch size (more batches = more cache pressure)
- XCD load balancing depends on grid dimensions

### 9.4 Note on Omniperf Deprecation

As of ROCm 6.x, Omniperf is being renamed to "AMD Instinct MI Profiler" (or "rocprof --perfmon" mode). The core functionality remains the same, but command-line options may change. Check the latest ROCm documentation for current tool names.

---

## References

- [ROCm Profiler (rocprof) Documentation](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [Omniperf Documentation](https://rocm.docs.amd.com/projects/omniperf/en/latest/)
- [Omnitrace Documentation](https://rocm.docs.amd.com/projects/omnitrace/en/latest/)
- [AMD MI300/MI200 Performance Counters](https://instinct.docs.amd.com/latest/gpu-arch/mi300-mi200-performance-counters.html)
- [AMD CDNA3 ISA Reference (MFMA Instructions)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [ROCm for AI: Profiling Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/profiling.html)
- [TileLang Flash Attention on MI300X](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html)
