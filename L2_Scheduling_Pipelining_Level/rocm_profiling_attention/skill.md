---
skill_name: AMD ROCm Profiling for Flash Attention Kernels
description: Profiler-driven methodology for identifying scheduling, pipelining, and memory bottlenecks in attention kernels on AMD MI300X/MI250X using rocprof, Omniperf, and Omnitrace
level: L2 - Scheduling & Pipelining Level
target_hardware: AMD MI300X (CDNA3), MI250X (CDNA2)
relevance: When debugging Flash Attention kernel performance on AMD GPUs, diagnosing MFMA utilization gaps, LDS bank conflicts, HBM bandwidth saturation, or wavefront scheduling inefficiencies
---

# AMD ROCm Profiling for Flash Attention Kernels

## What It Is
A profiler-driven optimization methodology for Flash Attention kernels on AMD Instinct GPUs using the ROCm profiling tool suite: rocprof (v1/v2) for hardware counter collection, Omniperf for roofline-guided bottleneck analysis, and Omnitrace for system-wide timeline tracing. This approach mirrors the Nsight Compute methodology used on NVIDIA GPUs but adapts to AMD's CDNA architecture -- 64-thread wavefronts, MFMA matrix instructions, LDS (Local Data Share) instead of SMEM, and chiplet-based XCD topology. The methodology systematically identifies whether an attention kernel is limited by MFMA throughput, LDS bandwidth, HBM bandwidth, wavefront scheduling, or instruction mix issues.

## Key Concepts
- **rocprof (ROCProfiler)**: Low-level hardware counter collection tool. rocprof v1 uses simple text-based counter specification; rocprof v2 (rocprofiler-sdk) provides a programmatic C API with more flexible counter grouping. Both collect raw PMC (Performance Monitor Counter) values per kernel dispatch.
- **Omniperf (now AMD Instinct MI Profiler)**: High-level roofline analysis tool built on top of rocprof. It collects counters across multiple passes, computes derived metrics (MFMA utilization, LDS efficiency, HBM throughput), and presents them in a web dashboard or CLI. Omniperf is the AMD equivalent of Nsight Compute's guided analysis.
- **Omnitrace**: System-wide tracing tool (AMD's equivalent of Nsight Systems) providing CPU/GPU timeline, kernel launch latencies, HIP API tracing, and multi-GPU correlation. Use Omnitrace first for coarse-grained bottleneck identification.
- **MFMA utilization**: The fraction of cycles where MFMA (Matrix Fused Multiply-Add) units are active. For attention kernels, target >60% for compute-bound regimes. Measured via `SQ_INSTS_VALU_MFMA` counters.
- **LDS bandwidth and bank conflicts**: LDS has 32 banks with 4-byte granularity. Bank conflicts serialize wavefront memory access. Measured via `SQ_LDS_BANK_CONFLICT` counter. Critical for Q*K^T tile loads.
- **HBM throughput**: Measured via `TCC_EA_RDREQ` and `TCC_EA_WRREQ` counters at L2 cache level. MI300X theoretical peak is 5.3 TB/s; attention kernels should target >70% for memory-bound phases.
- **Wavefront occupancy**: Number of active wavefronts per SIMD unit. Limited by VGPR usage, LDS allocation, or workgroup size. Measured via `SQ_WAVES` and `SQ_BUSY_CYCLES`.
- **XCD load balancing**: MI300X has 8 XCDs with independent L2 caches. Uneven kernel dispatch across XCDs causes load imbalance visible in Omniperf's CU-level metrics.

## Scheduling Strategy / Pseudo-code
```
# ROCm PROFILING WORKFLOW FOR ATTENTION KERNELS:

# Step 0: System-wide timeline (identify which kernel is the bottleneck)
omnitrace-instrument -- python model.py
# or:
omnitrace-sample --trace-gpu -- python model.py
# Open perfetto trace: identify attention kernel duration vs total

# Step 1: Collect hardware counters with rocprof
# Option A: rocprof v1 (simple, text-based)
echo "pmc: SQ_WAVES SQ_INSTS_VALU_MFMA SQ_INSTS_VALU SQ_LDS_BANK_CONFLICT" > counters.txt
echo "pmc: TCC_EA_RDREQ_32B_sum TCC_EA_WRREQ_32B_sum TCC_HIT_sum TCC_MISS_sum" >> counters.txt
echo "pmc: SQ_WAIT_INST_VMEM SQ_WAIT_INST_LDS SQ_BUSY_CYCLES GRBM_GUI_ACTIVE" >> counters.txt
rocprof -i counters.txt --kernel-name "attention" python script.py

# Option B: rocprof v2 (programmatic, more flexible)
rocprofv2 --hip-trace --kernel-trace --pmc SQ_WAVES,SQ_INSTS_VALU_MFMA -- python script.py

# Step 2: Guided analysis with Omniperf
omniperf profile -n attention_profile -- python script.py
omniperf analyze -p workloads/attention_profile/MI300X/

# KEY METRICS TO CHECK IN OMNIPERF:

# 2a. Roofline position
#   -> Is the kernel compute-bound or memory-bound?
#   -> What is arithmetic intensity (FLOP/byte)?
#   -> For FA with d=128: expect memory-bound for decode, compute-bound for prefill

# 2b. Compute utilization
#   -> MFMA utilization (% of cycles with MFMA active)
#   -> VALU utilization (scalar FP ops for softmax)
#   -> SALU utilization (scalar/control flow)
#   -> Target: >60% MFMA for compute-bound attention

# 2c. Memory hierarchy
#   -> HBM bandwidth achieved vs peak (5.3 TB/s on MI300X)
#   -> L2 cache hit rate (expect high for K/V reuse in MHA)
#   -> LDS bandwidth and bank conflict rate
#   -> VGPR usage per wavefront (limits occupancy)

# 2d. Wavefront scheduling
#   -> Active wavefronts per SIMD
#   -> Stall reasons: VMEM wait, LDS wait, MFMA wait, barrier
#   -> Occupancy limiters: VGPRs, LDS, workgroup size

# 2e. Instruction mix
#   -> MFMA instructions vs VALU instructions
#   -> Special function unit usage (v_exp, v_rcp for softmax)
#   -> Branch divergence (causal masking)

# Step 3: Targeted optimization based on findings

# If MFMA utilization is low:
#   -> Check if softmax computation dominates (VALU-heavy instruction mix)
#   -> Check if LDS bank conflicts stall MFMA operand delivery
#   -> Check if HBM bandwidth is saturated (memory-bound, not compute-bound)

# If HBM bandwidth is saturated:
#   -> Increase tile size to improve arithmetic intensity
#   -> Check L2 hit rate for K/V reuse opportunities
#   -> Consider split-Q parallelism to improve multi-CU L2 sharing

# If LDS bank conflicts are high:
#   -> Profile bank conflict counter: SQ_LDS_BANK_CONFLICT
#   -> Check K/V tile layout in LDS (row-major vs column-major)
#   -> Apply swizzling: T.use_swizzle in TileLang, or manual padding

# If wavefront occupancy is low:
#   -> Check VGPR usage: reduce accumulator tile size
#   -> Check LDS allocation: reduce tile size or share across wavefronts
#   -> Consider num_stages=1 (AMD lacks TMA, multi-stage less beneficial)
```

## Performance Impact
- **Omniperf roofline analysis** immediately reveals whether attention is compute-bound or memory-bound, directing optimization effort correctly
- **LDS bank conflict identification** can reveal 2-16x bandwidth waste (similar to NVIDIA SMEM bank conflicts causing 93.75% efficiency loss)
- **MFMA utilization tracking** shows whether tensor core instructions are being starved by softmax computation or data loading
- **XCD-level load balancing** prevents 2-8x performance degradation from uneven dispatch across chiplets
- **HBM bandwidth measurement** validates whether memory-bound kernels achieve near the 5.3 TB/s MI300X peak
- **Wavefront occupancy analysis** identifies register pressure from FP32 accumulators as the typical limiter for attention kernels

## When to Use
- Debugging a Flash Attention kernel that underperforms on AMD MI300X or MI250X
- When you suspect MFMA utilization is low but are unsure whether the cause is data starvation, softmax overhead, or scheduling
- After implementing a new attention variant on AMD and needing to find the primary bottleneck
- When porting an NVIDIA-optimized attention kernel to AMD and need to understand different performance characteristics
- Comparing attention kernel performance across Composable Kernel, TileLang, Triton, and hipBLASLt implementations
- Before and after kernel optimization to quantify improvement

## When NOT to Use
- Quick iteration during initial kernel development (use HIP event timers or `hipExtModuleLaunchKernel` timing first)
- When the kernel is already achieving >70% of theoretical peak for its bottleneck resource
- When profiling on a different AMD GPU architecture than the deployment target (counter semantics and available PMCs differ between CDNA2 and CDNA3)
- For end-to-end model profiling without kernel-level detail (use Omnitrace alone)
- On NVIDIA GPUs (use Nsight Compute/Nsight Systems instead)

## Source Code Examples

### rocprof Counter Specification (bash)

Create a counter specification file with multiple passes (4-8 PMCs per pass due to hardware counter multiplexing):

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

### Targeted Counter Collection (bash)

```bash
# Quick check: is attention compute-bound or memory-bound?
rocprof --pmc SQ_INSTS_VALU_MFMA,TCC_EA_RDREQ_32B_sum,GRBM_GUI_ACTIVE \
    --kernel-name "attention" -- python script.py

# Compute utilization breakdown
rocprof --pmc SQ_INSTS_VALU_MFMA,SQ_INSTS_VALU_TRANS,SQ_INSTS_VALU,SQ_INSTS_SALU \
    --kernel-name "attention" -- python script.py

# LDS efficiency
rocprof --pmc SQ_INSTS_LDS,SQ_LDS_BANK_CONFLICT,SQ_WAIT_INST_LDS \
    --kernel-name "attention" -- python script.py
```

### Omniperf Profile and Analyze Commands (bash)

```bash
# Profile: collects counters across multiple auto-generated passes
omniperf profile -n flash_attn_profile -- python model.py

# Analyze: compute derived metrics and display results
omniperf analyze -p workloads/flash_attn_profile/MI300X/

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

### Python Script for Computing Derived Metrics from rocprof Output

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

## Key Takeaways
- Omniperf is the AMD equivalent of Nsight Compute -- it provides roofline analysis, derived metrics, and guided bottleneck identification; always start here for structured analysis
- Omnitrace is the AMD equivalent of Nsight Systems -- use it first for timeline analysis to identify which kernel to profile deeply
- rocprof collects raw PMC counters; Omniperf builds on rocprof to provide actionable metrics like MFMA utilization percentage and LDS efficiency
- AMD's LDS has the same 32-bank structure as NVIDIA's SMEM -- bank conflict patterns and fixes (swizzling, layout transpose) transfer directly
- MI300X's 8-XCD chiplet topology introduces a profiling dimension absent on NVIDIA: per-XCD load balancing and cross-XCD L2 cache behavior
- MFMA instructions operate per-wavefront (64 threads) unlike NVIDIA's per-warpgroup WGMMA (128 threads) -- occupancy analysis must account for different thread granularity
- `num_stages=1` is typical on AMD (no TMA hardware), so pipeline stall patterns differ from NVIDIA's multi-stage TMA pipelines
- Counter multiplexing in Omniperf requires multiple kernel replays -- ensure deterministic kernel behavior for accurate profiling

## References
- [ROCm Profiler Documentation](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [Omniperf Documentation (AMD Instinct MI Profiler)](https://rocm.docs.amd.com/projects/omniperf/en/latest/)
- [Omnitrace Documentation](https://rocm.docs.amd.com/projects/omnitrace/en/latest/)
- [AMD MI300/MI200 Performance Counters](https://instinct.docs.amd.com/latest/gpu-arch/mi300-mi200-performance-counters.html)
- [ROCm Profiling Guide for AI/ML Workloads](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/profiling.html)
- [Nsight Compute Profiling for Flash Attention (NVIDIA equivalent)](../nsight_profiling_flash_attention/skill.md)
