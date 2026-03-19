---
skill_name: Nsight Compute Profiling for Flash Attention Occupancy and Scheduling
description: Profiler-driven methodology for identifying and fixing occupancy, bank conflict, and pipeline bottlenecks in Flash Attention kernels
level: L2 - Scheduling & Pipelining Level
target_hardware: NVIDIA GPUs (demonstrated on RTX 2070/Turing SM 7.5, applicable to Ampere/Hopper/Blackwell)
relevance: When debugging Flash Attention kernel performance, diagnosing why occupancy is low, finding shared memory bank conflicts, or understanding pipeline stalls
---

# Nsight Compute Profiling for Flash Attention Occupancy and Scheduling

## What It Is
A profiler-driven optimization methodology for Flash Attention kernels using NVIDIA Nsight Compute (ncu). This approach iteratively profiles, identifies the top bottleneck, fixes it, and re-profiles. Applied to a Triton-based Flash Attention implementation, it demonstrated how to diagnose: (1) excessive HBM traffic from wrong loop ordering, (2) shared memory bank conflicts causing 93.75% efficiency loss, (3) MIO pipeline stalls from softmax special math operations, and (4) occupancy limitations from shared memory pressure. The methodology achieved 4.9x total speedup through systematic bottleneck elimination.

## Key Concepts
- **Occupancy analysis**: Theoretical occupancy limited by shared memory per block (e.g., 28 KB -> max 2 blocks/SM -> 25% occupancy). Use ncu's occupancy calculator to identify the limiter (registers, shared memory, or thread count).
- **Shared memory bank conflicts**: K matrix stored row-major causes 16-way bank conflicts when accessed column-wise for QK^T. Pre-transposing K to column-major eliminates the conflict pattern.
- **MIO pipeline throttling**: MIO handles both shared memory loads AND special math (exp, max, log). Heavy softmax computation competes with memory access on the same pipeline, causing 50.7% of warp stall time.
- **Loop order and HBM traffic**: Outer-KV/inner-Q loop forces Q and O to reload from HBM every iteration (11.6 GB). Outer-Q/inner-KV keeps Q in registers and streams K/V once (412 MB -- a 92% reduction).
- **Deferred normalization**: Moving the division operation out of the inner loop to the kernel epilogue reduces MUFU pipeline pressure.
- **Occupancy is non-monotonic**: 25% occupancy (v1) was insufficient for latency hiding; 63% (v2) was sufficient. Beyond the threshold, more occupancy can hurt by reducing per-thread resources.

## Scheduling Strategy / Pseudo-code
```
# PROFILING WORKFLOW:
# Step 1: Profile with full metrics
ncu --set full --kernel-name "attn_kernel" -o profile.ncu-rep python script.py

# Step 2: Check occupancy section
#   -> What limits occupancy? (shared memory, registers, or max threads)
#   -> Theoretical vs achieved occupancy
#   -> Active warps per scheduler

# Step 3: Check memory throughput section
#   -> Total GMEM reads/writes (is it >> expected?)
#   -> SMEM throughput and bank conflicts
#   -> L2 hit rate

# Step 4: Check warp stall analysis
#   -> Top stall reasons (MIO throttle? Memory dependency? Barrier?)
#   -> Cycles between instruction issue

# Step 5: Check instruction mix
#   -> FFMA vs HMMA (tensor core) instructions
#   -> MUFU (special function) count
#   -> Compute vs memory instruction ratio

# KEY OPTIMIZATION SEQUENCE FOR FLASH ATTENTION:

# Fix 1: Loop order (eliminate HBM thrashing)
# Before: grid=(B, N_h), outer loop over K/V, reload Q every iteration
# After:  grid=(S/Bc, B*N_h), Q stays in registers, stream K/V
# Effect: -92% HBM reads

# Fix 2: Memory layout (eliminate bank conflicts)
# Before: K in row-major, column access -> 16-way bank conflicts
# After:  K pre-transposed to column-major -> sequential bank access
# Effect: 6.3x -> 3.4x average bank conflict, +145% speedup

# Fix 3: Deferred normalization (reduce MIO pressure)
# Before: division in inner loop (MUFU reciprocal + refinement)
# After:  accumulate unnormalized, single divide at end
# Effect: Reduced MIO stalls

# Fix 4: Tile size tuning (balance occupancy vs work per tile)
# Shared memory per block = (2*Bc + 3*Bc*D + Bc^2) * sizeof(float)
# Increase Bc -> fewer iterations, fewer exp/max ops, but lower occupancy
# Decrease Bc -> more iterations, more special math, but higher occupancy
```

## Performance Impact
- **Loop restructuring**: 92% reduction in HBM traffic (11.6 GB -> 412 MB)
- **Bank conflict fix (K transpose)**: 4.9x total speedup (166 ms -> 34 ms)
- **Bank conflict reduction**: 6.3-way -> 3.4-way average
- **Eligible warps per cycle**: +153% increase after bank conflict fix
- **Occupancy improvement**: 25% -> 63% (v1 -> v2) via reduced SMEM footprint
- **Overall**: profiler-guided optimization delivered 4.9x speedup on RTX 2070

## When to Use
- Debugging a Flash Attention kernel that underperforms expected throughput
- When you suspect occupancy is limiting performance but are unsure of the cause
- After implementing a new attention variant and needing to find bottlenecks
- When shared memory bank conflicts are suspected (signs: low SMEM throughput relative to requests)
- When porting attention kernels to new GPU architectures and need to understand resource limits
- Before and after any kernel optimization to verify it actually helped

## When NOT to Use
- Quick iteration during initial kernel development (ncu full profile is slow -- use nsys or torch.profiler first)
- When the kernel is already at >60% of theoretical peak (diminishing returns)
- When profiling on a different GPU architecture than the target deployment (bank conflict patterns and occupancy limits differ)
- For end-to-end model profiling (use Nsight Systems instead for timeline analysis)

## Key Takeaways
- Always profile before optimizing -- each ncu report reveals the SPECIFIC next bottleneck
- Shared memory bank conflicts can waste 93.75% of memory bandwidth; memory layout (row vs column major) is critical
- The MIO pipeline is shared between shared memory access and special math (exp, max); heavy softmax use creates contention
- Occupancy has a threshold effect: below the threshold, adding more warps helps hide latency; above it, more warps just reduce per-thread resources
- Loop ordering determines whether Q is reloaded from HBM every iteration (catastrophic) or stays in registers (optimal)
- Deferred normalization moves expensive division out of the inner loop
- `ncu --set full` collects everything but is slow; for quick checks, use `ncu --set default` or specific metric groups

## References
- [Reimplementing FlashAttention for Performance and Giggles (AmineDiro)](https://aminediro.com/posts/flash_attn/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
