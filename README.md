# Flash Attention Optimization Knowledge Base

## 1. Purpose and Motivation

### Why This Knowledge Base Exists

Flash Attention is the dominant attention algorithm in modern LLM training and inference. However, optimizing it for a specific GPU architecture remains a labor-intensive, expert-driven process. Each generation of GPU hardware (NVIDIA Ampere → Hopper → Blackwell, AMD CDNA2 → CDNA3 → CDNA4) introduces new matrix instructions, memory hierarchies, and pipeline capabilities that demand kernel rewrites. The optimization space is vast, spanning multiple distinct classes of decisions, from high-level framework selection down to individual instruction scheduling.

This knowledge base was built to give an **AI agent** (or a human kernel engineer) the structured reference material needed to navigate that optimization space systematically. Instead of searching through hundreds of papers, blog posts, and codebases, the agent can consult a curated set of **67 skill cards** — each one a self-contained decision guide for a specific optimization technique, tied to a specific level of the hierarchy.

### Why a Multi-Level Hierarchy?

Flash Attention optimization is not a single problem. It decomposes into a stack of interdependent decisions:

```
L0  Which kernel implementation should I use?          (framework / library choice)
L1  What algorithm should the kernel implement?        (FA1, FA2, FA3, FA4, MLA, Ring)
L2  How should work be scheduled and pipelined?        (partitioning, occupancy, producer-consumer)
L3  Where should data live and how should threads cooperate? (memory hierarchy, warp roles, synchronization)
L4  How to optimize the inner compute loop?            (tensor core config, softmax, instruction scheduling)
L5  What numerical precision to use?                   (FP16, BF16, FP8, INT4, mixed precision)
```

An AI agent descending this hierarchy can make decisions top-down (starting from "which library?" and drilling into "which PTX instruction?"), or it can enter at any level when a specific bottleneck is identified via profiling. Each level's skill cards provide the **when-to-use**, **when-not-to-use**, and **key trade-offs** that make the decision actionable.

### Why These Specific Resources?

Every entry in this knowledge base was selected against three criteria:

1. **Direct relevance**: Does it help make a concrete Flash Attention optimization decision? Generic GPU programming guides that don't connect to attention are excluded.
2. **Actionability**: Does it provide enough detail (pseudo-code, performance numbers, code examples) for an agent to act on it, not just understand it abstractly?
3. **Uniqueness**: Does it teach something no other entry covers? Cross-level duplicates are eliminated — each source document appears at exactly one level (its best fit).

The result is a lean, high-signal collection where every entry earns its place.

---

## 2. What the Knowledge Base Contains

### At a Glance

| Metric | Value |
|--------|-------|
| Total topics | 73 |
| Skill cards (`skill.md`) | 73 |
| Technical documents (`document.md`) | 73 |
| Original research papers (PDF) | 21 |
| Source code files | 255 |
| AMD GEAK knowledge files | 14 |

### Hardware Coverage

| Vendor | Architectures | Key Differences for Flash Attention |
|--------|--------------|--------------------------------------|
| **NVIDIA Ampere** (A100) | SM80, 80GB HBM2e, 2 TB/s | FA2 baseline; per-warp MMA; no TMA |
| **NVIDIA Hopper** (H100) | SM90, 80GB HBM3, 3.35 TB/s | WGMMA (warpgroup MMA), TMA hardware, warp specialization → FA3 |
| **NVIDIA Blackwell** (B200) | SM100, 192GB HBM3e, 8 TB/s | UMMA, TMEM, asymmetric SFU scaling → FA4 |
| **AMD CDNA2** (MI250X) | gfx90a, 128GB HBM2e, 3.2 TB/s | MFMA (wave-64), LDS 64KB/CU, no TMA equivalent |
| **AMD CDNA3** (MI300X) | gfx942, 192GB HBM3, 5.3 TB/s | 304 CUs (8 XCDs), FP8 MFMA, chiplet-aware scheduling |
| **AMD CDNA4** (MI355X) | gfx950, 288GB HBM3E, 8 TB/s | MXFP8/FP6/FP4 block-scaled MFMA, 160KB LDS, 2× transcendental rate, 10 PF FP4 peak |

### Attention Variants Covered

| Variant | Where in KB | Key Optimization Challenge |
|---------|-------------|---------------------------|
| Standard MHA | L1/original_flash_attention through L1/flash_attention_4 | Tiling, online softmax, pipeline design |
| Grouped-Query Attention (GQA) | L1/flashinfer_attention_engine | K/V head sharing changes memory access patterns |
| Multi-Head Latent Attention (MLA) | L1/flash_mla_deepseek | Fusing KV decompression into attention; 93.3% KV cache reduction |
| Ring Attention | L1/ring_attention | Distributing FA across GPUs with overlapped communication |
| Sparse Attention | L4/amd_fmha_kernel_internals | CSR-like patterns reducing O(N^2) to O(N*k) |
| Paged Attention | L4/amd_fmha_kernel_internals, L1/flashinfer | Block table indirection for serving workloads |

### AI-Driven Optimization Tools

| Tool | Approach | Relevance |
|------|----------|-----------|
| QiMeng-Attention | LLM + "Thinking Language" | Generates FA kernels surpassing cuDNN on diverse GPUs (up to 35x speedup) |
| GEAK | Multi-agent + profiling + knowledge retrieval | Autonomous Triton/HIP kernel optimization on AMD GPUs |
| AutoTriton | LLM + RL (GRPO) | Automated Triton code generation; 8B model matches frontier LLMs |
| ThunderKittens | Tile-based CUDA abstractions | From the FA research group; #1 on H100 attention benchmarks |
| FlagAttention | Triton attention operators | Portable flash/paged/split-KV attention across GPU architectures |

---

## 3. How the Knowledge Base Is Organized

### Level-by-Level Guide

Each level addresses a distinct class of optimization decisions. The levels are ordered from coarse-grained (L0) to fine-grained (L5), mirroring the natural top-down progression of kernel development.

#### L0 — Implementation Selection (11 topics)
**Question**: *Which attention implementation should I call, and how?*

This is the entry point. Before writing any kernel code, you must decide which existing implementation to use — or whether to generate a new one. This level covers PyTorch SDPA dispatch, FlexAttention for custom patterns, xFormers for multi-backend support, AMD's Composable Kernel, serving frameworks (vLLM, TensorRT-LLM), NVIDIA's cuDNN backend, and emerging AI-driven kernel generation tools.

| Topic | Hardware | Key Content |
|-------|----------|-------------|
| `pytorch_sdpa` | All | SDPA backend dispatch, `sdpa_kernel()` context manager, backend forcing |
| `flexattention_fa4` | NVIDIA Hopper/Blackwell | FlexAttention API, `score_mod`/`mask_mod`, CuTeDSL, 1.2-3.2x over Triton |
| `xformers_attention` | NVIDIA + AMD | Three backends (CK, Flash, Triton), composable biases, variable-length batching |
| `amd_composable_kernel` | AMD MI250X/MI300X | CK-Tile Flash Attention, MFMA instructions, wavefront-64 differences |
| `flash_attention_free_lunch` | All | Integration guide, version comparison (FA1→FA4), migration decision framework |
| `flash_attention_3_hopper` | NVIDIA Hopper | FA3 deployment guide, pingpong scheduling, FP8, 75% H100 utilization |
| `ai_driven_kernel_optimization` | All | QiMeng-Attention, GEAK, AutoTriton, FlagAttention — AI-driven FA kernel generation |
| `thunderkittens` | NVIDIA Hopper | Hazy Research tile primitives, #1 H100 attention benchmark, LSCF kernel design |
| `cudnn_fused_attention` | NVIDIA Ampere/Hopper/Blackwell | cuDNN fused attention graph API, PyTorch cuDNN backend dispatch, FP8 support |
| `vllm_attention_backends` | NVIDIA + AMD | vLLM backend selection (FA2/FlashInfer/xFormers/Triton), PagedAttention, chunked prefill |
| `tensorrt_llm_attention` | NVIDIA | TensorRT-LLM FMHA, cuDNN FP8, XQA decode kernel for GQA, paged KV cache |

**Why it matters for AI**: An AI agent must first decide whether to use an off-the-shelf kernel, extend one via FlexAttention, use a serving framework's built-in attention, or generate a new kernel with AI-driven tools. This level provides the decision framework.

#### L1 — Algorithm Design (13 topics)
**Question**: *What mathematical decomposition should the kernel use?*

Flash Attention's core innovation is fusing tiled GEMM with online softmax. But the specific algorithm varies significantly across versions (FA1→FA4) and variants (MLA, Ring, Sparse, Paged, Linear). This level covers the mathematical foundations, the evolution of algorithmic ideas, novel attention architectures, and the primary algorithmic alternative (linear attention / SSMs).

| Topic | Key Content |
|-------|-------------|
| `original_flash_attention` | IO-aware tiling, O(N²)→O(N) memory, recomputation in backward pass |
| `online_softmax_to_flash_attention` | Full mathematical derivation: 3-pass → 2-pass → 1-pass fused attention |
| `flash_attention_2` | Non-matmul FLOP reduction, sliced-Q warp partitioning, 230 TFLOPs/s A100 |
| `flash_attention_3` | Warp-specialized pipeline, 2-stage GEMM-softmax, FP8, 740 TFLOPs/s H100 |
| `flash_attention_4` | 5 warp roles, cubic exp approximation, conditional rescaling, 1613 TFLOPs/s B200 |
| `flash_mla_deepseek` | Multi-Head Latent Attention, KV cache compression (93.3%), 660 TFLOPS decode |
| `ring_attention` | Distributed FA across GPUs, near-infinite context, zero communication overhead |
| `flashinfer_attention_engine` | Serving-aware FA: prefill/decode/append kernels, paged KV-cache, GQA optimization |
| `triton_flash_attention_implementation` | Practical Triton FA code, exp2 optimization, float32 accumulation |
| `native_sparse_attention` | DeepSeek NSA: local + compressed + selected three-branch sparse attention, 8x speedup |
| `differential_attention` | Attention as difference of two softmax maps, noise cancellation, dual QK^T kernels |
| `paged_attention` | KV cache as virtual memory pages, page table indirection, vLLM's core algorithm |
| `linear_attention_mamba2` | SSM-attention duality via semiseparable matrices, Mamba-2 block decomposition |

**Why it matters for AI**: When an agent needs to implement a new attention variant, understand why a specific FA version is faster, or decide between softmax attention and linear alternatives, this level provides the algorithmic basis for reasoning about correctness and performance.

#### L2 — Scheduling & Pipelining (14 topics)
**Question**: *How should work be distributed and pipelined across the GPU?*

The same algorithm can perform very differently depending on how work is partitioned across thread blocks and how data movement overlaps with computation. This level covers scheduling strategies, profiling methodology, and the producer-consumer pipeline architectures that FA3 and FA4 depend on. Work partitioning and pipelining are tightly coupled — changing one almost always requires changing the other. The level also includes the specific pipeline designs of FA3 (producer-consumer with pingpong) and FA4 (5 specialized warp roles), since these are fundamentally pipeline architecture choices that the AI evaluates before diving into memory or inner-loop details.

| Topic | Key Content |
|-------|-------------|
| `fa2_scheduling` | Sliced-Q vs sliced-K warp partitioning, sequence-length parallelism |
| `flexattention_scheduling` | Block-sparse iteration, skipping masked regions, score_mod scheduling |
| `nsight_profiling_flash_attention` | Profiler-driven optimization: HBM thrashing, bank conflicts, MIO throttling, 4.9x speedup |
| `warp_specialization` | Producer-consumer pattern, 3 enabling conditions, TMA/WGMMA overlap |
| `cutlass_ping_pong_gemm` | 3-warpgroup ping-pong architecture (1 producer + 2 consumers) |
| `flashattention3_pipelining` | FA3's 2-stage GEMM-softmax pipeline, 250x throughput gap exploitation |
| `learn_cutlass_hard_way` | GEMM optimization progression: naive → 89.5x speedup, tiling/pipelining/autotuning |
| `hopper_tma_tutorial` | TMA mechanics: descriptors, mbarrier, phase-based sync, pipeline integration |
| `fa3_warp_specialization` | FA3 producer/consumer split, pingpong scheduling, 570→661 TFLOPs/s ablation |
| `fa4_warp_roles` | FA4's 5 specialized warps (Load, MMA, Softmax, Correction, Epilogue) |
| `cutlass_pipelining_warp_spec` | CUTLASS pipeline abstraction: dual barriers, circular buffers, setmaxnreg |
| `rocm_profiling_attention` | AMD ROCm profiling tools (rocprof, Omniperf, Omnitrace) for attention kernels |
| `kperfir_compiler_profiling` | KPerfIR (OSDI'25): compiler-centric Triton IR profiling, FA3 24.1% speedup, cross-platform |
| `split_kv_decode_scheduling` | FlashDecoding split-KV for decode phase, parallel reduction, adaptive num_splits |

**Why it matters for AI**: Scheduling decisions determine GPU occupancy and load balance. Without pipelining, even a mathematically optimal algorithm will leave the GPU idle 50%+ of the time. FA3's warp specialization enabled a 2x improvement over FA2, and FA4's 5-role pipeline is the key architectural innovation for Blackwell. The AI designs the pipeline architecture at this level before implementing memory and inner-loop details.

#### L3 — Memory & Thread Cooperation (13 topics)
**Question**: *Where should each tensor fragment reside, and how should threads cooperate at the hardware level?*

Flash Attention is fundamentally a memory-optimization technique — it exists because naive attention is memory-bound. This level focuses on data placement across the GPU memory hierarchy (HBM → L2 → shared memory → registers), tiling strategies, hardware-specific memory features (TMA deep mechanics, swizzling, bank conflicts), and vendor-specific thread cooperation primitives (AMD wave-64 model). Once the pipeline architecture is designed (L2), this level answers how to implement the data movement that feeds it.

| Topic | Hardware | Key Content |
|-------|----------|-------------|
| `gpu_memory_hierarchy` | All | Register→L1→L2→HBM latencies and bandwidths |
| `flash_attention_tiling_recomputation` | All | FA's tiling strategy, O(N²)→O(N) memory, backward recomputation |
| `why_flash_attention_memory_bound` | All | Roofline analysis, 97% traffic is N×N intermediates |
| `tma_deep_dive` | NVIDIA Hopper | Conveyor-belt TMA model, multi-dim tiling, triple buffering |
| `cutlass_cute_layout` | NVIDIA | CuTe layout algebra, Layout<Shape,Stride>, MMA/Copy atoms |
| `flash_attention_tile_tuning` | NVIDIA | "Large tile trap" (18-43% regression), fast-math rescue, 918 TFLOPS B200 |
| `shared_memory_swizzling_bank_conflicts` | All | XOR swizzling, CuTe Swizzle<B,M,S>, bank conflict elimination |
| `amd_mi300x_flash_attention` | AMD MI300X | CDNA3 chiplet architecture, 5.3 TB/s HBM3, MFMA, num_stages=1 tuning |
| `amd_cdna_architecture_guide` | AMD MI300X/MI355X | CDNA3/CDNA4 chiplet architecture, CU microarchitecture, cache hierarchy, HBM specs |
| `mi300_compute_memory_partitioning` | AMD MI300X/MI355X | SPX/CPX/DPX/QPX compute modes, NPS1/NPS4 memory modes, 5-15% bandwidth improvement |
| `amd_wavefront_cooperation` | AMD MI250X/MI300X | Wave-64 model, MFMA scheduling, buffer-to-LDS transfers, butterfly reductions |
| `blackwell_tmem` | NVIDIA Blackwell | Tensor Memory (TMEM): 128x256x32-bit buffer, TMEM allocation, FA4 usage |
| `thread_block_clusters` | NVIDIA Hopper/Blackwell | Cluster launch, distributed shared memory (DSMEM), TMA multicast for KV |

**Why it matters for AI**: The single biggest performance variable in attention is how data moves through the memory hierarchy. Getting tiling, swizzling, or TMA usage wrong can cost 2-10x performance.

#### L4 — Compute Kernel Optimization (14 topics)
**Question**: *How to optimize the innermost compute loop — tensor core config, softmax, and instruction scheduling?*

The innermost loop of Flash Attention executes matrix multiply-accumulate instructions (WGMMA on NVIDIA, MFMA on AMD) interleaved with softmax reductions. This level combines three aspects of inner-loop optimization that are tuned together in practice: (1) MMA tile shapes and instruction selection, (2) softmax implementation and approximation, and (3) instruction-level scheduling and PTX tuning. A tile shape change affects softmax register pressure which affects instruction overlap — they cannot be optimized independently.

| Topic | Hardware | Key Content |
|-------|----------|-------------|
| `wgmma_hopper_tutorial` | NVIDIA Hopper | WGMMA m64nNk16, SS vs RS variants, descriptor-based SMEM, swizzle modes |
| `flashattention2_hopper_cutlass` | NVIDIA Hopper | FA2 microkernel: tile shape crisis (128×128 collapses at d=256), SS/RS selection |
| `amd_fmha_kernel_internals` | AMD MI300X/MI350X | FMHA V3 fwd/bwd/splitkv/FP8/prefill kernels, wave-group scheduling, MFMA pipeline |
| `amd_mfma_matrix_core_programming` | AMD MI300X/MI355X | MFMA intrinsics, tile shapes, data layouts, FP32/FP16/FP8/FP4 examples, block scaling |
| `amd_gfx9_kernel_optimization` | AMD MI300X/MI355X | GFX9 register usage, LDS bank conflicts, global memory patterns, cross-lane primitives |
| `online_softmax_algorithm` | All | Running max/sum recurrence for tiled softmax, practical kernel implementation |
| `fa4_official_paper` | NVIDIA Blackwell | FA4 polynomial exp (degree-3, 8.77e-5 error), conditional rescaling (τ=8.0), 1613 TFLOPs/s |
| `fast_math_softmax` | NVIDIA | flush_to_zero + approx rounding: 34-72% speedup, SASS instruction comparison |
| `fast_softmax_cuda_kernel` | NVIDIA | Warp-level shuffles, vectorized loads, block reductions, 50% speedup on A100 |
| `cutlass_instruction_mapping` | NVIDIA | CuTe layout → hardware instruction sequences, MMA count formulas |
| `instruction_overlap_warp_spec` | NVIDIA Hopper | TMA + WGMMA concurrent execution, quasi-out-of-order on in-order GPU |
| `handwritten_ptx_optimization` | NVIDIA | Branchless `setp`+`selp`, fast exp2/reciprocal via inline PTX, 7-14% gains |
| `sfu_bottleneck_asymmetric_scaling` | NVIDIA Blackwell | 512:1 tensor core vs SFU imbalance, FA4's pipeline co-design for Blackwell |
| `umma_blackwell` | NVIDIA Blackwell | UMMA (Unified MMA) on SM100, TMEM operands, CTA-group operations, FA4 usage |

**Why it matters for AI**: A bad tile shape choice can cause register spills that destroy performance (128×128 drops from 308 to 36.7 TFLOPs). Softmax is where FA3→FA4's biggest algorithmic innovations happen (polynomial exp, conditional rescaling). The SFU bottleneck analysis shows that on Blackwell, the exponential function unit is 512x slower than tensor cores — understanding this asymmetry drove FA4's entire design.

#### L5 — Numerical Precision (8 topics)
**Question**: *What precision should the kernel use, and how to maintain accuracy?*

FP8 attention can deliver 2x the throughput of FP16, but naive quantization destroys accuracy. This level covers precision formats from FP16 down to FP4, quantization strategies (block quantization, Hadamard incoherent processing, microscaling), and mixed-precision techniques that make low-precision attention viable. Blackwell's FP4 tensor cores and OCP microscaling formats represent the next frontier.

| Topic | Key Content |
|-------|-------------|
| `fa3_fp8_fp16` | FA3 FP8/FP16: block quantization, Hadamard incoherent processing, 1.2 PFLOPS |
| `nvidia_fp8_formats` | E4M3 vs E5M2, per-tensor/delayed/MXFP8 scaling, Transformer Engine integration |
| `quip_incoherent_processing` | Randomized Hadamard transform, 2.6x quantization error reduction, E8 lattice codebooks |
| `pytorch_sdpa_precision` | Backend precision differences, float32 upcast in math backend, reproducibility |
| `sageattention2_mixed_precision` | INT4 Q/K + FP8 P/V, outlier smoothing, 3-5x over FA2 with better accuracy |
| `mxfp_microscaling_formats` | OCP MX standard (shared exponent per block of 32), MXFP4/MXFP8, Blackwell native support |
| `amd_cdna_low_precision_types` | AMD FP4/FP6/FP8 formats, OCP MXFP block scaling, E8M0 scale factors, CDNA3/CDNA4 support |
| `blackwell_fp4_attention` | FP4 (E2M1) tensor cores on B200, 2x FP8 throughput, sub-byte quantization for attention |

**Why it matters for AI**: Precision is the highest-leverage performance knob — going from FP16 to FP8 can double throughput. But getting the quantization strategy wrong can silently corrupt model outputs. This level provides the trade-off analysis.

---

## 4. How to Use This Knowledge Base

### File Structure Per Topic

Each of the 67 topic directories follows a consistent structure:

```
topic_name/
├── skill.md          # Decision card — START HERE
├── document.md       # Detailed technical content
├── *.pdf             # Original research paper (when available)
├── Report_*.md       # AMD GEAK kernel analysis reports (when available)
├── hip-*.md          # AMD HIP optimization guides (when available)
└── src/              # Source code from GitHub repos (when available)
```

### For an AI Agent: Reading Protocol

1. **Identify the optimization level** — Is the bottleneck at the algorithm level? Memory/threading? Compute inner loop? Use the L0-L5 hierarchy to narrow down.

2. **Read `skill.md` first** — Each skill card has YAML frontmatter (`skill_name`, `target_hardware`, `relevance`) followed by structured sections: What It Is, Key Concepts, When to Use, When NOT to Use, Code/Pseudo-code, Key Takeaways. This gives enough context to decide whether to read further.

3. **Read `document.md` for depth** — When the skill card indicates relevance, the document provides full technical details: performance numbers, algorithmic derivations, architecture diagrams, and implementation guidance.

4. **Consult `src/` for implementation** — When writing or modifying kernel code, the source files from official repositories (flash-attention, CUTLASS, FlashInfer, FlashMLA, ThunderKittens, FlagAttention, GEAK) provide ground truth.

5. **Read `*.pdf` for authority** — When a claim needs verification or the full context of a research contribution is needed, the original paper is the authoritative source.

### Navigation Shortcuts

**By optimization task:**

| If you need to... | Start at |
|---|---|
| Choose which FA version/library to use | L0 |
| Understand or modify FA's algorithm | L1 |
| Fix low GPU occupancy, load imbalance, or pipeline stalls | L2 |
| Fix memory bottlenecks, tune tiling, or configure warp roles | L3 |
| Tune MMA/MFMA tile shapes, optimize softmax, or schedule instructions | L4 |
| Choose precision (FP16/BF16/FP8/INT4) or debug numerical issues | L5 |

**By target GPU:**

| Target GPU | Key entries |
|---|---|
| **AMD MI250X / MI300X / MI355X** | L0/amd_composable_kernel, L3/amd_cdna_architecture_guide, L3/amd_mi300x_flash_attention, L3/mi300_compute_memory_partitioning, L3/amd_wavefront_cooperation, L4/amd_fmha_kernel_internals, L4/amd_mfma_matrix_core_programming, L4/amd_gfx9_kernel_optimization, L5/amd_cdna_low_precision_types |
| **NVIDIA Ampere (A100)** | L1/flash_attention_2, L2/fa2_scheduling, L4/flashattention2_hopper_cutlass |
| **NVIDIA Hopper (H100)** | L1/flash_attention_3, L2/flashattention3_pipelining, L2/fa3_warp_specialization, L4/wgmma_hopper_tutorial |
| **NVIDIA Blackwell (B200)** | L1/flash_attention_4, L2/fa4_warp_roles, L4/fa4_official_paper, L4/sfu_bottleneck_asymmetric_scaling |
| **AI-driven (any GPU)** | L0/ai_driven_kernel_optimization, L0/thunderkittens |

**By FA version:**

| Version | Primary entry | Key innovation |
|---------|--------------|----------------|
| FA1 | L1/original_flash_attention | IO-aware tiling + online softmax |
| FA2 | L1/flash_attention_2 | Sliced-Q, reduced non-matmul FLOPs |
| FA3 | L1/flash_attention_3 | Warp specialization, 2-stage pipeline, FP8 |
| FA4 | L1/flash_attention_4 | 5 warp roles, polynomial exp, conditional rescaling |
| FlashMLA | L1/flash_mla_deepseek | Low-rank KV compression, fused decompression |
| FlashInfer | L1/flashinfer_attention_engine | Serving-aware prefill/decode/append |

### For a Human Reader

- **New to Flash Attention?** Start with L1/original_flash_attention and L1/online_softmax_to_flash_attention for the mathematical foundations, then L0/flash_attention_free_lunch for the practical integration guide.
- **Kernel engineer?** Go directly to the level matching your current bottleneck (L2-L4 are the core kernel engineering levels).
- **Porting NVIDIA→AMD?** Read L0/amd_composable_kernel for framework differences, L3/amd_cdna_architecture_guide for hardware architecture, L3/amd_mi300x_flash_attention for FA-specific tuning, L3/mi300_compute_memory_partitioning for deployment modes, L3/amd_wavefront_cooperation for wave-64 vs warp-32, L4/amd_mfma_matrix_core_programming for MFMA instructions (vs WGMMA), L4/amd_gfx9_kernel_optimization for low-level optimization, L4/amd_fmha_kernel_internals for kernel implementation details, and L5/amd_cdna_low_precision_types for AMD precision formats.
- **Exploring AI-driven optimization?** Start with L0/ai_driven_kernel_optimization for the landscape, then L0/thunderkittens for tile abstractions.

---

## 5. How the Knowledge Base Was Constructed

### Source Material

The knowledge base draws from three categories of sources:

**21 Research Papers (PDF):**
FlashAttention 1/2/3/4, Online Softmax derivation, FA2-on-Hopper/CUTLASS, FlashInfer (MLSys 2025 Best Paper), SageAttention2 (ICML 2025), QuIP# (ICML 2024), FP8 Formats for Deep Learning, Ring Attention, DeepSeek-V2, PTX ISA Reference, AI-driven Kernel Generation Survey (arXiv 2601.15727), GEAK Agent (arXiv 2507.23194)

**14 AMD GEAK Knowledge Files:**
FMHA V3 Forward/Backward/Split-KV/FP8/Batch-Prefill kernel reports from AMD's Composable Kernel team, HIP Advanced Optimization and Native Examples guides, CDNA Architecture and Performance Counter references, Unified MMA Multi-Architecture and Async Buffer Addressing reports

**255 Source Code Files from 9 GitHub Repositories:**
`Dao-AILab/flash-attention` (FA2/FA3/FA4 CUDA kernels), `NVIDIA/cutlass` (CuTe, WGMMA, ping-pong GEMM), `facebookresearch/xformers` (FMHA backends), `flashinfer-ai/flashinfer` (serving attention kernels), `deepseek-ai/FlashMLA` (MLA CUDA kernels), `triton-lang/triton` (FA tutorial), `HazyResearch/ThunderKittens` (tile abstractions + H100 attention), `flagos-ai/FlagAttention` (Triton attention operators), `AMD-AGI/GEAK` (AMD kernel examples + knowledge base)

### Construction Process

1. **Seed collection** — Started from a curated instruction document listing key resources at each optimization level, with 18 specific URLs to papers and blog posts.

2. **Systematic web search** — For each optimization level, searched for additional high-quality resources beyond the seed set. Evaluated ~100+ candidate documents.

3. **Original file acquisition** — Downloaded original PDFs from arXiv and conference proceedings. Cloned GitHub repositories and extracted relevant kernel source files. Downloaded AMD GEAK knowledge base files for Flash Attention kernel analysis.

4. **Skill card generation** — For each retained document, generated a structured `skill.md` with YAML frontmatter and standardized sections (What It Is, Key Concepts, When to Use, When NOT to Use, Code/Pseudo-code, Key Takeaways, References).

5. **Gap analysis** — Identified missing coverage for AMD GPUs, attention variants (MLA, Ring Attention), and AI-driven kernel generation tools. Searched specifically for these topics and added entries.

6. **Quality audit** — Read all 61 initial skill.md files. Assessed each for quality, relevance to Flash Attention, and uniqueness. Flagged 15 cross-level duplicates (same source at multiple levels) and 1 irrelevant entry (ASIC hardware design paper).

7. **Consolidation** — Removed 29 entries total: 15 cross-level duplicates (kept at best-fit level), 1 irrelevant entry, 13 empty stubs. Removed all 41 HTML blog downloads (content preserved in document.md; PDFs and source code are more authoritative).

8. **External source integration** — Explored [awesome-LLM-driven-kernel-generation](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation) (32 resources evaluated across 5 tiers of relevance) and [AMD GEAK](https://github.com/AMD-AGI/GEAK) (55 knowledge files evaluated across 9 tiers). Integrated the most relevant entries: survey papers, ThunderKittens, FlagAttention, FMHA kernel reports, HIP optimization guides.

9. **Final verification** — Confirmed all 53 entries have complete skill.md + document.md, verified all PDFs are valid, all source code files are intact, and no entry duplicates another.

10. **Hierarchy consolidation** — Flattened the original 10-level hierarchy (L0-L9) to 6 levels (L0-L5) by merging levels that represent parallel optimization concerns at the same abstraction depth rather than distinct levels in the top-down workflow: work partitioning + pipelining → L2, memory hierarchy + warp cooperation → L3, tensor core + softmax + instruction scheduling → L4.

---

## 6. Directory Structure

```
knowledge_base/
├── L0_Model_Invocation_Level/                       (8 topics)
│   ├── ai_driven_kernel_optimization/               — QiMeng, GEAK, AutoTriton, FlagAttention
│   ├── amd_composable_kernel/                       — CK-Tile FA on AMD GPUs
│   ├── flash_attention_3_hopper/                    — FA3 deployment on H100
│   ├── flash_attention_free_lunch/                  — FA integration & version guide
│   ├── flexattention_fa4/                           — FlexAttention + FA4 backend
│   ├── pytorch_sdpa/                                — PyTorch SDPA dispatch
│   ├── thunderkittens/                              — Hazy Research tile primitives
│   ├── vllm_attention_backends/                     — vLLM backend selection, PagedAttention
│   ├── xformers_attention/                          — xFormers multi-backend attention
│   └── tensorrt_llm_attention/                      — TRT-LLM FMHA, XQA decode, paged KV
│
├── L1_Algorithmic_Stage_Level/                      (13 topics)
│   ├── original_flash_attention/                    — FA1 (Dao et al., 2022)
│   ├── online_softmax_to_flash_attention/           — Mathematical derivation
│   ├── flash_attention_2/                           — FA2 algorithm
│   ├── flash_attention_3/                           — FA3 algorithm
│   ├── flash_attention_4/                           — FA4 algorithm
│   ├── flash_mla_deepseek/                          — FlashMLA (DeepSeek)
│   ├── ring_attention/                              — Distributed FA for long context
│   ├── flashinfer_attention_engine/                 — Serving-aware FA
│   ├── triton_flash_attention_implementation/       — FA in Triton
│   ├── native_sparse_attention/                     — DeepSeek NSA: local+compressed+selected
│   ├── differential_attention/                      — Attention as difference of two softmax maps
│   ├── paged_attention/                             — KV cache as virtual memory pages (vLLM)
│   └── linear_attention_mamba2/                     — SSM-attention duality, Mamba-2
│
├── L2_Scheduling_Pipelining_Level/                  (14 topics)
│   ├── fa2_scheduling/                              — Sliced-Q/K partitioning, seq-length parallelism
│   ├── flexattention_scheduling/                    — Block-sparse iteration, masked region skipping
│   ├── nsight_profiling_flash_attention/            — Profiler-driven optimization methodology
│   ├── warp_specialization/                         — Producer-consumer pattern, TMA/WGMMA overlap
│   ├── cutlass_ping_pong_gemm/                      — 3-warpgroup ping-pong (1 producer + 2 consumers)
│   ├── flashattention3_pipelining/                  — FA3 2-stage GEMM-softmax pipeline
│   ├── learn_cutlass_hard_way/                      — GEMM optimization: naive → 89.5x speedup
│   ├── hopper_tma_tutorial/                         — TMA descriptors, mbarrier, phase sync
│   ├── fa3_warp_specialization/                     — FA3 producer/consumer, pingpong scheduling
│   ├── fa4_warp_roles/                              — FA4 5 specialized warps
│   ├── cutlass_pipelining_warp_spec/                — CUTLASS dual barriers, circular buffers
│   ├── rocm_profiling_attention/                    — AMD ROCm profiling (rocprof, Omniperf)
│   ├── kperfir_compiler_profiling/                  — KPerfIR compiler-centric Triton IR profiling (OSDI'25)
│   └── split_kv_decode_scheduling/                  — FlashDecoding split-KV, parallel reduction
│
├── L3_Memory_Thread_Cooperation_Level/              (13 topics)
│   ├── gpu_memory_hierarchy/                        — Register→L1→L2→HBM latencies/bandwidths
│   ├── flash_attention_tiling_recomputation/        — O(N²)→O(N) tiling, backward recomputation
│   ├── why_flash_attention_memory_bound/            — Roofline analysis, 97% N×N traffic
│   ├── tma_deep_dive/                               — Conveyor-belt TMA, multi-dim tiling
│   ├── cutlass_cute_layout/                         — CuTe layout algebra, MMA/Copy atoms
│   ├── flash_attention_tile_tuning/                 — "Large tile trap", fast-math rescue
│   ├── shared_memory_swizzling_bank_conflicts/      — XOR swizzling, bank conflict elimination
│   ├── amd_mi300x_flash_attention/                  — CDNA3 chiplet, 5.3 TB/s HBM3, MFMA
│   ├── amd_cdna_architecture_guide/                 — CDNA3/CDNA4 chiplet arch, CU, cache, HBM specs
│   ├── mi300_compute_memory_partitioning/           — MI300 SPX/CPX modes, NPS1/NPS4, deployment
│   ├── amd_wavefront_cooperation/                   — Wave-64 model, MFMA, butterfly reductions
│   ├── blackwell_tmem/                              — Tensor Memory (TMEM), 128x256x32-bit
│   └── thread_block_clusters/                       — Cluster launch, DSMEM, TMA multicast
│
├── L4_Compute_Kernel_Optimization_Level/            (14 topics)
│   ├── wgmma_hopper_tutorial/                       — WGMMA m64nNk16, SS/RS variants
│   ├── flashattention2_hopper_cutlass/              — FA2 tile shape crisis at d=256
│   ├── amd_fmha_kernel_internals/                   — FMHA V3 fwd/bwd/splitkv/FP8/prefill
│   ├── amd_mfma_matrix_core_programming/            — MFMA intrinsics, tile shapes, FP8/FP4 examples
│   ├── amd_gfx9_kernel_optimization/                — Register usage, LDS, global mem, cross-lane ops
│   ├── online_softmax_algorithm/                    — Running max/sum recurrence
│   ├── fa4_official_paper/                          — Polynomial exp, conditional rescaling
│   ├── fast_math_softmax/                           — flush_to_zero, 34-72% speedup
│   ├── fast_softmax_cuda_kernel/                    — Warp shuffles, vectorized loads
│   ├── cutlass_instruction_mapping/                 — CuTe → hardware instruction sequences
│   ├── instruction_overlap_warp_spec/               — TMA + WGMMA concurrent execution
│   ├── handwritten_ptx_optimization/                — Branchless PTX, fast exp2/reciprocal
│   ├── sfu_bottleneck_asymmetric_scaling/           — 512:1 tensor core vs SFU imbalance
│   └── umma_blackwell/                              — UMMA on SM100, TMEM operands, CTA-group
│
├── L5_Numerical_Precision_Level/                    (8 topics)
│   ├── fa3_fp8_fp16/                                — Block quantization, Hadamard processing
│   ├── nvidia_fp8_formats/                          — E4M3/E5M2, MXFP8, Transformer Engine
│   ├── quip_incoherent_processing/                  — Randomized Hadamard, 2.6x error reduction
│   ├── pytorch_sdpa_precision/                      — Backend precision, float32 upcast
│   ├── sageattention2_mixed_precision/              — INT4 Q/K + FP8 P/V, 3-5x over FA2
│   ├── mxfp_microscaling_formats/                   — OCP MX standard, MXFP4/MXFP8, Blackwell
│   ├── amd_cdna_low_precision_types/                — AMD FP4/FP6/FP8, MXFP block scaling, CDNA3/4
│   └── blackwell_fp4_attention/                     — FP4 (E2M1) tensor cores, 2x FP8 throughput
│
└── README.md                                        (this file)
```
