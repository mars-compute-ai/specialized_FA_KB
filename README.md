# Flash Attention Optimization Knowledge Base

## 1. Purpose and Motivation

### Why This Knowledge Base Exists

Flash Attention is the dominant attention algorithm in modern LLM training and inference. However, optimizing it for a specific GPU architecture remains a labor-intensive, expert-driven process. Each generation of GPU hardware (NVIDIA Ampere → Hopper → Blackwell, AMD CDNA2 → CDNA3 → CDNA4) introduces new matrix instructions, memory hierarchies, and pipeline capabilities that demand kernel rewrites. The optimization space is vast, spanning at least 10 distinct levels of granularity, from high-level framework selection down to individual instruction scheduling.

This knowledge base was built to give an **AI agent** (or a human kernel engineer) the structured reference material needed to navigate that optimization space systematically. Instead of searching through hundreds of papers, blog posts, and codebases, the agent can consult a curated set of **53 skill cards** — each one a self-contained decision guide for a specific optimization technique, tied to a specific level of the hierarchy.

### Why a Multi-Level Hierarchy?

Flash Attention optimization is not a single problem. It decomposes into a stack of interdependent decisions:

```
L0  Which kernel implementation should I use?          (framework / library choice)
L1  What algorithm should the kernel implement?        (FA1, FA2, FA3, FA4, MLA, Ring)
L2  How should work be distributed across the GPU?     (thread blocks, warps, occupancy)
L3  How should I pipeline data movement and compute?   (producer-consumer, software pipelining)
L4  Where should each tensor live in the memory hierarchy? (HBM, L2, shared memory, registers)
L5  How should warps cooperate within a thread block?  (warp roles, synchronization, barriers)
L6  What tile shapes and matrix instructions to use?   (WGMMA, MFMA, MMA atom selection)
L7  How to implement softmax efficiently?              (online softmax, fast-math, polynomial approx)
L8  How to schedule individual instructions?           (PTX tuning, SFU avoidance, instruction overlap)
L9  What numerical precision to use?                   (FP16, BF16, FP8, INT4, mixed precision)
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
| Total topics | 53 |
| Skill cards (`skill.md`) | 53 |
| Technical documents (`document.md`) | 53 |
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
| **AMD CDNA4** (MI350X) | gfx950, next-gen | New MFMA variants, enhanced memory subsystem |

### Attention Variants Covered

| Variant | Where in KB | Key Optimization Challenge |
|---------|-------------|---------------------------|
| Standard MHA | L1/original_flash_attention through L1/flash_attention_4 | Tiling, online softmax, pipeline design |
| Grouped-Query Attention (GQA) | L1/flashinfer_attention_engine | K/V head sharing changes memory access patterns |
| Multi-Head Latent Attention (MLA) | L1/flash_mla_deepseek | Fusing KV decompression into attention; 93.3% KV cache reduction |
| Ring Attention | L1/ring_attention | Distributing FA across GPUs with overlapped communication |
| Sparse Attention | L6/amd_fmha_kernel_internals | CSR-like patterns reducing O(N^2) to O(N*k) |
| Paged Attention | L6/amd_fmha_kernel_internals, L1/flashinfer | Block table indirection for serving workloads |

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

Each level addresses a distinct class of optimization decisions. The levels are ordered from coarse-grained (L0) to fine-grained (L9), mirroring the natural progression of kernel development.

#### L0 — Model/Invocation Level (8 topics)
**Question**: *Which attention implementation should I call, and how?*

This is the entry point. Before writing any kernel code, you must decide which existing implementation to use — or whether to generate a new one. This level covers PyTorch SDPA dispatch, FlexAttention for custom patterns, xFormers for multi-backend support, AMD's Composable Kernel, and emerging AI-driven kernel generation tools.

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

**Why it matters for AI**: An AI agent must first decide whether to use an off-the-shelf kernel, extend one via FlexAttention, or generate a new one with AI-driven tools. This level provides the decision framework.

#### L1 — Algorithmic/Mathematical Stage Level (9 topics)
**Question**: *What mathematical decomposition should the kernel use?*

Flash Attention's core innovation is fusing tiled GEMM with online softmax. But the specific algorithm varies significantly across versions (FA1→FA4) and variants (MLA, Ring). This level covers the mathematical foundations, the evolution of algorithmic ideas, and novel attention architectures.

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

**Why it matters for AI**: When an agent needs to implement a new attention variant or understand why a specific FA version is faster, this level provides the algorithmic basis for reasoning about correctness and performance.

#### L2 — Work-Partition/Schedule Level (3 topics)
**Question**: *How should computation be distributed across the GPU?*

The same algorithm can perform very differently depending on how work is partitioned across thread blocks, warps, and the sequence/batch/head dimensions. This level covers scheduling strategies and profiling methodology.

| Topic | Key Content |
|-------|-------------|
| `fa2_scheduling` | Sliced-Q vs sliced-K warp partitioning, sequence-length parallelism |
| `flexattention_scheduling` | Block-sparse iteration, skipping masked regions, score_mod scheduling |
| `nsight_profiling_flash_attention` | Profiler-driven optimization: HBM thrashing, bank conflicts, MIO throttling, 4.9x speedup |

**Why it matters for AI**: Scheduling decisions determine GPU occupancy and load balance. The Nsight profiling entry teaches a methodology (not just a tool) for identifying which scheduling choice is causing a bottleneck.

#### L3 — Pipeline/Loop-Nest Level (5 topics)
**Question**: *How to overlap data movement with computation?*

Modern GPUs (especially Hopper and beyond) achieve peak performance only when memory transfers run concurrently with matrix math. This level covers the producer-consumer pipeline patterns that FA3 and FA4 depend on.

| Topic | Key Content |
|-------|-------------|
| `warp_specialization` | Producer-consumer pattern, 3 enabling conditions, TMA/WGMMA overlap |
| `cutlass_ping_pong_gemm` | 3-warpgroup ping-pong architecture (1 producer + 2 consumers) |
| `flashattention3_pipelining` | FA3's 2-stage GEMM-softmax pipeline, 250x throughput gap exploitation |
| `learn_cutlass_hard_way` | GEMM optimization progression: naive → 89.5x speedup, tiling/pipelining/autotuning |
| `hopper_tma_tutorial` | TMA mechanics: descriptors, mbarrier, phase-based sync, pipeline integration |

**Why it matters for AI**: Without pipelining, even a mathematically optimal algorithm will leave the GPU idle 50%+ of the time. This level teaches the patterns that turn a correct kernel into a fast one.

#### L4 — Memory-Hierarchy/Data-Movement Level (8 topics)
**Question**: *Where should each tensor fragment reside, and how should data move?*

Flash Attention is fundamentally a memory-optimization technique — it exists because naive attention is memory-bound. This level covers GPU memory hierarchies (for both NVIDIA and AMD), tiling strategies, TMA hardware, shared memory swizzling, and the AMD MI300X's unique chiplet architecture.

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

**Why it matters for AI**: The single biggest performance variable in attention is how data moves through the memory hierarchy. Getting tiling, swizzling, or TMA usage wrong can cost 2-10x performance.

#### L5 — Intra-CTA Cooperation / Warp Level (4 topics)
**Question**: *How should warps cooperate within a thread block?*

FA3 and FA4 assign different roles to different warps — some load data, others compute matrix multiplies, others handle softmax. This level covers how those roles are designed, synchronized, and adapted for both NVIDIA (warp-32) and AMD (wavefront-64) architectures.

| Topic | Hardware | Key Content |
|-------|----------|-------------|
| `fa3_warp_specialization` | NVIDIA Hopper | FA3 producer/consumer split, pingpong scheduling, 570→661 TFLOPs/s ablation |
| `fa4_warp_roles` | NVIDIA Blackwell | FA4's 5 specialized warps (Load, MMA, Softmax, Correction, Epilogue) |
| `cutlass_pipelining_warp_spec` | NVIDIA Hopper | CUTLASS pipeline abstraction: dual barriers, circular buffers, setmaxnreg |
| `amd_wavefront_cooperation` | AMD MI250X/MI300X | Wave-64 model, MFMA scheduling, buffer-to-LDS transfers, butterfly reductions |

**Why it matters for AI**: Warp specialization is the mechanism that enabled FA3's 2x improvement over FA2. An agent tuning an attention kernel must understand how to assign and synchronize warp roles.

#### L6 — Micro-Kernel/Tensor-Core Level (3 topics)
**Question**: *What matrix instruction shapes and configurations to use?*

The innermost loop of Flash Attention executes matrix multiply-accumulate instructions (WGMMA on NVIDIA, MFMA on AMD). Choosing the wrong tile shape, operand source (registers vs shared memory), or instruction variant can collapse performance.

| Topic | Hardware | Key Content |
|-------|----------|-------------|
| `wgmma_hopper_tutorial` | NVIDIA Hopper | WGMMA m64nNk16, SS vs RS variants, descriptor-based SMEM, swizzle modes |
| `flashattention2_hopper_cutlass` | NVIDIA Hopper | FA2 microkernel: tile shape crisis (128×128 collapses at d=256), SS/RS selection |
| `amd_fmha_kernel_internals` | AMD MI300X/MI350X | FMHA V3 fwd/bwd/splitkv/FP8/prefill kernels, wave-group scheduling, MFMA pipeline |

**Why it matters for AI**: A bad tile shape choice can cause register spills that destroy performance (128×128 drops from 308 to 36.7 TFLOPs). This level provides the data to avoid that trap.

#### L7 — Softmax/Reduction Level (4 topics)
**Question**: *How to implement softmax efficiently inside a fused kernel?*

Softmax is the non-GEMM bottleneck in attention. It requires global reductions (max, sum) that are fundamentally at odds with tiling. This level covers the online softmax algorithm, polynomial approximations of the exponential function, and fast-math trade-offs.

| Topic | Key Content |
|-------|-------------|
| `online_softmax_algorithm` | Running max/sum recurrence for tiled softmax, practical kernel implementation |
| `fa4_official_paper` | FA4 polynomial exp (degree-3, 8.77e-5 error), conditional rescaling (τ=8.0), 1613 TFLOPs/s |
| `fast_math_softmax` | flush_to_zero + approx rounding: 34-72% speedup, SASS instruction comparison |
| `fast_softmax_cuda_kernel` | Warp-level shuffles, vectorized loads, block reductions, 50% speedup on A100 |

**Why it matters for AI**: Softmax is where FA3→FA4's biggest algorithmic innovations happen (polynomial exp, conditional rescaling). An agent optimizing attention performance must understand these techniques.

#### L8 — Instruction-Mix/Low-Level Optimization (4 topics)
**Question**: *How to squeeze out the last few percent of performance?*

After the algorithm, scheduling, pipeline, and tile shapes are set, the final optimization frontier is instruction-level: replacing expensive instructions with approximate ones, overlapping different instruction types, and using inline PTX for branchless execution.

| Topic | Key Content |
|-------|-------------|
| `cutlass_instruction_mapping` | CuTe layout → hardware instruction sequences, MMA count formulas |
| `instruction_overlap_warp_spec` | TMA + WGMMA concurrent execution, quasi-out-of-order on in-order GPU |
| `handwritten_ptx_optimization` | Branchless `setp`+`selp`, fast exp2/reciprocal via inline PTX, 7-14% gains |
| `sfu_bottleneck_asymmetric_scaling` | 512:1 tensor core vs SFU imbalance, FA4's pipeline co-design for Blackwell |

**Why it matters for AI**: The SFU bottleneck analysis shows that on Blackwell, the exponential function unit is 512x slower than tensor cores — understanding this asymmetry drove FA4's entire design. An agent must reason at this level to design kernels for new hardware.

#### L9 — Numerical/Data-Type Level (5 topics)
**Question**: *What precision should the kernel use, and how to maintain accuracy?*

FP8 attention can deliver 2x the throughput of FP16, but naive quantization destroys accuracy. This level covers the precision formats, quantization strategies (block quantization, Hadamard incoherent processing), and mixed-precision techniques that make low-precision attention viable.

| Topic | Key Content |
|-------|-------------|
| `fa3_fp8_fp16` | FA3 FP8/FP16: block quantization, Hadamard incoherent processing, 1.2 PFLOPS |
| `nvidia_fp8_formats` | E4M3 vs E5M2, per-tensor/delayed/MXFP8 scaling, Transformer Engine integration |
| `quip_incoherent_processing` | Randomized Hadamard transform, 2.6x quantization error reduction, E8 lattice codebooks |
| `pytorch_sdpa_precision` | Backend precision differences, float32 upcast in math backend, reproducibility |
| `sageattention2_mixed_precision` | INT4 Q/K + FP8 P/V, outlier smoothing, 3-5x over FA2 with better accuracy |

**Why it matters for AI**: Precision is the highest-leverage performance knob — going from FP16 to FP8 can double throughput. But getting the quantization strategy wrong can silently corrupt model outputs. This level provides the trade-off analysis.

---

## 4. How to Use This Knowledge Base

### File Structure Per Topic

Each of the 53 topic directories follows a consistent structure:

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

1. **Identify the optimization level** — Is the bottleneck at the algorithm level? Memory? Instructions? Use the L0-L9 hierarchy to narrow down.

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
| Fix low GPU occupancy or load imbalance | L2 |
| Design or debug a producer-consumer pipeline | L3 |
| Fix memory bottlenecks or optimize data movement | L4 |
| Configure warp/wavefront roles or synchronization | L5 |
| Tune MMA/MFMA tile shapes or instruction selection | L6 |
| Optimize the softmax inner loop | L7 |
| Squeeze out last-mile performance with PTX/instruction tuning | L8 |
| Choose precision (FP16/BF16/FP8/INT4) or debug numerical issues | L9 |

**By target GPU:**

| Target GPU | Key entries |
|---|---|
| **AMD MI250X / MI300X** | L0/amd_composable_kernel, L4/amd_mi300x_flash_attention, L5/amd_wavefront_cooperation, L6/amd_fmha_kernel_internals |
| **NVIDIA Ampere (A100)** | L1/flash_attention_2, L2/fa2_scheduling, L6/flashattention2_hopper_cutlass |
| **NVIDIA Hopper (H100)** | L1/flash_attention_3, L3/flashattention3_pipelining, L5/fa3_warp_specialization, L6/wgmma_hopper_tutorial |
| **NVIDIA Blackwell (B200)** | L1/flash_attention_4, L5/fa4_warp_roles, L7/fa4_official_paper, L8/sfu_bottleneck_asymmetric_scaling |
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
- **Kernel engineer?** Go directly to the level matching your current bottleneck (L3-L6 are the core kernel engineering levels).
- **Porting NVIDIA→AMD?** Read L0/amd_composable_kernel for framework differences, L4/amd_mi300x_flash_attention for architecture differences, L5/amd_wavefront_cooperation for wave-64 vs warp-32, and L6/amd_fmha_kernel_internals for AMD kernel implementation details.
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

2. **Systematic web search** — For each of the 10 levels, searched for additional high-quality resources beyond the seed set. Evaluated ~100+ candidate documents.

3. **Original file acquisition** — Downloaded original PDFs from arXiv and conference proceedings. Cloned GitHub repositories and extracted relevant kernel source files. Downloaded AMD GEAK knowledge base files for Flash Attention kernel analysis.

4. **Skill card generation** — For each retained document, generated a structured `skill.md` with YAML frontmatter and standardized sections (What It Is, Key Concepts, When to Use, When NOT to Use, Code/Pseudo-code, Key Takeaways, References).

5. **Gap analysis** — Identified missing coverage for AMD GPUs, attention variants (MLA, Ring Attention), and AI-driven kernel generation tools. Searched specifically for these topics and added entries.

6. **Quality audit** — Read all 61 initial skill.md files. Assessed each for quality, relevance to Flash Attention, and uniqueness. Flagged 15 cross-level duplicates (same source at multiple levels) and 1 irrelevant entry (ASIC hardware design paper).

7. **Consolidation** — Removed 29 entries total: 15 cross-level duplicates (kept at best-fit level), 1 irrelevant entry, 13 empty stubs. Removed all 41 HTML blog downloads (content preserved in document.md; PDFs and source code are more authoritative).

8. **External source integration** — Explored [awesome-LLM-driven-kernel-generation](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation) (32 resources evaluated across 5 tiers of relevance) and [AMD GEAK](https://github.com/AMD-AGI/GEAK) (55 knowledge files evaluated across 9 tiers). Integrated the most relevant entries: survey papers, ThunderKittens, FlagAttention, FMHA kernel reports, HIP optimization guides.

9. **Final verification** — Confirmed all 53 entries have complete skill.md + document.md, verified all PDFs are valid, all source code files are intact, and no entry duplicates another.

---

## 6. Directory Structure

```
knowledge_base/
├── L0_Model_Invocation_Level/                    (8 topics)
│   ├── ai_driven_kernel_optimization/            — QiMeng, GEAK, AutoTriton, FlagAttention
│   ├── amd_composable_kernel/                    — CK-Tile FA on AMD GPUs
│   ├── flash_attention_3_hopper/                 — FA3 deployment on H100
│   ├── flash_attention_free_lunch/               — FA integration & version guide
│   ├── flexattention_fa4/                        — FlexAttention + FA4 backend
│   ├── pytorch_sdpa/                             — PyTorch SDPA dispatch
│   ├── thunderkittens/                           — Hazy Research tile primitives
│   └── xformers_attention/                       — xFormers multi-backend attention
│
├── L1_Algorithmic_Stage_Level/                   (9 topics)
│   ├── original_flash_attention/                 — FA1 (Dao et al., 2022)
│   ├── online_softmax_to_flash_attention/        — Mathematical derivation
│   ├── flash_attention_2/                        — FA2 algorithm
│   ├── flash_attention_3/                        — FA3 algorithm
│   ├── flash_attention_4/                        — FA4 algorithm
│   ├── flash_mla_deepseek/                       — FlashMLA (DeepSeek)
│   ├── ring_attention/                           — Distributed FA for long context
│   ├── flashinfer_attention_engine/              — Serving-aware FA
│   └── triton_flash_attention_implementation/    — FA in Triton
│
├── L2_Work_Partition_Schedule_Level/             (3 topics)
├── L3_Pipeline_Loop_Nest_Level/                  (5 topics)
├── L4_Memory_Hierarchy_Data_Movement_Level/      (8 topics)
├── L5_Intra_CTA_Warp_Cooperation_Level/          (4 topics)
├── L6_Micro_Kernel_Tensor_Core_Level/            (3 topics)
├── L7_Softmax_Reduction_Level/                   (4 topics)
├── L8_Instruction_Mix_Low_Level_Optimization/    (4 topics)
├── L9_Numerical_Data_Type_Level/                 (5 topics)
└── README.md                                     (this file)
```
