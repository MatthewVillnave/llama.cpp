# PRT Phase 16A — Post-Phase-15 Strategy Decision

## Verdict

**RECOMMEND_14B_FEASIBILITY** ✅

---

## Context

Phase 15I froze the INT6 optimized pipeline checkpoint on Qwen2.5-7B. The CLI-level optimization work is complete. Matt's broader goal is to make CPU inference materially more viable — enabling 14B-class or larger models to run without a GPU on CPU-only/edge systems. PRT is the sidecar-backed FFN replacement proof path inside the broader SDI vision.

---

## What Phase 15 Proved

1. **INT8 sidecar path works correctly** — near-native generation tok/s, zero quality degradations, 8-prompt validation on 7B
2. **INT6 compressed sidecar path works correctly** — generation parity confirmed across offline parity, runtime canary, 8-prompt validation, longer-gen, repeatability
3. **Provenance logging is essential** — the duplicated-sidecar blind spot is eliminated; every run logs 28/28 unique SHAs
4. **Manifest cache is effective** — eliminated ~811ms/run of repeated SHA hashing
5. **mmap reduces I/O overhead** — MAP_PRIVATE + MADV_SEQUENTIAL saves ~416ms vs fread
6. **LUT4x unpack is correct but modest** — ~32ms improvement; diminishing returns on scalar-only code
7. **Total setup reduction** — 2,218ms → 913ms (−59%) across Phase 15E–H
8. **INT4 per-row is offline NO-GO** — insufficient accuracy for production use

---

## What Phase 15 Did Not Prove

1. **Model-scale viability** — all validation on 7B only; no evidence the approach generalizes to 14B or larger
2. **Memory feasibility on larger models** — 15GB RAM machine; 14B FFN_up would be ~2× larger per layer; no measurement of whether it fits
3. **Backend integration feasibility** — the current CLI-sidecar approach has inherent per-run overhead; whether ggml can absorb this is unproven
4. **Generation quality at scale** — only tested short-to-medium generation (≤320 tokens); no evidence of stability over long contexts with sidecars
5. **Architecture generalization** — Phase 14B's tensor-shape-detection approach (blk.%d.ffn_up pattern) was never validated on models other than Qwen2.5

---

## Why CLI-Level Optimization Is Exhausted

The remaining ~910ms setup is dominated by memory-bandwidth-bound CPU unpack (1.9B elements across 28 layers). LUT4x hit diminishing returns. Further scalar loop tweaks will not move this number materially. The next gain requires either:
- Backend integration (avoid per-run mmap overhead by pre-loading into ggml tensor space)
- Larger model scale (proportionally more compute time amortizes setup overhead)
- Neither requires more CLI optimization

---

## Option 1 — Backend / ggml Integration Design

**Purpose:** Move PRT/INT6 deeper into ggml/backend so sidecar decode/layout/compute is handled closer to execution instead of CLI-level setup.

### Expected Upside
- Eliminates per-run mmap/file-open overhead (~30ms saved, marginal)
- Enables ggml-level tensor reuse (sidecar data stays in tensor space across tokens)
- Cleaner API for external callers (not just CLI)
- Foundation for true production deployment

### Complexity
- **High.** Requires understanding ggml tensor lifecycle, model loading hooks, and backend architecture deeply
- Integration point is not obvious — need to decide where in the ggml graph PRT sidecars attach

### Risk
- **High.** ggml internals are non-trivial; integration could introduce subtle correctness issues
- No prior art to reference; this is novel architecture work
- Could break existing working CLI path

### Likely Files/Subsystems
- `src/ggml/*.cpp` — new PRT tensor type and compute node
- `src/llama.cpp` — model loading hooks for sidecar pre-loading
- `tools/cli/cli.cpp` — CLI integration for sidecar paths (already exists)

### Prerequisite Knowledge Needed
- ggml tensor graph architecture
- How llama.cpp manages KV cache and tensor lifecycle
- Where MLP_UP is computed in the ggml graph

### Safe First Phase
Design only: write a design document describing the target architecture, integration points, and success criteria. No code changes.

### Success Criteria
- Design document with clear integration plan
- Identified hooks: where sidecars enter ggml tensor space, how they persist across tokens
- Risk assessment for correctness and performance
- Decision on whether to proceed with implementation

### Failure Modes
- Design reveals fundamental incompatibility with ggml architecture
- Integration complexity exceeds available time/resources
- Correctness risks are too high to proceed safely

---

## Option 2 — 14B Feasibility Canary

**Purpose:** Test whether the Phase 15 pipeline can scale to a 14B-class model and whether CPU-only inference viability improves with larger model size.

### Expected Upside
- **Directly tests Matt's stated goal** — whether PRT enables 14B-class CPU inference
- If 14B works at acceptable speed, proves the approach generalizes
- If 14B doesn't fit or is too slow, provides a definitive no-go without wasted integration effort
- High information gain either way

### Hardware/RAM/Disk Risks
| Risk | Assessment |
|------|-------------|
| RAM — model weights | Qwen2.5-14B-Q4_K_M ≈ 9GB in memory (model) + KV cache + sidecars |
| RAM — 28×14B sidecars | Each INT8/INT6 sidecar is ~2× the 7B size; 28 layers × ~100MB each ≈ 2.8GB |
| RAM — total | Model (~9GB) + KV (~1-2GB) + sidecars (~2.8GB) + ggml overhead ≈ 14-15GB |
| **Risk** | **Borderline on 15GB machine — need careful measurement** |
| Disk | Need ~15GB free; 126GB available ✅ |
| Model acquisition | Need Qwen2.5-14B-Q4_K_M GGUF; can be downloaded |

### Model Acquisition
- `Qwen2.5-14B-Instruct-Q4_K_M.gguf` — available from HuggingFace
- Estimated size: ~9GB
- Download time: depends on connection (not my problem, Matt handles it)

### Sidecar Size Estimates for INT8 and INT6
| Format | 7B per-layer | 14B per-layer | 28 layers total |
|--------|-------------|-------------|----------------|
| INT8 | ~67.9MB | ~130MB | ~3.6GB |
| INT6 | ~51MB (packed) | ~98MB (packed) | ~2.7GB |

### Expected Setup Cost
- **INT6:** ~2× current ≈ 1,800ms (2× elements, same memory bandwidth)
- **INT8:** ~2× current ≈ 1,800ms
- Setup is one-time; generation time is per-token

### Expected Runtime Cost
- 14B forward pass is slower per token than 7B (more parameters)
- But generation tok/s for a larger model on the same CPU is already lower — PRT overhead is a smaller fraction
- Key question: does 14B with PRT still produce usable tok/s (e.g., >5 tok/s)?

### Safe Canary Plan
1. Download Qwen2.5-14B-Q4_K_M (Matt's responsibility)
2. Generate INT8 sidecars for 14B (reuse existing `llama_prt_sidecar_extract.cpp`)
3. Run tiny canary: `--n 8 --c 128` — does it load and generate?
4. If yes, run 8-prompt validation matrix (same as Phase 14P)
5. Compare generation tok/s vs native 14B baseline
6. If tok/s is acceptable (say >5 tok/s), proceed to full validation

### Success Criteria
- 14B model loads with INT8/INT6 sidecars without OOM
- Generation produces coherent output (not garbage)
- Generation tok/s ≥ 5.0 on the measured CPU setup
- Quality degradations = 0 across 8 prompts
- Provenance logs verify 28/28 unique SHAs (or appropriate layer count for 14B)

### Failure Modes
- OOM on 15GB machine → NO-GO for 14B on current hardware
- Generation tok/s < 5.0 → viable but slow; indicates hardware limit not architecture problem
- Sidecar generation fails for 14B dimensions → need to adapt generator

---

## Option 3 — Benchmark Harness / Provenance Integration

**Purpose:** Make validation more repeatable and trustworthy by integrating provenance logs into benchmark summaries automatically.

### Expected Upside
- Eliminates manual log parsing after each run
- Improves research credibility (verifiable, automated provenance)
- Low risk — mostly automation work

### Complexity
- Low. Python or shell scripting around existing CLI + log parsing.

### What It Would Automate
- Parse provenance logs → structured JSON
- Generate per-run summary: SHA verification, load time, gen tok/s
- Compare against baseline runs
- Flag regressions automatically

### How It Improves Credibility
- External reviewers can verify claims from raw log data
- Reduces human error in manual log parsing
- Creates audit trail for performance claims

### Success Criteria
- Single command generates a provenance-clean benchmark report
- Report includes: model SHA, sidecar SHA array, setup time, gen tok/s, pass/fail verdict

### Failure Modes
- Unlikely to fail — this is a thin automation layer

---

## Option 4 — Phase 15 Technical/Public Writeup

**Purpose:** Document current results and claim boundaries for internal/public communication.

### Expected Upside
- Preserves institutional knowledge
- Enables external credibility and collaboration
- Clarifies what claims are allowed/forbidden
- Could inform upstream llama.cpp community interest

### Risk of Overclaiming
- **High.** The temptation to generalize from 7B results to all models is strong. A public writeup needs extremely precise claim boundaries or it will be misused.

### What Should Be Public
- The INT8/INT6 quantization approach for PRT sidecars
- The validation methodology and pass criteria
- The generation tok/s measurements on Qwen2.5-7B
- The provenance logging system design

### What Should NOT Be Public
- Specific setup time numbers (hardware-dependent, easily misinterpreted)
- Any claim about larger models without 14B validation
- "Production ready" or "universally faster" framing
- The specific GGUF models and sidecar files

### Whether to Wait for 14B First
**Yes.** The writeup is much stronger with 14B validation. Publishing now would invite criticism of scale-limited claims. Wait for Option 2 to complete (or fail).

---

## Ranking Table

| Option | Upside | Risk | Time Cost | Info Gain | CPU/SDI Alignment | Rank |
|--------|--------|------|-----------|-----------|-------------------|------|
| **14B Feasibility Canary** | High | Low-Medium | Medium | **High** | **High** | **1** |
| Backend / ggml Integration | High | **High** | High | Medium | Medium | 2 |
| Benchmark Harness | Low | Low | Low | Low | Low | 3 |
| Public Writeup | Medium | **High** (overclaim) | Medium | Medium | Medium | 4 |

---

## Recommended Phase 16B

**Phase 16B: 14B Feasibility Canary**

### 10-Bullet Plan
1. Matt downloads `Qwen2.5-14B-Instruct-Q4_K_M.gguf` from HuggingFace to `~/models/gguf/qwen2.5/`
2. Run `sha256sum` on the downloaded model; update expected SHA in the plan
3. Modify `llama_prt_sidecar_extract.cpp` or create a 14B variant that uses correct M, N, K for Qwen2.5-14B (M=24576 or appropriate for 14B; verify from model tensor shapes)
4. Generate INT8 sidecars for 14B into `/tmp/prt_sidecars_14b_int8/`
5. Run tiny canary: `n=8, c=128` — does it load without OOM and generate coherent output?
6. If canary passes: run 8-prompt × 80-token validation matrix (INT8, same as Phase 14P)
7. If canary OOMs: try INT6 instead (smaller sidecars)
8. Compare generation tok/s vs native 14B baseline
9. Record: loaded layer count, unique SHA count, gen tok/s, quality assessment
10. Freeze with verdict: FEASIBLE_14B / PARTIAL_14B_FIT / INFEASIBLE_14B_HARDWARE

### Success Thresholds
- **FEASIBLE_14B:** 14B loads without OOM, gen tok/s ≥ 5.0, quality degradations = 0
- **PARTIAL_14B_FIT:** Loads but gen tok/s < 5.0 — architecture works, hardware limit
- **INFEASIBLE_14B_HARDWARE:** OOM even with INT6 sidecars — need more RAM

---

## Backup Path

**Backend / ggml Integration Design (Phase 16B-alt)**

If Matt decides to prioritize architecture over scale testing:
1. Read ggml source: `src/ggml.cpp`, `src/ggml-backend.cpp`, `src/llama.cpp` (model loading)
2. Identify the MLP_UP compute node and tensor lifecycle
3. Write `examples/speculative/ggml_prt_integration_design.md` with:
   - Target architecture diagram
   - Integration hooks: where sidecar tensors attach to ggml graph
   - Estimated performance impact (if any)
   - Correctness risk assessment
   - Decision gate: proceed to implementation or not
4. Review with Matt; get approval before any code changes

---

## What Not To Do Next

❌ Do NOT continue CLI-level scalar loop optimization — Phase 15H proved diminishing returns

❌ Do NOT write a public writeup without 14B validation — it will invite scale-limited criticism

❌ Do NOT attempt backend integration without a design phase — ggml internals are complex and correctness-critical

❌ Do NOT claim production readiness or universal speedup under any circumstances

❌ Do NOT attempt INT4 per-row work — offline NO-GO remains

---

## Allowed Claims

✅ Phase 15 proved the INT8 and INT6 sidecar approach on Qwen2.5-7B in Matt's measured CPU setup

✅ CLI-level optimization is exhausted; further gains require architecture-level changes

✅ 14B feasibility is unknown and requires direct measurement

✅ INT8 remains the validated runtime path; INT6 remains experimental

---

## Forbidden Claims

❌ Do not claim Phase 15 results generalize to all models

❌ Do not claim production readiness

❌ Do not claim universal speedup

❌ Do not claim CPU inference viability without 14B validation

❌ Do not claim INT6 replaces INT8

❌ Do not claim larger-than-7B support without evidence

❌ Do not claim backend integration is straightforward or low-risk

---

**Test environment:**
- Branch: `experimental/prt-phase14a-packed-sidecars`
- Commit: `3a4299619`