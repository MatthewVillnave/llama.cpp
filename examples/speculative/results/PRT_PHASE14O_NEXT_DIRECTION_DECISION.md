# PRT Phase 14O — Next Research Direction Decision

## Current State

- **PRT float32 path**: Quality-valid but speed-negative. Phase 13 proved PRT active replacement quality, but float32 sidecars were materially slower than native ggml computation.
- **PRT INT8 path**: Validated on 0.5B, 3B, and 7B. Per-row INT8 quantization with packed sidecar format recovers near-native throughput on all measured model sizes.
- **3B INT8**: Reached native-parity repeatability (INT8/native avg ratio 1.000×) — checkpoint tagged `PRT_PHASE14G_3B_INT8_REPEATABILITY_CHECKPOINT`.
- **7B INT8**: Reached near-native repeatability (INT8/native avg ratio 0.977×, median ratio 0.990×) — checkpoint tagged `PRT_PHASE14N_7B_INT8_CHECKPOINT`.
- **Current claims**: Scoped strictly to measured CPU setup. No universal, production, GPU, or larger-than-7B claims are permitted.

## Validated Claims

The following claims are supported by experimental evidence:

- Qwen2.5-3B INT8 PRT reached repeatable native-parity throughput on the measured CPU setup (21.0 t/s both modes, 10/10 runs).
- Qwen2.5-7B INT8 PRT retained ~97.7% average and ~99.0% median native generation throughput on the measured CPU setup.
- INT8 sidecars preserved tested output quality across all measured prompts and model sizes (0 quality degradations).
- INT8 sidecars were much faster than float32 PRT sidecars (e.g., 0.5B: 94.3 t/s INT8 vs 50.1 t/s float32).
- Sidecar loading succeeded 28/28 on 7B and 36/36 on 3B. Fallback was limited to force-native layers 11 and 15.
- The packed sidecar format (combined int8 weights + per-row scales in one file) scales across model sizes.

## Forbidden Claims

The following are NOT supported and must not be stated:

- ❌ No universal speedup claim.
- ❌ No production readiness claim.
- ❌ No GPU comparison claim.
- ❌ No larger-than-7B claim.
- ❌ No exact equivalence across all prompts/tasks.
- ❌ No deployment claim.
- ❌ No claim outside this measured CPU setup.

---

## Candidate Next Directions

### Option 1 — Full 8-prompt 7B validation

**Goal**: Run the same 8-prompt suite used for 3B against Qwen2.5-7B native and INT8 PRT.

| | |
|---|---|
| **Pros** | Makes the 7B quality story symmetric with 3B; strengthens external credibility; low engineering risk |
| **Cons** | Does not advance speed; more validation work without new capability |
| **Risk** | Low |
| **Value** | High |
| **Time cost** | Low to medium |
| **Dependency** | None |

### Option 2 — Longer-context / larger-n stability study

**Goal**: Test whether native parity survives at higher token counts and longer context.

| | |
|---|---|
| **Pros** | Tests practical usefulness at production-scale generation lengths; could catch memory/swap degradation |
| **Cons** | More resource-intensive; likely slower; does not advance new capability |
| **Risk** | Medium |
| **Value** | High |
| **Time cost** | Medium |
| **Dependency** | None |

### Option 3 — Optimization toward native-beating throughput

**Goal**: Turn native parity into actual speedup over native.

Possible targets:
- INT8 AVX2/VNNI kernel refinement
- Reduce per-token dequantization overhead
- Improve sidecar memory layout / cache behavior
- Reduce force-native layer count (currently 11, 15)
- Native ggml/backend integration

| | |
|---|---|
| **Pros** | Directly attacks the next major milestone; could produce real speedup |
| **Cons** | May require deeper kernel/backend work; risk of diminishing returns |
| **Risk** | Medium/High |
| **Value** | Very High |
| **Time cost** | High |
| **Dependency** | Should wait until 8-prompt 7B validation is complete |

### Option 4 — INT4 sidecar prototype

**Goal**: Build INT4 sidecar extraction and runtime, test if quality survives at 2× compression.

| | |
|---|---|
| **Pros** | Could reduce memory traffic another 2×; tests next compression frontier |
| **Cons** | Higher quality risk; more complex kernel; needs offline parity validation before runtime |
| **Risk** | High |
| **Value** | High |
| **Time cost** | Medium/High |
| **Dependency** | Requires offline parity validation before runtime test |

### Option 5 — Native ggml/backend integration design

**Goal**: Move PRT off custom callback and into ggml backend scheduling.

| | |
|---|---|
| **Pros** | More durable path to real performance; removes custom callback limitations; could improve scheduling and backend efficiency |
| **Cons** | Highly invasive; high engineering complexity; may take longer before measurable results |
| **Risk** | High |
| **Value** | Very High |
| **Time cost** | High |
| **Dependency** | Should follow after Phase 10E/Phase 14 full validation is done |

### Option 6 — Public/lab writeup package

**Goal**: Document Phase 14 results while they are fresh. Build X/blog/GitHub credibility with strict claims discipline.

| | |
|---|---|
| **Pros** | Captures result; forces claims discipline; useful for external credibility; low risk |
| **Cons** | Does not advance technical performance |
| **Risk** | Low |
| **Value** | High |
| **Time cost** | Low/Medium |
| **Dependency** | None |

---

## Ranking

By credibility value, technical upside, and dependency order:

| Rank | Option | Rationale |
|------|--------|-----------|
| **1** | Full 8-prompt 7B validation | Completes the 7B story symmetrically; low risk; no dependencies; highest credibility value per time invested |
| **2** | Public/lab writeup package | Captures results; disciplines claims; low cost; no dependencies |
| **3** | Longer-context / larger-n stability | Tests practical deployment range; important signal; no dependencies |
| **4** | INT4 sidecar prototype | High value, tests next compression frontier; needs offline validation first |
| **5** | Optimization toward native-beating throughput | Highest ceiling; depends on completing 7B quality story first |
| **6** | Native ggml/backend integration design | Highest architectural value; longest timeline; depends on having clean INT8 base |

---

## Recommendation

**Recommended next phase: Phase 14P — Full 8-Prompt 7B INT8 Validation**

### Reason

Before trying to beat native throughput, publish broadly, or invest in deeper backend integration, make the 7B INT8 quality claim as clean and credible as the 3B claim. The 3B story used 8 prompts to build a strong quality foundation. The 7B story currently rests on 4 prompts. Completing the symmetry between 3B and 7B is:

- **Low risk**: The 7B INT8 repeatability result (Phase 14M) makes failure unlikely.
- **High credibility value**: External validation of the 7B result requires matching the 3B benchmark structure.
- **Low time cost**: The infrastructure and scripts already exist. Run time is predictable.
- **Required for next steps**: All higher-value options (optimization, INT4, backend integration) benefit from a clean, validated 7B checkpoint as the baseline.

After 14P, two paths branch cleanly:

- **Path A**: Lab writeup package (Phase 14Q) → then INT4 prototype or optimization
- **Path B**: INT4 prototype (Phase 14Q) → then writeup

The writeup should precede publication or external communication. The INT4 work should precede optimization (since INT4 kernel refinement and optimization share similar tooling).

---

## Proposed Next Phase

**Phase 14P: Full 8-Prompt 7B INT8 Validation**

**Goal**: Run the same 8-prompt suite used for 3B INT8 validation against Qwen2.5-7B native and INT8 PRT. Confirm quality preservation, near-native timing, shape evidence, sidecar loading, fallback behavior, and memory stability across all 8 prompts.

**Pass criteria**:
- Native completed: 8/8
- INT8 PRT completed: 8/8
- Clean outputs: 8/8
- Exact or semantic matches: 8/8
- Quality degradations: 0
- Sidecars loaded: 28/28
- Fallback limited to layers 11, 15
- INT8/native t/s ratio ≥ 0.90 (avg or median)

**Do not run benchmarks during this phase. Do not modify PRT math or sidecar format.**

---

## What Not to Do Next

- ❌ Do not jump to larger models before 7B full 8-prompt validation is complete.
- ❌ Do not claim universal speedup before completing optimization path.
- ❌ Do not start invasive backend work before documenting the current result.
- ❌ Do not run parallel large-model tests on this machine.
- ❌ Do not modify existing tags.
- ❌ Do not stage models, sidecars, or binaries.