# PRT Phase 17A — Post-Phase-16 Strategy Decision

**Date:** 2026-05-09
**Branch:** `experimental/prt-phase14a-packed-sidecars`
**HEAD:** `fe7ef760b` (Phase 16O complete)
**Tag:** `PRT_PHASE16M_14B_INT6_EXPERIMENTAL_CHECKPOINT`

---

## Verdict

**RECOMMEND_NEXT_WEIGHT_CLASS** (with backend integration as subsequent phase)

**Reason in one line:** The highest-value next step for Matt's stated goal — "make CPU inference more viable on hardware people assume is impractical" — is to test whether 32B fits and works on the current machine. 14B proved it's possible at the edge of practical. 32B proves whether the edge moves.

---

## Context

**Phase 14–16 summary:** Packed INT8/INT6 sidecar-backed FFN replacement (PRT) is now validated end-to-end on Qwen2.5 models from 0.5B through 14B on Matt's measured CPU-only setup (Dell OptiPlex 7010, 15GB RAM). The pipeline preserves tested behavior with near-native generation throughput (~0.977–1.009×). The experimental checkpoint is frozen at `PRT_PHASE16M_14B_INT6_EXPERIMENTAL_CHECKPOINT`.

**Matt's stated goal:** Make CPU inference materially more viable on CPU-only / GPU-less / edge systems by using CPU-native / sub-dense inference paths instead of brute-forcing GPU-shaped dense inference.

**Matt's mini-goal:** Get models to run usefully on CPU-only hardware that people might normally assume is borderline or impractical.

**The implicit question:** Does the edge stop at 14B, or does it move to 32B? Answering that question matters more right now than optimizing the 14B path.

---

## What We Have Proven

These are the only claims the evidence supports:

- Packed INT8 and INT6 sidecar paths preserved tested behavior on Qwen2.5 models in Matt's measured CPU setup ✅
- Qwen2.5-14B Q4_K_M INT6 PRT passed: tiny canary, 8-prompt suite, n=320 longer prose, c=2048 larger context ✅
- Generation throughput was ~0.977–1.009× native in the tested prompt suite ✅
- Memory remained stable at ~12GB available throughout 14B validation ✅
- The sidecar loader has hardcoded size checks for 7B and 14B only ✅
- INT6 remains experimental; INT8 remains the validated runtime path ✅

---

## What We Have Not Proven

- No production readiness (no hardening, error handling, or edge-case coverage)
- No universal speedup (single CPU setup, limited prompt suite)
- No larger-than-14B support (30B+, 70B+ entirely unknown)
- No GPU comparison (CPU-only data only)
- No broad long-context guarantee (only tested to c=2048)
- No automatic model detection (hardcoded size checks in loader)
- No larger-model sidecar generation feasibility at 30B+
- No clean-state repeatability suite run
- No backend/ggml integration

---

## Option 1 — Backend/ggml Integration

**What it means:** Move PRT sidecar decode/replacement deeper into ggml/backend instead of CLI-level sidecar loading. Replace hardcoded size checks with gguf tensor dimension extraction via `ggml_get_tensor` + `gguf_get_tensor_info`. Design a fused INT6→matvec kernel as a ggml custom operation.

**Expected upside:**
- Automatic model detection (eliminates the silent-skip bug that Phase 16J caught)
- Kernel-level INT6 unpack enables genuine per-token speedups, not just throughput parity
- Self-healing fallback when sidecar loading fails (no more hand-specified layers 11, 15)
- Makes PRT a first-class compute path, not a CLI workaround
- Better long-term architecture — scales without hardcoded constants

**Complexity:** High. Requires deep ggml/src/backend core work. Custom op design and kernel optimization are non-trivial.

**Engineering risk:** Medium-high. Modifying ggml core could introduce regressions in the compute graph. Custom op integration requires careful API compliance.

**Required code areas:**
- `ggml/src/ggml-backend.cpp` — backend registration for PRT path
- `ggml/src/ggml-kernel-*.cpp` — fused INT6 unpack + matvec kernel
- `ggml/src/gguf.cpp` — tensor dimension extraction for model-size detection
- `tools/cli/cli.cpp` — replace hardcoded size checks with gguf API calls

**First safe implementation phase:** Extract M and K from the GGUF file's tensor metadata at load time instead of hardcoding. Use `gguf_get_tensor_by_name(ctx, "blk.*.ffn_up.weight")` to get shape, then verify against known sidecar sizes. This is a pure refactor — no kernel changes, no new compute path.

**Wall/setup overhead reduction:** Possibly. If the kernel-level INT6 unpack is faster than the current mmap+LUT4x path, per-token overhead could decrease. But this is speculative without benchmarking.

**Alignment with CPU-inference goal:** High long-term, medium short-term. The infrastructure fix is important but doesn't directly demonstrate that larger models work on CPU-only hardware.

**Information gain for this phase:** Medium. We'd know whether the loader can generalize to new model sizes without manual size-list updates.

**Likely next-step value:** High — unlocks the ability to run PRT on any model size automatically, which is required for production use and larger models.

---

## Option 2 — 14B Repeatability/Hardening

**What it means:** Re-run the full 14B pipeline from a clean state: fresh sidecar generation + clean runtime + 3–5 repeatability runs + more varied prompts (code, reasoning, longer factual). Validate memory/swap stability across multiple runs.

**Expected upside:**
- Strengthens the evidence base for public claims
- Validates that the pipeline works when rebuilt from source (reproducibility confirmation)
- More prompt coverage reduces the "only 8 prompts tested" concern
- Memory/swap stability confirmation across runs

**Complexity:** Low. Uses existing tools and already-validated pipeline.

**Engineering risk:** Low. Pure re-run, no new code.

**Hardware risk:** None. Already proven on this machine.

**Alignment with CPU-inference goal:** Low-medium. Valuable for credibility but doesn't expand the boundary of what works.

**Information gain:** Low-medium. The 14B result is already strong. Repeatability adds confidence but doesn't reveal anything new about the approach's limits.

**Likely next-step value:** Medium for credibility; low for expanding capability.

**Assessment:** Important as a credibility step but not as a next move. Matt's mini-goal is about expanding what's possible, not re-proving what's already shown.

---

## Option 3 — Bigger Model / Next Weight Class

**What it means:** Test whether Qwen2.5-32B (or similar 30B-range model) can run on the current 15GB RAM machine in native mode first, then with PRT sidecars. The goal is to answer: does the CPU-only boundary move from 14B to 32B?

**Possible next model sizes:**

| Model | Approx GGUF Size (Q4_K_M) | Feasibility on 15GB Machine |
|-------|--------------------------|----------------------------|
| Qwen2.5-32B | ~16–18GB | Tight — model + context could exceed RAM |
| Llama-3.1-8B | ~4.5GB | Easy — well within range |
| Llama-3.1-70B | ~40GB | NO — exceeds available disk and RAM |
| Mistral-7B | ~4GB | Easy — well within range |
| Qwen2.5-14B | ~8.4GB | Already validated |

**RAM analysis for 32B:**
- Machine has 15GB total, ~11GB available at idle
- 32B Q4_K_M GGUF: ~16–18GB (needs split or careful context sizing)
- Context window: each token adds to KV cache. With 512–1024 context, KV cache could add 2–4GB
- Total potential demand: 18GB model + 4GB KV = 22GB vs 15GB → OOM likely
- However: lower context (256–512) and no other processes might squeeze it
- **Verdict: 32B is tight but not impossible. A feasibility canary is required before full attempt.**

**Expected sidecar size for 32B:**
- FFN_UP shape for 32B: likely ~{8192, 28672} = 235,929,600 elements per layer
- Per layer INT6 sidecar: header(16) + 8192×4(scales) + packed(~62MB) = ~63MB per layer
- With ~64 layers: ~4GB total sidecar set
- Sidecar generation time: linear in layers. ~9min for 40 layers (14B). 64 layers → ~14min.

**Expected native load risk:**
- Native load without sidecars: OOM likely at full context
- Must start with very small context (128–256) to confirm model loads at all
- If 32B loads at small context, scale up incrementally

**What a safe feasibility canary looks like:**
1. Download Qwen2.5-32B Q4_K_M GGUF (verify split-merge if needed)
2. Run native llama-cli with minimal context (n=32, ctx=256) — just to confirm load
3. Check memory footprint after load and after first token
4. If stable: run tiny canary generation (8 tokens)
5. If all stable: proceed to full sidecar generation plan
6. If OOM: document 32B as exceeding this machine, recommend 8B or 7B as alternative

**Information gain:** High. If 32B works at all on this machine, it directly expands the demonstrated CPU-only boundary. If it doesn't, we learn the exact memory limit.

**Alignment with CPU-inference goal:** Very high. This directly tests Matt's mini-goal: can a model that people assume is impractical run on this hardware?

**Engineering risk:** Medium. Sidecar generation for 64 layers will take longer. Schema should generalize (16-byte header works for all sizes). Loader fix from Phase 16J may need another update for 32B size.

**Hardware risk:** Medium. 32B may OOM. Need to start with small context canary.

**Likely next-step value:** Very high — either proves the edge moves to 32B or defines exactly where this machine's boundary is.

---

## Option 4 — Further INT6 Optimization

**What it means:** Try to improve the current CLI-level INT6 pipeline without backend integration.

**Assessment:** Phase 15H (LUT4x unpack) delivered modest gains (~32ms). Phase 15G (mmap) reduced I/O overhead. Phase 15F (manifest cache) removed SHA recomputation. CLI-level optimization is largely exhausted.

**Remaining CLI-level opportunities:**
- Parallel sidecar loading (load multiple layers concurrently) — but CPU-bound, likely minimal gain
- Reduced SHA verification overhead — already cached in Phase 15F
- Pre-unpack scales at load time — already done via mmap

**Wall/setup overhead (~1.10–1.15×) is memory-bandwidth-bound.** CLI-level optimization cannot fix a memory-bandwidth bottleneck. Only kernel-level changes can address this.

**Conclusion:** Further CLI-level optimization is low-value. The remaining gain is in backend integration, not CLI polishing.

**Alignment with CPU-inference goal:** Low. Doesn't expand what's possible.

**Likely next-step value:** Very low. Not recommended as a standalone phase.

---

## Option 5 — Public/Research Packaging

**What it means:** Post the Phase 16O package (X thread, article, FAQ) to get early feedback before more experiments.

**Assessment:** Phase 16O is already created and committed. The question is whether to publish/post it now.

**Upside:**
- Early feedback from the research community
- Potential collaboration or correction from peers
- Starts building public track record

**Risk:**
- Evidence base still limited (8 prompts, single machine, no repeatability suite)
- Public claims that get challenged without repeatability data could undermine credibility
- Posting before repeatability means critics can say "run it again and show me"

**Recommended approach:** Do the next experiment first (32B feasibility canary), then post with that result included. The 16O package can be posted when there's a stronger next result to attach to it.

**Alternative:** Post 16O now with explicit caveat "results from one run on one machine, repeatability testing in progress." This is honest and avoids overclaiming.

**Decision:** This is a parallel track, not a blocking step. Can happen independently of the experiment path.

**Alignment with CPU-inference goal:** Medium. Feedback is valuable but not critical to the research direction.

---

## Ranking Table

| Option | Information Gain | Engineering Risk | Hardware Risk | CPU-Goal Alignment | Next-Step Value | Rank |
|--------|-----------------|------------------|---------------|-------------------|-----------------|------|
| 3. Bigger Model (32B) | **High** | Medium | Medium | **Very High** | **Very High** | **1** |
| 1. Backend/ggml Integration | Medium | Medium-High | Low | High (long-term) | High | **2** |
| 2. 14B Repeatability | Low-Medium | Low | None | Low-Medium | Medium | **3** |
| 5. Public Posting | Low | Low | None | Medium | Low-Medium | **4** |
| 4. Further INT6 Optimization | Very Low | Low | None | Low | Very Low | **5** |

---

## Recommended Next Phase

**Phase 17B: 32B CPU Feasibility Canary**

### Reason
The highest-value experiment for Matt's stated goal is to test whether the CPU-only boundary extends beyond 14B. If 32B can run at all on this machine — even with reduced context — it directly serves the "make CPU inference more viable on hardware people assume is impractical" objective. We learn the boundary. Either it moves to 32B or we define exactly where this machine's limit is.

### 5–10 Bullet Plan

1. **Check disk and download 32B GGUF** — Confirm ~20GB available on NVMe, download Qwen2.5-32B Q4_K_M from HuggingFace (handle split-merge if needed)
2. **Run native tiny canary (ctx=256, n=32)** — Just confirm the model loads and generates without OOM. Check memory footprint immediately after load.
3. **If OOM:** Document 32B as exceeding this machine's capacity. Pivot to 8B as the next weight class (well within 15GB). Do NOT force larger model onto insufficient RAM.
4. **If stable:** Run native tiny generation — confirm "Paris" output matches expected model behavior
5. **Check FFN_UP tensor shape** — Extract via gguf API to confirm expected dimensions for loader update
6. **Update loader for 32B** — Add 32B to known-size list (same pattern as Phase 16J fix)
7. **Generate 32B INT6 sidecar canary (2–4 layers)** — Validate sidecar format works for 32B before full generation
8. **If canary passes:** Generate full 64-layer INT6 sidecar set (expect ~14min)
9. **Run 32B PRT tiny canary** — Confirm 40+ sidecar layers load correctly
10. **Document 32B boundary result** — Whether it fits or fails, record the finding

### Pass/Fail Criteria

**Pass:** 32B Q4_K_M loads in native mode at ctx=256, generates without OOM, memory footprint understood, sidecar canary (2–4 layers) produces correct matvec output.

**Conditional pass:** 32B loads but OOMs at context >512. Document the context limit. Still a meaningful result — we know the boundary.

**Fail:** 32B OOMs at load or during first token generation at minimal context. Document as exceeding this machine. Recommend alternative model size.

### Not a blocker if 32B fails
If 32B doesn't fit, the backup path (8B) still tests the next-weight-class question, just at a smaller step.

---

## Backup Path

**If 32B OOMs or is otherwise blocked:**

**Phase 17C: Next Weight Class — 8B Validation**

Instead of 32B, test the 7B→14B step pattern at the next size: Qwen2.5-8B or Llama-3.1-8B (both ~4–5GB GGUF, well within 15GB RAM).

**Why 8B as backup:**
- Easy load on current hardware (plenty of RAM headroom)
- Tests whether PRT generalizes across model families (Llama vs Qwen architecture)
- Sidecar generation fast (~4–5 min for ~40 layers)
- Result adds diversity to the validation chain (now covers Qwen 0.5B/3B/7B/14B and potentially Llama 8B)

**Even if 8B, the question being answered is the same:** does the next weight class work on this CPU-only setup?

---

## What Not To Do Next

- **Do not attempt 70B or larger** — Requires more RAM than this machine has and more disk storage than is reasonable for this project
- **Do not spend more time on CLI-level INT6 optimization** — Phase 15H largely exhausted the easy wins; remaining gains are in the kernel, not the CLI
- **Do not post public claims without repeatability** — The 8-prompt suite is solid but critics will ask "run it again." Wait for at least one repeatability confirmation before public posting
- **Do not force 32B if it OOMs** — If the feasibility canary shows 32B doesn't fit, pivot to 8B. Forcing it wastes time and risks machine instability
- **Do not touch existing frozen tags** — Phase 16M is locked. Don't modify it to try new experiments

---

## Final Recommendation to Matt

> Given the current evidence, I recommend going to 32B next because the highest-value experiment for your stated goal — proving CPU inference works on hardware people assume is impractical — is to test whether the boundary extends beyond 14B. The 14B result is strong and complete. The next question is whether it generalizes to larger models, and 32B is the natural next step. Run the feasibility canary first (native load at small context), and if it doesn't fit, pivot to 8B as the backup. This directly serves your mini-goal of expanding what's possible on CPU-only hardware.

---

## Safety Scan Results

(to be executed before commit — see Phase 17A-K)

**Expected state:** Only `.md` and `.json` files staged from `results/`. No `.gguf`, `.bin`, `.safetensors`, `.pt`, `.pth` files. No private paths. No secrets.

---

*Phase 17A | Strategy decision | Branch: experimental/prt-phase14a-packed-sidecars*