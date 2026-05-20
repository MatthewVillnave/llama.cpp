# PRT Phase 24S: Architecture Decision Checkpoint

## Status: COMPLETE ✅

**Date:** 2026-05-20
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Previous HEAD:** `ce2366a73` (Phase 24Q)
**New HEAD:** `TBD` (docs-only commit)
**Scope:** Architecture decision documentation — no new code, no new timing runs

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD (Phase 24Q)
`ce2366a73`

## C. New HEAD
`TBD` — docs-only commit. Source changes from Phase 24R probe remain unstaged/local.

## D. Current Validated State

### 3B (Qwen2.5-3B-Instruct-Q4_K_M)
- **Canonical INT8 layer0:** ✅ Works. Sidecar: `prt_sidecars_3b_int8_phase24f`
- **Repeat stability:** ✅ Passed (multiple runs, output consistent)
- **Output correctness:** ✅ "Paris..." generation confirmed
- **Timing (c=4, n=8, layer0-only):**
  - Native: **2.30–2.39s** avg
  - PRT INT8 (custom op): **3.47–3.52s** avg
  - Slowdown: **+1.08–1.19s** (~47% slower in this narrow smoke test)
- **Known overhead:** ~470ms custom-op graph/scheduling + ~610ms kernel compute

### 7B (Qwen2.5-7B-Instruct-Q4_K_M)
- **Canonical INT8 layer0:** ✅ Works after sidecar regeneration (Phase 24G canonical)
- **Output correctness:** ✅ "Parisian..." / "Paris..." generation confirmed
- **Timing:** ❌ Blocked — 7B INT8 sidecar decode takes ~5s+ (memory constraints)
- **Multi-layer:** ❌ Not tested in Phase 24S scope

### 0.5B (Qwen2.5-0.5B-Instruct-Q4_K_M)
- **INT6 fallback:** ✅ Operational
- **Canonical INT8 layer0:** ⚠️ Partial — explicit 0.5B canonical INT8 pending
- **Status:** 0.5B uses INT6 path as working alternative

### Canonical INT8 Layout (Validated)
```
Position 0..K-1:        q[0], q[1], ..., q[K-1]       (quantized int8 values)
Position K..K+M-1:      scale[0], scale[1], ..., scale[M-1]  (float32 scales)
Total: K + M bytes

Decoded formula:
  W[k,j] = q[k*M + j] * scale[j]    for k in [0,K), j in [0,M)

Kernel formula (GGML custom op):
  Y[j,n] = sum_k X[k,n] * W[k,j]    (k in [0,K), j in [0,M), n in [0,N])
```

- Scales are **first**, int8 data is **second** — same as Phase 24H canonical
- Sidecar is decoded once, cached in f32 for the session
- This layout is designed for the custom AVX2 kernel, not native `ggml_mul_mat`

---

## E. What Phase 24Q Proved

**The slowdown is real and understood:**

| Component | Time | Proven? |
|-----------|------|---------|
| Marker spam | ~0ms | ✅ Phase 24O proved spam removed, timing unchanged |
| Repeated sidecar decode | ~0ms after first | ✅ Phase 24P counters show `decode=1` only |
| Selector calls (540×) | ~0μs | ✅ Phase 24P counters: `selector_us=0` |
| ENV/compat gate | ~10ms | ✅ Mode B vs Mode A: only ~10ms difference |
| **Custom-op graph/scheduling** | **~470ms** | ✅ Mode E (stub) − Mode A |
| **AVX2 kernel compute** | **~610ms** | ✅ Mode D − Mode E (8 calls × ~82ms) |
| INT8 decode loop | ~129ms once | ✅ Phase 24P counters: `decode_us=129098` |
| Sidecar file read | ~7ms once | ✅ Phase 24P counters: `read_us=7013` |

**The 470ms custom-op overhead is pure GGML infrastructure** — graph node creation, scheduling, dispatch per call. Even the stub kernel (which only writes zeros) adds 470ms over native.

**The 610ms kernel compute is correct** — 8 sequential decode calls × ~82ms each, running autoregressively.

**Phase 24Q did NOT prove:** That the custom-op path can be made faster without architectural change. It proved WHERE the time goes, not how to eliminate it.

---

## F. What Phase 24R Proved

**Native `ggml_mul_mat` path is incompatible with current sidecar layout:**

1. `ggml_mul_mat` requires weights in `[M,K]` column-major format
2. Current canonical sidecar decodes to `[K,M]` row-major
3. Three approaches tried (all failed):
   - **Transpose + call:** `W_T[M,K]` has `ne[0]=M ≠ K` (cur's `ne[0]`) → GGML assertion fails
   - **Reshape to 4D:** `ggml_can_mul_mat` passes, but `ggml_reshape_2d` on result fails → `ne[0]*ne[1]` mismatch
   - **Transpose view + contiguous copy:** `ggml_is_transposed` check fails on result

4. **Root cause:** The custom AVX2 kernel handles the `[K,M]→[M,K]` transpose explicitly as part of decode+compute. Native `ggml_mul_mat` cannot absorb this layout without:
   - A physical 90MB transposition per run (erases efficiency gain), OR
   - A separate sidecar generated in `[M,K]` layout (requires re-extraction)

5. **Phase 24R also confirmed:** Mode A (native) and Mode D (PRT) still work correctly after all the probe attempts.

---

## G. Closed Paths

The following are **closed** unless new information emerges:

| Path | Why Closed |
|------|-----------|
| **Speedup from current custom-op path** | 3B layer0 is ~47% slower than native. Overhead is structural. |
| **More timing before architecture changes** | Timing is clean. The slowdown is understood. More runs won't change the architecture. |
| **Batching normal autoregressive decode tokens** | Each token depends on the previous. Batch of 1 is correct. Speculative decoding is the only batchable case. |
| **Repeated native-mulmat attempts with current `[K,M]` layout** | Phase 24R proved layout incompatibility. Would need a `[M,K]` sidecar to proceed. |
| **7B timing on current hardware** | 7B decode takes ~5s+ for the INT8 step alone. Not viable without 15GB+ RAM or model changes. |
| **0.5B canonical INT8 explicit canonical** | INT6 path works. Not worth the effort for 0.5B. |
| **14B support** | Out of scope for PRT phase series. |

---

## H. Plausible Future Paths

These remain as **options**, not commitments:

### A. Native-Layout Sidecar Experiment
- Generate a separate sidecar in GGML-compatible `[M,K]` layout
- Decode directly into a tensor ready for `ggml_mul_mat` without per-run transpose
- **Risk:** Decoded f32 weights (90MB for 3B) are larger than native Q4_K (~45MB). May not speed up.
- **Feasibility:** Requires re-extraction tooling + validation

### B. True GGML Backend Op
- Implement a real `ggml_backend` kernel instead of custom op wrapper
- Goal: eliminate custom-op scheduling tax (~470ms)
- **Risk:** High complexity. The kernel itself is already fast (~82ms/call).
- **Feasibility:** Unknown without deep GGML backend work

### C. Keep PRT as Representation/Compression Path
- PRT is useful as canonical INT8 sidecar format for research
- Not currently a speed path for inference
- **Status:** Valid for correctness, compression, sidecar exchange

### D. Speculative Decoding Integration
- Batching only makes sense for multiple **accepted** tokens (speculative decoding)
- Normal greedy decode has no batch opportunity
- **Status:** Not relevant for current PRT scope

### E. Multi-Layer PRT (After Overhead Model Fixed)
- Current layer0-only is already slower than native
- Adding more layers without fixing custom-op overhead would worsen runtime
- **Status:** Do not proceed until per-layer overhead is resolved

### F. Accept + Document as Is
- PRT INT8 overhead (~1.08s for 3B, c=4, n=8) is known
- For batch/speculative scenarios the overhead profile may differ
- Document the ceiling and stop unless there is a specific reason to continue

---

## I. Claims Boundary

### Allowed Claims ✅
- Canonical INT8 sidecar layout validated for 3B and 7B layer0 canaries
- 3B repeat stability passed (multiple runs)
- Current custom-op path is **correct** but **slower** in narrow timing smoke
- Overhead source identified (custom-op graph + kernel compute)
- Native-mulmat path blocked by `[K,M]` vs `[M,K]` layout mismatch
- INT8 decode is a one-time ~129ms cost per session
- PRT sidecar enables canonical representation and compression

### Forbidden Claims ❌
- **Speedup claim** — PRT is slower in current narrow smoke test
- **Performance advantage claim** — native is faster
- **Production readiness** — 3B layer0-only smoke, not full model
- **Multi-layer support** — not tested
- **14B support** — out of scope
- **Broad semantic equivalence** — only "Paris..." output checked
- **7B timing result** — blocked
- **0.5B canonical INT8 pass** — INT6 works, explicit INT8 not done
- **Multi-batch advantage** — not tested

---

## J. Recommended Next

**Recommendation: Option 1 (Conservative) — Phase 24T paper/probe design only.**

Rationale: The Phase 24Q/24R investigation is complete. The overhead is understood. The layout blocker is confirmed. Before spending more runtime on implementation, write a clear paper/probe design for the native-layout sidecar experiment (Option A) and get explicit approval on next steps.

**If Matt wants to proceed with implementation: Option 2 — Phase 24T native-layout sidecar feasibility.**

Generate a 3B `[M,K]` layout sidecar and test direct `ggml_mul_mat` compatibility. This is the only remaining viable path from Phase 24R. But this is a non-trivial re-extraction + validation effort.

**If Matt wants to stop: Option 3 — Preserve checkpoint and pause PRT speed work.**

Document the canonical INT8 sidecar as a research artifact. PRT overhead (~1.08s for 3B) is known. Do not claim speedup. Continue only if a specific use case emerges.

---

## K. Safety Scan

```
git status:                           Clean (docs only, no binary artifacts staged)
Large files:                         Build artifacts + models (not staged, not in repo)
Secrets:                             None detected (grep hits are checklist template)
Tags:                                None touched in Phase 24S
```

## L. Files in This Commit

```
A examples/speculative/results/PRT_PHASE24S_ARCHITECTURE_DECISION_CHECKPOINT.md
A examples/speculative/results/phase24s_architecture_decision_checkpoint.json
```

## M. Phase Series Summary

| Phase | Subject | Key Result |
|-------|---------|------------|
| 24H | Canonical INT8 layout | ✅ Canonical format defined |
| 24K | State forensics | ✅ State machine understood |
| 24L/24O | Timing cleanup | ✅ Clean baseline timing |
| 24P | Overhead isolation | ✅ ~1.19s overhead identified |
| 24Q | Custom-op vs kernel | ✅ ~470ms graph + ~610ms kernel |
| 24R | Native matmul probe | ❌ Blocked by GGML layout |
| **24S** | **Architecture decision** | **✅ Decision checkpoint** |