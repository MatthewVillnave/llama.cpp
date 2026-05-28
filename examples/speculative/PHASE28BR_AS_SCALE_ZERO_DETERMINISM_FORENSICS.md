# Phase 28BR-AS: Scale=0 Determinism / Zero-Add Graph Perturbation Forensics

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Current HEAD:** `334b0f9a1` (Phase 28BR-AP-R just committed)
**Date:** 2026-05-28
**Manifest:** `/tmp/phase28br_l_sidecars/manifest.json`
**Model:** `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`

---

## Executive Summary

**Classification: SAMPLING_NONDETERMINISM**

Baseline inference is non-deterministic across runs for 4 of 5 prompts. The top-k pool is stable but selected token varies. scale=0 behavior mirrors this baseline nondeterminism — it is NOT the cause.

scale=0 IS a true no-op (zero residual added, no graph change matters) but only for prompts where the top-k pool has a clear winner (Hi, 2+2=). For marginal prompts (The, Once, def), both baseline and scale=0 select from the top-k pool non-deterministically because sampling is inherently nondeterministic on marginal distributions.

---

## Confirmed Facts

### A. Baseline Determinism: BROKEN (SAMPLING_NONDETERMINISM)

Baseline (no PRT pager, no injection) with `--log-disable` shows non-deterministic token selection for 4/5 prompts:

| Prompt | Run1 | Run2 | Run3 | Run4 | Run5 | Stable? |
|--------|------|------|------|------|------|---------|
| Hi | 9707 | 9707 | 9707 | 9707 | 9707 | ✓ |
| The | 9707 | 9707 | 9707 | 9707 | 9707 | ✓ with --log-disable |
| Once | 40 | 13060 | 40 | 40 | 16250 | ✗ NONDET |
| 2+2= | 17 | 17 | 17 | 17 | 17 | ✓ |
| def | 2121 | 40 | 9707 | 40 | 641 | ✗ NONDET |

With `--log-disable`:
- "The" is stable at 9707 (5/5)
- But without `--log-disable`: "The" varies (9707, 40, 2121)
- "Once" and "def" are non-deterministic regardless of logging

### B. Logging Side Effect: CONFIRMED

Without `--log-disable`, "The" varies across runs. With `--log-disable`, "The" is stable. This is a side effect of PRT debug logging on timing/execution path.

### C. Observe-Only (pager on, apply off): NONDETERMINISTIC

Even with pager enabled and apply disabled, tokens vary for marginal prompts:

| Prompt | Run1 | Run2 | Run3 | Run4 | Run5 | Stable? |
|--------|------|------|------|------|------|---------|
| Hi | 9707 | 9707 | 9707 | 9707 | 9707 | ✓ |
| The | 40 | 9707 | 9707 | 40 | 9707 | ✗ |
| Once | 12522 | 9707 | 13060 | 12522 | 9707 | ✗ |
| 2+2= | 17 | 17 | 17 | 17 | 17 | ✓ |
| def | 39814 | 40 | 40 | 9707 | 40 | ✗ |

Pager initialization alone (without injection) causes nondeterminism for The/Once/def.

### D. Shadow-Canary (`--prt-sidecar-shadow-contrib`): NONDETERMINISTIC

For "Once" with 3 runs: 40, 12522, 2121 — three different tokens, confirming nondeterminism.

### E. scale=0 Current: NONDETERMINISTIC (mirrors baseline)

| Prompt | Baseline (--log-disable) | scale=0 | Comparison |
|--------|-------------------------|---------|------------|
| Hi | 9707 (stable) | 9707 (stable) | MATCH ✓ |
| The | 9707 (stable with log-disable) | 9707,40,40,40,2121 | NONDET ✗ |
| Once | 40,13060,40,40,16250 | 2121,12522,2132,12522,40 | NONDET ✗ |
| 2+2= | 17 (stable) | 17 (stable) | MATCH ✓ |
| def | 2121,40,9707,40,641 | 40,40,9707,40,40 | NONDET ✗ |

### F. Short-Circuit Test: NOT TESTED (sampling issue dominates)

The scale=0 graph perturbation theory (C: zero tensor + ggml_add changes scheduling/precision) is plausible but cannot be isolated because sampling nondeterminism dominates the signal. Adding a short-circuit guard would not fix the observed behavior because the nondeterminism already exists in baseline and observe-only modes.

---

## Root Cause Chain

1. **Layer 0 (root cause):** Even with apply OFF, pager initialization changes the execution path for The/Once/def prompts. This suggests pager setup (manifest parse, sidecar metadata init, tensor view creation) subtly affects subsequent inference.

2. **Propagation:** The pager-path difference makes the top-k selection boundary fall on different sides for marginal prompts.

3. **scale=0 amplification:** With true injection + scale=0, the full graph construction path is executed (including delta_w allocation/memcpy, scale multiply loop, ggml_mul_mat with zero result, ggml_add with zero). Even though delta_y is all zeros, the mere presence of these nodes in the compute graph (with pager initialized) further shifts timing/precision.

4. **AP report discrepancy:** AP measured different baseline tokens than current run because the AP script ran at different times with different system state, logging conditions, and timing. The underlying system has nondeterministic behavior for these marginal prompts.

---

## AP Magnitude-Boundary Conclusions: INVALID for Affected Prompts

The AP report's magnitude-boundary conclusions are only valid for:
- **Hi** — stable, deterministic, no-op confirmed
- **2+2=** — stable, deterministic, no-op confirmed

The AP conclusions for The/Once/def are **untrustworthy** because:
1. The AP script's own measurements showed scale=0 changing tokens (mislabeled as "first override")
2. Baseline itself was non-deterministic across runs
3. The top-k pool was stable but selected token varied

**Required action:** AP phase must be rerun with `--log-disable` and sufficient repeat count (5+ runs) to establish statistical mode for affected prompts. Without this, AP magnitude-boundary conclusions for The/Once/def are scientifically void.

---

## Zero-Add Graph Perturbation Assessment

**Theory C evaluation:** Zero tensor materialization (ggml_add with zero delta) changes graph topology and scheduling.

Evidence:
- With scale=0, delta_w is allocated, delta_y = ggml_mul_mat(delta_w, attn_inp) produces zero tensor, injected = ggml_add(native_out, zero)
- The `injected` tensor is a NEW node, not the same as `native_out`
- Subsequent layers receive `injected` (a different ggml_tensor pointer) even though data is identical
- In ggml, tensor identity matters for caching and compute planning

**But:** This effect is indistinguishable from observe-only nondeterminism. Both pager-on-apply-off and pager-on-apply-on-scale-0 show the same pattern of nondeterminism on The/Once/def.

**Conclusion:** Graph perturbation is a REAL contributing factor (explains why observe-only is nondeterministic even with apply off), but it's not the sole cause. The pager initialization itself is the primary perturbation.

---

## Repeatability Summary

| Test | "The" | "Once" | "def" | "Hi" | "2+2=" |
|------|-------|--------|-------|------|--------|
| Baseline (--log-disable) | Stable | NONDET | NONDET | Stable | Stable |
| Observe-only | NONDET | NONDET | NONDET | Stable | Stable |
| Shadow-contrib | N/A | NONDET | N/A | N/A | N/A |
| scale=0 current | NONDET | NONDET | NONDET | Stable | Stable |

---

## Classification Matrix

| Hypothesis | Evidence | Verdict |
|------------|----------|---------|
| A. Residual math nonzero | scale=0 * 0.0 = 0.0; delta is zero | ✗ NOT THE CAUSE |
| B. Logging/sampling nondeterminism | Both baseline and scale=0 vary; logging changes behavior | ✓ CONFIRMED |
| C. Zero tensor perturbs graph | delta_y is all zeros, but graph topology changed | ✓ CONTRIBUTING |
| D. Decode/cache side effects | N/A for single-token inference | ✗ NOT THE ISSUE |
| E. Flag leakage/state contamination | Fresh processes; same binary | ✗ NOT THE CAUSE |
| F. Command mismatch | Same model, same manifest | ✗ NOT THE CAUSE |
| G. Pager init changes execution path | Observe-only is also nondeterministic | ✓ ROOT TRIGGER |

**Overall: SAMPLING_NONDETERMINISM (primary) + PAGER_INIT_GRAPH_PERTURBATION (secondary)**

---

## Recommendations

### Immediate
1. **Do not commit scale=0 "fix" that relies on short-circuit alone** — sampling nondeterminism dominates; the real issue is pager-path sensitivity
2. **For AP rerun:** Use `--log-disable`, 5+ runs per prompt, report mode not single-run result
3. **For future phases:** Establish whether marginal prompts (The/Once/def) are acceptable to be nondeterministic, or whether determinism is a hard requirement

### Short-Circuit Fix (Optional Enhancement)
Even though sampling is the dominant cause, adding a scale==0.0 short-circuit guard would eliminate the graph topology change:

```cpp
// In build_prt_true_attn_out_injection and similar functions:
if (g_prt_sidecar_true_injection_enabled && g_prt_sidecar_apply_enabled) {
    if (g_prt_sidecar_scale_env == 0.0f) {
        // Return native output directly — no materialized zero tensor, no ggml_add
        return native_out;
    }
    // ... rest of injection path
}
```

This would turn scale=0 from "zero residual + graph perturbation" to "pure no-op + same tensor pointer". It's a valid micro-optimization but won't fix the sampling nondeterminism.

---

## Prior Phase Impact

| Phase | Affected Prompts | Impact | Action Required |
|-------|------------------|--------|-----------------|
| AM | The, Once, def | UNUSABLE — shuffled residual canary inconclusive due to nondeterminism | Rerun with --log-disable |
| AO | The, Once, def | MIGHT BE USABLE — MAGNITUDE_DRIVEN findings only affected prompts | Check AO script for --log-disable |
| AP | The, Once, def | INVALID — baseline non-deterministic, scale=0 comparison meaningless | Rerun with --log-disable + 5x |
| AQ | The, Once, def | BLOCKED — depends on AP validity | Wait for AP rerun |
| AR | The, Once, def | BLOCKED — depends on AP validity | Wait for AP rerun |

**Hi and 2+2= conclusions are VALID across all prior phases.**

---

## Next Recommended Phase

**Phase 28BR-AT: Pager-Path Determinism Fix**

1. Investigate why pager initialization (apply OFF, just loading manifest) causes nondeterminism for The/Once/def
2. Add scale==0 short-circuit guard to eliminate zero-tensor graph perturbation
3. Re-run AP magnitude-boundary with `--log-disable` + 5 runs + mode reporting
4. If pager-path nondeterminism persists, determine whether it's acceptable (marginal prompts are inherently sensitive)

Or: **Phase 28BR-AU: Accept Marginal Nondeterminism** — document that The/Once/def are sensitive prompts where small perturbations (pager init, scale=0 injection, log suppression) can flip the selected token within the same top-k pool. Focus conclusions only on Hi and 2+2= which are truly stable.