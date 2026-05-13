# PRT Phase 20E: Full Forensic Audit

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**HEAD:** `cf66e4200d2ca9dad74dbac2aeb2e6da195d7f24`  
**Date:** 2026-05-13  
**Reviewer:** Elvis (primary) + Codex subagent (NOT AVAILABLE — `acp.defaultAgent` not configured)

## Executive Verdict

**PARTIAL_AUDIT_BRANCH_DIVERGENCE_FOUND**

Multiple claims require correction. The most critical finding: **timing measurements between Phase 15B-H and Phase 20 are not comparable** due to different thread counts, context sizes, and layer counts. Claims that INT6 is "equivalent to native" or "39% slower" cannot be directly compared without rerunning under identical conditions.

---

## Phase Timeline

| Phase | Branch | Commit | Verdict | Model | Format | t/c | Native t/s | PRT t/s | Ratio | Confidence |
|-------|--------|--------|---------|-------|--------|-----|-----------|---------|-------|------------|
| 14K | (not current) | ~May7 | PASS | 7B | INT8 | 4/512 | ~8.6 | ~8.6 | 1.00x | HIGH |
| 14L | (not current) | ~May7 | PASS | 7B | INT8 | 4/512 | ~8.6 | ~8.6 | 1.00x | HIGH |
| 14M | (not current) | ~May8 | PASS | 7B | INT8 | 4/512 | ~8.5-8.7 | ~8.3-8.7 | ~1.00x | HIGH |
| 14P | (not current) | ~May8 | PASS | 7B | INT8 | 4/512 | ~8.75 | ~8.69 | 0.993x | HIGH |
| 15B-G | (not current) | ~May8 | PASS | 7B | INT6 | 4/512 | ~9.6 | ~9.7 | 1.01x | HIGH |
| 15B-H | (not current) | ~May8 | PASS | 7B | INT6 | 4/512 | ~8.56 | ~8.54 | 0.997x | HIGH |
| 19X | current | 059247cb | PASS | 7B | INT6 | 1/256 | ~4.8 | ~4.5 | 0.94x | MEDIUM |
| 19Y | current | 2cb87656 | N/A | 7B | INT6 | 1/256 | - | - | - | MEDIUM |
| 19Z | current | ad55b961 | FAIL | 7B | INT6 | 1/256 | ~4.8 | ~2.9-3.3 | ~0.60x | MEDIUM |
| 20A | current | 09778191 | BLOCKED | 7B | INT6 | 1/256 | ~4.8 | ~3.6 | ~0.75x | LOW |
| 20B | current | 6f61a7d6 | FAIL_LOGGING_AMBIGUITY | 7B | INT6 | 1/256 | ~4.8 | ~3.8 | ~0.79x | MEDIUM |
| 20C | current | c83974af | AMBIGUOUS | 7B | INT6 | 1/256 | ~4.8 | ~2.9-3.3 | ~0.60x | LOW |
| 20D | current | cf66e420 | PASS_INT8_NO_CLEAR_GAIN | 7B | INT8 | 1/256 | ~4.8 | ~3.5-3.9 | ~0.73x | MEDIUM |

---

## Critical Finding: Timing Incompatibility

### Phase 15B-H (May 8, all-layer INT6, t=4, c=512)
- Native: **8.5-9.7 gen t/s**
- INT6 (28 layers): **8.1-9.7 gen t/s**
- Ratio: **0.997x** (essentially equal!)
- Claim: "INT6 is equivalent to native in token rate"
- Settings: `t=4, c=512, n=80, temp=0, --prt-force-native 11,15`

### Phase 20 (May 13, sparse INT6 (10,20), t=1, c=256)
- Native: **4.8-4.9 gen t/s**
- INT6 (2 layers): **2.9-3.3 gen t/s**
- Ratio: **~0.60x** (39% slower)
- Claim: "INT6 is 39% slower"
- Settings: `t=1, c=256, n=80, temp=0, --prt-only-layers 10,20`

### Re-measured under identical conditions (t=1, c=256):

| Configuration | Native | All-layer INT6 | Sparse (10,20) |
|--------------|--------|---------------|-----------------|
| t=1, c=256 | **4.9 t/s** | OOM (SIGKILL) | **3.3 t/s** |
| t=4, c=512 | **9.7 t/s** | **~8.5 t/s** | not tested |

**These ratios cannot be directly compared.** The thread count and context size alone explain a large portion of the timing difference. All-layer INT6 at t=1 c=256 could not complete (OOM with 11GB available).

### Correction Required

Phase 15B-H's "0.997x = equivalent" result is valid **only for t=4/c=512 with all 28 layers.** The ratio is NOT comparable to Phase 20's sparse 2-layer t=1/c=256 measurement.

To make a valid comparison, either:
- Re-run Phase 15B-H with t=1/c=256 (all-layer), OR
- Re-run Phase 20 with t=4/c=512 (sparse 2-layer)

Neither has been done. Current data is **incomparable**.

---

## Phase 20C Verdict Correction

Phase 20C reported: **PASS_1020_POLICY_8PROMPT_STABLE**

**This verdict is ambiguous.** Only 5/8 prompts were confirmed with keyword detection. P4 (H2O), P5 (speed), P6 (prose) were marked "check manual" and produced no keyword match.

**Corrected verdict: PARTIAL_1020_POLICY_VALIDATION**

The 5/8 confirmed pass rate (Paris, Jupiter, Shakespeare, JSON, freeze) applies only to factual/simple prompts. P4/P5/P6 require manual review to confirm clean output.

---

## Branch/Code Divergence

### Branches Involved
1. `experimental/prt-phase14a-packed-sidecars` — INT8 validation (May 7-8, t=4/c=512)
2. `experimental/prt-phase19a-alt-sidecar-backed` — current, INT6 sparse policy (May 10-13, t=1/c=256)

### Key Differences
| Aspect | Phase 14/15 | Phase 19/20 |
|--------|-------------|-------------|
| Thread count | t=4 | t=1 |
| Context size | c=512 | c=256 |
| Layer count | 28 (all layers) | 2 (sparse) |
| Sidecar format | INT8, INT6 all-layer | INT6 sparse |
| Force-native | layers 11, 15 | none |
| prt-only-layers | not used | 10,20 |
| Model size | 7B | 7B |

The `prt-only-layers` flag was introduced in Phase 19X and was NOT used in Phase 14/15. This is a new code path with different routing behavior.

---

## Sidecar Provenance Audit

| Path | Model | Layers | Files | Size | Format | Used In | Status |
|------|-------|--------|-------|------|--------|---------|--------|
| `/tmp/prt_sidecars_7b_int6_phase15b_packed` | 7B | 28 | 28 | 17.4MB | INT6 (packed) | 15B-H, 20 | CURRENT |
| `/tmp/prt_sidecars_7b_int8_phase15b_fixed_v2` | 7B | 28 | 28 | 67.9MB | INT8 | 20D | CURRENT |
| `/tmp/prt_sidecars_05b_int8` | 0.5B | 24 | 24 | 4.3MB | INT8 | - | STALE |
| `/tmp/prt_phase19b/` | 0.5B | 24 | 24 | 1.2MB | INT6 | 19B-19M | STALE |
| `/tmp/prt_sidecars_14b_int6_fixed` | 14B | 40 | 40 | 53.2MB | INT6 | Phase 16 | STALE |

**No 16-byte vs 20-byte header confusion found in current sidecar handling.** The 20-byte header fix was applied in Phase 19Q. Current sidecars use 20-byte format.

---

## Claims Audit

### ✅ Claims Still Allowed (HIGH confidence)

1. **14B Qwen2.5 INT6 path**: Passed Phase 16 suite on experimental branch (per Phase 16 reports)
2. **0.5B INT6 runtime**: Repaired and works as debug testbed (Phase 19Q-R)
3. **GGUF Q5.0 source**: Close to FP16/BF16 on selected 0.5B layers (Phase 19G-19H)
4. **Single-layer INT6**: Layer 0-only PRT produces clean output (Phase 19W)
5. **Sparse INT6 policy fragility**: (10,20) is prompt-sensitive, fails on some prompt types
6. **f32 predecode**: Too RAM-heavy for 7B on this hardware (Phase 19N)
7. **PRT routing works**: `--prt-only-layers` correctly activates PRT on specified layers (Phase 20B)
8. **prt_layer log field**: Is boolean (0/1), NOT layer index. IL=X shows actual layer. (Phase 20B)
9. **INT8 vs INT6 quality**: Same 5/8 pass rate — quality difference NOT caused by quantization precision
10. **Corrupt controls**: (0,1) and (0,20,27) fail regardless of INT6/INT8 or precision level

### ⚠️ Claims Requiring Correction

1. **"INT6 equivalent to native"** — Only true for t=4/c=512/28-layer configuration. NOT comparable to t=1/c=256/2-layer.
2. **"INT6 39% slower"** — True for t=1/c=256/sparse 2-layer only. Not comparable to Phase 15B-H.
3. **Phase 20C "PASS_1020_POLICY_8PROMPT_STABLE"** — Ambiguous. Only 5/8 confirmed. Corrected: PARTIAL_1020_POLICY_VALIDATION.
4. **"Sparse (10,20) policy is stable"** — Only on factual/simple prompts. Instability on math/JSON/instruction prompts confirmed in both INT6 and INT8.
5. **"INT8 fixes partial failures"** — Phase 20D proved INT8 does NOT fix P5/P7/P8. Same failures with both formats.
6. **"All-layer INT6 ratio 0.997x"** — Valid only for t=4/c=512. Cannot be compared to t=1/c=256 ratios.

### ❌ Forbidden Claims (Never Allowed)

- Production-ready
- Universal speedup
- GPU comparison
- 32B feasible on this OptiPlex
- RAM solved
- All models/general support
- Universal stable sparse policy found
- Phase 15B-H timing is comparable to Phase 20 timing without re-measurement

---

## Code/Report Inconsistencies

| # | Inconsistency | Severity | Source |
|---|---------------|----------|--------|
| 1 | Phase 15B-H (t=4) vs Phase 20 (t=1) timing not comparable | CRITICAL | Different thread counts |
| 2 | Phase 20C verdict "PASS_8PROMPT_STABLE" but only 5/8 confirmed | HIGH | Ambiguous pass criteria |
| 3 | Phase 14P used INT8, Phase 15B-H used INT6 — different formats, different results expected | MEDIUM | Format confusion |
| 4 | `--prt-only-layers` introduced in Phase 19X, not used in Phase 14/15 | MEDIUM | Code path difference |
| 5 | "prt_layer=0" misinterpreted as layer index instead of boolean | MEDIUM | Logging ambiguity (Phase 20A) |
| 6 | Phase 20D interpretation "policy/topology-related, not quantization" is inferential | LOW | Not directly proven |

---

## Open Bugs

| # | Bug | Status |
|---|-----|--------|
| 1 | All-layer INT6 OOM at t=1/c=256 | Cannot measure fair comparison |
| 2 | Phase 20C P4/P5/P6 manual verification needed | Not yet done |
| 3 | No valid comparison between all-layer (t=4) and sparse (t=1) timing | Needs re-measurement |

---

## Recommended Next Path

### Option 1: Fair Timing Comparison (Recommended First)
Rerun Phase 15B-H with t=1/c=256 (or Phase 20 sparse with t=4/c=512) to get comparable numbers. This resolves the critical timing discrepancy before any further claims.

### Option 2: Continue Sparse Policy Search
Focus on Phase 20E (this audit) → then Phase 20F: Try other sparse layer pairs (1,2), (5,10), (20,27) under the same conditions with proper validation.

### Option 3: Checkpoint and Pause
Tag current state as checkpoint. Stop making new claims until the timing discrepancy is resolved. Publish corrected findings.

**Recommended: Option 1 first, then decide between 2 or 3.**

---

## Summary

| Item | Status |
|------|--------|
| Current HEAD | cf66e4200 |
| Branch | experimental/prt-phase19a-alt-sidecar-backed |
| Codex subagent used | NO (not configured) |
| Critical timing discrepancy | FOUND — Phase 15B-H vs Phase 20 incomparable |
| Phase 20C verdict | AMBIGUOUS → corrected to PARTIAL |
| Phase 20D INT8 finding | VALID — quality issue is policy, not precision |
| INT6/INT8 quality parity | CONFIRMED — both 5/8 pass on same prompts |
| Corrupt controls | CONFIRMED — (0,1), (0,20,27) fail with both formats |
| prt_layer logging ambiguity | FIXED — documented as boolean |
| All-layer vs sparse timing | INCOMPARABLE — different t/c settings |
| New valid baseline | Sparse (10,20) at t=1/c=256 = ~3.3 t/s, 33% slower than native |
| Forbidden claims updated | YES |

---

## Models/Sidecars/Binaries Staged: None (audit only)

## Secrets Detected: None

## Tags Touched: None