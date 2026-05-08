# PRT Phase 15B-J — INT6 Experimental Checkpoint

## Checkpoint

| Field | Value |
|-------|-------|
| **Tag** | `PRT_PHASE15B_J_INT6_EXPERIMENTAL_CHECKPOINT` |
| **Branch** | `experimental/prt-phase14a-packed-sidecars` |
| **Base commit** | `59b1721ae` |
| **Date** | 2026-05-08 |
| **Verdict** | **PASS_INT6_EXPERIMENTAL_CHECKPOINT** |

---

## Context

After Phase 15B-A through 15B-E established fresh INT8 sidecars and tested INT4 (NO-GO) and INT6 (PASS) offline parity, the INT6 experimental path was validated through four sequential runtime tests:

- Phase 15B-G: Tiny runtime canary — confirmed the packed INT6 format loads and runs
- Phase 15B-H: 8-prompt validation — confirmed quality across diverse prompts
- Phase 15B-I: Longer-generation smoke — confirmed quality holds under extended output

**INT8 remains the validated runtime path.** INT6 is experimental.

---

## Frozen INT6 Validation Chain

### Phase 15B-F: INT6 Offline Parity
- **Verdict:** PASS_INT6_OFFLINE_PARITY
- **Commit:** `17a42a093`
- **Tool:** `phase15b_int6_offline_parity.cpp`
- **Selected layers tested:** 0, 1, 10, 11, 15, 20, 27
- **Minimum weight cosine:** 0.997873
- **Minimum matvec cosine:** 0.997419
- **Weakest layer:** 0
- **INT6 vs INT8 degradation:** ~0.001–0.002 WC
- **INT6 vs INT4 improvement:** +0.014–0.022 WC
- **Estimated packed size:** ~51 MB/layer vs INT8 ~68 MB/layer (25% reduction)

### Phase 15B-G: Packed INT6 Tiny Runtime Canary
- **Verdict:** PASS_INT6_TINY_RUNTIME_CANARY
- **Commit:** `f02dceb17`
- **Tool:** `phase15b_int6_packed_sidecar.cpp`
- **Format:** PRT6 magic + version/rows/cols + scales[float32, 18944] + packed payload (4 INT6→3 bytes, offset-32)
- **Packed sidecar dir:** `/tmp/prt_sidecars_7b_int6_phase15b_packed/`
- **Files:** 28, unique SHA: 28, total ~1.4 GB
- **Tiny canary:** native → "Paris.", INT6 → "Paris." ✅ exact match
- **Runtime:** 28/28 sidecars loaded, format=int6 confirmed, native 4.238s, INT6 6.647s (timing diagnostic)

### Phase 15B-H: Packed INT6 8-Prompt Validation
- **Verdict:** PASS_INT6_8PROMPT_QUALITY
- **Commit:** `79efcdf8e`
- **Prompts tested:** 8 (factual, prose, code, JSON, reasoning, technical, summarization, edge)
- **Native completed:** 8/8
- **INT6 completed:** 8/8
- **Exact matches:** 8/8
- **Quality degradations:** 0
- **Collapse/repetition:** 0
- **Native avg tok/s:** 8.562, median 8.400
- **INT6 avg tok/s:** 8.537, median 8.400
- **INT6/Native ratio:** 0.997× avg, 1.000× median (essentially 1:1)
- **Wall overhead:** ~1.19× (diagnostic only, unpack/setup overhead)
- **28/28 sidecars loaded:** 8/8 runs
- **Format=int6 logged:** 8/8 runs

### Phase 15B-I: Packed INT6 Longer-Generation Smoke
- **Verdict:** PASS_INT6_LONGER_GEN_SMOKE
- **Commit:** `59b1721ae`
- **Test A (n=320 prose):** Native 8.30 tok/s, INT6 8.20 tok/s, ratio 0.988×, EXACT match
- **Test B (n=160 factual):** Native 8.40 tok/s, INT6 8.20 tok/s, ratio 0.976×, EXACT match
- **Quality degradations:** 0
- **Repetition/collapse:** 0
- **28/28 sidecars loaded:** 2/2 tests
- **Format=int6 confirmed:** 2/2 tests
- **Average INT6/Native tok/s ratio:** 0.982×
- **Wall overhead:** ~5–11% (diagnostic only)

---

## Key Results Summary

| Metric | Value |
|--------|-------|
| Offline min WC | 0.997873 |
| Offline min MC | 0.997419 |
| 8-prompt exact matches | 8/8 |
| Longer-gen exact matches | 2/2 |
| Total quality degradations | 0 |
| Total collapse/repetition | 0 |
| Sidecars loaded | 28/28 (all runs) |
| Format confirmed | int6 (all runs) |
| Fallback layers | 11, 15 (forced native) |
| Avg tok/s ratio (INT6/Nat) | 0.990× |
| Wall overhead | ~5–19% (diagnostic, unpack/setup) |
| Packed size reduction vs INT8 | ~25% (51 MB vs 68 MB/layer) |

---

## Allowed Claims

> **Packed INT6 PRT passed offline parity, a tiny runtime canary, an 8-prompt validation, and longer-generation smoke tests on Qwen2.5-7B in Matt's measured CPU setup, with near-native generation token rate but measurable wall-time unpack overhead.**

- ✅ Packed INT6 passed all four runtime validation phases
- ✅ INT6 showed near-native generation token rate (0.990× avg across all tests)
- ✅ INT6 showed 25% size reduction vs INT8 at offline level
- ✅ INT6 remains experimental; INT8 remains the validated runtime path

## Forbidden Claims

- ❌ Production readiness
- ❌ Universal speedup demonstrated
- ❌ INT6 replaces INT8
- ❌ Full deployment readiness
- ❌ All-long-context stability (tests were c=512–1024, not 4096+)
- ❌ Larger-than-7B support validated
- ❌ GPU comparison or cross-platform speed claims
- ❌ Speedup beyond measured diagnostics
- ❌ Broad quality equivalence beyond tested prompts

## Known Caveats

- **Wall-time overhead** from unpack/setup remains visible (~5–19% in measured setup). No speedup claim is made.
- **INT6 is not optimized.** Unpack path adds startup overhead that does not appear in generation tok/s but does affect wall time.
- **Sidecars are not public/staged.** Stored only in `/tmp/` for this experimental phase.
- **Tests limited to Matt's measured CPU setup** (TheForgeHQ, 15GB RAM, Qwen2.5-7B Q4_K_M).
- **No larger-than-7B validation.** 0.5B and 3B not tested with INT6.
- **INT8 remains the safer validated path.** INT6 is experimental and not recommended for production use until further validation.
- **Layer 14 anomaly:** GGUF finiteness anomaly detected (67439616/67895296 finite), WC=-nan for layer 14. Separate from format validation.

## Recommended Next Phase

**Phase 15C: Runtime sidecar SHA/provenance logging**, then **INT6 timing repeatability**.

This adds provenance to the sidecar loading path (log which SHA was loaded, when, from where) and then checks whether the 0.990× tok/s ratio holds consistently across repeated runs of the same prompts.

Alternative paths:
- Phase 15C: INT6 repeatability timing (5-repeat benchmark)
- Phase 15C: optimize INT6 unpack/dequant path
- Phase 15C: backend/ggml integration design
- Phase 15C: public/internal writeup update

---

## Tag Information

```
Tag: PRT_PHASE15B_J_INT6_EXPERIMENTAL_CHECKPOINT
Commit: 59b1721ae
Message: PRT Phase 15B-J: freeze INT6 experimental checkpoint
```

## Related Files

| Phase | File |
|-------|------|
| 15B-F | `examples/speculative/results/PRT_PHASE15B_F_INT6_OFFLINE_PARITY.md` |
| 15B-H | `examples/speculative/results/PRT_PHASE15B_H_INT6_8PROMPT_VALIDATION.md` |
| 15B-I | `examples/speculative/results/PRT_PHASE15B_I_INT6_LONGER_GENERATION_SMOKE.md` |
| 15B-J | `examples/speculative/results/PRT_PHASE15B_J_INT6_EXPERIMENTAL_CHECKPOINT.md` (this file) |

## Previous Checkpoints

| Tag | Description |
|-----|-------------|
| `PRT_PHASE14F_3B_INT8_CHECKPOINT` | 3B INT8 sidecar validation |
| `PRT_PHASE14G_3B_INT8_REPEATABILITY_CHECKPOINT` | 3B INT8 repeatability |
| `PRT_PHASE14N_7B_INT8_CHECKPOINT` | 7B INT8 checkpoint |
| `PRT_PHASE14Q_7B_INT8_FULL_VALIDATION_CHECKPOINT` | 7B INT8 full validation |
| `PRT_PHASE15A_7B_INT8_LONG_CONTEXT_STABILITY_CHECKPOINT` | 7B INT8 long-context |
| `PRT_PHASE15B_J_INT6_EXPERIMENTAL_CHECKPOINT` | INT6 experimental (this tag) |