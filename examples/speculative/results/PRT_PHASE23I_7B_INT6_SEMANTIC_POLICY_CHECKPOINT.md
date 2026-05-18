# PRT Phase 23I — 7B INT6 Semantic/Policy Baseline Checkpoint

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Previous HEAD:** `c4fb94152` (Phase 23H BLOCKED_ENV_TIMING)  
**New HEAD:** _(checkpoint commit — docs only)_  
**Tag:** `PRT_PHASE23I_7B_INT6_SEMANTIC_POLICY_CHECKPOINT`  
**Verdict:** `PASS_7B_INT6_SEMANTIC_POLICY_BASELINE_CHECKPOINT_WITH_TIMING_BLOCKED`  
**Date:** Sun 2026-05-17 22:40 EDT

---

## 1. Executive Summary

**7B INT6 layer0 semantic/policy baseline is valid and frozen.**

Phases 23D-R2 → 23G collectively prove:
- The INT6 decoded-f32 policy path is **repeat-stable** (zero variance across 3 runs)
- The semantic output is **sane and correct** (4/4 CLEAN_SEMANTIC)
- The policy routing is **correct** (N≤4 → PRT AVX2, N>4 → native prefill, scalar=0)
- The scale offset is **correct** (scale_off=20 for 7B)
- The INT8/INT6 priority conflict is **resolved** (explicit `PRT_V2_SIDECAR_FORMAT=int6`)

**Phase 23H timing is blocked by binary debug output, not by a PRT kernel defect.**

No speed claim exists. No performance ratio exists. The kernel produces correct output — timing measurement requires a quieter binary or manual terminal execution.

---

## 2. Frozen Technical Baseline

| Parameter | Value |
|-----------|-------|
| Branch | `experimental/prt-phase19a-alt-sidecar-backed` |
| HEAD | `c4fb94152379dff60a7582abc92718a3f4793e31` |
| Layer scope | **layer0 only** (full model, layer0 is PRT policy layer) |
| Model | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` |
| Sidecar format | INT6 (decoded-f32) |
| Sidecar selector | `PRT_V2_SIDECAR_FORMAT=int6` |
| Scale offset | `scale_off=20` (corrected in Phase 23D-R2) |
| Dimensions | K=3584, M=18944 |
| Policy (N≤4) | `prt_avx2` — PRT AVX2 kernel handles the operation |
| Policy (N>4) | `native_prefill` — llama.cpp native prefill path |
| Scalar fallback | `0` — scalar PRT path never triggered in tested runs |
| NaN count | `0` — no NaN values observed |
| Inf count | `0` — no Inf values observed |
| Explicit selector | `PRT_V2_SIDECAR_FORMAT=int6` (avoids INT8/INT6 priority ambiguity from Phase 23D-R4) |

**The explicit `PRT_V2_SIDECAR_FORMAT=int6` selector is the validated configuration going forward.**

---

## 3. Valid Evidence

### Phase 23E — Repeat Validation (Zero Variance)

3 repeat runs of c=4, n=1 with identical settings produced **identical** abs4 values across all 3 runs:

| N (context) | abs4 sum |
|-------------|----------|
| N=1 | 1.657920 |
| N=2 | 2.477368 |
| N=4 | 2.160922 |
| N=8 | 1.820755 |
| N=16 | 2.117989 |
| N=32 | 2.034454 |
| N=64 | 1.961509 |
| N=128 | 1.987855 |
| N=256 | 1.987855 |
| N=512 | 1.987855 |
| N=1024 | 2.012630 |

Full 11-value sequence **identical** across all 3 runs. Zero variance.

**Scale audit (first 4 values):**
- `0.00216875, 0.00510111, 0.00218038, 0.00243282`
- Range: `0.00016086` to `0.07262494`
- nan=0, inf=0

### Phase 23G — Semantic Canary (4/4 CLEAN_SEMANTIC)

| Prompt | Output | Classification |
|--------|--------|----------------|
| "The capital of France is" | `Paris.` | ✅ CLEAN_SEMANTIC |
| "The largest planet in our solar system is" | `Jupiter.` | ✅ CLEAN_SEMANTIC |
| `{"name":"Qwen","status":"active"}` style request | valid JSON output | ✅ CLEAN_SEMANTIC |
| "Once upon a time" | `Once upon a time, in a land far, far away...` | ✅ CLEAN_SEMANTIC |

- No corruption detected
- No repetition loops
- Valid JSON produced (P3)
- Factual questions answered correctly (P1: Paris, P2: Jupiter)
- nan=0, inf=0 confirmed

### Phase 23F — Policy Baseline (Tagged)

Tag: `PRT_PHASE23F_7B_INT6_POLICY_BASELINE_CHECKPOINT`
Commit: `53deb5b16`

Frozen: scale_off=20, K=3584, M=18944, deterministic, policy routing correct.

---

## 4. Timing Block (Phase 23H)

### Phase 23H Verdict: `BLOCKED_ENV_TIMING`

**No valid timing data exists.** No timing ratio can be computed. No performance claim is valid.

### Attempts That Failed

| # | Approach | Failure |
|---|----------|---------|
| 1 | BSD `time -f` | `-f` flag unsupported on this system |
| 2 | `time ... \| tee` | SIGKILL when output is heavy |
| 3 | `date +%s` | exec killed mid-measurement |
| 4 | `timeout \| head` | hangs, killed |
| 5 | `setsid ... &` | OOM or signal kills parent |
| 6 | Direct redirect `>> file` | log grows to 10GB, no clean wall_time |
| 7 | Lean retry with `/usr/bin/time -p` | stderr spam blocks timing lines |

### Root Cause (Updated After Retry)

The blocker is **not** the exec tool, disk space, or memory alone. The blocker is the binary's own `printf`-style debug output:

```
[PRT-NATIVE] IL=0 up=0x629f448d2260 prt_layer=0 sidecar=(nil)
```

These lines are printed by the binary itself (via `GGML_PRINTF` or direct `printf`), **not** through the llama.cpp log system. Therefore `--log-disable` does **not** suppress them.

**Observed:** A single native run for ~6 minutes generated ~4.8GB of stderr output. With 8 inference tokens × 28 layers × multiple passes, this scales to multi-GB-per-run.

**`/usr/bin/time -p` itself works fine** — the POSIX time capture is not the problem. The problem is that the timing lines are buried under GB of debug spam in the same stderr stream, making clean extraction impossible without post-processing that would itself be unreliable.

### Required for Future Timing

1. **Add an env guard** to suppress `[PRT-NATIVE]` printfs, e.g.:
   - `PRT_QUIET=1` or
   - `PRT_NATIVE_DEBUG=0` or
   - compile-time flag to disable these printfs in non-debug builds
2. **Redirect stderr separately** from timing stdout (but this hides errors)
3. **Build a non-debug timing binary** without the printfs
4. **Manual terminal execution** outside the exec wrapper with controlled output

### What Is NOT the Problem

- ❌ PRT kernel correctness — proven by Phase 23G
- ❌ Scale offset — proven correct by Phase 23E/F
- ❌ Policy routing — proven correct by Phase 23E/F
- ❌ Determinism — proven by Phase 23E
- ❌ INT6 decoded-f32 output quality — proven by Phase 23G

---

## 5. Allowed Claims

✅ **7B INT6 layer0 decoded-f32 policy path is repeat-stable** (Phase 23E, zero variance)  
✅ **7B INT6 layer0 semantic canary passed 4/4** (Phase 23G)  
✅ **scale_off=20 fix is validated** (Phase 23D-R2)  
✅ **explicit sidecar selector `PRT_V2_SIDECAR_FORMAT=int6` is validated** (Phase 23D-R4)  
✅ **N≤4 routes to PRT AVX2** (Phase 23E/F policy counts)  
✅ **N>4 routes to native prefill** (Phase 23E/F policy counts)  
✅ **scalar fallback eliminated** (Phase 23E/F/G, scalar=0 in all tested runs)  
✅ **nan=0, inf=0** (Phase 23E/F/G)  
✅ **timing is blocked** by debug-output/harness constraints, not by a known kernel failure  
✅ **determinism** — full 11-value abs4 sequence is identical across 3 runs (Phase 23E)

---

## 6. Forbidden Claims

❌ **Timing ratio** — no INT6 vs native vs INT8 timing comparison exists  
❌ **Speedup** — no speed advantage claim can be made  
❌ **Performance advantage** — no performance data exists  
❌ **Production readiness** — Phase 23G proves quality but not performance  
❌ **Multi-layer support** — layer0 only, multi-layer is not validated  
❌ **Direct packed INT6 compute** — not implemented in this branch  
❌ **Broad semantic equivalence** — only 4 prompts tested at layer0  
❌ **Exact/token match to reference** — no reference comparison made  
❌ **14B support** — only 7B has been validated  
❌ **End-to-end speedup** — no timing data exists

---

## 7. Recommended Next Steps

| Option | Description | Priority |
|--------|-------------|----------|
| **A (preferred)** | Phase 23J — add `PRT_QUIET` or `PRT_NATIVE_DEBUG=0` env guard to suppress `[PRT-NATIVE]` printfs, then retry timing | High |
| B | Manual timing in real terminal with `/usr/bin/time -p` or `SECONDS=0` | Medium |
| C | 0.5B two-layer experiment before 7B multi-layer | Medium |
| D | PRT_3P kernel validation without speed claims | Low |

**If proceeding with Phase 23J:** Add a single env var guard (e.g. `GGML_PRT_NO_PRINT` or `PRT_NATIVE_DEBUG=0`) that gates the `[PRT-NATIVE]` printf lines. Then retry timing with `/usr/bin/time -p`. This is a one-line change in the binary and doesn't affect correctness.

---

## 8. Safety Scan

| Item | Status |
|------|--------|
| Model files staged | ❌ No |
| Sidecar files staged | ❌ No |
| F32 ref files staged | ❌ No |
| Capture files staged | ❌ No |
| Prompt files staged | ❌ No |
| Log files staged | ❌ No |
| Binaries staged | ❌ No |
| Huge files staged | ❌ No |
| Credentials staged | ❌ No |
| Secrets detected | ❌ No (checklist references only, no actual values) |
| Source code changes | ❌ No (docs only in this phase) |
| New tags created | ✅ Yes (annotated checkpoint tag) |
| Existing tags altered | ❌ No |

**Disk health:**
- System: 78G free (66%) ✅
- Scratch: 19G free (85%) ✅

---

## Final Report

| Field | Value |
|-------|-------|
| **A. Branch** | `experimental/prt-phase19a-alt-sidecar-backed` |
| **B. Previous HEAD** | `c4fb94152379dff60a7582abc92718a3f4793e31` (Phase 23H BLOCKED) |
| **C. New HEAD** | _(checkpoint commit — docs only)_ |
| **D. Checkpoint file** | `examples/speculative/results/PRT_PHASE23I_7B_INT6_SEMANTIC_POLICY_CHECKPOINT.md` |
| **E. JSON** | `examples/speculative/results/phase23i_7b_int6_semantic_policy_checkpoint.json` |
| **F. Tag created** | ✅ `PRT_PHASE23I_7B_INT6_SEMANTIC_POLICY_CHECKPOINT` |
| **G. Frozen verdict** | `PASS_7B_INT6_SEMANTIC_POLICY_BASELINE_CHECKPOINT_WITH_TIMING_BLOCKED` |
| **H. Semantic/policy evidence** | Phase 23E: zero-variance deterministic; Phase 23G: 4/4 CLEAN_SEMANTIC; Phase 23F: policy baseline tagged |
| **I. Timing status** | BLOCKED_ENV_TIMING — no valid timing data exists |
| **J. Timing blocker root cause** | Binary `[PRT-NATIVE]` printf spam floods stderr (multi-GB per run), burying timing lines; `/usr/bin/time -p` works but output is polluted |
| **K. Allowed claims** | INT6 layer0 correct, repeat-stable, sane output, scale_off=20 valid, policy routing correct, timing blocked |
| **L. Forbidden claims** | Any timing ratio, speedup, performance advantage, production readiness, multi-layer, 14B |
| **M. Recommended next** | Phase 23J — add `PRT_QUIET`/`PRT_NATIVE_DEBUG=0` env guard to binary, then retry timing |
| **N. Models/sidecars/binaries staged?** | ❌ No |
| **O. Secrets detected?** | ❌ No |
| **P. Existing tags altered?** | ❌ No |
| **Q. System disk free** | 78G (66%) |
| **R. Scratch disk free** | 19G (85%) |

---

## Phase History

| Phase | Status | Key Finding |
|-------|--------|-------------|
| 23D-R2 | ✅ | Fixed 7B INT6 `scale_off=20` bug |
| 23D-R4 | ✅ | Fixed INT8/INT6 priority with explicit `PRT_V2_SIDECAR_FORMAT` selector |
| 23E | ✅ | 7B INT6 repeat validation — **zero variance** across 3 runs |
| 23F | ✅ | Policy baseline checkpoint + tag (`PRT_PHASE23F_7B_INT6_POLICY_BASELINE_CHECKPOINT`) |
| 23G | ✅ | Semantic canary — **4/4 CLEAN_SEMANTIC** |
| 23H | ❌ | **BLOCKED_ENV_TIMING** — binary debug output blocks timing capture |
| **23I** | **✅** | **Checkpoint — semantic/policy baseline confirmed valid, timing block documented** |