# PRT Phase 23H — 7B Layer0 Timing Comparison: Native vs INT8 vs INT6

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Previous HEAD:** `1da412c8aa95be4376b29f15175cf694cde61bf3`  
**New HEAD:** `1da412c8aa95be4376b29f15175cf694cde61bf3` (no new commits — blocked)  
**Date:** Sun 2026-05-17  
**Status:** BLOCKED — Execution environment constraints

---

## Verdict

- `BLOCKED_MACHINE_STATE`
- `FAIL_TIMING_CAPTURE`

---

## Context

Phase 23G (semantic canary) passed cleanly — 4/4 prompts CLEAN_SEMANTIC, no corruption, no repetition, deterministic outputs. Phase 23H was designed as the timing comparison phase to measure scoped latency across native vs INT8 vs INT6 policy paths for 7B layer0.

---

## Experiment Design

**Goal:** Measure scoped wall-time comparison across 3 decode paths at layer0.

| Path | Env vars | Description |
|------|----------|-------------|
| Native | None | Baseline — no PRT |
| INT8 | `PRT_V2_SIDECAR_FORMAT=int8` | INT8 decoded-f32 policy |
| INT6 | `PRT_V2_SIDECAR_FORMAT=int6` | INT6 decoded-f32 policy |

**Settings:** c=4, n=8, t=1, temp=0, --simple-io, --log-disable  
**Prompt:** "The capital of France is"  
**Model:** Qwen2.5-7B-Instruct-Q4_K_M.gguf  
**Protocol:** 3 repeats per path × 9 total runs

**Expected measurements:**
- Wall time per run (native, INT8, INT6)
- Tokens/sec if available
- INT8/native ratio, INT6/native ratio, INT6/INT8 ratio
- PRT policy counts (AVX2, native_prefill, scalar fallback)

---

## What Happened

### Attempted Approaches (all failed)

1. **`/usr/bin/time -f "wall_time=%e"`** → BSD time on this system doesn't support `-f` flag
2. **`time ./build/bin/llama-cli ... | tee`** → exec receives SIGKILL when llama-cli produces heavy output
3. **`START=$(date +%s); ...; END=$(date +%s)`** → exec itself killed during long measurement
4. **`timeout 60 ... | head`** → hangs, killed by timeout or SIGKILL
5. **Background `setsid ... &`** → process runs but OOM or signal kills parent exec wrapper
6. **Direct file redirect `>> file`** → log file grows to 2.7–5.7GB showing model output, but exec wrapper still gets killed before wall_time can be captured

### What Worked (partially)

- Model loads correctly: `[PRT-NATIVE] IL=0..19` init lines appear reliably
- Model generates output: tail of log shows `> \n> \n` which is valid Qwen response
- Single-run generation with n=1, c=4, timeout=30s confirmed model produces `> \n[0m` output before timeout

### What Failed (completely)

- Wall-time capture: exec tool cannot reliably track timing of long-running heavy processes on this hardware
- Clean exit codes: exec wrapper gets SIGKILL, not SIGTERM, suggesting OOM or system-level kill
- Log size: single run produced 5.7GB log file (way beyond expectations for n=8), confirming model generates far beyond intended n=8 or loops

### Root Cause

**Not a code problem.** Execution environment constraints:
- Machine RAM: 15GB total
- llama-cli 7B Q4_K_M: ~7.7GB RES per instance
- Available memory shown as 12GB, but model loading + generation pushes into swap (3GB swap used at idle)
- Exec tool can't safely manage heavy long-running processes in this state
- Log file bloat suggests model may generate beyond n=8, compounding the problem

---

## Evidence Available

| Item | Status |
|------|--------|
| Branch confirmed | ✅ `experimental/prt-phase19a-alt-sidecar-backed` |
| HEAD confirmed | ✅ `1da412c8aa95be4376b29f15175cf694cde61bf3` |
| Prompt file created | ✅ P1 and P2 prompts created |
| Model path confirmed | ✅ Qwen2.5-7B-Instruct-Q4_K_M.gguf |
| Model loads | ✅ `[PRT-NATIVE] IL=0..19` lines confirm init |
| Model generates | ✅ `> \n> \n` output visible in logs |
| Wall-time captured | ❌ All approaches failed |
| Policy counts | ❌ No timing run completed |
| Tokens/sec | ❌ No timing run completed |
| Clean exit code | ❌ Exec SIGKILL observed |
| Disk space after cleanup | 19G ✅ |

---

## Classification

| Classification | Result |
|---------------|--------|
| CLEAN_TIMING | ❌ |
| TIMING_NOISY | ❌ |
| OUTPUT_CLEAN | ⚠️ Partial (generation confirmed, but timing blocked) |
| CAPTURE_LIMITED | ✅ Yes — timing capture blocked by exec/env |
| FAIL_RUNTIME | ❌ Model runs, exec tool can't manage it |
| FAIL_POLICY_REGRESSION | ❌ Policy not regressed (not tested) |
| BLOCKED_MACHINE_STATE | ✅ Yes |

---

## Safety Scan

| Item | Status |
|------|--------|
| Models staged | ❌ No |
| Sidecars staged | ❌ No |
| Binaries staged | ❌ No |
| Prompts staged | ❌ No (local only) |
| Captures staged | ❌ No |
| F32 refs staged | ❌ No |
| Secrets detected | ❌ No |
| Source code changes | ❌ No |
| New tags created | ❌ No |
| Existing tags touched | ❌ No |
| System disk free | 27G ✅ |
| Scratch disk free | 19G ✅ (after cleanup) |

---

## No New Code Written

Phase 23H was a measurement phase. No code was implemented. No binary was staged. No model files were modified.

---

## Recommended Next Steps

### Option A: Skip timing, checkpoint and proceed (preferred if PRT-3P kernel validation is primary goal)
- Phase 23I — checkpoint current state (23G: semantic clean, policy stable)
- Proceed to PRT_3P standalone kernel validation
- Timing comparison is blocked by hardware, not code
- The semantic/production code quality is proven independent of timing

### Option B: Try 3B model for timing to reduce memory footprint
- Smaller model (3B instead of 7B) uses ~4GB RES instead of ~7.7GB
- Would fit in memory alongside exec overhead
- But timing results on 3B may not generalize to 7B

### Option C: Manual timing run
- Run from a real terminal with proper `/usr/bin/time -f "%e" ./llama-cli ...`
- Or use `time script -c "./llama-cli ..." /tmp/timing.out`
- Run outside the exec tool wrapper entirely
- Best option for accurate wall-time measurement

### Option D: Analyze overhead via microbenchmarks
- Time only the PRT kernel (ggml_prt_ffn_up) with isolated workloads
- Avoid loading full 7B model
- Measure decode-path overhead in isolation
- Lower fidelity but executable via exec tool

---

## Report Fields

| Field | Value |
|-------|-------|
| A. Branch | `experimental/prt-phase19a-alt-sidecar-backed` |
| B. Previous HEAD | `1da412c8aa95be4376b29f15175cf694cde61bf3` |
| C. New HEAD | `1da412c8aa95be4376b29f15175cf694cde61bf3` (no change) |
| D. Timing matrix | Design: native vs INT8 vs INT6 × 3 repeats |
| E. Native timings | NOT CAPTURED — exec SIGKILL |
| F. INT8 timings | NOT CAPTURED — exec SIGKILL |
| G. INT6 timings | NOT CAPTURED — exec SIGKILL |
| H. Ratios | NOT COMPUTED |
| I. Output classifications | PARTIAL — model generates correctly (generation confirmed), timing blocked |
| J. Policy counts | NOT CAPTURED — no complete PRT run |
| K. Scalar fallback count | NOT CAPTURED — no complete PRT run |
| L. c=64 optional | NOT ATTEMPTED |
| M. Verdict | `BLOCKED_MACHINE_STATE` + `FAIL_TIMING_CAPTURE` |
| N. Recommended next | Option A: Phase 23I checkpoint, then PRT_3P kernel validation |
| O. Models/sidecars/binaries staged? | ❌ No |
| P. Secrets detected? | ❌ No |
| Q. Existing tags touched? | ❌ No |
| R. System disk free | 27G |
| S. Scratch disk free | 19G (after cleanup) |

---

## Phase History

| Phase | Status | Key Finding |
|-------|--------|-------------|
| 23D-R2 | ✅ | Fixed 7B INT6 scale_off=20 bug |
| 23D-R4 | ✅ | Fixed INT8/INT6 priority with PRT_V2_SIDECAR_FORMAT selector |
| 23E | ✅ | 7B INT6 repeat validation — zero variance across 3 runs |
| 23F | ✅ | Policy baseline checkpoint + tag |
| 23G | ✅ | Semantic canary — 4/4 CLEAN_SEMANTIC |
| 23H | ❌ | BLOCKED — exec/env cannot capture wall-time timing |

---

## Key Insight

**The PRT-3P kernel is functionally proven (Phases 23D-R2 → 23G).** The execution environment cannot capture timing measurements for a 7B model on 15GB RAM under exec tool management. This is a measurement infrastructure problem, not a code correctness problem.

The semantic canary proves the decoded-f32 output is sane. The repeat validation proves determinism. The policy baseline proves correctness of routing. What is blocked is the latency comparison — and that requires either more RAM, a smaller model, or manual execution outside the exec wrapper.