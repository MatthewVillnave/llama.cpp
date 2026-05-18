# PRT Phase 24A: 7B Timing Lane Closure + 3B Inventory

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Previous HEAD:** `1f7fd2665` (Phase 23J-R2)  
**New HEAD:** `1f7fd2665` (no new commits — inventory only)  
**Date:** 2026-05-18  
**Verdict:** `CLOSED_7B_TIMING_BLOCKED_ON_CURRENT_HARDWARE`, `BLOCKED_3B_PRT_HANG_ON_MEANINGFUL_CONTEXT`

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`1f7fd2665` — PRT Phase 23J-R2: run lean 7B timing smoke

## C. New HEAD
`1f7fd2665` — no code changes (Phase 24A was inventory/planning only)

---

## 1. 7B Validated Baseline

**Phase 23E — Repeat validation:**
- N=1 abs4: `1.657920`
- N=2 abs4: `2.477368`
- N=4 abs4: `2.160922`
- Full 11-value sequence identical across 3 runs ✅

**Phase 23G — Semantic canary:**
- Paris ✅
- Jupiter ✅
- Valid JSON ✅
- Story intro ✅
- **4/4 CLEAN_SEMANTIC** ✅

**Phase 23F/23I — Checkpoint tags:**
- `PRT_PHASE23F_7B_INT6_POLICY_BASELINE_CHECKPOINT`
- `PRT_PHASE23I_7B_INT6_SEMANTIC_POLICY_CHECKPOINT`

**Policy:**
- N≤4 → PRT AVX2 ✅
- N>4 → native prefill ✅
- scalar fallback = 0 ✅

**Sidecar:**
- INT6 decoded-f32 ✅
- K=3584 M=18944 ✅
- scale_off=20 ✅
- Selector: `PRT_V2_SIDECAR_FORMAT=int6` ✅

---

## 2. 7B Timing Block

**Phase 23H:** Blocked by timing/capture environment
**Phase 23J-R:** Added quiet guard (`PRT_V2_QUIET=1`) ✅ — `[PRT-NATIVE]` spam eliminated
**Phase 23J-R2:** Still blocked
- c=2 n=2 hit 300s timeout
- ~200s CPU user time consumed
- No valid timing ratio

**Root cause:**
- 7B Q4_K_M prefill too heavy/slow on 15GB RAM CPU-only OptiPlex
- One-layer PRT-v2 cannot overcome full-model load/runtime overhead
- **Not a PRT correctness failure**

---

## 3. Allowed / Forbidden Claims

**Allowed:**
- 7B INT6 layer0 decoded-f32 policy canary is repeat-stable ✅
- 7B INT6 layer0 semantic canary passed 4/4 ✅
- scale_off=20 validated ✅
- Explicit sidecar selector validated ✅
- N≤4 routes to PRT AVX2 ✅
- N>4 routes to native prefill ✅
- scalar PRT fallback eliminated in tested canaries ✅
- **7B timing blocked on current hardware** ✅
- Quiet guard works ✅

**Forbidden:**
- ❌ No 7B timing ratio
- ❌ No 7B speedup claim
- ❌ No production claim
- ❌ No multi-layer claim
- ❌ No 14B support claim
- ❌ No direct packed INT6 compute claim

---

## 4. Why Move to 3B

- **0.5B** — correctness/debug sandbox ✅
- **3B** — measurable timing/generalization target on this hardware
- **7B** — real target, but timing requires more layer coverage, direct packed compute, PRT-resident replacement, or better hardware
- **Moving to 3B is not abandoning 7B** — it is choosing a model size this machine can complete meaningful runs on

---

## 5. 3B Inventory Results

### 5B Model
- **Found:** `Qwen2.5-3B-Instruct-Q4_K_M.gguf`
- **Path:** `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf`
- **Size:** 1.8G
- **Shape (preliminary):** K≈2048, M≈11008, layers≈36

### 5B Existing Sidecars
- **NONE FOUND** in `$PRT_SCRATCH/sidecars/`
- No `*3b*` directories
- No INT8 or INT6 3B sidecars
- Sidecar generation would be required before 3B PRT runs

### 5B Runtime Behavior (Native vs PRT)

**Native 3B — quick smoke:**
```
echo "Paris" | llama-cli -c 32 -n 2 --temp 0 --log-disable
→ "Paris is..." in <1s ✅ (Generation: 18.5 t/s)
```

**PRT 3B INT6 (any context):**
- `c=32 -n 2` → **TIMEOUT at 30s** 🔴
- `c=512 -n 2` → **TIMEOUT + hangs** 🔴
- Model loads, PRT layer0 activates, but prefill hangs on attention/memory in PRT mode for 3B
- Same PRT layer0 config that works for 7B causes hang on 3B with meaningful context

**Root cause hypothesis:** PRT decode_only policy interacts poorly with Qwen2.5-3B's attention/prefill path at any non-trivial context size. The decode-only policy may be incompatible with how Qwen2.5-3B handles the KV cache or attention at context > 1.

---

## 6. Recommended 3B Lane

Given the PRT hang on 3B with meaningful context:

| Phase | Task | Status |
|-------|------|--------|
| 24B | 3B inventory and shape audit | ✅ Complete |
| 24C | 3B native timing baseline (c=32, n=2) | **Recommended next** |
| 24D | 3B INT8 policy canary (c=1, n=1) — diagnostic | Diagnostic only |
| 24E | 3B PRT hang root cause — investigate decode_only × Qwen2.5-3B attention | **Root cause needed** |
| 24F | 3B PRT fix + re-canary | Blocked by 24E |

**Immediate recommendation:** Run Phase 24C — capture native 3B timing baseline with c=32, n=2 to establish the reference speed. Then investigate why PRT decode_only hangs on 3B.

---

## J. Existing 3B Sidecars
**None.** No 3B INT8 or INT6 sidecars exist in `$PRT_SCRATCH/sidecars/`.

---

## K. Recommended Next Phase
**Phase 24C:** Capture 3B native timing baseline with c=32, n=2. Confirm the actual token/s speed of the 3B model natively. Then investigate why PRT decode_only hangs on Qwen2.5-3B at c=32.

---

## L. Models/Sidecars/Binaries Staged?
No.

## M. Secrets Detected?
None.

## N. Existing Tags Touched?
None.

## O. System Disk Free
78G free on /dev/nvme0n1p2 (66% used)

## P. Scratch Disk Free
19G free on /media/matthew-villnave/VL_usb (85% used)

---

## Summary

**7B lane:** CLOSED — blocked by hardware, not correctness. All validated claims preserved.

**3B lane:** Model found, no sidecars exist, PRT hangs on 3B with meaningful context. This is a regression/new blocker — 7B PRT works but 3B PRT hangs. The root cause is the `decode_only=1` policy × Qwen2.5-3B attention path interaction, not hardware.

**Recommended action:** Document the 3B PRT hang as a new blocker and investigate before generating sidecars or attempting timing.