# Phase 28BR-H: MethodA True Injection Canary

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Old HEAD:** `d85ffcf86` (Phase 28BR-G)
**New HEAD:** pending (mmap fix committed)
**Timestamp:** `2026-05-24T14:05 EDT`
**Verdict:** `PASS`

---

## Goal

Apply the mmap fallback fix (Phase 28BR-H-R2) to the `build_prt_true_attn_out_injection` injection point, then run the full A-G test suite to confirm that MethodA materialization works inside the llama-graph context.

Target: `layer=0`, `family=attn_out`, `Qwen2.5-0.5B`, `prompt="Hi"`, `n_predict=1`.

---

## Key Finding

**TRUE INJECTION SUCCEEDED.** The mmap fallback fixes the `no_alloc=true` materialization blocker that stopped Phase 28BR-F:

```
[PRT-INJECT-CANARY] il=0 family=attn_out action=mutated_output
R=[896,896] X=[896,30] out=[896,30]
injection_attempts=1 injection_successes=1 injection_failures=0
contribution_finite_before_injection=1
sidecar_math_influenced_output=1
```

---

## Mechanism

**The blocker:** `ggml_new_tensor(ctx0, ...)` with `ctx0` from `no_alloc=true` context → `delta_w->data == nullptr` during graph construction. `memcpy` into null pointer would crash.

**The fix:** When `delta_w->data == nullptr` after `ggml_new_tensor`, manually `mmap` a buffer and assign `delta_w->data = w_buf` before `memcpy`.

```cpp
// src/llama-graph.cpp:1365-1373
if (delta_w == nullptr || delta_w->data == nullptr) {
    size_t w_bytes = (size_t) r_rows * r_cols * sizeof(float);
    void * w_buf = mmap(NULL, w_bytes, PROT_READ|PROT_WRITE,
                        MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (w_buf == MAP_FAILED) {
        prt_true_injection_record_attempt_result(false, "delta_weight_mmap_failed");
        return native_out;
    }
    delta_w->data = w_buf;
}
```

Precedent: Phase 21G at `llama-graph.cpp:1928` (LoRA weight loading path already uses mmap for similar case).

---

## Counter Values (Test D — True Injection)

| Counter | Value |
|---|---|
| materialization_attempts | 1 |
| materialization_successes | 1 |
| materialization_failures | 0 |
| bytes_copied_to_ggml_tensor | 3,211,264 |
| contribution_attempts | 1 |
| contribution_successes | 1 |
| injection_attempts | 1 |
| injection_successes | 1 |
| injection_failures | 0 |
| decoded_residual_nan | 0 |
| decoded_residual_inf | 0 |
| decoded_residual_finite | true |
| sidecar_math_influenced_output | true |
| raw_bytes_cast_to_float | false |

---

## Test Results (A-G)

### A — Baseline ✅
- No pager, no prt-mode, no apply, no injection
- Exit: 0 | injection_attempts: 0 | sidecar_math_influenced_output: false

### B — Observe-only ✅
- pager + prt-mode 5700 + valid manifest + no apply
- Exit: 0 | activation_successes: 1 | decoded_views: 0 | injection_attempts: 0

### C — Shadow-only ✅
- pager + prt-mode 5700 + apply ON + true injection OFF
- Exit: 0 | decoded_views: 1 | app_attempts: 1 | app_success: 1 | injection_attempts: 0

### D — True Injection ✅
- pager + prt-mode 5700 + apply ON + true injection ON + layer0/attn_out
- Exit: 0 | materialization_successes: 1 | injection_successes: 1 | action: mutated_output
- R finite before injection: YES
- sidecar_math_influenced_output: true

### E — Wrong Target (layer=1) ✅
- true injection ON + layer=1 (not in manifest)
- Exit: 0 | injection_skipped: 1 | reason: injection_skipped_wrong_layer
- No crash ✅

### F — Budget=0 ✅
- true injection ON + budget=0
- Exit: 0 | budget_rejects: 5 | activation_successes: 0 | injection_attempts: 0
- No crash ✅

### G — Missing Manifest ✅
- true injection ON + missing manifest
- Exit: 0 | error: "PRT sidecar pager manifest not found"
- Deterministic failure, no silent fallback ✅

---

## Boundaries

- No clamping added.
- No NaN zeroing added.
- No fake injection.
- No quality claim.
- No speed claim.
- No Q2→Q4 recovery claim.
- No production-readiness claim.
- Output observation only: "experimental sidecar-influenced output differed from baseline."

---

## What Was NOT Proven

- Output correctness or quality
- Long generation stability
- Multi-layer injection
- Multi-family injection
- Q2→Q4 recovery
- Speedup
- 30B feasibility
- Production readiness

---

## Commit

```text
git add src/llama-graph.cpp
git commit -m "Phase 28BR-H: mmap fallback fixes true injection materialization"
```

---

## Phase Chain

- Phase 28BR-A: decode-once residual buffer lifecycle canary
- Phase 28BR-B: guarded residual contribution shadow compare
- Phase 28BR-C: runtime decoder forensics rerun
- Phase 28BR-E: pager global ODR wiring fix
- Phase 28BR-F: guarded true injection canary (BLOCKED: no_alloc=true)
- Phase 28BR-G: GGML constant materialization probe (MethodA proven standalone)
- **Phase 28BR-H: MethodA true injection canary — PASS**