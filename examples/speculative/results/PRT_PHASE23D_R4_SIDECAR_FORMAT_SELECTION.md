# Phase 23D-R4: Explicit Sidecar Format Selection

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## Previous HEAD (23D-R2)
`5812fc277`

## New HEAD (23D-R4)
`[NEW_COMMIT]`

---

## 1. Executive Summary

**Verdict: PASS — All phases cleared.**

- `PRT_V2_SIDECAR_FORMAT` env var selector works correctly
- INT6 explicitly selected when `=int6`, INT8 when `=int8`
- 0.5B INT6 regression check: PASS (scale_off=16, abs4 matches baseline)
- INT8 non-regression: PASS (abs4 matches baseline)

---

## 2. Sidecar Selection Fix

**Problem:** `llama-graph.cpp` had INT8 path with hardcoded priority — INT8 loaded first and marked `f32_weight_loaded=true`, blocking INT6 from ever running when both sidecars existed.

**Fix:** Added `PRT_V2_SIDECAR_FORMAT` env var selector in `src/llama-graph.cpp`:
- `int` = only try INT8 path
- `int6` = only try INT6 path
- `f32` = only try f32 path
- `auto` or unset = existing default behavior

**Implementation:**
```cpp
int g_prt_sidecar_format_override = 0;  // 0=auto, 1=int8, 2=int6, 3=f32

static struct PRTFormatAutoInit {
    PRTFormatAutoInit() {
        const char * e = getenv("PRT_V2_SIDECAR_FORMAT");
        if (e) {
            if (strcmp(e, "int8") == 0) g_prt_sidecar_format_override = 1;
            else if (strcmp(e, "int6") == 0) g_prt_sidecar_format_override = 2;
            else if (strcmp(e, "f32") == 0) g_prt_sidecar_format_override = 3;
            else g_prt_sidecar_format_override = 0;
        }
    }
} g_prt_format_auto_init;
```

**INT8 loading** now wrapped with:
```cpp
if (g_prt_sidecar_format_override == 0 || g_prt_sidecar_format_override == 1) {
    // Try INT8 path (only for auto or int8 override)
}
```

**INT6 loading** — if `override=2` (int6), error blocks fallback to INT8:
```cpp
if (g_prt_sidecar_format_override == 2) {
    prt_logf("[PRT_V2_SIDECAR_ERROR] requested=int6 missing path=%s — BLOCK fallback\n", int6_path);
}
```

---

## 3. INT6 Selection Test (0.5B, c=4)

**Env:** `PRT_V2_SIDECAR_FORMAT=int6`

| Evidence | Result |
|---|---|
| `requested=int6 skipped=int8` | ✅ |
| `selected=int6 path=.../prt_phase21h_v_int6_from_f32/ffn_up_layer0_prt.int6` | ✅ |
| `scale_off=16 reason=0.5B_or_default_16` | ✅ |
| `nan=0 inf=0` | ✅ |
| N=2 PRT AVX2 | ✅ |
| scalar fallback | 0 ✅ |

**0.5B INT6 abs4 values:**
| N | abs4 | Known Baseline | Match |
|---|---|---|---|
| 2 | 3.795631 | 3.795631 | ✅ |
| 2 | 1.952418 | 1.952418 | ✅ |
| 4 | 5.928558 | 5.928558 | ✅ |

**Scale audit:**
- Range: 0.00155972 to 0.00900958
- Mean: 0.00223749
- All sane (no NaN/Inf)

---

## 4. INT8 Non-Regression Test (0.5B, c=4)

**Env:** `PRT_V2_SIDECAR_FORMAT=int8`

| Evidence | Result |
|---|---|
| `requested=int8 selected=int8` | ✅ |
| N=2 PRT AVX2 | ✅ |
| scalar fallback | 0 ✅ |

**0.5B INT8 abs4 values:**
| N | abs4 | Known Baseline | Match |
|---|---|---|---|
| 2 | 8.409694 | 8.409694 | ✅ |
| 2 | 5.138701 | 5.138701 | ✅ |

---

## 5. Verdicts

| Verdict | Status |
|---|---|
| PASS_SIDECAR_FORMAT_SELECTOR | ✅ |
| PASS_05B_INT6_REGRESSION_CANARY | ✅ |
| PASS_INT8_NON_REGRESSION | ✅ |
| FAIL_INT6_STILL_BLOCKED_BY_INT8 | ❌ (FIXED) |
| FAIL_05B_INT6_SCALE_OFFSET_REGRESSION | ❌ (NOT REGRESSED) |
| FAIL_INT8_REGRESSION | ❌ (NOT REGRESSED) |
| BLOCKED_MACHINE_STATE | ❌ (CLEAN) |

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`5812fc277`

## C. New HEAD
`[NEW_COMMIT]`

## D. Sidecar selector env
`PRT_V2_SIDECAR_FORMAT=int8|int6|f32|auto`

## E. Sidecar selection fix
Added `g_prt_sidecar_format_override` global + auto-init struct.
INT8 block wrapped with format override guard.
INT6 error blocks silently fall back to INT8 when `=int6` requested.

## F. INT6 selected explicitly?
YES — `requested=int6 selected=int6` confirmed in logs

## G. 0.5B INT6 scale_off
`scale_off=16` (correct for 0.5B)

## H. 0.5B INT6 abs4
N=2: 3.795631, 1.952418 | First N=4: 5.928558
Matches prior known values exactly ✅

## I. INT8 non-regression
N=2: 8.409694, 5.138701 — matches prior baseline exactly ✅

## J. Scalar fallback count
0 (zero — all N routed to PRT AVX2)

## K. Verdict
PASS_SIDECAR_FORMAT_SELECTOR + PASS_05B_INT6_REGRESSION_CANARY + PASS_INT8_NON_REGRESSION

## L. Recommended next
Phase 23E — 7B INT6 repeat validation/checkpoint with `PRT_V2_SIDECAR_FORMAT=int6`

## M. Models/sidecars/binaries staged?
NO

## N. Secrets detected?
NO

## O. Existing tags touched?
NO (no tag created)