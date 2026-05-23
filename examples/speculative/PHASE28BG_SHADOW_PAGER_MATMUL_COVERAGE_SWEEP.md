# Phase 28BG: Shadow/Pager Matmul Coverage Sweep

**Verdict: PASS** ✅ | 2026-05-23

## Problem

`test_positive()` called `run_shadow_test()` with hardcoded `tensor_family="ffn_up"` for all 8 test cases. Each case uses a different family (`ffn_gate`, `ffn_down`, `attn_out`), so `run_shadow_test()` requested the wrong key, got null, and failed pass criteria. Only the 2 `ffn_up` cases passed.

## Fix Applied

### 1. `prt_shadow.h` — Add `tensor_family` parameter to `run_shadow_test()` and `prt_shadow_test()`

**Declaration (line 198):**
```cpp
// Before
ShadowResult run_shadow_test(int layer, int batch, const float* X_act,
                             const float* Y_float, int M, int N);
// After
ShadowResult run_shadow_test(int layer, int batch, const float* X_act,
                             const float* Y_float, int M, int N,
                             const char* tensor_family);
```

**Implementation (line 216):**
```cpp
// Before
prt_residual_view view = prt_get_residual_view(layer, "ffn_up");
// After
prt_residual_view view = prt_get_residual_view(layer, tensor_family);
```

**`prt_shadow_test()` wrapper (lines 329–330):**
```cpp
// Before
ShadowResult prt_shadow_test(int layer, int batch, const float* X_act,
                              const float* Y_float, int M, int N);

// After
ShadowResult prt_shadow_test(int layer, int batch, const float* X_act,
                              const float* Y_float, int M, int N,
                              const char* tensor_family);
```

### 2. `phase28bg_shadow_pager_coverage_sweep.cpp` — Pass correct family per case (line 201)

```cpp
// Before
ShadowResult sr = run_shadow_test(0, 1, X_act.data(), Y_ref.data(), (int)M, (int)N);
// After
ShadowResult sr = run_shadow_test(0, 1, X_act.data(), Y_ref.data(),
                                  (int)M, (int)N, tensor_family.c_str());
```

## Results — 8/8 PASS

| Case | Family | Layer | pager_hits | legacy_hits | null_views | Y_err | Cosine |
|------|--------|-------|-----------|-------------|-----------|-------|--------|
| ffn_up_l0_row | ffn_up | 0 | 1 | 0 | 0 | 0.0e+00 | 1.0000 |
| ffn_gate_l0_col | ffn_gate | 0 | 1 | 0 | 0 | 0.0e+00 | 1.0000 |
| ffn_down_l0_rc | ffn_down | 0 | 1 | 0 | 0 | 0.0e+00 | 1.0000 |
| attn_out_l0_awk | attn_out | 0 | 1 | 0 | 0 | 0.0e+00 | 1.0000 |
| ffn_up_l1_col | ffn_up | 1 | 1 | 0 | 0 | 0.0e+00 | 1.0000 |
| ffn_gate_l1_row | ffn_gate | 1 | 1 | 0 | 0 | 0.0e+00 | 1.0000 |
| ffn_down_l1_awk | ffn_down | 1 | 1 | 0 | 0 | 0.0e+00 | 1.0000 |
| attn_out_l5_rc | attn_out | 5 | 1 | 0 | 0 | 0.0e+00 | 1.0000 |

**All 8 positive tests:** `pager_hits=1`, `legacy_hits=0`, `null_views=0`, `Y_max_abs_err=0.0e+00`, `Y_cosine=1.0000`

## Controls (4/4 PASS)

| Control | Pass | Detail |
|---------|------|--------|
| DISABLED_MODE | ✅ | is_null=1, legacy_hits=0 |
| MISSING_SIDECAR | ✅ | init_correctly_failed |
| BAD_TENSOR_KEY | ✅ | is_null=1, null_views=0 |
| BUDGET_REJECT | ✅ | activate_correctly_failed |

## Files Modified

- `examples/speculative/prt_shadow.h` — added `tensor_family` parameter to `run_shadow_test()` and `prt_shadow_test()`
- `examples/speculative/phase28bg_shadow_pager_coverage_sweep.cpp` — pass per-case tensor_family to `run_shadow_test()`
- `examples/speculative/results/phase28bg_shadow_pager_matmul_coverage_sweep.json` — results (written by harness)