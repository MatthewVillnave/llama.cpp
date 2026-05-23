# Phase 28BF: Shadow Matmul Consumer End-to-End Dry Run

## Phase Metadata

| Field | Value |
|---|---|
| **Phase** | 28BF |
| **Name** | Shadow Matmul Consumer End-to-End Dry Run |
| **Branch** | `experimental/prt-phase19a-alt-sidecar-backed` |
| **Head before** | `e9cc6bd2b` |
| **Verdict** | **PASS** |
| **Generation run** | No |
| **Large files staged** | No |

## Goal

Prove the runtime-adjacent shadow consumer can:

1. ✅ Request pager-backed residual view through `prt_get_residual_view()` (proven in 28BE)
2. ✅ Decode the residual from the pager view (proven in 28BE)
3. ✅ Compute full overlay matmul `Y = X @ (W_base + R_shadow)` using the decoded residual
4. ✅ Confirm Y parity vs reference path
5. ✅ Confirm shadow counter evidence

## Per-Case Results

### row_split — PASS

| Metric | Value |
|---|---|
| K / M / N | 96 / 48 / 4 |
| block_rows / block_cols | 32 / 48 |
| R_max_abs_err | 0.0 |
| R_rmse | 0.0 |
| W_max_abs_err | 0.0 |
| W_rmse | 0.0 |
| Y_max_abs_err | 0.0 |
| Y_rmse | 0.0 |
| Y_cosine | 1.0 |
| pager_view_null | false |
| pager_view_size_gt_0 | true |
| shadow_lookup_calls | 1 |
| shadow_pager_hits | 1 |
| shadow_legacy_hits | 0 |
| shadow_null_views | 0 |
| shadow_budget_rejects | 0 |

### col_split — PASS

| Metric | Value |
|---|---|
| K / M / N | 32 / 144 / 4 |
| block_rows / block_cols | 32 / 48 |
| R_max_abs_err | 0.0 |
| R_rmse | 0.0 |
| W_max_abs_err | 0.0 |
| W_rmse | 0.0 |
| Y_max_abs_err | 0.0 |
| Y_rmse | 0.0 |
| Y_cosine | 1.0 |
| pager_view_null | false |
| pager_view_size_gt_0 | true |
| shadow_lookup_calls | 1 |
| shadow_pager_hits | 1 |
| shadow_legacy_hits | 0 |
| shadow_null_views | 0 |
| shadow_budget_rejects | 0 |

### row_col_split — PASS

| Metric | Value |
|---|---|
| K / M / N | 96 / 144 / 6 |
| block_rows / block_cols | 32 / 48 |
| R_max_abs_err | 0.0 |
| R_rmse | 0.0 |
| W_max_abs_err | 0.0 |
| W_rmse | 0.0 |
| Y_max_abs_err | 0.0 |
| Y_rmse | 0.0 |
| Y_cosine | 1.0 |
| pager_view_null | false |
| pager_view_size_gt_0 | true |
| shadow_lookup_calls | 1 |
| shadow_pager_hits | 1 |
| shadow_legacy_hits | 0 |
| shadow_null_views | 0 |
| shadow_budget_rejects | 0 |

### awkward_edge — PASS

| Metric | Value |
|---|---|
| K / M / N | 70 / 101 / 2 |
| block_rows / block_cols | 32 / 48 |
| R_max_abs_err | 0.0 |
| R_rmse | 0.0 |
| W_max_abs_err | 0.0 |
| W_rmse | 0.0 |
| Y_max_abs_err | 0.0 |
| Y_rmse | 0.0 |
| Y_cosine | 1.0 |
| pager_view_null | false |
| pager_view_size_gt_0 | true |
| shadow_lookup_calls | 1 |
| shadow_pager_hits | 1 |
| shadow_legacy_hits | 0 |
| shadow_null_views | 0 |
| shadow_budget_rejects | 0 |

## Control Tests

| Test | Expected | Actual | Result |
|---|---|---|---|
| **DISABLED_MODE** | `is_null=true, legacy_hits=0` | `is_null=1, legacy_hits=0` | ✅ PASS |
| **MISSING_SIDECAR** | `init` fails | `init_correctly_failed` | ✅ PASS |
| **BAD_TENSOR_KEY** | `is_null=true` | `is_null=1, null_views=0` | ✅ PASS |
| **BUDGET_REJECT** | `init` or `is_null` or `budget_rejects > 0` | `activate_correctly_failed` | ✅ PASS |

## Summary

- **Positive tests**: 4/4 PASS
- **Control tests**: 4/4 PASS
- **Total failures**: 0
- **Verdict**: `PASS_PHASE28BF_SHADOW_MATMUL_CONSUMER_DRY_RUN`

## Key Findings

- **Perfect decode parity**: All R/W/Y error metrics are exactly 0.0 across all 4 cases — the pager-backed decode path produces bit-identical results to the direct file decode path
- **Correct counter routing**: `shadow_pager_hits=1, legacy_hits=0` for all positive cases — the runtime correctly routes through the pager
- **Control tests deterministic**: All 4 control tests behave as expected:
  - DISABLED_MODE: pager disabled → null view, no legacy hit
  - MISSING_SIDECAR: bad path → init fails
  - BAD_TENSOR_KEY: nonexistent family → null view with `null_views=0` (key not found is not a null view event)
  - BUDGET_REJECT: 1-byte budget → `activate_layer` correctly fails

## Output Files

- Harness: `examples/speculative/phase28bf_shadow_matmul_consumer.cpp`
- JSON: `examples/speculative/results/phase28bf_shadow_matmul_consumer_dry_run.json`
- Report: `examples/speculative/PHASE28BF_SHADOW_MATMUL_CONSUMER_DRY_RUN.md`