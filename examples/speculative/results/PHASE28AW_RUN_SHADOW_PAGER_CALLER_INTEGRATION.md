# Phase 28AW: Run Shadow Pager Caller Integration

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`93bf442d9` (Phase 28AV)

## C. run_shadow_test() Modification
Modified `examples/speculative/prt_shadow.h`:
- Added `#include "prt_sidecar_runtime_link.h"` for `prt_get_residual_view()` and `prt_residual_view`
- Changed `g_sidecars` from `static` to `extern` (was previously changed during 28AV)
- Changed `g_sidecars_loaded` from `static` to `extern` for consistency
- Added `g_shadow_lookup_calls`, `g_shadow_pager_hits`, `g_shadow_legacy_hits`, `g_shadow_null_views`, `g_shadow_budget_rejects` counters
- Modified `run_shadow_test()` lookup path (behind `PRT_SIDECAR_PAGER_EXPERIMENTAL`):
  - Calls `prt_get_residual_view(layer, "ffn_up")`
  - If view is valid: `W_sidecar = reinterpret_cast<const float*>(view.data)`
  - If view is null: increments `g_shadow_null_views++`, falls back to legacy `g_sidecars.find()`
  - Stats counters updated per lookup outcome
- Added `prt_get_shadow_stats()` and `prt_reset_shadow_stats()` helper functions

## D. Tensor Family/Source Mapping
- `run_shadow_test()` calls `prt_get_residual_view(layer, "ffn_up")` — fixed "ffn_up" tensor family
- `matmul_prt_3plane(X_act, W_sidecar, Y_prt, batch, M, N)` receives the sidecar data pointer
- `prt_residual_view.data` (uint8_t*) reinterpret_cast to float* for matmul

## E. Caller Harness Path
`examples/speculative/prt_run_shadow_lookup_harness.cpp` — tests run_shadow_test() directly (not just the wrapper)

## F. Legacy Disabled Result
```
run_shadow_test(0): has_sidecar=true computed=true ✅
run_shadow_test(99): has_sidecar=false ✅
stats: calls=2 legacy_hits=1 null_views=1 ✅
Result: PASS_LEGACY_DISABLED ✅
```

## G. Pager Enabled Result
```
activate_layer(0): ACTIVATED ✅
run_shadow_test(0): has_sidecar=true computed=true ✅
stats: calls=1 pager_hits=1 legacy_hits=0 ✅
Result: PASS_PAGER_ENABLED ✅
```

## H. Fallback/Budget Result
```
Unknown layer 99 → has_sidecar=false null_views=1 ✅
Budget reject (strict 256KB): has_sidecar=true (legacy fallback) ✅
stats: calls=1 null_views=0 legacy_hits=1 ✅
Result: PASS_FALLBACK ✅ | PASS_BUDGET_REJECT ✅
```

## I. Build/Default Behavior
- Build: `g++ -O2 -std=c++17 -DPRT_SIDECAR_PAGER_EXPERIMENTAL -I. -Iggml/include -Iinclude prt_sidecar_pager.cpp prt_run_shadow_lookup_harness.cpp -o harness`
- 4/4 tests PASS
- **llama-cli default preserved**: no behavior change without flag

## J. Stats/Counters
- `g_shadow_lookup_calls`: total lookup calls
- `g_shadow_pager_hits`: pager provided valid view
- `g_shadow_legacy_hits`: legacy sidecar used
- `g_shadow_null_views`: pager returned null
- `g_shadow_budget_rejects`: activation rejected
- `prt_get_shadow_stats()` / `prt_reset_shadow_stats()` interface

## K. Limitations
- `ggml.h/llama.h/common.h` removed from `prt_shadow.h` for standalone compilation
- `g_sidecars` and `g_sidecars_loaded` changed to `extern` — requires single-definition constraint
- Synthetic .trit files only

## L. Recommended Next Phase
Phase 28AX: Full end-to-end dry run with synthetic manifest, --enable-prt-sidecar-pager flag, no generation.

## M. Models/Sidecars/F32 Refs Staged?
No. Synthetic `/tmp` only.

## N. Secrets Detected?
None.

## O. Tags Touched?
No.