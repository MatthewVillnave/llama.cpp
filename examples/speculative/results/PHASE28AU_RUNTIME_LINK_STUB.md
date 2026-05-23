# Phase 28AU: Runtime Link Stub for prt_get_residual_view()

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`8dfc1c38d` (Phase 28AT)

## C. Files Changed
- `examples/speculative/prt_shadow.h` — added cmath include; changed g_sidecars to extern; removed ggml.h/llama.h/common.h (standalone compatibility)
- `examples/speculative/prt_sidecar_runtime_link.h` — NEW: prt_get_residual_view() routing wrapper, g_prt_pager/g_prt_pager_enabled externs
- `examples/speculative/prt_sidecar_runtime_link_harness.cpp` — NEW: 4-mode harness (--legacy, --pager, --fallback, --budget)

## D. Residual Wrapper Path
`prt_get_residual_view(int layer_idx, const std::string& tensor_family)` in `prt_sidecar_runtime_link.h`:
- **Pager enabled**: routes to `g_prt_pager->get_residual(layer_idx, tensor_family)`
- **Pager disabled**: falls back to legacy `g_sidecars` map lookup

## E. Pager Global/Stub Behavior
- `g_prt_pager: prt_sidecar_pager* = nullptr` — nullptr by default
- `g_prt_pager_enabled: bool = false` — disabled by default
- Only allocated when `prt_init_pager()` called with flag enabled
- No manifest required unless flag enabled
- `PRT_SIDECAR_PAGER_EXPERIMENTAL` compile guard gates all active code

## F. Legacy Disabled Result
```
prt_get_residual_view(0, ffn_up): is_null=false reason=legacy
Layer 0 via legacy: PASS
prt_get_residual_view(99, ffn_up): is_null=true reason=not_found
Missing layer: PASS
Result: PASS_LEGACY_DISABLED ✅
```

## G. Pager Enabled Result
```
prt_init_pager() = true
Pager stats: resident=0 reads=0
prt_get_residual_view(0, ffn_up): is_null=false size=393504 reason=
Result: PASS_PAGER_ENABLED ✅
```

## H. Missing/Budget Fallback Result
```
Unknown family → falls back to legacy (correct behavior)
Unknown layer → is_null=true reason=not_found
activate_layer(0) with 256KB budget: REJECTED ✅
budget_rejects = 1
Result: PASS_MISSING_FALLBACK ✅ | PASS_BUDGET_REJECT ✅
```

## I. Build/Default Behavior Checks
- Build: `g++ -O2 -std=c++17 -DPRT_SIDECAR_PAGER_EXPERIMENTAL -I. -Iggml/include -Iinclude prt_sidecar_pager.cpp prt_sidecar_runtime_link_harness.cpp -o harness`
- 4 tests: Legacy=PASS, Pager=PASS, Fallback=PASS, Budget=PASS
- **llama-cli default preserved**: no runtime behavior change, pager behind flag

## J. Limitations
- `prt_shadow.h` modified: ggml.h/llama.h removed for standalone compilation; full build system needed for actual llama-cli integration
- `g_sidecars` changed from `static` to `extern` — requires single-definition constraint across TUs
- Synthetic .trit files used (no real model data)

## K. Recommended Next Phase
Phase 28AV: Hook prt_get_residual_view() into actual llama-cli PRT matmul path behind flag — stats-only, no generation change.

## L. Models/Sidecars/F32 Refs Staged?
No. Synthetic .trit files under `/tmp` only.

## M. Secrets Detected?
None.

## N. Tags Touched?
No.