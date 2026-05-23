# Phase 28AV: PRT Matmul Lookup Pager Hook

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`55300e763` (Phase 28AU)

## C. Lookup Path Identified
**File:** `examples/speculative/prt_shadow.h`
**Function:** `run_shadow_test()` → `g_sidecars.find(layer)` → `matmul_prt_3plane()`
**Lookup expression:**
```cpp
if (!g_sidecars_loaded || g_sidecars.find(layer) == g_sidecars.end()) {
    r.has_sidecar = false;
    return r;
}
```
**Current behavior:** Returns `has_sidecar=false` if sidecar missing.
**Proposed hook:** `prt_get_residual_view(layer_idx, tensor_family)` routes to pager or legacy.

## D. Wrapper Integration
`prt_get_residual_view()` from `prt_sidecar_runtime_link.h`:
```cpp
prt_residual_view prt_get_residual_view(int layer_idx, const std::string& tensor_family) {
    if (g_prt_pager != nullptr && g_prt_pager_enabled) {
        view = g_prt_pager->get_residual(layer_idx, tensor_family);
        if (!view.is_null) return view;  // pager hit
    }
    // Legacy fallback
    auto it = g_sidecars.find(layer_idx);
    if (it != g_sidecars.end()) {
        return {reinterpret_cast<uint8_t*>(it->second.data), it->second.size, false, "legacy"};
    }
    return {nullptr, 0, true, "not_found"};
}
```
**Compile guard:** `PRT_SIDECAR_PAGER_EXPERIMENTAL` — only active when flag set.

## E. Harness Path
`examples/speculative/prt_matmul_sidecar_lookup_harness.cpp` — 5 test modes (--legacy, --pager, --fallback, --budget, --stats)

## F. Legacy Disabled Result
```
g_sidecars.size=1
prt_get_residual_view(0, ffn_up): is_null=false reason=legacy size=2048 ✅
prt_get_residual_view(99, ffn_up): is_null=true reason=not_found ✅
Result: PASS_LEGACY_DISABLED ✅
```

## G. Pager Enabled Result
```
prt_init_pager() = true
Pager stats: resident=0 reads=0
prt_get_residual_view(0, ffn_up): is_null=false size=393504 reason= ✅
Result: PASS_PAGER_ENABLED ✅
```

## H. Fallback/Missing Result
```
unknown_family → falls back to legacy ✅
unknown layer → is_null=true reason=not_found ✅
Result: PASS_FALLBACK ✅
```

## I. Budget Reject Result
```
activate_layer(0) with 256KB budget: REJECTED ✅
Fallback to legacy after reject ✅
budget_rejects=1 ✅
Result: PASS_BUDGET_REJECT ✅
```

## J. Build/Default Behavior
- Build: `g++ -O2 -std=c++17 -DPRT_SIDECAR_PAGER_EXPERIMENTAL -I. -Iggml/include -Iinclude prt_sidecar_pager.cpp prt_matmul_sidecar_lookup_harness.cpp -o harness`
- 5/5 tests PASS
- **llama-cli default preserved**: no behavior change without flag

## K. Limitations
- `prt_shadow.h` modified: `ggml.h/llama.h/common.h` removed for standalone; full build needed for actual integration
- Synthetic .trit files (no real model data)
- `g_sidecars` temporarily changed to `extern` — single-TU constraint

## L. Recommended Next Phase
Phase 28AW: Connect `prt_get_residual_view()` into actual `run_shadow_test()` caller behind flag — stats-only, no generation change.

## M. Models/Sidecars/F32 Refs Staged?
No. Synthetic `.trit` files under `/tmp` only.

## N. Secrets Detected?
None.

## O. Tags Touched?
No.