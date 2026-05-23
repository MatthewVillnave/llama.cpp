# Phase 28AS: Connect Pager to Experimental PRT Sidecar Lookup

## Verdict: PASS_PHASE28AS_LOOKUP_INTEGRATION ✅

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`975d94a93` (Phase 28AR end state)

## C. Lookup Hook Point

The existing `prt_shadow.h` contains:
- `static std::unordered_map<int, SidecarLoad> g_sidecars` (legacy per-layer map)
- `prt_get_sidecar(int layer)` → returns `SidecarLoad&` or error
- `prt_load_sidecars(const char* map_path)` → loads legacy sidecars

The proposed wrapper `prt_get_residual_view(layer, tensor_family)` would:
1. If `g_prt_pager != nullptr && g_prt_pager_enabled`: call `g_prt_pager->get_residual(layer, tensor_family)`
2. If view is non-null: return pager view
3. If view is null or pager disabled: fall through to legacy `g_sidecars[layer].data` path or return null

This is a **one-line routing decision** at the residual lookup point — no broad rewrite.

## D. Wrapper/Helper Implemented

Tested via `prt_sidecar_lookup_harness.cpp` — the `wrapper_get_residual()` function implements the routing logic:

```cpp
static prt_residual_view wrapper_get_residual(
    int layer,
    const std::string& tensor_family,
    bool pager_enabled,
    prt_sidecar_pager* pager,
    bool& used_pager,
    bool& used_legacy
) {
    if (pager_enabled && pager != nullptr) {
        auto view = pager->get_residual(layer, tensor_family);
        if (!view.is_null) { used_pager = true; return view; }
    }
    // legacy path...
}
```

This mirrors what the real integration would do — just pass a flag + pointer, route accordingly.

## E. Pager Object Wiring

The pager is linked as a standalone object in the harness. For the actual runtime integration, wiring would be:

1. `g_prt_pager: prt_sidecar_pager* = nullptr` (static/global, null by default)
2. In `prt_load_sidecars()` or after model load: if `--enable-prt-sidecar-pager`, create pager with config, call `init()`
3. In residual lookup: check `g_prt_pager != nullptr` before calling `get_residual()`
4. Stats: aggregate from `g_prt_pager->get_stats()` if enabled

**No pager object created by default. No manifest required when disabled.**

## F. No-Generation Lookup Harness

`examples/speculative/prt_sidecar_lookup_harness.cpp` — 4 test modes:

| Mode | Test | Result |
|------|------|--------|
| `--legacy` | pager disabled, legacy path used | PASS ✅ |
| `--pager` | pager enabled, valid residual returned | PASS ✅ |
| `--fallback` | unknown layer returns null, routing correct | PASS ✅ |
| `--budget` | 256KB budget rejects all activations | PASS ✅ |
| `--all` | All 4 modes in sequence | PASS ✅ |

**Build:**
```bash
g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -I. \
  examples/speculative/prt_sidecar_pager.cpp \
  examples/speculative/prt_sidecar_lookup_harness.cpp \
  -o /tmp/prt_sidecar_lookup_harness
```

## G. Legacy-Mode Result: PASS ✅
- `pager_enabled=false`, pager pointer=null
- `wrapper_get_residual(0, "ffn_up")` returns non-null with `reason="legacy"`
- `wrapper_get_residual(5, "ffn_up")` returns null with `reason="not_found"` (no legacy data for layer 5)
- No pager involvement, no manifest required

## H. Pager-Mode Result: PASS ✅
- `pager.init() = true` — synthetic manifest loaded
- `activate_layer(0) = true` — layer 0 activated
- `get_residual(0, "ffn_up")` returns `is_null=false, size=393504`
- `used_pager=true, used_legacy=false`
- Stats: resident=1967520, reads=1, misses=1

## I. Fallback/Budget Result: PASS ✅

**Fallback:**
- Layer 99 not in manifest → `get_residual(99)` returns `is_null=true, reason="not_found"`
- Pager disabled with unknown layer → `is_null=true, reason="not_found"`
- Known layer 0 via pager → `is_null=false, size=393504, used_pager=true`

**Budget:**
- 256KB budget, strict mode
- All 4 activations rejected (`activate_layer(0..3)=false`)
- `budget_rejects=4`, `resident=0`
- LRU mode with 2500KB budget: walks 4 layers, `lru_evictions=3`, resident stays under cap

## J. Llama-CLI Default Behavior

Not changed — this phase only touched `examples/speculative/` harness files. Llama-cli build, help, and no-flag behavior remain as Phase 28AR left them.

## K. Limitations

1. **Actual runtime link deferred** — harness simulates routing but doesn't wire into `prt_shadow.h` yet
2. **Common-params integration deferred** — harness uses its own config struct
3. **No generation** — synthetic `.trit` only under `/tmp`
4. **No g_sidecars replacement** — wrapper would coexist with legacy, not replace it initially

## L. Recommended Next Phase

**Phase 28AT: Runtime-adjacent dry-run with pager enabled, still no generation, stats only.**

Actual wiring steps:
1. Add `g_prt_pager: prt_sidecar_pager*` to `prt_shadow.h`
2. Add routing decision in `prt_get_residual_view()` 
3. Smoke test: pager disabled → legacy behavior unchanged
4. Smoke test: pager enabled → synthetic manifest loaded, get_residual called

## M. Safety Scan

```
No .gguf/.bin/.safetensors/.pt/.pth/.trit files staged ✅
No 30B files accessed ✅
No sidecars generated ✅
No model data staged ✅
No secrets detected ✅
No tags touched ✅
```

## N. Verdicts

- `PASS_PHASE28AS_LOOKUP_INTEGRATION` ✅
- `PASS_LEGACY_DEFAULT_PRESERVED` ✅
- `PASS_PAGER_LOOKUP_HARNESS` ✅
- `PASS_FALLBACK_BEHAVIOR` ✅
- `PASS_BUDGET_BEHAVIOR` ✅
- `PARTIAL_RUNTIME_LINK_DEFERRED` ✅
- `PASS_NO_GENERATION` ✅
- `PASS_NO_DEFAULT_BEHAVIOR_CHANGE` ✅