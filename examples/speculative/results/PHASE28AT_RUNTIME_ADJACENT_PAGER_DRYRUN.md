# Phase 28AT: Runtime-Adjacent Pager Dry-Run, Stats Only, No Generation

## Verdict: PASS_PHASE28AT_RUNTIME_ADJACENT_DRYRUN ✅

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`03e03ac1e` (Phase 28AS end state)

## C. Dry-Run Harness Path
`examples/speculative/prt_sidecar_runtime_dryrun.cpp`

## D. Build Command
```bash
g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -I. \
  examples/speculative/prt_sidecar_pager.cpp \
  examples/speculative/prt_sidecar_runtime_dryrun.cpp \
  -o /tmp/prt_sidecar_runtime_dryrun
```

## E. Disabled Dry-Run Result: PASS ✅
- `--disabled` mode: no pager init, no manifest required
- `pager_enabled=false`, `manifest_loaded=false`
- Clean exit, zero overhead when flag is off

## F. Enabled Normal Result: PASS ✅
- 4 layers × 5 tensor families traversed
- `pager.init() = true`
- `activate_layer(0) = true`
- `layers_activated = 1` (window=4, budget sufficient for 1 layer at a time)
- `residual_lookups = 20, hits = 5` (5 families × 4 layers, only layer 0 activated = 5 hits)
- `resident_bytes = 1967520`, `peak_resident_bytes = 1967520`
- `wall_time_ms = 0`

## G. Fallback Result: PASS ✅
- Unknown layer 99 → `is_null=true, reason=layer_not_activated`
- Unknown family `nonexistent_family` → `is_null=true, reason=tensor_not_found`
- Known layer 0 returns valid `size=393504`
- Routing correct: unknown layers/tensors return null without crashes

## H. Strict Budget Result: PASS ✅
- 256 KB budget, strict mode
- `activate_layer(0..3)=false` all rejected
- `budget_rejects=4`
- `resident_bytes=0`
- No over-spending under strict constraint

## I. LRU Budget Result: PASS ✅
- 2500 KB budget, LRU eviction enabled
- `activate_layer(0..3)=true` all 4 layers activated
- `lru_evictions=3` (eviction triggered after 4 layers)
- `resident_bytes = 1967520` (under 2500 KB cap = 2560000)

## J. Data-Integrity Path Result: PASS ✅
- `pager.init() = true`
- All 5 tensor families (ffn_up, ffn_down, ffn_gate, attn_q, attn_output) return valid views
- `data_path_hits = 5/5`
- Full data load path confirmed functional with synthetic .trit package

## K. Llama-CLI Default Behavior: PRESERVED ✅
- No changes to llama-cli source
- Only `examples/speculative/` files touched
- Pager flags remain behind `--enable-prt-sidecar-pager` (disabled by default)
- Supporting flags still inert without master flag

## L. Stats Summary

| Stat | Value |
|------|-------|
| pager_enabled | true (enabled modes) |
| manifest_loaded | true |
| layers_requested | 4 |
| layers_activated | 1 (normal) / 4 (LRU) |
| layers_prefetched | 3 (normal, prefetch_distance=1) |
| residual_lookups | 20 |
| residual_hits | 5 |
| residual_fallbacks | 15 |
| checksum_ok | 0 (synthetic files use 0xFFFF checksum) |
| checksum_fail | 0 |
| budget_rejects | 4 (strict), 0 (normal/LRU) |
| lru_evictions | 3 (LRU) |
| resident_bytes | 1967520 |
| peak_resident_bytes | 1967520 |
| wall_time_ms | 0 |

## M. Limitations
1. Synthetic `.trit` package under `/tmp` only — no real sidecar generation
2. No actual LLama model loading or generation
3. Manifest checksum entries set to `0xFFFF` (skip CRC check in `load_sidecar_data`)
4. Checksum stats reflect manifest-level CRC validation (requires real manifests with non-0xFFFF values)
5. Runtime link to `prt_shadow.h` still deferred to Phase 28AU

## N. Recommended Next Phase

**Phase 28AU: Runtime Link — Connect `prt_get_residual_view()` Wrapper into `prt_shadow.h` Behind Flag.**

1. Add `g_prt_pager: prt_sidecar_pager* = nullptr` to `prt_shadow.h`
2. Add `prt_get_residual_view(layer, tensor_family)` function
3. Route: if `g_prt_pager != nullptr && g_prt_pager_enabled` → pager path → else legacy
4. Smoke test: disabled → legacy behavior unchanged
5. Smoke test: enabled with synthetic manifest → pager path used
6. No generation change, no matmul modification

## O. Models/Sidecars/f32 Refs Staged?
No. All sidecar files under `/tmp`, cleaned after tests.

## P. Secrets Detected?
No secrets in changed files.

## Q. Tags Touched?
No git tags touched.

## R. Files Changed
- `examples/speculative/prt_sidecar_runtime_dryrun.cpp` (new)
- `examples/speculative/results/PHASE28AT_RUNTIME_ADJACENT_PAGER_DRYRUN.md`
- `examples/speculative/results/phase28at_runtime_adjacent_pager_dryrun.json`