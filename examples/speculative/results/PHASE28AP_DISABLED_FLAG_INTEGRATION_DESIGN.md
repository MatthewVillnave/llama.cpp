# Phase 28AP: Disabled-Flag Sidecar Pager Integration Design

## Verdict: PASS_PHASE28AP_DISABLED_FLAG_DESIGN | PASS_FLAG_SCOPE_DEFINED | PASS_WIRING_POINTS_DEFINED | PASS_DEFAULT_BEHAVIOR_PRESERVED | RECOMMEND_DISABLED_STUB_WIRING

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`4ed68bba1`

---

## C. Integration Flag Design

### Primary Flag
`--enable-prt-sidecar-pager` (default: **disabled**)

### Supporting Flags
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--enable-prt-sidecar-pager` | bool | false | Master switch — enables pager |
| `--prt-sidecar-manifest` | string | "" | Path to manifest.json |
| `--prt-sidecar-budget-mb` | int | 512 | Max resident bytes in MB |
| `--prt-sidecar-policy` | string | "strict" | "strict" or "lru" |
| `--prt-sidecar-prefetch-distance` | int | 1 | Layers ahead to prefetch |
| `--prt-sidecar-window-size` | int | 4 | Active window size |
| `--prt-sidecar-lru` | bool | false | True = LRU eviction, False = strict reject |
| `--prt-sidecar-checksum` | bool | true | Validate .trit header checksums |

### CLI Parsing
- All `--prt-sidecar-*` flags are **conditionally required** only when `--enable-prt-sidecar-pager=true`
- When flag is disabled, all other flags are ignored (no manifest required, no config needed)
- Parse in the same location as existing PRT/sidecar config

---

## D. Code Wiring Points

### 1. Config Parsing
**File:** Likely `common/` params parsing or `llama.cpp` CLI adapter
**Point:** Add `--enable-prt-sidecar-pager` and supporting flags to arg parser
**Action:** Store in `llama_params` or similar struct; pass to PRT context init

### 2. Pager Initialization
**File:** `prt_shadow.h` or PRT context initialization
**Point:** After model loaded, before first layer processed
**Action:**
```cpp
if (params.enable_prt_sidecar_pager) {
    prt_sidecar_pager_config cfg;
    cfg.manifest_path = params.prt_sidecar_manifest;
    cfg.max_resident_bytes = params.prt_sidecar_budget_mb * 1024 * 1024;
    cfg.policy = params.prt_sidecar_policy;
    cfg.eviction_lru = params.prt_sidecar_lru;
    cfg.checksum_enabled = params.prt_sidecar_checksum;
    cfg.prefetch_distance = params.prt_sidecar_prefetch_distance;
    cfg.window_size = params.prt_sidecar_window_size;
    g_prt_pager = new prt_sidecar_pager(cfg);
    g_prt_pager->init();
}
```

### 3. Layer Activation
**File:** PRT layer processing loop (where residual sidecars are consumed)
**Point:** Before computing layer L — call `activate_layer(L)` then `prefetch_layer(L+prefetch_distance)`
**Action:**
```cpp
if (g_prt_pager) {
    g_prt_pager->activate_layer(current_layer);
    if (params.prt_sidecar_prefetch_distance > 0) {
        g_prt_pager->prefetch_layer(current_layer + params.prt_sidecar_prefetch_distance);
    }
}
```

### 4. Residual Lookup Replacement
**File:** `phase10b_shadow_test.cpp:178` or equivalent matmul consumption point
**Point:** Where `g_sidecars[sidecar_key]` is currently called
**Action:** Replace with:
```cpp
if (g_prt_pager) {
    auto view = g_prt_pager->get_residual(layer, tensor_family);
    if (!view.is_null) {
        // Use view.data for PRT matmul
    } else {
        // Fallback to base weight
    }
} else {
    // Legacy sidecar path unchanged
}
```

### 5. Stats Emission
**File:** After each layer or end of inference
**Action:** Log or expose via `prt_sidecar_pager_stats`
```cpp
if (g_prt_pager) {
    auto stats = g_prt_pager->get_stats();
    LLAMA_LOG("pager: resident=%zu peak=%zu reads=%zu evictions=%zu\n",
        stats.resident_bytes, stats.peak_resident_bytes,
        stats.reads, stats.evictions);
}
```

### 6. Fallback Behavior
**File:** At residual lookup point
**Action:** When `get_residual()` returns `is_null=true`:
- Log: `"residual missing, using base weight"`
- Use existing base weight path
- Increment `fallback` counter

---

## E. Disabled Behavior (Default)

When `--enable-prt-sidecar-pager=false` or not set:
- `g_prt_pager = nullptr`
- Existing `g_sidecars` map unchanged
- `prt_load_sidecars()` called normally
- No manifest required
- No new failures possible
- No stats emitted
- **Current behavior fully preserved**

---

## F. Enabled Behavior

### Initialization
1. Parse CLI flags
2. Validate manifest path exists
3. Init pager with config
4. Log: `"PRT sidecar pager enabled, manifest=<path>"`

### Per-Layer Flow
```
For layer L:
  1. activate_layer(L)
     → load residuals for L into memory
     → enforce budget (strict or LRU)
     → evict old layers if needed
  2. prefetch_layer(L + prefetch_distance)
     → async hint for future layer
  3. get_residual(L, tensor_family)
     → return prt_residual_view {data, size, is_null, reason}
     → if is_null: fallback to base weight
  4. Update stats
```

### Error Handling
| Condition | Strict Mode | LRU Mode |
|-----------|-------------|----------|
| Manifest missing | Abort with error | Abort with error |
| Budget exceeded | Return false; fallback | Evict oldest; retry |
| Checksum fail | Mark residual null; fallback | Mark residual null; fallback |
| File missing | Mark null; fallback | Mark null; fallback |

---

## G. Runtime Stats

Emitted when `--enable-prt-sidecar-pager=true`:

| Stat | Type | Description |
|------|------|-------------|
| `pager_enabled` | bool | Always true when stats emitted |
| `manifest_loaded` | bool | Whether init() succeeded |
| `resident_bytes` | size_t | Current resident memory |
| `peak_resident_bytes` | size_t | Peak resident during run |
| `reads` | size_t | Number of file reads |
| `evictions` | size_t | Number of layer evictions |
| `lru_evictions` | size_t | Evictions due to LRU policy |
| `cache_hits` | size_t | Prefetch hits (already loaded) |
| `cache_misses` | size_t | Loads done on demand |
| `fallbacks` | size_t | Get_residual returned null |
| `checksum_ok` | size_t | .trit headers validated |
| `checksum_fail` | size_t | Failed checksum validations |
| `budget_rejects` | size_t | Activate rejected due to budget |

---

## H. Minimal Implementation Phase

**Recommended: Phase 28AQ — Disabled Stub Wiring**

Scope:
1. Add CLI `--enable-prt-sidecar-pager` flag to common/llama.cpp params
2. Add supporting `--prt-sidecar-*` flags (even if unused initially)
3. Instantiate `prt_sidecar_pager` only when flag enabled
4. Smoke test: disabled mode preserves existing behavior
5. No generation required
6. No active PRT math changes

**Alternative: Phase 28AQ — Harness Only**
- Integrate into `examples/speculative/` test harness only
- Not in `llama-cli` main binary
- Faster to validate without full model load

---

## I. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Accidentally changing default sidecar behavior | HIGH | Flag defaults to false; existing path untouched |
| Manifest missing causing runtime failure | MEDIUM | Require explicit `--enable` to activate pager |
| get_residual lifetime mismatch | HIGH | View valid until `evict_layer()` or `shutdown()` |
| Pager view invalid after eviction | HIGH | Caller must check `is_null` before each use |
| Duplicate ownership (g_sidecars + pager) | MEDIUM | Use pager ONLY when enabled; never double-load |
| Thread safety in concurrent prefetch | LOW | Prefetch is read-only; activation is sequential |
| Stats overhead | LOW | Only when `--enable-prt-sidecar-pager=true` |
| Silent fallback hiding errors | MEDIUM | Log every fallback with reason string |

---

## J. Recommended Next Phase

**Phase 28AQ: Disabled Stub Wiring**

- Add CLI flags to common params
- Instantiate pager when enabled
- Smoke test: verify disabled = no behavior change
- Enabled test: load fake manifest, verify activate/get_residual works
- No generation, no model load required

---

## K. Safety Scan

```
No .gguf/.bin/.safetensors/.pt/.pth files staged ✅
No 30B files accessed ✅
No sidecars generated ✅
No model data staged ✅
No secrets detected ✅
No tags touched ✅
No code changes (design only) ✅
```

**Safety verdict:** CLEAN — no changes made, design only.

---

## Verdict Flags
- PASS_PHASE28AP_DISABLED_FLAG_DESIGN
- PASS_FLAG_SCOPE_DEFINED
- PASS_WIRING_POINTS_DEFINED
- PASS_DEFAULT_BEHAVIOR_PRESERVED
- RECOMMEND_DISABLED_STUB_WIRING
- PASS_NO_CHANGES_MADE

---

## Tags Touched?
NO.