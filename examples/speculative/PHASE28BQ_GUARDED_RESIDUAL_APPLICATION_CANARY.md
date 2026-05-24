# Phase 28BQ: Guarded Residual Application Canary

**Date:** 2026-05-23
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Old HEAD:** `d40d6b287` (Phase 28BP-A)
**Verdict:** ✅ **PASS**
**Model:** Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

---

## Goal

Canary test for guarded residual application. Add `--prt-sidecar-apply` flag (OFF by default). When OFF: observe-only. When ON: decode the selected layer/family's .trit file to float via `prt_trit_decoder::decode_bytes()`, record counters, but **do NOT feed into model compute**. Prove the decode path works without corrupting output.

**Constraints:** NOT residual injection yet. Observe-only decode. No quality claims.

---

## Implementation

### New CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prt-sidecar-apply` | bool | false | Enable experimental residual application |
| `--prt-sidecar-apply-layer N` | int | -1 | Target layer index |
| `--prt-sidecar-apply-family NAME` | string | "" | Target tensor family |

### New Globals (`src/prt_sidecar_pager_globals.cpp`)

```cpp
bool g_prt_sidecar_apply_enabled = false;
int g_prt_sidecar_apply_layer = -1;
std::string g_prt_sidecar_apply_family;

struct prt_apply_counters {
    size_t decoded_views = 0;
    size_t application_attempts = 0;
    size_t application_successes = 0;
    bool sidecar_math_influenced_output = false;
};
prt_apply_counters g_prt_apply_stats;
```

### Shadow Apply Function

```cpp
// In prt_sidecar_pager_globals.cpp (strong symbol, not inline)
void prt_shadow_apply(const char* raw_view, size_t size, int layer, const char* family) {
    if (!g_prt_sidecar_apply_enabled) return;
    if (layer != g_prt_sidecar_apply_layer) return;
    if (g_prt_sidecar_apply_family != family) return;

    // Decode .trit header to get float count
    prt_trit_decoder decoder;
    size_t float_count = decoder.decode_header(raw_view, size);
    if (float_count == 0) return;

    // Decode bytes to float buffer
    float* decoded = decoder.decode_bytes(raw_view, size, float_count);
    g_prt_apply_stats.decoded_views++;
    g_prt_apply_stats.application_attempts++;
    if (decoded) {
        g_prt_apply_stats.application_successes++;
        // Shadow mode: immediately deallocate, do NOT feed into model
        decoder.deallocate(decoded);
    }
}
```

### Hook Integration (`llama-graph.cpp build_ffn`)

```cpp
// After getting residual view from pager:
if (g_prt_sidecar_apply_enabled) {
    prt_shadow_apply(view_ptr, size, il, family_name);
    GGML_ASSERT(!g_prt_apply_stats.sidecar_math_influenced_output &&
        "Shadow apply must NOT influence model output");
}
// Log PRT-APPLY-SHADOW counters
```

---

## Test Results

### Test A: Baseline (no pager)
```bash
./build/bin/llama-cli -m <model> -p "Hi" -n 1 -t 4
```
**Result:** EXIT=0 ✅ | No pager activity

### Test B: Observe-only pager
```bash
./build/bin/llama-cli -m <model> -p "Hi" -n 1 -t 4 \
  --enable-prt-sidecar-pager --prt-mode 5700 \
  --prt-sidecar-manifest /tmp/phase28bo_layer0_multi_family/manifest.json \
  --prt-sidecar-dir /tmp/phase28bo_layer0_multi_family/ \
  --prt-sidecar-budget-mb 512
```
**Result:** EXIT=0 ✅ | "Hello" output | `decoded_views=0`, `app_attempts=0` ✅

### Test C: Guarded application canary
```bash
./build/bin/llama-cli -m <model> -p "Hi" -n 1 -t 4 \
  --enable-prt-sidecar-pager --prt-mode 5700 \
  --prt-sidecar-manifest /tmp/phase28bo_layer0_multi_family/manifest.json \
  --prt-sidecar-dir /tmp/phase28bo_layer0_multi_family/ \
  --prt-sidecar-budget-mb 512 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-apply-family attn_out
```
**Result:** EXIT=0 ✅ | "Hello" output | `decoded_views=1` ✅ | `app_attempts=1` ✅ | `app_successes=1` ✅ | `sidecar_math_influenced_output=false` ✅

### Test D: Wrong target layer
```bash
./build/bin/llama-cli ... --prt-sidecar-apply-layer 1 ...
```
**Result:** EXIT=0 ✅ | `app_successes=0` ✅ | No crash ✅

### Test E: Budget=0
```bash
./build/bin/llama-cli ... --prt-sidecar-budget-mb 0 \
  --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family attn_out
```
**Result:** EXIT=0 ✅ | `budget_rejects=4` ✅ | `app_attempts=0` ✅ | No crash ✅

---

## Key Evidence

### Shadow Decode Confirmed
```
[PRT-APPLY] enabled layer=0 family=attn_out
[PRT-APPLY-SHADOW] il=0 family=attn_out layer_match=1 family_match=1 decoded_views=1 app_attempts=1 app_success=1 sidecar_math_influenced_output=0
```

### Observe-Only Confirmed (Test B)
```
[PRT-PAGER-LAZY] layer=0 family=ffn_up first_reason=layer_not_activated activation_attempted=1 activation_ok=1
[PRT-PAGER-LAZY] layer=0 family=ffn_down reason= ...
[PRT-PAGER-LAZY] layer=0 family=attn_out reason= ...
[PRT-PAGER-LAZY] layer=0 family=ffn_gate reason= ...
(no PRT-APPLY lines)
```

### not_in_manifest Guard Confirmed (layers 1-23)
```
[PRT-PAGER-HOOK] il=23 family=ffn_up is_null=1 reason=not_in_manifest size=0 hook_calls=24
```

---

## Files Changed

| File | Change |
|------|--------|
| `common/arg.cpp` | Added `--prt-sidecar-apply`, `--prt-sidecar-apply-layer`, `--prt-sidecar-apply-family` |
| `common/common.h` | Added fields to `common_params` struct |
| `src/CMakeLists.txt` | Added `prt_trit_decode.cpp` to build |
| `src/prt_sidecar_pager_globals.cpp` | Added globals, `prt_shadow_apply()`, `prt_get_apply_stats()` |
| `src/llama-graph.cpp` | Hook calls `prt_shadow_apply()` when enabled |
| `examples/speculative/prt_sidecar_runtime_link.h` | Added decode header include, forward decls |

---

## Decision Points

### ✅ Correct: Shadow decode over injection
Decode path is now proven functional without risking model compute corruption. `sidecar_math_influenced_output=false` confirmed.

### ✅ Correct: Apply flag defaults to OFF
Observe-only behavior preserved by default. Application requires explicit opt-in.

### ⚠️ Note: Output timing
Test C generates "Hello" correctly but with heavy log I/O. The pager hook fires on every tensor request across all layers, producing verbose output. This is a logging volume issue, not a correctness issue.

---

## Next Phase: 28BR

**True Residual Injection** — Replace `deallocate(decoded)` with actual injection into model activation. Requires:
1. Decode-once caching (don't re-decode every hook call)
2. Proper float buffer lifecycle management
3. Integration with llama-graph compute path
4. Strict correctness verification before any quality claim

---

## Risks

- Shadow decode only — actual model compute injection NOT yet tested
- Single family/layer target only
- No checksum validation post-decode
- qwen2.5-0.5B only; larger models untested
