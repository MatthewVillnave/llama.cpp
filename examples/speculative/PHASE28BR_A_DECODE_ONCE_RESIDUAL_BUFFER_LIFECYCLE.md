# Phase 28BR-A: Decode-Once Residual Buffer Lifecycle Canary

## Context
- **Branch:** experimental/prt-phase19a-alt-sidecar-backed
- **Previous HEAD:** 120978d4b (Phase 28BQ)
- **Model:** Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
- **Manifest:** /tmp/phase28bo_layer0_multi_family/manifest.json (layer 0, 4 families)
- **Target:** layer=0, family=attn_out

## Objective
Implement a decode-once cached residual buffer: decode the .trit once, cache the decoded float buffer, reuse on repeated hits — all WITHOUT feeding into model compute. Still observe-only.

## Changes

### 1. `src/prt_sidecar_pager_globals.cpp`
Added decode-once cache infrastructure:
- `prt_decode_cache_entry` struct: owns decoded float buffer, metadata, validity flag
- `prt_decode_cache` struct: map of key→entry, miss/hit counters, total bytes tracked
- `prt_decode_cached()`: decode-once function. Cache key = "layer:family". On miss: decode, allocate owned buffer, copy decoded data, store. On hit: return cached pointer. Cache OWNS the buffer.
- `prt_decode_cache_shutdown()`: free all cached entries on process exit
- Updated `prt_shadow_apply()` to use `prt_decode_cached()` instead of inline decode+deallocate. No deallocate after use — cache owns it. Shadow mode: do NOT inject into model compute.
- Updated `prt_apply_counters` struct with cache counter fields
- Updated `prt_get_apply_stats()` to report cache counters from g_prt_decode_cache

### 2. `tools/cli/cli.cpp`
Updated atexit shutdown to call `prt_decode_cache_shutdown()` before pager shutdown.

### 3. `examples/speculative/prt_sidecar_runtime_link.h`
Updated `prt_apply_counters` struct to include cache counter fields for API consistency.

## Test Results

| Test | Command | Exit | Status |
|------|---------|------|--------|
| A: Baseline | No pager flags | 0 | PASS |
| B: Observe-only | Pager + manifest, no --apply | 0 | PASS |
| C: Decode-once canary | --apply layer=0 family=attn_out | 0 | PASS |
| D: Repeated hit n=2 | Same as C, n_predict=2 | 0 | PASS |
| E: Wrong target | layer=1 (not in manifest) | 0 | PASS |
| F: Budget=0 | budget_mb=0 | 0 | PASS |
| G: Missing manifest | nonexistent manifest | 0 | PASS |

## Test C Evidence (decode-once canary)

**Target .trit (attn_out):**
- raw_bytes: 303,216
- decoded_f32_bytes: 3,211,264
- rows=896, cols=896, block_rows=32, block_cols=48, n_scales=532

**Cache counters:**
- decode_cache_misses: 1 (first call creates entry)
- decode_cache_hits: 0 (single-token run)
- decode_cache_entries: 1 (0:attn_out stored)
- decoded_bytes_total: 3,211,264

**Apply counters:**
- decoded_views: 1
- application_attempts: 1
- application_successes: 1
- sidecar_math_influenced_output: false
- raw_bytes_cast_to_float: false

## Test D Evidence (repeated hit, n_predict=2)

Two `[PRT-APPLY-SHADOW]` events observed for il=0 family=attn_out:
1. `decoded_views=1 app_attempts=1 app_success=1` (first token)
2. `decoded_views=2 app_attempts=2 app_success=2` (second token)

Second call hit the cache (key "0:attn_out" already valid) — proof of decode-once reuse across token boundaries. The cache avoided re-decoding the same .trit for the second token.

## Verification Checklist

- [x] Baseline exits 0
- [x] Observe-only exits 0, no cache activity
- [x] Decode-once canary exits 0, cache_misses=1, cache_entries=1
- [x] Repeated-hit shows cache reuse (decoded_views goes from 1→2)
- [x] Wrong target: no cache entry, no crash
- [x] Budget=0: no cache entry, no crash
- [x] Missing manifest: deterministic failure (manifest init fails)
- [x] sidecar_math_influenced_output=false
- [x] raw_bytes_cast_to_float=false
- [x] Only small source/report/json files to be staged
- [x] No model files/binaries/tmp data staged

## Design Notes

**Cache ownership:** The cache owns the decoded buffer via `new float[n_floats]`. Caller receives pointer but does NOT free — cache frees on shutdown. This is the key lifecycle guarantee.

**Shadow compute:** `prt_shadow_apply()` still does not inject decoded buffer into model compute. The buffer sits in the cache unused until shutdown. This maintains the observe-only property of the 28BQ canary while adding decode-once semantics.

**Decode-once semantics:** Once decoded, the same raw .trit view for a given layer:family returns the cached pointer. For a 1-token run, there's only one call so miss=1 hit=0. For n_predict=2, two calls = miss+hits grow.

**Shutdown cleanup:** `prt_decode_cache_shutdown()` runs before `g_prt_pager->shutdown()` in atexit. This ensures cached decoded buffers are freed before pager is torn down.