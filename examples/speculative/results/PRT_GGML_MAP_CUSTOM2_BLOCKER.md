# PRT_GGML_MAP_CUSTOM2_BLOCKER.md

## Blocker: ggml_map_custom2 Memory Corruption

**Date:** 2026-04-30
**Severity:** BLOCKING
**Phase:** 10E-7S → 10E-8, 10F

## Description

When PRT sidecar data is loaded into llama.cpp via `llama_set_prt_sidecar()` and the custom op substitution is active via `ggml_map_custom2`, the ggml computation graph produces garbage output. Decoded tokens contain path string fragments from the sidecar filename (e.g., `ffn_up_layer35_prt.bin`).

This corruption occurs regardless of what the custom op does internally — even a pure identity copy (src0→dst with verified correct floats) produces garbage output.

## Root Cause

The corruption is in the ggml_map_custom2 integration inside llama.cpp. The matmul result tensor (`src0` in the custom op) arrives at the custom op already containing corrupted data with path string fragments.

## Why This Blocks Everything

| Next Phase | Why Blocked |
|------------|-------------|
| 10E-8 Quality Canary | Cannot measure quality if output is corrupted |
| 10F Broader Benchmark | Cannot run benchmarks if output is corrupted |
| Production Integration | Output is garbage, not usable |

## Evidence

1. **Sidecar file audit:** CLEAN — correct size, no path strings, finite values
2. **Loader pointer audit:** CLEAN — all pointers within allocated range
3. **Standalone PRT compute:** CLEAN — produces correct output outside llama.cpp
4. **Custom op identity copy:** VERIFIED CORRECT — src0 and dst both show correct floats inside the op
5. **Final output:** CORRUPT — path string fragments appear in decoded tokens

## What This Is NOT

- ❌ NOT a sidecar format issue
- ❌ NOT a loader bug
- ❌ NOT a PRT algorithm bug
- ❌ NOT a custom op code bug
- ❌ NOT memcpy self-copy undefined behavior

## What This IS

**A ggml tensor lifecycle / custom op integration bug in llama.cpp.**

The interaction between:
- `llama_set_prt_sidecar()` storing sidecar pointers in `g_prt_sidecar_data[]`
- `ggml_map_custom2()` registering a custom op callback
- `ggml_graph_compute()` running the computation graph

produces memory corruption that leaks path string data into tensor buffers.

## Investigation Leads

1. The path string "ffn_up_layer35_prt.bin" (layer 35, last loaded) appears in ALL layer outputs, suggesting a shared metadata region being overwritten
2. The corruption appears only when sidecar is non-nil — identity fallback (sidecar=nil) works correctly
3. The custom op fires 432+ times per generation run — the corruption accumulates

## Resolution

**UNRESOLVED** — requires deeper ggml_map_custom2 internals investigation.

## Workaround Options

1. Disable sidecar loading (identity fallback) — works but no PRT acceleration
2. Investigate ggml tensor buffer reuse / aliasing
3. Move to a cleaner custom op implementation route (external sidecar engine)
4. Use a different backend for PRT computation (not ggml_map_custom2)

## Status

**ARCHIVED** — Do not continue active generation tests on this branch.