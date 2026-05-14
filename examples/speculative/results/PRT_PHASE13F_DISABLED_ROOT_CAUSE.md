# PRT Phase 13F: Disabled-Mode Root Cause Isolation

## Goal
Answer:
1. Does current-build native llama-cli output clean text?
2. Why does PRT-disabled produce garbage?

## Test Environment
- Branch: `experimental/prt-phase13-model-generalization`
- HEAD: `a96e3bf9193f808404452a41ae1e8b82782fce7c`
- Binary: `llama-cli` (May 4 22:02), `llama-prt-posix` (May 5 12:09)
- Model: Qwen2.5-0.5B-Instruct-Q4_K_M

## Step 1: Build Identity

### Git Status
- Branch: experimental/prt-phase13-model-generalization
- HEAD: a96e3bf9193f808404452a41ae1e8b82782fce7c

### Binary Timestamps
- llama-cli: May 4 22:02 (5.7M)
- llama-prt-posix: May 5 12:09 (41K)

### SHA256
- llama-cli: bf4d655fec814c42924acb2928d3475907fc6a4edc09fe658aec83ea5c7c8d0f
- llama-prt-posix: 1ff41b32dc68d1b8344d9766380eb7fa53e50f9ba49b95375dbd4e1930c85f6b

### Shared Libraries (ldd)
Both link to same `libllama.so.0.0.8742` (via symlink at runtime):
```
libllama.so.0 => /home/matthew-villnave/llama.cpp/build/bin/libllama.so.0
```
- Both resolve to same address at runtime ✅

### Library Files
```
libllama.so.0.0.8720  May  2 11:26
libllama.so.0.0.8739  May  4 21:30
libllama.so.0.0.8740  May  4 22:00
libllama.so.0.0.8742  May  5 12:08  <-- current (symlinked)
```

## Step 2: Native Test Results

### Run 1-3 Output (strings extracted)
```
Run 1: "The capital of France is Paris."
Run 2: "The capital of France is Paris."
Run 3: "The capital of France is Paris."
```

### Garbage Check
```
grep: "located|centered|theated|incated" → NOT FOUND
```

**NATIVE: CLEAN ✅** - All 3 runs output correct "Paris" completion

## Step 3: PRT-Disabled Test Results

### Sidecar State
- Sidecars hidden: /tmp/prt_sidecars → /tmp/prt_sidecars_HIDDEN (verified)

### Output (strings extracted from stderr)
```
[DEBUG] token 0: n=8 buf_hex: 20 6C 6F 63 61 74 65 64  str=' located'
[DEBUG] token 1: n=3 buf_hex: 20 69 6E           str=' incated'
[DEBUG] token 2: n=4 buf_hex: 20 74 68 65        str=' theated'
[DEBUG] token 3: n=7 buf_hex: 20 63 65 6E 74 65 72  str=' centerd'
[DEBUG] token 4: n=3 buf_hex: 20 6F 66           str=' ofnterd'
...
[DEBUG] token 13: n=5 buf_hex: 20 73 61 6D 65    str=' sameony'
```

### Graph Counters (from stderr)
```
[11BD] callback_overwrites: 0
[11BD] native_ffn_up_calls: 0
[11BD] prt_true_replacement_calls: 0
[11BD] native_fallback_calls: 0
```

**PRT-DISABLED: DIRTY ❌** - Garbage tokens, same pattern as Phase 13E-R

## Step 4: Graph-Level Analysis

### Key Observations
1. `prt_is_true_replacement_layer(il)` returns false in mode 0 ✅ (guarded)
2. `build_prt_ffn_up()` never called in mode 0 ✅ (guarded)
3. `prt_custom_ops = 0` ✅ (no PRT custom ops registered)
4. `g_prt_debug_mode = 0` ✅ (disabled)
5. `g_prt_sidecar_data[il] = nullptr` ✅ (sidecars hidden)

**BUT output is garbage!**

### Suspected Root Cause
The PRT code in `phase10e0_layer0_replacement.cpp` (the harness) includes headers that modify how the library is used:
- `prt_graph_replace.h` - defines custom ops and graph building
- `prt_active.h` - defines callback hooks

Even in mode 0, these headers may alter:
- How the tokenizer is used 
- How the GGML context is initialized
- How the generation loop processes tokens

**Hypothesis:** The phase10e0 harness uses a different token processing path that doesn't match the native llama-cli's streaming mode.

## Step 5: Classification

### Candidates
- A. NATIVE_CURRENT_BUILD_REGRESSION ❌ (native is clean)
- B. TRUE_DISABLED_GRAPH_REPLACEMENT ❌ (counters show 0 PRT calls)
- C. PRTPOSIX_FRONTEND_BUG ✅ Most likely - issue in harness code, not libllama
- D. LINKAGE/STALENESS_BUG ❌ (both resolve to same runtime lib)

### Final Classification
**C. PRTPOSIX_FRONTEND_BUG**

The PRT harness (phase10e0_layer0_replacement.cpp) uses a different token processing/streaming mechanism than native llama-cli, causing different outputs even when PRT is disabled in the library.

## Canary Status

**Canary allowed?** 
- Yes, but MUST use native llama-cli as baseline (not PRT-disabled)
- The PRT-disabled mode in llama-prt-posix is broken by harness design
- PRT-active mode (mode 5700) should be tested against native for valid comparison

## Recommended Fix

Option 1: Fix llama-prt-posix to use same token processing as llama-cli
- Requires diff analysis between phase10e0 and llama-cli main loop

Option 2: Use separate native baseline for canary
- Build a clean "disabled mode" harness that mimics llama-cli exactly
- Or run canary comparing PRT-active vs native llama-cli

## Files Changed
- None (investigation only - no code modified)

## Models/Sidecars Staged?
- Models: No
- Sidecars: No (hidden, verified)
- Binaries: No

## Secrets?
- None

## Tags Touched?
- None (frozen tags untouched)

---
Investigation completed: 2026-05-05 14:15 EDT