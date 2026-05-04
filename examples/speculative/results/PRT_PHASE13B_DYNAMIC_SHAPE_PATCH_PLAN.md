# PRT Phase 13B: Dynamic Shape Patch Plan

**Date:** 2026-05-03

## Files to Modify

1. `examples/speculative/phase10e0_layer0_replacement.cpp`
2. `examples/speculative/prt_active.h` (header constants — low priority)

## Exact Changes

### Change 1: Remove TOTAL_LAYERS hardcode

**File:** `examples/speculative/phase10e0_layer0_replacement.cpp`

**Before (line 23):**
```cpp
static const int TOTAL_LAYERS = 36;  // Phase 10E-5: all 36 layers
```

**After (replace with dynamic):**
```cpp
// Dynamic: read from llama_model at runtime
// TOTAL_LAYERS is now llama_model_n_layer(model) — see line 151
```

**Search locations that use TOTAL_LAYERS:**
- Line 153: `for (int l = 0; l < TOTAL_LAYERS; l++)`
- Line 163: `fprintf(stderr, "[PRT] Loaded %d/%d sidecars\n", loaded, TOTAL_LAYERS);`
- Line 190: `for (int l = 0; l < TOTAL_LAYERS; l++)`
- Line 259: `&& layer < TOTAL_LAYERS`
- Line 274: `layer >= TOTAL_LAYERS`

**Replacement:** All `TOTAL_LAYERS` references should use `model_n_layers` (a variable set once at model load from `llama_model_n_layer(model)`).

### Change 2: Fix hardcoded M/N in sidecar struct init

**File:** `examples/speculative/phase10e0_layer0_replacement.cpp`

**Lines 95, 112, 125, 132:** Change:
```cpp
g_sidecars[layer] = {layer, data, 2048, 11008};
```

**After:**
```cpp
// Read from the actual tensor that was loaded for this layer
// The tensor name pattern: blk.{layer}.ffn_up.weight
// We already have file size, so derive M/N from expected bytes:
// expected_bytes = M * N * sizeof(float)
// For safety, pass the actual tensor or file size
g_sidecars[layer] = {layer, data, n_embd, n_ff};  // from model/tensor metadata
```

**Implementation approach:**

The cleanest fix is to pass the actual tensor dimensions from the GGML tensor inspection. At the point where `load_all_sidecars()` is called, the model is loaded and we can inspect a sample tensor:

```cpp
// In load_all_sidecars():
// Get model layer count
int n_layer = llama_model_n_layer(model);

// For a sample tensor to get M/N:
struct ggml_context * ctx = /* get from model */;
struct ggml_tensor * sample = ggml_get_tensor(ctx, "blk.0.ffn_up.weight");
int n_ff = sample->ne[0];     // FFN dimension
int n_embd = sample->ne[1];   // Hidden dimension

// Expected bytes per sidecar:
int64_t expected_bytes = (int64_t)n_ff * n_embd * sizeof(float);

// Then use these for validation and struct init:
// g_sidecars[layer] = {layer, data, n_embd, n_ff};
```

### Change 3: Fix static zero sidecar (line 131-132)

**Before:**
```cpp
static float s_zero_sidecar[2048 * 11008] = {0};
g_sidecars[layer] = {layer, s_zero_sidecar, 2048, 11008};
```

**After (simplest fix — remove this path):**
```cpp
// If we need a static fallback, use dynamic allocation:
// static float * s_zero_sidecar = nullptr;
// This is used for load_sidecar_static() which is a DEVELOPMENT fallback only
// Most models will use load_sidecar_posix/mmap, so this path is unlikely to be hit
```

Note: The `load_sidecar_static()` path is a development-only fallback. For production use, it would need to allocate dynamically.

### Change 4: Fix bounds check (lines 259, 274)

**Before:**
```cpp
if ((sscanf(...) == 1) && layer >= 0 && layer < TOTAL_LAYERS) { ... }
if (layer < 0 || layer >= TOTAL_LAYERS) return true;
```

**After:**
```cpp
// Use the dynamic n_layer variable:
if ((sscanf(...) == 1) && layer >= 0 && layer < model_n_layers) { ... }
if (layer < 0 || layer >= model_n_layers) return true;
```

## Proposed New Struct and Init Pattern

```cpp
struct PRTConfig {
    int n_layer;       // from llama_model_n_layer(model)
    int n_embd;        // hidden dimension
    int n_ff;          // FFN dimension  
    int64_t sidecar_bytes;  // n_embd * n_ff * 4
    int * force_native; // array of layer IDs to run native
    int n_force_native;
};

// Global config (set once at model load)
static PRTConfig g_prt_cfg = {0};

// Init at model load:
void prt_init_config(struct llama_model * model) {
    g_prt_cfg.n_layer = llama_model_n_layer(model);
    // ... read tensor dims from sample ffn_up tensor
    g_prt_cfg.n_embd = n_embd;   // from tensor->ne[1]
    g_prt_cfg.n_ff = n_ff;       // from tensor->ne[0]
    g_prt_cfg.sidecar_bytes = (int64_t)g_prt_cfg.n_embd * g_prt_cfg.n_ff * 4;
}
```

## LOC Estimate

- Remove TOTAL_LAYERS constant: **-1 line**
- Replace 4 TOTAL_LAYERS uses: **+0 lines** (just variable rename)
- Replace 4 hardcoded M/N struct inits: **+4 lines** (use dynamic vars)
- Fix static zero sidecar: **+5 lines** (dynamic or guard)
- Fix 2 bounds checks: **+0 lines** (variable rename)

**Total: ~10 lines added/changed, no algorithmic changes**

## Test Plan After Patch

1. Load Qwen2.5-0.5B with `--prt-mode 5700 --prt-force-native 7,10`
2. Check: `[PRT] Loaded 24/24 sidecars` (not 36/36)
3. Run 5-prompt canary
4. Verify counters: `callback_overwrites=0`, `identity_fallback_calls=0`

## Pre-commit Safety Check

```bash
# Verify no model/sidecar binaries:
git ls-files | grep -E "\.gguf$|\.bin$|\.safetensors$" || echo "CLEAN"

# Verify no secrets:
grep -r -i "api_key\|secret\|token\|password" examples/speculative/ || echo "CLEAN"
```