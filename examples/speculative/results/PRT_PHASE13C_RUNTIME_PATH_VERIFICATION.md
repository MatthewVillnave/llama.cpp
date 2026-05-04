# PRT Phase 13C: Runtime Path Verification

**Date:** 2026-05-03  
**Branch:** `experimental/prt-phase13-model-generalization`  
**Commit:** `f108192b6`

---

## Runtime Path: `--prt-mode 5700`

### Step 1: `--prt-mode 5700` sets debug mode

**File:** `src/llama.cpp` lines 1276-1281  
```cpp
extern "C" LLAMA_API void llama_set_prt_debug_mode(int mode);
void llama_set_prt_debug_mode(int mode) {
    g_prt_debug_mode = mode;
}
```

### Step 2: Mode 5700 activates Route A in build_ffn

**File:** `src/llama-graph.cpp` line 46 (include):
```cpp
#include "../examples/speculative/prt_graph_replace.h"
```

**File:** `src/llama-graph.cpp` line 1107:
```cpp
bool prt_layer = prt_is_true_replacement_layer(il);
```
`prt_is_true_replacement_layer()` checks `g_prt_debug_mode >= 5700` → true for mode 5700.

### Step 3: If PRT layer and sidecar exists, call build_prt_ffn_up

**File:** `src/llama-graph.cpp` lines 1122-1123:
```cpp
} else if (up && prt_layer && g_prt_sidecar_data[il]) {
    ggml_tensor * prt_result = build_prt_ffn_up(ctx0, cur, il);
```

### Step 4: build_prt_ffn_up uses g_prt_sidecar_M/N

**File:** `examples/speculative/prt_graph_replace.h` lines 175-187:
```cpp
if (layer_id < 0 || layer_id >= 36) return nullptr;
if (!g_prt_sidecar_data[layer_id]) return nullptr;

int hidden = g_prt_sidecar_M[layer_id];   // READS FROM g_prt_sidecar_M[]
int ffn    = g_prt_sidecar_N[layer_id];   // READS FROM g_prt_sidecar_N[]
int n_tokens = (int)cur->ne[1];

PRTUserData * ud = &g_prt_ud_pool[layer_id];
ud->sidecar  = g_prt_sidecar_data[layer_id];
ud->M        = hidden;   // SETS ud->M
ud->N        = ffn;      // SETS ud->N
```

The custom op kernel reads `ud->M` and `ud->N` at compute time.

### Step 5: Sidecars registered via llama_set_prt_sidecar

**File:** `src/llama.cpp` lines 1265-1272:
```cpp
void llama_set_prt_sidecar(int layer, const float * data, int M, int N) {
    if (layer >= 0 && layer < 36) {
        g_prt_sidecar_data[layer] = data;
        g_prt_sidecar_bytes[layer] = (size_t)M * N * sizeof(float);
        g_prt_sidecar_M[layer] = M;    // ← DYNAMIC VALUE STORED HERE
        g_prt_sidecar_N[layer] = N;    // ← DYNAMIC VALUE STORED HERE
    }
}
```

### Step 6: llama_set_prt_sidecar called by phase10e0_layer0_replacement.cpp

**File:** `examples/speculative/phase10e0_layer0_replacement.cpp` line 199:
```cpp
llama_set_prt_sidecar(l, it->second.data, it->second.M, it->second.N);
```

`it->second.M` and `it->second.N` are set from `g_prt_M` and `g_prt_N` (the dynamic values).

### Step 7: g_prt_M/g_prt_N set by Phase 13B patch

**File:** `examples/speculative/phase10e0_layer0_replacement.cpp` lines 165-193:
```cpp
g_n_layer = llama_model_n_layer(model);  // 24 for 0.5B
// ...
g_prt_M = 896;  // derived from file size
g_prt_N = 4864; // derived from file size
fprintf(stderr, "[PRT] Dynamic shape: n_layer=%d, M=%d, N=%d\n",
        g_n_layer, g_prt_M, g_prt_N);
```

---

## Verified: Dynamic Shape Flow

```
llama_model_n_layer(model) → g_n_layer (24)
file_size heuristic        → g_prt_M (896), g_prt_N (4864)
Sidecar struct init         → g_sidecars[layer] = {layer, data, 896, 4864}
llama_set_prt_sidecar      → g_prt_sidecar_M[layer] = 896
                           → g_prt_sidecar_N[layer] = 4864
build_prt_ffn_up           → reads g_prt_sidecar_M[layer] = 896
                           → reads g_prt_sidecar_N[layer] = 4864
Custom op kernel           → uses ud->M=896, ud->N=4864
```

---

## Hardcoded Values in Active Path (RESIDUAL)

| Location | Hardcode | Risk | Status |
|----------|----------|------|--------|
| `prt_graph_replace.h:15` | `int M; // hidden = 2048` | LOW (comment only) | NEEDS FIX |
| `prt_graph_replace.h:16` | `int N; // ffn = 11008` | LOW (comment only) | NEEDS FIX |
| `prt_graph_replace.h:22` | `g_prt_ud_pool[36]` | MED — hardcoded pool size | NEEDS FIX |
| `prt_graph_replace.h:41` | `int hidden = ud->M; // 2048` | LOW (comment only) | NEEDS FIX |
| `prt_graph_replace.h:42` | `int ffn = ud->N; // 11008` | LOW (comment only) | NEEDS FIX |
| `prt_graph_replace.h:175` | `layer_id >= 36` | MED — bounds check uses 36 | NEEDS FIX |
| `llama-graph.cpp:39` | `g_prt_force_native_layer[36]` | MED — array size 36 | NEEDS FIX |
| `src/llama.cpp:1265` | `layer >= 0 && layer < 36` | MED — bounds check 36 | NEEDS FIX |
| `src/llama.cpp:1266` | `(size_t)M * N * sizeof(float)` | NONE — just byte calc | OK |

**But critically:** The actual COMPUTATION uses `g_prt_sidecar_M[layer]` and `g_prt_sidecar_N[layer]` — which are set by the dynamic Phase 13B path. So the computation is correct even though some bounds checks still use 36.

---

## Phase 13B Test Result (from Phase 13B)

```
Model: n_layer=24, n_embd=896, n_ff=4864
[PRT] Dynamic shape: n_layer=24, M=896, N=4864
[PRT] Sidecar set: layer=0 M=896 N=4864 ptr=... bytes=17432576  ← CORRECT!
[PRT] Loaded 24/24 sidecars
[PRT-11BP] Required PRT layers: 22  ← 24 - 2 force-native
[11BD] prt_true_replacement_calls: 374
[11BD] native_fallback_calls: 16  ← 8*2 for layers 11+15 (prompt+gen)
```

**Verdict:** Phase 13B patch IS on the active path. Dynamic M/N correctly flows from `phase10e0_layer0_replacement.cpp` → `llama_set_prt_sidecar()` → `g_prt_sidecar_M/N[]` → `build_prt_ffn_up()` → custom op kernel.

---

## Residual Issues for Future Fixes

1. **Bounds checks still use 36** — `g_prt_force_native_layer[36]`, `layer < 36` checks, `g_prt_ud_pool[36]` — these are safe for now (0.5B has 24 layers, well under 36) but must be fixed before claiming full model generalization
2. **Comments say 2048/11008** — misleading but non-functional
3. **File size heuristic** — works for known models but a proper fix would read from GGML tensor directly at ctx init time

**These residuals do NOT block the 0.5B canary** — they are latent bugs for models with >36 layers.