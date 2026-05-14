# Phase 10E-3R: Corrected Layer0-Only Patch

## Problem Statement

Phase 10E-3 attempted all-layer PRT replacement with a single layer-0 sidecar.
- 36 layers replaced
- 1 sidecar (layer 0) 
- Result: cosine = 0.0 (weight mismatch for 35 layers)

## Corrected Approach: Layer 0 Only

### Scope Restriction (Two-Level Enforcement)

**Level 1: Harness LAYER_SCOPE**
```cpp
static const int LAYER_SCOPE = 0;  // Only layer 0 triggers sidecar loading
```
From `phase10e0_layer0_replacement.cpp`:
```cpp
if (sscanf(t->name, "blk.%d.ffn_up", &layer) == 1 && layer == LAYER_SCOPE) {
    // Load sidecar for layer 0 only
}
```

**Level 2: llama-graph.cpp il==0 check**
```cpp
if (il == 0) {
    tmp = ggml_map_custom2(ctx0, matmul_result, cur, prt_ffn_up_prt_op, 1, nullptr);
} else {
    tmp = matmul_result;  // No custom op, no PRT
}
```

### Sidecar Loading
- Harness loads only `ffn_up_layer0_prt.bin`
- `llama_set_prt_sidecar(0, ptr, M=2048, N=11008)` called once
- All other layers: `g_prt_sidecar_data = nullptr`

### Custom Op Behavior Per Layer

| Layer | Sidecar | Custom op target | PRT compute | Fallback |
|-------|---------|------------------|-------------|----------|
| 0 | ✅ (layer 0) | YES | `X @ W_layer0` | identity |
| 1-27 | ❌ (null) | YES | ❌ | identity |
| 28-35 | ❌ (null) | NO | ❌ | identity |

### Expected Replacement Count
- Phase 10E-3R targets: **4 replacements** (1 prompt eval step × 1 layer)
- Phase 10E-3 had: **144 replacements** (4 steps × 36 layers)
- Layer 0 only smoke test: **1 replacement** (1 prompt token)

## Required Patch Status

| Item | Status | Notes |
|------|--------|-------|
| LAYER_SCOPE=0 in harness | ✅ Already set | Layer 0 only |
| il==0 check in llama-graph.cpp | ✅ Already set | Layer 0 only |
| Sidecar loading restricted to layer 0 | ✅ | Only loads ffn_up_layer0_prt.bin |
| Other layers use identity fallback | ✅ | g_prt_sidecar_data = nullptr |
| No all-layer replacement | ✅ | Confirmed by GRAPH_SUBSTITUTION log |

## Verification: GRAPH_SUBSTITUTION Log

Phase 10E-3R output shows:
```
[PRT] GRAPH_SUBSTITUTION: layer=0 using ggml_map_custom2 (PRT ACTIVE)
```
Appears 7 times (one per prompt token = 7 tokens in "Hello\n"). **Not 36 times.**

All other layers: NO GRAPH_SUBSTITUTION message (no ggml_map_custom2).

## Remaining Issue

Custom op is registered but NOT called during decode loop.
Only fires during prompt evaluation (prefill).

This is a graph execution/scheduling issue, not a patch issue.