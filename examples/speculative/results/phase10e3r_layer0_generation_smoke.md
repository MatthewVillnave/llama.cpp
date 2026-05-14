# Phase 10E-3R: Layer0-Only Generation Smoke Test

## Test Setup

- Binary: `llama-phase10e0-layer0`
- Model: Qwen2.5-3B-Instruct-Q4_K_M.gguf
- Sidecar: layer 0 only (`ffn_up_layer0_prt.bin`, M=2048, N=11008)
- Harness: LAYER_SCOPE=0 (layer 0 only)
- llama-graph.cpp: `if (il == 0)` check (layer 0 only)
- Temperature: 0 (deterministic)
- Seed: 42
- Prompt: "Hi" (2 tokens)

## Results

### Graph Substitution (Prompt Eval)
```
[PRT] GRAPH_SUBSTITUTION: layer=0 using ggml_map_custom2 (PRT ACTIVE)
```
- Appears 8 times (2 prompt tokens × multiple graph builds)
- All are layer=0 only ✅

### Custom Op Execution
```
[PRT] prt_op_entry #1: layer=35 sidecar=0x750f2485e010 il_prev=35
[PRT] PRT_OP ENTRY: layer=35 sidecar=0x750f2485e010
[PRT] PRT_OP: layer=35 fallback=identity (sidecar=0x750f2485e010 layer=0 wanted=35)
```

- Only 2 prt_op_entry calls (decode steps)
- Both show layer=35 with fallback
- **Layer 0 custom op NOT called** ❌

### Replacement Count
```
Total PRT replacements: 0  [fallback: 2]
Last cosine: 0.000000
```

- 0 PRT replacements (custom op)
- 2 fallback calls (identity)
- Cosine: 0.0

### Generation Output
```
0: 'H
```
- Single token "H"
- Not obviously broken (but no PRT applied)

## Layer Scope Analysis

| Layer | Sidecar | ggml_map_custom2 | Custom op fires | PRT compute |
|-------|---------|------------------|-----------------|-------------|
| 0 | YES | YES | ❌ NO | ❌ NO |
| 1-35 | NO (null) | NO | ❌ NO | ❌ NO |

**Only layer 0 was targeted** ✅
**But custom op did not execute** ❌

## What Happened

1. Harness loaded only layer 0 sidecar
2. llama-graph.cpp substituted only layer 0 with ggml_map_custom2
3. During execution, the custom op function (`prt_ffn_up_prt_op`) was never called
4. Instead, `prt_op_entry #1` shows layer=35 with fallback
5. The `prt_op_entry` counter (entry logging) shows layer=35, NOT layer=0

**The prt_ffn_up_prt_op function is being called, but with `g_prt_ffn_up_layer_last=35`.**

This means: **During decode, the custom op fires for the LAST processed layer (35), not layer 0.** The graph structure has a custom op that fires at some point, but `g_prt_ffn_up_layer_last` was set to 35 when the op was called.

This is because the GRAPH_SUBSTITUTION fires at graph BUILD time, but the custom op is scheduled and called at compute time. The `g_prt_ffn_up_layer_last = il` is set during graph BUILD (when `build_ffn` is called). During compute, the op is invoked multiple times as the graph is executed, but `g_prt_ffn_up_layer_last` retains its LAST value from graph building (35).

Wait, let me re-examine. The LOG shows:
```
[PRT] prt_op_entry #1: layer=35 sidecar=0x750f2485e010 il_prev=35
```

`prt_op_entry` is called at the START of `prt_ffn_up_prt_op()`. The `g_prt_ffn_up_layer_last` was set to 35. But during prompt eval, `build_ffn` was called for layers 0, 1, 2, ... 35. The LAST call was for layer 35, setting `g_prt_ffn_up_layer_last = 35`.

BUT during decode, only layer 0's graph (with the custom op) should be active. Why does the op fire with layer=35?

The answer: **The custom op tensor (`ffn_up.prt.layer0`) was created during prompt eval graph build, which set `g_prt_ffn_up_layer_last = 35` by the end. Then during decode, the SAME op function is called but `g_prt_ffn_up_layer_last` is still 35 from the build phase.**

This is a **state bleed bug**: `g_prt_ffn_up_layer_last` is set during graph building but persists into execution, and since the last layer built was 35, it shows 35.

But wait, during decode graph build, only layer 0 should be built... Let's check. Actually, during decode, llama builds a fresh graph for each decode step. If the decode graph includes ALL layers (even if only 1 token), then `build_ffn` is called for all 36 layers again, setting `g_prt_ffn_up_layer_last = 35` at the end of graph construction.

Then when the custom op executes, it reads `g_prt_ffn_up_layer_last = 35`. The condition `if (g_prt_ffn_up_layer_last == 0)` fails, so it falls back to identity.

**Root cause identified: `g_prt_ffn_up_layer_last` set during graph build (per-layer loop), but the value is the LAST layer processed, not the specific layer whose op is executing. The custom op function has no per-invocation layer context.**

## Pass Criteria Check

| Criterion | Expected | Actual | Pass? |
|-----------|----------|--------|-------|
| Replacement fires for layer0 only | YES | NO (fallback fires) | ❌ |
| No other layers replaced | YES | N/A (no PRT) | ✅ |
| Cosine >= 0.95 | YES | 0.0 | ❌ |
| No crash | YES | NO crash | ✅ |
| Output coherent | YES | "H" (garbage) | ❌ |

## Verdict: FAIL ❌

The layer0-only patch is correctly scoped but broken by execution semantics:
1. `g_prt_ffn_up_layer_last` bleeds to last-built-layer value
2. Custom op called but sees wrong layer value
3. Fallback fires instead of PRT compute
4. No actual PRT modification occurs