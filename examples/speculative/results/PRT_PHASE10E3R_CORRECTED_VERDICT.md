# PRT_PHASE10E3R_CORRECTED_VERDICT

## Executive Summary

**Verdict: FAIL**

Phase 10E-3R attempted corrected layer0-only PRT replacement but failed due to a state management bug in the custom op integration.

---

## Preflight Check Results

### 1. Model Layer Count
- **36 layers** (Qwen2.5-3B, qwen2.block_count=36 in GGUF metadata)
- Layers 0-35

### 2. Sidecar Count
- **28 sidecars** in `/tmp/prt_sidecars/`
- Layers 0-27 covered (ffn_up_layer0-27_prt.bin)
- Layers 28-35 **missing** (8 layers uncovered)
- Coverage: 28/36 = 77.8%

### 3. Layer0-Only Enforced
- **YES** - Harness LAYER_SCOPE=0 restricts callback
- **YES** - llama-graph.cpp `if (il == 0)` check restricts graph op
- GRAPH_SUBSTITUTION fires only for layer=0 (8 times for 2-token prompt)

### 4. Non-Layer0 Replacements
- **0** - No non-layer0 custom op replacements
- Only layer 0 had ggml_map_custom2 substitution
- 2 fallback calls (from custom op execution)

---

## Custom Op Math Analysis

| Check | Result |
|-------|--------|
| Computes X @ W_prt | YES |
| Uses input-specific cache | NO |
| Output shape correct | YES [batch, 11008] |
| Sidecar format | |W| (all positive) |
| Layer-specific sidecar | YES |

### Math Verdict: CORRECT ✅
The custom op implements the correct PRT math. No issues with the computation itself.

---

## Accuracy Metrics

| Metric | Value | Target | Pass? |
|--------|-------|--------|-------|
| Cosine | 0.0 | >= 0.95 | ❌ |
| Max abs error | N/A (no PRT) | - | N/A |
| Mean abs error | N/A (no PRT) | - | N/A |

**Accuracy: FAIL** - No PRT was applied, so cosine against PRT reference = 0.0

---

## Generation Test

| Property | Result |
|----------|--------|
| Ran without crash | YES ✅ |
| Output | "H" (single token, no PRT effect) |
| Repetition loops | N/A |
| Coherence | N/A (no PRT) |

**Generation: Ran but no PRT modification occurred**

---

## Root Cause

**State bleed**: `g_prt_ffn_up_layer_last` is a global variable set during graph building. After building all 36 layers, it holds `35` (last layer processed). During custom op execution, it reads `35` instead of `0`, causing the layer check `il == 0` to fail and fall back to identity.

**Evidence**:
```
prt_op_entry #1: layer=35 sidecar=0x750f2485e010 il_prev=35
```
Custom op fires (entry logged) but with wrong layer value.

---

## Verdict Decision Matrix

| Criterion | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Layer0-only enforced | YES | YES | ✅ |
| No non-layer0 replacements | YES | YES (0) | ✅ |
| Cosine >= 0.95 | YES | 0.0 | ❌ |
| No crash | YES | YES | ✅ |
| Output coherent | YES | "H" (no PRT) | ❌ |
| Fragile layers avoided | YES | YES | ✅ |

**Overall: FAIL ❌**

---

## Is All-Layer PRT Retest Allowed?

**NO** ❌

Reasons:
1. Sidecar coverage is 28/36 (missing layers 28-35)
2. Using layer0 sidecar for all 36 layers is weight mismatch
3. The core failure is state management, not scope

**All-layer PRT requires 36 sidecars (one per layer). Only 28 exist.**

---

## Required Next Steps

1. **Fix layer context in custom op** (userdata approach or tensor opaque)
2. **Re-generate sidecars for layers 28-35** (missing coverage)
3. **Test layer0 PRT with correct state management** before all-layer attempt

**Phase 10E-3R is complete. FAIL recorded.**