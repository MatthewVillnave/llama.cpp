# PRT Phase 13A: Model Generalization Canary - BLOCKED

**Date:** 2026-05-03  
**Branch:** `experimental/prt-phase13-model-generalization`  
**Base checkpoint:** `bca5a1f32` (PRT_PHASE12E_L11_L15_CHECKPOINT)

---

## Selected Model

- **Model:** Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
- **Path:** `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
- **Size:** 380MB
- **Layers:** 24
- **Hidden:** 896
- **FFN:** 4864
- **Quantization:** Q5_0 / Q4_K

---

## Model Inspection

The 0.5B model was successfully loaded:

- Architecture: Qwen2
- 24 layers (compatible with PRT's 36-layer assumption)
- Anchor layers for testing: 7 and 12 (~1/3 and ~1/2 depth)

---

## Sidecar Generation

### Attempted Method

Used `llama-prt-ffn-up-extract` binary to extract FFN_UP weights:

```bash
./build/bin/llama-prt-ffn-up-extract -m /path/to/0.5B.gguf --layer N
```

**Result:** Successful - generated 24 sidecar files at ~17.4MB each (total ~418MB)

### Incompatibility Found

**CRITICAL:** The PRT sidecar loader has hardcoded dimensions:

```cpp
// From phase10e0_layer0_replacement.cpp
static const int TOTAL_LAYERS = 36;  // Hardcoded to 36!
g_sidecars[layer] = {layer, data, 2048, 11008};  // Hardcoded M=2048, N=11008
```

The 0.5B model has hidden=896, ffn=4864, but the PRT code expects hidden=2048, ffn=11008.

When sidecars were generated:
- 0.5B sidecar dims: 896 × 4864 = 4.36M floats (17.4MB)
- PRT code expects: 2048 × 11008 = 22.54M floats (90MB)

**The PRT system cannot directly use 0.5B sidecars** without code modification to:
1. Change `TOTAL_LAYERS` from 36 to 24
2. Fix the hardcoded M=2048, N=11008 dimensions in the sidecar struct

---

## Canary Test Results

### Test Attempt

Attempted to run 5-prompt canary with both native baseline and PRT:

- **Model loading:** Successful (24 layers, 896 hidden, 4864 FFN)
- **Native inference:** Started but extremely slow (>3 minutes per prompt)
- **PR Test:** Not executed due to sidecar dimension incompatibility

### Reason for Blocked Status

1. **Hardcoded dimensions:** TOTAL_LAYERS=36, M=2048, N=11008 in PRT code
2. **System slowness:** 0.5B model took >3 minutes per prompt on TheForgeHQ
3. **Disk pressure:** Disk at 97% (7.6GB available of 233GB)

---

## Verdict: **BLOCKED**

| Check | Result |
|-------|--------|
| Native baseline works? | YES (but slow) |
| Sidecar generation? | YES (but wrong dimensions) |
| Sidecars load in PRT? | NO - hardcoded dims |
| PRT path runs without crash? | CANNOT TEST - dimension mismatch |
| Speedup measurable? | CANNOT TEST |

---

## Recommendations

1. **Fix PRT sidecar loader** to support dynamic model dimensions:
   - Read M/N from model metadata
   - Make TOTAL_LAYERS dynamic
   - Generate compatible sidecar format

2. **Try 3B model on fresh system** with more disk space (current disk at 97%)

3. **Alternative model:** Try a model with matching dimensions to existing sidecars

4. **Pause and return to Phase 12E validation** - the current PRT works well on 3B model

---

## What Was Generated

No results files committed - the test was blocked before completion.

Sidecar files (if needed for future work):
- Location: `/tmp/prt_phase13_sidecars/` (~418MB total)
- Format: 24 files × 17.4MB each