# PRT Phase 13E-R: Disabled-Mode Regression Isolation

## Context
- Phase 13D-S: Reported PASS - disabled mode clean
- Phase 13E: Reports FAIL - disabled mode garbage
- Need to resolve contradiction

## Step 0: Binary Verification
- Branch: experimental/prt-phase13-model-generalization
- HEAD: ef4a0c16d
- Binary: 1ff41b32dc68d... (built with diagnostics, May 4 22:28)
- libllama.so.0.0.8720: Built May 2 (before Phase 13D-S)

## Step 1: Environment Check
- No PRT env vars: ✅ CLEAN
- No aliases: ✅ CLEAN

## Step 2: Three-Way Test

### A. Native llama-cli
```
> The capital of France is
The capital of France is Paris.
```
**STATUS:** CLEAN ✅

### B. PRT disabled (sidecars hidden)
```
[PRT-BOOT] argc=13, g_prt_debug_mode initial=0
[PRT-BOOT] After model init: g_prt_debug_mode=0, will_load=0, will_callback=0

[GEN] step 0: token_id=7407 str=' located'
[GEN] step 1: token_id=304 str=' incated'
[GEN] step 2: token_id=279 str=' theated'
[GEN] step 3: token_id=4126 str=' centerd'
...
```
**STATUS:** GARBAGE ❌

### C. PRT disabled (sidecars present)
Same garbage as B - sidecar presence doesn't matter.

## Root Cause: DISABLED_GRAPH_REPLACEMENT

**Evidence:**
- Guard logs confirm `g_prt_debug_mode=0`: ✅
- No sidecar loading: will_load=0: ✅  
- No callback install: will_callback=0: ✅
- BUT output is GARBAGE: ❌

The guards in `phase10e0_layer0_replacement.cpp` work correctly - they prevent sidecar loading and callback installation. 

However, the garbage output comes from GGML GRAPH replacement - specifically, the PRT graph code paths in `libllama.so` are being executed even with g_prt_debug_mode=0.

The libllama library (built May 2) contains:
1. `prt_is_true_replacement_layer()` in llama-graph.cpp (via prt_graph_replace.h)
2. Graph replacement logic that checks g_prt_debug_mode before replacing layers

HOWEVER: Even with g_prt_debug_mode=0, either:
- The graph code paths in libllama are different/broken from native, OR
- Something in the PRT graph setup corrupts the computation

**Key finding:** The PRT example binary uses a different libllama than native - one that has PRT graph code compiled in.

## Classification: D (DISABLED_GRAPH_REPLACEMENT)

The graph construction in libllama produces fundamentally different results even when mode=0, likely because:
1. The library was compiled with PRT code
2. Something in the graph initialization changes behavior
3. The mode=0 check in prt_is_true_replacement_layer() returns false but graph setup still differs

## Files Changed
- examples/speculative/phase10e0_layer0_replacement.cpp (added diagnostics only)

## Verdict: FAIL

Native: CLEAN
PRT-disabled: GARBAGE

Active canary NOT allowed - disabled mode fundamental difference must be resolved first.

## Recommended Next Steps
1. Rebuild libllama with new build OR
2. Use same llama-cli binary as PRT binary (share library link)
3. Verify which specific graph changes cause the difference
4. PRT code in libllama must gracefully degrade when disabled

## Safety
- No model/file artifacts staged: ✅
- No secrets: ✅  
- Tags untouched: ✅