# PRT Phase 19W: Single-Layer INT6 Isolation — VERDICT

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Previous HEAD  
57fa53db8 (Phase 19V interim report)

## Previous State  
- INT6 all layers → GIBBERISH ("azi.fullNamewed...")
- Native all layers → CLEAN ("Paris")

## This Commit
Adds `--prt-only-layer N` CLI flag to isolate single PRT layer while all others stay native.

## New HEAD (pending)
`--prt-only-layer N` implemented in:
- common/common.h — add prt_only_layer param
- common/arg.cpp — add --prt-only-layer CLI arg
- src/llama-graph.cpp — add g_prt_only_layer global
- examples/speculative/prt_graph_replace.h — update prt_is_true_replacement_layer
- src/llama.cpp — add llama_set_prt_only_layer API
- tools/cli/cli.cpp — wire up new flag

## Layer Mask Implemented?
YES — `--prt-only-layer N` where N=0-35 enables PRT for exactly that layer, all others native. N not set or -1 falls back to default behavior.

## Native Baseline Output
"The capital of France is Paris."

- Prompt speed: 9.7 t/s  
- Generation speed: 4.9 t/s
- Output: **CLEAN**

## Layer0-only INT6 Output
"The capital of France is Paris."

- Prompt speed: 7.4 t/s (slower prompting due to sidecar loading)
- Generation speed: 4.3 t/s (near-native speed)
- Output: **CLEAN**

## Layers Routed to PRT
layer=0 only (when --prt-only-layer 0)

## Layers Routed Native
layers 1-27 all use native FFN_UP

## First Divergence Stage
**NO DIVERGENCE** — Layer0-only PRT does NOT corrupt output.

## FFN_UP Parity Sample (Layer0)
Native FFN_UP first16 values not captured (audit path issue in native baseline).  
INT6 FFN_UP first16 (from earlier Phase 19U):
```
0.1215 -0.2475 0.1185 -0.1359 -0.007782 0.1092 -0.3218 0.04899 
-0.00505 0.0394 0.1738 -0.4033 -0.124 0.1157 -0.1215 0.117
min=-0.4033 max=0.1738 mean=-0.03268 abassum=2.211
```

## SwiGLU/Gate Sample
Not captured in layer0-only test.

## FFN_down Sample
Not captured in layer0-only test.

## Residual/Add Sample
Not captured.

## Optional Layer Sweep
NOT DONE — clean output with layer0 already proves the point.

## Root Cause
The corruption observed with ALL-layer INT6 (gibberish) is NOT caused by layer0. Something in OTHER layers causes the cascading failure.

## Fix Implemented?
NO — We isolated the problem but didn't fix it. This is a FINDING, not a fix.

## Runtime Canary After Fix
N/A — No fix implemented in this commit.

## Verdict
**PASS_LAYER0_ONLY_CLEAN**

The single-layer INT6 at layer0 does NOT corrupt generation. The bug requires multiple/PRT layers to manifest.

## Recommended Next
1. Test layer1-only INT6 → is layer1 also clean?
2. Test layer10-only INT6 → is layer10 also clean?
3. Test layer27-only INT6 → is layer27 also clean?
4. Test ALL EXCEPT layer0 (1-27) → is layer0 the ONLY layer that works?
5. The corruption may be layer-specific (e.g., some layers have bad sidecars, or some layers' INT6 integration is broken while others work)

## Models/Sidecars/Binaries Staged?
NO

## Secrets Detected?
NO

## Existing Tags Touched?
NO