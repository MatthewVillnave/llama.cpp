# PRT Phase 21F — 0.5B F32 Tensor Plumbing Report

## Summary
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Previous HEAD:** `bbc757f96`  
**Status:** Partial — CLI arg parsing blocker prevents full validation

---

## Phase 21F Results

| Check | Status |
|-------|--------|
| 0.5B model found | ✅ Qwen2.5-0.5B-Instruct-Q4_K_M.gguf (380MB) |
| f32 weights located | ✅ /tmp/prt_phase21f_layer0_W_f32.bin (17MB) |
| W/scales shape | ✅ K=896, M=4864 |
| build_ffn code | ✅ Real f32 weight loading + synthetic fallback |
| CLI flag added | ⚠️ Arg parsing issue (blocked) |
| Native behavior unchanged | ✅ When flag not triggered |

### CLI Arg Parsing Issue
The `--prt-ggml-op-test` flag with `--prt-ggml-op-layer` causes an argument parsing error. The arg parser seems to expect a value for the flag. Attempts to fix:
- Changed from `int` to `string` handler → still fails
- Tried multiple argument orderings → same error
  
**Root cause unknown** — other similar flags work but this one doesn't.

---

## Changes Made

### 1. Tensor Loading (src/llama-graph.cpp)
```cpp
// Phase 21F: Load real f32 weights from file
static float * g_f32_weights[36] = {nullptr};
const char * weight_path = "/tmp/prt_phase21f_layer0_W_f32.bin";

// Load once if file exists and size matches
if (!weight_loaded && il >= 0 && il < 36) {
    FILE * wf = fopen(weight_path, "rb");
    if (wf) {
        // Load K*M*sizeof(float) bytes
        ...
        weight_loaded = true;
    }
}

// Use real weights if loaded, else synthetic transpose
if (g_f32_weights[il]) {
    W = ggml_new_tensor(ctx0, GGML_TYPE_F32, 2, dims_w);
    memcpy(W->data, g_f32_weights[il], K*M*sizeof(float));
}
```

### 2. CLI Default (tools/cli/cli.cpp)
```cpp
// Default layer to 0 if test enabled but layer is -1
int target_layer = params.prt_ggml_op_layer;
if (params.prt_ggml_op_test && target_layer < 0) {
    target_layer = 0;  // Default to layer 0
}
if (params.prt_ggml_op_test) {
    llama_set_ggml_op_test(1, target_layer);
}
```

### 3. Weight File Created
Location: `/tmp/prt_phase21f_layer0_W_f32.bin`  
Size: 17MB (896×4864×4 bytes)  
Source: Copied from `/tmp/prt_phase19b/layer0_float.bin`

---

## Verdict: PARTIAL_CLI_ARG_BLOCKED

### Known Issues
- CLI argument `--prt-ggml-op-test` triggers "expected value for argument" error
- CLI `--prt-ggml-op-layer` doesn't parse correctly in combination with test flag
- Runtime flag can be set via llama_set_ggml_op_test() API but CLI is broken

### What Works
- Build succeeds
- Native behavior unchanged when test not enabled
- Weight loading code is in place with proper fallbacks
- Null scales handled (identity scaling)
- Shape logging present

### Recommended Next
1. Fix CLI argument parsing for --prt-ggml-op-test or use environment variable instead
2. Once CLI works: Run native baseline vs PRT-v2 output comparison
3. Then verify semantic canary (should say "Paris" for "The capital of France is")

---

## Files Changed (to commit)
- src/llama-graph.cpp (f32 weight loading)
- tools/cli/cli.cpp (default layer fallback)
- common/arg.cpp (arg definitions)

---

## Safety Scan
- Model files: NOT staged ✅
- Sidecars: NOT staged ✅
- Secrets: NOT detected ✅
- Tags: NOT touched ✅