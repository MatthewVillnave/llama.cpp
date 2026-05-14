# PRT Phase 9: Guarded MLP_UP PRT_3P Sidecar Integration

## Overview
Implement PRT_3P as a guarded sidecar path for MLP_UP tensor operations in llama.cpp speculative decoding.

## Target
- **Tensor pattern**: `blk.%d.ffn_up` 
- **Operation**: `ggml_mul_mat(up, cur)`
- **Shape**: {2048, 11008} (typical for large models)
- **Batch**: 16/17 speculative verification tokens

## Modes

### Mode 1: STUB (`--prt-mlp-up --prt-mlp-up-mode=stub`)
- Detect eligible MLP_UP tensors (shape {2048, 11008})
- Log would-use PRT_3P intent
- Execute original float path unchanged
- **Pass criteria**: output identical, no crashes

### Mode 2: SHADOW (`--prt-mlp-up --prt-mlp-up-mode=shadow`)
- Execute float MLP_UP normally
- Also compute PRT_3P sidecar in parallel
- Compare outputs (cosine similarity, max_abs_error)
- Do NOT use PRT output for model (safety)
- **Pass criteria**: cosine >= 0.95, PRT hit count > 0

### Mode 3: ACTIVE (`--prt-mlp-up --prt-mlp-up-mode=active`)
- Use PRT_3P output for MLP_UP during verification batches
- Float fallback on any error/mismatch
- **Pass criteria**: acceptance rate stable, speedup measurable

## Implementation

### 1. New Parameters (common/common.h)
```cpp
struct common_params_speculative {
    // ... existing fields ...
    bool prt_enable = false;  // existing
    bool prt_mlp_up = false; // NEW: enable PRT for MLP_UP
    enum { PRT_MLP_UP_MODE_STUB = 0, PRT_MLP_UP_MODE_SHADOW, PRT_MLP_UP_MODE_ACTIVE };
    int prt_mlp_up_mode = PRT_MLP_UP_MODE_STUB;
};
```

### 2. Command-line Flags (common/arg.cpp)
```
--prt-mlp-up              enable PRT for MLP_UP layer
--prt-mlp-up-mode [mode]   stub|shadow|active (default: stub)
```

### 3. PRT_3P Sidecar Builder
The PRT_3P representation converts float weights to 3-plane ternary:
- Plane 0: sign (+1, 0, -1)
- Plane 1: magnitude tier (small/medium/large)
- Plane 2: residual for reconstruction

### 4. Integration Points
- Hook into `llama_decode()` before MLP_UP computation
- Detect verification batch (batch.n_tokens >= 16)
- Select mode-appropriate execution path

## Files to Modify
1. `common/common.h` - Add prt_mlp_up and prt_mlp_up_mode params
2. `common/arg.cpp` - Add --prt-mlp-up and --prt-mlp-up-mode flags
3. `examples/speculative/speculative.cpp` - Implement PRT hook integration

## Output Files
- `PRT_PHASE9_SIDECAR_PLAN.md` - This plan
- `results/prt_phase9_stub_logs.json` - Stub mode logs
- `results/prt_phase9_shadow_accuracy.json` - Shadow accuracy metrics
- `results/prt_phase9_active_benchmark.json` - Active mode benchmarks
- `PRT_PHASE9_INTEGRATION_VERDICT.md` - Final verdict

## Deterministic Test
```bash
./speculative \
  --model /path/to/target.gguf \
  --model-draft /path/to/draft.gguf \
  --prt-mlp-up --prt-mlp-up-mode=[stub|shadow|active] \
  --temp 0 -s 42 \
  -p "The quick brown fox"
```

## Pass Criteria
| Mode | Criteria |
|------|----------|
| STUB | Output identical to baseline, no crashes |
| SHADOW | Cosine similarity >= 0.95, PRT hit count > 0 |
| ACTIVE | Acceptance rate within 5% of baseline, speedup measurable |
