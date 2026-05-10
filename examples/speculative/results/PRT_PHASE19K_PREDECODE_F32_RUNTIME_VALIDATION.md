# PRT Phase 19K — INT6→F32 Predecode Runtime Validation

## Verdict

**BLOCKED_MACHINE_STATE**

## Context

Phase 19J added `--prt-predecode-f32` flag and `llama_set_prt_sidecar_int6_predecode_f32()` function.
Phase 19K aimed to validate that the AVX2 path actually activates and measure speedup.

## Implementation Verification

- **Flag**: `--prt-predecode-f32` parsed correctly (verified via `--help` output)
- **CLI routing**: Conditional in `tools/cli/cli.cpp:825` fixed indentation, routes to predecode when enabled
- **Predecode function**: `llama_set_prt_sidecar_int6_predecode_f32()` in `src/llama.cpp:1352`
  - Precomputes: `f32[j*K+k] = int8[j*K+k] * scales[j]`
  - Sets format=0 (activates AVX2 path)
  - Logs `[PRT-PREDECODE]` per layer
- **Format gate**: `prt_graph_replace.h:454` sets `kernel_mode=1` when `format==0 && kernel_mode==1`

## Code Path Verification (Static Analysis)

```
CLI --prt-predecode-f32 → g_prt_predecode_f32_enabled=1
  → loader calls llama_set_prt_sidecar_int6_predecode_f32()
    → mallocs float32[M*K]
    → precomputes int8 * scales → float32
    → g_prt_sidecar_data[layer] = f32_data
    → g_prt_sidecar_format[layer] = 0
Custom op: kernel_mode = (format==0 && g_prt_kernel_mode==1) ? 1 : 0
  → format=0 → kernel_mode=1 → AVX2 path activates
```

## Root Cause of Blocked Runtime

Machine state: generation is extremely slow. Even native mode (no PRT) with n=2, c=32 takes 20+ seconds per token. This is consistent with Phase 19J experience where generation crawled/timed out.

Likely causes:
1. CPU frequency scaling (i5-13500T running at ~24% MHz)
2. Swap pressure (4GB swap fully used)
3. Memory pressure (5.6GB used / 15GB total)
4. Background processes competing for CPU

## Fix Applied During This Phase

Fixed indentation bug in CLI loader conditional:
- Old: `llama_set_prt_sidecar_int6_predecode_f32` was outside proper if/else block
- New: Properly indented inside `if (g_prt_predecode_f32_enabled) { } else { }` block

## Recommended Next Phase

Phase 19K-Redo: Run on machine with better CPU/memory state, or use batched-bench for isolated throughput measurement.

## Allowed Claims

- Implementation code paths verified via static analysis
- Flag parsing works
- Predecode function logic is correct
- Format routing to AVX2 is correct

## Forbidden Claims

- No runtime speedup measured
- No AVX2 activation confirmed
- No timing data collected
