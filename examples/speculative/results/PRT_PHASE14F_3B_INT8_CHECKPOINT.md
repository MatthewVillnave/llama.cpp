# PRT Phase 14F — 3B INT8 Quality + Timing Checkpoint

## Verdict

**PASS_3B_INT8_FULL_QUALITY_AND_TIMING**

## Branch

`experimental/prt-phase14a-packed-sidecars`

## Commit

`82726efd72172cc649aa6211788b460677b03141`

## What this checkpoint proves

- Qwen2.5-3B INT8 PRT runs through the clean llama-cli PTY runner.
- INT8 sidecars load successfully: 36/36.
- INT8 sidecar format is active.
- PRT_SHAPE_DETAIL confirms:
  - n_layer=36
  - hidden=2048
  - ffn=11008
  - sidecar_rows=11008
  - sidecar_cols=2048
  - runtime_M=11008
  - runtime_N=2048
  - format=int8
- Native completed 8/8 validation prompts.
- Float32 PRT completed 8/8 validation prompts.
- INT8 PRT completed 8/8 validation prompts.
- INT8 clean outputs were extracted 8/8.
- INT8 semantic matches were 8/8.
- INT8 exact matches were 6/8 (ANSI artifact — not actual text differences).
- Quality degradations were 0.
- Repetition/collapse was 0.
- JSON prompt passed (valid JSON output).
- Code prompts were plausible (Python function, explanation).
- Fallback was limited to force-native layers 11 and 15.
- INT8 PRT was measured at 18.9 t/s versus native at 19.1 t/s on this setup.
- INT8 PRT was 2.45× faster than float32 PRT on this setup.

## What this checkpoint does NOT prove

- ❌ No universal speedup claim.
- ❌ No production-readiness claim.
- ❌ No larger-than-3B generalization claim.
- ❌ No exact equivalence across all prompts.
- ❌ No claim that INT8 PRT always beats native.
- ❌ No claim outside this exact model, prompt suite, hardware, and runtime setup.

## Key comparison

| Mode | Avg generation throughput |
|------|-------------------------|
| Native 3B | 19.1 t/s |
| Float32 PRT 3B | 7.7 t/s |
| INT8 PRT 3B | **18.9 t/s** |

## Interpretation

Phase 13 showed that float32 sidecars preserved quality but were speed-negative (~0.40× native).

Phase 14 showed that packed INT8 sidecars preserve quality while recovering nearly all native throughput on Qwen2.5-3B (~0.99× native).

This supports the packed-sidecar thesis:
the PRT idea was not the dead part; the float32 sidecar representation was too memory-heavy.

## Allowed claims

✅ Qwen2.5-3B INT8 PRT passed full 8-prompt clean quality validation with 8/8 semantic matches and 0 quality degradations.
✅ Qwen2.5-3B INT8 PRT loaded 36/36 sidecars with INT8 format active.
✅ Qwen2.5-3B INT8 PRT measured 18.9 t/s versus native 19.1 t/s on this setup.
✅ INT8 PRT was 2.45× faster than float32 PRT on this setup.
✅ INT8 sidecars substantially reduce sidecar size versus float32 (~4×).

## Forbidden claims

- ❌ Do not claim universal speedup.
- ❌ Do not claim production readiness.
- ❌ Do not claim larger-than-3B success.
- ❌ Do not claim exact equivalence across all prompts.
- ❌ Do not claim INT8 PRT always beats native.
- ❌ Do not claim results outside the measured setup.

## Recommended next phase

**Phase 14G: Stability / repeatability validation.**

Suggested goals:
- rerun 3B INT8 timing across more runs
- verify variance
- validate memory/swap health
- optionally run longer n=100 prompt
- optionally run 4-prompt timing sweep
- do not expand claims until repeatability is measured

**Alternative:**
Phase 14G: Optimization toward consistent native-beating throughput:
- AVX2/VNNI INT8 kernel refinement
- mmap INT8 path if not already used
- lower precision INT4 or ternary experiment
- native ggml/backend integration