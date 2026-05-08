# PRT Phase 14N — 7B INT8 Repeatability Checkpoint

## Verdict

**PASS_7B_INT8_REPEATABLE_NATIVE_PARITY** ✅

## Branch

`experimental/prt-phase14a-packed-sidecars`

## Commit

`b6639a60e`

## What this checkpoint proves

- Qwen2.5-7B INT8 PRT runs through the clean llama-cli PTY runner.
- Qwen2.5-7B INT8 sidecars load successfully: 28/28.
- INT8 sidecar format is active.
- PRT_SHAPE_DETAIL confirms:
  - n_layer=28
  - hidden=3584
  - ffn=18944
  - format=int8
- Native completed repeatability runs cleanly.
- INT8 PRT completed repeatability runs cleanly.
- INT8 PRT preserved output exactly/semantically across the tested repeatability set.
- Quality degradations were 0.
- Failures, timeouts, and outliers were 0.
- Longer generation smoke test passed with identical output.
- Mixed prompt timing passed across 4 prompt types.
- Fallback was limited to force-native layers 11 and 15.
- INT8 PRT retained near-native throughput on this measured CPU setup:
  - native avg: 9.66 t/s
  - INT8 PRT avg: 9.44 t/s
  - avg ratio: 0.977×
  - median ratio: 0.990×

## What this checkpoint does NOT prove

- No universal speedup claim.
- No production-readiness claim.
- No larger-than-7B generalization claim.
- No GPU comparison claim.
- No guarantee across all prompts/tasks.
- No exact equivalence beyond tested prompts.
- No claim outside this measured CPU setup.

## Key timing table

| Mode | Avg t/s | Median t/s | Stddev | Clean |
|------|---------|------------|--------|-------|
| Native 7B | 9.66 | 9.6 | 0.08 | 10/10 |
| INT8 PRT 7B | 9.44 | 9.5 | 0.24 | 10/10 |

## Ratios

- INT8/native avg ratio: 0.977×
- INT8/native median ratio: 0.990×

## Longer generation result

**Prompt**: "Once upon a time in a distant galaxy"

**Result**:
- Native and INT8 PRT produced identical output.
- Both measured about 8.5 t/s.
- No collapse.
- No repetition loop.
- No debug contamination.

## Mixed prompt result

4 mixed prompts passed:
1. "The capital of France is" — exact match
2. "Write a Python function that reverses a list." — semantic match (slicing approach)
3. "Return JSON with keys name and status." — exact match, JSON valid
4. "Explain CPU inference in one sentence." — exact match

**Result**: 4/4 matched. JSON prompt valid. Code prompt plausible. No quality degradation.

## Phase history for this checkpoint

| Phase | Commit | Result |
|-------|--------|--------|
| 14C | b03b1ca73 | Validate 0.5B INT8 sidecar runtime (8/8 exact, 1.81× faster) |
| 14E | cd7068c98 | Validate 3B INT8 sidecar runtime |
| 14F | c751e3e11 | Freeze 3B INT8 checkpoint (`PRT_PHASE14F_3B_INT8_CHECKPOINT`) |
| 14G | cd7068c98 | Validate 3B INT8 repeatability |
| 14G-tag | 0978d957e | Freeze 3B INT8 repeatability (`PRT_PHASE14G_3B_INT8_REPEATABILITY_CHECKPOINT`) |
| 14H | 54a9bdf5c | Plan 7B INT8 sidecar feasibility |
| 14J | 257a1b18c | Diagnose 7B INT8 parity failure |
| 14K | 257a1b18c | Run 7B INT8 single prompt canary (PASS) |
| 14L | 772654ade | Validate 7B INT8 4-prompt validation (PASS 4/4) |
| 14M | b6639a60e | Validate 7B INT8 repeatability (10 runs, 0 failures) |
| **14N** | **b6639a60e** | **Freeze 7B INT8 checkpoint** |

## Allowed claims

- Qwen2.5-7B INT8 PRT repeatability validation preserved tested output exactly/semantically on this measured CPU setup.
- Qwen2.5-7B INT8 PRT retained ~97.7% average and ~99.0% median native generation throughput on this measured CPU setup.
- Qwen2.5-7B INT8 sidecars loaded 28/28 with fallback limited to force-native layers 11 and 15.
- The packed INT8 sidecar path scales from 3B to 7B in the tested setup.

## Forbidden claims

- Do not claim universal speedup.
- Do not claim production readiness.
- Do not claim larger-than-7B success.
- Do not claim GPU comparison.
- Do not claim exact equivalence across all prompts/tasks.
- Do not claim results outside this measured CPU setup.

## Recommended next phase

**Phase 14O**: Decide next research direction.

Recommended options:
1. Full 8-prompt 7B validation if we want symmetry with 3B.
2. Optimization toward consistent native-beating throughput.
3. INT4 sidecar prototype.
4. Native ggml/backend integration design.
5. Longer-context / larger-n generation stability study.
6. Publish/lab writeup of Phase 14 packed-sidecar result.