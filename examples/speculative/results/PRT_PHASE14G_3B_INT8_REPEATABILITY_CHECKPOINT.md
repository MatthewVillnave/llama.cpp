# PRT Phase 14G — 3B INT8 Repeatability Checkpoint

## Verdict

**PASS_3B_INT8_REPEATABLE_NATIVE_PARITY**

## Branch

`experimental/prt-phase14a-packed-sidecars`

## What this checkpoint proves

- Qwen2.5-3B INT8 PRT repeatably matches native llama.cpp throughput on the measured CPU setup.
- The Phase 14F 3B INT8 native-parity result was **not a lucky run**.
- INT8 PRT completed 10/10 repeat timing runs cleanly.
- Native completed 10/10 repeat timing runs cleanly.
- Float32 PRT completed 10/10 repeat timing runs cleanly.
- INT8 PRT produced clean output 10/10.
- INT8 PRT showed no crashes, no timeouts, and no output corruption.
- INT8 PRT remained **much faster** than float32 PRT.
- Longer generation smoke test passed at n=160, c=512.
- Mixed prompt timing showed near-native parity across multiple prompt types.

## What this checkpoint does NOT prove

- ❌ No universal speedup claim.
- ❌ No production-readiness claim.
- ❌ No larger-than-3B generalization claim.
- ❌ No 7B result yet.
- ❌ No guarantee across all prompts/tasks.
- ❌ No GPU comparison.
- ❌ No claim outside this measured CPU setup.

## Key timing table

| Mode | Avg t/s | Median t/s | Stddev | CV | Clean |
|------|---------|------------|--------|----|-------|
| Native | 21.0 | 21.0 | 0.14 | 0.7% | 10/10 |
| Float32 PRT | 8.5 | 8.5 | 0.09 | 1.1% | 10/10 |
| **INT8 PRT** | **21.0** | **21.1** | **0.18** | **0.8%** | **10/10** |

## Ratios

- INT8/native avg ratio: **1.000×**
- INT8/native median ratio: **1.002×**
- INT8/float32 PRT ratio: **2.467×**

## Longer generation result

**Prompt:** "Once upon a time in a distant galaxy"

**Settings:** n=160, c=512, t=4

| Mode | Gen t/s | Clean | Collapse | Repetition |
|------|---------|-------|----------|------------|
| Native | 18.3 | yes | none | none |
| INT8 PRT | 18.4 | yes | none | none |

**Ratio: 1.005×**

## Mixed prompt timing

| Prompt | Native t/s | INT8 t/s | Ratio |
|--------|------------|----------|-------|
| The capital of France is | 20.7 | 20.9 | 1.010× |
| Write a Python function that reverses a list. | 18.5 | 18.2 | 0.984× |
| Return JSON with keys name and status. | 18.8 | 19.2 | 1.021× |
| Explain CPU inference in one sentence. | 19.1 | 19.1 | 1.000× |

**Average ratio: 1.004×**

## Allowed claims

- ✅ Qwen2.5-3B INT8 PRT repeatability validation showed stable native-parity throughput on this measured CPU setup.
- ✅ INT8 PRT matched native throughput across 10 independent runs on the measured prompt.
- ✅ INT8 PRT preserved clean output in repeatability, longer-generation, and mixed-prompt tests.
- ✅ INT8 PRT was about 2.47× faster than float32 PRT on this measured setup.
- ✅ Packed INT8 sidecars validated the Phase 14 thesis within this measured setup.

## Forbidden claims

- ❌ Do not claim universal speedup.
- ❌ Do not claim production readiness.
- ❌ Do not claim 7B or larger-model success.
- ❌ Do not claim exact equivalence across all prompts/tasks.
- ❌ Do not claim GPU comparison.
- ❌ Do not claim beyond this measured CPU setup.

## Recommended next phase

**Phase 14H: 7B sidecar generation and validation plan.**

Before any 7B runtime testing:
- Estimate RAM/disk requirements
- Verify model shape
- Verify sidecar size
- Generate sidecars only if disk/RAM safe
- Run sidecar validation first
- Do not run 7B active PRT until sidecar validation passes