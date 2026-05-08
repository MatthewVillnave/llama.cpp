# PRT Phase 15A — 7B INT8 Longer-Context Stability Checkpoint

## Checkpoint Tag: PRT_PHASE15A_7B_INT8_LONG_CONTEXT_STABILITY_CHECKPOINT

| Field | Value |
|-------|-------|
| **Branch** | `experimental/prt-phase14a-packed-sidecars` |
| **Base Phase 15A commit** | `9cf81b392f41e3ada52e6c2c7e37d8272cbd2281` |
| **Verdict** | `PASS_7B_INT8_LONG_CONTEXT_STABLE` |

## Context

Phase 14 established that packed INT8 sidecars recovered near-native throughput for PRT on Qwen2.5 3B and 7B Q4_K_M models in Matt's measured CPU setup. Phase 15A tested whether the 7B INT8 PRT path remained stable beyond the short Phase 14Q 8-prompt validation suite by increasing generation length and context size.

## Frozen Result

Phase 15A passed. The Qwen2.5-7B INT8 PRT path remained clean, stable, and near-native under longer-generation and larger-context smoke tests on Matt's measured CPU setup.

## Tests

### Test A — 7B Longer Generation

Settings:
- model: Qwen2.5-7B-Instruct-Q4_K_M
- n_predict: 320
- context: 1024
- temp: 0
- clean llama-cli frontend
- INT8 sidecars
- PRT mode 5700
- force-native layers 11 and 15

Result:
- Native: 8.3 t/s
- INT8 PRT: 8.2 t/s
- Ratio: 0.988×
- Quality: same story/characters (Zeltronians/Lira/crystal orb)
- Collapse/repetition: none

### Test B — 7B Larger Context Factual Prompt

Settings:
- context: 2048
- n_predict: 120
- temp: 0

Result:
- Native: 8.1 t/s
- INT8 PRT: 8.1 t/s
- Ratio: 1.000×
- Quality: exact answer match (Phase 13 float32 → Phase 14 INT8 transition captured correctly)
- Collapse/repetition: none

### Test C — 7B Larger Context JSON Prompt

Settings:
- context: 2048
- n_predict: 160
- temp: 0

Result:
- Native: 8.3 t/s
- INT8 PRT: 8.2 t/s
- Ratio: 0.988×
- JSON validity: valid JSON both native and INT8 (phase13, phase14, status keys correct)
- Collapse/repetition: none

### Optional Test D — 3B Longer Generation Spot Check

Result:
- Native: 18.1 t/s
- INT8 PRT: 17.8 t/s
- Ratio: 0.983×
- Quality: same story (Terra Nova/Elara)
- Collapse/repetition: none

## Evidence

- 7B sidecars loaded: 28/28
- Sidecar format: INT8
- Fallback limited to force-native layers 11 and 15
- Clean frontend: llama-cli (not llama-prt-posix)
- No stdout contamination (stderr log artifact only, not model output)
- No quality collapse
- No repetition loop
- Memory: 11 GB RAM available
- Swap stable at 4.0 GB
- No leaked llama processes
- No models, sidecars, binaries, temp logs, or huge files staged
- No secrets detected
- Existing tags untouched before this checkpoint

## Path Fragment Note

Some INT8 run metadata reported `contains_path_fragment: true`. This was attributed to a `--prt-log-file` stderr/log artifact, not model stdout contamination. The actual generated text output was clean in all tests.

## Allowed Claim

> "Qwen2.5-7B INT8 PRT remained stable and near-native under longer-generation and larger-context smoke tests on Matt's measured CPU setup."

## Forbidden Claims

Do not claim:
- production readiness
- universal speedup
- all-long-context guarantee
- larger-than-7B support
- GPU comparison
- all-task equivalence
- exact equivalence beyond tested prompts
- deployment readiness
- that INT8 PRT always beats native
- that sidecars or model files are public/staged

## Interpretation

Phase 15A strengthens the Phase 14 result by showing that the 7B INT8 PRT path was not only a short-prompt artifact. It remained stable under longer generation (n=320 vs n=80 in Phase 14Q) and larger context (c=2048 vs c=512 in Phase 14Q) while preserving near-native throughput in the measured setup.

## Recommended Next Phase

**Recommended: Phase 15B — INT4 sidecar prototype**

Suggested Phase 15B order:
1. Offline INT4 quantization/parity checks
2. Selected-layer cosine/matvec validation
3. Runtime single-prompt canary only if offline parity passes
4. Full validation only if canary passes

**Alternative: Phase 15B — ggml/backend integration design**

Do not begin Phase 15B in this checkpoint phase.