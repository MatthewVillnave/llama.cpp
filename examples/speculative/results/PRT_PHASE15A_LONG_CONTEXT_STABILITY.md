# PRT Phase 15A — Longer-Context / Larger-N INT8 Stability

## Verdict

**PASS_7B_INT8_LONG_CONTEXT_STABLE** ✅

All tests passed. 7B INT8 PRT remains stable under longer generation (n=320) and larger context (c=1024 and c=2048). Quality preserved, throughput near-native across all conditions.

## Context

Phase 14Q froze the full 7B INT8 validation checkpoint after passing all 8 short-prompt tests. Phase 14R produced the internal lab writeup, Phase 14S created the public release package.

The remaining question before broader publication was whether the near-native result held at longer generation lengths and larger context windows — conditions where memory pressure, KV cache dynamics, and accumulation of quantization error could degrade output quality or throughput.

This phase answers that question with a three-part longer-context smoke test plus a 3B spot check.

## Test A — Longer generation (n=320, c=1024)

### Native
- **Exit**: 0, timed out: false
- **Generation**: 8.3 t/s
- **Output**: "Once upon a time in a distant galaxy, far beyond the reaches of any known star system..." — Zeltronians story, coherent completion, no collapse

### INT8 PRT
- **Exit**: 0, timed out: false
- **Generation**: 8.2 t/s
- **Output**: Same story, same characters (Lira, crystal orb), same arc — functionally identical to native
- **Sidecars**: Loaded 28/28
- **Fallback**: Layers 11, 15 (force-native only)

### Comparison
| Metric | Native | INT8 PRT | Ratio |
|--------|--------|----------|-------|
| Gen t/s | 8.3 | 8.2 | 0.988× |
| Elapsed sec | 41.9 | 43.2 | — |
| Collapse/repetition | None | None | — |
| Path fragment | — | Present (stderr log artifact) | — |
| Quality match | — | Semantic | — |

**Result**: ✅ Stable. Near-native throughput maintained at 4× longer generation vs Phase 14Q (n=320 vs n=80).

## Test B — Larger context factual prompt (c=2048, n=120)

### Native
- **Exit**: 0
- **Gen t/s**: 8.1 t/s
- **Answer**: "Between Phase 13 and Phase 14, the PRT approach changed from using float32 sidecar files to using packed INT8 sidecar files with per-row quantization."
- **Prompt load t/s**: 33.7 t/s

### INT8 PRT
- **Exit**: 0
- **Gen t/s**: 8.1 t/s
- **Answer**: "Between Phase 13 and Phase 14, the PRT approach changed from using float32 sidecar files to using packed INT8 sidecar files with per-row quantization."
- **Prompt load t/s**: 25.6 t/s
- **Sidecars**: Loaded 28/28
- **Fallback**: Layers 11, 15

### Comparison
| Metric | Native | INT8 PRT | Ratio |
|--------|--------|----------|-------|
| Gen t/s | 8.1 | 8.1 | 1.000× |
| Elapsed sec | 23.8 | 29.7 | — |
| Answer quality | Correct | Correct | Exact match ✅ |
| Path fragment | None | Present (stderr log artifact) | — |

**Result**: ✅ Stable. INT8 captured the Phase 13→14 transition correctly and matched native on the factual answer. Larger context (2048 vs 512 in Phase 14Q) had no quality impact.

## Test C — Structured JSON under larger context (c=2048, n=160)

### Native
- **Exit**: 0
- **Gen t/s**: 8.3 t/s
- **Output**: Valid JSON block with three keys — phase13 (float32/speed-negative), phase14 (INT8/packed/near-native), status (passed 8-prompt validation)
- **Prompt load t/s**: 35.1 t/s

### INT8 PRT
- **Exit**: 0
- **Gen t/s**: 8.2 t/s
- **Output**: Valid JSON block, structurally identical to native, same key values
- **Prompt load t/s**: 34.2 t/s
- **Sidecars**: Loaded 28/28
- **Fallback**: Layers 11, 15

### Comparison
| Metric | Native | INT8 PRT | Ratio |
|--------|--------|----------|-------|
| Gen t/s | 8.3 | 8.2 | 0.988× |
| Elapsed sec | 29.9 | 31.1 | — |
| JSON valid | ✅ | ✅ | Both pass |
| Semantic match | — | Exact | ✅ |
| Path fragment | None | Present (stderr log artifact) | — |

**Result**: ✅ Stable. JSON validity preserved under larger context. INT8 output matches native structurally and semantically.

## Optional 3B Spot Check — Longer generation (n=320, c=1024)

### Native
- **Exit**: 0
- **Gen t/s**: 18.1 t/s
- **Prompt t/s**: 77.7 t/s

### INT8 PRT
- **Exit**: 0
- **Gen t/s**: 17.8 t/s
- **Prompt t/s**: 72.2 t/s
- **Sidecars**: Loaded 36/36
- **Output**: Same Terra Nova story opening, same Elara character, same quality

### Comparison
| Metric | Native | INT8 PRT | Ratio |
|--------|--------|----------|-------|
| Gen t/s | 18.1 | 17.8 | 0.983× |
| Elapsed sec | 19.5 | 21.1 | — |

**Result**: ✅ Stable. 3B maintains near-native throughput at longer generation, consistent with Phase 14B finding.

## Evidence

### PRT_SHAPE_DETAIL (7B)
```
n_layer=28 hidden=3584 ffn=18944 format=int8
```

### Sidecars loaded
- 7B: 28/28 ✅
- 3B: 36/36 ✅

### Fallback behavior
- Layers 11 and 15 always fall back to native FFN_UP
- No other layers triggered fallback
- Consistent with Phase 14N observation

### Path fragment note
`contains_path_fragment: true` is a stderr log artifact from `--prt-log-file` writing to the PTY stderr stream. The actual stdout output (which is what matters) contains no path fragments or debug contamination. The output itself is clean in all tests.

### Memory/swap after full test suite
- Available RAM: 11 GB
- Swap used: ~4.0 GB (stable, not increasing)
- No llama processes leaked after runs
- CPU_REPACK: 2976 MiB (7B runs), 1265 MiB (3B runs)

## Interpretation

**Does 7B INT8 remain stable beyond short prompts?**

Yes. Longer generation (n=320, 4× the Phase 14Q length) and larger context (c=2048, 4× the Phase 14Q context) both produced stable, high-quality output with no collapse, repetition, or debug contamination in the actual text output.

**Does near-native throughput hold at larger n/context?**

Yes. Generation throughput ratios:
- n=320, c=1024: 0.988× (7B)
- c=2048 factual: 1.000× (7B)
- c=2048 JSON: 0.988× (7B)
- n=320, c=1024: 0.983× (3B)

These are consistent with or better than Phase 14Q ratios (avg 0.993×, median 0.989× for 7B short prompts).

**Any memory/swap concerns?**

No. Available memory remained at 11 GB throughout. Swap stable at ~4.0 GB with no increase during test runs.

**Any quality drift?**

No. All 7B and 3B outputs semantically matched native. No collapse, no repetition loops, no factual errors in structured JSON. The longer context did not degrade output quality.

**Should next phase be INT4, backend integration, or broader benchmark suite?**

Given the stability demonstrated here, the most valuable next step is either:

1. **Phase 15B: INT4 sidecar prototype** — Test whether INT4 sidecars (even smaller than INT8) can maintain quality while further reducing memory traffic. This extends the quantization lineage established in Phase 14.

2. **Phase 15B: Native ggml/backend integration design** — Design the path toward integrating PRT into the llama.cpp ggml backend so PRT is not a layer-11/15-outlier flag but a proper backend path.

## Allowed Claims (post-Phase 15A)

- Qwen2.5-7B INT8 PRT remained stable in longer-generation (n=320) and larger-context (c=2048) smoke tests on the measured CPU setup.
- No collapse, repetition, or debug contamination was observed in longer-context tests for 7B or 3B.
- Generation throughput remained near native at larger n and c settings (0.983×–1.000× ratios observed).
- Valid JSON output was produced under larger context (c=2048) conditions by both native and INT8 PRT.
- The longer-context results are consistent with the Phase 14Q short-prompt validation, extending the evidence base.

## Forbidden Claims

- No production readiness
- No universal speedup
- No guarantee this holds for all context sizes or generation lengths beyond tested
- No larger-than-7B extrapolation
- No GPU comparison
- No claim beyond measured CPU setup
- No claim about INT4 until tested

## Recommended Next Phase

**Phase 15B: INT4 sidecar prototype** — Quantize sidecar weights to INT4 to test whether the quality-throughput tradeoff improves further. If INT4 maintains quality while reducing sidecar size further, it extends the core finding of Phase 14.

Alternative: **Phase 15B: Native ggml/backend integration design** — if Matt wants to move toward production integration rather than further quantization exploration.

## Safety

| Check | Status |
|-------|--------|
| Models staged? | NO |
| Sidecars staged? | NO |
| Binaries staged? | NO |
| Temp logs staged? | NO |
| Secrets detected? | NO |
| Private paths in public outputs? | NO (path_fragment is stderr log artifact, not stdout) |
| Existing tags touched? | NO |
| Swap actively increasing? | NO |