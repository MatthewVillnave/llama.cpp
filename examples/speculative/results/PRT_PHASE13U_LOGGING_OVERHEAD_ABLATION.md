# PRT Phase 13U: Logging Overhead Ablation

**Verdict: COMPUTE_DOMINATES** ✅

## Summary

PRT `--prt-log-level` implemented (debug/summary/quiet). Ablation confirms logging accounts for only **2.5% of PRT overhead** — removing 94% of log volume saves just 44ms on a 1.78s PRT overhead. The dominant cost is CPU compute in the PRT custom op matmul, not I/O or fprintf overhead.

## Implementation

**Flag added:** `--prt-log-level {debug|summary|quiet}`

| Level | Per-call debug logs | Init/summary logs | Use case |
|-------|---------------------|-------------------|----------|
| debug (default) | All AUTH/IL/custom-op logs | All init events | Development |
| summary | Suppressed | PRT_SHAPE, sidecar_load_ms, force-native | Production debug |
| quiet | Suppressed | Suppressed | Performance testing |

**Files changed:**
- `common/common.h` — `prt_log_level` field added to `common_params`
- `common/arg.cpp` — `--prt-log-level` CLI argument
- `src/llama-graph.cpp` — `g_prt_log_level` global, `prt_logf()` gated, `prt_log_summary()` helper, `llama_set_prt_log_level()` API
- `src/llama.cpp` — extern declarations for `g_prt_log_level` and `llama_set_prt_log_level()`
- `tools/cli/cli.cpp` — calls `llama_set_prt_log_level()` after setting log file; `prt_log_level` extern
- `examples/speculative/prt_graph_replace.h` — per-call debug logs gated behind `g_prt_log_level >= 2`

**Key gating decisions:**
- Per-call `[PRT-11BB] custom op` / `IL=N` / `AUTH` logs → debug level only
- Per-call `[PRT-11BG] FORCE-NATIVE` / `FALLBACK` → debug level only
- Init: `PRT_SHAPE`, `sidecar_load_ms`, `force-native config`, `Debug mode set` → all levels (summary+)
- Fatal errors (missing sidecar) → always logged regardless of level

## Benchmark Results

**5 runs each, Qwen2.5-0.5B, 80 tokens, ctx=256, temp=0, 4 threads**

### Wall Time

| Mode | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | **Avg** | **Stddev** |
|------|-------|-------|-------|-------|-------|---------|------------|
| Native | 0.608 | 0.618 | 0.586 | 0.583 | 0.609 | **0.601s** | 0.014s |
| PRT verbose | 2.442 | 2.424 | 2.321 | 2.362 | 2.364 | **2.383s** | 0.042s |
| PRT summary | 2.355 | 2.336 | 2.338 | 2.338 | 2.336 | **2.341s** | 0.008s |
| PRT quiet | 2.322 | 2.332 | 2.346 | 2.342 | 2.349 | **2.338s** | 0.010s |

### Log Volume

| Mode | Size per run | Lines (est.) | Contents |
|------|-------------|--------------|----------|
| PRT verbose | 35,720 bytes | ~3,520 | 24 init + 3,496 per-call (AUTH + IL + custom op) |
| PRT summary | 1,994 bytes | ~32 | 24 sidecar inits + 8 summary events |
| PRT quiet | 1,994 bytes | ~32 | Same as summary (summary+quiet skip same logs) |

### Overhead Decomposition

```
Native wall:        0.601s
PRT verbose wall:   2.383s
PRT quiet wall:     2.338s
PRT overhead (verbose):  1.782s
PRT overhead (quiet):    1.737s

Logging savings (verbose→quiet): 0.044s = 44ms
Logging as % of PRT overhead:  2.49%
Logging as % of total PRT time: 1.86%
```

### Sidecar Load

| Metric | Value |
|--------|-------|
| Avg sidecar load ms | 122.5ms |
| Stddev | ~1ms |
| Consistency | Very stable across all runs |

## Interpretation

### What the data shows

**Logging is not the bottleneck.** Removing 33,726 bytes of per-call debug logging (94.4% of log volume) saves only 44ms. The PRT custom op's CPU matmul is the dominant cost.

**Overhead breakdown:**

| Component | Time | % of PRT overhead |
|-----------|------|-------------------|
| Native reference | ~0.601s | baseline |
| Sidecar load (instrumented) | ~0.122s | 6.9% |
| Debug logging I/O (verbose→quiet) | ~0.044s | 2.5% |
| **PRT compute/custom-op (residual)** | **~1.612s** | **~90.5%** |

### What Phase 13T was measuring

Phase 13T's ~1.78s PRT overhead was **not polluted by logging**. The logging ablation confirms:
- Logging overhead: ~44ms (negligible)
- Sidecar load: ~122ms (7% of overhead)
- PRT compute: ~1.61s (90%+ of overhead)

The 4.71× generation slowdown (19.7 t/s vs 92.8 t/s) is real compute overhead, not artifact.

## Quality Verification

| Check | Result |
|-------|--------|
| Native outputs clean | 5/5 ✓ |
| PRT verbose outputs clean | 5/5 ✓ |
| PRT summary outputs clean | 5/5 ✓ |
| PRT quiet outputs clean | 5/5 ✓ |
| All outputs semantically identical | ✓ |
| stdout not contaminated | ✓ |
| PRT evidence (shape, sidecar_load_ms, force-native) | Present in summary+quiet ✓ |

## Verdict Rationale

**COMPUTE_DOMINATES** — quiet mode barely changes speed (1.86% improvement). The PRT custom op's CPU-only matmul is the bottleneck, not logging, not sidecar loading.

## Allowed Claims

- Logging accounts for only 2.5% of PRT overhead (measured via `--prt-log-level quiet` vs `debug`)
- Sidecar loading accounts for ~7% of PRT overhead (measured via `[PRT_TIMING]`)
- PRT compute/custom-op accounts for ~90%+ of overhead (residual)
- `--prt-log-level` successfully gates per-call debug logs (94.4% log reduction)
- Summary mode preserves essential PRT evidence: SHAPE, sidecar_load_ms, force-native config
- Phase 13T timing results are valid — logging was not a material confounder
- No speedup claim possible — PRT is slower in all measured configurations

## Forbidden Claims

- No speedup claim
- No production readiness
- No larger-model extrapolation
- Logging does not "dominate" the slowdown

## Safety
- No models staged: ✓
- No secrets leaked: ✓
- Tags untouched: ✓
- Source changes verified clean (no model/bin/log files in diff): ✓