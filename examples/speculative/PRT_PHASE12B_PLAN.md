# PRT Phase 12B Plan — Speed Attribution

**Version:** 1.0
**Date:** 2026-05-03
**Branch:** `experimental/prt-route-a-phase12`

---

## Goal

Determine where the Phase 12A ~1.79x speedup comes from.

The speedup is observed at the wall-clock level. But:
- Is it from FFN_UP replacement (the custom op)?
- Is it from reduced memory bandwidth (sparse ternary vs dense matmul)?
- Is it from reduced compute (ternary vs float32)?
- Is it from something else entirely?

Phase 12B measures component-level time to answer this question.

---

## Key Question

> If FFN_UP is the hot path being replaced, why is the speedup not larger? And is there room for more?

Answering this requires isolating:
1. Time spent in native FFN_UP (the baseline)
2. Time spent in PRT custom op (the replacement)
3. Time spent in force-native layers L12/L15 (the fallback anchors)
4. Time overhead from graph dispatch, sidecar loading, and validation

---

## Test Setup

Use a fixed subset of 3 prompts from Phase 12A:

1. Narrative: "Once upon a time in a" (n=100)
2. Factual: "What is the capital of France?" (n=100)
3. Code: "Write a Python function to reverse a list." (n=100)

**Runs per prompt:**
- Native baseline (--prt-mode 0)
- PRT L12+L15 (--prt-mode 5700 --prt-force-native "12,15")
- Pure PRT all-36 (--prt-mode 5700, no force-native) — for comparison

---

## Measurements to Capture

### Wall-clock breakdown (via `time` + internal timing if available)

| Component | How to measure |
|-----------|---------------|
| Total wall time | `time` real |
| Token loop overhead | profiler or repeated n=1 runs |
| FFN_UP time (native) | native baseline, isolate from total |
| FFN_UP time (PRT) | PRT custom op calls |
| L12+L15 time (native fallback) | native_fallback_calls × estimated per-call |
| Sidecar load time | measure at startup |
| Graph dispatch overhead | ggml custom op call overhead |

### Expected pattern

If FFN_UP replacement is the primary driver:
- PRT wall time ≈ native wall time − FFN_UP time savings
- L12+L15 overhead should be measurable vs pure PRT

If FFN_UP is NOT the primary driver:
- Speedup persists even when isolating non-FFN_UP paths
- Something else is the bottleneck

---

## Approaches

### Approach 1: Instrumented Benchmark (Preferred)

Add timing regions to `build_ffn` in `phase10e0_layer0_replacement.cpp`:

```cpp
struct {
  uint64_t t_native_ffn = 0;
  uint64_t t_prt_ffn = 0;
  uint64_t t_sidecar_load = 0;
  uint64_t t_validation = 0;
  uint64_t t_graph_dispatch = 0;
} timing;

// Around native path:
timing.t_native_ffn += get_time_ns();

// Around PRT path:
timing.t_prt_ffn += get_time_ns();
```

Print totals at end of generation.

### Approach 2: External profiler (if instrumented build unavailable)

Use `perf` or `valgrind --tool=callgrind` on a single prompt to identify where time is spent.

```bash
perf record -g -- ./build/bin/llama-prt-posix [args]
perf report
```

### Approach 3: Differential timing

Run same prompt in native vs PRT mode at multiple n_predict values:
- n=1: mostly startup + first token
- n=10: more token loop
- n=50: steady state
- n=100: full run

Plot time(n) curves for both. If curves are parallel, the overhead is fixed. If they converge or diverge, something is scaling differently.

---

## Output Files

After measurement:

1. `examples/speculative/results/PRT_PHASE12B_SPEED_ATTRIBUTION.md`
   - Component-level time breakdown
   - Which component accounts for how much of the speedup
   - Interpretation

2. `examples/speculative/results/phase12b_timing.json`
   - Structured timing data for all runs

3. `examples/speculative/results/PRT_PHASE12B_VERDICT.md`
   - Where does the ~1.79x speedup actually come from?
   - Is FFN_UP replacement the primary driver?
   - Are there other contributing factors?

---

## Constraints

- Fixed seed, temp=0, same model throughout
- Sidecar path fixed: `/tmp/prt_sidecars/`
- n=100 for timing runs
- Measure 3 prompts × 3 modes = 9 runs minimum
- Do not claim upstream applicability without justification

---

## What's Known From Phase 12A

- Speedup is consistent: 1.51x–1.95x across 24 prompts
- No speedup outliers in either direction (no prompts slower)
- PRT custom op is doing real work: `prt_true_replacement_calls = 3706` per n=100 run
- L12+L15 adds 16 native fallback calls per run (2 layers × 8 tokens)

## What's Unknown

- What fraction of native wall time is FFN_UP vs other matmuls (attn, kvm, etc.)?
- Does AVX2 kernel vs scalar kernel change the speedup ratio?
- Is memory bandwidth the actual bottleneck being bypassed?
- Is there headroom for > 2x speedup with further optimization?

---

## Next Action

Implement timing instrumentation in `build_ffn` and run the 3-prompt differential suite.

---

*Plan by ELVIS for Matthew Villnave / The ForgeHQ*