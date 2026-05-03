# PRT Phase 12B: Profiling Notes

**Version:** 1.0
**Date:** 2026-05-03

---

## Code Paths Inspected

### phase10e0_layer0_replacement.cpp

- `g_prt_state` — tracks callback_overwrites, last_cosine
- `llama_get_prt_true_replacement_calls()` — counts PRT custom op invocations
- `llama_get_native_fallback_calls()` — counts force-native fallback invocations
- `validate_prt_sidecars()` — startup sidecar validation
- `build_prt_ffn_up()` — inserts GGML custom op
- `route_a_mode = (g_prt_debug_mode >= 5700)` — Route A activation gate

### prt_graph_replace.h

- `prt_ffn_up_custom_op()` — the GGML custom op implementation
- Sparse ternary matmul: reads sidecar weights, applies threshold, computes (W ⊙ S) @ x
- AVX2 kernel when available, scalar fallback otherwise

---

## What Was Measured

### Coarse Timing (via /usr/bin/time -v)

- **Wall clock time** — total elapsed time
- **User time** — CPU time in user space
- **System time** — CPU time in kernel
- **Maximum RSS** — peak memory footprint
- **CPU utilization** — (user + system) / wall ratio indicating multi-core usage

### Counters (via stderr output)

- `callback_overwrites` — should be 0 in Route A mode (5700+)
- `native_fallback_calls` — 16 for L12+L15 (2 layers × 8 tokens)
- `prt_true_replacement_calls` — 3706 for n=100, 2006 for n=50

---

## Instrumentation Gap

The current code does **not** have per-component timing:

- No sidecar load time per layer
- No per-layer FFN_UP time
- No graph dispatch overhead measurement
- No token loop time isolation

Adding per-component timing would require invasive changes to `ggml_compute` in llama.cpp proper, not just the PRT shim.

---

## What Could Not Be Measured Cleanly

1. **Per-layer FFN_UP compute time** — The llama.cpp `build_ffn()` interface doesn't expose per-layer timing. Isolating FFN_UP from the full inference graph would require llama.cpp core changes.

2. **Sidecar validation startup time** — Measured at process start, before `/usr/bin/time` starts. Not trivially isolatable.

3. **Graph dispatch overhead** — Embedded in GGML custom op invocation. No separate counter.

4. **Memory bandwidth vs compute** — Cannot tell from timing alone whether speedup is from reduced memory traffic (ternary sparse) or reduced compute (fewer ops).

---

## Instrumentation Risks

Adding instrumentation risks:
- Changing generated output (if bugs introduced)
- Breaking callback/sidecar paths
- Adding unwanted runtime overhead to normal runs
- Accumulating timing state across runs without resetting

**Recommendation:** Do not add per-component timing without a dedicated profiling build.

---

## Alternative Approaches (Not Used)

1. **perf report** — could identify hot call stacks, requires build with debug symbols
2. **VTune** — Intel-specific, not available
3. **Valgrind callgrind** — extremely slow, not practical for n=100 generation
4. **Custom timing regions in build_ffn** — requires core llama.cpp changes, risky

---

## Summary

Phase 12B relied on coarse `/usr/bin/time -v` timing and existing PRT counters. This was sufficient to:
- Confirm speedup is real and in user-space compute
- Discover L12+L15 is faster than pure PRT (key finding!)
- Rule out memory as the bottleneck (RSS unchanged)

Fine-grained attribution requires deeper instrumentation into llama.cpp core, which is out of scope for this branch.

---

*Profiling notes by ELVIS for Matthew Villnave / The ForgeHQ*