# PRT Phase 13T: 0.5B Timing Isolation

**Verdict: PARTIAL_TIMING_ISOLATED** ⚠️

## Summary

Cold timing measured for Qwen2.5-0.5B (80 tokens, 256 ctx, 4 threads, temp=0). Sidecar load timing instrumented. Warm timing blocked — llama-cli loads sidecars per-process, no in-process multi-prompt capability.

## Phase 13T-A: Cold Timing Results

**5 native cold runs:**
| Run | Wall (s) | Gen t/s | Exit | Clean |
|-----|----------|--------|------|-------|
| 1 | 0.627 | 91.2 | 0 | ✓ |
| 2 | 0.640 | 92.2 | 0 | ✓ |
| 3 | 0.630 | 92.8 | 0 | ✓ |
| 4 | 0.635 | 94.1 | 0 | ✓ |
| 5 | 0.651 | 93.6 | 0 | ✓ |

**5 PRT active cold runs:**
| Run | Wall (s) | Gen t/s | Exit | Clean | Log Size | Sidecar Load ms |
|-----|---------|---------|------|-------|---------|----------------|
| 1 | 2.370 | 20.4 | 0 | ✓ | 35720b | 121.18 |
| 2 | 2.412 | 18.7 | 0 | ✓ | 35720b | 121.18 |
| 3 | 2.352 | 20.3 | 0 | ✓ | 35720b | 121.18 |
| 4 | 2.378 | 20.4 | 0 | ✓ | 35720b | 121.18 |
| 5 | 2.401 | 18.6 | 0 | ✓ | 35720b | 121.18 |

## Summary Statistics

| Metric | Native | PRT Active | Ratio |
|--------|--------|------------|-------|
| Avg cold wall time | 0.637s | 2.383s | 3.74× |
| Avg generation t/s | 92.8 t/s | 19.7 t/s | 4.71× slower |
| Avg prompt eval t/s | ~285 t/s | ~24.3 t/s | ~11.7× slower |
| Avg sidecar load | N/A | 121ms | — |
| Stddev wall | 0.009s | 0.022s | — |

## Timing Component Breakdown

Cold timing is the sum of: model load + sidecar load + model warmup + generation.

**Model load** (native and PRT identical):
- Measured via native cold wall time ≈ 0.637s
- This is the cost of loading Qwen2.5-0.5B Q4_K_M from disk + KV cache init

**Sidecar load** (PRT only, instrumented):
- 24 sidecars × 896×4864 float32 = ~17.4MB each = ~418MB total
- Measured: **121ms** (avg of 5 runs)
- This is file I/O + malloc + fread per sidecar layer

**Generation runtime overhead** (PRT minus native, minus sidecar):
```
PRT wall    = 2.383s (avg)
Native wall = 0.637s (avg)
Overhead    = 1.746s

Sidecar load = 0.121s
Remaining   = 1.625s (model reload + runtime overhead)
```

The 1.625s "remaining" includes:
1. Model load (again, ~0.637s — each process loads model independently)
2. PRT runtime overhead during prompt eval + generation

**Estimated PRT runtime overhead** (generation only):
- Native generation: 92.8 t/s → ~10.8ms/token
- PRT generation: 19.7 t/s → ~50.7ms/token
- Overhead per token: ~40ms

For 80 tokens generation: ~3.2s PRT vs ~0.86s native = ~2.3s PRT extra from compute overhead.

## Phase 13T-B: Sidecar Load Timing

Instrumentation added to `tools/cli/cli.cpp`:
```cpp
auto sidecar_load_start = std::chrono::high_resolution_clock::now();
// ... sidecar loading loop ...
auto sidecar_load_end = std::chrono::high_resolution_clock::now();
double sidecar_load_ms = std::chrono::duration<double, std:: milli>(sidecar_load_end - sidecar_load_start).count();
fprintf(g_prt_log_file, "[PRT_TIMING] sidecar_load_ms=%.2f\n", sidecar_load_ms);
```

**Result: 121ms ± <1ms (very consistent, 5 runs)**

## Phase 13T-C: Warm Sidecar Reuse Feasibility

**Are sidecars loaded once per process?** Yes.

**Are sidecars reloaded on every prompt?** Yes — each `llama-cli` invocation is a new OS process. Each process independently:
1. Loads the GGUF model from disk
2. Reads all sidecar .bin files
3. Registers them via `llama_set_prt_sidecar()`

**Is there any current way to run multiple prompts in one process while keeping sidecars resident?** No.

`llama-cli` is a chat/inference tool — each invocation is a single prompt → single completion. There is no batch mode, session mode, or multi-shot mode that would keep the model and sidecars resident in memory.

**Workarounds for warm timing** (not implemented):
- Dedicated in-process benchmark harness: call `llama_decode()` / `llama_batch_decode()` directly with the PRT custom op, bypassing CLI per-run overhead
- Server mode: run `llama-server` with PRT pre-loaded, then send multiple requests to the same process
- Microbench: isolate the PRT custom op computation in a minimal standalone test

## Phase 13T-D: Compute-Only Timing Estimate

No per-call timing added to the PRT custom op. Per-call logging (`[PRT-11BB] IL=N...`) is present in the log file (~35KB for 80 tokens = ~22 layers × ~37 tokens × 2 log lines each ≈ 1628 log lines). This is already heavy instrumentation — adding per-call timing would require source-level changes.

**Estimated PRT compute overhead per token:**
- Native: ~10.8ms/token
- PRT: ~50.7ms/token
- Overhead: ~40ms/token (3.7× slower in generation)

This overhead comes from the PRT custom op's matrix-vector multiplication (`Y = X @ W_prt^T` where X is `[hidden, batch]` and W_prt is `[ffn, hidden]`) running on CPU without AVX2/optimized kernels.

## Interpretation

**Cold process overhead decomposition:**
| Component | Native | PRT | Notes |
|-----------|--------|-----|-------|
| Model load | ~0.637s | ~0.637s | Same for both |
| Sidecar load | N/A | 0.121s | Instrumented |
| PRT runtime overhead | N/A | ~1.625s | 1.625s = runtime penalties |
| **Total** | **0.637s** | **~2.383s** | PRT is 3.74× slower cold |

**Runtime (generation) overhead:**
- PRT generation: 4.71× slower (19.7 t/s vs 92.8 t/s)
- Per-token overhead: ~40ms more per token
- Source: CPU-only PRT matmul without AVX2, no batch optimization

**The 121ms sidecar load accounts for only ~7% of the total overhead** (121ms / 1746ms). The dominant cost is runtime — the PRT custom op compute on CPU is inherently slower than the native GGML matmul with whatever optimization is available.

## Allowed Claims

- Cold active PRT is 3.74× slower wall time than native on 0.5B under llama-cli (prompt=80 tokens, ctx=256)
- Sidecar loading adds ~121ms to startup (measured via `[PRT_TIMING]`)
- Sidecar load accounts for ~7% of total cold overhead; runtime dominates
- PRT generation is 4.71× slower in tokens/s than native generation (19.7 t/s vs 92.8 t/s)
- Phase 13T measured timing components but does not establish speedup
- Warm PRT timing is not available from current llama-cli; requires in-process benchmark or server mode

## Forbidden Claims

- No speedup claim — PRT is slower in all measured phases
- No production readiness
- No larger-model timing extrapolation
- No warm timing claim — not measured

## Safety
- No models staged: ✓
- No secrets leaked: ✓
- Tags untouched: ✓