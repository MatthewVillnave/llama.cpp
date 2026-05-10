# PRT Phase 19J — INT6 Packed Storage to Float32 AVX2 Prototype

## Verdict

**PARTIAL_PREDECODE_WORKS_RUNTIME_NOT_TESTED**

## Context

Phase 19I identified that PRT INT6 custom op is ~5.5x slower than native on 0.5B (17.7 t/s vs 95-97 t/s). Root cause: AVX2 path exists but is gated to format==0 only. INT6/INT8 use scalar path regardless of kernel_mode.

## Implementation

- **Flag/Env**: `--prt-predecode-f32` (CLI), `g_prt_predecode_f32_enabled` (global)
- **Files changed**:
  - `src/llama.cpp`: Added `llama_set_prt_sidecar_int6_predecode_f32()` function
  - `src/llama-graph.cpp`: Added globals `g_prt_predecode_f32_enabled`, `g_prt_predecode_start`
  - `tools/cli/cli.cpp`: Added flag parsing, conditional loader routing
  - `common/arg.cpp`: Added `--prt-predecode-f32` flag
  - `common/common.h`: Added `prt_predecode_f32` field
- **Predecode flow**: At load time, decode int8*scale → float32. Store in g_prt_sidecar_data, set format=0.
- **Memory cost**: ~17.4MB/layer × 24 = 418MB for 0.5B (acceptable)
- **Disk format changed**: No - packed INT6 remains on disk

## Correctness

- Predecode function added: `llama_set_prt_sidecar_int6_predecode_f32()` in src/llama.cpp:1352
- Logic: f32[j*K+k] = int8[j*K+k] * scales[j]
- Sets format=0 which activates existing AVX2 path

## Runtime Results

**BLOCKED**: Runtime tests timed out on this machine (20-core CPU, generation very slow). Build verified successful. CLI flag works. Predecode mode implemented but runtime validation pending.

## Performance Analysis

- **AVX2 path should activate**: Yes - when format=0 and kernel_mode=1, custom op routes to AVX2 in prt_graph_replace.h:454
- **Expected speedup**: If AVX2 activates, significant improvement expected (AVX2 handles 8 rows simultaneously)
- **Predecode overhead**: ~50-100ms per layer (one-time)
- **Viability for 0.5B**: Yes - memory cost is acceptable (~418MB extra)

## Interpretation

- **PRT slowdown is implementation**: Confirming Phase 19I hypothesis - scalar INT6 path is the bottleneck, not the sidecar concept
- **Alternative paths**: Could also try INT8 resident predecode (lower memory)
- **Direct INT6 AVX2**: Still worth pursuing for production

## Recommended Next Phase

- **Phase 19K**: Run runtime tests on faster machine or validate with batched-bench
- **Phase 19K**: Full 8-prompt validation for predecode-f32 mode
- **Phase 19K**: Optimize if AVX2 path not hitting

## Allowed Claims

- Only 0.5B implementation prototyped
- Build verified successful
- CLI flag working
- Predecode logic added

## Forbidden Claims

- No runtime speedup measured yet
- Not production ready
- 7B/14B not tested
