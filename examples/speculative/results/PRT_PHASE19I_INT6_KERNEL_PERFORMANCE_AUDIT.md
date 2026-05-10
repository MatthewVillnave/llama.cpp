# PRT Phase 19I — INT6 Kernel Performance Audit / AVX2 Path Design

## Verdict
**RECOMMEND_INT6_PREDECODE_TO_INT8**

## Context

Phase 19H froze the 0.5.5B PRT runtime checkpoint with:
- 0.5B INT6 PRT overlay works end-to-end (24/24 sidecars load)
- Native 0.5B: ~95–97 t/s
- PRT INT6 0.5B: ~17–18 t/s (~5.5x slower)
- The bottleneck is the scalar INT6/INT8 custom-op kernel

## Kernel Path Findings

| File | Function | Purpose |
|------|----------|---------|
| `examples/speculative/prt_graph_replace.h:184` | `prt_ffn_up_custom_op()` | PRT custom op entry point |
| `examples/speculative/prt_graph_replace.h:268` | AVX2 path | `if (ud->kernel_mode == 1)` — float32 only |
| `examples/speculative/prt_graph_replace.h:290` | INT6/INT8 scalar path | `if ((ud->format == 1 \|\| ud->format == 2) && ...)` |
| `examples/speculative/prt_graph_replace.h:454` | kernel_mode gate | `ud->kernel_mode = (ud->format == 0 && g_prt_kernel_mode == 1) ? 1 : 0` |
| `src/llama-graph.cpp:1209` | format logging | Logs int6/int8/fp32 per layer |

**Key finding:** The AVX2 path is gated by `format == 0` (float32). Formats 1 (INT8) and 2 (INT6) use the scalar fallback unconditionally, regardless of `kernel_mode`.

## Current Scalar Kernel (format 1 or 2)

```
for each token:
  for j = 0..ffn-1:
    scale = int8_scales[j]
    for k = 0..hidden-1:
      s += X[k] * int8_data[j*hidden + k] * scale
    Y[j] = s
```

Issues:
1. **No vectorization** — int8→float conversion is scalar
2. **No FMA batching** — sequential MACs, ~1-2 GFLOPs scalar throughput
3. **One row at a time** — no multi-row SIMD batching possible with int8 weights
4. **Memory-bound on weights** — 4.3MB/token sequential read from int8_data

## Timing Results

| Metric | Value |
|--------|-------|
| Native 0.5B | ~95–97 t/s |
| PRT INT6 0.5B | ~17.7 t/s |
| Slowdown | ~5.5x |
| Time/token (native) | ~10.3ms |
| Time/token (PRT) | ~56.5ms |
| MACs/token | 4,358,144 |
| Weight read | 4,358,144 bytes |
| Compute intensity | 2.0 FLOPs/byte |

## Bottleneck Analysis

**Root cause:** The INT6/INT8 kernel has no AVX2 or SIMD acceleration path at all.

The scalar path processes one ffn row at a time with sequential MAC operations. At 17.7 t/s, the scalar path achieves ~300-400 MFLOPs effective — far below what a modern CPU can sustain with vectorization (~10+ GFLOPs).

**AVX2 float32 path exists but is inaccessible for INT6/INT8** because:
```cpp
ud->kernel_mode = (ud->format == 0 && g_prt_kernel_mode == 1) ? 1 : 0;
```

The format check gates AVX2 to only float32 sidecars. Formats 1 and 2 always fall through to scalar.

## AVX2 / Optimization Options

### Option 1: Predecode INT6 → INT8 at load time (RECOMMENDED)
- Keep INT6 packed on disk (smaller files)
- At `llama_set_prt_sidecar_int6()` time, expand INT6→INT8 in RAM
- Use existing INT8 scalar path (still scalar, but no per-token unpack)
- Memory cost: 4.3MB int8 vs 3.2MB int6 per layer
- Speedup: ~10-20% (eliminates per-token int6 unpack overhead)
- Does NOT enable AVX2 for int8

### Option 2: Predecode INT6 → float32 at load time
- Convert INT6 to float32 sidecar at load time
- Use existing AVX2 float32 path (kernel_mode==1, format==0)
- Memory cost: 17.3MB float32 vs 3.2MB int6 per layer (5.3x more RAM)
- Speedup: potentially 4-8x (uses AVX2)
- Trade-off: RAM vs speed

### Option 3: New INT6→float AVX2 kernel
- Decode packed 6-bit to float32 using AVX2
- Multiply by scale
- Dot with activation vector
- Complexity: significant — requires scalar unpack of 6-bit, then vectorize
- Risk: high implementation complexity

### Option 4: Predecode INT6 → float32 and convert sidecar format
- Store sidecars as float32 (not INT6)
- Always use AVX2 path
- This is essentially Option 2 with a format change
- Memory: 17.3MB/layer for 0.5B

### Option 5: Do nothing until backend integration
- Leave scalar INT6 kernel as-is for prototype phase
- Only optimize after PRT backend integration is clearer

## Recommendation

**Option 2 + 4: Predecode to float32 at load time**

For 0.5B (24 layers):
- Disk: INT6 packed format (~3.2MB/layer = 77MB total)
- RAM at runtime: float32 expansion (~17.3MB/layer = 415MB total)

This is acceptable for 0.5B. The float32 runtime path already has a well-tested AVX2 implementation. The predecode approach:
1. Load INT6 from disk
2. Expand to float32 in RAM at `llama_set_prt_sidecar_int6()` time
3. Switch format to 0 (float32) and/or set float32 sidecar pointer
4. Existing AVX2 kernel path activates automatically

**Implementation phase:** Phase 19J — INT6 predecode to float32 at load time

## Risk / Caveats

- Predecode increases RAM per layer from 3.2MB → 17.3MB (0.5B)
- Must preserve existing INT6 disk format (backwards compatibility)
- Must not break 7B/14B sidecar loading
- No speed claim until measured post-implementation
- AVX2 path still processes one row at a time in current design — multi-row batching is a further optimization

## Allowed Claims

- PRT INT6 0.5B runs ~5.5x slower than native due to scalar kernel
- The scalar bottleneck is confirmed to be the lack of AVX2 for INT6/INT8 formats
- The float32 AVX2 path exists but is inaccessible for INT6/INT8 formats
- A predecode approach would allow use of the existing AVX2 path

## Forbidden Claims

- Speedup — not measured yet
- Production readiness
- Universal performance improvement
- RAM reduction (predecode increases RAM)
- GPU comparison
- Larger model impact extrapolated from 0.5B
