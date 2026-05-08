# PRT Phase 14B — INT8 Runtime Canary

## Verdict: BLOCKED_SWAP_EXHAUSTION

## Summary

Phase 14B implemented the INT8 sidecar loading and runtime compute path. Build succeeded. Runtime canary tests blocked due to swap exhaustion (99.99% used).

## Implementation Status

### Runtime format — IMPLEMENTED ✅

- **CLI flag added**: `--prt-sidecar-format float32|int8` (default: float32)
- **Float32 path**: Unchanged, works via `.bin` files
- **INT8 path**: Implemented, loads `.int8` files with per-row scales

### Runtime format details

- `--prt-sidecar-format float32`: Uses traditional `.bin` files (float32)
- `--prt-sidecar-format int8`: Uses `.int8` files with appended scales

### File layout (INT8)

- File suffix: `.int8`
- Format: `[M*K bytes int8 data][M*4 bytes float32 scales]`
- Layout: Row-major [M][K] where M=ffn_dim, K=hidden_dim
- Scale scheme: Per-row (one float32 scale per output dimension)
- Shape validation: File size inference (supports 0.5B and 3B)

### INT8 loader — IMPLEMENTED ✅

- File detection: `.int8` extension
- Scale loading: Reads M*4 bytes appended after int8 data
- mmap support: NO (future work)
- Memory-efficient: YES (22.5MB/layer vs 90MB/layer for float32)

### File naming

- Float32: `ffn_up_layer{N}_prt.bin`
- INT8: `ffn_up_layer{N}_prt.int8`

### INT8 runtime compute path — IMPLEMENTED ✅

- Dequantization: Per-row scalar computation (Y[j] = scale[j] * sum_k(int8[j*K+k] * X[k]))
- AVX2: NOT implemented (scalar path for correctness-first)
- Format tracking: Per-layer format flag (float32/int8)
- Fallback: Uses existing scalar float32 path for float32 sidecars

### Files modified

| File | Changes |
|------|---------|
| common/common.h | +1 line (prt_sidecar_format) |
| common/arg.cpp | +11 lines (--prt-sidecar-format) |
| tools/cli/cli.cpp | +139/-28 lines (INT8 loader) |
| src/llama.cpp | +36 lines (INT8 setter API) |
| src/llama-graph.cpp | +9 lines (INT8 globals) |
| examples/speculative/prt_graph_replace.h | +73/-62 lines (INT8 kernel path) |

## Build status

- **Build**: ✅ SUCCESS
- **Binary**: `build/bin/llama-cli` (5.9MB)

## Generated sidecars (NOT committed)

### 3B INT8 sidecars (36 layers)

- Location: `/tmp/prt_sidecars_3b_int8/`
- Count: 36 layers (0-35)
- Compression: 3.969x (90.2MB → 22.7MB per layer)
- Weight cosine: ~0.999 (all layers ≥ 0.9999)

### 0.5B INT8 sidecars (16 layers)

- Location: `/tmp/prt_sidecars_05b_int8/`
- Count: 16 layers (0-15)
- Compression: 3.93x (17.4MB → 4.4MB per layer)
- Weight cosine: ~0.999 (all layers ≥ 0.9999)

## Offline parity (verification before blocking)

Based on Phase 14A/14B quantization microbench:

- **Tested layers**: 3B (layers 0, 23, 35), 0.5B (layers 0, 15)
- **Output cosine**: ≥ 0.9999 (pass)
- **Relative L2**: ≤ 0.02 (pass)
- **Max abs error**: ≤ 0.003

## Runtime canary — BLOCKED

Cannot run canary tests due to:

```
Swap exhaustion: 4.0Gi / 4.0Gi (99.99% used)
Available RAM: 9Gi
```

## What this phase proves

- INT8 sidecar format design is correct
- Build compiles and links
- 0.5B and 3B INT8 sidecars can be generated offline
- Offline microbench shows parity ≥ 0.999

## What this phase does NOT prove

- Cannot run runtime canary (blocked by swap exhaustion)
- No generation quality validation
- No timing benchmarks
- Cannot verify INT8 at runtime

## Recommended next phase

**Phase 14C: INT8 runtime canary after memory recovery**

Options:
1. Retry Phase 14B when swap/RAM available
2. Add int8 mmap support
3. Add AVX2 INT8 kernel path

## Safety scan

- **Models/sidecars staged**: NO ✅
- **Secrets**: NONE ✅
- **Phase 13 tags untouched**: YES ✅

## Git status

```
 common/arg.cpp                           |  11 +++
 common/common.h                          |   1 +
 examples/speculative/prt_graph_replace.h |  73 +++++++++++-----
 src/llama-graph.cpp                      |   9 ++
 src/llama.cpp                            |  36 ++++++++
 tools/cli/cli.cpp                        | 139 ++++++++++++++++++---------
 6 files changed, 207 insertions(+), 62 deletions(-)
```

Branch: `experimental/prt-phase14a-packed-sidecars`
Previous HEAD: `a9c198950`
Commit: (not committed — source changes ready)