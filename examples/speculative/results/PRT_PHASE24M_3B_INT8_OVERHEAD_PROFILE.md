# PRT Phase 24M: 3B INT8 Overhead Profile

## Date
2026-05-20 17:19+

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Previous HEAD
da62fcefe

## New HEAD
7a028cebf (plus instrumentation)

## Phase 24L Baseline
- Native avg: 2.40s
- PRT avg: 3.44s
- Delta: ~1.04s

## Profile Run (c=4 n=8, 3B canonical INT8)

### Layer 0 Decode Profile (first invocation)
- `sidecar_read_us=0` — file already in OS page cache
- `int8_decode_us=7046` — fread 22MB sidecar + scales
- `int8_loop_us=128711` — K*M decode loop (2048*11008 = 22.5M iterations)
- `total_us=135757` — **~136ms for INT8→f32 decode**
- `op_call_us=1` — kernel execution essentially FREE (0.001ms)

### Calls per run
- `R3_ENTRY` appears for IL=0 on every graph build call (every prefill/decode)
- `ggml_prt_ffn_up` called with n_tokens=1, 2, 8, 16 at various stages
- `op_call_us` consistently 0-1μs regardless of n_tokens

### Overhead Breakdown (first invocation)
| Stage | Time | % of delta |
|-------|------|-------------|
| Sidecar fread | ~0μs | cached |
| Scales+INT8 fread | 7ms | ~0.7% |
| Decode loop (K*M) | 129ms | ~12.4% |
| Kernel call | ~0ms | ~0% |
| **Total** | **~136ms** | **~13% of 1.04s delta** |

### Observation
The decode loop accounts for ~130ms. The remaining ~900ms of the 1.04s delta is NOT from the decode loop. It must be from:
1. Repeated R3_ENTRY check overhead (37 layers * multiple calls per run)
2. Graph build overhead (build_lora_mm vs ggml_prt_ffn_up path differences)
3. Memory allocation patterns

### Kernel is NOT the bottleneck
`op_call_us=0-1` consistently. The custom op itself is fast.

### Sidecar cached after first load
`sidecar_read_us=0` on subsequent calls means the file stays in OS page cache. The decode loop (129ms) re-runs on every graph build though — it's not cached per se.

## Verdict
PARTIAL_OVERHEAD_SOURCE_IDENTIFIED

The decode loop is the identifiable overhead source (~136ms), but it only explains ~13% of the total 1.04s delta. The rest (~900ms) is unaccounted and likely from:
- Repeated layer eligibility checks across 36 layers on every call
- Graph build path divergence (build_lora_mm vs ggml_prt_ffn_up)
- Memory allocation/fragmentation

## Allowed Claims
- INT8 decode loop is the main identified overhead (~130ms per decode)
- Kernel execution is essentially free (1μs)
- The remaining ~900ms overhead source is unclear

## Forbidden Claims
- No optimization recommendations without further profiling
- No production performance claims

## Recommended Next
Phase 24N — instrument the repeated R3_ENTRY call overhead to identify where the other ~900ms goes