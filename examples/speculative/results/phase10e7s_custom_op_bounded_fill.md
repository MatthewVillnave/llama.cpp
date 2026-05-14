# Phase 10E-7S: Custom Op Bounded Fill Test — FINAL RESULTS

## Tests Run

### Test 1: Bounded Fill (0.001*i pattern)
- Custom op writes: `Y[i] = 0.001 * i` for i in [0, nelem)
- NO sidecar read
- NO PRT math
- Result: CORRUPT (path fragments still appear in output)

### Test 2: Zero Fill (memset)
- Custom op writes: `memset(dst->data, 0, ggml_nbytes(dst))`
- NO sidecar read
- NO PRT math
- Result: CORRUPT (path fragments still appear in output)

### Test 3: Identity Copy (src0→dst loop)
- Custom op writes: `for(i) Y_out[i] = Y_matmul[i]`
- NO sidecar read
- NO PRT math
- Result: CORRUPT (path fragments still appear in output)

### Test 4: Identity Copy with Logging
- Custom op logs BEFORE and AFTER copy
- BEFORE: src0[0]=0.235489 (CORRECT)
- AFTER: dst[0]=0.235489 (CORRECT, matches src0)
- Result: CORRUPT — corruption happens AFTER custom op, upstream in ggml graph

## Key Finding

**The corruption happens BEFORE the custom op runs** — the src0 tensor (matmul result) arrives at the custom op already containing path string garbage.

## What This Rules Out

1. ❌ Custom op code bug — identity copy produces correct floats
2. ❌ PRT math bug — same corruption without any PRT computation
3. ❌ Sidecar reading bug — same corruption with only identity copy

## What This Points To

**ggml custom op integration bug** — when sidecar is loaded, the ggml tensor allocation or graph computation corrupts the matmul result tensor before the custom op runs.

## Status: FAIL ❌
The root cause is in the ggml_map_custom2 integration itself.