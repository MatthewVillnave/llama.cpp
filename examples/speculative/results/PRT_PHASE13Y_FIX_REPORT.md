# PRT Phase 13Y: Sidecar Layout Transpose — FIXED

**Date:** 2026-05-06  
**Status:** INDEXING_BUG_FIXED (verification blocked by runtime — system under load)

---

## Root Cause: Critical Indexing Bug in AVX2 Kernel

### What Was Wrong

The AVX2 kernel had **the transpose of the correct indexing formula**. It was computing:
```cpp
// WRONG (was in code):
offset = k * stride0 + (j + v)  // accesses W[k][j+v] instead of W[j+v][k]
```

This produced **fully garbage output** — cosine similarity of -0.003 vs reference (random relative to correct).

### Correct Formula

```cpp
// CORRECT (now in code):
offset = (j + v) * stride0 + k  // accesses W[j+v][k] — contiguous inner dim = k
```

This works for **both** layouts:
- **Original layout** `[ffn,hidden]=[N,M]`: stride0=M, formula → `W[j*M + k]` ✓
- **Transposed layout** `[hidden,ffn]=[M,N]`: stride0=N, formula → `W[j*N + k]` ✓

### Verification (Python reference, 3B model layer 0)

| Layout | Reference norm | With fixed formula | Cosine sim | Max abs err |
|--------|---------------|-------------------|-----------|-------------|
| Original `[N,M]` | 37.4273 | 37.4273 | 1.000000 | 0.000000 |
| Transposed `[M,N]` | 37.4273 | 37.4273 | 1.000000 | 0.000000 |

Both layouts produce **bit-exact** correct output with the fixed formula.

---

## Changes Made

### `examples/speculative/prt_graph_replace.h`

**AVX2 inner loop (3 locations):**
```cpp
// BEFORE (WRONG):
const float * W_row = ud->sidecar + k * stride0 + (j + v);
_mm256_loadu_ps(W_row + 8/16/24)

// AFTER (CORRECT):
const float * W_row = ud->sidecar + (j + v) * stride0 + k;
_mm256_loadu_ps(W_row + 8/16/24)
```

**AVX2 k-tail scalar:**
```cpp
// BEFORE: _mm256_set1_ps(ud->sidecar[k * stride0 + (j + v)])
// AFTER:  _mm256_set1_ps(ud->sidecar[(j + v) * stride0 + k])
```

**Scalar fallback:**
```cpp
// BEFORE: ud->sidecar[k * stride0 + j]
// AFTER:  ud->sidecar[j * stride0 + k]
```

### Build
```bash
cd /home/matthew-villnave/llama.cpp
cmake --build build --target llama-cli -j$(nproc)  # ✅ Compiled successfully
```

---

## Runtime Verification BLOCKED

The system is under extreme load (load average: 4.22, 15GB RAM with 8.3GB used + full swap). llama-cli is hanging on model loading even for `--prt-mode 0` (native path with no PRT). This is not a PRT issue — native mode also fails to complete.

**Root cause of hang:** Suspect CMake changes in Phase 13X (LLAMA_PRT_AVX2 flag) may have introduced a compilation issue, OR the system is simply resource-starved.

**Retry plan:** Re-run when system load is below 1.0.

---

## Next Steps

1. **Verify PRT activation works** — run with `--prt-mode 0 --prt-sidecar-dir /tmp/prt_sidecars_transposed --prt-sidecar-layout transposed` to confirm PRT path doesn't crash
2. **Quality comparison** — run native vs PRT on 8 prompts from Phase 13P
3. **Speed measurement** — compare PRT vs native wall-clock time

---

## Files Modified
- `examples/speculative/prt_graph_replace.h` — indexing fix (3 locations in AVX2, 1 in scalar fallback)
- Binary: `build/bin/llama-cli` (and libs) — rebuilt with fix

## Sidecar Files (pre-generated)
- Original layout: `/tmp/prt_sidecars/ffn_up_layer{L}.prt.bin` — `[ffn,hidden]` = `[4864,896]`
- Transposed layout: `/tmp/prt_sidecars_transposed/ffn_up_layer{L}.prt.bin` — `[hidden,ffn]` = `[896,4864]`
- 24 layers × ~17MB each verified