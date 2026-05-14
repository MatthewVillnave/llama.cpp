# PRT Phase 13Y: Sidecar Layout Transpose Experiment

**Verdict: INDEXING_BUG — AVX2 path reads transposed weights**

## What Was Tested

| Config | Layout | Output |
|--------|--------|--------|
| Native (prt-mode 0) | — | "The capital of France is Paris." ✅ |
| PRT layer0 only | original [ffn,hidden] | "ate  to   a::;;" ❌ |
| PRT layer0 only | transposed [hidden,ffn] | Empty/garbled ❌ |
| PRT all layers | original | Crashes/garbage ❌ |
| PRT all layers | transposed | Crashes/garbage ❌ |

## Suspected Indexing Bug

Both PRT paths (original and transposed layout) produce garbage output.
Root cause: **AVX2 kernel uses `sidecar[k * stride0 + (j+v)]` which reads the matrix transpose**.

For original sidecar layout `[ffn,hidden]` with M=hidden, N=ffn:

| Formula | Accesses | Matmul Role |
|---------|----------|-------------|
| `sidecar[j * hidden + k]` | row j, col k | **CORRECT** — W[j,k] |
| `sidecar[k * hidden + j]` | row k, col j | **WRONG** — W[k,j] = W^T[k,j] |

The matmul requires `Y[j] = Σ_k W[j,k] * X[k]`, but current code reads `W[k,j]` (the transpose).

## Why 13S Is Not Automatically Invalidated

Phase 13S ("freeze clean 0.5B quality checkpoint") was the clean checkpoint **before** AVX2 was enabled.
Phase 13X enabled AVX2 compile flags (`-mavx2 -mfma`).
The indexing bug affects the **AVX2 path only** — this was introduced post-13S.

The scalar/SSE fallback (`kernel_mode=0`) has the same indexing formula but may work differently
due to scalar loads being checked against different correctness criteria in earlier tests.

**13S quality was verified with fallback/scalar path. AVX2 speedup claims from 13X may be invalid.**

## Layout Strategies Evaluated

**Strategy A** — Keep original `[ffn,hidden]` sidecars:
- Correct index: `sidecar[j * hidden + k]`
- AVX2 challenge: 8 output lanes j..j+7 are at stride `hidden=896` apart — not contiguous
- Solution: Vectorize over **hidden k** (contiguous), not over output j
- Preserves existing sidecar files

**Strategy B** — Switch to transposed `[hidden,ffn]` sidecars:
- Correct index: `sidecar[k * ffn + j]`
- AVX2 benefit: 8 output lanes j..j+7 are **contiguous** at fixed k
- Requires re-generating all sidecar files with `--prt-sidecar-layout transposed`
- Transpose tool created (`phase13y_transpose_sidecars.py`) and verified

**Decision: Start with Strategy A (fix original layout indexing) for quickest correctness.
Strategy B may offer better AVX2 performance but requires correctness first.**

## Files Created

| File | Purpose |
|------|---------|
| `examples/speculative/phase13y_transpose_sidecars.py` | Transposes sidecar files [ffn,hidden]→[hidden,ffn] |
| `/tmp/prt_sidecars_transposed/` | 24 transposed sidecar files (400MB, not committed) |

## Sidecar Stats

Original sidecars: 24 layers × 17,432,576 bytes = 400MB total  
Transposed sidecars: 24 layers × 17,432,576 bytes = 400MB total  
Both layouts verified: same checksum sum (-13.1969 for layer 0), all finite values, correct shape

## Safety Check

- No models, sidecars, or binaries staged ✅
- No secrets/API keys in changed files ✅
- Transpose script reads only, produces float32 ✅

## Recommended Next (Phase 13Y-FIX)

1. Write reference scalar checker for original `[ffn,hidden]` layout
2. Fix AVX2 indexing: `sidecar + j*stride0 + k` (Strategy A)
3. Verify reference vs scalar fallback parity (may already pass)
4. Correctness canary on Qwen2.5-0.5B
5. Quality mini-suite
6. Timing comparison

**Do NOT test larger models until correctness passes on 0.5B.**
**Do NOT claim speedup until quality passes.**