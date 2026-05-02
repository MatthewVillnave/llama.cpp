# PRT Route A RC1 Tag Notes

**Tag Name:** `prt-route-a-rc1`
**Date:** 2026-05-02
**Commit:** (to be recorded at tag time)
**Branch:** phase11bm-archived
**Status:** Release-Candidate — NOT PRODUCTION READY

---

## What This Tag Represents

A frozen snapshot of the Route A + L12+L15 native fallback implementation, validated across Phase 11BM (mini-suite) and Phase 11BN (controlled benchmark).

**This is a research/development tag.** Do not confuse with a production release.

---

## How to Use This Tag

### Enable PRT Route A + L12+L15:
```bash
./build/bin/llama-prt-posix \
  -m model.gguf \
  -p "Once upon a time in a" \
  -n 100 \
  --prt-mode 5700 \
  --prt-force-native 12,15
```

### What This Does:
1. Loads PRT sidecar matrices for all 36 layers
2. Route A: Replaces native FFN_UP matmul with PRT custom op for 34 layers
3. Force-native: L12 and L15 use native FFN_UP (no PRT)
4. Speedup: ~1.84x vs pure native at n=100

---

## What's Included

### Core Implementation
- `src/llama-graph.cpp` — Route A hook in `build_ffn()`
- `src/llama.cpp` — Force-native API + counter getters
- `examples/speculative/prt_graph_replace.h` — GGML custom op
- `examples/speculative/prt_avx2_kernel.h` — AVX2 SIMD

### Benchmark Binary
- `examples/speculative/phase10e0_layer0_replacement.cpp`
- Binary: `build/bin/llama-prt-posix`

### Sidecars
- `/tmp/prt_sidecars/ffn_up_layer{L}_prt.bin` for L=0..35
- 2048 × 11008 float32 per file

---

## Known Limitations

| Limitation | Severity | Note |
|------------|----------|------|
| Missing sidecar = silent fallback | MEDIUM | Should add startup check |
| JSON/structured not fully tested | MEDIUM | Resource-limited machine |
| Broader suite not validated | LOW | 6 prompts tested |
| AVX2 only (no AVX-512) | INFO | Falls back to scalar |

---

## Test Results (Phase 11BN)

| Metric | Value |
|--------|-------|
| Prompts tested | 6 |
| Average speedup | ~1.84x |
| Known failure fixed | YES |
| callback_overwrites | 0 (all runs) |
| Collapse/repetition | None |
| Memory stable | YES |

---

## How to Reproduce

```bash
# 1. Extract sidecars (if not already done)
./build/bin/llama-prt-ffn-up-extract -m model.gguf

# 2. Run benchmark
./build/bin/llama-prt-posix \
  -m model.gguf \
  -p "Once upon a time in a" \
  -n 100 \
  --prt-mode 5700 \
  --prt-force-native 12,15
```

Expected output:
- `[11BD] callback_overwrites: 0`
- `[11BD] native_fallback_calls: 16`
- `[11BD] prt_true_replacement_calls: 3706`
- Speedup ~1.85x vs native

---

## Allowed Claims for RC1

- "Release-candidate experimental branch"
- "Approximately 1.84x speedup on tested prompts"
- "Tested on narrative, code, factual, and JSON prompts"
- "Production-candidate fallback policy (not production-ready)"
- "Callbacks disabled, no overwrite/correction path"

## Forbidden Claims for RC1

- "Production-ready"
- "Universally safe"
- "2x speedup guaranteed"
- "Validated on all prompts"

---

## Next Steps After RC1

1. Code review + merge to experimental branch
2. Add missing-sidecar startup validation
3. Full suite on >32GB machine
4. JSON/structured validation at n=50+
5. Performance profiling (where is time spent?)

---

*End of RC1 Tag Notes*
