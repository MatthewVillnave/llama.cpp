# Phase 28BA: Decoded Residual Matmul Microprobe

**Status:** ✅ PASS — All test cases cleared tight floating-point tolerance

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**HEAD before:** `8addc0eb3`  
**Date:** 2026-05-23

---

## What Was Proven

Decoded `.trit` residual bytes can participate in **mathematically correct residual overlay matmul in isolation**, with no llama.cpp runtime, no model weights, no generation involved.

Specifically:
- Ternary matrices written to `.trit` via `prt_trit_io.write_trit()` round-trip through the canonical reader (`prt_trit_io.read_trit()`) with **zero loss**
- The decoded float values (`decoded_ternary.astype(np.float32) * scale`) reproduce reference matmul results within `max_err < 1e-4`

### Test Results

| Case       | K   | M   | N   | seed | max_abs_err | mean_abs_err | rmse  | Pass |
|------------|-----|-----|-----|------|-------------|--------------|-------|------|
| K=8_M=16_N=1 | 8   | 16  | 1   | 42   | 0.00e+00     | 0.00e+00      | 0.00e+00 | ✅    |
| K=17_M=19_N=4 | 17  | 19  | 4   | 43   | 0.00e+00     | 0.00e+00      | 0.00e+00 | ✅    |
| K=31_M=33_N=7 | 31  | 33  | 7   | 44   | 0.00e+00     | 0.00e+00      | 0.00e+00 | ✅    |

**All pass:** `True`

---

## What Is NOT Proven

This microprobe **does not** prove any of the following:

- ❌ Generation works or is faster with sidecar-backed residuals
- ❌ Quality parity with full-precision models
- ❌ Runtime inference correctness or throughput
- ❌ Any speedup claim (wall-clock, memory bandwidth, or compute)
- ❌ Real model weight interaction — all weights are random
- ❌ End-to-end llama.cpp integration — no generation path was exercised
- ❌ Sidecar file format correctness for production use
- ❌ Any claim beyond isolated matmul arithmetic on decoded residuals

---

## Implementation Notes

- All test weights (`X`, `W_base`) are random `float32` — no real model data
- Ternary residual: alternating ±1 pattern row-wise
- Single-scale block (block_rows=K, block_cols=M → 1 block, 1 scale)
- Canonical `prt_trit_io` used throughout — no bypass paths

---

## Files Added

- `phase28ba_matmul_microprobe.py` — self-contained proof script
- `results/phase28ba_decoded_residual_matmul_microprobe.json` — machine-readable results