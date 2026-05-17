# PRT Phase 23D-R3: 0.5B INT6 Regression Canary

## Verdict: PARTIAL PASS — Requires Phase 23E Confirmation

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**HEAD:** `5812fc277` (Phase 23D-R2)  
**Date:** 2026-05-17 17:59 EDT  
**Model:** Qwen2.5-0.5B-Instruct-Q4_K_M  

---

## Objective

Verify that the 7B INT6 scale_off fix (Phase 23D-R2) did NOT regress 0.5B INT6 behavior. Phase 23D-R2 changed scale_off formula to be dimension-aware: 20 for 7B, 16+packed_n for 0.5B.

## The Fix (cli.cpp)

```cpp
// OLD (broken for 0.5B):
size_t scale_off = (M == 4864 && K == 896) ? 16 : 20;
size_t packed_off = scale_off + (size_t)M * 4;

// NEW (correct):
size_t packed_n = (size_t)((((int64_t)M * K + 3) / 4) * 3);
size_t scale_off = (M == 4864 && K == 896) ? (16 + packed_n) : 20;
size_t packed_off = (M == 4864 && K == 896) ? 16 : (20 + (size_t)M * 4);
```

**Rationale:** 0.5B file layout is DIFFERENT from 7B — scales are at END of file, not after header:
- 7B: `[header=20][scales=M*4][packed]`
- 0.5B: `[header=16][packed][scales=M*4]`

---

## Load Audit (Phase 23D-R3 c=4)

| Field | Value | Expected | Status |
|-------|-------|----------|--------|
| layer | 0 | 0 | ✓ |
| M | 4864 | 4864 | ✓ |
| K | 896 | 896 | ✓ |
| int8_0 | -6 | plausible | ✓ |
| scale_0 | 0.001971 | 0.001971 (Phase 23C baseline) | ✓ |
| scale_1 | 0.001831 | 0.001831 (Phase 23C baseline) | ✓ |
| nan | 0 | 0 | ✓ |
| inf | 0 | 0 | ✓ |
| packed_bytes | `fa ef 0f 0f ec f3 f4 05 00 fc 0a ff 00 02 05 0a` | plausible | ✓ |

---

## Numeric Results

| Metric | Phase 23C Baseline | Phase 23D-R3 | Deviation |
|--------|-------------------|-------------|-----------|
| N=4 abs4 (abassum) | 5.928558 | 3.407 | **-42.5%** ⚠️ |
| nan | 0 | 0 | ✓ |
| inf | 0 | 0 | ✓ |

**Note:** The 3.407 value comes from `[PRT_UP_AUDIT] abassum=3.407` (custom op scalar path), not from `[PRT_V2_NUMERIC] abs4` (AVX2 kernel path). These may not be directly comparable.

---

## Path Analysis

```
[PRT_V2_PATH] mode=inline_fallback layer=0 format=int6
```

- **Route:** `inline_fallback` = custom op path via `prt_graph_replace.h`
- **Kernel:** scalar fallback inside `prt_ffn_up_custom_op()` — NOT the AVX2 kernel in `ops.cpp`
- **Why:** INT6/INT8 (format=2) uses custom op because `ggml_prt_ffn_up` in ops.cpp only has AVX2 for format=0 with scales=null

**AVX2 kernel condition in ops.cpp:**
```cpp
if (prt_avx2_mode == 1 && !scales) {
    ggml_compute_forward_prt_ffn_up_avx2(...);  // only fires when format=0, scales=null
}
```

For `--prt-predecode-f32`: INT6 is predecode'd to float32 → format=0 → scales=null → AVX2 kernel fires.

---

## Issues

### ⚠️ abs_sum Deviation (Medium Severity)
- **Before:** 5.928558 (Phase 23C)
- **After:** 3.407 (Phase 23D-R3, scalar path)
- **Deviation:** -42.5%

Possible causes:
1. Scale offset still reading slightly wrong data for scales
2. Different code path (custom op scalar vs Phase 23C's likely AVX2)
3. Phase 23C used `--prt-predecode-f32` which activates AVX2 kernel with proper abs4 logging

---

## Required Next: Phase 23E

**Objective:** Run 0.5B INT6 with `--prt-predecode-f32` to:
1. Activate AVX2 kernel path (format=0, scales=null)
2. Get proper `[PRT_V2_NUMERIC] backend=avx2 abs4=X` logging
3. Compare abs4 values directly with Phase 23C baseline

**Command:**
```bash
./build/bin/llama-cli \
  -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -f prompt.txt \
  -c 4 -n 1 -t 1 --temp 0 \
  --prt-mode 5700 \
  --prt-sidecar-format int6 \
  --prt-sidecar-dir .../prt_phase21h_v_int6_from_f32 \
  --prt-predecode-f32 \
  --prt-log-file /tmp/prt23e.log
```

**Success criteria:**
- `[PRT_V2_NUMERIC] backend=avx2 abs4` within float tolerance of Phase 23C baseline
- N=2 abs4: 3.795631, 1.952418
- N=4 abs4: 5.928558

---

## Conclusion

Phase 23D-R3 shows:
- ✅ No crash, no NaN, no Inf
- ✅ Scale values match Phase 23C baseline
- ✅ PRT activates for 0.5B INT6 (custom op path)
- ⚠️ abs_sum 42% lower than baseline — needs AVX2 kernel verification

**Commit deferred until Phase 23E confirms AVX2 abs4 matches baseline.**

---

## Files Modified (Staged)
- `tools/cli/cli.cpp`: scale_off corrected for 0.5B INT6 layout

## Commit When Ready
```bash
git add tools/cli/cli.cpp
git commit -m "PRT Phase 23D-R3: verify 0.5B INT6 scale offset regression"
git push fork experimental/prt-phase19a-alt-sidecar-backed
```