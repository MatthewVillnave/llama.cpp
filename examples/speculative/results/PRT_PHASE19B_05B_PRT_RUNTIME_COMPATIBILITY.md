# PRT Phase 19B: 0.5B PRT Runtime Compatibility

## Phase 19B Summary

**Goal:** Fix 0.5B sidecar acceptance and activate PRT compute path

**Status:** PARTIAL PASS - Infrastructure works, but PRT custom op has shape mismatch

---

## A. Preflight ✅
- Branch: experimental/prt-phase19a-alt-sidecar-backed
- Machine health: OK

## B. Sidecar Size Fix ✅
- Added 0.5B size (3288080 bytes) to INT6 loader paths in cli.cpp
- Fixed condition to check BOTH g_prt_sidecar_data AND g_prt_int8_data

## C. PRT Compute Logging ✅
- Added [PRT_COMPUTE] logging in build_ffn()

## D. Sidecars Generated ✅
- 24 INT6 sidecars generated (one per layer)
- Format: PRT6 magic, 16-byte header, M=4864, K=896
- Each ~3.2MB

## E. Runtime Test Results

**Tiny Canary:**
- Loaded: 24/24 sidecars
- PRT compute path: ACTIVATED ✅ (first time!)
- Log shows: `[PRT_COMPUTE] layer=0 mode=int6 hit=1`
- Output: "The capital of France is Paris" ✅ (correct)
- Generation speed: ~95 t/s (native ~94 t/s)

**Crash:**
- GGML shape assertion failed during PRT custom op execution
- Cause: Tensor shape mismatch in PRT custom op
  - GGUF FFN_UP: {K=896, M=4864}
  - Sidecar: {M=4864, N=896} (possibly transposed)

## F. Root Cause Analysis

Found and fixed TWO issues:
1. **Loader:** 0.5B sidecar size (3,288,080 bytes) not in hardcoded size list → Added
2. **Condition:** `g_prt_sidecar_data[il]` is NULL for INT6 (float data), but INT6 uses `g_prt_int8_data[il]` → Changed condition to check BOTH

Now PRT path IS activated (confirmed by PRT_COMPUTE log). The crash is in the CUSTOM OP's tensor handling, not the infrastructure.

---

## Verdict

**PARTIAL_05B_SIDECAR_ACCEPTED_COMPUTE_ACTIVATED** 
- 24/24 sidecars loaded ✅
- PRT compute path activated ✅  
- Output correct before crash ✅
- Shape mismatch in custom op causes crash ❌

## Files Modified

- tools/cli/cli.cpp (size lookup fix)
- src/llama-graph.cpp (condition fix + logging)

## Next Steps

1. Fix tensor shape in PRT custom op (transpose handling)
2. Re-test to get clean output before committing
3. Proceed to Phase 19C (sidecar-backed substitution)

---

*Date: 2026-05-10*
*Commit: 7ef80c13a + fixes*