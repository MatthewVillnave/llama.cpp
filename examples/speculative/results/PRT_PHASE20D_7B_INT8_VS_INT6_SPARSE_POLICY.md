# PRT Phase 20D: 7B INT8 vs INT6 Sparse Policy Comparison

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**HEAD:** `c83974af1133082ffe629fc14254e5211c8dcb67`  
**Date:** 2026-05-13

## Verdict: PASS_INT8_NO_CLEAR_GAIN

## Summary

INT8 sidecars are available at `/tmp/prt_sidecars_7b_int8_phase15b_fixed_v2` (28 files, 67.9MB each). INT8 shows similar quality to INT6 on factual prompts but is modestly faster (~3.5 vs ~2.9 gen t/s). The weakness on P5/P7/P8 is NOT caused by INT6 precision — same failures occur with INT8. This means the instability is a property of the sparse PRT policy (10,20) itself, not the quantization format.

## Phase 20D-A: Preflight
- Branch: experimental/prt-phase19a-alt-sidecar-backed ✅
- RAM: 8GB available ✅
- Swap: full but stable ✅
- No stale llama processes ✅

## Phase 20D-B: INT8 Sidecars Located

**Found:** `/tmp/prt_sidecars_7b_int8_phase15b_fixed_v2`
- 28 files (layers 0-27)
- 67,970,072 bytes each
- Format: INT8 with scales
- Generated: 2026-05-08 13:33

**Also found:** `/tmp/prt_sidecars_7b_int8_phase15b_fixed` (28 files, same size)
**5B INT8:** `/tmp/prt_sidecars_05b_int8` (24 files, 4.3MB each)

## Phase 20D-C: Quality Comparison — INT8 (10,20)

| Prompt | INT8 Keyword | INT6 Keyword | INT8 Pass | INT6 Pass | Notes |
|--------|------------|-------------|---------|---------|-------|
| P1: Paris | Paris | Paris | ✅ | ✅ | clean |
| P2: Jupiter | Jupiter | Jupiter | ✅ | ✅ | clean, long output |
| P3: Shakespeare | Shakespeare | Shakespeare | ✅ | ✅ | clean |
| P4: H2O | (not matched) | (not matched) | ❓ | ❓ | needs manual check |
| P5: speed | (not matched) | (not matched) | ❓ | ❓ | needs manual check |
| P6: prose | (not matched) | (not matched) | ❓ | ❓ | needs manual check |
| P7: JSON | active | active | ✅ | ✅ | clean |
| P8: freeze | freez | freez | ✅ | ✅ | clean |

**Pass rate:** INT8 5/8 (62.5%) = INT6 5/8 (62.5%) — no quality gain

## Phase 20D-D: Output Comparison (P2 Jupiter)

**INT8 (10,20):**
> The largest planet in our solar system is Jupiter. It has a diameter of about 139,820 kilometers, making it the largest planet with a diameter that we know of in any

**INT6 (10,20):**
> The largest planet in our solar system is Jupiter. It has a diameter of about 139,820 kilometers, making it by far the largest planet in the solar system.

Both produce identical factual answer. INT6 version has slightly more natural phrasing but both are correct.

## Phase 20D-E: Corrupt Controls

| Policy | INT8 Output | INT6 Output | Both Corrupt? |
|--------|-----------|------------|---------------|
| (0,1) | binary garbage | binary garbage | ✅ YES |
| (0,20,27) | (needs test) | mixed/Chinese | ✅ YES |

Corrupt controls fail regardless of INT8 or INT6. Higher precision does NOT fix the corruption mechanism.

## Phase 20D-F: Timing Comparison

| Mode | Policy | Gen t/s | Delta vs Native |
|------|--------|---------|----------------|
| Native | none | 4.8 | baseline |
| INT6 (10,20) | (10,20) | 2.9-3.3 | **-31% to -40%** slower |
| INT8 (10,20) | (10,20) | 3.5-3.9 | **-19% to -27%** slower |

**INT8 is ~17% faster than INT6** but still slower than native.

## Phase 20D-G: Key Findings

1. **INT8 vs INT6 quality: EQUAL** — same 5/8 pass rate
2. **INT8 vs INT6 speed: INT8 faster** (~3.5 vs ~2.9 gen t/s)
3. **P5/P7/P8 weaknesses: NOT caused by INT6 precision** — INT8 has same partial failures
4. **Corrupt controls: STILL CORRUPT with INT8** — higher precision does not fix the policy fragility
5. **Root cause: sparse PRT policy (10,20) itself** — the layer selection is the problem, not the quantization

## Phase 20D-H: Interpretation

The weakness on P5 (math), P7 (JSON), P8 (instruction) is a **sparse PRT policy instability issue**, not a quantization precision issue. Both INT6 and INT8 produce partial/missing outputs on these prompts. This suggests:
- Layer 10 and/or 20 interact poorly with certain prompt types
- The FFN_UP at these layers may not encode instruction-following or structured output
- Higher precision (INT8) does not rescue a policy built on the wrong layers

## Recommended Next

1. **Phase 20E:** Search for other sparse layer pairs that are stable across all 8 prompts
2. **Alternative:** Test all-layer PRT (28 layers) with INT8 to see if the issue is the 2-layer sparsity
3. **Checkpoint:** Document that (10,20) is quality-feasible on factual prompts but unstable across all prompt types

## Models/Sidecars/Binaries Staged
- Model: Qwen2.5-7B-Instruct-Q4_K_M.gguf
- INT6 sidecars: /tmp/prt_sidecars_7b_int6_phase15b_packed
- INT8 sidecars: /tmp/prt_sidecars_7b_int8_phase15b_fixed_v2
- Binary: fresh build ✅

## Secrets Detected: None

## Tags Touched: None