# Phase 10E-7R Memory Safety Verdict

## Final Response Format

1. **Root cause found:** **PARTIAL** - Identified correlation, not yet confirmed root cause

2. **Root cause:** 
When sidecar = (nil): identity fallback produces CORRECT output ("Hello").
When sidecar = loaded (non-null): PRT active produces GARBAGE ("amup/prt_sidecars/...").
The issue is specifically with sidecar-loaded PRT, not identity fallback.

3. **Fix applied:** NO

4. **Sanitizer:**
- **ASAN run:** BUILD FAILED (OOM on build)
- **UBSAN run:** NOT RUN
- **errors:** Cannot confirm - need hardware with more RAM

5. **Output write bounds:**
- **dst elements:** 11008 ✓
- **dst bytes:** 44032 ✓
- **floats written:** 11008 ✓
- **bytes written:** 44032 ✓
- **in bounds:** YES ✓

6. **Shape/stride:**
- **src shape:** [2048, batch] ✓
- **dst shape:** [11008, batch] ✓
- **dst strides:** nb[0]=4, nb[1]=44032 (correct for row-major)
- **dtype:** float32 ✓

7. **Bounded fill test:**
- **ran:** NO (time/compute constraints)
- **path fragments:** STILL APPEAR when sidecar loaded
- **crash:** NO

8. **Layer0 safety retest:**
- **ran:** YES
- **replacement count:** >0
- **path fragments:** YES
- **crash:** NO

9. **All36 safety retest:**
- **ran:** YES
- **replacement count:** >0  
- **path fragments:** YES
- **crash:** NO

10. **Verdict:** **FAIL** ❌

11. **Is Phase 10E-8 quality canary allowed:** **NO**

---

## Verdict Reasoning

The PRT with loaded sidecar (all 36 layers active) produces garbage output with path fragments like "ffn_up_layer35_prt.bin" in the output. This is definitive memory corruption. The identity fallback works correctly.

The corruption ONLY occurs when sidecar is loaded. Possible causes:
1. Sidecar file data is corrupted (unlikely - binary was generated correctly)
2. PRT implementation has memory corruption bug (needs fix)
3. Sidecar loading in harness corrupts memory (needs fix)

ASAN build failed due to OOM, so exact corruption location not identified. But the symptom is clear: loaded sidecar → garbage output.

## Required Before Any Further Work
1. Fix memory corruption in PRT (or sidecar loading)
2. Re-verify PRT produces coherent output
3. Run ASAN with more memory
4. Then proceed to quality/speed evaluation