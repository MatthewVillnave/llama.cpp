# CLAIMS_FORBIDDEN_PHASE10E.md

## Forbidden Claims After Phase 10E

The following claims are **FORBIDDEN** based on Phase 10E results:

### PRT Active Generation Claims

1. ❌ **PRT active generation is working**
   - Status: BLOCKED by ggml_map_custom2 memory corruption
   - Output is garbage (path string fragments)
   - Do not claim PRT generation is functional

2. ❌ **PRT is faster end-to-end**
   - Status: CANNOT MEASURE — output is corrupted
   - No speedup measurement is valid
   - Do not claim any speedup

3. ❌ **PRT quality passed**
   - Status: CANNOT MEASURE — output is corrupted
   - Phase 10E-8 quality canary is BLOCKED
   - Do not claim any quality improvements

4. ❌ **Phase 10F benchmark is allowed**
   - Status: BLOCKED by Phase 10E-7S failure
   - Broader benchmark cannot proceed
   - Do not claim benchmark results

5. ❌ **Production integration is ready**
   - Status: BLOCKED by memory corruption
   - ggml_map_custom2 integration is not safe
   - Do not claim production readiness

### Misrepresentation Warnings

6. ❌ **Do not claim "PRT works" without the "standalone" qualifier**
   - PRT_3P standalone kernel works (validated offline)
   - PRT active generation does NOT work (blocked by corruption)

7. ❌ **Do not claim "sidecar loading works" as if end-to-end works**
   - Sidecar loading is clean (verified)
   - But loading + ggml integration = corruption

8. ❌ **Do not claim "ggml_map_custom2 substitution works" as if generation works**
   - Substitution infrastructure is proven functional
   - But output is corrupted, making the substitution useless in practice

## Summary

**The gap:** PRT infrastructure is proven correct at the component level (sidecar, loader, standalone kernel, graph substitution), but the ggml_map_custom2 integration introduces memory corruption that makes end-to-end generation unusable.

**What this means:** PRT research is valid and promising at the kernel level, but the llama.cpp integration path via ggml_map_custom2 is blocked by a fundamental memory corruption bug.