# PRT Phase 21A: GGML_OP_PRT_FFN_UP Skeleton

**Verdict:** `PASS_GGML_OP_SKELETON_BUILDS` + `PASS_GGML_OP_SKELETON_NATIVE_SMOKE`

---

## Phase 21A Summary

### What Was Added

1. **Op enum in ggml.h:**
   - Added `GGML_OP_PRT_FFN_UP` to the enum before `GGML_OP_COUNT`
   - Enum count now 97 (was 96)

2. **Op name mapping in ggml.c:**
   - Added "PRT_FFN_UP" to `GGML_OP_NAME[]` array
   - Added "prt_ffn_up(X,W,scales)" to `GGML_OP_SYMBOL[]` array
   - Updated static_assert to check for 97 ops

3. **Constructor stub in ggml.h + ggml.c:**
   - Added `ggml_prt_ffn_up()` declaration in ggml.h
   - Added stub implementation that asserts with message: "ggml_prt_ffn_up is a stub — not implemented yet"

4. **Behavior when invoked:**
   - If anyone tries to call `ggml_prt_ffn_up()`, it will `GGML_ASSERT(false)` with clear message
   - If the op is accidentally created via other means, CPU backend default will `GGML_ABORT("op not implemented: PRT_FFN_UP")`

### Files Changed
- `ggml/include/ggml.h` - enum + constructor declaration
- `ggml/src/ggml.c` - name mapping + stub implementation

### Build and Smoke Test

| Test | Result |
|------|--------|
| `cmake --build build -j4` | ✅ SUCCESS (llama-cli, llama-server built) |
| Native smoke (no PRT) | ✅ SUCCESS - "Paris" at 9.6 t/s |
| Behavior when unused | ✅ No change - normal native path |

### Validation

- **Build:** Compiles cleanly without errors
- **Native behavior:** Unchanged - goes through normal `build_lora_mm()` path
- **No PRT invocation:** When no PRT flags used, `ggml_prt_ffn_up()` stub is never called
- **Explicit failure:** If stub IS called, fails with clear message "not implemented yet"

### Recommended Next Phase

**Phase 21B:** Scalar reference kernel in ggml-cpu/ops.cpp

Implementation path:
- Replace stub that returns NULL with actual tensor creation
- Add `ggml_compute_forward_prt_ffn_up()` stub in ggml-cpu/ops.cpp  
- Implement scalar reference kernel (triple-nested loop for correctness)
- Verify tensor shapes: input [M,n_tokens], output [K,n_tokens]
- Synthetic test: known weights × known input → known output
- Next canary: 0.5B model single-layer PRT vs native comparison

---

## Phase 21A-A through 21A-G Complete

### A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

### B. Previous HEAD
`9f76445fb` (Phase 20K)

### C. New HEAD
Pending commit

### D. Files Changed
- ggml/include/ggml.h (+18 lines)
- ggml/src/ggml.c (+25 lines, -2 lines static_assert changes)

### E. Op enum added?
YES - `GGML_OP_PRT_FFN_UP` added to enum before GGML_OP_COUNT

### F. Op name/string added?
YES - "PRT_FFN_UP" in GGML_OP_NAME["PRT_FFN_UP"]

### G. Dispatch/switches updated?
NO - default handler catches unknown ops with GGML_ABORT (intentional - stub not yet used)

### H. Constructor stub added?
YES - `ggml_prt_ffn_up()` declared and implemented as stub (asserts if called)

### I. Build result
✅ SUCCESS - Both llama-cli and llama-server built

### J. Native smoke result  
✅ SUCCESS - "The capital of France is Paris" at 9.6 t/s

### K. Behavior change when unused?
NO - Native path unchanged

### L. Verdict
PASS_GGML_OP_SKELETON_BUILDS + PASS_GGML_OP_SKELETON_NATIVE_SMOKE

### M. Recommended next
Phase 21B - scalar reference kernel / synthetic test

### N. Models/sidecars/binaries staged?
NO

### O. Secrets detected?
NONE

### P. Existing tags touched?
NO

---

*Phase 21A Complete*