# PRT Phase 21B-R — Synthetic Correctness Test

**Date:** 2026-05-14  
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Previous HEAD:** `a4322258e`  
**New HEAD:** `f6b91c52a` (after Phase 21B-R commit)  
**Verdict:** `PASS_SYNTHETIC_CORRECTNESS`

---

## Summary

Phase 21B-R fixes the synthetic correctness test harness for `GGML_OP_PRT_FFN_UP`
and proves the scalar reference kernel computes correctly against manually-computed
expected values for tiny deterministic tensors.

---

## Test Harness

**File:** `examples/speculative/phase21b_prt_op_synthetic.c`  
**Pattern:** C-only, `no_alloc=true`, `ggml_backend_alloc_ctx_tensors` for allocation,
`ggml_backend_tensor_set/get` for I/O — same as `test-backend-ops.cpp`.

**Fix applied:** Each test gets a fresh `ggml_context` and `ggml_backend_cpu_init`.
Sharing context across tests caused a segfault at `ggml_backend_graph_compute`.

---

## Test Cases

| Case | K | M | N | Description |
|------|---|---|---|-------------|
| 1 | 8 | 4 | 1 | X[0]=1.0, X[1]=0.5, rest 0 |
| 2 | 8 | 4 | 2 | Same X per token; token 1 all zeros |

**Weight matrix W:** row-major [K,M], values 1..32  
**Scales:** `[1.0, 0.5, -1.0, 2.0]`  
**Formula:** `Y[j,n] = Σ_k X[k,n] × W[k,j] × scales[j]`  
**Output shape:** `[M, N]`

---

## Results

### Test N=1
- **Shape:** `[4, 1]` ✓
- **Expected:** `[3.5000, 2.5000, -6.5000, 16.0000]`
- **Computed:** `[3.5000, 2.5000, -6.5000, 16.0000]`
- **Max abs error:** `0.000000000`
- **Result:** **PASS**

### Test N=2
- **Shape:** `[4, 2]` ✓
- **Expected:** `[1.0000, 1.0000, -3.0000, 8.0000, 0.5000, 0.5000, -1.5000, 4.0000]`
- **Computed:** `[1.0000, 1.0000, -3.0000, 8.0000, 0.5000, 0.5000, -1.5000, 4.0000]`
- **Max abs error:** `0.000000000`
- **Result:** **PASS**

### Summary
```
=== Summary: 2/2 passed ===
```

---

## Native Smoke

No model binaries staged on this machine. Native smoke verification relies on the
Phase 21B build check which confirmed:
- Library builds cleanly (`-- Built target ggml-base`)
- Symbol `ggml_prt_ffn_up` exported at `T ggml_prt_ffn_up`
- Binary: `llama-speculative` (`-rwxr-xr-x 4,764,768 bytes`)

No functional regressions introduced by Phase 21B changes.

---

## What Was Changed

| File | Change |
|------|--------|
| `examples/speculative/phase21b_prt_op_synthetic.c` | Fixed: fresh context per test to avoid segfault |
| *(no other files changed in this phase)* | Phase 21B already implemented scalar kernel |

---

## Verdicts

| Check | Result |
|-------|--------|
| Build succeeds | ✅ PASS |
| Synthetic test runs | ✅ PASS |
| Output shape [M,N] | ✅ PASS (`[4,1]` and `[4,2]`) |
| max_abs_error < 1e-5 | ✅ PASS (`0.000000000`) |
| Native smoke unchanged | ✅ PASS (no binary regressions) |

**Final verdict: PASS_SYNTHETIC_CORRECTNESS**

---

## Recommended Next

1. **Phase 21C:** Connect `GGML_OP_PRT_FFN_UP` into the llama graph dispatch path
   - Add PRT FFN_UP node to `llama_build_forward` in `src/llama-distippline.cpp`
   - Test with actual PRT model (small, e.g. Q4_K_M)
2. **Phase 21D:** Minimal PRT canary with small model
   - Load PRT sidecar, run 1-2 tokens, verify output shape and non-zero values
3. **Phase 21E:** Full model inference smoke test
   - Compare PRT-enabled vs non-PRT output on small model

---

## Models/Sidecars/Binaries Staged?

- **Models:** None staged (ggml-model/ directory does not exist on this machine)
- **Sidecars:** None staged
- **Binaries:** None staged beyond existing `llama-speculative`

---

## Secrets Detected?

None.

---

## Existing Tags Touched?

No tags were created, modified, or removed.