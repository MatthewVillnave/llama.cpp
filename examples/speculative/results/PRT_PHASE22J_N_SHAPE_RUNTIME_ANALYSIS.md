# PRT Phase 22J: N-Shape Runtime Analysis

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Previous HEAD
ddd50a2c6 (Phase 22I)

## New HEAD
a4f7c9d12 (Phase 22J with N-shape logging)

---

# Executive Summary

**Critical finding: AVX2 kernel NEVER executes during normal runtime.**

All GGML_OP_PRT_FFN_UP calls produce N=2 (batch size 2), regardless of prompt, generation length, or layer. The AVX2 kernel in prt_ffn_up_avx2.h only handles N=1, so it rejects every call with `REJECTED_N_NE_1`. All layers fall back to scalar for all invocations.

This means the current AVX2 kernel provides zero acceleration for real workloads on this hardware/harness combination.

---

# 1. N-Shape Data

## 0.5B (K=896, M=4864), AVX2 enabled

| Test | Prompt | n= | c= | N seen | AVX2 execution | Kernel time |
|------|--------|---|-----|--------|----------------|-------------|
| 1 | Hi | 1 | 64 | N=30 | REJECTED | 0us |
| 2 | Hi | 1 | 4 | N=4 | REJECTED | 0us |
| 3 | "The capital..." | 1 | 4 | N=4 | REJECTED | 0us |
| 4 | "The capital..." | 2 | 4 | N=4 | REJECTED | 0us |
| 5 | Hi | 1 | 2 | N=2 | REJECTED | 0us |
| 6 (no --prt-only-layer) | Hi | 1 | 2 | N=2 | REJECTED | 0us |

**Finding:** ALL calls produce N>1. N=2 is the minimum batch size observed. N=30 and N=4 are prompt-phase batch aggregations.

## 7B (K=3584, M=18944), AVX2 enabled

From Phase 22I: all 7B calls showed N=2 and were rejected.

---

# 2. AVX2 Kernel Constraint

```c
// prt_ffn_up_avx2.h
if (n_tokens != 1) {
    return; // caller falls back to scalar
}
```

The kernel is designed for N=1 only. When N!=1, it returns immediately without doing any work.

---

# 3. Caller Behavior

The caller (ops.cpp) tries AVX2 first when `PRT_V2_AVX2=1`:
1. Calls AVX2 kernel with N
2. Kernel returns immediately for N!=1
3. Caller falls back to scalar
4. Scalar path executes full computation

---

# 4. Impact Assessment

| Scenario | AVX2 used? | Reason |
|----------|-----------|--------|
| Prompt/prefill (N>1) | ❌ NO | Kernel rejects N!=1 |
| Token generation (N>1) | ❌ NO | Kernel rejects N!=1 |
| Single-token batch | ❌ NO | llama-cli batches 2+ tokens |
| Current runtime | ❌ NO | Zero AVX2 acceleration |

---

# 5. Root Cause

llama_decode() batches tokens in groups of 2+ for this model/harness. The GGML batch传递给GGML_OP_PRT_FFN_UP的n_tokens dimension is always >=2 during both prefill and decode phases.

---

# 6. Verdict

`FAIL_AVX2_NEVER_EXECUTES + N_SHAPE_CONFIRMED_N_GE_2`

The AVX2 kernel is unreachable with current runtime batching. Scalar path is the only active path.

---

# 7. Allowed Claims

✅ GGML_OP_PRT_FFN_UP sees N>=2 for all calls during prompt and decode
✅ AVX2 kernel rejects all N!=1 calls
✅ All layers fall back to scalar path
✅ No AVX2 acceleration occurs in current harness

---

# 8. Forbidden Claims

❌ No AVX2 speedup in any real workload
❌ No kernel-level AVX2 timing captured (kernel never executes)
❌ No production-ready AVX2 acceleration

---

# 9. Recommended Next

**Phase 22K:** Implement N=2 AVX2 kernel for GGML_OP_PRT_FFN_UP.

Rationale: N=2 is the minimum batch size, so accelerating N=2 gives maximum real-world benefit. The AVX2 kernel needs to handle N>1 to be useful. A simple N=2 vectorized kernel would eliminate the scalar fallback.

---

# 10. Safety Scan

| Check | Status |
|-------|--------|
| No model files staged | ✅ |
| No sidecars staged | ✅ |
| No f32 refs staged | ✅ |
| No captures staged | ✅ |
| No huge logs staged | ✅ |
| No secrets | ✅ |
| System disk free | ~58G |
| Scratch disk free | ~51G |

---

**Phase 22J Verdict:** `FAIL_AVX2_NEVER_EXECUTES + N_SHAPE_CONFIRMED_N_GE_2`
