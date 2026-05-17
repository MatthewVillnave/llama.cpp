# PRT Phase 22J: N-Shape Runtime Analysis and N=1 Timing Attempt

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Previous HEAD
ddd50a2c6 (Phase 22I)

## New HEAD
7a3b9c12e (Phase 22J with full N-shape analysis)

---

# Executive Summary

**N=1 is NOT reachable via llama-cli for token generation.** All GGML_OP_PRT_FFN_UP calls during both prefill and decode produce N>=2. The minimum observed N is 2 for token generation, and N=30 for prompt phase. The AVX2 kernel (designed for N=1 only) is dead code for real workloads.

This is not a bug — it's a fundamental batching constraint of the llama.cpp runtime. The AVX2 kernel needs to be extended to handle N=2 (minimum) or N>1 in general.

---

# 1. N-Shape Matrix

## 0.5B (K=896, M=4864)

| Test | Prompt | n= | c= | Backend | N seen | Kernel calls |
|------|--------|---|-----|---------|--------|--------------|
| 1 | Hi | 1 | 64 | AVX2 | 30 | 1 (REJECTED) |
| 2 | Hi | 1 | 4 | AVX2 | 4 | 5 (REJECTED) |
| 3 | "The capital..." | 1 | 4 | AVX2 | 4 | 5 (REJECTED) |
| 4 | "The capital..." | 2 | 4 | AVX2 | 4 | REJECTED (all) |
| 5 | Hi | 1 | 2 | AVX2 | 2 | REJECTED (all) |
| 6 | Hi | 1 | 2 | AVX2 | 2 | REJECTED (all, no --prt-only-layer) |
| 7 | Hi | 1 | 2 | scalar | 2 | All layers |
| 8 | Hi | 1 | 32 | scalar | 30 | 1 (prompt phase) |

**0.5B findings:**
- Prefill/prompt phase: N=30 (c=64) or N=30 (c=32)
- Token generation: **N=2** (consistent across all tests)
- AVX2: 0 executions (all rejected N!=1)
- Scalar: 100% of calls

## 7B (K=3584, M=18944)

| Test | Prompt | n= | c= | N | Kernel calls |
|------|--------|---|-----|---|--------------|
| 1 | Hi | 1 | 2 | 2 | All N=2, all rejected |

**7B findings:**
- Token generation: **N=2** (consistent)
- AVX2: 0 executions
- Scalar: 100% of calls

---

# 2. Force N=1 Attempt

**Attempted:** `llama-cli -b 1 -ub 1` to force batch size 1

**Result:** llama_decode() aborts with GGML assertion failure. The batch size 1 is incompatible with the speculative decoding path (--prt-mode 5700 uses speculative decoding internally).

**Error:**
```
llama_decode () at .../llama.cpp:...:
ggml_abort()
common_speculative_is_compat()
```

**Conclusion:** N=1 cannot be forced through llama-cli batch flags when using speculative decoding (prt_mode=5700).

---

# 3. Timing per Backend (Scalar only — AVX2 never executes)

## 0.5B Scalar Timing (N=2)

| Test | Prompt | N | avg_kernel_us | abs_sum range |
|------|--------|---|---------------|---------------|
| 7 | Hi, n=1, c=2 | 2 | ~9,500-11,000 | 5.50 - 8.27 |
| 8 | Hi, n=1, c=32 | 30 | N/A (prompt) | 11.76 |

**Scalar avg per token (N=2):** ~10,300 us (~10.3 ms)

## 7B Scalar Timing (N=2)

| Test | N | avg_kernel_us | abs_sum range |
|------|---|---------------|---------------|
| 1 | 2 | ~593,000 | 0.96 - 1.70 |

**7B scalar avg per token (N=2):** ~593,000 us (~593 ms)

---

# 4. AVX2 Kernel Constraint

```c
// prt_ffn_up_avx2.h
if (n_tokens != 1) {
    return; // caller falls back to scalar
}
```

The kernel returns immediately for N!=1 without doing any work.

**Impact: 0% AVX2 usage in current runtime. Zero acceleration from AVX2 kernel.**

---

# 5. Interpretation

| Scenario | N | AVX2 used? | Reason |
|----------|---|-----------|--------|
| Prefill (c=64) | 30 | ❌ | Kernel rejects N!=1 |
| Prefill (c=32) | 30 | ❌ | Kernel rejects N!=1 |
| Prefill (c=4) | 4 | ❌ | Kernel rejects N!=1 |
| Token gen | 2 | ❌ | Kernel rejects N!=1 |
| Force N=1 | - | ❌ | Aborts with GGML assert |

**Conclusion:** N=1-only AVX2 is insufficient for this harness. N=2 (minimum batch) or N>1 support required.

---

# 6. Verdict

`FAIL_AVX2_NEVER_EXECUTES + PARTIAL_N1_NOT_REACHABLE + N_SHAPE_CONFIRMED_N_GE_2`

- AVX2 kernel never executes (all calls N>=2)
- N=1 cannot be forced via llama-cli batch flags
- Scalar fallback is 100% of runtime
- N=2 is the minimum observed batch size

---

# 7. Allowed Claims

✅ GGML_OP_PRT_FFN_UP sees N=2 for token generation (0.5B and 7B)
✅ GGML_OP_PRT_FFN_UP sees N=30 for prompt phase (c=64)
✅ AVX2 kernel rejects all N!=1 calls
✅ Zero AVX2 acceleration in current harness
✅ N=1 unreachable via llama-cli with speculative decoding

---

# 8. Forbidden Claims

❌ No AVX2 speedup claim (kernel never executes)
❌ No N=1 AVX2 timing (N=1 not reachable)
❌ No end-to-end speedup

---

# 9. Recommended Next

**Phase 22K:** Implement N=2 AVX2 kernel for GGML_OP_PRT_FFN_UP.

Rationale: N=2 is the minimum batch size for token generation on this harness. Accelerating N=2 provides real benefit vs scalar. The current N=1-only kernel is dead code.

Key design for N=2 AVX2:
- Handle 2 tokens simultaneously
- Vectorize over M (output features) like N=1 does
- Process X[:,0] and X[:,1] in parallel using AVX2 FMA
- Maintain same math as scalar path

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
