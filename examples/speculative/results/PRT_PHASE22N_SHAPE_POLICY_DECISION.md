# Phase 22N: Shape Policy Decision — N=30 AVX2 vs Native Prefill Fallback

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## Previous HEAD
`d50c32a46` (Phase 22M: AVX2 N4 kernel)

## New HEAD
TBD after commit

---

## D. Shape Distribution Matrix

Compiled from all captured AVX2 log data (c=2 to c=64):

| Test | N=2 | N=4 | N=14 | N=16 | N=30 | N=34 | REJECT | Total | AVX2% |
|------|-----|-----|------|------|------|------|--------|-------|-------|
| Hi c=2 n=1 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 17 | 100% |
| Hi c=4 n=1 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | 11 | 100% |
| Hi c=16 n=1 | 2 | 0 | 1 | 1 | 0 | 0 | 2 | 4 | 50% |
| cap c=4 n=1 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | 11 | 100% |
| cap c=64 n=1 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 3 | 67% |
| cap c=64 n=1 (22L) | 2 | 0 | 0 | 0 | 1 | 0 | 1 | 3 | 67% |

**AVX2 coverage by context size:**
- c=2: **100%** (all decode, all N=2)
- c=4: **100%** (decode N=2 + small batch N=4)
- c=16: **50%** (N=2 decode + N=14/N=16 prefill fallback)
- c=64: **67%** (N=2 decode + N=34 prefill fallback)

---

## E. N=2 / N=4 Count Summary

| N value | Observed in | Count | Classification |
|---------|-------------|-------|----------------|
| N=2 | All tests | ~26 total | Decode (single-token generation) |
| N=4 | c=4 tests only | 16 total | Small batch (prefill of short prompts) |
| N=14 | Hi c=16 | 1 | Prefill large context |
| N=16 | Hi c=16 | 1 | Prefill large context |
| N=30 | cap c=64 | 1 | Prefill large context |
| N=34 | cap c=64 | 1 | Prefill large context |

---

## J. Prefill vs Decode Interpretation

**Clear pattern identified:**

| N range | Classification | Frequency | AVX2 support |
|---------|---------------|-----------|--------------|
| N=2 | Decode (single-token generation) | Common, every generation step | ✓ N=2 |
| N=3-4 | Small batch / short prompt | c=4 tests | ✓ N=4 |
| N=5-16 | Medium prefill | c=16 tests | ✗ Falls to scalar |
| N=17-64 | Large prefill / context | c=64 tests | ✗ Falls to scalar |

**Key insight:** N=2 is always decode (token-by-token generation after KV fill). N>4 is always prefill/large context — these are GGML native matmul territory, not where PRT should replace FFN_UP.

**PRT AVX2 sweet spot:** N<=4 covers all decode scenarios and small batch. Beyond N=4, the native Q4 matmul path in GGML is already optimized for large-matrix operations.

---

## K. Policy Options Evaluated

### Policy A — PRT all calls (current state)
- N=2/N=4 use AVX2 ✓
- N>4 falls back to scalar PRT (slow)
- **Problem:** Scalar PRT for N=30 is ~35ms per call (from 22L timing)
- **Risk:** Large prefill shapes trigger slow scalar fallback

### Policy B — Native prefill, PRT decode (RECOMMENDED)
- Use PRT AVX2 only when N<=4
- Force native FFN_UP (ggml_op matmul) for N>4
- **Benefit:** GGML native Q4 matmul is already optimized for large N
- **Risk:** Low — doesn't require new kernel implementation
- **Implementation:** Route by N in ops.cpp dispatch

### Policy C — Implement N=30/generic-N AVX2
- **Complexity:** High — 30-pass temp-store or scatter/gather pattern
- **Bandwidth:** Would read W 30× per 8-wide block
- **Comparison:** Native GGML Q4 matmul is already highly optimized for N=30
- **Verdict:** Not worth implementation complexity when native path exists

---

## L. Recommended Next Policy

**Policy B — Native prefill, PRT decode routing**

Implementation in ops.cpp dispatch:
```c
if (prt_avx2_mode == 1 && !scales) {
    if (n_tokens <= 4) {
        // Use PRT AVX2 kernel
        ggml_compute_forward_prt_ffn_up_avx2(...);
    } else {
        // Force native path for N>4 (large prefill)
        // Fall through to native ggml_compute_forward_mul_mat or return early
        // with a flag that causes the custom op to skip PRT insertion
    }
}
```

**Rationale:**
1. N<=4: PRT AVX2 gives ~5× (N=2) and ~1.15× (N=4) speedup over scalar
2. N>4: Native GGML matmul is already optimized for large-matrix operations
3. The scalar PRT fallback for N=30 (~35ms) is slower than GGML native would be
4. This avoids implementing a complex N=30 AVX2 kernel

---

## N. Verdict

**PASS_SHAPE_POLICY_DECISION + PARTIAL_NEED_NATIVE_PREFILL_POLICY**

- Shape distribution clearly shows N=2/N=4 = decode, N>4 = prefill
- N=30/N=34 observed in c=64 tests are prefill shapes, not decode
- Policy B (native prefill, PRT decode) is the correct architecture choice
- No N=30 AVX2 needed — native path is better for large N

---

## O. Recommended Next

**Phase 22O — Implement N<=4 PRT decode routing with native fallback for N>4**

Changes needed:
1. ops.cpp: add N<=4 check before calling AVX2 kernel
2. For N>4: implement native fallback routing (either skip PRT or use GGML native matmul path)
3. Verify with c=16 and c=64 tests that native fallback triggers for N>4
4. Confirm output validity with native fallback

**Do NOT implement N=30 AVX2. Do NOT move to INT6 until this policy is validated.**

---

## Safety Checklist

| Item | Status |
|------|--------|
| Models staged | No |
| Sidecars staged | No |
| Credentials | No |
| Tags touched | No |
| System disk free | 45G |
| Scratch disk free | 51G |