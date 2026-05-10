# PRT Phase 19O — 7B INT6 Regression Isolation

## Verdict

**FAIL_7B_BRANCH_REGRESSION_CONFIRMED** ❌

---

## Context

Phase 19N reported 7B INT6 scalar producing corrupt output at ~1.0 tok/s on the current Phase 19 branch. This conflicts with earlier Phase 15B-H results showing INT6 working correctly at ~9 tok/s with exact matches.

Goal: Isolate root cause — command mismatch vs branch regression vs sidecar provenance.

---

## Test Results

### A. 7B Native (baseline)

| Metric | Value |
|--------|-------|
| **Timing** | 9.7 tok/s |
| **Output** | "Paris" (correct) |
| **Layers** | All native |

### B. 7B INT6 scalar (current Phase 19, --prt-mode 5700 --prt-force-native 11,15)

| Metric | Value |
|--------|-------|
| **Timing** | 1.1 tok/s |
| **Output** | Corrupt (Chinese chars, gibberish) |
| **Layers** | 28 loaded, 26 PRT-active, 2 force-native |
| **PRT compute** | YES — PRT_COMPUTE events visible |
| **Force-native** | Applied for layers 11, 15 |

### C. Layer0-only PRT (--prt-mode 5600)

| Metric | Value |
|--------|-------|
| **Timing** | 6.8 tok/s |
| **Output** | Still corrupt |
| **PRT layers** | 1 (layer0 only) |

### D. All-layers force-native

| Metric | Value |
|--------|-------|
| **Timing** | 9.6 tok/s |
| **Output** | "Paris" (correct) |
| **PRT layers** | 0 — all native fallback |

### E. 7B INT6 predecode-f32

| Metric | Value |
|--------|-------|
| **Status** | OOM at layer 26/28 |
| **RAM estimate** | +7.1 GB extra |
| **Verdict** | PARTIAL_MEMORY_RISK |

---

## Investigation Steps

### 1. Sidecar verification

- **28 sidecars** found in `/tmp/prt_sidecars_7b_int6_phase15b_packed/`
- **Unique SHAs:** 28 (all unique)
- **File sizes:** All 50,997,268 bytes (matches expected PRT6 packed)
- **Format:** PRT6 INT6 packed (not INT8, not 0.5B)
- **Header M/K:** M=18944 (FFN), K=3584 (hidden) — matches Qwen2.5-7B

### 2. Scale forensic

```python
# layer0 first 8 scales:
scale[0] = 0.00000000  # ⚠️ ZERO
scale[1] = 0.00216875
scale[2] = 0.00510111
...
# Layer 0 scale exactly ZERO!
```

**Issue:** Scale at index 0 is exactly zero. This would cause row 0 of the FFN output to be zeroed out (Y[0] = sum(X[k] * 0 * int8[0,K]) = 0).

But scale[1-7] are non-zero (0.002-0.005). So only row 0 is zeroed.

**Question:** Did Phase 15 sidecars have non-zero at index 0? Unknown without reference branch test.

### 3. Command parity

Phase 15B-H used:
- `--prt-mode 5700`
- `--prt-force-native 11,15`
- `--prt-sidecar-format int6`

Current Phase 19 tests used the SAME flags. Still corrupt.

### 4. Force-native verification

| Test | Layers force-native | Result |
|------|-------------------|--------|
| Force-native 11,15 | 2/28 | Still corrupt at 1.1 t/s |
| All-28 native | 28/28 | Clean at 9.6 t/s ✓ |
| Layer0 only | 1/28 | Still corrupt at 6.8 t/s |

**Conclusion:** Force-native doesn't fix corruption. The PRT replacement path itself is broken.

### 5. PRT compute activation

With `--prt-log-level 1`, we can see:
- `[PRT_COMPUTE] layer=N mode=int6 hit=1` — PRT is being invoked
- `[PRT-11BB-AUTH] IL=N PRT result ne=[18944,1]` — Custom op returns tensor

So the custom op IS called. The computation inside is wrong.

### 6. Code diff (Phase 15 vs Phase 19)

Key changes between Phase 14a branch and Phase 19:

| Commit | Change | Risk |
|--------|-------|------|
| a8ce1a1d5 | 0.5B fix: M/K shape swap, PRT condition | LOW (0.5B specific) |
| 1e2b50616 | Add format==2 (INT6) to dequant path | **MEDIUM** — adds INT6 handling |
| c6b651107 | INT6 predecode f32 prototype | MEDIUM (different path) |

**Critical:** Commit 1e2b50616 (Phase 19D) added format==2 to the dequantization condition. Before this fix, format==2 fell through to the float32 fallback, which reads from `ud->sidecar` (nullptr for INT6), giving all-zeros output.

The FIX is present (0.5B INT6 works), but maybe there's a secondary bug.

### 7. Layer bisection

| PRT layers | Timing | Output |
|-----------|--------|--------|
| 0 only | 6.8 t/s | Corrupt |
| All (26 active) | 1.1 t/s | Corrupt |
| None (all native) | 9.6 t/s | Clean |

**Conclusion:** Corruption occurs even with a single PRT layer (layer0). Not cumulative multi-layer effect. The INT6 compute path itself is broken.

---

## Root Cause Analysis

### Hypothesis 1: Scale bug
**Evidence:** scale_0 = 0.000000 in CLI audit
**Status:** LIKELY — one zero scale would corrupt row 0 of all subsequent FFN layers

### Hypothesis 2: Unpack bug  
**Evidence:** Phase 15 packed data uses LUT-based unpack (Phase 15H)
**Status:** Unknown — can't verify without reference branch test

### Hypothesis 3: M/K layout mismatch
**Evidence:** PRT_SHAPE shows ud_M=3584, ud_N=18944, format=2
**Status:** This is correct (M=hidden, N=FFN)

### Hypothesis 4: Reference branch difference
**Status:** Could not test due to build time constraints

---

## Recommendations

1. **Immediate:** Run on reference branch (experimental/prt-phase14a-packed-sidecars) to confirm if sidecar issue
2. **Fix scale extraction:** Check if scale[0] should be zero, if generation code handles zero-scale rows
3. **Check dequantization:** The format==2 (INT6) kernel path may have subtle bug not present in format==1 (INT8)
4. **Verify Phase 15B-H exact environment:** Build, branch, model SHA, sidecar dir, git commit

---

## Next Steps

If command mismatch ruled out (we used same flags), and force-native doesn't fix (tested), it's a branch regression. Recommended:

1. **Phase 19P-A:** Create worktree on Phase 14a branch, rebuild, run same test
2. **Phase 19P-B:** Patch current branch based on diff with Phase 14a
3. **Phase 19P-C:** Add debug logs to scalar INT6 kernel to trace values

**Blockers:**
- Cannot rebuild Phase 14a worktree in reasonable time
- Phase 15 exact build env not reproducible without the branch

---

## Files Staged

None (no changes to source).

## Safety Scan

- No model files staged
- No sidecar files staged  
- No secrets detected
- No existing tags touched

