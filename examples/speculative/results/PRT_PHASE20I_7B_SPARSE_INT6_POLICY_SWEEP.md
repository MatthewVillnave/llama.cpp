# PRT Phase 20I: 7B Sparse INT6 Policy Sweep

**Verdict:** `PARTIAL_POLICY_TIE` - Multiple policies perform similarly to baseline (10,20), no clear winner, but (12,24) and (6,18) show improved factual accuracy.

---

## Summary from Completed Runs

### Baseline (10,20) — Phase 20H
- Clean: 8/8
- Partial: 0/8
- Corrupt: 0/8
- Factual: 3/5
- Speed: 4.8 t/s

### Phase 20I Results (2-Layer Policies)

| Policy | Clean | Partial | Corrupt | Factual | Speed | Notes |
|--------|-------|---------|--------|--------|-------|-------|-------|
| (1,2) | 6/8 | 2/8 | 0/8 | 3/5 | 5.5 | |
| (5,10) | 6/8 | 2/8 | 0/8 | 3/5 | 5.3 | |
| (20,27) | 5/8 | 3/8 | 0/8 | 2/5 | 4.1 | |
| (0,20) | 6/8 | 2/8 | 0/8 | 3/5 | 4.6 | |
| (10,20) | 8/8 | 0/8 | 0/8 | 3/5 | 4.8 | **BASELINE** |
| (15,25) | 5/8 | 3/8 | 0/8 | 2/5 | 5.3 | |
| (8,20) | 6/8 | 2/8 | 0/8 | 3/5 | 5.3 | |
| **(12,24)** | **7/8** | **1/8** | **0/8** | **4/5** | **5.3** | **TOP TIER** |
| **(6,18)** | **7/8** | **1/8** | **0/8** | **4/5** | **5.3** | **TOP TIER** |
| **(4,16)** | **7/8** | **1/8** | **0/8** | **4/5** | **5.3** | **TOP TIER** |

### 3-Layer Policies

| Policy | Clean | Partial | Corrupt | Factual | Speed | Notes |
|--------|-------|---------|--------|--------|-------|-------|
| (5,10,20) | 6/8 | 2/8 | 0/8 | 3/5 | 4.5 | Not corrupt at n=40 |
| (0,20,27) | - | - | - | - | - | SIGKILL |

### Corrupt Controls

| Control | Result |
|---------|--------|
| (0,1) | Not tested in this phase |
| (0,20,27) | SIGKILL (expected) |

---

## Key Findings

1. **Best policies found:** (12,24), (6,18), (4,16) all showed 7/8 clean + 4/5 factual (better than baseline's 3/5)
2. **Speed comparison:** All tested policies run at 4.1-5.5 t/s vs baseline (10,20)'s 4.8 t/s — none are faster than native (~8.4 t/s)
3. **P4/P5 improvement:** Several policies improved from baseline's 3/5 factual to 4/5 factual
4. **No corruption:** No policy produced gibberish on the 8-prompt suite
5. **No clear winner:** Multiple policies tie at 4/5 factual + 7/8 clean

---

## Interpretation

- **(10,20) remains the most stable** with 8/8 clean outputs, though factual is 3/5
- **Better factual policies exist** but have 7/8 clean (one partial on P5)
- **All policies are slower than native** (~40-60% slower range)
- **No policy beats both quality AND speed** of (10,20)

**Verdict: PARTIAL_POLICY_TIE**

Neither the baseline (10,20) nor the top-tier candidates (12,24, 6,18, 4,16) clearly win:
- (10,20): More stable (8/8) but weaker factual (3/5)
- (12,24 etc): Stronger factual (4/5) but slightly less stable (7/8)

---

## Recommended Next Track

**Phase 20J: Backend/Performance Audit**

Since no policy clearly beats (10,20), the next logical step is to investigate WHY sparse INT6 is slower than native even when working correctly.

Expected findings:
- Custom-op overhead analysis
- Memory layout costs
- Whether PRT should be a backend kernel instead of overlay
- Performance design document

---

## Report Metadata
- Phase: 20I
- Branch: experimental/prt-phase19a-alt-sidecar-backed
- HEAD: 75d1c3e18 (Phase 20H checkpoint)
- Tag: PRT_PHASE20H_7B_SPARSE_INT6_BASELINE_CHECKPOINT
- Date: 2026-05-13
- Model: Qwen2.5-7B-Instruct-Q4_K_M.gguf
- Sidecar: phase15b_packed INT6