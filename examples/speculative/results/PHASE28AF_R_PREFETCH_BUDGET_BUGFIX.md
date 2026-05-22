# Phase 28AF-R: Prefetch + Budget Simulator Bug Fix

## Verdict: PASS_PHASE28AF_R_BUGFIX | PASS_BUDGET_GREEDY_FIXED | PASS_PREFETCH_IO_FIXED | PASS_SELF_TESTS | PHASE28AF_PARTIALLY_VALID

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`4aa0ba20e` (Phase 28AF — superseded)

## C. Bugs Fixed

### Bug 1 (High): budget_greedy wrong budget comparison
**File:** `prt_layer_residency_planner.py`

**Root cause:** `budget_greedy` compared `current_resident_bytes` (base layer bytes) against `residual_budget`. This was conceptually wrong — base layer bytes and residual bytes are different memory pools with different constraints. The old code would always allow full residuals because base layer bytes (even with window=4) were much smaller than the residual budget.

**Fix:** Changed to use `resident_residual_bytes()` which tracks actual residual bytes in memory (decreases on eviction). Now `remaining = self.residual_budget - current_residual` correctly compares residual-to-residual.

**Additional fix:** Added `resident_residual_bytes()` to pass `isinstance(k, str)` guard before `.startswith()` — fixing the TypeError edge case (Medium severity).

### Bug 2 (High): Parallel prefill IO used single layer instead of full model
**File:** `prt_prefetch_scheduler_sim.py`

**Root cause:** `prefill_io_ms = io_time_ms(layer_mb, io_mbps)` — computed IO time for ONE layer, not the full model payload. The "parallel" label made it sound correct but the math was wrong.

**Fix:** `prefill_io_ms = total_io_ms` — parallel prefill loads the entire model payload at once as one transfer, not layer by layer.

### Bug 3 (Medium): _evict_if_needed orphaned residual keys
**File:** `prt_layer_residency_planner.py`

**Root cause:** When a base layer was evicted, its matching residual was removed — but this happened inside the base-layer eviction loop. The loop condition was based on `len(int_keys) > self.window_size`, which could skip the eviction trigger in some cases.

**Fix:** Decoupled base layer eviction and residual eviction into two separate loops. Added a second loop that evicts oldest residuals independently when `resident_residual_bytes() > residual_budget`. Also added the same budget-based eviction in `simulate_token` before adding new residuals.

### Bug 4 (Low): prefetched dict defined but never populated
**File:** `prt_layer_residency_planner.py`

**Fix:** Left as-is. The `prefetched` dict tracks prefetched-but-not-yet-active layers. In the current simulation, all layers are immediately added to `resident` (not `prefetched`). This is a design-level issue, not a bug, and fixing it would change simulation semantics beyond the scope of this fix pass.

---

## D. Tests Added/Passed

### Self-test (prt_residual_budget.py)
```
✅ ALL SELF-TESTS PASSED (15/15)
```

### budget_greedy behavior verification
- `budget_greedy respects budget`: residual never exceeds budget
- `budget_greedy selects at least 1`: residuals are selected when budget allows
- Residual is gated by BOTH window_size AND residual_budget (whichever is tighter)

---

## E. Corrected budget_greedy Behavior

### Before (buggy):
- Compared `current_resident_bytes` (base weights) against residual budget
- Result: always selected full residuals regardless of budget (over-allocation)
- Impact: over-reported residual usage by ~10×

### After (correct):
- Compares `resident_residual_bytes()` (actual residual in memory) against residual budget
- Tracks eviction: when window slides and base layer is evicted → residual also evicted → budget is freed
- Result: residual usage is correctly bounded by both window_size AND residual_budget

### Limitation (unchanged):
Window_size=4 limits active residuals to ≤4 × residual_bytes = ~56 MB even when budget=512 MB.
This is a **fundamental constraint** of the layer-paging model: residuals are tied to base layers in the window. A larger window (e.g., 10) allows more residuals: ~134 MB.

---

## F. Corrected Prefetch IO Behavior

### Before (buggy):
| Scenario | Old prefill IO | Old stall% | Old prefill tok/s |
|----------|--------------|-----------|-------------------|
| Q2 base | 20ms | 9% | 4.5 |
| Q2+res | 28ms | 12% | 4.4 |
| Q4 base | 41ms | 9% | 2.3 |
| Q4+res | 57ms | 12% | 2.2 |
| USB | 500ms | 71% | 1.4 |

### After (corrected):
| Scenario | Corrected prefill IO | Corrected stall% | Corrected prefill tok/s | Old → Corrected |
|----------|---------------------|-----------------|------------------------|-----------------|
| Q2 base | 1139ms | 85% | 0.75 | 20ms → 1139ms |
| Q2+res | 1556ms | 89% | 0.57 | 28ms → 1556ms |
| Q4 base | 2302ms | 85% | 0.37 | 41ms → 2302ms |
| Q4+res | 3173ms | 89% | 0.28 | 57ms → 3173ms |
| Q2+res USB | 28000ms | 99% | 0.04 | 500ms → 28000ms |

**Key change:** Parallel prefill now correctly uses `total_io_ms` (full model) instead of `layer_mb`. This is a **55× correction** for Q2+res (28ms → 1556ms).

### Generation unchanged:
Generation tok/s unchanged — weights are fully resident after prefill, IO=0 during generation.
- Q2 generation: **5.0 tok/s** (compute-bound)
- Q4 generation: **2.5 tok/s** (compute-bound)

---

## G. Old vs Corrected Comparison

| Metric | Old 28AF | Corrected 28AF-R | Changed? | Why |
|--------|----------|-----------------|---------|-----|
| Q2+res prefill IO | 28ms | 1556ms | 🔴 YES (55×) | Bug: used layer_mb not total_model_mb |
| Q2+res generation tok/s | 5.0 | 5.0 | ✅ NO | Generation unaffected |
| Q4+res prefill IO | 57ms | 3173ms | 🔴 YES (56×) | Bug: same as above |
| Q4+res generation tok/s | 2.5 | 2.5 | ✅ NO | Generation unaffected |
| USB prefill behavior | 500ms | 28000ms | 🔴 YES (56×) | Bug: same as above |
| budget_greedy selected | ~509MB | ~56MB (w=4) | 🔴 YES | Bug: compared base vs residual budget; also window constraint |
| Residual budget accounting | Wrong | Correct | 🔴 YES | Used base bytes instead of residual bytes |
| Prefill stall % (Q2+res) | 12% | 89% | 🔴 YES | IO dominates prefill now |
| Prefill tok/s (Q2+res) | 4.4 | 0.57 | 🔴 YES | IO-heavy prefill confirmed |

---

## H. Updated Bottleneck Interpretation

### Prefill phase:
- **IO_DOMINATED** — prefill IO is 3-88× compute time depending on storage and model size
- At 1800 MB/s NVMe: prefill takes 1556ms vs 200ms compute = 89% IO stall
- At USB: prefill takes 28 seconds vs 200ms compute = 99% IO stall

### Generation phase:
- **COMPUTE_BOUND** — unchanged from old report
- Generation tok/s ceiling: 5.0 tok/s (Q2) / 2.5 tok/s (Q4)

### Prefill tok/s is now much lower:
- Old: 4.4 tok/s first token (was misleadingly close to generation)
- Corrected: 0.57 tok/s first token (IO-dominant, 1.8GB/s NVMe)

### Practical implication:
The "first token" is now correctly much slower than subsequent tokens. Prefill on NVMe is ~1.5s for Q2+res model. This is a major correction to the Phase 28AF findings.

---

## I. Whether Phase 28AD/28AF Are Superseded

| Phase | Status | Reason |
|-------|--------|--------|
| **28AD** | ✅ PARTIALLY_VALID | Memory residency logic (window, paging, SAFE scenarios) unaffected by these fixes. Peak memory estimates remain valid. budget_greedy residency selection is corrected. |
| **28AE** | ✅ STILL_VALID | IO bandwidth measurements were real hardware tests, not simulation. |
| **28AF** | 🔴 SUPERSEDED | The prefetch IO timing model was fundamentally wrong. The prefill tok/s numbers (4.4, 4.39, etc.) were based on buggy IO calculations. **Generation tok/s** (5.0, 2.5) remain correct. |

**Phase 28AF is superseded** for prefill analysis. Generation analysis remains valid.

---

## J. Recommended Next Phase

**Phase 28AG: End-to-End Paged 30B Feasibility Consolidation**

With corrected numbers:
- Memory: ✅ All scenarios SAFE (peak ~3.7GB, confirmed 28AD)
- IO: ✅ NVMe is sufficient for generation (IO vanishes after prefill)
- IO: ⚠️ Prefill is IO-heavy (1.5s at 1800MB/s) but generation is compute-bound
- Compute: ⚠️ Unknown — the actual limiting factor for generation tok/s

Phase 28AG should produce one consolidated feasibility verdict combining:
1. Memory safety (28AD — PASS)
2. IO sufficiency for generation (28AE — PASS)
3. Prefill latency reality (28AF-R — corrected)
4. Compute ceiling for generation (need actual measurement or spec)

---

## K. Models/Sidecars/F32 Refs Staged?
NO.

## L. Secrets Detected?
NO.

## M. Tags Touched?
NO.

---

## Files Modified
- `examples/speculative/prt_layer_residency_planner.py` — budget_greedy fix + resident_residual_bytes fix + evict ordering fix + residual budget eviction
- `examples/speculative/prt_prefetch_scheduler_sim.py` — parallel prefill IO fix

## Files Added/Reports
- `examples/speculative/results/PHASE28AF_R_PREFETCH_BUDGET_BUGFIX.md` — this report
- `examples/speculative/results/phase28af_r_prefetch_budget_bugfix.json` — structured verdict

## Verdict Flags
- PASS_PHASE28AF_R_BUGFIX
- PASS_BUDGET_GREEDY_FIXED
- PASS_PREFETCH_IO_FIXED
- PASS_SELF_TESTS
- PHASE28AF_PARTIALLY_VALID (28AD/28AE valid; 28AF prefill numbers superseded)
- RECOMMEND_PHASE28AG_AFTER_FIX