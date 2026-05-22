# Phase 28AD: Simulated Layer Residency Planner

## Verdict: PASS_PHASE28AD_RESIDENCY_PLANNER ✅ | PASS_30B_MEMORY_FEASIBLE_IN_SIM ✅

## Summary
Implemented `prt_layer_residency_planner.py` — a metadata-only layer traversal simulator. Ran 9 scenarios across Q2/Q4 base and residual policies. All scenarios pass with >12 GB headroom. Memory is NOT the bottleneck. IO bandwidth is the next blocker.

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`207223a26`

## C. Planner Path
`examples/speculative/prt_layer_residency_planner.py`

### Architecture

**Inputs:** layers, layer_bytes_mb, residual_bytes_mb, window_size, prefetch_distance, kv_mb, runtime_buffer_mb, os_headroom_mb, ram_gb, policy, max_tokens

**Simulation model:**
- Each token: sequential layer traversal (0 → num_layers)
- Resident set: base layers + residual overlays
- Eviction: LRU — evict layers behind current layer when window exceeded
- Prefetch: load N layers ahead during compute
- Peak tracking: records maximum resident at any point

**Output:** peak resident, headroom, SAFE/UNSAFE, IO/token estimate, eviction/prefetch counts

---

## D. Scenario Results

| Scenario | Layers | Layer MB | Res MB | Window | Context | Peak MB | Headroom MB | SAFE | IO/token |
|---------|--------|---------|--------|--------|--------|---------|-------------|------|----------|
| 30B Q2 base, c=1024, win=4 | 56 | 36.6 | 0.0 | 4 | 1024 | 3294 | 13090 | ✅ | 36.6 |
| 30B Q2 base, c=2048, win=4 | 56 | 36.6 | 0.0 | 4 | 2048 | 3296 | 13088 | ✅ | 36.6 |
| 30B Q2+res, c=1024, win=4 | 56 | 36.6 | 13.4 | 4 | 1024 | 3374 | 13010 | ✅ | 50.0 |
| 30B Q2+res, c=2048, win=4 | 56 | 36.6 | 13.4 | 4 | 2048 | 3376 | 13008 | ✅ | 50.0 |
| 30B Q4 base, c=2048, win=4 | 56 | 74.0 | 0.0 | 4 | 2048 | 3520 | 12864 | ✅ | 74.0 |
| 30B Q4 base, c=4096, win=4 | 56 | 74.0 | 0.0 | 4 | 4096 | 3524 | 12860 | ✅ | 74.0 |
| 30B Q2+res, c=2048, win=8, pref=2 | 56 | 36.6 | 13.4 | 8 | 2048 | 3613 | 12771 | ✅ | 50.0 |
| 30B Q4+res, c=2048, win=4 | 56 | 74.0 | 28.0 | 4 | 2048 | 3688 | 12696 | ✅ | 102.0 |
| 30B Q2+res, c=2048, win=4, budget=512MB | 56 | 36.6 | 13.4 | 4 | 2048 | 3376 | 13008 | ✅ | 50.0 |

---

## E. Memory Feasibility Result

**All 9 scenarios are SAFE.** Peak resident never exceeds 3.7 GB against a 16 GB budget.

**Why the headroom is so large:**
- The active layer window (4 layers × ~37 MB) = ~150–220 MB peak
- Adding KV (~4 MB), runtime buffer (1 GB), OS headroom (2 GB) = ~3.3 GB total
- The 13 GB remaining headroom is a massive safety margin

**Critical confirmation:** 30B Q2+residual with layer paging fits in 16 GB RAM with 12+ GB headroom under ALL tested configurations.

**30B Q4+residual (the most demanding scenario tested) peaks at 3.7 GB — also SAFE.**

---

## F. Unsafe Scenarios

**None observed in simulation.** The smallest window that still passes would require pushing layer_bytes or context much higher than estimated.

Even 30B Q4 base at c=4096 with 4-layer window is SAFE (3.5 GB peak vs 16 GB budget).

To find an unsafe scenario, we'd need to either:
1. Reduce RAM to < 4 GB (unrealistic)
2. Increase layer_bytes dramatically (e.g., f32 instead of Q2)
3. Increase window_size beyond feasible I/O capacity

---

## G. Bottleneck Interpretation

### Memory is NOT the bottleneck

The simulation puts peak resident at 3–4 GB for all scenarios — well within 16 GB with 12+ GB headroom. This is the **key finding**.

### IO bandwidth IS the bottleneck

At steady state (10 tokens/sec):
- **Q2 base alone:** 37 MB/token × 10 tok/sec = **370 MB/s IO required**
- **Q2+all residuals:** 50 MB/token × 10 tok/sec = **500 MB/s IO required**
- **Q4+residuals:** 102 MB/token × 10 tok/sec = **1,020 MB/s IO required**

**NVMe throughput:** 3–7 GB/s — handles all of the above comfortably
**SATA SSD:** ~500 MB/s — marginal for Q4+residential at 10 tok/sec, OK for Q2
**USB 3.0:** ~100–125 MB/s — **inadequate for all scenarios**

The dominant risk has shifted from memory to **IO latency**: prefetch must complete before the layer is needed, or the decode stalls. At higher throughput targets (20–30 tok/sec), even SATA becomes borderline.

### Second bottleneck: Prefetch latency

With prefetch_distance=1, each layer must load in < layer_compute_time. Layer compute for Q2 30B is ~50–200 ms per layer (estimated). If IO takes longer than compute, the pipeline stalls.

### Third bottleneck: Thermal/power

Continuous NVMe reads at 500 MB/s = ~2.2 GB/min = moderate thermal load. Manageable on desktop NVMe. Risky on passive-cooled mobile NVMe.

---

## H. Recommended Window/Policy

**Window size recommendation:**
- `window=4` is the sweet spot: ~150–220 MB peak layer resident, sufficient prefetch overlap
- `window=8` adds only ~300 MB peak but reduces eviction frequency (beneficial for I/O pattern stability)

**Policy recommendation:**
- `all_residuals` is feasible (3.4 GB peak total) and provides maximum quality recovery
- `budget_greedy @ 512 MB` produces identical peak to `all_residuals` in simulation (all tensors fit within budget anyway)
- At 16 GB RAM, the budget constraint doesn't bind — residual policy choice is quality-driven, not memory-driven

**For maximum quality:** `all_residuals` — all 5 validated families, peak ~3.4 GB
**For maximum speed (if IO is tight):** `attention_partial` — smallest residual set

---

## I. Next Blocker: IO Bandwidth

The simulation shows memory feasibility. The next real question is **IO bandwidth**.

**Phase 28AE — IO Bandwidth/Latency Estimate** must answer:
1. What NVMe read bandwidth is needed per token at target throughput?
2. Can NVMe prefetch hide IO behind compute (i.e., does compute take longer than IO)?
3. What's the minimum prefetch_distance for no-stall operation at 5/10/20 tok/sec?
4. Is SATA usable for Q2-only paging at 5 tok/sec? At 10 tok/sec?
5. Is USB storage feasible at all?

**Key numbers to measure:**
- Layer compute time per layer for Q2/Q4 (requires actual inference timing)
- NVMe sequential read speed for ~37 MB contiguous reads
- NVMe random read speed for ~37 MB (worst case)
- Prefetch overlap ratio: how much of IO can be hidden by compute?

---

## J. Recommended Next Phase

**Phase 28AE — IO Bandwidth/Latency Estimation**

Goal: Determine whether NVMe IO can sustain the layer paging rate required for real-time decode.

Steps:
1. Estimate or measure layer compute time for Q2/Q4 30B (if possible)
2. Compute required IO bandwidth for target tok/sec
3. Compare against NVMe/SATA actual throughput
4. Determine minimum prefetch_depth for no-stall operation
5. Assess USB feasibility as lower bound

**No model files required** — use metadata + known NVMe benchmarks.

---

## K. Models/Sidecars/F32 Refs Staged?
**NO.** No model files, no sidecars, no f32 refs staged.

## L. Secrets Detected?
None.

## M. Tags Touched?
None.

---

## Files Committed
- `examples/speculative/prt_layer_residency_planner.py` — simulated layer residency planner
- `examples/speculative/results/PHASE28AD_SIMULATED_LAYER_RESIDENCY_PLANNER.md` — this report
- `examples/speculative/results/phase28ad_simulated_layer_residency_planner.json` — structured results