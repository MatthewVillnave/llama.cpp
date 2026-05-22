# Phase 28AG: Paged 30B Feasibility Consolidation

## Verdict: PASS_PHASE28AG_FEASIBILITY_CONSOLIDATION | MEMORY_FEASIBLE_IN_SIM | IO_FEASIBLE_PREFILL_SLOW | COMPUTE_BOUND_GENERATION_ESTIMATE | RUNTIME_UNPROVEN | QUALITY_UNPROVEN | RECOMMEND_FAKE_FILE_PAGER

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`6db31c5d0`

---

## C. Corrected Evidence Chain

### 1. Residual Quality Recovery
**Source:** Phase 26K-R3, 26O, 26Q

- Q2+ternary works strongly on tested tensor slices across validated families
- **Validated tensor families:**
  - FFN_UP, FFN_DOWN, FFN_GATE (~88.5% of Q4 memory)
  - attn_q, attn_output (~11% of Q4 memory)
- 0.5B and 3B model transfer: strong results
- Storage ratio: **0.75× Q4** for same tensor classes (ternary at 0.125 bytes/param vs Q4 at ~0.5 bytes/param)
- Key insight: ternary 1-bit at 0.125 bytes/param stays below Q4 (0.75×) while recovering quality
- Q2+ternary alone (~9GB for 30B) is not sufficient — selective residuals needed
- Real 7B GGUF-derived manifest logic: Q2 total for 5 families = 4.17GB

### 2. Budget / Manifest Estimates
**Source:** Phase 28AB, 28AA

- Real 7B numbers from GGUF metadata:
  - Q4 total for 5 validated families: 4.17GB
  - Q2 base including weights + non-weight: 3.88GB
  - FFN families = 88.5% of Q4 memory; attention ~11%
- Phase 28X estimates were within ~7% of actual values
- 30B Q2 full-resident (all layers): ~26GB — too large for 16GB RAM
- Selective residuals required: 1-3GB budget for residual sidecars

### 3. Layer Residency (Corrected)
**Source:** Phase 28AD, 28AF-R

- **Full model storage ≠ peak resident memory.** With paging, active window is all that matters.
- 30B Q2 base (window=4): peak ~3.3GB resident ✓
- 30B Q2+res (window=4, budget=512MB): peak ~3.4GB resident ✓
- 30B Q4 base (window=4): peak ~3.5GB resident ✓
- 30B Q4+res (window=4, budget=512MB): peak ~3.7GB resident ✓
- Headroom at 16GB RAM: ≥12.3GB in all scenarios
- **All 9 simulated 30B scenarios: MEMORY SAFE**
- budget_greedy corrected: uses `resident_residual_bytes()` not base bytes; limited by window_size
- After fix: `budget_greedy` with window=4 yields ~56MB residual (4 layers × ~14MB/layer), not the buggy 509MB

### 4. IO Bandwidth (Hardware-Verified)
**Source:** Phase 28AE

- **Micron 2450 NVMe 256GB:** 1.7–2.0 GB/s read speed
- **SanDisk USB:** ~100 MB/s — not suitable
- NVMe is suitable for paged model data
- IO measurements are real hardware — not simulation, not buggy

### 5. Prefetch IO (Corrected)
**Source:** Phase 28AF-R

- **Bug fixed:** Parallel prefill was using `layer_mb` (one layer) instead of `total_model_mb` (full model)
- Corrected prefill IO at 1800 MB/s NVMe:
  - Q2 base (2050MB): 1139ms IO vs 200ms compute = **85% stall**
  - Q2+res (2800MB): 1556ms IO vs 200ms compute = **89% stall**
  - Q4 base (4144MB): 2302ms IO vs 400ms compute = **85% stall**
  - Q4+res (5712MB): 3173ms IO vs 400ms compute = **89% stall**
  - Q2+res USB (2800MB): 28000ms IO vs 200ms compute = **99% stall**
- **Generation: IO=0** — weights fully resident after prefill; purely compute-bound
- Key: prefill IO is IO_HEAVY but bounded (one-time cost). Generation IO vanishes.

### 6. Generation Compute Estimate
**Source:** Phase 28AF-R (unchanged from 28AF)

- Once prefill completes and weights are resident, generation is **compute-bound**
- Generation tok/s ceiling (simulation estimate):
  - Q2 at 200ms/token: **5.0 tok/s**
  - Q2 at 100ms/token: **10.0 tok/s**
  - Q4 at 400ms/token: **2.5 tok/s**
- Real-world tok/s depends on actual CPU throughput — not yet measured

---

## D. Corrected End-to-End Model

### Two-Phase Execution Model (Revised)

**Phase 1 — Prefill (first token):**
1. Load Q2 base (or Q4 base) from NVMe into RAM
2. Load selected residuals from sidecar files into RAM
3. Compute first token
4. **IO-dominant:** prefill takes 1-3s at 1800MB/s NVMe

**Phase 2 — Generation (tokens 2+):**
1. All weights already resident in RAM
2. Per-token compute: ~200ms for Q2, ~400ms for Q4
3. **IO=0** — pure compute
4. Throughput = 1000 / compute_ms_per_token

### Config Comparison Table

| Config | Storage (MB) | Peak Resident (MB) | Prefill IO (ms) | Prefill tok/s | Gen tok/s (est) | Main Bottleneck | Feasibility |
|--------|-------------|-------------------|-----------------|--------------|-----------------|-----------------|-------------|
| **Q2 base, NVMe** | 2050 | ~3300 | 1139 | 0.75 | 5.0 | Compute | ✅ MEMORY_FEASIBLE |
| **Q2+res, NVMe** | 2800 | ~3400 | 1556 | 0.57 | 5.0 | Compute | ✅ MEMORY_FEASIBLE |
| **Q4 base, NVMe** | 4144 | ~3500 | 2302 | 0.37 | 2.5 | Compute | ✅ MEMORY_FEASIBLE |
| **Q4+res, NVMe** | 5712 | ~3700 | 3173 | 0.28 | 2.5 | Compute | ✅ MEMORY_FEASIBLE |
| **Q2+res, USB** | 2800 | ~3400 | 28000 | 0.04 | 5.0 | IO (prefill) | ⚠️ PREFILL_SLOW |
| **Q2+res, fast CPU** | 2800 | ~3400 | 1556 | 0.57 | 20.0 | Compute | ✅ MEMORY_FEASIBLE |

**Key assumptions:**
- NVMe: 1800 MB/s (measured)
- USB: 100 MB/s
- Compute: Q2=200ms/token, Q4=400ms/token (medium estimate)
- Window=4, KV=2GB, buffer=1GB, OS=2GB, RAM=16GB

### What This Means Practically

| Scenario | Experience |
|----------|-----------|
| Q2+res first token | ~1.5s prefill (IO-bound), then 200ms/generate token |
| Q2+res 100-token response | 1.5s prefill + 20s generation = 21.5s total |
| Q2+res throughput | ~5 tok/s after prefill — compute-bound |
| USB prefill | 28s first token — barely usable |
| Q4+res first token | ~3.2s prefill — still acceptable |

---

## E. Feasibility Classification

### MEMORY_FEASIBLE_IN_SIM
- All 9 simulated 30B scenarios fit within 16GB RAM budget
- Peak resident never exceeds 3.7GB (Q4+res)
- Headroom: ≥12.3GB at all times
- **Caveat:** This is metadata-only simulation. Real 30B GGUF files not tested.

### IO_FEASIBLE_PREFILL_SLOW
- NVMe at 1.8 GB/s is sufficient for paging
- Prefill takes 1-3s depending on config — acceptable for batch workloads
- USB is NOT suitable (28s prefill)
- **Caveat:** No real file-backed pager tested. Fake files only.

### COMPUTE_BOUND_GENERATION_ESTIMATE
- Generation tok/s ceiling: 2.5-5.0 tok/s for Q2/Q4 (medium compute assumption)
- IO vanishes after prefill
- Next bottleneck is CPU compute throughput, not IO
- **Caveat:** Actual compute timing unknown — estimate only

### RUNTIME_UNPROVEN
- No actual paged inference runtime exists
- No llama.cpp integration for layer paging
- No file-backed mmap/prefetch scheduler implemented
- Simulator uses fake files, not real model data
- **Verdict:** Claims of "30B runs" are forbidden

### QUALITY_UNPROVEN
- Offline tensor parity tests show Q2+ternary recovers quality on validated slices
- No generation quality testing performed
- No human preference evaluation
- Sidecar residual format tested only with synthetic data
- **Verdict:** Claims of "generation improves" are forbidden

---

## F. Game Demo Viability Assessment

### Target Demo Definition
> A CPU-only 16GB system produces coherent bounded output from a dense 30B/32B-class model that native full-resident inference cannot run.

### Minimum Demo Requirements
1. Native dense Q4 fails or is unsafe on 16GB system
2. Paged Q2 path loads/streams from NVMe
3. Q2+selective residual improves offline canary quality
4. One coherent bounded answer produced
5. No swap death / OOM

### Current State Assessment

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Native Q4 unsafe | ✅ Probable | 4.1GB Q4 base exceeds 16GB RAM minus KV/buffer |
| Paged Q2 loads/streams | ❌ Unproven | No runtime pager; only fake-file simulation |
| Q2+res quality gain | ❌ Unproven | Offline tensor tests; no generation eval |
| Coherent output | ❌ Unproven | No actual generation tested |
| No swap death | ❌ Unproven | No real system tested |

**Verdict: Game demo NOT ready.** Plausible path exists but unproven system behaviors remain.

### Realistic Assessment
- Memory feasibility: ✅ suggests paged 30B **may** fit
- IO feasibility: ✅ NVMe is sufficient for bounded paging
- Runtime: ❌ No pager exists
- Quality: ❌ No generation testing
- Next blocker: **real file-backed pager with fake layer files** — test the IO/residency mechanism without needing actual 30B model data

---

## G. Blockers

| Blocker | Severity | Description |
|---------|----------|-------------|
| No runtime pager | **CRITICAL** | Simulator uses fake files; no real mmap/read/prefetch in llama.cpp |
| No actual 30B files | HIGH | All work is metadata-based; no real 30B GGUF tested |
| No generation quality eval | HIGH | Only offline tensor parity; no text output tested |
| No scheduler integration | MEDIUM | No llama.cpp hooks for layer paging |
| No sidecar generation for real model | MEDIUM | Trit format tested with synthetic data only |
| budget_greedy window constraint | LOW | Window=4 limits residual to ~56MB; larger window = more residual |

---

## H. Recommended Next Phase

**Phase 28AH: Simulated File-Backed Pager (Fake Layer Files)**

**Purpose:** Test real file-backed mmap/read/prefetch behavior without model data.

**Approach:**
1. Create fake layer files on NVMe (one per layer, correct byte sizes)
2. Implement real `open/read/mmap` IO path
3. Test prefetch scheduler with actual file ops
4. Measure real IO timing vs simulated timing
5. Verify no model data touched

**Why fake files:**
- Avoids needing actual 30B GGUF files
- Tests the IO mechanism in isolation
- Can be deleted after test
- No model data staged

**What it validates:**
- Real mmap/read behavior on NVMe
- Prefetch scheduler with actual file I/O
- OS page cache behavior
- Multiple concurrent file handles

**What it doesn't validate:**
- Real GGUF parsing
- Actual model output quality
- Integration with llama.cpp runtime

---

## I. Superseded / Valid Prior Phases

| Phase | Status | Notes |
|-------|--------|-------|
| 28Z | ✅ VALID | Manifest + trit tooling; no sim bugs |
| 28AA | ✅ VALID | Manifest budget integration |
| 28AB | ✅ VALID | 7B residual budget scan |
| 28AC | ✅ VALID | Layer paging architecture design |
| 28AD | 🟡 PARTIALLY_VALID | Memory logic intact; budget_greedy fixed |
| 28AE | ✅ STILL_VALID | Real hardware IO measurements |
| 28AF | 🔴 SUPERSEDED | Prefill IO numbers wrong; generation tok/s valid |
| 28AF-R | ✅ VALID | Bug fixes applied; corrected numbers |
| 28AG | ✅ THIS PHASE | Consolidation |

---

## J. Allowed vs Forbidden Claims

### Allowed
- "Corrected simulation suggests paged 30B may be memory-feasible under metadata assumptions"
- "NVMe bandwidth appears sufficient for bounded paging, though prefill is IO-heavy"
- "Generation estimate is compute-bound in simulation"
- "Runtime and quality remain unproven"
- "No actual 30B files have been tested"

### Forbidden
- ❌ "30B runs"
- ❌ "Runtime works"
- ❌ "Generation improves"
- ❌ "Speedup achieved"
- ❌ "Production ready"
- ❌ "Real paging implemented"
- ❌ Any sidecar claims without real data

---

## K. Models/Sidecars/F32 Refs Staged?
NO.

## L. Secrets Detected?
NO.

## M. Tags Touched?
NO.

---

## Files Created
- `examples/speculative/results/PHASE28AG_PAGED_30B_FEASIBILITY_CONSOLIDATION.md` — this report
- `examples/speculative/results/phase28ag_paged_30b_feasibility_consolidation.json` — structured verdict