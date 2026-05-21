# Phase 26L: Model-Residency SDI Roadmap

**Verdict:** `PASS_PHASE26L_MODEL_RESIDENCY_ROADMAP` | `RECOMMEND_STANDALONE_SDI_PACKET_RUNTIME` | `RECOMMEND_MEMORY_RESIDENCY_GUARD` | `RECOMMEND_7B_SAFE_CONTEXT_HARNESS` | `PARK_PRT_V3_WEIGHT_RESIDENCY`

**Date:** Thu 2026-05-21 10:35 EDT
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Current HEAD:** `efa3f0319`

---

## A. Branch
```
experimental/prt-phase19a-alt-sidecar-backed
```

## B. Current HEAD
```
efa3f0319be72adfa4886fe2824e8939f4a57e9d
```

## C. Corrected SDI Thesis

> **Sub-Dense Inference is a CPU/RAM survival strategy for dense models:** reduce how much context, KV, weight memory, and dense compute must be active per token while preserving correctness through fallback and verification.

**What it is NOT:**
- NOT agentic
- NOT OpenClaw integration
- NOT Smart Agent Router
- NOT "just routing"
- NOT classic MoE

**What it IS:**
Making dense AI models run inside constrained CPU/RAM limits they were not naturally designed for.

---

## D. The Actual CPU/RAM Problem

### Hardware Reality

Running a model on hardware it "has no right running" means:

| Constraint | What Happens | Why It Kills Us |
|------------|-------------|-----------------|
| **RAM capacity** | Model weights + KV + context must fit in physical RAM | 7B Q4 ≈ 4–5 GB bare, context at c=8K adds ~1–2 GB KV. Leaves ~8–10 GB for OS + llama.cpp + buffers. Tight. |
| **RAM bandwidth** | Every token requires reading weights from RAM | CPU-RAM bandwidth is ~25–50 GB/s. A 7B model reads ~4–5 GB per token. That's 80–200 ms/token minimum — before compute. |
| **KV/context growth** | Active context grows linearly with tokens | At c=8192, KV for 7B ≈ 1.6 GB. At c=16K, ≈ 3.2 GB. KV eviction or OOM. |
| **Swap death** | Linux swap kicks in when RAM exhausted | Swapping to disk at ~0.5 MB/s = catastrophic latency. System unusable. Observed: swap starts ~420 MB used at idle. |
| **Dense per-token weight streaming** | Every token reads full weight matrices | No weight reuse across tokens in standard decode. FFN layers dominate memory bandwidth. |
| **CPU-only runtime** | No GPU memory bandwidth advantage | GPU→CPU gap: ~10x in bandwidth, ~100x in compute density. Can't close with clever software. |

### Target

**Run the largest useful model possible within CPU/RAM limits, with no swap death, acceptable quality, and controlled degradation.**

The question is not "can the model run?" — it's "what must be ACTIVE per token?"

---

## E. Three-Layer Model

### Layer 1 — Context/KV Residency

**Problem:** Too much active context makes KV/cache/memory pressure grow linearly with token count.

**Current solution candidate:** SDI_CONTEXT_PACKET
- Pinned facts (critical verbatim state)
- Recent window (bounded, N most recent turns)
- Structured summary (what happened)
- Open loops (pending decisions)
- Dropped-context report (what was discarded)
- 33–94% token reduction observed (Phase 26J)

**Near-term goal:** Keep 7B inside safe c=2K–8K active context while preserving task-critical facts.

**Evidence so far:**
- Packet builder: 5/5 scenarios, 43/43 checks passing
- Ollama qwen2.5:0.5b probe: 0.83/1.0 avg, dramatic improvement vs no-packet baseline
- Swap stable with packets (0 KB delta)

### Layer 2 — Weight Residency / Bandwidth

**Problem:** Even if context is small, dense model weights must be streamed from RAM every token.

**Possible future mechanisms:**
- Native-layout compressed sidecars
- PRT v3 (speculative FFN pre-computation)
- Selective FFN/channel/block execution
- Model weight paging without swap death
- CPU-native quant formats (e.g., Q4_0 vs Q4_K_M layout differences)
- Full backend op, not custom-op hacks

**Current status:** Research only. Do not implement yet.

**PRT lesson:** Custom-op hacking (Phase 10E) taught us: do not blind-implement complex memory-side mechanisms without a clean native backend path. PRT v3 is parked until a real backend path exists.

### Layer 3 — Activation / Verification

**Problem:** Dense models activate too much computation every token — every FFN layer, every channel.

**Possible mechanisms:**
- Speculative decoding (draft + verify)
- Selective layer/FFN activation (skip low-importance components)
- Early exit (stop when confident)
- Fallback/verification (accept-block verification)
- Self-correcting output verification

**Current status:** Speculative decoding is the most practical component. It does not solve RAM capacity alone — it solves throughput.

**Key insight:** Verification-based methods (accept-block, self-correct) are the most promising because they add resilience without requiring architectural changes.

---

## F. Ranked First-Build Options

### Option A — Standalone SDI Packet Runtime ✅ **Recommended**
Build a CLI tool that:
- Takes raw context + pinned facts + current request
- Builds SDI_CONTEXT_PACKET
- Chooses tier based on memory budget
- Optionally sends to selected local backend
- Records quality and token reduction

| Criterion | Score | Notes |
|-----------|-------|-------|
| Directly solves CPU/RAM limit | ★★★ | Reduces active context directly |
| Time-to-demo | ★★★ | Leverages existing packet builder |
| Engineering risk | ★★☆ | Builder exists; CLI wrapper is simple |
| Quality preservation | ★★★ | Phase 26K proved packets preserve facts |
| Usefulness to SDI thesis | ★★★ | First tangible runtime evidence |

### Option B — Memory Residency Guard ✅ **Recommended (paired with A)**
Build a guard that:
- Checks RAM/swap/model/context before inference
- Refuses unsafe local inference
- Chooses safe context budget
- Outputs policy decision

| Criterion | Score | Notes |
|-----------|-------|-------|
| Directly solves CPU/RAM limit | ★★★ | Prevents swap death |
| Time-to-demo | ★★★ | Simple policy checks |
| Engineering risk | ★★★ | Direct sysinfo reads |
| Quality preservation | ★★☆ | Indirect — prevents failure modes |
| Usefulness to SDI thesis | ★★☆ | Infrastructure, not core mechanism |

### Option C — 7B Safe Context Harness ⚠️ **Recommended after A+B**
Build a controlled harness to run 7B with packetized context at c=2K/4K/8K and prove no swap + quality retention.

| Criterion | Score | Notes |
|-----------|-------|-------|
| Directly solves CPU/RAM limit | ★★★ | Real 7B in safe bounds |
| Time-to-demo | ★★☆ | Needs llama.cpp harness + eval |
| Engineering risk | ★★☆ | llama-cli available; harness is new |
| Quality preservation | ★★★ | Tests real model at target scale |
| Usefulness to SDI thesis | ★★★ | Proves 7B viability with packets |

### Option D — PRT v3 Design 🅿️ **Parked**
Design what a real native PRT backend would require.

| Criterion | Score | Notes |
|-----------|-------|-------|
| Directly solves CPU/RAM limit | ★★★ | Weight streaming is the real problem |
| Time-to-demo | ★☆☆ | Design only, no implementation |
| Engineering risk | ★☆☆ | High — requires native backend |
| Quality preservation | ★★★ | Correctness through verification |
| Usefulness to SDI thesis | ★★★ | But too early; context/KV must be solved first |

**Ranking: A > B > C > D (for near-term), D is long-term.**

---

## G. Recommended Solution Path

### Step 1 — Build Standalone SDI Packet Runtime + Memory Guard ✅ (Next: Phase 26M)

**Why first:**
- Proven component (packet builder) + low-risk addition (guard)
- Delivers a runnable tool with measurable outputs
- No model changes, no PRT, no weight residency complexity
- Establishes the evaluation baseline for context/KV SDI

**Core deliverables:**
- `sdi_packet_runtime.py` CLI: accepts conversation + pinned facts → SDI_CONTEXT_PACKET + quality report
- `memory_guard.py`: checks RAM/swap/model/context → safe/tier decision
- Integration: runtime calls guard before building packet

### Step 2 — Use Step 1 to Keep 7B Under Safe Context Limits

**Why after Step 1:**
- Need the runtime working before testing with 7B
- 7B at c=8K with packet should be safe; need to prove it
- Can benchmark: packet quality vs full context at same c

### Step 3 — Revisit Weight-Residency SDI (after context/KV proven)

**What to revisit:**
- PRT v3 design (native backend required — do not hack ggml)
- Selective FFN/block activation
- Model weight paging without swap
- Native-layout sidecar format

**Why wait:**
- PRT phase (10E) taught us: don't implement memory-side mechanisms without clean backend
- Context/KV SDI is lower risk and already producing results
- Weight residency is the deeper invention but higher risk — prove context first

---

## H. Success Metrics

### For Standalone SDI Runtime (Step 1)

| Metric | Target | How Measured |
|--------|--------|---------------|
| Active context reduction | 50–90% | Packet tokens vs original conversation tokens |
| Swap growth | 0 KB | Swap measured before/after inference |
| Critical facts retained | 100% | Phase 26K-style assertions |
| Local model can answer from packet | >0.8 avg score | Phase 26K-style quality probe |
| Deterministic packet output | Identical packet for identical input | Run same input twice, diff should be empty |
| Dropped-content report | Non-empty | Dropped sections report present |

### For Future Weight-Residency SDI

| Metric | Target | How Measured |
|--------|--------|---------------|
| Resident memory reduction | >20% vs native Q4 | `ps_mem.py` or `/proc/$PID/status` |
| Output matches native | <5% quality delta | Perplexity or task-based eval |
| No custom-op overhead trap | <10% latency overhead | Wall-clock timing |
| Native backend from day one | No ggml hacks | Clean implementation path |

---

## I. Kill Criteria

**Kill a mechanism if ANY of:**
- Packet loses critical facts (assertion failure)
- Output quality collapses (>20% drop vs baseline)
- Memory savings do not reduce swap risk (swap still grows)
- Mechanism only routes around the model instead of reducing active residency
- Implementation complexity exceeds payoff
- Custom-op path re-emerges without clean backend

**Specific to weight-residency SDI:**
- If PRT v3 requires modifying ggml internals → park it
- If selective FFN activation requires model retraining → park it
- If sidecar format requires new model files → park it

---

## J. What NOT to Do

- ❌ Do NOT integrate with OpenClaw
- ❌ Do NOT integrate with Smart Agent Router
- ❌ Do NOT frame as agentic
- ❌ Do NOT build routing around "which model to use" as the main invention
- ❌ Do NOT implement PRT custom ops without a clean native backend
- ❌ Do NOT assume 7B at c=16K is safe (it isn't — swap death observed at ~16K)
- ❌ Do NOT conflate routing intelligence with model-residency intelligence
- ❌ Do NOT build MoE — this is about making dense models survive constrained hardware, not expert selection

---

## K. Recommended Next Phase

**Phase 26M — Build Standalone SDI Packet Runtime + Memory Guard**

Concrete deliverables:
1. `sdi_packet_runtime.py` — CLI tool wrapping the packet builder
   - Input: raw conversation file, pinned facts JSON, optional backend URL
   - Output: SDI_CONTEXT_PACKET, token reduction report, tier decision, memory guard check
   - Records output to `sdi_runtime.log` and `sdi_packet_<timestamp>.txt`
2. `memory_guard.py` — standalone memory check utility
   - Reads `/proc/meminfo`, `swapinfo`, model size, context length
   - Outputs: SAFE/TIER_0/TIER_1/TIER_2/UNSAFE with reason
   - Used before packet building to choose tier
3. Integration test: run packet runtime on Phase 26K-R2 scenarios, verify same quality
4. Commit docs/JSON only (no binaries, no models)

---

## L. Models/Sidecars/F32 Refs Staged?
```
NO
```
Phase 26L is a planning phase. No inference, no benchmarks, no code changes beyond docs.

---

## M. Secrets Detected?
```
NO
```

---

## N. Tags Touched?
```
NO
```

---

## Safety Scan

```
git status --short
 M ggml/src/ggml-cpu/ops.cpp
?? examples/speculative/results/PRT_PHASE24R_NATIVE_MULMAT_PROBE.md
?? examples/speculative/results/phase24r_native_mulmat_probe.json

No models staged.
No inference run.
No code changes.
Docs only.
```

---

*Phase 26L complete. Roadmap defined. Next: Phase 26M — standalone SDI Packet Runtime + Memory Guard.*