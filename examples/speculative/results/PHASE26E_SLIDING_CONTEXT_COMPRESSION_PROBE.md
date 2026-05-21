# Phase 26E: Sliding Context Compression Policy Probe

**Verdict:** `BLOCKED_MACHINE_STATE — SWAP_DEATH`

**Date:** Wed 2026-05-20 23:45 EDT
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Current HEAD:** `a02f8bff5` (Phase 26D commit)

---

## A. Branch
```
experimental/prt-phase19a-alt-sidecar-backed
```

## B. Current HEAD
```
a02f8bff5
```

## C. Model
```
/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf (Q4_K_M, 4.4 GB)
```

## D. Baseline Memory Finding (from Phase 26D)

| Context | Swap Δ | Status |
|---------|---------|--------|
| c=2048 | 0 MB | ✅ Safe |
| c=4096 | +73 MB | ✅ Safe |
| c=8192 | +66 MB | ⚠️ Low pressure |
| c=16384 | +394 MB | 🔴 Danger zone |

Key finding: c=16384 activates ~400 MB swap. c=8192 is the safe ceiling for RAM-resident operation on 7B Q4_K_M.

---

## E. Prompt Design

**Synthetic long-context task created at:**
`/media/matthew-villnave/VL_usb/prt_scratch/phase26e/full_context_task.txt`

**Design:**
- Section 1 (PINNED): Project setup facts — codename, lead, API key (FAKE), decision, preference, constraint, date
- Section 2 (DISTRACTOR): ~2000 words of Roman Empire text — pure memory pressure filler
- Section 3 (PINNED): Project mid-status — phase, budget, hardware, team
- Section 4 (RECENT): 10-question task requiring retrieval from pinned facts and Roman Empire distractor

**Answer key:** `/media/matthew-villnave/VL_usb/prt_scratch/phase26e/answer_key.md`

**Scoring:** 10 questions × 1 point each. 9–10 = excellent, 7–8 = good, 5–6 = partial, 0–4 = failing.

---

## F. Variants Created

| Variant | File | Approx tokens | Purpose |
|---------|------|--------------|---------|
| A — Full context | `full_context_task.txt` | ~3,000 | Baseline (large ctx) |
| B — Recent window only | `variant_B_recent_window.txt` | ~300 | Quality-loss baseline — no early facts |
| C — Pinned facts + recent | `variant_C_pinned_facts.txt` | ~450 | Manual extraction of key facts |
| D — Summary + pinned + recent | `variant_D_summary.txt` | ~515 | Summarized old context + pinned |
| E — ContextOS packet + recent | `variant_E_contextos.txt` | ~465 | Structured packet format |

---

## G. Machine State — SWAP DEATH (Critical Finding)

**Discovery:** Machine entered **swap death** during Phase 26E preflight and testing.

**Swap state at time of discovery:**
```
SwapTotal:  4.0 GB
SwapUsed:   4.0 GB  (FULL)
SwapFree:   484 KB  (essentially zero)
```

**What happened:**
1. Previous Phase 26D runs at c=16384 cumulatively pushed ~400 MB into swap
2. OS processes and background activity filled remaining swap
3. litert-proxy (server.py, ~10 GB resident when including model) was still occupying ~4.8 GB RAM
4. When litert-proxy was killed to free RAM, swap remained saturated from prior paging
5. Subsequent llama-cli attempts were killed by SIGKILL/SIGTERM as the OS couldn't allocate memory

**Key insight:** Swap usage is **persistent**. Even after killing processes, the OS doesn't automatically reclaim swap pages. Swap fills cumulatively and must be explicitly managed.

**Evidence of swap death from attempts:**
- c=2048 full context test: PRT-NATIVE logs printed (model loading) then SIGKILL
- Variant B recent window test: SIGKILL after ~90 seconds
- Variant C pinned facts test: SIGTERM after ~40 seconds with swap at ~4 GB used
- Simple "The capital of France is" test: returned `[0m]` with exit code 124 (timeout) — model loading was slow/swapping

**Safe pre-flight memory (ideal state from earlier today):**
```
MemAvailable: ~9.0–10.0 GB
SwapUsed:     ~390–520 MB
SwapFree:     ~3.5–3.8 GB
```

**Current blocked state (swap death):**
```
MemAvailable: ~6.4 GB (but not accessible due to swap full)
SwapUsed:     ~4.0 GB (FULL)
SwapFree:     ~484 KB
```

---

## H. Memory/Swap Table — BLOCKED

| Variant | Attempted ctx-size | Result | Swap Δ | Notes |
|---------|-------------------|--------|--------|-------|
| A — Full | 16384 | ❌ SIGKILL | unknown | Swap death |
| B — Recent | 2048 | ❌ SIGKILL | unknown | Swap death |
| C — Pinned | 1024 | ❌ SIGTERM | unknown | Swap death |
| D — Summary | — | Not attempted | — | Blocked |
| E — ContextOS | — | Not attempted | — | Blocked |

**No variant could complete.** Machine was in swap death state throughout Phase 26E attempt.

---

## I. Quality Score Table — NOT TESTED

No variants completed due to swap death. Quality scoring is **blocked**.

---

## J. Best Policy — THEORETICAL RANKING

Based on design analysis (no live data due to swap death):

| Policy | Token reduction | Quality preservation | Complexity | Ranking |
|-------|---------------|---------------------|------------|---------|
| Pinned facts + recent window (C) | ~85% | High (manual extraction) | Low | 🥇 Recommended first |
| ContextOS packet + recent (E) | ~85% | High (structured) | Medium | 🥈 If C works well |
| Summary + pinned + recent (D) | ~83% | Medium-high (LLM summary) | Medium | 🥉 If C insufficient |
| Recent window only (B) | ~90% | **Low** (loses all early facts) | None | ⚠️ Quality baseline only |
| Full context (A) | 0% | 100% | None | Baseline |

**Theoretical conclusion:** Pinned facts extraction (Variant C) is the right first policy — it preserves the most important information in the smallest token footprint, without requiring LLM summarization.

---

## K. Recommended Active Window Size — DETERMINED THEORETICALLY

From Phase 26D memory matrix:
- c=8192: safe, no meaningful swap
- c=16384: danger zone (+394 MB swap)
- To fit c=16384 safely: need ~50% KV reduction

**Recommended operating parameters:**
- Active context: ~4,096 tokens (safe, no swap)
- Pinned section: ~500 tokens (project facts, constraints, system prompt)
- Total effective context: ~4,600 tokens (vs 16,384 ceiling)
- Reduction: ~72% of max context

This gives a working system that stays RAM-resident at what would otherwise be c=16K workload, while preserving all important facts in the pinned section.

---

## L. Success Criteria Result

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| 50%+ active context reduction | Yes | ✅ (theoretical) | BLOCKED — not tested |
| No meaningful swap increase | Yes | ❌ (swap death) | FAILED — swap fully utilized |
| Quality degradation <10–15% | <15% | N/A | BLOCKED — no tests ran |
| Output remains sane | Yes | N/A | BLOCKED — no tests ran |

**Blocked by:** Machine state — swap death. No inference possible in current state.

---

## M. Kill Criteria Result

| Criterion | Kill condition | Result |
|-----------|--------------|--------|
| Loses critical facts | Hallucination | N/A (blocked) |
| No memory improvement | <20% reduction | N/A (blocked) |
| Quality collapses | >50% deg | N/A (blocked) |

**Not evaluated due to swap death.**

---

## N. Recommended Next Phase

**Phase 26F: Machine Recovery + Swap Reclaim**

**Before any further inference testing:**
1. Drop swap: `swapoff -a && swapon -a` (clears accumulated swap pages)
2. Verify clean state: `free -h` showing swap free
3. Kill any background llama-server or litert-proxy processes
4. Confirm: SwapFree ~3.5 GB+ before running more inference

**Then retry Phase 26E with Variant C (pinned facts) first:**
- Variant C is simplest and theoretically highest quality
- Run with c=1024 to c=2048 (well within safe zone)
- Establish baseline quality score
- If Variant C works, test Variant E (ContextOS packet)
- Compare swap behavior between variants

**Why Variant C first:** It requires no LLM summarization step, no format parsing, no complex infrastructure — just a well-structured pinned section that a human (or simple heuristic) can extract. If this preserves quality, it proves the concept without building full automation.

---

## O. Models/Sidecars/F32 Refs Staged?

**No.** Only the synthetic prompt/answer key files were created (not staged — they stay in `/media/matthew-villnave/VL_usb/prt_scratch/phase26e/`).

---

## P. Secrets Detected?

**No.** The synthetic task uses only fake values:
- API key: `sk-fake-argus-testkey-abc123xyz789`
- Project: Project ARGUS
- Lead: Dr. Sarah Chen
- All other values (budget, dates, etc.) are synthetic

---

## Q. Tags Touched?

**No tags touched.**

---

## Safety Scan

```
git status --short
A  examples/speculative/results/PHASE26D_KV_CONTEXT_MEMORY_BASELINE.md
A  examples/speculative/results/phase26d_kv_context_memory_baseline.json
 M ggml/src/ggml-cpu/ops.cpp
?? examples/speculative/results/PRT_PHASE24R_NATIVE_MULMAT_PROBE.md
?? examples/speculative/results/phase24r_native_mulmat_probe.json

No models/sidecars/f32 refs staged.
No secrets found.
No tags touched.
```

---

## Critical Machine State Note

⚠️ **The 15GB machine is in swap death.** Before any further PRT or SDI testing:

```bash
sudo swapoff -a && sudo swapon -a
```

Or reboot. The accumulated swap pages from Phases 26D and 26E's memory pressure tests have filled the 4GB swap to capacity, and the OS will not recover it automatically.

**Never let swap exceed ~1 GB on this machine** — it cannot recover gracefully. When swap reaches ~1 GB used during a long inference run, abort and reclaim.

---

## Verdicts

| Verdict | Value |
|---------|-------|
| `BLOCK_PHASE26E_TESTS` | ✅ All tests blocked by swap death |
| `BLOCKED_MACHINE_STATE` | ✅ Swap at 4.0 GB / 4.0 GB (FULL) |
| `BLOCKED_7B_RUNTIME` | ✅ No 7B inference possible in current state |
| `DISCOVERED_SWAP_DEATH` | ✅ Machine accumulates swap pages that persist after process death |
| `RECOMMEND_SWAP_RECOVERY_FIRST` | ✅ Must clear swap before continuing |
| `RECOMMEND_VARIANT_C_FIRST` | ✅ Pinned facts = simplest, highest quality |
| `THEORETICAL_PROBE_VALUES_VALID` | ✅ Design analysis valid, execution blocked |

---

*Phase 26E partially executed. Design, variants, and answer key created. Machine state blocked all testing. Must recover swap before retry.*