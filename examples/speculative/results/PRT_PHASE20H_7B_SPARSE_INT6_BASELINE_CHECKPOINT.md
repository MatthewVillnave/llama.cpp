# PRT Phase 20H — 7B Sparse INT6 Baseline Checkpoint

**Verdict:** `PASS_7B_SPARSE_INT6_BASELINE_CHECKPOINT`

---

## 1. Executive Summary

Sparse INT6 policy (10,20) is the current best corrected 7B baseline:

- Produces clean, non-gibberish outputs on the 8-prompt fair-settings suite
- Not fully quality-stable (P4 chemistry and P5 math show degraded responses)
- Slower than native (0.57x — approximately 43% slower)
- This is a **research baseline, not production-ready**

---

## 2. Current Trusted Baseline

| Parameter | Value |
|-----------|-------|
| Branch | `experimental/prt-phase19a-alt-sidecar-backed` |
| HEAD | `727108d98` |
| Model | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` |
| Sidecar dir | `/tmp/prt_sidecars_7b_int6_phase15b_packed` |
| Sidecar format | INT6 (packed, offset-32) |
| Policy | `--prt-only-layers 10,20` |
| Prompt suite | 8 factual/prose/code/JSON/reasoning/instruction prompts |
| Settings | n=40, c=512, t=4, single-turn |
| Timing | (10,20) ~4.8 t/s vs native ~8.4 t/s |
| Quality result | 8/8 clean, 3/5 factual correct, 2 partial |

---

## 3. What Held Up

- **Routing works:** `--prt-only-layers 10,20` activates only layers 10 and 20
- **Corrupt controls validate test harness:** (5,10,20) at n=40 produces degraded but not gibberish output; (5,10,20) at n=20 SIGKILLs (timeout)
- **Single-layer INT6 is clean:** Layer 0, 10, 20 individually produce clean output
- **Some sparse policies can avoid gibberish:** (10,20) is corruption-stable on the 8-prompt suite
- **(10,20) is corruption-stable** under Phase 20G fair settings

---

## 4. What Was Corrected

- **Speedup claims removed:** (10,20) is approximately 43% slower than native, not faster
- **Phase 20C "8-prompt stable" was too strong:** Outputs are clean but quality-degraded on some prompts
- **Phase 20C partial outputs were weak/partial, not hard corruption**
- **`prt_layer=0` was a misleading boolean log**, not a layer ID — layer ID is `IL=X`
- **Timing across Phase 15 vs Phase 20 cannot be compared** unless settings match (different thread count, context, n_predict)

---

## 5. What Remains Unsolved

- **Sparse policy quality weakness:** P4 chemistry (H2O) and P5 math (train speed) responses are degraded compared to native
- **Current PRT custom-op path is performance-negative:** ~43% slower than native
- **Best layer policy is not yet known:** (10,20) works but may not be optimal
- **No production-ready policy exists**
- **Backend/layout path likely needed for real speed:** Custom-op overlay approach appears to double-pay overhead

---

## 6. Allowed Claims

- ✅ 7B sparse INT6 (10,20) can produce clean, non-gibberish outputs on the tested 8-prompt fair-settings suite
- ✅ (10,20) is prompt-sensitive and quality-degraded on some factual/reasoning prompts
- ✅ (10,20) is slower than native under Phase 20G settings (0.57x)
- ✅ Sparse layer selection matters; full/broad INT6 replacement can corrupt output
- ✅ Corrupt controls validate the test harness
- ✅ Single-layer INT6 produces clean output
- ✅ Routing is confirmed functional via PRT_COMPUTE logs

---

## 7. Forbidden Claims

- 🚫 Production readiness
- 🚫 Universal stability across all prompts
- 🚫 Speedup (is slower than native)
- 🚫 Native-equivalent quality
- 🚫 All-prompt safety guarantee
- 🚫 GPU comparison claims
- 🚫 32B feasibility
- 🚫 RAM problem solved
- 🚫 All-layer INT6 works
- 🚫 INT8 fixes quality issues
- 🚫 (10,20) is final or best possible policy
- 🚫 Phase 15 timing comparable to Phase 20 timing
- 🚫 Phase 20C was a full PASS

---

## 8. Recommended Next Tracks

### Track A — Phase 20I: Better Sparse Policy Sweep

**Goal:** Search for policies better than (10,20) on quality while preserving no-gibberish stability.

**Candidate policies to test:**
- `(1,2)` — early layer pair
- `(5,10)` — mid-low pair
- `(20,27)` — late layer pair
- `(1,2,10)` — 3-layer early
- `(1,10,20)` — spread across depth
- `(2,10,20)` — spread across depth
- `(10,20,27)` — late-heavy 3-layer

**Rules:**
- Same fair settings as Phase 20G (n=40, c=512, t=4)
- 8-prompt suite
- Corrupt controls mandatory
- No speedup claims
- Primary metric = quality stability
- Secondary metric = timing

**Success criteria:**
- Cleaner factual responses on P4/P5 while maintaining 8/8 clean outputs
- Stable across repeat runs
- Timing acceptable (within 50% of native)

### Track B — Phase 20J: Backend / Performance Audit

**Goal:** Investigate why even clean sparse PRT is slower than native.

**Focus areas:**
- Custom-op overhead (ggml_custom_4d per token overhead)
- GGML graph integration cost
- Memory layout / cache behavior of INT6 sidecars
- Scalar vs vector compute path
- Tensor stride / output handoff between PRT custom op and SwiGLU
- Whether PRT should be a ggml-native backend kernel instead of custom-op overlay
- Whether current approach double-pays overhead (custom op + graph overhead)

**Rules:**
- No new quality claims
- No sidecar format changes until design is written
- Produce architecture/performance design first

**Recommended order:**
1. Finish Phase 20H checkpoint ← **HERE**
2. Run Phase 20I policy sweep
3. Run Phase 20J backend/performance design
4. Decide whether to continue this branch or fork PRT-v2

---

## Phase 20 Series Summary

| Phase | Verdict | Key Finding |
|-------|---------|-------------|
| 20A | BLOCKED | PRT routing not activating |
| 20B | PASS | Routing confirmed working |
| 20C | PARTIAL | 5/8 confirmed, not 8/8 |
| 20D | PARTIAL | INT8 vs INT6 same pattern |
| 20E | PARTIAL_AUDIT | Forensic audit complete |
| 20F | PASS_DIFF | Root cause found (multi-layer) |
| 20G | PARTIAL | (10,20) stable but slow |
| 20H | CHECKPOINT | Baseline frozen |

---

**Checkpoint frozen at:** `727108d98`
**Tag:** `PRT_PHASE20H_7B_SPARSE_INT6_BASELINE_CHECKPOINT`
**Date:** 2026-05-13