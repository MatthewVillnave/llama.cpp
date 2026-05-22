# SDI Runtime Phase 26/27: Final Summary

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Checkpoint:** `SDI_PHASE26W_RUNTIME_V0_1_1_CHECKPOINT`
**HEAD:** `c747e0845`

---

## What Is SDI Runtime?

**Sub-Dense Inference (SDI) Runtime** is a CPU/RAM model-residency strategy for dense AI models. The core problem: dense models (like Qwen2.5) are not designed for CPU/RAM-constrained environments, where memory capacity, memory bandwidth, KV/context growth, and swap risk are the primary constraints — not raw compute.

SDI is not:
- An agentic framework
- An integration with OpenClaw or Smart Agent Router
- A KV cache modification
- A weight-residency solution
- A speedup mechanism

SDI is: a **structured context selection system** that decides what context to present to a model at inference time, using a packet format, memory guard, and auto policy — without modifying model weights or llama.cpp internals.

---

## The Problem

CPU/RAM-limited dense inference is constrained by:
- **Memory capacity:** Models like qwen2.5:7B at c=16384 can exceed available RAM
- **Memory bandwidth:** Dense per-token weight streaming is bandwidth-heavy on CPU
- **KV/context growth:** Long conversations and noisy context consume RAM linearly with token count
- **Swap risk:** Unbounded context can trigger swap, poisoning subsequent runs

The goal: make dense models survive inside bounded RAM through smarter context management, not by making them faster.

---

## The PRT Lesson

The project explored a **PRT custom-op speed path** (Phase 10E) for CPU-side dense weight access. The work produced correct ggml kernels, but the path was not a speed win — it parked at a complexity/benefit crossroads. PRT remains a possible future native mechanism but is not part of the current SDI Runtime.

---

## SDI Runtime Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Memory Guard        │ ← Pre-run RAM/swap check
│  evaluate_guard()    │   Blocks if unsafe
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Auto Policy         │ ← Selects context strategy
│  sdi_policy()        │
│  (recent_only |      │
│   simple_summary |   │
│   sdi_packet |       │
│   no_packet | auto)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Packet Builder       │ ← Builds structured context
│  build_sdi_packet()  │   or passes through
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Ollama Backend       │ ← Local inference
│  (qwen2.5:0.5b/3b/7b)│
└─────────┬───────────┘
          │
          ▼
   Model Output
```

### Components

1. **Packet Builder** (`sdi_packet_builder.py`): Builds structured context packets from long/noisy conversation history, with deterministic output, metadata, and safety checks.

2. **Memory Guard** (`sdi_memory_guard.py`): Reads host RAM/swap state, checks for stale processes, returns conservative go/no-go for local inference.

3. **Auto Policy** (`sdi_packet_runtime.py`): Selects context strategy based on conversation characteristics — recent_only, simple_summary, sdi_packet, no_packet, or auto.

4. **Exact-Tool Mode**: Specialized routing for tool-output contexts where exactness is critical.

5. **Targeted Policy Gates**: Scenario-specific routing overrides for benchmark, license, and constraint scenarios.

---

## SDI Runtime v0.1.1 Results

### Evidence Table

| Phase | Evidence | Status |
|-------|---------|--------|
| 26I | Packet builder deterministic; 14 tests | PASS |
| 26J | Synthetic eval 5/5 scenarios, 43/43 checks | PASS |
| 26K-R2 | qwen2.5:0.5b packet probe avg 0.83, swap stable | PASS |
| 26M | Standalone runtime built; hostile eval mixed | PARTIAL |
| 26N | Packet v0.2 refinement; avg improved to 0.770, wins 4/5 | PASS |
| 26O | Auto policy 9/10 win/tie vs best fixed baseline | PASS |
| 26T | v0.1 checkpoint frozen | PASS |
| 26U | Expanded eval mixed; eager packet issue found | PARTIAL |
| 26V | Targeted policy gates; avg 0.950, 7/8 win/tie | PASS |
| 26W | v0.1.1 frozen; checkpoint tag created | PASS |
| 27B-R | qwen2.5:3b edge-case: Sc25 model ceiling confirmed | PASS_NARROW |
| 27E | Tiny 7B canary: 8 runs, swap stable, no aborts | PASS_NARROW |
| 27E-R | Scoring audit: no false pass, scores valid | PASS |
| 27F | Schema bug fixed (required+must_include union); no headline changes | PASS |

### Headline Results (v0.1.1, qwen2.5:0.5b)

- **Auto policy average:** 0.950 (after targeted gates)
- **Win/tie vs best fixed:** 7/8 scenarios
- **Gap to best fixed baseline:** narrowed to 0.013
- **Swap behavior:** Stable across all eval runs (0–12 MB delta)

### Narrow 3B Edge-Case Validation (Phase 27B-R)

- qwen2.5:3b confirmed closing Sc25 gap (0.750 → 0.850)
- Interpretation: **Model ceiling, not policy failure**
- Auto policy remains correct at both 0.5B and 3B scale
- No broad 3B validation claim warranted

### Tiny 7B Canary (Phase 27E)

- **8 runs completed:** Sc21 + Sc23 at c=2048/4096
- **Swap stable:** 677 MB throughout all runs
- **No abort criteria triggered**
- **Auto policy correctly selected recent_only on 7B**
- Sc21: 1.000 at all baselines
- Sc23: 0.850 (F1 miss, consistent with 0.5B/3B ceiling)
- **No broad 7B validation claimed**

---

## Allowed Claims

✅ **You MAY say:**
- SDI Runtime v0.1.1 is a standalone CPU/RAM model-residency prototype
- It uses structured context packets, memory guard, and auto policy to reduce harmful context pressure
- It improved auto-policy behavior on qwen2.5:0.5b evals, reaching 0.950 avg and 7/8 win/tie after targeted gates
- qwen2.5:3b targeted comparison confirmed Sc25 was a 0.5B model ceiling, not a policy failure
- A tiny guarded qwen2.5:7B canary completed at c=2048 and c=4096 auto with stable swap
- Eval schema (required+must_include union) was fixed with no headline conclusion changes

❌ **You MAY NOT say:**
- Production readiness
- Speedup achieved
- Broad 7B validation
- 14B support
- Long-context solved
- KV cache modified
- Weight-residency solved
- Agent/OpenClaw/SAR integration
- PRT speedup
- Universal superiority
- Token savings on small contexts
- qwen2.5:0.5b solves all exact/natural constraint tasks

---

## Known Limitations

- **No broad 7B eval:** Only a tiny 2-task canary at c=2048/4096 was run
- **Phase 27H-B/J correction:** The earlier WS-6144/c=8192 "cliff" (Phase 27H-C) was superseded. Clean forensic runs (Phase 27J) passed WS-512 through WS-6144 at c=8192. No memory/swap cliff found. Still not broad 7B validation.
- **No 14B:** Never tested
- **No KV cache modification:** SDI operates above the KV layer
- **No weight-residency solution yet:** PRT custom-op path parked
- **Model ceilings remain:** qwen2.5:0.5B has hard limits on natural constraint and exact tool-output tasks
- **Packet overhead:** Real on small contexts; exact-tool mode has high token cost
- **Current evals are local/limited:** Not public benchmark-level validation
- **PRT custom-op speed path:** Parked at complexity/benefit crossroads
- **Eval fixture schema bug:** Fixed in 27F; took 8 scenarios off critical list

---

## How to Reproduce

```bash
cd llama.cpp

# Run SDI eval harness
python3 examples/speculative/sdi_packet_runtime.py \
  --conversation fixtures/sdi_packet_eval/<SCENARIO>/conversation.txt \
  --pinned fixtures/sdi_packet_eval/<SCENARIO>/pinned_facts.json \
  --expected fixtures/sdi_packet_eval/<SCENARIO>/expected.json \
  --model qwen2.5:0.5b \
  --baseline auto \
  --backend ollama \
  --active-context-target 2048 \
  --out /tmp/sdi_out.txt \
  --meta /tmp/sdi_meta.json

# Run regression tests
python3 examples/speculative/test_sdi_packet_builder.py
```

---

## Recommended Next Directions

1. **Phase 27L** — Update public article/thread package with corrected bounded 7B result, then pause 7B probing
2. **Parked** — PRT v3 weight-residency design (until SDI context path stabilizes)
3. **Parked** — KV memory probe design (requires careful scoping)

---

## Phase 27G Verdicts

- **PASS_PHASE27G_FINAL_SUMMARY**
- **PASS_EVIDENCE_CONSOLIDATED**
- **PASS_CLAIM_BOUNDARIES_FINALIZED**
- **PASS_LIMITATIONS_DOCUMENTED**