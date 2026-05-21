# SDI Runtime v0.1.1: A Model-Residency Layer for CPU-Constrained Dense Inference

**Status:** Prototype | **Models tested:** qwen2.5:0.5B, 3B, 7B (tiny canary) | **Eval:** Local, not public benchmark

---

## Hook

Dense models aren't just compute-heavy on CPU. They're memory-residency problems.

When a 4B+ dense model runs on a 15GB RAM machine at c=4096, the KV cache alone can consume multiple gigabytes. At c=16384, the model weights + KV can exceed available RAM and trigger swap — which then poisons every subsequent inference run. That's not a performance issue. It's a survivability issue.

This is what Sub-Dense Inference (SDI) is trying to solve.

---

## The Problem Space

CPU inference is constrained by five compounding factors:

1. **RAM capacity** — Model weights + KV must fit. At c=16384 with qwen2.5:7B, they may not.
2. **Memory bandwidth** — Dense per-token weight streaming is bandwidth-heavy on CPU, not compute-bound.
3. **KV/context growth** — Long conversations grow KV linearly. No free lunch.
4. **Swap risk** — Unbounded context can trigger swap, poisoning the next inference call.
5. **Dense per-token work** — Unlike pruned or quantized sparse models, dense models do full compute at every token.

The goal: make dense models survive inside bounded RAM, not make them faster.

---

## What SDI Means

Sub-Dense Inference is a **CPU/RAM model-residency strategy**:

> Reduce how much context, KV, weight memory, and dense compute must be active per token while preserving correctness through fallback and verification.

SDI is not:
- An agent framework
- A routing layer for agents
- A KV cache modification
- A weight-residency solution
- A speedup mechanism

SDI is: a **structured context selection layer** that decides what context to present to the model at inference time, without modifying model weights or llama.cpp internals.

The current prototype focuses on the **first layer**: active context pressure.

---

## What the Prototype Does

SDI Runtime v0.1.1 is a standalone system with four components:

### 1. Packet Builder
Builds structured context packets from long/noisy conversation history. Deterministic output. Includes safety checks for secrets, filler, and repetition. Emits structured fields: Summary, Open Loops, Constraints, Recent Conversation.

### 2. Memory Guard
Pre-run RAM/swap check. Detects stale llama/ollama processes. Returns conservative go/no-go. Blocks if swap > 1GB or available RAM < 3GB. Prevents the swap-death scenario.

### 3. Auto Policy
Five strategies: `recent_only`, `simple_summary`, `sdi_packet`, `no_packet`, `auto`. Policy is selected per-conversation based on length, noise, filler, and context size. `auto` uses per-scenario heuristic rules.

### 4. Exact-Tool Mode
Specialized routing for contexts where exact tool output (paths, commit hashes, numeric values) must be preserved exactly. Higher token cost but critical for correctness in tool-use scenarios.

### Eval Harness
Ollama-backed standalone eval harness. Runs conversations against qwen2.5:0.5B, 3B, or 7B. Scores responses against fixture criteria. Records swap delta before/after each run.

---

## What Was Tested

All testing is **local eval only**, not public benchmark-level validation.

### qwen2.5:0.5B — Full Eval
- **17 scenarios** including pinned facts, open loops, constraints, tool outputs, numeric recall, benchmark interpretation
- Auto policy: **0.950 average**, **7/8 win-tie** vs best fixed baseline after targeted policy gates
- Gap to best fixed: **0.013** (narrowed from 0.069 in Phase 26K)
- Swap: stable (0–12 MB delta) across all runs

### qwen2.5:3B — Edge-Case Probe
- Targeted Sc21/Sc25/Sc26 comparison
- **Sc25 result:** qwen2.5:3B scored 0.850 (auto and no_packet matched), vs qwen2.5:0.5B's 0.750 gap
- **Interpretation:** Sc25's 0.5B failure is a **model ceiling**, not a policy failure
- Auto policy validated at both 0.5B and 3B scale for Sc21/Sc26

### qwen2.5:7B — Tiny Canary
- 2 tasks (Sc21 research handoff, Sc23 benchmark interpretation)
- c=2048 and c=4096 auto only
- **8 runs completed, 0 abort criteria triggered, swap stable at 677 MB**
- Auto correctly selected `recent_only` for both tasks
- **No broad 7B validation claimed**

### Eval Schema Fix
- Phase 27F: Fixed a bug where fixture `required` field was silently ignored by `score_response()`
- Unioned `required` + `must_include` as critical check fields
- **No headline SDI v0.1.1 conclusion changes**

---

## What Worked

- **Packetization helps** long/noisy/pinned/open-loop/exact-tool contexts
- **Auto policy routing matters** — always-on packets are wrong, always-off is worse
- **Memory guard is effective** — swap stayed flat during eval runs
- **Context reduction is real** — packet builder produces structured summaries with verifiable factual preservation
- **3B closes a ceiling** — larger models close some gaps that look like policy failures at 0.5B

---

## What Did Not Work / Limitations

- **No speedup claim.** SDI is about memory survival, not latency.
- **No broad 7B validation.** Only a tiny 2-task canary at c=2048/4096 was run.
- **No 14B tested.**
- **No KV cache modification.** SDI operates above the KV layer.
- **No weight-residency solution yet.** PRT custom-op speed path was correct but not faster; parked.
- **Small contexts bloat.** Packet overhead can exceed savings on short inputs.
- **Exact-tool mode costs tokens.** High fidelity exactness requires more context budget.
- **Current evals are local/limited.** Not public benchmark-level validation.
- **Model ceilings remain.** qwen2.5:0.5B has hard limits on natural constraint and exact tool-output tasks.

---

## Why This Matters

This is not an agent framework. This is not routing-as-the-invention.

This is a **model-residency layer** for constrained local inference — the layer between "the model wants unbounded context" and "the machine has bounded RAM."

The key insight: for dense models on constrained hardware, the bottleneck is memory residency, not compute. If you can reduce active context pressure without losing the facts that matter, you can make dense models run where they otherwise couldn't.

That's what SDI is trying to be. The current prototype is real but narrow. It shows signal. It doesn't claim to be done.

---

## What Comes Next

If this line of work continues:

1. **Bounded 7B validation** — a few more scenarios at c=4096, strict guard, explicit approval
2. **KV/context memory probe** — map memory pressure curves for 7B at higher context lengths
3. **Stricter real-task eval** — more realistic scenarios, not fixture-based synthetic tasks
4. **Weight-residency / PRT v3** — only if SDI context path stabilizes and justifies the complexity

Or: this is the prototype. It ships as evidence of the approach, not as a product.

---

## Reproduce

```bash
cd llama.cpp

# Run a single eval scenario
python3 examples/speculative/sdi_packet_runtime.py \
  --conversation fixtures/sdi_packet_eval/21_research_handoff/conversation.txt \
  --pinned fixtures/sdi_packet_eval/21_research_handoff/pinned_facts.json \
  --expected fixtures/sdi_packet_eval/21_research_handoff/expected.json \
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

## Claims Boundary

**Allowed:** Standalone CPU/RAM prototype, structured context selection, memory guard, auto policy improvement on qwen2.5:0.5B, 0.950 avg/7/8 win-tie in local eval, 3B confirms 0.5B ceiling, tiny 7B canary with stable swap, eval schema fix with no conclusion changes.

**Forbidden:** Speedup, production readiness, broad 7B validation, 14B support, long-context solved, KV cache modified, weight-residency solved, agent/OpenClaw/SAR integration, PRT speedup, universal superiority, token savings on small contexts.