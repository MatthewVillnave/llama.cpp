# SDI Runtime v0.1.1 — X Thread Draft

**8 posts. No hype. Technical. Honest about limitations.**

---

**Post 1/8:**
Dense models on CPU aren't just compute-heavy — they're memory-residency problems.

When qwen2.5:7B runs at c=16384 on a 15GB RAM machine, KV + weights can exceed available memory and trigger swap, which poisons every subsequent inference call.

That's not a performance issue. It's a survivability issue.

---

**Post 2/8:**
So we built a context-selection layer called SDI: Sub-Dense Inference.

It doesn't make models faster. It doesn't modify KV cache or model weights. It decides what context to present to the model at inference time, using structured packets + a memory guard + an auto policy.

Goal: make dense models survive bounded RAM without losing the facts that matter.

---

**Post 3/8:**
SDI Runtime v0.1.1 components:
- Packet builder: structured context summaries from long/noisy conversations
- Memory guard: pre-run RAM/swap check, blocks if unsafe
- Auto policy: selects recent_only / simple_summary / sdi_packet / no_packet per conversation
- Exact-tool mode: preserves exact paths, hashes, numeric values
- Standalone Ollama-backed eval harness

No llama.cpp internals modified. No model weights changed.

---

**Post 4/8:**
Results (local eval, not public benchmark):

qwen2.5:0.5B auto policy: 0.950 avg, 7/8 win-tie vs best fixed baseline after targeted gates. Swap stable across all runs.

qwen2.5:3B edge-case: confirmed a scenario that looked like a policy failure at 0.5B was actually a model ceiling — auto policy correct at both scales.

---

**Post 5/8:**
qwen2.5:7B tiny canary: 8 runs, 2 tasks at c=2048 and c=4096 auto-only. Swap stable. Auto correctly selected recent_only. No aborts triggered.

This is not broad 7B validation. It's one tiny canary under strict memory guard.

---

**Post 6/8:**
What didn't work:
- Always-on context packets are wrong for small inputs (overhead exceeds savings)
- PRT custom-op speed path: correct but not faster — parked
- qwen2.5:0.5B has hard ceilings on natural constraint tasks and exact tool outputs
- No KV cache modification, no weight-residency solution yet
- Current evals are fixture-based, not public-benchmark-level

---

**Post 7/8:**
The key insight: for dense models on constrained hardware, the bottleneck is memory residency, not compute.

If you reduce active context pressure without losing the facts that matter, dense models can run where they otherwise couldn't.

SDI is trying to be that layer. Current prototype is real but narrow.

---

**Post 8/8:**
What would you test next for a system like this:

A) Bounded 7B validation at c=4096 (with strict memory guard)
B) KV/context memory pressure mapping across context lengths
C) Structured real-world task eval instead of fixtures

Genuine question — we have signal but not a roadmap.

---

**Mandatory disclaimers (must stay in thread):**
- Not a speedup claim
- Not production-ready
- Not broad 7B validation
- Context/KV pressure first, weight residency later
- PRT speed path parked after correctness validation but no speed win