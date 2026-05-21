# SDI Runtime v0.1.1 — Technical Summary

## What This Is

SDI Runtime v0.1.1 is a prototype for reducing active context pressure during local CPU/RAM inference. It builds compact structured context packets from long or noisy conversation histories and uses an auto-policy to decide when packetization helps versus when it adds overhead. Built and tested on qwen2.5:0.5b via local Ollama.

## The Problem

Dense models on CPU face memory pressure from accumulating context. Active context grows with every conversation turn. When it exceeds available RAM, systems swap to disk — and inference grinds to a halt. This is especially acute for models not designed for CPU-only serving.

## The Approach

SDI Runtime constructs a compact [SDI PACKET] — a structured representation of pinned facts, open loops, state flags, and exact-tool fields — and uses an auto-policy to select between:

- **no_packet:** raw context, minimal overhead
- **recent_only:** last N lines, small context
- **simple_summary:** natural language summary of pinned facts
- **sdi_packet:** compact structured packet
- **auto:** policy selects the best mode for the question and context shape

The auto-policy is the core contribution. It routes based on question type, context shape, memory pressure, and content risk flags. Key routing rules:

- Long/noisy/benchmarked context → sdi_packet
- Simple factual recall (e.g., "verify the license") → recent_only/no_packet
- Hard constraint questions ("can I override?") → simple_summary for natural format
- Open-loop continuation ("resume where we left off") → recent_only for natural flow
- Multi-commit code review → simple_summary

## What Was Tested

Over 10+ phases, the system was evaluated on:

- Unit/integrity tests for packet builder
- Synthetic 5-scenario probe
- Hostile eval (long/filler/noisy context pressure)
- Auto-policy refinement across 8 realistic engineering scenarios
- Swap stability monitoring across all runs

On qwen2.5:0.5b (the only model tested), the auto-policy reached:
- **0.950 average score** on the 8-scenario eval
- **7/8 win/tie** against best per-scenario fixed baselines
- **0 MB swap delta** across all runs (memory stable)

## What Improved (v0.1 → v0.1.1)

Three targeted policy gates were added in v0.1.1:

1. **Factual recall gate:** Simple "verify/confirm X" questions where X is already in pinned facts now route to `recent_only` instead of `sdi_packet`, improving score from 0.850 to 1.000.
2. **Hard-constraint gate:** Questions about overriding hard constraints now route to `simple_summary` (natural format) instead of exact-tool mode, improving score from 0.600 to 0.750.
3. **Continuation gate:** "Resume" questions route to `recent_only` for natural conversational flow.

Overall: auto average improved from 0.894 → 0.950; gap to best fixed baseline narrowed from 0.069 → 0.013.

## Limitations

**This is not production software.** Key limitations:

- All results are on qwen2.5:0.5b. No 7B/14B validation.
- Packet overhead on small contexts is real — targeted gates mitigate but don't eliminate it.
- One failure case (Scenario 25) remains below baseline because qwen2.5:0.5b cannot produce natural-format hard-constraint answers. This is a model ceiling issue, not a policy failure.
- No KV cache modification. This is a prompt-composition technique only.
- No weight-residency mechanism yet. PRT custom-op speed path is parked due to an unresolved memory corruption bug.

## Next Steps

Recommended next work:
- Expand the fixture set with more realistic engineering scenarios
- Investigate prompt variants for natural-format hard-constraint answers
- Optional (with approval): qwen2.5:3b comparison on hardest remaining tasks

PRT v3 native model-residency and KV/cache pressure work are parked until a cleaner design path is justified.

---

*SDI Runtime v0.1.1 — claim-bounded prototype, not production software*