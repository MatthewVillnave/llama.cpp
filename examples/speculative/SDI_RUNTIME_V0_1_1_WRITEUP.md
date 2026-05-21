# SDI Runtime v0.1.1: Context Compression as CPU/RAM Model-Residency Strategy

**Tag:** `SDI_PHASE26W_RUNTIME_V0_1_1_CHECKPOINT`
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**HEAD:** `aa6ebd460`
**Date:** 2026-05-21

---

## 1. Summary

SDI Runtime v0.1.1 is a standalone prototype that reduces active context pressure for local CPU/RAM-constrained inference. It constructs compact structured context packets from long or noisy conversation histories and uses an auto-policy to decide when packetization is helpful versus when it adds unnecessary overhead. All evaluation was performed on qwen2.5:0.5b via local Ollama; no claims are made about 7B/14B behavior.

---

## 2. Problem Statement

Dense model inference on CPU is constrained by:

- **RAM capacity:** Models like Llama/Gemma/Qwen were not designed for CPU-only serving; their KV/context growth is aggressive.
- **Memory bandwidth:** Dense per-token compute requires weights to be streamed from RAM, not cache.
- **KV/context growth:** Active context accumulates linearly with conversation length; long histories dominate memory.
- **Swap risk:** When active context exceeds available RAM, the system swaps to disk, destroying inference performance.
- **Context bloat:** Verbose conversation logs, repeated filler, benchmark tables, and tool outputs all inflate active context without adding proportional value.

This project addresses **context/KV pressure first**. It does not yet claim to solve weight bandwidth — that requires PRT v3 or a native model-residency mechanism, both parked pending further design.

---

## 3. What SDI Is

**Sub-Dense Inference** is a CPU/RAM survival strategy for dense models: reduce how much context, KV, weight memory, and dense compute must be active per token while preserving correctness through fallback and verification.

SDI Runtime is the implementation vehicle: a prompt-composition layer that builds structured context packets and selects between baseline and packetized prompts using an auto-policy. It does not modify KV cache internals, model weights, or llama.cpp compute paths.

---

## 4. What SDI Runtime v0.1.1 Includes

| File | Role |
|------|------|
| `sdi_packet_builder.py` | Constructs compact [SDI PACKET] with pinned facts, open loops, state flags, exact-tool fields |
| `sdi_memory_guard.py` | Estimates RAM pressure and recommends tier |
| `sdi_packet_runtime.py` | Runs baselines (no_packet, recent_only, simple_summary, sdi_packet) and auto-policy |
| `run_sdi_demo_eval.py` | Automated eval runner with scoring |
| `fixtures/sdi_packet_eval/` | 8 realistic engineering scenarios for eval |
| `SDI_RUNTIME_V0_1_1_CHECKPOINT.md` | Full v0.1.1 checkpoint with claim boundaries |

---

## 5. Evaluation History

| Phase | Result |
|-------|--------|
| 26I | Packet builder deterministic — 14/14 unit tests passing |
| 26J | 5/5 synthetic scenarios, 43/43 integrity checks |
| 26K-R2 | qwen2.5:0.5b packet probe — avg 0.83/1.0, swap stable |
| 26L | Measurement baseline + hostile eval plan |
| 26M | Standalone runtime + memory guard built; mixed hostile eval: SDI avg 0.535 |
| 26N | Packet v0.2 refinement: avg 0.770, 4/5 wins/ties |
| 26O | Auto policy v0.2: avg 0.764, 9/10 win/tie |
| 26T | **v0.1 checkpoint frozen** |
| 26U | Expanded eval (8 scenarios): auto avg 0.894, 5/8 win/tie |
| 26V | Targeted policy gates: auto avg **0.950**, 7/8 win/tie, gap to best fixed: 0.013 |
| 26W | **v0.1.1 checkpoint frozen** |

---

## 6. Key Technical Lesson

Always-on packets are not ideal. The value is **policy selection**:

- **Packetize** long/noisy/high-risk context (benchmark tables, tool outputs, pinned facts)
- **Avoid packet overhead** on tiny/self-contained prompts (simple factual recall)
- **Use simple_summary** for multi-commit code review and broad interpretation questions
- **Use exact-tool mode** only when exact path/hash/number/status matters
- **Use recent_only/no_packet** for simple factual recall when context is already small
- **Use compact style** for token-sensitive contexts

The auto-policy is what makes this useful — not the packet itself.

---

## 7. Allowed Claims

✅ **You MAY claim:**
- SDI Runtime v0.1.1 is a standalone CPU/RAM model-residency prototype.
- It improves auto-policy behavior on qwen2.5:0.5b local Ollama evals.
- It reached 0.950 average score on the expanded 8-scenario real-task eval.
- It achieved 7/8 win/tie against best fixed baselines in that eval.
- Targeted policy gates reduced eager packet selection.
- Swap remained stable (0 MB delta) in all tested qwen2.5:0.5b runs.
- It is useful for long/noisy/pinned-fact/open-loop/exact-tool contexts.

---

## 8. Forbidden Claims

❌ **Do NOT claim:**
- Production readiness
- Token speedup or latency improvement
- 7B or 14B validation
- KV cache modification
- Weight-residency solution
- Agent, OpenClaw, or Smart Agent Router integration
- PRT speedup (PRT active generation claims are forbidden until `ggml_map_custom2` memory corruption is resolved)
- Universal superiority
- Token savings on small contexts (packet overhead is real there)
- qwen2.5:0.5b as a universal proxy for larger models
- Scenario 25 "solved" — the 0.750 vs 0.850 gap remains a model ceiling issue

---

## 9. Known Limitations

1. **Scenario 25 model ceiling:** qwen2.5:0.5b produces structured constraint output ("Hard Constraint: Never...") rather than natural conversational answers ("No — constraint. Recommended action: [safe action]."). The policy routes correctly, but the model cannot produce the expected answer format.

2. **Packet overhead on small contexts:** SDI packet adds overhead on contexts <500 tokens. Targeted gates route small/factual questions away from packet, but overhead remains for contexts that legitimately need the packet.

3. **Exact-tool mode token cost:** Exact-tool routing has high token overhead on tiny contexts. Should only fire when truly needed.

4. **No 7B long-context validation:** All evals are on qwen2.5:0.5b. Claims do not transfer to 7B/14B models.

5. **No KV cache modification:** This is a prompt-composition technique. It does not modify KV tensor internals.

6. **No weight-residency mechanism yet:** PRT v3 (native model-residency) is parked. This work is about prompt selection, not compute scheduling.

7. **PRT speed path parked:** `ggml_map_custom2` memory corruption bug when sidecar + custom op are active together. End-to-end speedup claims are forbidden until resolved.

---

## 10. How to Reproduce

```bash
cd /home/matthew-villnave/llama.cpp
python3 examples/speculative/run_sdi_demo_eval.py \
  --model qwen2.5:0.5b \
  --out-dir /tmp/sdi_demo_eval
```

Prerequisites:
- Ollama must be running locally
- `qwen2.5:0.5b` must be installed (`ollama pull qwen2.5:0.5b`)
- Generated outputs are written to `/tmp`, not staged to repo

---

## 11. Next Research Directions

**Recommended (no model pull):**
- Formal writeup and external publication of claim-bounded results
- Expand fixture set with more realistic engineering scenarios
- Investigate natural-constraint-answer prompt variants

**Optional (requires explicit approval, no auto-pull):**
- qwen2.5:3b comparison on Sc21/Sc25/Sc26 only, to determine if larger model closes Scenario 25 ceiling gap

**Later (after 7B safe-context validation):**
- 7B safe-context validation with swap guard
- KV/cache pressure work
- PRT v3 native model-residency design

---

*SDI Runtime v0.1.1 — `aa6ebd460` — experimental/prt-phase19a-alt-sidecar-backed*