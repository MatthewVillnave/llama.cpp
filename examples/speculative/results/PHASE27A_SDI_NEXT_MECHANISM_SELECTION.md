# Phase 27A: SDI Next Mechanism Selection

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`da1fb954c`

## C. Checkpoint Verified
✅ `SDI_PHASE26W_RUNTIME_V0_1_1_CHECKPOINT` exists at `aa6ebd460`
✅ Phase 26X docs exist and committed

---

## D. Phase 26 Proved

| What worked | Evidence |
|-------------|----------|
| Deterministic packet builder | 14/14 unit tests, Phase 26I |
| Memory guard | Active in all runs, RAM monitoring working |
| Standalone runtime | Fully repo-agnostic, no llama.cpp internals touched |
| Auto-policy with targeted gates | Phase 26V: auto avg 0.894→0.950, win/tie 5/8→7/8 |
| Hostile eval methodology | Consistent across 26I–26V |
| Swap stability on 0.5B | 0 MB delta across all tested runs |

**Core valid SDI claim:** Context-packet SDI improves auto-policy on qwen2.5:0.5b for long/noisy/pinned-fact/open-loop contexts.

---

## E. Phase 26 Did NOT Prove

| What was NOT proven | Why |
|---------------------|-----|
| 7B long-context validation | Only tested on qwen2.5:0.5b |
| KV cache reduction | No KV internals touched |
| Speedup | No timing benchmarks run |
| Weight-residency solution | PRT speed path parked due to ggml_map_custom2 corruption |
| Production readiness | Prototype only |
| Universal superiority | 1/8 scenario still loses to no_packet |
| Token savings on small contexts | Packet overhead is real on small context |

**Key lesson:** Context-packet SDI is a valid first layer. It attacks active prompt/context pressure. It does NOT yet solve weight bandwidth or KV cache internals.

---

## F. Option Ranking

| Option | Solves CPU/RAM goal? | Risk | Time-to-signal | HW risk | Notes |
|--------|---------------------|------|----------------|---------|-------|
| **qwen2.5:3b comparison** | Medium | Low | Fast | None (if approved) | Validates if model ceiling closes Sc25; ~1GB model |
| **7B safe-context validation** | High | Medium | Medium | Medium | Closest to original CPU/RAM goal; tight RAM margin (~12.9GB avail) |
| **KV/Context memory probe** | High | Low | Medium | None | Designs external KV pressure measurement; aligns with reviewer feedback |
| **Natural-constraint refinement** | Low | Very Low | Fast | None | Likely diminishing returns; Sc25 is model ceiling |
| **Public/internal release package** | Low | Very Low | Fast | None | Communicates progress; forces claim discipline |
| **PRT v3 / Weight-residency design** | Very High | High | Long | None | High complexity; not implementation-ready yet |

### Detailed reasoning:

**qwen2.5:3b** (Recommended first step):
- Lowest risk next step
- Matt has to explicitly approve the pull
- Validates whether the Sc25 failure (0.750 vs 0.850) is a qwen2.5:0.5b ceiling issue
- If 3B closes the gap → confirms policy is correct, model is the limiter
- If 3B doesn't close the gap → reveals deeper problem with the constraint routing
- Time-to-signal: ~30 min eval once model is pulled
- No hardware risk if pull is approved

**7B safe-context validation** (Recommended second step, conditional):
- Only if 3B comparison is informative AND explicit approval for 7B scope
- Hardware risk is real: 12.9GB available on a 15GB machine, 7B needs ~10GB+ weights + KV
- Swap risk exists if context grows large
- Must have strict swap guard and tiny scope (only test 1-2 scenarios, short context limits)
- This is the honest test of "run models inside CPU/RAM limits"

**KV/Context memory probe** (Worth designing now):
- Does not require model pulls or hardware risk
- Designs the measurement before modifying KV internals
- Aligns with likely reviewer feedback: "how do you measure KV pressure?"
- Output is a design doc + eval plan, not a new implementation
- Can run in parallel with 3B comparison

**Natural-constraint refinement** (Deprioritize):
- Sc25 is likely a model ceiling issue
- Improving the prompt format is worth one more attempt, but diminishing returns
- Low priority given 3B would settle the question faster

**Public/internal release package** (Worth doing eventually):
- SDI v0.1.1 is clean enough for a structured internal post
- Forces claim discipline
- Does not need to be the next technical step

**PRT v3 / Weight-residency** (Park):
- High complexity, not implementation-ready
- PRT phase 10E failure showed ggml_map_custom2 is not stable
- Need a clean native backend design before resuming
- Park until SDI context work is fully validated

---

## G. Recommended Next Phase

**Recommended:** Phase 27B-R — qwen2.5:3b comparison gate on Sc21/Sc25/Sc26 only.

**Conditions:**
- Matt must explicitly approve the `ollama pull qwen2.5:3b`
- Scope is limited to the 3 hardest scenarios from v0.1.1 eval
- Success criteria: does 3B close the Sc25 gap (0.750 → 0.850)?
- If yes: policy is validated, model is the limiter
- If no: reveals a policy flaw that targeted gates didn't catch

**Parallel:** KV/Context memory probe design (no model pull, no hardware risk) can be started independently as Phase 27B-K.

**Then (conditional on 3B results and explicit approval):** Phase 27C — 7B safe-context validation with strict swap guard and tiny scope.

---

## H. What NOT to Do Next

❌ Do NOT run 7B validation without explicit approval and a swap guard
❌ Do NOT pull qwen2.5:3b without explicit approval
❌ Do NOT claim speedup or production readiness
❌ Do NOT modify KV internals without a probe design first
❌ Do NOT resume PRT speed path work until ggml_map_custom2 is resolved
❌ Do NOT run broad new evals

---

## I. Models/sidecars/f32 refs staged?
No. qwen2.5:0.5b via Ollama HTTP API. No model files, sidecars, f32 refs, or binaries staged.

## J. Secrets detected?
No.

## K. Tags touched?
No. Existing tags not modified. No new tag created.

---

## Verdict

**PASS_PHASE27A_NEXT_MECHANISM_SELECTION**
**RECOMMEND_3B_COMPARISON** (first step, requires explicit approval)
**RECOMMEND_7B_SAFE_CONTEXT_VALIDATION** (conditional second step)
**RECOMMEND_KV_MEMORY_PROBE** (parallel design work)
**PARK_PRT_V3**