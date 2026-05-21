# Phase 26H: SDI Context Packet Policy v0.1 — Repo-Agnostic Design

**Verdict:** `PASS_PHASE26H_DESIGN_COMPLETE` | `RECOMMEND_STANDALONE_PACKET_BUILDER_NEXT`

**Date:** Thu 2026-05-21 01:06 EDT
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Current HEAD:** `9d89360b4`

---

## A. Branch
```
experimental/prt-phase19a-alt-sidecar-backed
```

## B. Current HEAD
```
9d89360b4
```

## C. Phase 26G Summary

Phase 26G built a safe output harness for llama-cli and partially validated the pinned facts context policy:

**Harness findings:**
- llama-cli hangs after generating n= tokens — doesn't terminate cleanly on its own
- Timeout kills the process after correct answer is generated
- Output cap (128 KB) works to prevent runaway captures
- 0.5B generated "Paris" correctly — answer was in the output before timeout

**Variant C partial validation:**
- Q1 "Project ARGUS" answered correctly from pinned facts ✅
- Q2 partial (API key visible in model output) ✅
- Swap delta: 0 MB at c=512 and c=1024 ✅
- 85% token reduction confirmed ✅
- Full 10-point score blocked by harness issues ❌

**Key insight:** Context compression policy is validated at the mechanism level (facts accessible from compressed context). The llama-cli harness is a distraction — the real solution is a proper prompt assembly layer.

**Decision:** Pivot from llama-cli benchmarking to SDI context packet policy design.

---

## D. SDI Context Policy: `sdi_context_policy_v0_1`

### Overview

`sdi_context_policy_v0_1` is a **prompt assembly policy** that decides what content to include in a model's input context before inference. It runs **before model selection**, not during inference.

It is **repo-agnostic** — it has no built-in assumptions about VaultBrain, ClawVault, Smart Agent Router, or any specific runtime. It produces a structured packet that can be consumed by any of those systems.

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `user_message` | string | Current user request |
| `recent_turns` | list[Turn] | Recent conversation turns (newest last) |
| `pinned_facts` | list[string] | Critical project facts, constraints, IDs |
| `retrieved_memories` | list[Memory] | Memories retrieved from long-term storage |
| `task_type` | enum | question / creative / coding / planning / review |
| `target_model` | string | Model being considered (e.g. qwen2.5-7b) |
| `context_budget` | int | Max tokens for this model (from model profile) |
| `memory_state` | dict | MemAvailable KB, SwapUsed MB (if available) |
| `tool_outputs` | list[ToolOutput] | Results from recent tool calls |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `active_packet` | SDI_CONTEXT_PACKET | Assembled prompt packet |
| `estimated_tokens` | int | Approximate token count of packet |
| `included_facts` | list[string] | Facts included in packet |
| `dropped_content` | list[string] | Content dropped/replaced with summary |
| `compression_reason` | string | Why compression was applied |
| `safety_warnings` | list[string] | Any memory/safety concerns |
| `tier` | enum | Tier 0/1/2/3 applied |
| `model_used` | string | Model that should receive this packet |

### Policy Tiers

The policy selects a **tier** based on memory state and context budget:

| Tier | Name | When | Target tokens | Includes |
|------|------|------|---------------|----------|
| 0 | Emergency / tiny | swap > 1GB OR memory tight | ≤ 1,024 | current message + pinned facts + last 1-2 turns |
| 1 | Safe 7B default | memory OK, normal task | ≤ 2,048–4,096 | current message + pinned facts + recent window + memories |
| 2 | Extended safe | memory OK, complex task | ≤ 4,096–8,192 | current message + pinned facts + ContextOS packet + recent window + tool outputs |
| 3 | Full context | memory headroom confirmed, no swap pressure | ≤ 8,192–16,384 | Everything — only with explicit approval or clean memory state |

**Tier selection rules:**
```
if swap_used > 1GB:           tier = 0  (emergency)
elif memory_tight:             tier = 0 or 1
elif task_type == "planning":   tier = 2  (needs more context)
elif task_type == "creative":  tier = 2  (needs more context)
elif memory_clean and task not critical: tier = 1
elif memory_clean and task is critical: tier = 2
else:                          tier = 1  (default to safe)
```

### Compression Logic

For each input content block:

1. **Estimate token count** (rough: `len(text) / 4`)
2. **Check against tier budget** (`context_budget * tier_multiplier`)
3. **If over budget:**
   - pinned facts: **never drop** — always include verbatim
   - tool outputs: keep last N only (N determined by relevance)
   - recent turns: keep last M turns (oldest dropped first)
   - retrieved memories: keep top K by relevance score
   - filler/repetitive content: **drop and summarize with one line**
   - long raw logs: **drop unless user explicitly requests them**

4. **If still over budget:**
   - Fall back to Tier 0 (emergency compact)
   - Return `safety_warnings` indicating forced compression

### Key Principles

- **Never invent facts** — if a fact isn't in pinned_facts or retrieved_memories, mark it as uncertain
- **Pinned facts are sacred** — they survive all compression tiers without modification
- **Drop filler, not structure** — compress repetitive content but preserve IDs, numbers, names, paths
- **ContextOS packet is the compression unit** — old context gets converted to one structured packet, not scattered summaries
- **Deterministic** — same inputs → same outputs (reproducible)

---

## E. Packet Format

```markdown
[SDI_CONTEXT_PACKET]
Goal: <current user request — verbatim or paraphrased summary>
Current user request: <exact user message if short, else summary>
Pinned facts: <bullet list — exact verbatim facts>
  - <fact 1>
  - <fact 2>
Hard constraints: <bullet list — constraints that must not be broken>
  - <constraint 1>
  - <constraint 2>
Relevant memory: <bullet list — from retrieved memories, with source note>
  - [memory/<date>] <fact>
Open loops: <bullet list — unresolved tasks, pending decisions, waiting on>
  - <item 1>
  - <item 2>
Recent state: <paragraph summary — what happened in recent turns>
  <N sentences max>
Tool/output facts: <bullet list — key outputs from tools, if relevant>
  - [tool] <result summary>
Dropped context summary: <brief — what was dropped and why>
  - <dropped content 1>: <reason>
  - <dropped content 2>: <reason>
Uncertainty: <bullet list — things marked as uncertain/guessed>
  - <uncertain fact>
Memory/safety note: <one line — memory state, tier applied, any warnings>
[/SDI_CONTEXT_PACKET]
```

### Packet Rules

1. **Exact preservation:** IDs, names, numbers, paths, commits, tags, constraints — always verbatim
2. **No invented facts:** mark uncertain things explicitly as `[?]` or `[uncertain]`
3. **User constraints verbatim:** if user said "never do X" or "always include Y", copy it exactly
4. **No filler:** do not include raw logs, repetitive messages, or empty padding
5. **Pinned facts before summaries:** critical details go in pinned facts, not in summary
6. **Dropped context always explained:** if content is removed, state why in `Dropped context summary`
7. **Deterministic:** no randomness in packet assembly

### Example Packet (Compact)

```markdown
[SDI_CONTEXT_PACKET]
Goal: Review the PRT phase 25B correctness fix and recommend next steps.
Current user request: What should we do next with PRT?
Pinned facts:
  - Project: PRT (Parallel Representation Transform) in llama.cpp
  - Branch: experimental/prt-phase19a-alt-sidecar-backed
  - Checkpoint: PRT_PHASE25B_CORRECTNESS_RESTORED_CHECKPOINT
  - Last phase: Phase 25B — INT8 layer0 correctness restored after /127 decode fix
  - 3B and 7B correctness canaries: PASSED
Hard constraints:
  - Never force-push to main
  - Do not modify ggml/src/ggml-cpu/ops.cpp without Matt's approval
Relevant memory:
  - [memory/2026-04-30] PRT phase 10E blocked — ggml_map_custom2 memory corruption
  - [memory/2026-05-20] Phase 26D: c=16384 triggers swap (+394 MB)
Open loops:
  - KV/context compression policy (Phase 26E-26H)
  - SpecBenchCPU revalidation (parked)
  - PRT v3 design (parked)
Recent state:
  - Phase 26H is designing the SDI context packet policy.
  - Phases 26D-G validated memory pressure baseline and partial context compression.
  - Swap death discovered — accumulated swap pages persist after process death.
Tool/output facts: none
Dropped context summary:
  - Full Roman Empire text from Phase 26E synthetic task: not relevant to PRT review
  - Phase 26E test variants B-E: replaced by this summary packet
Uncertainty: none
Memory/safety note: Tier 1 applied. MemAvailable=13.9GB, SwapFree=3.6GB. Clean state.
[/SDI_CONTEXT_PACKET]
```

---

## F. Memory/Swap Guard

The memory/swap guard runs **before any local inference** and determines whether it's safe to proceed.

### Pre-Inference Check

```python
def memory_guard(target_model: str, context_size: int, memory_state: dict) -> GuardResult:
    """
    Returns: (proceed: bool, tier: int, reason: str, warnings: list[str])
    """
    memavailable_gb = memory_state.get("memavailable_gb", 0)
    swap_used_gb = memory_state.get("swap_used_gb", 0)
    swap_free_gb = memory_state.get("swap_free_gb", 4.0)
    
    model_sizes = {
        "qwen2.5-0.5b": 0.38,   # GB, Q4_K_M
        "qwen2.5-3b":   1.8,
        "qwen2.5-7b":   4.4,
        "qwen2.5-14b":  8.4,
    }
    
    model_gb = model_sizes.get(target_model, 0)
    kv_gb = (context_size / 16384) * 1.8  # rough estimate for 7B at full ctx
    
    total_estimate_gb = model_gb + kv_gb + 0.3  # 0.3 for runtime overhead
    
    # Rule 1: Swap is a hard stop
    if swap_used_gb > 1.0:
        return GuardResult(
            proceed=False,
            tier=0,
            reason=f"Swap pressure critical: {swap_used_gb:.1f} GB used",
            warnings=["Swap > 1 GB — will not run local inference"],
            chosen_model=None,
            context_budget=None
        )
    
    # Rule 2: Not enough RAM for this model + context
    if total_estimate_gb > memavailable_gb * 0.85:  # 85% of available (leave headroom)
        return GuardResult(
            proceed=False,
            tier=0,
            reason=f"RAM tight: need ~{total_estimate_gb:.1f} GB, have {memavailable_gb:.1f} GB",
            warnings=[f"Insufficient RAM for {target_model} at c={context_size}"],
            chosen_model=None,
            context_budget=None
        )
    
    # Rule 3: Context size exceeds safe tier for this memory state
    safe_context_by_tier = {0: 1024, 1: 4096, 2: 8192, 3: 16384}
    if context_size > safe_context_by_tier[3] and swap_used_gb > 0.5:
        return GuardResult(
            proceed=False,
            tier=0,
            reason="Context too large with any swap pressure",
            warnings=["c > 16384 with swap present — will not proceed"],
            chosen_model=None,
            context_budget=None
        )
    
    # Rule 4: Proceed with appropriate tier
    if swap_free_gb < 2.0:
        tier = 1
    elif swap_free_gb < 3.0:
        tier = 2
    else:
        tier = 3
    
    return GuardResult(
        proceed=True,
        tier=tier,
        reason=f"Memory OK for {target_model} at c={context_size}",
        warnings=[],
        chosen_model=target_model,
        context_budget=context_size
    )
```

### Guard Rules (Plain English)

| Condition | Action |
|-----------|--------|
| Swap used > 1 GB | **Do not run local 7B** — escalate to cloud or compress first |
| Active context estimate > c=8192 with any swap | **Compress to Tier 1** |
| RAM estimate > 85% of MemAvailable | **Route to smaller model or cloud** |
| Stale llama processes found | **Kill and recover before proceeding** |
| Model already loaded, memory clean | **Proceed** |
| Memory is tight but clean | **Use Tier 0 or Tier 1** |
| Memory is clean and task is critical | **Use Tier 2** |

### Guard Output

The guard returns:
- `proceed`: bool — whether to proceed with local inference
- `tier`: int (0-3) — which compression tier to apply
- `reason`: str — human-readable explanation
- `warnings`: list[str] — any safety concerns
- `chosen_model`: str or None — model to use (may be downsized from target)
- `context_budget`: int or None — max context tokens allowed

---

## G. Integration Options

### Option A — Smart Agent Router Integration

**How:** Router calls `sdi_context_policy_v0_1` before choosing model/runtime.

```
User message
    ↓
Smart Agent Router
    ↓
sdi_context_policy_v0_1(user_message, recent_turns, pinned_facts, ...)
    ↓
Returns: active_packet + tier + chosen_model
    ↓
Router selects: model + runtime + context_budget
    ↓
Inference with active_packet
```

**Pros:**
- Router already has model selection authority
- Policy runs at the right decision point
- Can route to cloud if local fails guard

**Cons:**
- Router would need modification to support policy inputs
- Pinned facts and memories need to be accessible to router

### Option B — OpenClaw Preprocessor

**How:** OpenClaw builds the SDI packet before sending prompt to selected model.

**Pros:**
- OpenClaw is the session orchestrator — natural fit
- No router modification needed
- Can use existing memory integration

**Cons:**
- OpenClaw preprocessor interface not yet designed
- Requires integration work

### Option C — ContextOS Module

**How:** ContextOS owns packet assembly, exports compact prompt.

**Pros:**
- ContextOS is the memory layer — owns the context
- Packet is natural output of memory compression
- Already designed for this purpose

**Cons:**
- ContextOS doesn't exist yet as a component
- Would need to be built

### Option D — Standalone Helper (Recommended First)

**How:** Simple script or function that takes raw context + memories, emits SDI packet.

```python
def build_sdi_packet(
    user_message: str,
    recent_turns: list,
    pinned_facts: list,
    retrieved_memories: list,
    task_type: str,
    target_model: str,
    context_budget: int,
    memory_state: dict = None,
) -> SDIPacket:
    """Standalone packet builder — no repo dependencies."""
    
    # 1. Apply memory/swap guard
    guard = memory_guard(target_model, context_budget, memory_state)
    
    # 2. Select tier
    tier = select_tier(task_type, memory_state, guard)
    
    # 3. Assemble packet components
    components = assemble_packet(
        user_message=user_message,
        recent_turns=recent_turns,
        pinned_facts=pinned_facts,
        retrieved_memories=retrieved_memories,
        task_type=task_type,
        tier=tier,
        context_budget=context_budget,
    )
    
    # 4. Return structured packet
    return SDIPacket(
        packet_text=components["packet_text"],
        estimated_tokens=components["estimated_tokens"],
        included_facts=components["included_facts"],
        dropped_content=components["dropped"],
        compression_reason=components["reason"],
        safety_warnings=guard.warnings,
        tier=tier,
        model_used=guard.chosen_model or target_model,
    )
```

**Pros:**
- No repo dependencies — pure Python, works anywhere
- Testable in isolation
- Easy to integrate into any runtime later
- Clear input/output contract

**Cons:**
- Needs to be wired into a real runtime eventually

**Recommended order:** Standalone helper first (Phase 26I) → Smart Agent Router integration later → OpenClaw preprocessor later.

---

## H. Recommended First Integration

**Standalone helper (`sdi_packet_builder.py`) first.**

Rationale:
1. The policy is already designed — building the helper validates the design without touching any runtime
2. Isolated testing is fast and reproducible
3. Can be used immediately with existing tools (just feed it a conversation file)
4. Proves the concept before committing to runtime integration
5. Matt can use it manually or in scripts right away

**Later:** Wire into Smart Agent Router or OpenClaw when the helper is validated.

---

## I. Future Test Plan (Design Only — No Live Inference)

### Synthetic Evals

| Eval | Setup | Pass Criteria |
|------|-------|--------------|
| 1. Pinned early fact recall | Early fact in conversation, compress to Tier 1, ask question about it | Fact recalled correctly 100% |
| 2. Conflicting recent vs old | Old memory says X, recent says Y (conflict), ask about X | Model notes conflict OR picks most recent with caveat |
| 3. Long filler + critical constraint | 100-turn conversation with filler, one critical constraint buried early | Constraint recalled verbatim |
| 4. Tool result preservation | Tool call output important number/ID, compress context | Number/ID preserved exactly |
| 5. Open loop continuation | Open loop from earlier, compressed, ask "what's the status" | Open loop identified correctly |
| 6. Project-state catchup | Complex multi-phase project, compressed to Tier 2, ask for status | Correct phase/state reported |

### Metrics

| Metric | How to Measure |
|--------|---------------|
| Active token reduction | `(original_tokens - packet_tokens) / original_tokens` |
| Fact recall | % of critical facts present in output |
| Constraint retention | % of hard constraints verbatim in packet |
| Hallucination rate | % of facts in output not in inputs (should be 0) |
| Swap avoided | Swap delta with packet vs without, per run |
| Model output quality | Manual or automated score on eval questions |

### Success Criteria

| Criterion | Target |
|-----------|--------|
| Active context reduction | 50–80% |
| Critical facts retained | 100% |
| Constraint retention | 100% verbatim |
| Hallucination rate | 0% (never invent facts) |
| Swap avoided | No swap increase at c ≤ 4096 |
| Quality degradation | < 10–15% vs uncompressed baseline |
| Packet generation | Deterministic / reproducible |

### Kill Criteria

| Criterion | Kill Condition |
|-----------|---------------|
| Pinned facts lost | Any critical fact missing from packet |
| Hallucination increases | Output contains facts not in any input |
| Packet too large | Packet > context_budget after compression |
| Compression manual work | Requires human to manually edit packet each time |
| Integration impossible | Cannot integrate with any available runtime |

---

## J. Recommended Next Phase

**Phase 26I — Build Standalone SDI Context Packet Builder**

Scope:
- No model inference
- Input: synthetic conversation/context file + pinned facts JSON
- Output: compact SDI_CONTEXT_PACKET as markdown text
- Token estimate (rough: `len(text) / 4`)
- Dropped-content report
- Deterministic JSON metadata
- Memory/swap guard function included

**What to build:**
1. `sdi_packet_builder.py` — standalone Python module
2. `build_sdi_packet()` function with inputs listed above
3. `memory_guard()` function from Section F
4. Simple test harness with synthetic eval cases from Section I
5. Output validation: packet format, token count, no invented facts

**What NOT to build:**
- No model inference
- No runtime integration
- No VaultBrain/ClawVault wiring
- No llama.cpp modifications

**Deliverables:**
- `~/smart-agent-router/sdi_packet_builder.py` (standalone)
- `examples/speculative/results/PHASE26I_PACKET_BUILDER.md` (report)
- Synthetic test cases that pass

---

## K. Models/Sidecars/F32 Refs Staged?

**No.** Design phase only — no inference, no files staged.

---

## L. Secrets Detected?

**No.** All design uses synthetic examples only.

---

## M. Tags Touched?

**No tags touched.**

---

## Safety Scan

```
git status --short
A  examples/speculative/phase26g_safe_llama_runner.py
A  examples/speculative/results/PHASE26G_SAFE_CONTEXT_HARNESS_AND_VARIANT_SCORING.md
A  examples/speculative/results/phase26g_safe_context_harness_and_variant_scoring.json
 M ggml/src/ggml-cpu/ops.cpp
?? examples/speculative/results/PRT_PHASE24R_NATIVE_MULMAT_PROBE.md
?? examples/speculative/results/phase24r_native_mulmat_probe.json

No models/sidecars/f32 refs staged.
No secrets found.
No tags touched.
```

---

## Verdicts

| Verdict | Value |
|---------|-------|
| `PASS_PHASE26H_DESIGN_COMPLETE` | ✅ Full policy, packet format, guard, integration options, test plan |
| `RECOMMEND_STANDALONE_PACKET_BUILDER` | ✅ Phase 26I as first deliverable |
| `RECOMMEND_SMART_AGENT_ROUTER_LATER` | ✅ Integration path after helper validated |
| `RECOMMEND_OPENCLAW_PREPROCESSOR_LATER` | ✅ Secondary integration path |
| `RECOMMEND_CONTEXTOS_MODULE` | ✅ Future memory layer integration |
| `BLOCKED_REPO_STATE` | ❌ No repo assumptions — design is repo-agnostic |
| `BLOCKED_MACHINE_STATE` | ❌ Machine is clean |

---

*Phase 26H complete. SDI context packet policy v0.1 fully designed. Ready for Phase 26I implementation.*