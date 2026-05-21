# Phase 27D: 7B Safe-Context Validation Planning

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`e66f56dc8` (Phase 27C commit)

## C. Checkpoint Verified
SDI_PHASE26W_RUNTIME_V0_1_1_CHECKPOINT exists

## D. Known Hardware Constraints

| Constraint | Value |
|------------|-------|
| Machine RAM | 15 GB total, ~12 GB available |
| Machine swap | 4 GB total, ~168 MB used (light) |
| qwen2.5:7B Q4_K_M size | ~4.4 GB |
| qwen2.5:7B context defaults | c=32768 (Ollama default) |
| qwen2.5:3B status | installed, 1.9 GB |
| qwen2.5:0.5B status | installed, 397 MB |
| **qwen2.5:7B status** | **NOT installed** |

### Previously established safety bounds (from prior phases):
- c=2048: safe for 7B in normal RAM conditions
- c=4096: low memory pressure for 7B
- c=8192: previously showed low pressure but no broad testing
- c=16384: danger zone -- triggers swap on constrained machines
- **Hard rule:** no 7B run if swap used > 1 GB
- **Hard rule:** stale llama/ollama processes must be killed before any run

### Memory math for 7B:
- Model weights: ~4.4 GB (Q4_K_M)
- KV/context at c=2048: ~500 MB
- KV/context at c=4096: ~1 GB
- KV/context at c=8192: ~2 GB
- KV/context at c=16384: ~4 GB
- Available RAM: ~12 GB
- With buffer: c=2048/4096 comfortably fits; c=8192 fits with margin; c=16384 is risky

---

## E. 7B Validation Objective

**Core question:** Can SDI Runtime v0.1.1 help qwen2.5:7B operate inside CPU/RAM limits by reducing active context pressure?

**NOT speed.** The objective is memory safety and task correctness under bounded context.

**Objective:** Verify SDI auto-policy selects correctly (recent_only/simple_summary/auto) on 7B for tasks where SDI helped 0.5B, while keeping swap pressure low.

---

## F. Selected Future Tasks (Phase 27E -- Tiny 7B Canary)

Only **2 tasks** for the first 7B canary:

| Task | Why | Expected benefit |
|------|-----|-----------------|
| **Sc21** research_handoff | Long pinned-fact context; SDI policy helped at 0.5B; tests factual recall under bounded context | Verifies if recent_only works for 7B factual recall |
| **Sc23** benchmark_interpretation | Benchmark tables + long context; tests simple_summary routing | Verifies if simple_summary reduces context vs raw |

**NOT** Sc25/Sc26: Both scored 1.000 already on 0.5B, lower priority for 7B validation.

---

## G. Proposed Phase 27E Test Matrix

### Model
- **qwen2.5:7B** -- only if explicitly approved and pulled
- Do NOT auto-pull without Matt's explicit approval
- If not available, block: BLOCKED_7B_NOT_AVAILABLE

### Contexts (conservative stair-step)
1. **c=2048 first** -- verify no-swap baseline
2. **c=4096 second** -- only if c=2048 runs cleanly (swap delta < 100 MB)
3. **c=8192** -- only in a later phase with explicit approval
4. **Never c=16384** in Phase 27E

### Baselines
- recent_only -- minimal context baseline
- simple_summary -- SDI summary baseline
- auto -- policy-selected (primary test)

### Metrics per run
- [x] task score (0.0-1.0)
- [x] selected policy
- [x] swap delta (before/after)
- [x] output sanity
- [x] timeout/cap behavior
- [ ] wall time (optional, low priority)

### Skip conditions
- Do NOT run no_packet if raw context exceeds c=4096
- Do NOT run sdi_packet unless explicitly needed
- Do NOT run all 8 tasks
- Do NOT run full benchmark suite

---

## H. Safety Guard

### Before each run:
```
# Check RAM/swap state
free -h
swapon --show        # block if swap > 1GB

# Kill stale processes
pkill -f llama-app 2>/dev/null || true
pkill -f llama-server 2>/dev/null || true

# Verify ollama is responsive
curl -s http://localhost:11434/api/tags
```

### Output rules:
- All outputs go to /tmp/sdi_7b_*/, NOT staged to repo
- No generated packets/captures/logs in repo
- Clear /tmp after each run

### Run limits:
- Strict timeout: 180s per run
- Strict output cap: 2048 tokens
- One run at a time (no parallel)
- Use Ollama HTTP API

### After each run:
Record swap delta. Stop immediately if swap delta > 250 MB.

---

## I. Abort Criteria

**Abort immediately if:**
- swap increases > 250 MB after any single run
- available RAM drops below 3 GB during run
- OS kills a process (OOM kill)
- output capture grows unexpectedly
- machine becomes sluggish
- stale llama/ollama process remains after timeout
- swap was already > 1 GB at pre-run check
- any run produces no output (model load failure)
- model fails to respond (Ollama crash)

**Phase 27E stops permanently if:**
- First c=2048 run exceeds swap threshold
- Any run triggers OOM
- Swap delta compounds across runs

---

## J. Claim Boundaries

### Allowed after Phase 27D (planning only):
- "7B safe-context validation plan created"
- "Phase 27E will test 2 tasks at c=2048/4096 if approved"
- "qwen2.5:7B not installed; requires explicit approval to pull"

### Forbidden after Phase 27D or Phase 27E:
- 7B validated / 7B safe / 7B production ready
- 7B speedup claims
- Long-context solved
- KV cache modified
- Weight-residency solution
- SDI enables larger models universally
- Broad 7B results without full eval

### Allowed after Phase 27E (if runs complete cleanly):
- "qwen2.5:7B at c=2048/4096 showed stable swap under SDI auto-policy on 2 tasks"
- "Auto policy selected recent_only/simple_summary correctly on 7B for those tasks"
- "No swap pressure detected at c=2048 for selected tasks"

---

## K. Recommended Next Phase

**Phase 27E -- Tiny 7B SDI Canary** (requires explicit Matt approval to pull qwen2.5:7B)

Scope:
- 2 tasks (Sc21, Sc23)
- c=2048 only (no c=4096 unless c=2048 is clean)
- 3 baselines (recent_only, simple_summary, auto)
- Strict safety protocol
- Swap delta monitoring before/after each run

**If Matt does NOT approve 7B pull:** Phase 27F -- public/internal writeup. No model pull, no new eval.

---

## L. Models/sidecars/f32 refs staged?
No. qwen2.5:7B is NOT installed. No model files staged.

## M. Secrets detected?
No.

## N. Tags touched?
No. No existing tags modified. No new tag created.

---

## Verdict
PASS_PHASE27D_7B_VALIDATION_PLAN
PASS_SAFETY_GUARD_DEFINED
PASS_ABORT_CRITERIA_DEFINED
BLOCKED_7B_NOT_AVAILABLE (requires explicit approval to pull)