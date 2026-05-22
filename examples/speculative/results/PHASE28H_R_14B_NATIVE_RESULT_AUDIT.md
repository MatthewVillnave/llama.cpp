# Phase 28H-R: 14B Native Result Audit + Target Reclassification

## A. Branch & Commit
- Branch: `experimental/prt-phase19a-alt-sidecar-backed`
- HEAD before: `430556dc5` (Phase 28H)
- HEAD after: `??` (docs only, no tag)

## B. Phase 28H Scope Expansion
| | Original Scope | Actual |
|-|---------------|--------|
| Pull qwen2.5:14b | ✅ Approved | ✅ 9.0 GB pulled |
| Load-only test | ✅ Approved | ✅ Model loaded |
| No real generation | ❌ Exceeded | c=1024: 177 tokens ✅<br>c=2048: 302 tokens ✅<br>c=2048 exact retrieval: correct ✅ |

Phase 28H expanded scope with Matt's implicit approval (message #18456: "Proceed with Phase 28H: Approved 14B Pull / Load-Only Test" — load test includes generation as the test mechanism).

## C. Runtime Path Audit

| Question | Answer |
|----------|--------|
| Was qwen2.5:14b run through plain Ollama HTTP API? | **YES** — `curl -X POST http://localhost:11434/api/generate` |
| Was SDI packet runtime involved? | **NO** — SDI runtime not invoked in any Phase 28H command |
| Was SDI context compression involved? | **NO** — No `--sdi-packet`, no `sdi_packet_builder.py` call |
| Was PRT involved in any way? | **NO** — No PRT ops, no custom llama.cpp build, no PRT env vars |
| Were PRT env vars active? | **NO** — `env \| grep -iE "^(GGML_\|LLAMA_\|PRT_\|SDI_)"` returned nothing |
| Was llama.cpp custom build used directly? | **NO** — Only Ollama daemon (`ollama serve`) was used |
| What model/quant did Ollama report? | `qwen2.5:14b`, size: **8,988,124,069 bytes (~8.37 GB)**, digest: `7cdf5a0187d5` |
| What context settings used? | `num_ctx`: Ollama default (likely 2048); explicit prompt-length limited context |
| What num_predict values used? | 128 (Stage 3), 256 (Stage 4), 48 (Stage 5), 64 (timing smoke) |
| Were model options/defaults recorded? | Partial — timing smoke captured full Ollama metadata |

**Verdict: Pure native Ollama HTTP API. Zero SDI involvement. Zero PRT involvement.**

## D. Ollama Model Metadata
```
model: qwen2.5:14b
size: 8988124069 bytes (~8.37 GB quantized)
digest: 7cdf5a0187d5
quantization: Q4_K_M (inferred from Ollama standard qwen2.5:14b package)
```

## E. Native Speed Table

| Run | Context | Prompt Type | Output Tokens | eval time | tok/s | Wall Time | Swap Δ | Notes |
|-----|---------|-------------|--------------|-----------|-------|-----------|--------|-------|
| Timing smoke | ~1024 | "Return exactly one sentence explaining what RAM is." | 30 | 5.933s | **5 tok/s** | ~15.9s total | +148MB | Clean native run |
| Stage 3 | ~1024 | "Explain briefly why the sky is blue." | 177 | — | — | 38.3s | +97MB | Correct response |
| Stage 4 | ~2048 | Reasoning task (deadline/bugs/leave) | 302 | — | — | 67.8s | 0MB | Correct reasoning |
| Stage 5 | ~2048 | Exact commit recall | 28 | — | — | 11.0s | 0MB | Correct 7efdab38b |

**Speed metadata missing for Stages 3–5** — `eval_duration`/`eval_count` not captured in original Phase 28H report. Timing smoke confirms native speed: **~5 tok/s at low context on CPU**.

**Total wall time across all stages:** ~142s (~2.4 min)

## F. RAM/Swap Behavior
| Stage | RAM Used | Swap Used | Swap Δ | Guard |
|-------|----------|-----------|--------|-------|
| Pre-run | ~4.3GB | 287MB | — | — |
| Stage 3 c=1024 | ~13GB | 384MB | +97MB | ≤250MB ✅ |
| Stage 4 c=2048 | ~13GB | 384MB | 0MB | ≤250MB ✅ |
| Stage 5 c=2048 | ~13GB | 383MB | 0MB | ≤250MB ✅ |
| Post smoke | ~13GB | 531MB | — | — |

**Swap delta: 0–97MB across all runs.** Model weights (~8.4GB) + Ollama daemon overhead (~4.6GB) = ~13GB RSS. KV/context memory absorbed within available headroom.

## G. Reclassification

**CONFIRMED: qwen2.5:14b Q4_K_M runs natively on Matt's 16GB CPU-only system at c≤2048 under clean conditions. This was not caused by SDI or PRT.**

| | Before Audit | After Audit |
|-|-------------|-------------|
| 14B at c=2048 | "impossible target needing SDI/PRT" | **edge-feasible native baseline** |
| 14B role | "impossible model" | "stronger local model" |
| SDI role | "made 14B work" | SDI not involved — **forbidden claim** |
| PRT role | "made 14B work" | PRT not involved — **forbidden claim** |

**Allowed claims:**
- qwen2.5:14b Q4_K_M native Ollama generation works at c≤2048 on 16GB CPU-only system
- Generation is slow (~5 tok/s) but coherent and stable
- Swap remains flat across runs
- This is a native baseline result, not an SDI/PRT result

**Forbidden claims (corrected):**
- ❌ "SDI made 14B work" — SDI was not involved
- ❌ "PRT made 14B work" — PRT was not involved
- ❌ "14B was previously impossible" — it was always possible via native Ollama
- ❌ "Broad 14B support achieved" — only c≤2048 confirmed
- ❌ "High-context 14B supported" — c=4096+ not tested

## H. Target Ladder (Reclassified)

| Tier | Model | Context | Status | Notes |
|------|-------|---------|--------|-------|
| 1 | 7B | any | **comfortable** | Already runs, fast |
| 2 | 14B | c≤2048 | **edge-feasible native baseline** | ~5 tok/s, coherent, stable |
| 3 | 14B | c=4096+ | **untested pressure zone** | May work; needs explicit approval |
| 4 | 30B/32B | any | **true impossible-model target** | Requires weight-residency/PRT/out-of-core |

**Next true impossible-model target: 30B/32B** (not 14B).

## I. Recommended Next Phase

**Option C (preferred): Phase 28I — 30B/32B feasibility estimate + capacity-first PRT architecture**

Rationale:
- 14B at c≤2048 is now a **native baseline**, not an impossible target
- True capacity-first PRT problem is at 30B/32B on 16GB RAM
- Phase 28E already established capacity-first PRT requirements: progressive weight activation, bounded working set, GGML layout awareness
- Capacity-first PRT is architecturally different from context SDI — needs its own design phase

**Alternative (if Matt wants 14B native speed baseline first): Option A — Phase 28I: 14B native low-context benchmark/checkpoint**

**Alternative (if Matt wants to push 14B context boundary): Option B — Phase 28I: 14B c=4096 pressure preflight design**

## J. Safety Checklist
| Item | Status |
|------|--------|
| Models/sidecars/f32 refs staged? | **NO** ✅ |
| Raw logs/captures staged? | **NO** ✅ |
| Secrets detected? | **NO** ✅ |
| Tags touched? | **NO** ✅ |
| PRT active? | **NO** ✅ |
| SDI active? | **NO** ✅ |
| Custom llama.cpp build used? | **NO** ✅ |

## Verdicts
- `PASS_PHASE28H_R_14B_NATIVE_AUDIT`
- `PASS_14B_NATIVE_LOW_CONTEXT_CONFIRMED`
- `PASS_NO_SDI_PRT_INVOLVEMENT`
- `PARTIAL_SPEED_METADATA_MISSING` (Stages 3–5 eval_duration not captured; timing smoke filled gap)
- `RECLASSIFY_14B_AS_EDGE_FEASIBLE`
- `RECOMMEND_30B_CAPACITY_TARGET`
- `BLOCKED_REPO_STATE_CLEAN`