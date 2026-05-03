# PRT Phase 12 Plan

## Starting Point

- RC1 is frozen at `PRT_ROUTE_A_RC1`
- RC1 branch is published at `experimental/prt-route-a-rc1`
- Phase 12 active branch is `experimental/prt-route-a-phase12`
- RC1 commit: `aaa5f290240dbaacfe355ca073bcbce49b18fde7`

---

## Current Validated RC1 Claim

PRT Route A + L12/L15 native anchors achieved **~1.84x average speedup** on the controlled mini-suite (6 prompts), with clean counters, stable memory, no callback overwrite path in accelerated mode, and loud sidecar validation.

---

## Phase 12A — Broader Validation

Run a 20–30 prompt suite:
- narrative
- code
- factual
- JSON/structured
- instruction-following
- weird/edge prompts
- longer n_predict=100 and n_predict=200 where memory allows

**Record per prompt:**
- exact match native vs L12+L15
- first divergence step
- coherence
- collapse/repetition
- JSON validity
- code plausibility
- wall time native
- wall time L12+L15
- speedup
- counters (callback_overwrites, native_fallback_calls, prt_true_replacement_calls)
- RAM before/after

**Pass criteria:** No collapse, counters clean, speedup ≥ 1.5x, no regression from RC1 known-good prompts.

---

## Phase 12B — Speed Attribution

Profile where the ~1.84x speedup comes from:
- native token loop time (baseline)
- Route A token loop time (with L12+L15)
- FFN_UP replacement time (PRT custom op)
- sidecar access time (mmap/load)
- fallback layer time (L12+L15 native)
- startup sidecar validation time
- total wall-clock delta

**Goal:** Explain where the speedup comes from. If it's not from FFN_UP replacement, that's an important finding.

---

## Phase 12C — Sidecar Packaging

Design reproducible, safe sidecar distribution:
- `sidecar_manifest.json` — model, layer count, checksums
- model compatibility metadata (hash, architecture)
- checksum list per layer (L0-L35)
- layer coverage list (which layers have sidecars)
- required vs force-native exclusion list
- loud mismatch failure (if loaded sidecar != manifest checksum → FATAL)

**Goal:** Make sidecars reproducible and auditable before broader release.

---

## Phase 12D — Native Anchor Policy Search

Compare native anchor policies (no callback overwrite allowed):
- L12+L15 (current RC1 policy)
- L13+L14 (alternative pair — also fixes known failure)
- L12+L15+additional candidates (L5, L17)
- all36 pure PRT as negative control (should fail on known-failure prompt)

**Goal:** Find minimal native-anchor policy that preserves quality. Fewer native layers = more PRT coverage = higher speedup potential.

---

## Phase 12E — Broader Model Generalization

Test only after Phase 12C (sidecar packaging) is clean.

**Candidate models:**
- Qwen2.5-7B if available
- Qwen2.5-1.5B if available
- Other GGUF models

**Hard rule:** Do NOT claim generalization until passed. Each new model requires new sidecar extraction and re-validation.

---

## Phase 12F — Upstream Sync Strategy

- RC1 (`PRT_ROUTE_A_RC1`) is frozen forever — never rebase, never amend
- `experimental/prt-route-a-rc1` is the published checkpoint branch — never rebase
- `experimental/prt-route-a-phase12` is the active development branch — rebasing allowed
- Create a separate branch `experimental/prt-upstream-rebase` for rebasing against newer llama.cpp later
- Never push to upstream `ggerganov/llama.cpp` without PR review

---

## Phase 12 Naming

| Phase | Branch | Purpose |
|-------|--------|---------|
| RC1 | `experimental/prt-route-a-rc1` | Frozen checkpoint |
| Phase 12 | `experimental/prt-route-a-phase12` | Active development |
| Future | TBD | Broader validation |

---

## RC1 Frozen State

- Tag: `PRT_ROUTE_A_RC1` — NEVER MOVE
- Commit: `aaa5f290240dbaacfe355ca073bcbce49b18fde7` — NEVER REBASE
- Docs in `examples/speculative/` — never edit retroactively
- RC1 is the documented baseline for all future comparisons

---

## What's Still Unproven

- JSON/structured full validation (n=50+, RAM limited)
- >6 prompt broad suite
- Other model generalization
- Speed attribution (where does 1.84x actually come from?)
- Sidecar reproducibility across machines

---

*Plan created 2026-05-03 — Phase 12*
