# PRT Phase 13S Checkpoint Summary

## Verdict

PASS_CLEAN_QUALITY

## Commit

2e5ba8f9f

## Branch

experimental/prt-phase13-model-generalization

## What This Checkpoint Proves

- Qwen2.5-0.5B active PRT runs cleanly through llama-cli using the fixed Python PTY argv runner.
- Native completed 8/8 prompts.
- Active PRT completed 8/8 prompts.
- Clean generated output was extracted for native and PRT.
- PRT debug logs were routed separately with `--prt-log-file`.
- No PRT debug contamination remained in stdout.
- PRT outputs were semantically acceptable on 8/8 prompts.
- Exact matches occurred on 4/8 prompts.
- No quality degradations were detected.
- No repetition/collapse occurred.
- JSON and code prompts passed.

## What This Checkpoint Does NOT Prove

- No speedup claim.
- No production-readiness claim.
- No larger-model generalization claim.
- No universal PRT quality claim.
- No kernel-level performance advantage claim yet.

## Allowed Claim

PRT dynamic-shape active path preserves clean generation quality on Qwen2.5-0.5B across an 8-prompt clean llama-cli validation suite.

## Forbidden Claims

- Do not claim PRT is faster than native on 0.5B.
- Do not claim production readiness.
- Do not claim general model support.
- Do not claim larger model success from this checkpoint.
- Do not claim exact output equivalence across all prompts.

## Key Files

- `examples/speculative/phase13o_pty_argv_runner.py` — Python PTY argv runner
- `examples/speculative/results/PRT_PHASE13S_05B_CLEAN_QUALITY_COMPARISON.md` — full results
- `examples/speculative/results/phase13s_05b_clean_quality_comparison.json` — structured data
- `examples/speculative/results/PRT_PHASE13R_PRT_LOG_ROUTING.md` — log routing implementation

## Recommended Next Phases

### Phase 13T Option A — Timing Isolation

Separate sidecar load overhead from generation/runtime cost on 0.5B.

### Phase 13T Option B — Larger Model Generalization

Attempt 1.5B or 3B model validation using the same clean runner and `--prt-log-file` mechanism.

**Recommendation:** Run timing isolation first, then larger model generalization.

## Tag

`PRT_PHASE13S_05B_CLEAN_QUALITY_CHECKPOINT`

This tag is a quality/execution checkpoint only. It is not a speedup, production-readiness, or larger-model generalization claim.