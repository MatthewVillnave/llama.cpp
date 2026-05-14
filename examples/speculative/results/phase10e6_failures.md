# Phase 10E-6 Failures & Gaps

## Completed Successfully
- ✅ All 36 sidecar files located and loaded
- ✅ All 36 layers receive PRT substitution
- ✅ Zero wrong-sidecar events
- ✅ Zero fallback events
- ✅ No crashes
- ✅ Clean exit codes

## Failures / Incomplete Data

### 1. Full Generation Output Comparison — NOT COMPLETED
- Longer generation tests (n≥15) exceeded execution time budget
- Could not capture generated text for all 3 prompts with PRT active
- Cannot confirm visible degradation or quality change vs baseline

### 2. Acceptance Rate Metrics — NOT AVAILABLE
- llama-phase10e0-layer0 is a completion harness, not a speculative decoding harness
- Speculative accept/reject metrics are not tracked by this binary
- Would need speculative mode (`-sp`) to measure n_accept/n_draft ratios

### 3. Timing Comparison — INCOMPLETE
- Baseline wall-clock time not measured (no-PRT generation run)
- PRT wall-clock time not measured (full generation exceeded budget)
- tok/s baseline vs PRT comparison not performed

### 4. Subagent Execution Failure
- Full 3-prompt suite subagent run failed due to gateway restart during subagent execution
- Subagent task ultimately produced full model load output but did not reach PRT metrics

## Root Causes
1. Hardware execution speed — CPU inference on Qwen2.5-3B is slow for n≥15 tokens
2. Binary type — completion mode, not speculative mode
3. Execution budget — insufficient time available for full generation suite

## What's Needed to Complete
1. Run baseline and PRT with n=20 on faster hardware or with GPU
2. Use speculative mode (`-sp`) for accept rate tracking
3. Capture full generated text for all 3 prompts
4. Measure wall-clock time for both baseline and PRT