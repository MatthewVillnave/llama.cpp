# PRT Phase 13S: 0.5B Clean Quality Comparison

**Verdict: PASS_CLEAN_QUALITY** ✅

## Summary

Full 8-prompt native vs PRT quality comparison using `--prt-log-file` for clean output extraction. All runs completed with clean stdout (no PRT debug contamination). All PRT log files contain full PRT evidence.

## Results Table

| # | Prompt | Exact | Semantic | PRT Clean | Notes |
|---|--------|-------|----------|-----------|-------|
| 1 | The capital of France is | ✅ | ✅ | ✅ | Identical outputs |
| 2 | Write a Python function that reverses a list. | ❌ | ✅ | ✅ | Minor: "a list" vs "a given list" |
| 3 | Once upon a time in a | ❌ | ✅ | ✅ | Minor: "far-off land" vs "faraway land" |
| 4 | Explain CPU inference in one sentence. | ❌ | ✅ | ✅ | Minor: "computer's CPU" vs "CPU (CPU)" |
| 5 | Return JSON with keys name and status. | ✅ | ✅ | ✅ | Identical JSON |
| 6 | The fastest way to sort a list in Python is | ✅ | ✅ | ✅ | Identical outputs |
| 7 | In two sentences, explain what RAM does. | ❌ | ✅ | ✅ | Minor: "temporarily during" vs "temporarily. It allows" |
| 8 | Complete this phrase: artificial intelligence is | ✅ | ✅ | ✅ | Identical outputs |

## Detailed Diff Analysis

**Prompt 2:** "reverses a list" (N) vs "reverses a given list" (P) — same meaning
**Prompt 3:** "far-off land" (N) vs "faraway land" (P) — story variation, same narrative
**Prompt 4:** "a computer's central processing unit" (N) vs "a central processing unit (CPU)" (P) — same concept, minor rewording
**Prompt 7:** "temporarily during the execution" (N) vs "temporarily. It allows the computer" (P) — same meaning, different continuation

None of these represent quality degradation — both outputs are semantically equivalent and correct.

## Quality Checks

| Check | Result |
|-------|--------|
| Native completed | 8/8 ✓ |
| PRT completed | 8/8 ✓ |
| Clean output extracted (native) | 8/8 ✓ |
| Clean output extracted (PRT) | 8/8 ✓ |
| No PRT debug in generated stdout | 8/8 ✓ |
| PRT log contains PRT_SHAPE | 8/8 ✓ |
| PRT log contains sidecar evidence | 8/8 ✓ |
| PRT log contains replacement/auth | 8/8 ✓ |
| No timeouts | 8/8 ✓ |
| No invalid argument errors | 8/8 ✓ |
| No flag echo | 8/8 ✓ |
| No path fragments in output | 8/8 ✓ |
| No repetition/collapse | 8/8 ✓ |
| Exact matches | 4/8 (minor word variations in others) |
| Semantic matches | 8/8 ✓ |
| Quality degradation | 0/8 ✓ |
| JSON validity (prompt 5) | Native ✓, PRT ✓ |
| Code plausibility (prompt 2) | Both ✓ |
| Code plausibility (prompt 6) | Both ✓ |

## Timing

| Prompt | Native (s) | PRT (s) | Ratio |
|--------|-----------|---------|-------|
| 1 | 0.604 | 2.340 | 3.9× |
| 2 | 1.024 | 4.334 | 4.2× |
| 3 | 1.019 | 4.226 | 4.1× |
| 4 | 1.014 | 4.306 | 4.2× |
| 5 | 0.804 | 3.249 | 4.0× |
| 6 | 1.013 | 4.341 | 4.3× |
| 7 | 0.880 | 4.286 | 4.9× |
| 8 | 1.013 | 4.215 | 4.2× |

Note: No speedup claims made — timing shown for reference only.

## PRT Log File Sizes

| Prompt | Log Size |
|--------|---------|
| 1 | 35,684 |
| 2 | 148,745 |
| 3 | 148,694 |
| 4 | 148,605 |
| 5 | 85,134 |
| 6 | 148,626 |
| 7 | 148,683 |
| 8 | 148,723 |

Prompts 2/3/4/6/7/8 all ~148KB (full batch inference with PRT custom op logs for 22 layers × 36+ tokens).
Prompt 1/5 smaller (40 token generation with fewer custom op calls).

## Safety
- No models staged: ✓
- No secrets leaked: ✓
- Tags untouched: ✓