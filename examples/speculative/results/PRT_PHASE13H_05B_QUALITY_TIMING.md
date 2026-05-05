# PRT Phase 13H: Qwen2.5-0.5B Quality + Timing Comparison

## Branch
- **fork/experimental/prt-phase13-model-generalization**

## Context
Following Phase 13G (clean llama-cli frontend canary), this phase runs 8 diverse prompts through both native and PRT-active llama-simple to compare output quality and inference speed.

## Model & Environment
- **Model:** Qwen2.5-0.5B-Instruct-Q4_K_M (`Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`)
- **Binary:** `llama-simple` (interactive prompt mode, simpler than llama-cli)
- **PRT mode:** `--prt-mode 5700` (all layers active, full PRT)
- **Sidecars:** `/tmp/prt_sidecars/` (24 sidecars loaded for all PRT runs)
- **Force-native:** layers 11, 15

## Prompts & Results

| # | Prompt | Native Output (first 80 chars) | PRT Output (first 80 chars) | Verdict |
|---|--------|------------------------------|----------------------------|---------|
| 1 | "The capital of France is" | "The capital of France is Paris. It is the largest city in Europe..." | "The capital of France is Paris. The capital of France is Paris. The capital of France..." | **FAIL** — PRT loops |
| 2 | "Write a Python function that reverses a list." | "Write a Python function that reverses a list. The function should..." | "Write a Python function that reverses a list. The function should... Use the `reverse()` method..." | **PASS** — clean |
| 3 | "Once upon a time in a" | "Once upon a time in a faraway land, there was a magical tree..." | "--prt-mode 5700 Once upon a time in a land of knights..." | **FAIL** — flag echo |
| 4 | "Explain CPU inference in one sentence." | "Explain CPU inference in one sentence. CPU inference refers..." | "--prt-mode 5700 Explain CPU inference in one sentence. Explain CPU inference..." | **FAIL** — flag echo + repetition |
| 5 | "Return JSON with keys name and status." | "Return JSON with keys name and status. If status is 'active'..." | "--prt-mode 5700 Return JSON with keys name and status. If the status is 'error'..." | **FAIL** — flag echo |
| 6 | "The fastest way to sort a list in Python is" | "The fastest way to sort a list in Python is to use the built-in `sorted()`..." | "--prt-mode 5700 The fastest way to sort a list in Python is..." | **FAIL** — flag echo |
| 7 | "In two sentences, explain what RAM does." | "In two sentences, explain what RAM does. RAM (Random Access Memory) is a..." | "--prt-mode 5700 In two sentences, explain what RAM does. RAM (Random Access Memory) is..." | **FAIL** — flag echo |
| 8 | "Complete this phrase: artificial intelligence is" | "Complete this phrase: artificial intelligence is a(n) ________ of..." | "--prt-mode 5700 Complete this phrase: artificial intelligence is..." | **FAIL** — flag echo |

## Timing Results

| Prompt | Native (tok/s) | PRT (tok/s) | Delta | Notes |
|--------|---------------|-------------|-------|-------|
| 1 | 32.01 | 27.76 | -13.3% | PRT slower |
| 2 | 29.58 | 35.04 | +18.4% | PRT faster ⚠️ anomalous |
| 3 | 31.24 | 27.93 | -10.6% | PRT slower |
| 4 | 28.01 | 27.50 | -1.8% | ~same |
| 5 | 23.31 | 22.93 | -1.6% | ~same |
| 6 | 22.88 | 22.50 | -1.7% | ~same |
| 7 | 23.14 | 22.96 | -0.8% | ~same |
| 8 | 23.33 | 22.76 | -2.4% | ~same |
| **Avg** | **26.69** | **26.17** | **-2.0%** | |

## Quality Analysis

### Native Clean Count: **8/8** ✅
All 8 native runs produced clean, correct-looking output with no garbage tokens or path fragments.

### PRT Clean Count: **1/8** ❌
Only prompt 2 (Python function) produced clean output. The remaining 7 PRT runs all have `--prt-mode 5700` flag fragments echoed into the output stream.

### Key Issues

1. **Flag echo in output (7/8 PRT runs):** The `--prt-mode 5700` flag is being echoed as part of the output text for most prompts. This is a regression from the clean Phase 13G canary run on the capital-of-France prompt.

2. **Repetition/looping (Prompt 1):** PRT produced the same sentence 5+ times before cut-off — classic model instability under PRT perturbation.

3. **Anomalous speed (Prompt 2):** PRT ran 18% faster than native for the Python prompt — counter to the general trend and suggests timing variance.

## Path Fragment Check
- **Native path fragments:** 0/8 ✅
- **PRT path fragments (`ffn_up_layer`, `.bin`, `/tmp/prt`):** 0/8 ✅ (no file path leaks)

## Verdict: FAIL ❌

PRT active mode produces corrupt/flagged output on 7/8 prompts. The root cause is likely the flag echo into the output stream rather than a model quality issue per se. The Phase 13G canary on "The capital of France is" passed but the same flag fails on other prompts — suggesting the echo only manifests when the model doesn't immediately start generating (e.g., short factual prompts may get past it while longer prompts hit it).

## Recommended Next Steps
1. **Fix flag echo:** Investigate why `--prt-mode 5700` appears in stdout for prompts 3-8 but not 1-2 — likely a model/prompt interaction with the output stream
2. **Fix looping:** Investigate why Prompt 1 loops on PRT (model instability under full-layer perturbation)
3. **Validate PRT-disabled mode:** Run same 8 prompts with no PRT flags to confirm clean baseline

## Secrets Detected?
- **No** — no API keys, tokens, or credentials in any output files
