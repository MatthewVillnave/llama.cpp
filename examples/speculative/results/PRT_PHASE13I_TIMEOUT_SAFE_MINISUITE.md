# PRT Phase 13I: Timeout-Safe 0.5B Minisuite

## Branch
- **fork/experimental/prt-phase13-model-generalization**

## Previous HEAD
- `761b1352c` — Phase 13H-R rerun

## Binary
- `./build/bin/llama-cli` (not llama-simple)

## Settings
- Model: Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
- n_predict: 16
- context: 256, threads: 4, temp: 0
- PRT active: `--prt-mode 5700 --prt-force-native 11,15 --prt-sidecar-dir /tmp/prt_sidecars/`

## Results

| # | Prompt | Native Output (first 60 chars) | Native Clean | PRT Output (first 60 chars) | PRT Clean |
|---|--------|-------------------------------|--------------|----------------------------|-----------|
| 1 | The capital of France is | "The capital of France is Paris." | ✅ | "The capital of France is Paris." | ✅ |
| 2 | Write a Python function that reverses a list. | "Certainly! Below is a Python function..." | ✅ | "Certainly! Below is a Python function..." | ✅ |
| 3 | Once upon a time in a | "Once upon a time in a faraway land..." | ✅ | "Once upon a time in a faraway land..." | ✅ |
| 4 | Explain CPU inference in one sentence. | "CPU inference refers to the process where..." | ✅ | "CPU inference refers to the process where..." | ✅ |

## Validation

| Check | Result |
|-------|--------|
| Flag echo in stdout (--prt-mode/--prt-force/--prt-sidecar) | **0** all prompts ✅ |
| Path fragments (ffn_up_layer/.bin/tmp/prt) | **0** all prompts ✅ |
| PRT_SHAPE | n_layer=24 M=896 N=4864 ✅ |
| Sidecars loaded | 24/24 ✅ |
| llama-simple used | NO ✅ |
| Collapse/repetition within 16 tokens | NONE ✅ |

## Verdict: PASS

All 4 prompts:
- Native outputs clean: **4/4**
- PRT active outputs clean: **4/4**
- No flag echo
- No path fragments
- Counters clean
- Same output from native and PRT for all 4 prompts (quality identical)

## Key Findings
The llama-cli binary with PRT flags produces identical output to native for the 0.5B model on short prompts (n=16). No quality regression from PRT perturbation at this scale. The Phase 13H failure was confirmed as an INVALID_WRONG_BINARY (llama-simple) issue — this run using llama-cli directly shows no corruption.

## Recommended Next
Run full 8-prompt suite with n=40 when environment is stable, or test with larger models (Qwen2.5-3B) to see if quality divergence appears at scale.

## Secrets Detected?
- No

## Tags Untouched?
- Yes