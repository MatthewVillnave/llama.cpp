# Next Phase Plan: Controlled Mini-Suite Benchmark

**Version:** 1.0
**Date:** 2026-05-02
**Goal:** Complete broader suite validation on cleaned test machine

---

## Prerequisite: Memory Cleanup

Before running benchmark, ALWAYS clean memory:

```bash
# Kill Ollama runners
pkill -f "ollama runner"

# Verify memory freed
free -h
# Expected: ~12GB available
```

**If Ollama runners are needed:** Stop them before benchmark, restart after.

Alternative: Use machine with >32GB RAM to avoid cleanup.

---

## Benchmark Command Checklist

### 1. Narrative Prompts (n=100)

```bash
# Prompt 1
./build/bin/llama-prt-posix -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "Once upon a time in a" -n 100 --prt-mode 5700 --prt-force-native "12,15"

# Prompt 2
./build/bin/llama-prt-posix -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "Once upon a time in a distant galaxy" -n 100 --prt-mode 5700 --prt-force-native "12,15"

# Prompt 3
./build/bin/llama-prt-posix -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "Tell me a story about a dragon." -n 100 --prt-mode 5700 --prt-force-native "12,15"

# Prompt 4
./build/bin/llama-prt-posix -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "Complete this sentence: The old machine in the basement" -n 100 --prt-mode 5700 --prt-force-native "12,15"
```

### 2. Code Prompts (n=50)

```bash
# Code 1
./build/bin/llama-prt-posix -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "Write a Python function to reverse a list." -n 50 --prt-mode 5700 --prt-force-native "12,15"

# Code 2
./build/bin/llama-prt-posix -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "def quick_sort(arr):" -n 50 --prt-mode 5700 --prt-force-native "12,15"
```

### 3. Factual Prompts (n=50)

```bash
# Factual 1
./build/bin/llama-prt-posix -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "What is the capital of France?" -n 50 --prt-mode 5700 --prt-force-native "12,15"

# Factual 2
./build/bin/llama-prt-posix -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "Explain photosynthesis in one paragraph." -n 50 --prt-mode 5700 --prt-force-native "12,15"

# Factual 3
./build/bin/llama-prt-posix -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "List three reasons CPUs are hard for LLM inference." -n 50 --prt-mode 5700 --prt-force-native "12,15"
```

### 4. Structured Prompts (n=30-50)

```bash
# JSON 1 (start with n=30)
./build/bin/llama-prt-posix -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "Give me a JSON object with name and age." -n 30 --prt-mode 5700 --prt-force-native "12,15"

# JSON 2
./build/bin/llama-prt-posix -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "Return a JSON array of three fruits." -n 30 --prt-mode 5700 --prt-force-native "12,15"
```

---

## Pass/Fail Criteria

### Pass Conditions
- All prompts complete (no SIGKILL from mode failure)
- callback_overwrites = 0 on all runs
- No collapse/repetition in output
- Coherent output on narrative prompts
- Speedup ≥ 1.5x on all prompts

### Fail Conditions
- Any run produces collapse/repetition
- callback_overwrites > 0
- Speedup < 1.5x on any prompt
- Output diverges significantly from native on quality-critical prompts
- SIGKILL occurs after memory cleanup (indicates mode issue, not resource)

### Conditional Pass (needs investigation)
- JSON output not valid (syntax errors)
- Markdown table malformed
- Code output not syntactically plausible

---

## Expected Duration

| Prompts | n | Estimated Time |
|---------|---|---------------|
| 4 narrative | 100 | ~3 min each |
| 2 code | 50 | ~25 sec each |
| 3 factual | 50 | ~25 sec each |
| 2 structured | 30-50 | ~20 sec each |
| **Total** | | **~15 min** |

---

## Next Phase Name

**Phase 11BN:** Controlled Mini-Suite Completion

---

*End of Next Phase Plan*
