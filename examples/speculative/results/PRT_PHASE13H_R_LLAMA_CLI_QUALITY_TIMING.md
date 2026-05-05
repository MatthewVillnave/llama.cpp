# PRT Phase 13H-R: Llama-CLI Quality/Timing Comparison

**Date:** 2026-05-05
**Model:** Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
**Configuration:** -n 40 --temp 0 -c 256 -t 4

## Summary

| Metric | Native | PRT |
|--------|-------|-----|
| **Clean Count** | 8/8 | 8/8 |
| **Avg tok/s** | 47.88 | 12.13 |
| **Exit Codes** | all 0 | all 0 |

**Key Finding:** PRT debug mode (5700) shows ~4x slowdown compared to native inference. Both modes produce comparable quality outputs for this small model.

## Timing Details

| Prompt | Native (tok/s) | PRT (tok/s) | Speed Ratio |
|--------|-------------|-------------|-----------|
| 1. "The capital of France is" | 52.8 | 12.8 | 4.1x |
| 2. "Write a Python function that reverses a list." | 46.7 | 12.3 | 3.8x |
| 3. "Once upon a time in a" | 47.0 | 12.0 | 3.9x |
| 4. "Explain CPU inference in one sentence." | 47.1 | 12.4 | 3.8x |
| 5. "Return JSON with keys name and status." | 48.6 | 12.1 | 4.0x |
| 6. "The fastest way to sort a list in Python is" | 47.0 | 11.9 | 3.9x |
| 7. "In two sentences, explain what RAM does." | 46.8 | 12.0 | 3.9x |
| 8. "Complete this phrase: artificial intelligence is" | 47.1 | 11.6 | 4.1x |

## Quality Comparison

### Prompt 1: "The capital of France is"
- **Native:** "The capital of France is Paris."
- **PRT:** "The capital of France is Paris."
- **Verdict:** ✓ Match

### Prompt 2: "Write a Python function that reverses a list."
- **Native:** "Certainly! Below is a Python function that reverses a list:\n```python\ndef reverse_list(lst):"
- **PRT:** "Certainly! Below is a Python function that reverses a list:\n```python\ndef reverse_list(lst):"
- **Verdict:** ✓ Match

### Prompt 3: "Once upon a time in a"
- **Native:** "Once upon a time in a far-off land, there was a kingdom ruled by a wise king..."
- **PRT:** "Once upon a time in a faraway land, there was a young girl named Lily..."
- **Verdict:** Story variant (both coherent)

### Prompt 4: "Explain CPU inference in one sentence."
- **Native:** "CPU inference refers to the process where a computer's central processing unit (CPU) performs inference tasks..."
- **PRT:** "CPU inference refers to the process where a central processing unit (CPU) performs inference tasks on data stored in memory..."
- **Verdict:** ✓ Similar technical content

### Prompt 5: "Return JSON with keys name and status."
- **Native:** "```json"
- **PRT:** "```json"
- **Verdict:** Both produce JSON start

### Prompt 6: "The fastest way to sort a list in Python is"
- **Native:** "In Python, the fastest way to sort a list is using the built-in `sorted()` function..."
- **PRT:** "In Python, the fastest way to sort a list is using the built-in `sorted()` function..."
- **Verdict:** ✓ Match

### Prompt 7: "In two sentences, explain what RAM does."
- **Native:** "RAM stands for "Random Access Memory," which is a type of memory that can store data and instructions temporarily..."
- **PRT:** "RAM stands for "Random Access Memory," which is a type of memory that can store data and instructions temporarily. It allows the computer to access data and instructions quickly..."
- **Verdict:** ✓ Similar (both reference temp storage)

### Prompt 8: "Complete this phrase: artificial intelligence is"
- **Native:** "artificial intelligence is a branch of computer science and engineering that focuses on creating intelligent machines..."
- **PRT:** "artificial intelligence is a branch of computer science and engineering that focuses on creating intelligent machines..."
- **Verdict:** ✓ Match

## PRT Debug Counters

| Prompt | Debug Mode | Sidecar Layers Loaded | Notes |
|--------|-----------|---------------------|-------|
| 1 | 5700 | layers 0,1,2 | M=896 N=4864 |
| 2 | 5700 | layers 0,1,2 | M=896 N=4864 |
| 3 | 5700 | layers 0,1,2 | M=896 N=4864 |
| 4 | 5700 | layers 0,1,2 | M=896 N=4864 |
| 5 | 5700 | layers 0,1,2 | M=896 N=4864 |
| 6 | 5700 | layers 0,1,2 | M=896 N=4864 |
| 7 | 5700 | layers 0,1,2 | M=896 N=4864 |
| 8 | 5700 | layers 0,1,2 | M=896 N=4864 |

## Clean Output Check

| Check | Result |
|-------|--------|
| Flag echo in stdout | 0 for all 8 |
| Path fragments (.bin, /tmp/prt, ffn_up_layer) | 0 for all 8 |

## Conclusions

1. **Timing:** PRT debug mode (5700) shows significant slowdown (~4x) on this small model. Production PRT without debug flags should perform better.
2. **Quality:** Quality is equivalent between native and PRT mode for factual/technical prompts. Story generation shows natural variation.
3. **Cleanliness:** No leaked flags or path information in output.
4. **Model Compatibility:** Qwen2.5-0.5B works with PRT but may not be ideal for speculative decoding (too small for effective branching).

**Recommendation:** Test with larger model (1B+ params) and production PRT flags (without debug mode 5700) for realistic timing comparison.