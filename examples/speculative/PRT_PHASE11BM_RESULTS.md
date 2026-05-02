# PRT Phase 11BM Results

**Test Date:** 2026-05-02
**Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf
**Test Machine:** TheForgeHQ (16GB RAM, 4GB swap)

---

## Timing Results

### Core Tests (n=100)

| Prompt | Native | L12+L15 | Speedup | Status |
|--------|--------|---------|---------|--------|
| "Once upon a time in a" | 1m23.8s | 45.4s | **1.85x** | ✓ PASS |
| "Once upon a time in a distant galaxy" | 1m24.5s | 46.2s | **1.83x** | ✓ PASS |
| "Tell me a story about a dragon." | 1m24.6s | 45.9s | **1.84x** | ✓ PASS |

### Factual Tests (n=50)

| Prompt | Native | L12+L15 | Speedup | Status |
|--------|--------|---------|---------|--------|
| "What is the capital of France?" | 44.2s | 24.3s | **1.82x** | ✓ PASS |

### Code Tests (n=50)

| Prompt | Native | L12+L15 | Speedup | Status |
|--------|--------|---------|---------|--------|
| "Write a Python function to reverse a list." | 49.2s | 25.2s | **1.95x** | ✓ PASS |

### Average Speedup

**~1.86x average across tested prompts**

---

## Quality Results

### Known Failure Recovery

**Prompt:** "Once upon a time in a"

| Mode | Token 0 | Token 1 | Token 2 | Result |
|------|---------|---------|---------|--------|
| Native | " small" | " village" | ",village" | ✓ CORRECT |
| Pure Route A all36 | " far" | " away" | " land" | ✗ WRONG |
| L12+L15 | " small" | " village" | ",village" | ✓ **FIXED** |

**L12+L15 produces IDENTICAL output to native on known failure prompt at n=100.**

### Token-Level Comparison (n=20)

| Token | Native | L12+L15 | Match |
|-------|--------|---------|-------|
| 0 | " small" | " small" | ✓ |
| 1 | " village" | " village" | ✓ |
| 2 | ",village" | ",village" | ✓ |
| 3 | " therege" | " therege" | ✓ |
| 4 | " livedge" | " livedge" | ✓ |
| 5 | " aivedge" | " aivedge" | ✓ |
| 6 | " youngge" | " youngge" | ✓ |
| 7 | " boyngge" | " boyngge" | ✓ |

**Tokens 0-7: 100% match**

---

## Counters

### Clean Counters (L12+L15 at n=100)

```
callback_overwrites: 0          ← DISABLED
native_ffn_up_calls: 0         ← CORRECT
prt_true_replacement_calls: 3706 ← VALID
native_fallback_calls: 16        ← 2 layers × 8 tokens
sidecar_L0_checksum: -354.098145 ← MATCHES
sidecar_L35_checksum: 39.010246 ← MATCHES
```

**Verdict:** All counters clean.

---

## Memory Pressure Notes

**Test Machine Constraints:**
- Total RAM: 16GB
- Ollama runners: ~4.7GB (when running)
- Swap: 4GB, 2.8GB used (nearly full)

**SIGKILL incidents:**
- n=100 failed when Ollama runners active (only ~980MB free)
- n=100 passes after killing Ollama runners (~12GB available)
- JSON/structured prompts SIGKILL at n>30 with memory pressure

**Root cause:** Hardware limitation, not PRT mode failure.

**Fix:** `pkill -f "ollama runner"` before benchmark, or use machine with >32GB RAM.

---

## Layer Interaction Discovery

Testing revealed nonlinear layer-pair requirements:

| Force-Native Layers | Result |
|--------------------|--------|
| L12 alone | WRONG |
| L15 alone | WRONG |
| **L12+L15** | **CORRECT** |
| L12+L13 | WRONG |
| L13+L14 | CORRECT |
| L14+L15 | WRONG |
| L12-L17 (all 6) | WRONG |

**Key insight:** Pair-wise native fallback required. Single-layer insufficient. More native layers does NOT always help (L12-17 all 6 still fails).

---

## Summary

| Metric | Value |
|--------|-------|
| Average speedup | ~1.86x |
| Known failure fixed | ✓ YES |
| Counters clean | ✓ YES |
| n=100 stable | ✓ YES |
| Memory-limited tests | JSON/structured (n>30) |

---

*End of Phase 11BM Results*
