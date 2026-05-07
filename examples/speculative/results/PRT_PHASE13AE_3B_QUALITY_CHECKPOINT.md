# PRT Phase 13AE — 3B Full Quality Checkpoint

**Date:** 2026-05-07  
**Verdict:** PASS_3B_FULL_QUALITY  
**Commit:** 20d7911ed2a84d410e0baa4bce5645d38941e639  
**Branch:** `experimental/prt-phase13-model-generalization`

---

## What This Checkpoint Proves

- Qwen2.5-3B active PRT runs through the clean llama-cli PTY runner ✅
- 3B sidecars load successfully: 36/36 ✅
- Model shape is layers=36, hidden=2048, ffn=11008 ✅
- Native completed 8/8 validation prompts ✅
- Active PRT completed 8/8 validation prompts ✅
- Clean generated output was extracted for both native and PRT ✅
- PRT debug logs were routed separately with `--prt-log-file` ✅
- PRT debug contamination in stdout was 0/8 ✅
- PRT outputs were semantically acceptable on 8/8 prompts ✅
- Exact matches occurred on 3/8 prompts (prompts 1, 2, 5)
- No quality degradations were detected ✅
- No repetition/collapse occurred ✅
- JSON prompt passed for native and PRT (prompt 5: `{"name":"Qwen","status":"active"}` valid both) ✅
- Code prompts passed for native and PRT (prompts 2, 6: plausible code both) ✅
- PRT_SHAPE showed `n_layer=36 M=2048 N=11008` on all 8 runs ✅
- AVX2 replacement path was active (compile_flags=-mavx2 -mfma, PRT_KERNEL=avx2 on all 8) ✅
- Fallback calls were 0 across all 8 prompts ✅

---

## What This Checkpoint Does NOT Prove

- ❌ No 3B speedup claim
- ❌ No production-readiness claim
- ❌ No larger-than-3B generalization claim
- ❌ No universal PRT quality claim
- ❌ No exact output equivalence across all prompts (3/8 exact matches)
- ❌ No latency overhead optimization claim

---

## Allowed Claim

> Qwen2.5-3B active PRT passed an 8-prompt clean quality validation with 8/8 semantic matches, 0 quality degradations, 0 repetition/collapse, 36/36 sidecars loaded, AVX2 replacement active, and 0 fallback calls.

---

## Forbidden Claims

- Do not claim PRT is faster than native on 3B
- Do not claim production readiness
- Do not claim general model support
- Do not claim larger-than-3B success
- Do not claim exact equivalence across all prompts
- Do not claim latency overhead is solved

---

## Key Files

| File | Purpose |
|------|---------|
| `examples/speculative/phase13o_pty_argv_runner.py` | PTY runner used for clean output capture |
| `examples/speculative/results/PRT_PHASE13AE_3B_FULL_QUALITY_VALIDATION.md` | Full phase report |
| `examples/speculative/results/phase13ae_3b_full_quality_validation.json` | Structured validation data |
| `examples/speculative/results/PRT_PHASE13AE_3B_QUALITY_CHECKPOINT.md` | This checkpoint summary |

---

## 8-Prompt Validation Summary

| Prompt | Native | PRT | Exact | Semantic | Special |
|--------|--------|-----|-------|----------|---------|
| 1. The capital of France is | Paris. | Paris. | ✅ | ✅ | |
| 2. Write a Python function... | reverse_list code | reverse_list code | ✅ | ✅ | Code ✅ |
| 3. Once upon a time in a | Kingdom narrative | Kingdom narrative | ❌ | ✅ | |
| 4. Explain CPU inference... | CPU definition | CPU definition | ❌ | ✅ | |
| 5. Return JSON... | `{"name":"Qwen","status":"active"}` | `{"name":"Qwen","status":"active"}` | ✅ | ✅ | JSON ✅ |
| 6. The fastest way to sort... | sort() method | sort methods | ❌ | ✅ | Code ✅ |
| 7. In two sentences, explain RAM... | RAM description | RAM description | ❌ | ✅ | |
| 8. Complete: artificial intelligence is | Technological advancements | Next generation innovation | ❌ | ✅ | |

**Totals:** 3/8 exact, 8/8 semantic, 0 degradations, 0 collapse ✅

---

## Timing Evidence (Informational Only)

| Metric | Native Avg | PRT Avg | Ratio |
|--------|-----------|---------|-------|
| Generation tok/s | 19.4 | 7.9 | 0.41× |
| Wall time | 3.23s | 12.92s | 4.00× |
| Sidecar load (PRT) | — | 4579ms avg | — |

**Note:** PRT is slower. No speedup claim is made.

---

## AVX2 Evidence (All 8 Runs)

Each PRT log ends with:
```
[PRT-BUILD] PRT_KERNEL=avx2 (kernel_mode=1)
[PRT-BUILD] compile_flags=-mavx2 -mfma (LLAMA_PRT_AVX2)
[PRT-BUILD] __AVX2__=defined
[PRT-BUILD] __FMA__=defined
```

---

## Recommended Next Phase

**Phase 13AF:** Profile 3B PRT latency and identify overhead reduction targets.

Focus:
- Separate sidecar load, callback time, kernel time, graph dispatch, and wall time
- Do not change quality logic yet
- No speedup claim unless measured

---

**Tag:** `PRT_PHASE13AE_3B_QUALITY_CHECKPOINT`  
**Tagged at:** `20d7911ed2a84d410e0baa4bce5645d38941e639`  
**Tagged on:** `experimental/prt-phase13-model-generalization`