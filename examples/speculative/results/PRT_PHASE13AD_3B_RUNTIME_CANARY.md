# PRT Phase 13AD — 3B Runtime Canary

**Date:** 2026-05-07  
**Verdict:** PASS_3B_RUNTIME_CANARY ✅  
**Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf  
**Branch:** `experimental/prt-phase13-model-generalization`

---

## Verdict

3B PRT runtime is fully operational. Single-prompt canary passed. 4-prompt quality suite passed 4/4 semantic matches, 3/4 exact matches (prompt 4 differs by word choice but semantically equivalent), zero quality degradations. AVX2 kernel confirmed active on 3B model. Ready for full 8-prompt validation.

---

## Model and Sidecar Metadata

| Field | Value |
|-------|-------|
| **Model path** | `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf` |
| **Model size** | 1.84 GB |
| **Layers** | 36 |
| **Hidden** | 2048 |
| **FFN** | 11008 |
| **Sidecar dir** | `/tmp/prt_sidecars_3b/` |
| **Sidecars generated** | 36 |
| **Sidecar bytes/layer** | 90,177,536 |
| **Total sidecar storage** | 3.1 GB |
| **Sidecars loaded per run** | 36/36 ✅ |

---

## Single-Prompt Canary

| Metric | Native | PRT |
|--------|--------|-----|
| **Output** | "The capital of France is Paris." | "The capital of France is Paris." |
| **Exit code** | 0 | 0 |
| **Elapsed** | 2.1s | 9.84s |
| **Gen tok/s** | 20.8 | 8.8 |
| **Clean output** | ✅ | ✅ |
| **Semantic match** | — | ✅ |
| **Exact match** | — | ✅ |

**PRT log evidence:**
- `PRT_SHAPE: n_layer=36 M=2048 N=11008` ✅
- `total_calls=272` (34 active layers × 8 calls/layer) ✅
- `kern_total=3265.7ms` — all AVX2 ✅
- `fallback_calls=0` ✅
- `force-native layers: 11 15` ✅
- `compile_flags=-mavx2 -mfma (LLAMA_PRT_AVX2)` ✅

---

## 4-Prompt Quality Suite

| Prompt | Native Output | PRT Output | Exact | Semantic | Notes |
|--------|--------------|------------|-------|----------|-------|
| 1. "The capital of France is" | "Paris." | "Paris." | ✅ | ✅ | |
| 2. "Write a Python function..." | "Certainly! Below is... reverse_list(input_list):" | "Certainly! Below is... reverse_list(input_list):" | ✅ | ✅ | Code plausible |
| 3. "Return JSON with keys..." | `{"name": "Qwen", "status": "active"}` | `{"name": "Qwen", "status": "active"}` | ✅ | ✅ | Valid JSON |
| 4. "Explain CPU inference..." | "...using the Central Processing Unit to process data..." | "...using a CPU to execute inference tasks, which involves processing and analyzing data..." | ❌ | ✅ | Semantic equivalent, minor wording difference |

**Summary:** 3/4 exact matches, 4/4 semantic matches, 0 quality degradations ✅

---

## Per-Prompt PRT Evidence

| Prompt | PRT_SHAPE | Calls | Kern ms | Sidecar Load ms | AVX2 | Fallback |
|--------|-----------|-------|---------|-----------------|------|---------|
| 1 | n_layer=36 M=2048 N=11008 | 272 | 3265.7 | 4883 | ✅ | 0 |
| 2 | n_layer=36 M=2048 N=11008 | 1360 | 6394.4 | ~4800 | ✅ | 0 |
| 3 | n_layer=36 M=2048 N=11008 | 748 | 4772.8 | ~4800 | ✅ | 0 |
| 4 | n_layer=36 M=2048 N=11008 | 1156 | 5751.8 | ~4800 | ✅ | 0 |

---

## Timing (Informational Only — No Speedup Claim)

| Metric | Native | PRT | Ratio |
|--------|--------|-----|-------|
| Gen tok/s (prompt 1) | 20.8 | 8.8 | 2.36× slower |
| Wall time (prompt 1) | 2.1s | 9.84s | 4.69× slower |
| Gen tok/s (prompt 3 JSON) | ~19.5 | ~8.0 | 2.44× slower |
| Wall time (prompt 3 JSON) | ~2.8s | ~11.9s | 4.25× slower |

**Note:** PRT is slower than native on 3B (as expected with current implementation). This is consistent with 0.5B behavior. No speedup claim for this phase.

---

## Path Fragment False Positive

The `contains_path_fragment=True` flag in the PTY runner output is a **false positive** caused by the string `"Exiting..."` appearing in the tail. This is llama-cli shutdown text, not PRT debug contamination. All actual generated outputs are clean with no path fragments.

---

## What This Phase Proves

- 3B sidecars load and execute correctly at runtime ✅
- 3B active PRT produces clean outputs matching native ✅
- 3B active PRT preserves quality across diverse prompts (factual, code, JSON, definition) ✅
- AVX2 kernel fully active on 3B (272-1360 calls per prompt) ✅
- Zero fallback calls across all 4 prompts ✅
- PRT_SHAPE now correctly reports `n_layer=36 M=2048 N=11008` ✅
- No debug contamination in generated output ✅
- JSON validity preserved on code prompt ✅
- Code plausibility preserved ✅

---

## What This Phase Does NOT Prove

- ❌ No full 8-prompt 3B validation yet
- ❌ No 3B speedup claim
- ❌ No production readiness
- ❌ No larger model generalization beyond Qwen2.5-3B

---

## Recommended Next

**Phase 13AE:** Full 8-prompt 3B quality validation with timing baseline:
1. Run full 8-prompt native baseline
2. Run full 8-prompt PRT active
3. Confirm all 8 prompts quality-preserved
4. Document exact timing for 3B PRT vs native
5. Compare slowdown ratio to 0.5B results

---

## Safety

- No `.gguf`/`.bin`/`.safetensors` staged ✅
- No secrets in any changed file ✅
- PRT logs in `/tmp/` ✅

**Tag:** `PRT_PHASE13AD_3B_RUNTIME_CANARY`