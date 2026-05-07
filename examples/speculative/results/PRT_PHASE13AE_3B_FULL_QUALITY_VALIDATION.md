# PRT Phase 13AE — 3B Full Quality Validation

**Date:** 2026-05-07  
**Verdict:** PASS_3B_FULL_QUALITY ✅  
**Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf  
**Branch:** `experimental/prt-phase13-model-generalization`

---

## Verdict

8/8 prompts passed. Native and PRT both completed cleanly on all prompts with exit 0. Semantic match on all 8/8 prompts. No quality degradation, no repetition collapse, no corruption. JSON valid on both native and PRT for prompt 5. Code plausible on both for prompts 2 and 6. AVX2 kernel confirmed on all 8 runs. PRT is ~2.45× slower in generation, ~4× slower in wall time — informational only, no speedup claim.

---

## Model / Sidecar Metadata

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
| **PRT mode** | 5700 |
| **Force-native layers** | 11, 15 |

---

## Quality Table

| # | Prompt | Native Output | PRT Output | Exact | Semantic | Notes |
|---|--------|--------------|-----------|-------|----------|-------|
| 1 | The capital of France is | "Paris." | "Paris." | ✅ | ✅ | Identical |
| 2 | Write a Python function that reverses a list. | "Certainly! Below is... reverse_list(input_list):" | "Certainly! Below is... reverse_list(input_list):" | ✅ | ✅ | Code plausible ✅ |
| 3 | Once upon a time in a | "...there was a kingdom surrounded by lush forests..." | "...there lived a kingdom known for its beauty and prosperity..." | ❌ | ✅ | Both coherent narratives |
| 4 | Explain CPU inference in one sentence. | "...using the Central Processing Unit to process data..." | "...using a CPU to execute inference tasks, which involves processing..." | ❌ | ✅ | Both accurate definitions |
| 5 | Return JSON with keys name and status. | `{"name":"Qwen","status":"active"}` | `{"name":"Qwen","status":"active"}` | ✅ | ✅ | Valid JSON ✅ |
| 6 | The fastest way to sort a list in Python is | "...Python's built-in sort() method and sorted()..." | "...Python provides several sorting methods..." | ❌ | ✅ | Both discuss Python sorting |
| 7 | In two sentences, explain what RAM does. | "...store and quickly access data and instructions..." | "...volatile memory that temporarily stores data..." | ❌ | ✅ | Both accurate RAM descriptions |
| 8 | Complete this phrase: artificial intelligence is | "...driving force behind many technological advancements..." | "...driving force behind the next generation of technological innovation..." | ❌ | ✅ | Both complete the phrase coherently |

**Summary:** 3/8 exact matches, 8/8 semantic matches, 0 quality degradations ✅

---

## Per-Prompt Evidence

| Prompt | PRT_SHAPE | Sidecars | AVX2 | Fallback | Sidecar Load ms |
|--------|-----------|----------|------|----------|-----------------|
| 1 | n_layer=36 M=2048 N=11008 | 36/36 | ✅ | 0 | 4529.74 |
| 2 | n_layer=36 M=2048 N=11008 | 36/36 | ✅ | 0 | 5069.74 |
| 3 | n_layer=36 M=2048 N=11008 | 36/36 | ✅ | 0 | 4377.74 |
| 4 | n_layer=36 M=2048 N=11008 | 36/36 | ✅ | 0 | 4466.85 |
| 5 | n_layer=36 M=2048 N=11008 | 36/36 | ✅ | 0 | 4380.12 |
| 6 | n_layer=36 M=2048 N=11008 | 36/36 | ✅ | 0 | 4491.78 |
| 7 | n_layer=36 M=2048 N=11008 | 36/36 | ✅ | 0 | 4478.31 |
| 8 | n_layer=36 M=2048 N=11008 | 36/36 | ✅ | 0 | 4437.60 |

**AVX2 evidence (all 8 runs):** `[PRT-BUILD] PRT_KERNEL=avx2 (kernel_mode=1)` + `[PRT-BUILD] compile_flags=-mavx2 -mfma (LLAMA_PRT_AVX2)`  
**Force-native evidence (all 8 runs):** `[PRT-11BG] force-native enabled for 2 layers: 11 15`

---

## Timing (Informational — No Speedup Claim)

| Prompt | Native Elapsed | PRT Elapsed | Native Gen | PRT Gen | PRT/N Gen Ratio |
|--------|---------------|-------------|-----------|---------|-----------------|
| 1 | 2.05s | 9.82s | 21.0 t/s | 8.4 t/s | 0.40× |
| 2 | 3.92s | 15.08s | 18.9 t/s | 7.7 t/s | 0.41× |
| 3 | 3.83s | 13.93s | 19.0 t/s | 7.8 t/s | 0.41× |
| 4 | 3.12s | 13.45s | 19.2 t/s | 7.8 t/s | 0.41× |
| 5 | 2.96s | 11.74s | 19.5 t/s | 8.0 t/s | 0.41× |
| 6 | 3.85s | 14.38s | 19.1 t/s | 7.8 t/s | 0.41× |
| 7 | 3.36s | 13.73s | 19.3 t/s | 7.7 t/s | 0.40× |
| 8 | 2.76s | 11.24s | 19.4 t/s | 8.0 t/s | 0.41× |
| **Avg** | **3.23s** | **12.92s** | **19.4 t/s** | **7.9 t/s** | **0.41×** |

**Interpretation:** PRT generation is ~2.45× slower than native. PRT wall time is ~4× slower than native. Sidecar loading adds ~4.5s overhead per run (included in wall time). No speedup claim is made.

---

## Allowed Claims

- Qwen2.5-3B active PRT passed an 8-prompt clean quality validation ✅
- 3B sidecars loaded 36/36 on all 8 runs ✅
- Fixed AVX2 replacement path executed with 0 fallback across all runs ✅
- Quality was preserved on the tested 8 prompts (semantic match 8/8) ✅
- Prompt 5 JSON valid on both native and PRT ✅
- Prompts 2 and 6 code plausible on both native and PRT ✅

## Forbidden Claims

- ❌ No 3B speedup claim
- ❌ No production readiness
- ❌ No universal or general model support claim
- ❌ No larger-than-3B generalization claim
- ❌ No exact equivalence beyond tested prompts

---

## What This Phase Proves

- 3B PRT with active sidecars produces clean, semantically equivalent output on diverse prompt types ✅
- Factual (prompt 1), code (prompts 2, 6), creative narrative (prompt 3), definition (prompt 4), JSON (prompt 5), RAM explanation (prompt 7), phrase completion (prompt 8) — all preserved ✅
- 0 fallback calls across all 8 diverse prompts ✅
- AVX2 kernel confirmed on all 8 runs ✅
- PRT_SHAPE correctly reports `n_layer=36 M=2048 N=11008` on all 8 runs ✅
- No debug contamination, no path fragments, no corruption ✅
- JSON structure validity preserved under active PRT ✅
- Code plausibility preserved under active PRT ✅

---

## What This Phase Does NOT Prove

- ❌ No 3B speedup
- ❌ No production readiness
- ❌ No universal model generalization
- ❌ No larger model compatibility

---

## Recommended Next

**Phase 13AF:** Explore PRT speedup opportunity on 3B:
1. Profile PRT latency breakdown (sidecar load vs per-token overhead vs kernel time)
2. Target the per-token overhead reduction (currently 7.9 t/s vs 19.4 t/s native)
3. Consider: batch pretouch, memory mapping, kernel fusion, or async sidecar load
4. If overhead can be reduced enough, PRT could approach or exceed native speed

---

## Safety

- No `.gguf`/`.bin`/`.safetensors` staged ✅
- No secrets in any changed file ✅
- PRT logs in `/tmp/` ✅

**Tag:** `PRT_PHASE13AE_3B_FULL_QUALITY_VALIDATION`