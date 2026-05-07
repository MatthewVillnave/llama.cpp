# PRT Phase 13AB — 0.5B Post-Fix Checkpoint

**Date:** 2026-05-07  
**Commit:** 9a72dd479  
**Branch:** `experimental/prt-phase13-model-generalization`  
**Verdict:** PASS_SPEED_IMPROVED_BUT_STILL_SLOWER

---

## Verdict

0.5B post-fix PRT quality is confirmed.  
0.5B post-fix PRT timing is improved but still slower than native.

---

## What This Checkpoint Proves

- Qwen2.5-0.5B active PRT runs through clean llama-cli PTY runner ✅
- AVX2 replacement path is confirmed ✅
- Graph replacement is real ✅
- Custom op calls execute ✅
- AVX2 calls execute 176/176 ✅
- Fallback calls are 0 ✅
- Force-native layers 11 and 15 are respected ✅
- Clean output is preserved ✅
- Exact output match observed in Phase 13AA timing prompt (5/5) ✅
- Full 8-prompt clean quality confirmed in Phase 13Z-R (8/8 exact, 8/8 semantic) ✅
- Post-fix PRT is much faster than the broken/fallback path (2.51× improvement) ✅

---

## What This Checkpoint Does NOT Prove

- ❌ No speedup over native on 0.5B
- ❌ No production readiness
- ❌ No larger-model generalization
- ❌ No 3B validation yet
- ❌ No universal quality claim
- ❌ No throughput claim beyond this exact 0.5B setup

---

## Key Quality Checkpoint — Phase 13Z-R (commit a090c5d04)

| Metric | Result | Pass |
|--------|--------|------|
| Native completed | 8/8 | ✅ |
| PRT completed | 8/8 | ✅ |
| Exact matches | 8/8 | ✅ |
| Semantic matches | 8/8 | ✅ |
| Quality degradations | 0 | ✅ |
| Repetition/collapse | 0 | ✅ |
| JSON valid (native + PRT) | 1/1 + 1/1 | ✅ |
| Code prompts plausible | 2/2 | ✅ |
| PRT_SHAPE logs | 8/8 | ✅ |
| Sidecars loaded | 24/24 × 8 runs | ✅ |
| Replacement/custom-op evidence | Self-contained (176 from PRT-13V-SUMMARY) | ✅ |
| AVX2 evidence | 176/176 | ✅ |
| Fallback calls | 0 | ✅ |

---

## Key Timing Checkpoint — Phase 13AA (commit 9a72dd479)

| Metric | Native | PRT | Ratio |
|--------|--------|-----|-------|
| Avg tok/s (gen) | 94.86 | 49.54 | 1.92× slower |
| Avg wall (s) | 0.651 | 1.007 | 1.55× slower |
| Sidecar load (ms) | — | 119.14 | — |
| Custom-op/kernel total (ms) | — | 276.7 | — |
| Replacement calls | — | 176 | — |
| AVX2 calls | — | 176 | — |
| Fallback calls | — | 0 | ✅ |

**Phase 13T pre-fix comparison:**  
Post-fix PRT improved from 19.7 t/s → 49.5 t/s (2.51× faster)  
Slowdown vs native improved from 4.71× → 1.92×

---

## Allowed Claims

✅ PRT dynamic-shape active path preserves clean generation quality on Qwen2.5-0.5B across the post-fix 8-prompt clean suite.  
✅ The fixed AVX2 path executes real graph replacement with 0 fallback on Qwen2.5-0.5B.  
✅ AVX2/indexing fixes improved PRT performance substantially versus the broken/fallback path.  
✅ On Qwen2.5-0.5B, PRT remains slower than native llama.cpp.  

---

## Forbidden Claims

🚫 Do not claim speedup over native on 0.5B  
🚫 Do not claim production readiness  
🚫 Do not claim 3B or larger-model success  
🚫 Do not claim general model support  
🚫 Do not claim PRT is faster universally  
🚫 Do not claim exact equivalence beyond tested prompts  

---

## Recommended Next Phases

**Phase 13AC:** Generate/validate Qwen2.5-3B sidecars with correct shape (hidden=2048, ffn=11008).

**Phase 13AD:** Run 3B clean quality and timing validation using the fixed AVX2/indexing/log-routing/PTY runner stack.

---

## Safety

- No `.gguf`/`.bin`/`.safetensors` files staged ✅
- No secrets/API keys in any changed file ✅
- PRT log files kept in `/tmp` (not committed) ✅

**Tags:** `PRT_PHASE13AB_05B_POST_FIX_CHECKPOINT`
