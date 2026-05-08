# PRT Phase 14R — Packed-Sidecar Lab Writeup

## Short Version

PRT started as an active FFN_UP replacement path for llama.cpp.

Phase 13 proved that active replacement could preserve output quality, but the float32 sidecar representation was too memory-heavy and speed-negative.

Phase 14 replaced float32 sidecars with packed INT8 sidecars. That preserved tested output quality while recovering near-native CPU throughput on Qwen2.5-3B and Qwen2.5-7B in the measured setup.

---

## What PRT Is

PRT replaces selected FFN_UP computations with a sidecar-backed approximation path. The sidecar encodes the relevant FFN_UP weights in an alternate representation (INT8). The goal is not to change the model's external behavior, but to test whether a different representation can preserve output quality while improving CPU inference economics. Sidecars are loaded at initialization and swapped into the kernel path during generation.

---

## Phase 13: What Worked and What Failed

**What worked:**
- Active replacement — PRT replaced FFN_UP in the ggml graph and generated correct outputs
- Dynamic shape support — flexible kernel handled varying tensor sizes
- Clean llama-cli frontend — no dirty harness confusion, PTY-based output capture
- AVX2 path — vectorized float32 path was functional
- mmap sidecar loading — files were loaded without full RAM copy
- 0.5B quality validated — clean outputs and semantic matches
- 3B quality validated — 8/8 semantic matches, 0 degradations

**What failed (as a speed solution):**
- Float32 sidecars were speed-negative — memory traffic dominated, throughput was materially below native ggml
- Phase 12 speedup claim was not reproduced and was retired as a current claim

**Verdict on Phase 13:** Not a correctness failure. A representation failure. Active replacement was correct; float32 was the wrong sidecar format.

---

## Phase 14: The Packed-Sidecar Pivot

The core hypothesis was that float32 sidecars were the bottleneck: they required too much memory bandwidth relative to the native ggml path.

**The fix:** INT8 per-row quantized sidecars with packed layout:
- INT8 weights + per-row scales stored in one file per layer (no separate scale array)
- Reduces memory traffic roughly 4×
- Per-row scaling preserves offline parity when quantized from float32

**What this tested:**
1. Could the 0.5B float32 INT8 result replicate at 3B and 7B?
2. Did the packed layout scale to larger models?
3. Did runtime quality and throughput hold across the full suite?

**Answer:** Yes to all three.

---

## Evidence Summary

| Model | Validation depth | Quality result | Throughput result | Tag |
|-------|-------------------|----------------|-------------------|-----|
| Qwen2.5-0.5B | Full 8-prompt suite | 8/8 exact matches | INT8 1.81× faster than float32 PRT, near-native | commit 104863026 |
| Qwen2.5-3B | Full 8-prompt + 10-run repeatability | 8/8 semantic matches, 0 degradations | Native parity repeatability: 1.000× avg, 1.002× median (10 runs) | `PRT_PHASE14G_3B_INT8_REPEATABILITY_CHECKPOINT` |
| Qwen2.5-7B | Full 8-prompt + 10-run repeatability | 8/8 exact or semantic matches, 6/8 exact, 0 degradations | 0.993× avg and 0.989× median native throughput (8-prompt suite) | `PRT_PHASE14Q_7B_INT8_FULL_VALIDATION_CHECKPOINT` |

---

## Key Numbers

### 0.5B
- INT8 PRT completed: 8/8
- Exact matches: 8/8
- INT8 vs float32 PRT: ~1.81× faster
- Note: full repeatability not measured at 0.5B

### 3B
- Native repeatability avg: 21.0 t/s
- INT8 PRT repeatability avg: 21.0 t/s
- INT8/native avg ratio: 1.000×
- INT8/native median ratio: 1.002×
- INT8/float32 PRT ratio: 2.467×
- 8/8 semantic matches (10-run)
- 0 quality degradations
- 36/36 sidecars loaded

### 7B
- Full 8-prompt native avg/median: 8.75 / 8.60 t/s
- Full 8-prompt INT8 avg/median: 8.69 / 8.55 t/s
- INT8/native avg ratio: 0.993×
- INT8/native median ratio: 0.989×
- Every individual prompt ratio ≥ 0.976
- 8/8 exact or semantic matches
- 6/8 exact matches
- 0 quality degradations
- 28/28 sidecars loaded
- Fallback limited to force-native layers 11 and 15

---

## What Changed Technically

| Before (Phase 13) | After (Phase 14) |
|-------------------|------------------|
| Float32 sidecars | Packed INT8 sidecars (weights + per-row scales in one file) |
| Separate `.scale` file per layer | Scales appended in same file — no extra file I/O |
| `ffn_up_layer{L}.int8` + `ffn_up_layer{L}.scale` naming | `ffn_up_layer{L}_prt.int8` naming — loader expects this |
| Shape/orientation was audited | Found and fixed: M dimension was read as K in loader, causing corruption |
| 7B scale formula unknown | Fixed: added M=18944 recognition to `cli.cpp` loader |
| No INT8 fallback concept | `--prt-force-native 11,15` limits fallback to two layers only |
| Log output mixed with stdout | Log routed to `--prt-log-file`, stdout kept clean |
| Path fragments in output | Fixed by isolating log routing |

---

## Important Caveats

- Results are on one CPU setup (Dell OptiPlex 7010 / i5-13500T / 15 GiB RAM)
- Results are for Qwen2.5-0.5B, Qwen2.5-3B, and Qwen2.5-7B only (all Q4_K_M quantization)
- Results are for measured prompt suites only — longer contexts and larger-n stability are not yet tested
- This is **not production-ready**
- This is **not a universal speedup claim**
- This is **not a GPU comparison**
- This does not prove all tasks behave identically to native
- Larger-than-7B is entirely untested
- Only Qwen2.5 models at Q4_K_M quantization have been validated

---

## Allowed Claims

See `PRT_PHASE14R_CLAIMS_ALLOWED.md` for the full list. Core allowed claims:

1. PRT active replacement preserved tested output quality on Qwen2.5-0.5B, 3B, and 7B in the measured suites.
2. Float32 sidecars were speed-negative in Phase 13.
3. Packed INT8 sidecars reduced sidecar size by about 4× and recovered throughput.
4. Qwen2.5-3B INT8 PRT reached native-parity throughput (1.000×) in repeatability.
5. Qwen2.5-7B INT8 PRT retained near-native throughput (0.993× avg) in full 8-prompt validation.
6. 28/28 sidecars loaded on 7B; 36/36 on 3B.
7. Fallback limited to force-native layers 11 and 15.

Every claim must be scoped to the measured hardware, model files, prompt suites, and branch state.

---

## Forbidden Claims

See `PRT_PHASE14R_CLAIMS_FORBIDDEN.md` for the full list. Core forbidden claims:

- ❌ No universal speedup
- ❌ No production readiness
- ❌ No GPU comparison
- ❌ No larger-than-7B success
- ❌ No all-task equivalence
- ❌ No Phase 12 speedup reproduction
- ❌ No exact equivalence beyond tested prompts
- ❌ No claim outside this measured CPU setup
- ❌ No INT8 PRT always beats native

---

## What This Means

- PRT's active replacement path is **viable** — output quality is preserved
- Float32 was the **wrong sidecar representation** — memory traffic was the bottleneck
- Packed INT8 sidecars are the **right direction** — near-native throughput while preserving quality
- The architecture **scales from 0.5B to 7B** with the same quality and throughput profile
- Future work: INT4 prototype, longer-context stability, native ggml/backend integration, optimization toward native-beating throughput

---

## Recommended Next Technical Paths

| Rank | Path | Rationale |
|------|------|-----------|
| 1 | Longer-context / larger-n stability | Tests practical deployment range; catches memory/swap degradation |
| 2 | INT4 sidecar prototype | Could halve memory traffic again; requires offline parity first |
| 3 | Native ggml/backend integration | Removes custom callback limits; highest architectural value |
| 4 | Optimization toward native-beating throughput | Highest ceiling; depends on clean 7B validation as baseline |
| 5 | Public article / X thread package | Captures result with claims discipline; low cost, high credibility value |

---

## Final Statement

**PRT Phase 14 validates the packed-sidecar thesis on the measured CPU setup:** INT8 sidecars preserved tested output quality and recovered near-native throughput on Qwen2.5-3B and Qwen2.5-7B. This is a research checkpoint, not a production claim. The Phase 14 series (14A–14Q) demonstrates that the active replacement path works, the packed sidecar format is the right representation, and the quality-throughput trade-off is favorable on the measured setup. INT4, longer-context stability, backend integration, and optimization are the logical next research directions.