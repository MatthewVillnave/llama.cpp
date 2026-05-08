# PRT Phase 15B-E — INT4 Offline Parity with Fixed Extraction

## Verdict

**NO_GO_INT4_OFFLINE_PARITY**

INT4 per-row quantization with range [-7, +7] degrades too far vs INT8. All 7 selected layers fail to meet the 0.999 threshold. Even the 0.980 threshold is borderline — 2/7 layers fail outright, 5/7 are MAYBE at best.

## Context

Phase 15B-A found INT4 promising but unvalidated due to GGUF extraction being blocked. Phase 15B-C/D fixed the GGUF extraction and produced fresh unique INT8 sidecars. This phase reruns INT4 offline parity with those fresh sidecars as references.

## INT4 Format

- **Signed range:** [-7, +7]
- **Scale formula:** `scale[j] = row_max / 7.0` (per-row, same as INT8 but divided by 7 instead of 127)
- **Packing (probe files only):** 2 int4 values per byte (low nibble first, high nibble second), offset = (value + 8) to make unsigned
- **Orientation:** [ffn=18944, hidden=3584]
- **Probe format:** `[FFN*HIDDEN/2 packed bytes][FFN*4 float32 scales]`
- **Temp output:** `/tmp/prt_sidecars_7b_int4_phase15b_probe/` (7 probe files, 33MB each)
- **Note:** Runtime format NOT defined — this was purely offline parity probe

## Selected-Layer Results

| Layer | Finite | INT4 WC | INT4 cos_min | INT8 WC | WC Gap | MAE | Zero Frac | Sat7 Frac | Verdict |
|-------|:------:|:-------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | ✅ | 0.982504 | 0.979752 | 0.999791 | 0.0173 | 0.0028 | 0.0023 | 0.0006 | ❌ NO_GO |
| 1 | ✅ | 0.975859 | 0.975003 | 0.999869 | 0.0240 | 0.0030 | 0.0024 | 0.0005 | ❌ NO_GO |
| 10 | ✅ | 0.984259 | 0.983861 | 0.999949 | 0.0157 | 0.0030 | 0.0001 | 0.0006 | ⚠️ MAYBE |
| 11 | ✅ | 0.984582 | 0.984159 | 0.999951 | 0.0154 | 0.0051 | 0.0000 | 0.0006 | ⚠️ MAYBE |
| 15 | ✅ | 0.981068 | 0.980741 | 0.999939 | 0.0189 | 0.0037 | 0.0001 | 0.0006 | ⚠️ MAYBE |
| 20 | ✅ | 0.982185 | 0.981435 | 0.999942 | 0.0178 | 0.0059 | 0.0001 | 0.0006 | ⚠️ MAYBE |
| 27 | ✅ | 0.985633 | 0.985311 | 0.999955 | 0.0143 | 0.0041 | 0.0000 | 0.0006 | ⚠️ MAYBE |

**Summary:** Min INT4 WC = 0.975859 (L1) | Min INT4 MC = 0.975003 (L1) | PASS: 0, MAYBE: 5, NO_GO: 2

## INT4 vs INT8 Degradation

The degradation from INT8 to INT4 is significant and consistent across all layers:

- **WC gap:** 0.0143 to 0.0240 (INT4 loses 1.4–2.4% cosine similarity)
- **Mean WC gap:** ~0.018 (INT4 is ~1.8% worse than INT8)
- **Consistent pattern:** All 7 layers show similar degradation — not a weak-layer issue, a fundamental scheme issue

The problem is not layer-dependent — it's the INT4 quantization scheme itself. With only 4 bits (range [-7,+7]), you're representing the same continuous weights as INT8 with 8 bits (range [-127,+127]). The resolution loss is ~2x in quantization levels.

## INT4 Size Estimate

- Per layer: 34,023,424 bytes (~34 MB) — 50% of INT8's 65 MB
- Total for 28 layers: ~952 MB (vs INT8's ~1.8 GB)
- **Size savings: ~47%** (roughly 2x compression)

However, size savings are irrelevant if quality is unacceptable.

## Interpretation

### Is INT4 viable offline?
**No.** With WC ~0.976–0.986 and MC ~0.975–0.985, INT4 per-row is too lossy for this FFN_UP use case. The 0.98 threshold is violated on 2/7 layers (L0, L1). The 0.99 threshold is violated on all 7.

### Which layer is weakest?
**Layer 1** — WC=0.975859, MC=0.975003. But the spread is small (0.976–0.986) — it's a scheme issue, not a layer issue.

### Is runtime canary justified?
**No.** The offline results are clean enough to say: INT4 per-row [-7,+7] is not suitable. There's no edge case that a runtime canary would fix.

### Would INT6, groupwise INT4, or mixed INT8/INT4 be better?

**INT6:** Probably better than INT4, worse than INT8. Would need a separate offline test. INT6 = [-31, +31] (6 bits signed) — still a significant gap vs INT8's 127 levels.

**Groupwise INT4 (per-block):** Grouping weights into blocks (e.g., 32 or 64 elements per group) with shared scale would better utilize the 4-bit range and reduce per-row quantization loss. This is how Q4_K actually works — block-based scales. Could be worth trying offline.

**Mixed INT8/INT4:** Could use INT8 for critical layers and INT4 for others. But the question is which layers — and why would INT4 work there if it's failing everywhere offline?

**Per-row INT4 verdict:** The issue is the per-row scale inefficiency — with 18944 rows each having a scale, you're using 4 bits to represent a continuous value with very different magnitudes across rows. Groupwise would be more efficient with the same bit budget.

### Does unpack/dequant overhead risk erasing size gains?
**Yes.** Even if you fix the quality, INT4 requires:
1. Packed byte → two nibbles
2. Nibble → signed int4 → offset removal
3. Multiply by per-row scale

INT8 requires: int8 → multiply by per-row scale (1 step).

The unpack overhead is 2–3x per element. If the size savings is only 2x, the CPU-side overhead could easily cancel the memory bandwidth savings.

## Allowed Claims

- INT4 per-row offline parity on selected 7B FFN_UP layers is NO_GO
- INT4 WC range: 0.976–0.986 (all layers below 0.99)
- INT8 remains the validated runtime path
- INT4 degradation gap vs INT8: ~0.018 mean WC
- No runtime tests were performed

## Forbidden Claims

- INT4 runtime works
- INT4 speedup
- INT4 generation quality
- Production readiness
- Universal speedup
- Larger-than-7B support
- GPU comparison

## Recommended Next Phase

**Stop INT4 per-row and try groupwise INT4 offline** — The fundamental problem is per-row quantization is too inefficient with 4 bits. Block/groupwise quantization (e.g., 32 or 64 elements per scale) would better utilize the bit budget and might achieve WC ≥ 0.999.

**Alternative:** Try INT6 offline parity — 6 bits should give more headroom than INT4 while still being smaller than INT8.

**Do not:** Run INT4 runtime canary. The offline results are definitive.

## Files Created

- `examples/speculative/phase15b_int4_offline_parity.cpp` — INT4 offline parity tool
- Temp INT4 probes at `/tmp/prt_sidecars_7b_int4_phase15b_probe/` (7 files, NOT staged)
- `examples/speculative/results/phase15b_e_int4_offline_parity.json` — JSON metrics

## Safety

| Check | Status |
|-------|--------|
| Models staged | NO |
| Sidecars staged in repo | NO |
| Binaries staged | NO |
| Temp logs staged | NO |
| Secrets detected | NO |
| Existing tags touched | NO |