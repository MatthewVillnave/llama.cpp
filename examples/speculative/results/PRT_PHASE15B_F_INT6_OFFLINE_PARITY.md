# PRT Phase 15B-F — INT6 Offline Parity Prototype

## Verdict

**PASS_INT6_OFFLINE_PARITY**

All 7 selected layers pass both thresholds (WC ≥ 0.995, MC ≥ 0.995). INT6 per-row with range [-31,+31] preserves high-fidelity parity across all tested layers. The gap vs INT8 is small (~0.001–0.002 WC), and the improvement vs INT4 is large (+0.014–0.022 WC).

## Context

Phase 15B-E tested per-row INT4 and got a clean NO_GO (WC ~0.976–0.986, all layers below 0.99). The issue was the per-row scheme with only 4 bits — too lossy for FFN_UP weights. INT6 adds 2 bits (range [-31,+31] vs [-7,+7]), which should give meaningfully better fidelity while still being smaller than INT8 if packed.

## INT6 Format

- **Signed range:** [-31, +31]
- **Scale formula:** `scale[j] = row_max / 31.0` (per-row, same scheme as INT8/INT4)
- **Packing (probe only):** unpacked int8 storage for offline parity (1 byte per value). **Runtime format not yet designed.**
- **Probe orientation:** [ffn=18944, hidden=3584]
- **Probe format:** `[FFN*HIDDEN int8 bytes (INT6 values)][FFN*4 float32 scales]`
- **Temp output:** `/tmp/prt_sidecars_7b_int6_phase15b_probe/` (7 probe files, 65MB each)
- **Note:** For actual runtime, INT6 should be packed (~51MB/layer vs INT8's 68MB/layer). This phase only tested parity, not packing.

## Selected-Layer Results

| Layer | Finite | INT6 WC | INT6 cos_min | INT8 WC | Gap WC | Gap MC | vs INT4 ΔWC | Verdict |
|-------|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | ✅ | 0.997873 | 0.997419 | 0.999791 | 0.0019 | 0.0014 | +0.0154 | ✅ PASS |
| 1 | ✅ | 0.998271 | 0.998171 | 0.999869 | 0.0016 | 0.0007 | +0.0224 | ✅ PASS |
| 10 | ✅ | 0.999153 | 0.999112 | 0.999949 | 0.0008 | 0.0008 | +0.0149 | ✅ PASS |
| 11 | ✅ | 0.999186 | 0.999146 | 0.999951 | 0.0008 | 0.0008 | +0.0146 | ✅ PASS |
| 15 | ✅ | 0.998973 | 0.998912 | 0.999939 | 0.0010 | 0.0010 | +0.0179 | ✅ PASS |
| 20 | ✅ | 0.999037 | 0.998986 | 0.999942 | 0.0009 | 0.0010 | +0.0169 | ✅ PASS |
| 27 | ✅ | 0.999248 | 0.999225 | 0.999955 | 0.0007 | 0.0007 | +0.0136 | ✅ PASS |

**Summary:** Min WC = 0.997873 (L0) | Min MC = 0.997419 (L0) | PASS: 7, MAYBE: 0, NO_GO: 0

## INT6 vs INT8 vs INT4

| Metric | INT4 | INT6 | INT8 |
|--------|:----:|:----:|:----:|
| Min WC | 0.9759 | **0.9979** | 0.9998 |
| Min MC | 0.9750 | **0.9974** | 0.9996 |
| Size (unpacked) | 34 MB/layer | 65 MB/layer | 65 MB/layer |
| Size (packed estimate) | 34 MB/layer | **~51 MB/layer** | 68 MB/layer |
| Passes thresholds | 0/7 | **7/7** | 7/7 |
| vs GGUF fidelity | ❌ Too lossy | ✅ Good | ✅ Excellent |

**Key observations:**
- INT6 closes ~90% of the gap between INT4 and INT8
- INT6's weakest (L0: WC=0.9979) is still better than INT4's best (L27: WC=0.9856)
- INT6 vs INT8 gap is small (~0.001–0.002) and consistent across all layers
- If INT6 were packed (3 values per 2 bytes), it would be ~25% smaller than INT8 with much better quality than INT4

## Optional All-Layer Sweep

Skipped. Selected-layer results (7/7 PASS) were definitive. No need to spend time on all 28 layers when the verdict is already clear.

## Interpretation

### Is INT6 viable offline?
**Yes.** All selected layers pass the 0.995 threshold for both WC and MC. The scheme is sound.

### Which layer is weakest?
**Layer 0** — WC=0.997873, MC=0.997419. But even L0 passes the threshold. The spread across layers is small (0.998–0.999), indicating consistent behavior rather than a layer-specific issue.

### Is runtime canary justified?
**Yes, but with a caveat.** INT6 passes offline thresholds, but the gap vs INT8 (~0.001–0.002 WC) means runtime quality will be measurably worse than INT8. Whether that's *acceptable* depends on the size/quality tradeoff.

### Does INT6 look worth the extra format/runtime complexity?
**Yes, more than INT4.** INT4 required ~0.018 WC degradation vs INT8 but only saved ~50% size. INT6 requires ~0.001 WC degradation vs INT8 with potential for ~25% size savings (if packed). The efficiency ratio is much better.

### Is groupwise INT4 still worth testing?
**Maybe.** Groupwise (block-based) INT4 would be more efficient with the bit budget than per-row INT4. But INT6 is already showing good results — the marginal gain from groupwise over INT6 might not justify the additional complexity. This could be a Phase 15G optional test if time permits.

### Does unpack/dequant overhead risk erasing size gains?
**Depends on packing.** Unpacked INT6 is the same size as INT8 — no gain. Packed INT6 (~51MB/layer) would be ~25% smaller, but requires:
1. Unpack 3 values from 2 bytes
2. Sign-extend from 6 bits to int8
3. Multiply by per-row scale

That's 2–3x the per-element work of INT8. Whether the memory bandwidth savings from 25% smaller sidecars outweighs the CPU unpack overhead is an empirical question for runtime benchmarking.

## Allowed Claims

- INT6 per-row offline parity PASSED on selected 7B FFN_UP layers
- INT6 WC range: 0.9979–0.9992 across selected layers
- INT6 improvement over INT4: +0.014–0.022 WC
- INT8 remains the validated runtime path
- INT4 per-row remains NO-GO

## Forbidden Claims

- INT6 runtime works (untested)
- INT6 speedup (untested)
- INT6 size savings (format not designed/packed yet)
- Production readiness
- Universal speedup
- Larger-than-7B support

## Recommended Next Phase

**Phase 15B-G: Packed INT6 runtime format design + tiny canary.** Before running a runtime canary, we need:
1. Design a packed INT6 format (suggested: 3 values per 2 bytes)
2. Implement dequantization kernel for packed INT6
3. Run one tiny runtime canary to confirm the format works end-to-end

Alternative: **Phase 15B-G: Mixed precision policy** — use INT8 for critical layers and INT6 for others. But that requires knowing which layers are "critical" — which needs runtime profiling.

## Files Created

- `examples/speculative/phase15b_int6_offline_parity.cpp` — INT6 offline parity tool
- Temp INT6 probes at `/tmp/prt_sidecars_7b_int6_phase15b_probe/` (7 files, 65MB each, NOT staged)
- `examples/speculative/results/phase15b_f_int6_offline_parity.json` — JSON metrics

## Safety

| Check | Status |
|-------|--------|
| Models staged | NO |
| Sidecars staged in repo | NO |
| Binaries staged | NO |
| Temp logs staged | NO |
| Secrets detected | NO |
| Existing tags touched | NO |