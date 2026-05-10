# PRT Phase 19H — 0.5B Runtime + Source Quality Checkpoint

## Verdict
**PASS_05B_RUNTIME_SOURCE_QUALITY_CHECKPOINT**

## Branch / Commit

| Item | Value |
|------|-------|
| Branch | `experimental/prt-phase19a-alt-sidecar-backed` |
| Previous HEAD | `6f5efd99f` (Phase 19G) |
| Checkpoint | This commit |

## What Phase 19 Proved

- **0.5B PRT runtime compatibility restored.** End-to-end PRT INT6 overlay path runs on Qwen2.5-0.5B after Phase 19 repairs.
- **24/24 sidecars load.** All 0.5B INT6 sidecars are accepted by the runtime.
- **PRT compute activates.** PRT graph nodes are created for each layer during generation.
- **Tiny canary passes.** Canary strings pass through PRT mode without corruption.
- **PRT bypasses native ffn_up.weight for compute.** During active PRT compute, the PRT custom op reads from sidecar data, not from the native ffn_up.weight GGUF tensor. The overlay is compute-substitutive, not RAM-residency-substitutive.
- **GGUF Q5.0 source is sufficiently close to FP16/BF16.** For the 4 selected tested layers (0, 1, 12, 23) of Qwen2.5-0.5B, GGUF Q5.0 dequantized weights are within ~0.001 cosine of FP16/BF16 source (cosine range: 0.99899–0.99908, max MAE: 0.000718).
- **FP16/BF16-sourced PRT is not a priority for this model.** FP16-sourced INT6 improved layer0 cosine by only ~0.001 over GGUF-sourced INT6. The 2.5x storage cost (952MB vs 380MB) is not justified by this marginal gain on 0.5B tested layers.

## Bugs Fixed

| Bug | Phase | Fix |
|-----|-------|-----|
| 0.5B sidecar size acceptance | 19D | Sidecar format now handles 3.2MB 0.5B weights |
| INT6 activation condition | 19D | g_prt_sidecar_data OR g_prt_int8_data triggers PRT path |
| M/K shape swap | 19D | Fixed ffn_up shape [hidden, ffn] correctly interpreted |
| INT6 format handler in custom op kernel | 19D | INT6 handler added to scalar op dispatch |

## Key Metrics

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-0.5B |
| Layers | 24 |
| Hidden size K | 896 |
| FFN size M | 4,864 |
| Sidecar size (per layer) | 3,288,080 bytes (~3.2MB) |
| Sidecars generated | 24/24 |
| Source parity tested | Layers 0, 1, 12, 23 |
| Min GGUF-vs-FP16 cosine | 0.99899 (layer 12) |
| Max GGUF-vs-FP16 cosine | 0.99908 (layer 1) |
| Max MAE | 0.000718 (layer 23) |
| FP16-sourced INT6 gain | ~0.001 cosine on layer0 vs GGUF-sourced |
| Native 0.5B speed | ~95–97 t/s |
| PRT INT6 scalar speed | ~17–18 t/s |

## Current Limitations

- **INT6 kernel is slow/scalar on 0.5B.** The custom op INT6 path is scalar, yielding ~5x slowdown vs native.
- **No performance claim.** Do not claim speedup — the opposite is observed.
- **No RAM reduction claim.** PRT is overlay/compute-substitutive, not a weight residency replacement.
- **FP16-source conclusion is local.** Only applies to the tested Qwen2.5-0.5B Q5.0 GGUF layers. Do not generalize to larger models or other quantization formats.
- **Not production-ready.** The scalar INT6 kernel is a prototype, not an optimized implementation.
- **Overlay only.** PRT does not replace weights in memory; it computes an additional value that bypasses the native ffn_up.weight.

## Allowed Claims

- 0.5B PRT INT6 overlay runs end-to-end after Phase 19 fixes
- PRT compute bypasses native ffn_up.weight during active compute
- GGUF Q5.0 source is close to FP16/BF16 for selected tested 0.5B layers (cosine 0.999+, MAE < 0.001)
- FP16/BF16 source did not materially improve tested INT6 parity enough to justify extra storage (2.5x) in this tested case
- 24/24 0.5B INT6 sidecars load
- PRT bypass is compute-substitutive, not RAM-residency-substitutive

## Forbidden Claims

Do **not** claim:
- Production readiness
- Universal model support across all sizes/quantizations
- Speedup over native inference
- RAM savings or memory reduction
- FP16 source is never useful for any model/layer
- All layers/models behave the same as 0.5B
- Larger-model source parity extrapolated from this 0.5B test
- GPU performance comparison
- The INT6 kernel is optimized

## Recommended Next Phase

**Phase 19I — INT6 Kernel Performance Audit / AVX2 Path Design**

The representation and source-quality questions are now mostly closed for 0.5B. The next bottleneck is scalar INT6 custom-op performance (~5x slower than native). The next phase should audit the current kernel, identify the scalar bottleneck, and design a vectorized path (AVX2/VNNI) for INT6 matvec.
