# PRT Phase 14R — Allowed Claims

The following claims are supported by experimental evidence from Phase 13 and Phase 14 on `experimental/prt-phase14a-packed-sidecars`. Every claim must be scoped to the measured hardware (Dell OptiPlex 7010 / i5-13500T), model files (Qwen2.5-0.5B/3B/7B Q4_K_M), prompt suites, and branch state (`f9583337d` / `PRT_PHASE14Q_7B_INT8_FULL_VALIDATION_CHECKPOINT`).

## Active Replacement Quality

1. PRT active replacement preserved tested output quality on Qwen2.5-0.5B, Qwen2.5-3B, and Qwen2.5-7B in the measured suites.
2. On Qwen2.5-3B, INT8 PRT produced 8/8 semantic matches and 0 quality degradations across the full 8-prompt suite.
3. On Qwen2.5-7B, INT8 PRT produced 8/8 exact or semantic matches and 0 quality degradations across the full 8-prompt suite.

## Sidecar Representation

4. Float32 sidecars were speed-negative in Phase 13 — memory traffic dominated and throughput was materially below native ggml.
5. Packed INT8 sidecars reduced per-layer sidecar memory footprint by about 4× compared to float32.
6. The packed layout (combined int8 weights + per-row scales in one file) scales from 0.5B to 7B with correct shape handling.

## Throughput

7. Qwen2.5-3B INT8 PRT reached repeatable native-parity throughput on the measured CPU setup (1.000× avg, 1.002× median in 10 independent runs).
8. Qwen2.5-7B INT8 PRT retained near-native throughput on the measured CPU setup (0.993× avg, 0.989× median across the 8-prompt suite).
9. Qwen2.5-0.5B INT8 PRT ran at approximately 1.81× the speed of float32 PRT in the measured canary.
10. INT8 PRT was approximately 2.467× faster than float32 PRT at 3B scale.

## Sidecar Loading

11. Sidecars loaded fully and correctly on all validated model sizes:
    - Qwen2.5-0.5B: 8/8 sidecars loaded (commit `104863026`)
    - Qwen2.5-3B: 36/36 sidecars loaded (Phase 14E/14G)
    - Qwen2.5-7B: 28/28 sidecars loaded (Phase 14K–14Q)
12. Fallback to native FFN_UP was limited to force-native layers 11 and 15 in all measured runs.
13. No path-fragment or debug-log contamination appeared in output text when `--prt-log-file` was used.

## Checkpoint Evidence Chain

14. Phase 14 evidence is preserved in tagged commits and result documents on the `experimental/prt-phase14a-packed-sidecars` branch.
15. The 3B checkpoint is tagged `PRT_PHASE14G_3B_INT8_REPEATABILITY_CHECKPOINT` (`0978d957e`).
16. The 7B checkpoint is tagged `PRT_PHASE14Q_7B_INT8_FULL_VALIDATION_CHECKPOINT` (`f9583337d`).
17. The Phase 14 series (14A–14Q) represents increasing-depth validation from 0.5B to 7B.

## What the Evidence Shows

18. PRT's active replacement path is viable — output quality is preserved when sidecar representation is corrected.
19. Float32 was the wrong sidecar format — INT8 recovers the right quality-throughput trade-off.
20. The packed-sidecar thesis is validated within the measured setup.

---

*Every claim above must be accompanied by the scope qualifier: on this measured CPU setup, for the measured Qwen2.5 models, for the measured prompt suites.*
