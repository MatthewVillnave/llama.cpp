# PRT Packed-Sidecar Research Summary

## Status

Research prototype. Not production-ready.

---

## Summary

PRT is an experimental sidecar-backed active replacement path for transformer FFN_UP computation in llama.cpp. It routes selected FFN_UP layers through external weight files (sidecars) at runtime instead of using the model's native stored weights. The goal is to test whether a different weight representation can preserve output quality while improving CPU inference economics.

---

## Key Finding

Float32 sidecars preserved quality but were too memory-heavy. Packed INT8 sidecars (~4× smaller) preserved tested quality and recovered near-native throughput on measured Qwen2.5 CPU runs.

---

## Validation Snapshot

| Model | Quality | Throughput |
|-------|---------|------------|
| Qwen2.5-0.5B (Q4_K_M) | 8/8 exact matches | INT8 ~1.81× faster than float32 PRT |
| Qwen2.5-3B (Q4_K_M) | 8/8 semantic, 0 degradations | 1.000× native avg (10 runs) |
| Qwen2.5-7B (Q4_K_M) | 8/8 exact or semantic, 0 degradations | 0.993× native avg (8-prompt suite) |

All results on consumer CPU setup, Qwen2.5 Q4_K_M quantization, measured prompt suites only.

---

## Claim Boundary

This is a measured research checkpoint. The following are NOT supported:

- ❌ Universal speedup claim
- ❌ Production readiness claim
- ❌ GPU comparison
- ❌ Larger-than-7B generalization
- ❌ All-task equivalence
- ❌ Deployment claim
- ❌ Results outside measured CPU setup

Safe wording: "on the measured CPU setup," "in tested prompt suites," "near-native throughput," "research checkpoint."

---

## Reproducibility Status

Checkpoint commits and result documents exist on the `experimental/prt-phase14a-packed-sidecars` branch. Model files and sidecar datasets are not included in this package.

The evidence chain:
- Phase 13 (float32): correctness valid, throughput negative
- Phase 14 (INT8): correctness valid, throughput near-native
- Checkpoints tagged at 3B and 7B

A full reproducibility package (model downloads, sidecar generation scripts, run harnesses) is not yet prepared for public distribution.

---

## Project Structure

```
llama.cpp/
  examples/speculative/
    public/              ← public-facing materials (this folder)
      PRT_PUBLIC_X_THREAD.md
      PRT_PUBLIC_ARTICLE.md
      PRT_PUBLIC_README_SUMMARY.md
      PRT_PUBLIC_ONE_PAGE.md
      PRT_PUBLIC_CLAIMS_BOUNDARY.md
    results/             ← internal lab results
      PRT_PHASE14R_LAB_WRITEUP.md
      PRT_PHASE14R_CLAIMS_ALLOWED.md
      PRT_PHASE14R_CLAIMS_FORBIDDEN.md
      PRT_PHASE14Q_7B_INT8_FULL_VALIDATION_CHECKPOINT.md
      PRT_PHASE14G_3B_INT8_REPEATABILITY_CHECKPOINT.md
      ... (full Phase 14 series)
```

---

## Next Steps

1. Longer-context / larger-n stability validation
2. INT4 sidecar prototype
3. Native ggml/backend integration
4. Optimization toward native-beating throughput
5. Broader benchmark suites

---

## Contact / Citations

Results are tagged in the `experimental/prt-phase14a-packed-sidecars` branch of the llama.cpp fork. Cite the branch and relevant phase commits when referencing.
