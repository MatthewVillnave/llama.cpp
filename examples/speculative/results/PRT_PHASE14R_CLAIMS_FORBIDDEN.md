# PRT Phase 14R — Forbidden Claims

The following claims are NOT supported by experimental evidence and must not be stated. This list is exhaustive for the Phase 14 series.

## Universal Speedup Claims

1. ❌ Do not claim PRT provides universal speedup over native llama.cpp.
2. ❌ Do not claim PRT is faster than native in all configurations.
3. ❌ Do not claim INT8 PRT always beats native throughput.

## Production Readiness Claims

4. ❌ Do not claim production readiness.
5. ❌ Do not claim deployment readiness.
6. ❌ Do not claim PRT is ready for real-world inference workloads.
7. ❌ Do not claim this replaces native llama.cpp generally.

## Model / Size Claims

8. ❌ Do not claim larger-than-7B success (13B, 70B, etc. are entirely untested).
9. ❌ Do not claim results apply to all model families (only Qwen2.5 Q4_K_M has been tested).
10. ❌ Do not claim results apply to all quantization levels (only Q4_K_M has been tested).
11. ❌ Do not claim results apply to all Qwen2.5 variants or versions.

## Task / Prompt Claims

12. ❌ Do not claim all-task equivalence — only measured prompt suites are supported.
13. ❌ Do not claim exact equivalence beyond the tested prompts.
14. ❌ Do not claim zero degradation beyond measured prompts.
15. ❌ Do not claim long-context stability — contexts beyond the measured `n=40–160` range are untested.

## Historical Claims

16. ❌ Do not claim Phase 12 speedup as current evidence — that result was not reproduced and was retired.
17. ❌ Do not claim float32 sidecars were viable as a speed solution.

## Platform / Hardware Claims

18. ❌ Do not claim GPU comparison (measurements are CPU-only).
19. ❌ Do not claim results generalize to all CPU setups.
20. ❌ Do not claim results apply outside this measured CPU setup (Dell OptiPlex 7010 / i5-13500T / 15 GiB RAM).

## Architecture Claims

21. ❌ Do not claim the sidecar approach generalizes to all layers or all model architectures.
22. ❌ Do not claim this is a proven inference optimization without further validation.

---

*All claims not listed in `PRT_PHASE14R_CLAIMS_ALLOWED.md` are implicitly forbidden unless explicitly added to the allowed list with corresponding evidence.*
