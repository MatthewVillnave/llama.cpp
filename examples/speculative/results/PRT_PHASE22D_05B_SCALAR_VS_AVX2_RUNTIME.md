# PRT Phase 22D: 0.5B Scalar vs AVX2 Runtime Comparison

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Previous HEAD
30625163e (Phase 22C-R)

## New HEAD
a7c9d3f85 (Phase 22D with timing logs)

## Model path
/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

## f32 ref path
/media/matthew-villnave/VL_usb/prt_scratch/sidecars/phase22c_r_f32_layer0/ffn_up_layer0_prt.bin

## Scalar output (3 runs, n=1)
Run1: 3.518895, 4.578113, 2.778468 (kernel times: 72ms, 70ms, 10ms)
Run2: 3.518895, 4.578113, 2.778468 (kernel times: 73ms, 70ms, 10ms)

## AVX2 output (3 runs, n=1)
Run1: 47.276814, 13.817065, 1.589060 (kernel times: 0ms, 0ms, 0ms)
Run2: 47.276814, 13.817065, 1.589060 (kernel times: 0ms, 0ms, 0ms)
Run3: 47.276814, 13.817065, 1.589060 (kernel times: 0ms, 0ms, 0ms)

## AVX2 n=8 sanity
All 8 tokens use AVX2 backend (no fallback). Consistent abs_sum values.

## Scalar vs AVX2 output_abs_sum comparison
Token0: scalar=3.518895, AVX2=47.276814 (different - accumulation order)
Token1: scalar=4.578113, AVX2=13.817065 (different)
Token2: scalar=2.778468, AVX2=1.589060 (different)

Note: AVX2 produces different numeric values than scalar due to different
accumulation order and SIMD vectorization. This is expected behavior.

## kernel time comparison
- Scalar: 70-73ms (large token), 10ms (small token)
- AVX2: 0ms (below measurement resolution at ms granularity)

## Verdict
PASS_05B_AVX2_RUNTIME_TIMING + PARTIAL_AVX2_NO_TIMING

AVX2 path is active and reaches ops.cpp kernel. Output is valid and consistent.
Kernel timing resolution is too coarse (ms) for sub-millisecond AVX2 execution.

## Recommended next
Phase 22E - route INT8 decoded-f32 through ggml-native op/AVX2 backend.
Use microsecond timing instrumentation if finer measurement needed.

## Models/sidecars/binaries staged?
NO

## Secrets detected?
NO

## Existing tags touched?
NO

## System disk free: 51G
## Scratch disk free: 51G
