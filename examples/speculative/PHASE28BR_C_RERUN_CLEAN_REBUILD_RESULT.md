# Phase 28BR-C-RERUN: Clean Rebuild 28BR-B Test C

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**HEAD:** `779b28890` (Phase 28BR-C: add runtime decoder integration forensics)
**Timestamp:** `2026-05-24T07:26 EDT`
**Result:** `FAIL_NONFINITE_AFTER_CLEAN_REBUILD`
**Classification:** `CACHE_CORRUPTION_AFTER_DECODE`

---

## Rebuild

Clean object purge and rebuild:

```bash
cd build
rm -f src/CMakeFiles/llama.dir/__/examples/speculative/prt_trit_decode.cpp.o \
      src/CMakeFiles/llama.dir/__/examples/speculative/prt_trit_decode.cpp.o.d
cmake --build . --target llama
```

Evidence:

- `prt_trit_decode.cpp` was recompiled.
- `libllama.so` was relinked.
- CMake reported `ggml commit: 779b28890`.

## 28BR-B Test C Rerun

Command:

```bash
timeout 120s ./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p 'Hi' -n 1 -t 4 --no-display-prompt \
  --enable-prt-sidecar-pager \
  --prt-mode 5700 \
  --prt-sidecar-manifest /tmp/phase28bo_layer0_multi_family/manifest.json \
  --prt-sidecar-dir /tmp/phase28bo_layer0_multi_family/ \
  --prt-sidecar-budget-mb 512 \
  --prt-sidecar-apply \
  --prt-sidecar-apply-layer 0 \
  --prt-sidecar-apply-family attn_out \
  --prt-sidecar-shadow-contrib
```

The command hit the 120s timeout because `llama-cli` remained in interactive/chat prompt mode, but the 28BR-B Test C shadow lines were emitted before timeout.

## Runtime Evidence

```text
[PRT-APPLY-SHADOW] il=0 family=attn_out layer_match=1 family_match=1 decoded_views=1 app_attempts=1 app_success=1 sidecar_math_influenced_output=0
[PRT-CONTRIB-SHADOW] il=0 family=attn_out X_synthetic=I_KK R=[896x896] Y=[896x896] abs_sum=4.009560e+05 max_abs=1.000000e+00 nan=896 inf=896 finite=0
[PRT-PAGER-HOOK-TRIT] il=0 family=attn_out rows=896 cols=896 block_rows=32 block_cols=48 n_scales=532 payload_offset=32 scale_offset=301088 raw_bytes=303216 decoded_f32_bytes=3211264 raw_bytes_cast_to_float=0 sidecar_math_influenced_output=0
```

## Required Gate

| Gate | Required | Rerun |
|------|----------|-------|
| decoded R finite | true | false at runtime contribution scan |
| Y finite | true | false |
| nan | 0 | 896 |
| inf | 0 | 896 |
| sidecar_math_influenced_output | false | false |

## Classification

`BUILD_STALE_OBJECT / PSEUDO_NAN_FROM_STALE_DECODER` is rejected for this rerun because the decoder object was explicitly purged and rebuilt, and `libllama.so` was relinked at `779b28890`.

`CONTRIBUTION_LOOP_BUG` is also unlikely: the 28BR-B synthetic contribution path does not allocate or compute a separate Y buffer. It scans the cached `R` buffer directly because `X = I` and therefore `Y = R`. For finite ternary `R` values (`-1, 0, +1`), the loop cannot create exactly `nan=896, inf=896`.

**Current classification:** `CACHE_CORRUPTION_AFTER_DECODE`.

More precise wording: standalone decode remains finite, but the cached runtime `R` buffer is nonfinite by the time `prt_shadow_contribution_synthetic()` scans it. The next forensic step should scan immediately after `prt_decode_cached()` copies into cache and immediately before contribution, without clamping, injection, or output mutation.

## Boundary

No true residual injection was attempted.
No clamping was added.
No quality, speed, or runtime correctness claims are made.
