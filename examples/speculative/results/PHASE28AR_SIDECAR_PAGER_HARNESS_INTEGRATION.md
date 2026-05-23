# Phase 28AR: Sidecar Pager Harness Integration

## A. Branch

`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD

Start HEAD: `0327738c9` (`Phase 28AQ: add disabled sidecar pager stub wiring`)

## C. Harness Path

`examples/speculative/prt_sidecar_pager_harness.cpp`

The harness is standalone and runtime-adjacent only:

- no model generation
- no ggml graph
- no llama.cpp matmul changes
- no `g_sidecars` lookup replacement
- synthetic real-format `.trit` package under `/tmp`

## D. Build Command

```bash
g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -Iexamples/speculative \
  examples/speculative/prt_sidecar_pager.cpp \
  examples/speculative/prt_sidecar_pager_harness.cpp \
  -o /tmp/prt_sidecar_pager_harness
```

Build result: PASS.

Compiler warnings are limited to existing ignored `fread` return-value warnings in `prt_sidecar_pager.cpp`.

## E. Disabled-Mode Harness Result

Command:

```bash
/tmp/prt_sidecar_pager_harness --all
```

Disabled-mode result:

- pager object not created
- no manifest required
- exit path succeeds

Verdict: `PASS_DISABLED_MODE_NOOP`

## F. Enabled-Mode Harness Result

Synthetic package:

- path: `/tmp/prt_harness_28ar`
- layers: 4
- tensor families: `ffn_up`, `ffn_down`, `ffn_gate`, `attn_q`, `attn_output`
- files: real-format `.trit` headers with checksum validation enabled

Enabled-mode result:

- `pager.init()` succeeds
- `activate_layer(0)` succeeds
- `prefetch_layer(1)` is exercised
- valid residual views returned for all five requested families
- missing layer lookup returns null/fallback view
- checksum validation succeeds for loaded `.trit` files

Observed stats:

- `resident_bytes`: 1968800
- `reads`: 1
- `cache_misses`: 1
- `fallbacks`: 1
- `trit_checksum_ok`: 5
- `trit_checksum_fail`: 0

Verdicts:

- `PASS_ENABLED_PAGER_INIT`
- `PASS_RESIDUAL_VIEW_LOOKUP`

## G. Budget Stress Result

Strict budget mode:

- budget: 256 KB
- package: 4 layers, 5 families in manifest
- one layer is larger than the budget
- activation attempts for layers 0..3 are rejected
- `budget_rejects` increments to 4
- resident bytes remain 0

Verdict: `PASS_BUDGET_STRESS`

## H. LRU Result

LRU mode:

- budget: 2500 KB
- one 5-family layer fits
- two resident layers exceed budget
- walking layers 0..3 triggers eviction

Observed stats:

- activated layers: 4
- `lru_evictions`: 3
- `evictions`: 3
- `resident_bytes`: 1968800
- budget cap: 2500 KB
- `budget_rejects`: 0
- `trit_checksum_ok`: 20
- `trit_checksum_fail`: 0

Verdict: `PASS_LRU_HARNESS`

## I. llama-cli Default Behavior Check

Checks:

- `cmake --build build --target llama-cli -j2`: PASS
- `build/bin/llama-cli --help`: new pager flags visible
- `build/bin/llama-cli`: existing `error: --model is required`, no pager/manifest error
- `build/bin/llama-cli --prt-sidecar-manifest /tmp/fake_manifest.json`: existing `error: --model is required`, proving supporting pager flags remain inert without the master flag

No generation was run.

Verdict: `PASS_LLAMA_CLI_DEFAULT_PRESERVED`

## J. Limitations

- Harness accepts equivalent local flags rather than reusing `common_params` directly.
- Pager is linked into the standalone harness only, not active generation/runtime.
- No active PRT sidecar lookup is connected.
- No model files, full-model sidecars, 30B/32B files, generated captures, or runtime speed claims.

Verdict: `PARTIAL_COMMON_PARAM_LINK_DEFERRED`

## K. Recommended Next Phase

Phase 28AS: connect pager to experimental PRT sidecar lookup behind the disabled flag, still with no generation.

## L. Models/Sidecars/F32 Refs Staged?

No.

The synthetic `.trit` package was generated under `/tmp` only and was not staged.

## M. Secrets Detected?

No secrets detected in the changed diff.

The broad repository grep reports existing fixture/docs strings and API-key examples outside the changed diff; Phase 28AR did not introduce secrets.

## N. Tags Touched?

No.

## Verdicts

- `PASS_PHASE28AR_HARNESS_INTEGRATION`
- `PASS_DISABLED_MODE_NOOP`
- `PASS_ENABLED_PAGER_INIT`
- `PASS_RESIDUAL_VIEW_LOOKUP`
- `PASS_BUDGET_STRESS`
- `PASS_LRU_HARNESS`
- `PASS_LLAMA_CLI_DEFAULT_PRESERVED`
- `PARTIAL_COMMON_PARAM_LINK_DEFERRED`
