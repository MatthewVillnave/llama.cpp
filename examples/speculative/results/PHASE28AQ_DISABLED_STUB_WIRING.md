# Phase 28AQ: Disabled Stub Wiring for Sidecar Pager

## A. Branch

`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD

Start HEAD: `3c002c8a1` (`Phase 28AP: design disabled sidecar pager integration`)

## C. Files Changed

- `common/common.h`
- `common/arg.cpp`
- `common/common.cpp`
- `examples/speculative/results/PHASE28AQ_DISABLED_STUB_WIRING.md`
- `examples/speculative/results/phase28aq_disabled_stub_wiring.json`

## D. Config Fields

Added disabled-by-default sidecar pager fields to `common_params`:

- `bool prt_sidecar_pager_enabled = false`
- `std::string prt_sidecar_manifest`
- `size_t prt_sidecar_budget_mb = 0`
- `std::string prt_sidecar_policy = "strict"`
- `int prt_sidecar_prefetch_distance = 1`
- `int prt_sidecar_window_size = 4`
- `bool prt_sidecar_lru = false`
- `bool prt_sidecar_checksum = true`

Defaults preserve current behavior because `prt_sidecar_pager_enabled` is false.

## E. CLI Flags

Added CLI parsing for:

- `--enable-prt-sidecar-pager`
- `--prt-sidecar-manifest PATH`
- `--prt-sidecar-budget-mb N`
- `--prt-sidecar-policy POLICY`
- `--prt-sidecar-prefetch-distance N`
- `--prt-sidecar-window-size N`
- `--prt-sidecar-lru`
- `--prt-sidecar-checksum`
- `--no-prt-sidecar-checksum`

Supporting flags are inert unless `--enable-prt-sidecar-pager` is supplied.

## F. Pager Init Stub

Added enabled-only runtime-adjacent stub in `common_init_result` after model load:

- Disabled: does nothing.
- Enabled with empty manifest: clear error.
- Enabled with missing manifest path: clear error.
- Enabled with existing manifest: logs that pager linking is deferred.

No pager source is linked into active runtime in this phase. No residual lookup replacement was added. No matmul behavior was changed.

## G. Default Behavior Preservation

Checks:

- `cmake --build build --target llama-cli -j2`: PASS
- `build/bin/llama-cli --help`: PASS, new flags visible
- `build/bin/llama-cli`: returns existing `error: --model is required`, with no pager or manifest error
- `build/bin/llama-cli --prt-sidecar-manifest /tmp/fake_manifest.json`: returns existing `error: --model is required`, proving the supporting manifest flag is ignored when pager is disabled

No generation was run.

## H. Enabled-Mode Smoke

`build/bin/llama-cli --enable-prt-sidecar-pager` returns:

```text
error: --enable-prt-sidecar-pager requires --prt-sidecar-manifest
```

Exit code: `1`

Fake-manifest runtime init was not exercised because `llama-cli` requires a model before `common_init_result`; running a model load was intentionally skipped.

## I. Limitations

- Pager linkage is deferred.
- No `prt_sidecar_pager` instance is created in active runtime.
- No `g_sidecars` lookup is replaced.
- No layer activation, prefetch, residual lookup, or stats emission is wired into generation.
- No runtime correctness, speedup, 30B support, or production readiness is claimed.

## J. Recommended Next Phase

Phase 28AR: sidecar pager harness integration, not generation.

Focus:

- Link pager only in a controlled harness path.
- Exercise fake/small manifests without model generation.
- Keep default runtime behavior unchanged.

## K. Models/Sidecars/F32 Refs Staged?

No.

Existing tracked/untracked model-like files and build artifacts are present in the repository tree, but no `.gguf`, `.bin`, `.safetensors`, `.pt`, `.pth`, `.trit`, f32 refs, generated sidecars, binaries, logs, or captures were staged by this phase.

## L. Secrets Detected?

No secrets detected in the changed diff.

The broad requested grep over `examples/speculative/ common/ tools/ src/` reports existing fixture/docs strings, including fake keys and API-key examples. Those are pre-existing and not introduced by Phase 28AQ.

## M. Tags Touched?

No.

## Verdicts

- `PASS_PHASE28AQ_DISABLED_STUB_WIRING`
- `PASS_CONFIG_FIELDS_ADDED`
- `PASS_CLI_FLAGS_ADDED`
- `PASS_DEFAULT_BEHAVIOR_PRESERVED`
- `PASS_ENABLED_STUB_ERROR_HANDLING`
- `PARTIAL_PAGER_LINK_DEFERRED`
