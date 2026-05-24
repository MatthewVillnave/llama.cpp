# Phase 28BR-E: Pager Global ODR Fix

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Base HEAD:** `550614179` (Phase 28BR-C clean rebuild rerun)
**Timestamp:** `2026-05-24T09:48 EDT`
**Verdict:** `PASS_PHASE28BR_E_ODR_GLOBAL_FIX`

---

## Goal

Fix duplicate pager global storage so all runtime paths share the same:

- `g_prt_pager`
- `g_prt_pager_enabled`

No true residual injection was attempted. No clamping or NaN zeroing was added. No quality or speed claims are made.

## ODR Evidence Before Fix

Source grep found duplicate strong storage:

```text
src/prt_sidecar_pager_globals.cpp:159:prt_sidecar_pager* g_prt_pager = nullptr;
src/prt_sidecar_pager_globals.cpp:160:bool g_prt_pager_enabled = false;
src/prt_sidecar_runtime_link.cpp:20:prt_sidecar_pager* g_prt_pager = nullptr;
src/prt_sidecar_runtime_link.cpp:21:bool g_prt_pager_enabled = false;
```

`src/prt_sidecar_pager_globals.cpp` is linked into `libllama.so`; `src/prt_sidecar_runtime_link.cpp` existed as an untracked source file with duplicate definitions. The runtime-link header already declares both globals as `extern`.

## Fix

- Kept the single strong definitions in `src/prt_sidecar_pager_globals.cpp`.
- Changed `src/prt_sidecar_runtime_link.cpp` to extern-only references.
- Updated `examples/speculative/prt_sidecar_runtime_link.h` comments to point at the real definition owner.
- Added direct `PRT_FORENSIC_LOG` breadcrumbs for init, hook entry, and decode/cache checkpoints so address/value evidence does not depend on stderr capture.
- Fixed a 28BR-B contribution log varargs mismatch: the log format printed only R/Y shapes but passed X/R/Y shape fields, causing undefined varargs reads and false `nan=896 inf=896` output.

## Symbol Check After Fix

```text
00000000006bd908 B g_prt_pager
00000000006bd901 B g_prt_pager_enabled
```

There is one runtime `g_prt_pager` storage symbol and one runtime `g_prt_pager_enabled` storage symbol in `build/bin/libllama.so`.

## Runtime Canary

Command used:

```bash
PRT_FORENSIC_LOG=/tmp/phase28br_e_forensics.jsonl \
timeout 90s ./build/bin/llama-cli \
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

The process timed out after emitting the required forensic lines because `llama-cli` stayed in interactive prompt mode. The runtime evidence was captured before timeout.

## Shared Global Address Evidence

All runtime sites reported the same global storage addresses:

| Site | pager_ptr | g_prt_pager address | enabled | g_prt_pager_enabled address |
|------|-----------|---------------------|---------|------------------------------|
| init | `0x6512260cbbd0` | `0x6511fc86f798` | 1 | `0x6511fc86f880` |
| hook | `0x6512260cbbd0` | `0x6511fc86f798` | 1 | `0x6511fc86f880` |
| checkpoint A-D | `0x6512260cbbd0` | `0x6511fc86f798` | 1 | `0x6511fc86f880` |

## Decode / Cache Checkpoints

| Checkpoint | Location | nan | inf | finite | cache_hit |
|------------|----------|-----|-----|--------|-----------|
| A | immediately after `decode_bytes()` | 0 | 0 | 1 | 0 |
| B | immediately after owned copy | 0 | 0 | 1 | 0 |
| C | immediately after cache retrieval | 0 | 0 | 1 | 1 |
| D | immediately before contribution loop | 0 | 0 | 1 | 1 |

All checkpoints reported:

```text
rows=896 cols=896 float_count=802816 byte_count=3211264 abs_sum=4.009560e+05 max_abs=1.000000e+00
```

## Contribution Shadow Result

After the log varargs fix:

```text
[PRT-CONTRIB-SHADOW] il=0 family=attn_out X_synthetic=I_KK R=[896x896] Y=[896x896] abs_sum=4.009560e+05 max_abs=1.000000e+00 nan=0 inf=0 finite=1
```

`sidecar_math_influenced_output` remained false:

```text
[PRT-APPLY-SHADOW] il=0 family=attn_out layer_match=1 family_match=1 decoded_views=1 app_attempts=1 app_success=1 sidecar_math_influenced_output=0
```

## Classification

The earlier `nan=896 inf=896` was not residual corruption after the ODR fix. The final classification is:

`ODR_GLOBAL_SPLIT_PLUS_CONTRIB_LOG_VARARGS_BUG`

More precise split:

- 28BR-D checkpoint capture was blocked by split/global ambiguity and stderr capture fragility.
- 28BR-E unifies the runtime pager global state and proves decode/cache checkpoints fire.
- A-D prove decoded R remains finite before cache, after cache copy, after cache retrieval, and before contribution.
- The remaining nonfinite contribution line was a logging bug in `src/llama-graph.cpp`, not math: the varargs passed `X_rows/X_cols` into a format string that started at `R=[%zux%zu]`, shifting later fields.

## Boundary

- No true residual injection.
- No model output mutation.
- No clamping.
- No NaN zeroing.
- No quality claim.
- No speed claim.

## Artifacts

- `/tmp/phase28br_e_forensics.jsonl` — direct forensic JSONL from runtime
- `/tmp/phase28br_e_runtime.log` — raw runtime stderr/stdout capture, not staged
- `examples/speculative/PHASE28BR_E_ODR_GLOBAL_FIX.md`
- `examples/speculative/results/phase28br_e_odr_global_fix.json`
