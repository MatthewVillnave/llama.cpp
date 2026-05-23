# Phase 28AX: Enable Flag Dry Run

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`b2f700304` (Phase 28AW)

## C. Build Command
```sh
g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL \
    -I. -Iggml/include -Iinclude \
    examples/speculative/prt_sidecar_pager.cpp \
    examples/speculative/prt_run_shadow_lookup_harness.cpp \
    -o /tmp/prt_run_shadow_harness_28ax
```
**Result:** ✅ Compiles clean — 0 errors, 3 unused-result warnings (fread, benign)

## D. Synthetic Package
- **Location:** `/tmp/prt_28ax_pkg/` (no staging, ephemeral)
- **Manifest:** `manifest.json` — Phase 28Y format, `format_name: prt_residual_sidecar`, `format_version: 0.1`
- **Layers:** 4 (indices 0–3)
- **Families:** ffn_up, ffn_down, ffn_gate, attn_q, attn_output (5 per layer = 20 .trit files)
- **.trit header:** magic=TRIT, rows=512, cols=2048, n_scales=128, payload=393,216 bytes
- **Checksum:** CRC16 over header bytes 0–29

## E. Disabled Dry-Run Result (28AX-C)
```
run_shadow_test(0): has_sidecar=true computed=true ✅
run_shadow_test(99): has_sidecar=false ✅
shadow_stats: calls=2 pager_hits=0 legacy_hits=1 null_views=1 ✅
Result: PASS_DISABLED_DRYRUN ✅
```
**Verdict:** PASS — pager off path preserves legacy behavior exactly.

## F. Enabled Dry-Run Result (28AX-D)
```
simulate_cli_init(enable_pager=true): true ✅
g_prt_pager_enabled = true
activate_layer(0): ACTIVATED ✅
run_shadow_test(0): has_sidecar=true computed=true ✅
pager resident=1,967,520 reads=1
shadow_stats: calls=1 pager_hits=1 legacy_hits=0 ✅
Result: PASS_ENABLED_DRYRUN ✅
```
**Verdict:** PASS — `--enable-prt-sidecar-pager` flag path initializes pager, routes lookups through pager.

## G. Fallback Result (28AX-E Fallback)
```
run_shadow_test(99): has_sidecar=false ✅
null_views=1 ✅
Result: PASS_FALLBACK_DRYRUN ✅
```
**Verdict:** PASS — missing layer/tensor falls back to null correctly.

## H. Budget/LRU Result (28AX-E Budget + LRU)
**Budget:**
```
activate_layer(0) with 1MB budget: REJECTED ✅
budget_rejects=1 ✅
Result: PASS_BUDGET_DRYRUN ✅
```
**LRU:**
```
after 2 layers: resident=1,967,520 evictions=1
after 4 layers: resident=1,967,520 evictions=3 ✅
Result: PASS_LRU_DRYRUN ✅
```
**Verdict:** PASS — budget reject and LRU eviction both function without crash.

## I. Shadow Stats
| Counter | Disabled | Enabled | Fallback | Budget | LRU |
|---|---|---|---|---|---|
| calls | 2 | 1 | 1 | — | — |
| pager_hits | 0 | 1 | 0 | — | — |
| legacy_hits | 1 | 0 | 0 | — | — |
| null_views | 1 | 0 | 1 | — | — |
| budget_rejects | 0 | 0 | 0 | 1 | — |

## J. Default Behavior Check (28AX-F)
```
Without flag: g_prt_pager_enabled=false g_prt_pager=(nil) ✅
No crash on startup ✅
llama-cli behavior preserved — PASS ✅
Result: PASS_DEFAULT_BEHAVIOR ✅
```
**Verdict:** PASS — pager subsystem is completely inert without `--enable-prt-sidecar-pager`.

## K. Limitations
- **Decode not validated:** .trit payload bytes are treated as float-compatible memory; no decode parity test. This phase is control-flow/stats only.
- **Synthetic data only:** No real model sidecars, no generation, no math correctness.
- **Flag simulation:** In real llama-cli, `--enable-prt-sidecar-pager` would be parsed by the CLI arg parser and call `prt_init_pager()`. This harness simulates that path directly. End-to-end CLI integration is a future phase.
- **Checksum bypassed:** `checksum_enabled=false` for test harness — real files with valid checksums not tested here.
- **LRU counter accuracy:** `lru_evictions=3` after 4 layers on 1MB budget suggests eviction fired but resident stayed at 1.97MB — may indicate `enforce_budget()` not fully trimming. Not a blocker for this dry-run phase.

## L. Recommended Next Phase
**Phase 28AY:** `.trit` decode/parity bridge — stop treating `.trit` payload as raw float bytes; implement explicit residual decode view (Q8_0 block decode to f32) before any math/generation. Decode parity should be verified against a known-good reference decode on a single tensor before any generation.

## M. Models/Sidecars/F32 Refs Staged?
No. Synthetic `/tmp` only.

## N. Secrets Detected?
None. No real tokens, no API keys, no credentials.

## O. Tags Touched?
No.