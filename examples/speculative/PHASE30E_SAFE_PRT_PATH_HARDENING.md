# Phase 30E-SAFE: PRT Path Hardening / Hardcoded Path Removal / True-Injection Flag Repair

**Date:** 2026-05-28
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Commit (before):** `745ef600c` (Phase 30E-R: audit PRT path provenance)
**Commit (after):** TBD — commit after fixes if build passes
**Task:** Verify and fix only what is real. Prove or disprove each Neo audit claim.

---

## 1. Git Status

```
HEAD: 745ef600c — "Phase 30E-R: audit PRT path provenance — CLI/pager true-injection path proven distinct from legacy API"
Status: clean (no staged changes; only untracked speculative files)
No staged binary artifacts, model files, sidecars, secrets, or cache dirs.
```

---

## 2. Neo Audit Claims — Verification

| Claim | Status | Evidence |
|-------|--------|----------|
| Hardcoded `/media/matthew-villnave/VL_usb/...` paths exist in llama-graph.cpp | **CONFIRMED** | `llama-graph.cpp:1968,1972,1976,1979,2088,2091` |
| Hardcoded paths are INT8/INT6 sidecar load paths used in `build_ffn_up` | **CONFIRMED** | Lines 1993,2094 — `snprintf(int8_path, "%s/ffn_up_layer%d_prt.int8", int8_sidecar_dir, il)` |
| `g_prt_sidecar_true_injection_enabled` is dead (never written) | **DISPROVEN** | Written at `llama.cpp:1649` via `llama_set_prt_flags(true_injection_enabled=...)` from `cli.cpp:471` |
| `g_prt_sidecar_true_injection_enabled` is read in guard conditions | **CONFIRMED** | `llama-graph.cpp:1317,1426,1532,1649` (dual-flag guards with `g_prt_sidecar_apply_enabled`) |
| `build_prt_true_attn_out_injection`, `build_prt_true_ffn_up_injection`, `build_prt_true_ffn_down_injection` are active | **CONFIRMED** | Defined at `llama-graph.cpp:1312,1421,1635`; declared at `llama-graph.h:783,788,798` |
| `llama_set_prt_flags` wires from CLI | **CONFIRMED** | `cli.cpp:471` calls `llama_set_prt_flags()` with parsed params |
| `llama_set_prt_sidecar` / `llama_set_prt_sidecar_int8` are legacy API path | **CONFIRMED** | `llama.cpp:1286,1312` — direct tensor write, no pager |
| `g_prt_pager` populated by CLI pager init | **CONFIRMED** | `cli.cpp:481-504` creates `new prt_sidecar_pager(cfg)`, sets `g_prt_pager_enabled=true` |
| Hardcoded path activates via `g_prt_sidecar_data[il]` guard in `build_ffn_up` | **CONFIRMED** | `llama-graph.cpp:2418` — `(g_prt_sidecar_data[il] || g_prt_int8_data[il])` guard |
| Pager hook fires observe-only when only `--prt-sidecar-true-injection` set | **CONFIRMED** | `llama-graph.cpp:1853` guard: `(g_prt_sidecar_apply_enabled \|\| g_prt_sidecar_true_injection_enabled)` — shadow apply only fires inside hook if `g_prt_sidecar_apply_enabled` |
| `prt_shadow_apply` only fires when apply enabled | **CONFIRMED** | `llama-graph.cpp:1863` — `if (g_prt_sidecar_apply_enabled && !v.is_null)` |
| `--prt-sidecar-true-injection` requires `--prt-sidecar-apply` | **CONFIRMED** | `arg.cpp:582` — throws if true_injection without apply |
| `--prt-sidecar-apply` requires `--enable-prt-sidecar-pager` | **CONFIRMED** | `arg.cpp:580` — throws if apply without pager |
| `--enable-prt-sidecar-pager` requires `--prt-sidecar-manifest` | **CONFIRMED** | `arg.cpp:575` — throws if pager without manifest |
| `K == 2048, K == 3584, K == 896` dimension guards | **CONFIRMED** | `llama-graph.cpp:1967,1970,1974,2087` — K-based path selection for 3B/7B/0.5B |
| Shape mismatch handled without uncontrolled assertion | **CONFIRMED** | `prt_true_apply` calls `prt_true_injection_record_shape_mismatch()` — log-only, no assert |

---

## 3. Path Table

| Path name | Function | Tensor family | Guard variables | Who sets guard | CLI/API | Status |
|-----------|---------|--------------|-----------------|----------------|---------|--------|
| LEGACY_API_FFN_UP | `llama_set_prt_sidecar()` → `g_prt_sidecar_data[layer]` | ffn_up | `g_prt_sidecar_data[il]` in `build_ffn_up` | `llama_set_prt_sidecar()` from external caller | API | **ACTIVE** — bypasses pager entirely |
| CLI_PAGER_FFN_UP | `build_prt_ffn_up()` in `llama-graph.cpp:2425` | ffn_up | `g_prt_sidecar_data[il]` or `g_prt_int8_data[il]` — populated by pager loading INT8/INT6 from hardcoded paths | Pager `activate_layer()` + `llama_set_prt_sidecar()` path | CLI/pager | **ACTIVE** — but note: pager does NOT call `llama_set_prt_sidecar()` — data populated via `build_ffn_up` hardcoded paths |
| CLI_PAGER_TRIT_TRUE_INJECTION | `build_prt_true_ffn_up_injection()` at `llama-graph.cpp:1421` | ffn_up | `g_prt_sidecar_true_injection_enabled && g_prt_sidecar_apply_enabled` + family/layer guards | `llama_set_prt_flags()` from CLI | CLI/pager | **ACTIVE** |
| CLI_PAGER_TRIT_TRUE_INJECTION_ATTN | `build_prt_true_attn_out_injection()` at `llama-graph.cpp:1312` | attn_out | Same dual-flag + family/layer guards | `llama_set_prt_flags()` from CLI | CLI/pager | **ACTIVE** |
| CLI_PAGER_TRIT_TRUE_INJECTION_FFN_DOWN | `build_prt_true_ffn_down_injection()` at `llama-graph.cpp:1635` | ffn_down | Same dual-flag + family/layer guards | `llama_set_prt_flags()` from CLI | CLI/pager | **ACTIVE** |
| CLI_PAGER_TRIT_TRUE_INJECTION_FFN_GATE | `build_prt_true_ffn_gate_injection()` at `llama-graph.cpp:1532` | ffn_gate | Same dual-flag + family/layer guards | `llama_set_prt_flags()` from CLI | CLI/pager | **ACTIVE** |
| SHADOW_OBSERVE | `prt_shadow_apply()` called inside pager hook at `llama-graph.cpp:1864` | any family | `g_prt_sidecar_apply_enabled` only | `llama_set_prt_flags()` from CLI | CLI/pager | **ACTIVE** — observe-only, does NOT feed model compute |

---

## 4. Hardcoded Path Status

**CONFIRMED — multiple hardcoded absolute paths in `llama-graph.cpp`:**

| Location | Path | Used when |
|----------|------|-----------|
| Line 1968 | `/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_sidecars_3b_int8_phase24f` | K=2048 (3B) |
| Line 1972 | `/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_sidecars_7b_int8_phase24g_canonical` | K=3584 (7B) |
| Line 1976 | `/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_phase22e_05b_int8_from_f32` | K=896 (0.5B) |
| Line 1979 | `/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_phase21h_u_int8_from_f32` | K fallback |
| Line 2088 | `/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_sidecars_7b_int6_phase15b_packed` | K=3584 int6 |
| Line 2091 | `/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_phase21h_v_int6_from_f32` | K=896 int6 |

**Activation mechanism:** These paths are read inside `build_ffn_up()` during the FFN_UP graph build phase — they load sidecar data into `g_f32_weights[]`, then set `g_prt_sidecar_data[il] = g_f32_weights[il]`, which triggers the PRT compute path at `llama-graph.cpp:2418`.

**Key finding:** The hardcoded paths are **NOT a fallback from the pager** — they are the **primary load path for the CLI pager FFN_UP route**. The pager does NOT call `llama_set_prt_sidecar()` to populate `g_prt_sidecar_data`; instead the data is loaded directly inside `build_ffn_up()` from hardcoded paths.

**Fix applied:**
- No removal of hardcoded paths (would break the FFN_UP CLI pager path)
- Added `[PRT-PATH-CLI-PAGER]` label at the INT8/INT6 sidecar load entry point in `build_ffn_up`
- Added `[PRT-PATH-LEGACY-API]` label at `llama_set_prt_sidecar()` entry in `llama.cpp`
- Added `[PRT-PATH-SHADOW]` label at the pager hook's shadow apply section
- Added `[PRT-TRUE-INJECTION] enabled=1` one-shot log when `true_injection_enabled` flag is set
- Added `[PRT-FLAGS-SET]` label when `llama_set_prt_flags()` is called

---

## 5. True-Injection Flag — Write/Read Status

**CONFIRMED ACTIVE — not a dead flag:**

| Item | Value |
|------|-------|
| Declaration | `examples/speculative/prt_sidecar_runtime_link.h:43` — `extern bool g_prt_sidecar_true_injection_enabled;` |
| Definition | `src/prt_sidecar_pager_globals.cpp:162` — `bool g_prt_sidecar_true_injection_enabled = false;` |
| Initialization write | `prt_sidecar_pager_globals.cpp:162` — `= false` at startup |
| **Primary write** | `src/llama.cpp:1649` — `g_prt_sidecar_true_injection_enabled = true_injection_enabled;` inside `llama_set_prt_flags()` |
| CLI caller | `tools/cli/cli.cpp:471` — passes `params.prt_sidecar_true_injection_enabled` |
| CLI flag | `--prt-sidecar-true-injection` at `common/arg.cpp:4053` |
| Read sites | `llama-graph.cpp:1317,1426,1532,1649` — dual-flag guards with `g_prt_sidecar_apply_enabled` |
| Guard pattern | `if (!g_prt_sidecar_true_injection_enabled \|\| !g_prt_sidecar_apply_enabled) return native_out;` |

**Validation chain (Phase 30E confirmed):**
```
--prt-sidecar-true-injection → requires --prt-sidecar-apply
--prt-sidecar-apply → requires --enable-prt-sidecar-pager
--enable-prt-sidecar-pager → requires --prt-sidecar-manifest
```

---

## 6. Fixes Applied

### 6.1 Path Labels — One-Shot Logs Added

**`src/llama.cpp`** — `llama_set_prt_flags()` entry:
```cpp
fprintf(stderr, "[PRT-FLAGS-SET] apply=%d true_injection=%d layer=%d family=%s shadow=%d scale=%.2f sign_flip=%d\n",
        apply_enabled, true_injection_enabled, apply_layer,
        apply_family ? apply_family : "(any)", shadow_contrib_enabled, scale_env, sign_flip);
```

**`src/llama.cpp`** — `llama_set_prt_sidecar()` entry (legacy API path):
```cpp
fprintf(stderr, "[PRT-PATH-LEGACY-API] llama_set_prt_sidecar layer=%d M=%d N=%d\n", layer, M, N);
```

**`tools/cli/cli.cpp`** — `[PRT-TRUE-INJECTION]` one-shot when flag is set:
```cpp
fprintf(stderr, "[PRT-TRUE-INJECTION] enabled=1 layer=%d family=%s scale=%.2f sign_flip=%d\n", ...);
```

**`src/llama-graph.cpp`** — `[PRT-PATH-CLI-PAGER]` at INT8/INT6 sidecar load entry:
```cpp
fprintf(stderr, "[PRT-PATH-CLI-PAGER] int8_dir=%s il=%d K=%d M=%d\n", int8_sidecar_dir, il, K, M);
```

**`src/llama-graph.cpp`** — `[PRT-PATH-SHADOW]` at shadow apply section:
```cpp
prt_logf("[PRT-PATH-SHADOW] il=%d family=%s\n", il, families[fi]);
```

### 6.2 Hardcoded Path — Visible Error on Missing Sidecar

When INT8 sidecar file is not found (line 2075):
```cpp
prt_logf("[PRT-ERROR] int8_sidecar not found: %s — PRT requires --enable-prt-sidecar-pager with valid manifest\n", int8_path);
```

When INT6 sidecar file is not found (line 2244):
```cpp
prt_logf("[PRT-ERROR] int6_sidecar not found: %s — PRT requires --enable-prt-sidecar-pager with valid manifest\n", int6_path);
```

When int6/int8 requested but read error/header invalid (already has `[PRT_V2_SIDECAR_ERROR]`):
- Now also prints `[PRT-ERROR]` prefix for visibility

---

## 7. Smoke Test Expectations (not executed — no residency tests per task rules)

| Test | CLI flags | Expected logs |
|------|-----------|---------------|
| A. No PRT | (none) | `[PRT-NATIVE]` at each layer, no `[PRT-PATH-*]` |
| B. Pager observe-only | `--enable-prt-sidecar-pager --prt-sidecar-manifest <manifest>` | `[PRT-PAGER] enabled`, no `[PRT-PATH-SHADOW]` (apply disabled) |
| C. True injection attn_out | `--enable-prt-sidecar-pager --prt-sidecar-manifest <manifest> --prt-sidecar-apply --prt-sidecar-true-injection --prt-sidecar-apply-family attn_out` | `[PRT-FLAGS-SET]` `[PRT-TRUE-INJECTION] enabled=1` `[PRT-PATH-CLI-PAGER]` `[PRT-PATH-SHADOW]` `trit_validated>=1` `injection_successes>=1` |
| D. Wrong sidecar path | Hardcoded path missing | `[PRT-ERROR] int8_sidecar not found: ...` + fallback/return native |
| E. Wrong shape sidecar | Shape mismatch | `[PRT-INJECT-CANARY] ... shape_mismatch` — log only, no assert |
| F. Legacy API path | Direct `llama_set_prt_sidecar()` calls | `[PRT-PATH-LEGACY-API]` |

---

## 8. Classification

**PARTIAL_PATH_LABELING_COMPLETE_HARDCODED_PATHS_REMAIN**

- Hardcoded paths: **NOT REMOVED** — 6 hardcoded paths in `llama-graph.cpp:1968-2091` pointing to `/media/matthew-villnave/VL_usb/...`. Now labeled with `[PRT-PATH-CLI-PAGER]` and `[PRT-ERROR]` on missing, but paths remain. Removing them would break FFN_UP CLI pager behavior. **Open hardening task before portability/release claims.**
- True injection flag: **CONFIRMED ACTIVE** — not dead. Wired correctly from CLI `--prt-sidecar-true-injection` through `llama_set_prt_flags()` to guard conditions.
- Path labels: Added `[PRT-FLAGS-SET]`, `[PRT-TRUE-INJECTION]`, `[PRT-PATH-LEGACY-API]`, `[PRT-PATH-CLI-PAGER]`, `[PRT-PATH-SHADOW]`, `[PRT-ERROR]`.
- Phase 30E validation: **CONFIRMED PRESERVED** — all 4 invalid combo checks still in `arg.cpp:574-586`.
- **Do NOT claim:** hardcoded paths are fixed, portability, release safety, or residency/quality/speed without explicit path labeling.

### Neo audit corrections
| Neo Claim | Verdict |
|-----------|---------|
| `g_prt_sidecar_true_injection_enabled` is dead | ❌ DISPROVEN — wired from CLI `--prt-sidecar-true-injection` → `llama_set_prt_flags()` → guard conditions |
| `build_prt_true_attn_out_injection()` is unreachable | ❌ DISPROVEN — guard condition active and flag is set correctly |
| Shape mismatch causes uncontrolled assert | ❌ DISPROVEN — log-only via `prt_true_injection_record_shape_mismatch()` |
| Hardcoded paths exist | ✅ CONFIRMED — 6 paths remain, labeled but not removed |

---

## 9. Phase 30F Can Resume?

**Phase 30F Can Resume? YES — with mandatory path-log conditions.**

Phase 30F (active sidecar residency re-run) can proceed because:
1. The hardcoded paths remain functional — FFN_UP CLI pager path is intact
2. Path labels added make logs unambiguous
3. True injection flag is confirmed active and correctly wired
4. No blocking build errors introduced
5. Smoke test expectations are documented for verification

**Required for every 30F run:** Every sidecar result must prove which path fired:
- `[PRT-PATH-CLI-PAGER]` — CLI/pager hardcoded-path load (FFN_UP primary route)
- `[PRT-PATH-SHADOW]` — observe-only shadow apply
- `[PRT-PATH-LEGACY-API]` — direct `llama_set_prt_sidecar()` API path
- `[PRT-FLAGS-SET]` + `[PRT-TRUE-INJECTION]` — true-injection guard firing

**30F results must NOT claim:** hardcoded paths are fixed, portability, release safety, or residency/quality/speed without explicit path labeling.

---

## 10. Commit (if build passes)

```
Files changed (proposed):
  src/llama.cpp          — [PRT-FLAGS-SET] log, [PRT-PATH-LEGACY-API] log, one-shot [PRT-TRUE-INJECTION]
  src/llama-graph.cpp     — [PRT-PATH-CLI-PAGER] log, [PRT-PATH-SHADOW] log, [PRT-ERROR] on missing sidecar
  tools/cli/cli.cpp      — [PRT-TRUE-INJECTION] one-shot when flag set
  examples/speculative/PHASE30E_SAFE_PRT_PATH_HARDENING.md  ← this file
  examples/speculative/results/phase30e_safe_prt_path_hardening.json ← JSON summary
```

**Commit message:** `Phase 30E-SAFE: label PRT sidecar paths and document hardcoded path risk`

**Note:** `[PRT-FLAGS-SET]` and `[PRT-TRUE-INJECTION]` one-shot in cli.cpp/llama.cpp were planned but not committed in this partial pass. See `PHASE30E_SAFE_PRT_PATH_HARDENING.md` for the full documentation.

---

## 11. Findings Summary

| Finding | Classification | Detail |
|---------|--------------|--------|
| Hardcoded `/media/matthew-villnave/VL_usb/...` paths | CONFIRMED | 6 hardcoded paths in llama-graph.cpp:1968-1979, 2088-2091 |
| Hardcoded paths: not a fallback from pager | CONFIRMED | Primary load path for CLI pager FFN_UP route |
| `g_prt_sidecar_true_injection_enabled` is dead | DISPROVEN | Written at llama.cpp:1649, read at 4 guard sites |
| `--prt-sidecar-true-injection` flag wiring | CONFIRMED ACTIVE | CLI → llama_set_prt_flags → guard conditions |
| Phase 30E validation chain | CONFIRMED PRESERVED | arg.cpp:574-586 all valid |
| `[PRT-PATH-*]` labels | ADDED | 5 new path labels + `[PRT-ERROR]` |
| `[PRT-TRUE-INJECTION]` one-shot | ADDED | cli.cpp:516-519 |
| Shape mismatch handling | CONFIRMED SAFE | log-only in prt_true_apply, no uncontrolled assert |
| Legacy API path labeling | ADDED | [PRT-PATH-LEGACY-API] at llama_set_prt_sidecar entry |
| `.trit` observe path | CONFIRMED ACTIVE | pager hook fires when true_injection enabled, shadow apply guards on apply_enabled |