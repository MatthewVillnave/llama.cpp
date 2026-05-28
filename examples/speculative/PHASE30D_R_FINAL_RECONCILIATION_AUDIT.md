# Phase 30D-R — Final Reconciliation / Claim Boundary Audit

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`
**Old HEAD:** `14eb661aa` (Phase 30D doc commit)
**New HEAD:** `b5f3c7a21` (Phase 30D-R commit)

## Classification
`PASS_SCOPED_RESIDENCY_UNSUPPORTED`

---

## 1. Build/Flag Provenance

### Binary verification
| Item | Value |
|------|-------|
| git HEAD | `14eb661aa` |
| Binary built | May 28 18:18 (from Phase 29F commit `cbe60117c`) |
| CXX_FLAGS | `-DPRT_SIDECAR_PAGER_EXPERIMENTAL` present |
| `llama_set_prt_flags` | Defined at `0x9c390` in `libllama.so` |
| char[64] fix | Present (Phase 29F `strncpy` buffer fix in `llama_set_prt_flags`) |

### Phase 29F fix status — NOT an unblock
Phase 29F improved the `strncpy` buffer sizing inside `llama_set_prt_flags()` (avoiding truncation if family name were long). Phase 29E/30D-R found: **`llama_set_prt_flags()` is never called from the CLI**.

Evidence:
- `[PRT-FLAGS-SET]` debug print inside `llama_set_prt_flags()` never fires
- `[PRT-PAGER] enabled via --enable-prt-sidecar-pager` never fires
- Runtime globals always: `g_true_inj=0 g_apply=0 family_len=0`
- All layers: `action=guard_reject flags_disabled`

**Root cause:** The CLI pager init code path (cli.cpp line ~580-625) that calls `llama_set_prt_flags()` is either (a) not reached due to a missing initialization flag, or (b) the stderr output from initialization is being lost before the forward pass starts. The library's `[PRT-NATIVE]` and `[PRT-INJECT-DOWN]` prints appear during the forward pass, but CLI init prints (`[PRT-PAGER]`, `[PRT-APPLY]`, `[PRT-FLAGS-SET]`) do not appear.

**Phase 29F verdict:** NOT a flag propagation fix. It was a buffer-sizing improvement inside a function that is never called. The PRT flag pipeline remains BROKEN.

**Phase 29E subagent flag error:** Used `--prt-sidecar-apply attn_out` instead of `--prt-sidecar-apply --prt-sidecar-apply-family attn_out`. The subagent's "error: invalid argument: attn_out" was its own mistake, not the binary's fault.

---

## 2. RSS Table Provenance

### Q2/Q3 source verification
| Model | Path | Provenance | File Size |
|-------|------|------------|-----------|
| Q4_K_M | VL_usb `qwen2.5-0.5b-instruct-q4_k_m.gguf` | from FP16 | 469 MB |
| Q3_K_M | VL_usb `qwen2.5-0.5b-instruct-q3_k_m.gguf` | from FP16 | 413 MB |
| Q2_K | VL_usb `qwen2.5-0.5b-instruct-q2_k.gguf` | from FP16 | 396 MB |

**All three are official HF GGUFs — clean F16 provenance, NOT Q4-derived. Verified by source: HF repo `Qwen/Qwen2.5-0.5B-Instruct-GGUF`.**

### RSS measurement command
```bash
python3 /tmp/rss_test2.py
# Models: Q4, Q3, Q2 each with and without LLAMA_PRTSCDIR=/tmp/prt_sidecars_0_5b_layer0
# n=1, batch_size=512, prompt="Hello world", n_tokens=5, sleep=4s
```

### RSS Results — VERIFIED

| Configuration | RSS KB | vs Q4_K_M_baseline | Notes |
|---------------|--------|---------------------|-------|
| Q4_K_M_baseline | 967,436 KB | — | Clean FP16 Q4 |
| Q4_K_M + sidecars | 967,012 KB | -424 KB (-0.04%) | **Sidecars NOT activating** (flag pipeline broken) |
| Q2_K_baseline | 1,001,556 KB | **+34,120 KB (+3.5%)** | Official from-FP16 Q2 |
| Q2_K + sidecars | 1,001,540 KB | **+34,104 KB (+3.5%)** | Sidecars not activating |
| Q3_K_M_baseline | 1,062,688 KB | **+95,252 KB (+9.8%)** | Official from-FP16 Q3 |
| Q3_K_M + sidecars | 1,062,500 KB | **+95,064 KB (+9.8%)** | Sidecars not activating |

### Q4+sidecar note
Since PRT flag pipeline is broken, `g_prt_pager_enabled=FALSE` → sidecar loading is bypassed → Q4+sidecar ≈ Q4_baseline. The RSS delta of -424 KB is measurement noise, not sidecar effect.

### Sidecar manifest verified
- Path: `/tmp/prt_sidecars_0_5b_layer0/manifest.json`
- Format: Phase28Y-compatible (`entries` key, not `sidecars`)
- 3 entries: attn_out, ffn_up, ffn_down for layer 0
- All `.trit` files present and non-empty

---

## 3. Flag Smoke Result

**attn_out scale=1 test:**
```
[PRT-INJECT-DOWN-DEBUG] ENTER il=0 g_true_inj=0 g_apply=0
[PRT-INJECT-DOWN] il=0 family= family_len=0 action=guard_reject flags_disabled
```
- `trit_validated`: **NULL** (pager init never fired, no manifest load)
- `injection_successes`: **0** (g_true_inj=0 g_apply=0 at all layers)
- `family_len=0`: **CONFIRMED** — `llama_set_prt_flags()` never received the family parameter

**Flag smoke result: FAIL — FLAG_REGRESSION_AFTER_29F**

---

## 4. Conflict Resolution

| Phase | Claim |
|-------|-------|
| Phase 29F commit | "fix PRT family flag ABI propagation" — claimed to unblock injection |
| Phase 29E result | `injection_successes=0`, globals always FALSE |
| Phase 30D-R verification | `llama_set_prt_flags()` never called, `[PRT-FLAGS-SET]` never fires |

**Resolution:** Phase 29F improved buffer handling inside `llama_set_prt_flags()` but the call site in the CLI never executes. Phase 29F is NOT a flag propagation unblock. The regression is real and pre-dates Phase 30D.

**Note:** The Phase 29E subagent also made a flag syntax error (`--prt-sidecar-apply attn_out` instead of `--prt-sidecar-apply --prt-sidecar-apply-family attn_out`), which compounded the issue. Both the flag syntax error AND the missing call are contributing factors.

---

## 5. Final Scoped Claim

**Accepted claim (classification A — PASS_SCOPED_RESIDENCY_UNSUPPORTED):**

> "For Qwen2.5-0.5B on this CPU/llama.cpp setup, official/FP16-derived Q2/Q3 GGUFs were tested as resident base models. Q2_K uses +3.5% more RSS than Q4_K_M; Q3_K_M uses +9.8% more RSS than Q4_K_M. Sidecar injection could not be activated due to a PRT flag pipeline regression. The current low-bit-base + sidecar residency path is unsupported for this model/hardware setup."

**What this does NOT claim:**
- SDI is not globally disproven across all models/hardware/runtimes
- Sidecars can never work in any configuration
- Quality or correctness of any model
- Behavior on larger models

**What this confirms:**
- Q4_K_M is the lightest configuration for Qwen2.5-0.5B on TheForgeHQ
- Sidecars are non-functional in the current binary (flag pipeline regression)
- The RSS finding is based on official F16-derived models, not Q4-derived models

---

## 6. Recommended Next Phase

### Phase 30E — PRT Flag Pipeline Fix (required before any further sidecar testing)

**Root cause:** `llama_set_prt_flags()` is never called from the CLI despite the call site being compiled in.

**Tasks:**
1. Trace why the pager init block (cli.cpp ~580-625) is not reached
2. Add debug prints BEFORE the `#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL` block to confirm the code path is entered
3. Verify `params.prt_sidecar_pager_enabled` is TRUE when `--enable-prt-sidecar-pager` is passed
4. Add debug print AFTER `llama_set_prt_flags()` call to confirm it executes
5. Fix the call site so flags actually propagate
6. Re-run A-H smoke tests to confirm `trit_validated=1` and `injection_successes>=1`

**Alternative:** If the CLI path is fundamentally broken, consider a standalone harness that calls `llama_set_prt_flags()` directly before model loading.

---

## 7. Repo Hygiene

The following untracked files should be reviewed/cleaned:
- Phase 28BR series files (Forensics, Shadow Compare, etc.)
- Phase 27h/27j files
- Phase 26K/26L/26R/24R results

No GGUF model files or secrets staged.

---

## Commit
`b5f3c7a21` — Phase 30D-R: scoped residency claim confirmed; PRT flag pipeline regression confirmed; 29F not an unblock