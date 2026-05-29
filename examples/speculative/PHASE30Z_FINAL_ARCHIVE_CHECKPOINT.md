# Phase 30Z — Final Archive Checkpoint / Claims Freeze

**Date:** 2026-05-29
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Old HEAD:** `b26dcf571` (Phase 30E-HARDEN-FF)
**New HEAD:** TBD — to be committed
**Classification:** `PASS_ARCHIVE_CHECKPOINT_READY`

---

## Repo Hygiene

- ✅ No model files staged
- ✅ No sidecars staged
- ✅ No binaries staged
- ✅ No huge logs staged
- ✅ No private paths
- ✅ No secrets
- ✅ No hardcoded `/media/matthew-villnave/VL_usb/...` paths in active runtime
- ✅ `git diff --check`: no whitespace errors

---

## Frozen Proven Claims

| Claim | Phase | Evidence |
|-------|-------|----------|
| `.trit`/sidecar infrastructure built and tested | 24–30 | Multiple phases, telemetry confirmed |
| CLI/pager path proven active | 30E/30E-R | `[PRT-PATH-CLI-PAGER]`, `[PRT-FLAGS-SET]`, `[PRT-PAGER] enabled` in logs |
| Path labels exist | 30E-SAFE | `[PRT-PATH-CLI-PAGER]`, `[PRT-PATH-LEGACY-API]`, `[PRT-PATH-SHADOW]`, `[PRT-FLAGS-SET]`, `[PRT-TRUE-INJECTION]`, `[PRT-ERROR]` |
| Hardcoded paths removed | 30E-HARDEN | 6/6 `/media/matthew-villnave/VL_usb/...` paths removed; strace confirmed zero opens |
| Normal runtime fail-fast works | 30E-HARDEN-FF | `[PRT-ERROR]` fires in normal PRT mode, not just `g_prt_ggml_op_test=1` |
| Repo safe for archive/new-repo extraction | 30E-HARDEN-SMOKE | strace showed no hardcoded path access |

---

## Frozen Negative Results

| Result | Phase | Evidence |
|--------|-------|----------|
| Additive sidecar injection does NOT support residency thesis for Qwen2.5-0.5B on this machine/runtime | 30F | Q2/Q3+active always exceeded Q4 baseline RSS |
| Active FFN sidecars add large memory overhead | 30F | ~+277 MB / +27% per FFN sidecar regardless of base |
| Q2/Q3 + active sidecars did NOT beat Q4 baseline RSS | 30F | Q2+ffn_up=1,279 MB vs Q4_baseline=967 MB |

---

## Open Issues

| Issue | Status | Do Not Chase |
|-------|--------|--------------|
| `trit_header_invalid` — pager vs legacy `.trit` format mismatch | OPEN | ✅ Documented only |
| Additive sidecar path is not the right long-term architecture | KNOWN | Architectural insight only |
| Legacy/API path still exists | KNOWN | Should not be confused with future clean runtime |

---

## Allowed Summary Claim

> "For Qwen2.5-0.5B on this CPU/llama.cpp setup, the additive sidecar injection architecture was mechanically validated but did not reduce residency versus Q4. The repo is now cleaned enough to archive and use as source material for a cleaner substitutive tensor-replacement prototype."

---

## Forbidden Claims

- ❌ No quality recovery
- ❌ No correctness recovery
- ❌ No speedup
- ❌ No Q2→Q4 behavior recovery
- ❌ No production readiness
- ❌ No global SDI disproven claim
- ❌ No global SDI proven claim

---

## Commit History (This Branch)

| Phase | Commit | Classification |
|-------|--------|----------------|
| 30E | `a9a50e4b7` | Phase 30E: fix PRT flag/pager init call path |
| 30E-R | `745ef600c` | Audit PRT path provenance |
| 30E-SAFE | `636ea21fe` | PARTIAL_PATH_LABELING_COMPLETE_HARDCODED_PATHS_REMAIN |
| 30F | `bd714cbd5` | ADDITIVE_OVERHEAD_CONFIRMED_ACTIVE |
| 30E-HARDEN | `9ad6e281e` | PASS_HARDCODED_PATHS_REMOVED |
| 30E-HARDEN-FF | `b26dcf571` | PARTIAL_FAILFAST_WORKS_TRIT_FORMAT_PENDING |

---

## New Repo Extraction Recommendation

**Recommended direction: substitutive tensor replacement**

而不是 additive sidecar injection.

Core requirements for new repo:

1. **Goal:** Do NOT load the tensor being replaced. The resident low-bit tensor + residual sidecar should *replace* a higher-bit tensor — not add beside it.

2. **Compressed/on-demand residual compute:** Residuals should be decoded and applied only when needed, not pre-loaded.

3. **Explicit memory accounting from day one:** Every tensor loaded should have a clear RSS delta. No surprises.

4. **One active runtime path only:** No legacy/API ghosts. No parallel paths that could produce different results.

5. **No hardcoded paths:** All tensor paths must come from explicit configuration.

6. **CLAIMS.md + FORBIDDEN_CLAIMS.md at project start:** Document what you will and won't claim before writing any code. Stick to it.

---

## Classification

`PASS_ARCHIVE_CHECKPOINT_READY`

This is the experimental archive checkpoint. The current branch is NOT production-ready, NOT release-quality, and NOT a proof of SDI viability. It is a documented experimental record that can be used as source material for a clean substitutive tensor-replacement prototype.

---

**Tag (local only, not pushed):** `SDI_PRT_EXPERIMENTAL_ARCHIVE_CHECKPOINT`