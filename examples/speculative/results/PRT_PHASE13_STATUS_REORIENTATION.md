# PRT Phase 13 Status: Reorientation

## 1. Current Branch/State

**Branch:** `experimental/prt-phase13-model-generalization`
**HEAD:** `5fffcc721` — "PRT: document low-disk test IO protocol"

**Git status:** 88 untracked files in working tree (result docs, test code, old artifacts — no models, no sidecars, no large binaries)

**Recent commits (last 10):**
```
5fffcc721 PRT: document low-disk test IO protocol
a3d53e9e8 PRT Phase 13I: run timeout-safe 0.5B minisuite
761b1352c PRT Phase 13H-R: rerun quality timing with llama-cli
70e9d84bc PRT Phase 13H-R: Quality/timing comparison for 8 prompts
0bede7146 PRT Phase 13H: run clean 0.5B quality timing comparison
62212c2e6 PRT Phase 13G: add clean llama-cli frontend canary
a96e3bf91 PRT Phase 13E-R: isolate disabled-mode regression - FAIL
ef4a0c16d PRT Phase 13E: active 0.5B canary - FAIL
```

**PRT tags:**
```
PRT_ROUTE_A_RC1
PRT_PHASE12_VALIDATION_CHECKPOINT
PRT_PHASE12E_L11_L15_CHECKPOINT
```

---

## 2. Machine Health After Cleanup

| Item | Value |
|------|-------|
| Disk free | 170GB (24% used) |
| RAM available | 11GB / 15GB total |
| RAM used | 4.2GB |
| Swap used | 1.7GB / 4GB |
| Remaining llama processes | Only Ollama daemon (1550534, system service) |
| Hung processes | 0 after cleanup |
| Temp files removed | ~600MB of phase13*.txt, prt_out*.txt, etc. |
| /tmp/prt_sidecars/ preserved | YES |

---

## 3. Why Phase 13J Is Blocked

**Verdict:** `BLOCKED_BUILD_MODEL_LOAD_HANG`

**Details:**
- Binary: `./build/bin/llama-cli` (fresh rebuild with `LLAMA_BUILD_SERVER=ON`)
- Model: `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (398MB, file intact)
- Hung processes: 1 llama-cli process at 100% CPU, 0 tokens generated
- Native llama-cli model-load: **HUNG** — generation loop stalls after prompt eval
- Active PRT attempted: **NO** — blocked by model-load hang
- Any valid 13J result: **NO**

**Phase 13J is NOT:**
- NOT a PRT failure — no PRT code was running
- NOT a sidecar failure — no sidecars loaded
- NOT a quality result — no valid output produced
- NOT model corruption — same model works in llama-simple and llama-bench

---

## 4. Known-Good Ladder

### Phase 13G
- **Commit:** `62212c2e6`
- **Verdict:** PASS
- Clean llama-cli frontend with PRT flags built
- Native no-flags clean
- `--prt-mode 0` clean
- Active `--prt-mode 5700` ran
- Sidecars loaded 24/24
- PRT_SHAPE: n_layer=24, M=896, N=4864
- No garbage/path fragments

### Phase 13H
- **Commit:** `0bede7146`
- **Verdict:** INVALID_WRONG_BINARY
- Used `llama-simple` instead of `llama-cli`
- **Do not treat as a PRT failure or quality claim**

### Phase 13H-R
- **Commit:** `761b1352c`
- **Verdict:** PASS/PARTIAL
- Clean llama-cli rerun, 1/8 prompts completed
- Native clean 1/1
- PRT active clean 1/1
- No flag echo, no path fragments, counters clean

### Phase 13I
- **Commit:** `a3d53e9e8`
- **Verdict:** PASS
- Timeout-safe mini-suite, 4/4 prompts clean
- Native and PRT active both clean
- Low-disk I/O protocol adopted: stdout → /dev/null, stderr captured only
- No more GB-sized stdout files

### Phase 13J
- **Verdict:** BLOCKED_BUILD_MODEL_LOAD_HANG
- Full 8-prompt n=40 suite not completed
- No valid quality/timing claim

---

## 5. Current Allowed Claims

- PRT Route A 3B Phase 12/12E results stand unless separately disproven
- Phase 13G created a clean llama-cli PRT frontend
- Phase 13H llama-simple result is **invalid** (wrong binary)
- Phase 13H-R shows active 0.5B PRT can run cleanly through llama-cli on at least one prompt
- Phase 13I shows native and active PRT both clean on 4-prompt timeout-safe mini-suite
- Phase 13J is blocked by model-load/build/runtime hang

## Current Forbidden Claims

- Do not claim full 8-prompt 0.5B PRT validation
- Do not claim 0.5B speedup or quality metrics
- Do not claim general model support beyond what's validated
- Do not claim production readiness for 0.5B model
- Do not treat llama-simple output as valid PRT result
- Do not treat 13J hang as a PRT quality failure

---

## 6. Best Safe Resume Point

**Best safe resume commit:** `761b1352c` (Phase 13H-R)

**Reason:** 13H-R is the last commit where llama-cli ran cleanly with active PRT on the 0.5B model, completing 1 prompt with proper output. It has clean binary, clean output, and no hang.

**Alternative:** `a3d53e9e8` (Phase 13I) — last commit with clean native+PRT results on 4/4 prompts.

**Current HEAD state:** `5fffcc721` is clean (docs only), but the build has model-load issues. The HEAD is not dirty — the build is the problem, not the source tree.

**Recommendation:** Create a new branch from `761b1352c` for Phase 13K canary, or test from current HEAD after diagnosing the model-load hang.

---

## 7. Next Safe Resume Phase: 13K

**Do NOT resume with full Phase 13J.**

Phase 13K: Native model-load isolation

**Goal:** Prove current-build `./build/bin/llama-cli` can load Qwen2.5-0.5B reliably before any PRT.

**Minimal test only:**
- n_predict=1
- prompt: "test"
- stdout → /dev/null
- stderr captured
- timeout wrapper
- no PRT flags, no sidecars, no active canary

Compare:
- A. Current HEAD native llama-cli
- B. Commit `761b1352c` native llama-cli (known good)

Only after current native model-load is confirmed stable should Phase 13J be retried.

---

## 8. Safety Checklist

- [x] No model .gguf files staged or tracked
- [x] No sidecars staged or tracked
- [x] No large binaries staged or tracked
- [x] No huge logs staged
- [x] No secrets in git
- [x] Tags untouched (PRT_ROUTE_A_RC1, PRT_PHASE12_VALIDATION_CHECKPOINT, PRT_PHASE12E_L11_L15_CHECKPOINT)
- [x] /tmp/prt_sidecars/ preserved
- [x] Disk: 170GB free, healthy
- [x] RAM: 11GB available, 4.2GB used

**Git-tracked model files:** None (checked via `git ls-files`)

**Secrets check:** No API keys, tokens, or passwords in examples/speculative/, common/, tools/, or src/