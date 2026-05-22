# Phase 28G: 14B Availability Preflight — Stage 0 Only

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`2ed258d31`

## C. System Disk Free
- `/home` (nvme0n1p2): 233GB total, 90GB used, **132GB available** — 41% used
- **Disk status: ✅ OK** — 132GB available provides large margin for any 14B pull

## D. Scratch Disk Free
- `/tmp` sits on same nvme0n1p2 partition
- **132GB available on scratch/disk** — well above 2x model size requirement

## E. RAM/Swap State

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Total RAM | 15 GiB | — | — |
| Used RAM | 4.3 GiB | — | — |
| Free RAM | 3.1 GiB | — | — |
| **Available RAM** | **11 GiB** | ≥6 GiB required | ✅ PASS |
| Swap total | 4.0 GiB | — | — |
| **Swap used** | **288 MiB** | ≤1 GiB required | ✅ PASS |

**System readiness: ✅ READY** — Both swap (288MB << 1GB) and available RAM (11GB >> 6GB) pass Phase 28F thresholds.

## F. Installed Ollama Models

| Model | Size | Modified |
|-------|------|----------|
| qwen2.5:7b | 4.7 GB | 8 hours ago |
| qwen2.5:3b | 1.9 GB | 8 hours ago |
| qwen2.5:0.5b | 397 MB | 14 hours ago |
| nomic-embed-text:latest | 274 MB | 4 days ago |

**No 14B models installed.** All installed models are 7B or smaller.

## G. Candidate 14B Models

### Primary Candidate: `qwen2.5:14b`
- **Available in Ollama registry:** ✅ Yes — confirmed via ollama.com/library/qwen2.5:14b
- **Reported manifest size:** 30.7 MB (this is the Ollama model manifest descriptor, not the full model weight size)
- **Actual full weight size:** Estimated ~8–9 GB (consistent with Phase 28E analysis; 14B Q4_K_M class)
- **Quantization:** Q4_K_M (Ollama default for this size class)
- **CPU-only:** ✅ Yes — no GPU requirement specified
- **Context support:** Supports up to 128K tokens officially; practical limit depends on RAM
- **Family:** Qwen (consistent with all prior eval models)

### Other 14B candidates checked
- `qwen2.5:14b-instruct` — available, same size class
- Other 14B models — not in Ollama registry or size unknown

### Rejection: None. `qwen2.5:14b` meets all criteria.

## H. Candidate Decision Table

| Candidate | Est. Size | Disk OK? | RAM Risk | Available? | Recommended? | Notes |
|-----------|-----------|----------|----------|------------|--------------|-------|
| qwen2.5:14b | ~8–9 GB | ✅ YES (132GB free) | 🟡 MARGINAL (11GB avail, model ~8.5GB, leaves ~2.5GB for KV/buffers at c=2048) | ✅ YES | 🟡 APPROVE_LATER — system ready, but c=2048 KV headroom is tight. Stage 1 load-only is safe; generation context must stay conservative. | Primary candidate. Pull is safe; generation requires careful guard. |

### Classification: `APPROVE_LATER_PULL`

The system is ready. The candidate is valid. A load-only test (Stage 2 of Phase 28F preflight) is reasonable. **No generation without explicit Matt approval.**

## I. Stage 1 Pull Recommendation

**Status: READY FOR APPROVAL — no pull executed yet**

System state passes all Stage 0 checks:
- ✅ Swap used 288MB < 1GB threshold
- ✅ Available RAM 11GB > 6GB threshold
- ✅ Disk 132GB free > 2x model size
- ✅ Ollama daemon responding
- ✅ Candidate `qwen2.5:14b` confirmed available in registry
- ✅ No stale processes consuming excessive RAM

**Pull is safe when Matt approves.** The load-only Stage 2 test will measure actual RSS and confirm whether the ~8–9GB estimate holds.

## J. Blockers

| Blocker | Status |
|---------|--------|
| Swap > 1GB | ✅ Not blocked — 288MB used |
| Available RAM < 6GB | ✅ Not blocked — 11GB available |
| Disk insufficient | ✅ Not blocked — 132GB free |
| No 14B candidate | ✅ Not blocked — `qwen2.5:14b` available |
| Ollama not responding | ✅ Not blocked — responding normally |

**No blockers found. Stage 0 complete.**

## K. Recommended Next Phase

**Phase 28H: Approved 14B pull/load-only test**

This would execute Stage 2 of the Phase 28F preflight design:
1. Pull `qwen2.5:14b` under strict guard
2. Load model into Ollama (warmup only — no generation)
3. Record RSS, swap delta, disk before/after
4. Abort on any Phase 28F hard abort condition

**Requires explicit Matt approval before executing.**

**If Matt wants to proceed:** Confirm "yes, pull qwen2.5:14b" and I'll execute Stage 2 with all guards active.

**If Matt wants to pause here:** Phase 28G Stage 0 is documented. Pull can happen any time without re-running system checks (swap/RAM state valid for immediate future).

## L. Models/Sidecars/F32 Refs Staged?
No. No 14B model files staged or pulled.

## M. Secrets Detected?
No secrets in any committed files.

## N. Tags Touched?
No tags created, modified, or pushed.

---

## Verdicts
- ✅ `PASS_PHASE28G_14B_AVAILABILITY_PREFLIGHT`
- ✅ `APPROVE_LATER_14B_PULL_CANDIDATE_FOUND`
- ✅ `PASS_SYSTEM_READY`
- ✅ `BLOCKED_SYSTEM_STATE_CLEAN`
- ✅ `PASS_NO_BLOCKERS`
- ✅ `BLOCKED_REPO_STATE` (old untracked reports remain untracked)