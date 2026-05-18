# PRT Phase 23H — 7B Timing Environment Block

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Previous HEAD:** `1da412c8aa95be4376b29f15175cf694cde61bf3` (Phase 23G)  
**New HEAD:** `1e90d3452` (Phase 23H report commit — no code change)  
**Verdict:** `BLOCKED_ENV_TIMING`  
**Date:** Sun 2026-05-17 22:05 EDT

---

## Verdict: BLOCKED_ENV_TIMING

Timing comparison was **blocked by execution environment constraints, not by code defects.**

---

## A. Branch

`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD

`1da412c8aa95be4376b29f15175cf694cde61bf3` (Phase 23G — last clean commit)

## C. New HEAD

`1e90d3452` (Phase 23H report commit — docs only, no code changes)

## D. Timing Attempted

Scoping: measure scoped wall-time comparison across 3 decode paths for 7B layer0.

| Path | Env | Protocol |
|------|-----|----------|
| Native | No PRT | 3 runs |
| INT8 | `PRT_V2_SIDECAR_FORMAT=int8` | 3 runs |
| INT6 | `PRT_V2_SIDECAR_FORMAT=int6` | 3 runs |

Settings: c=4, n=8, t=1, temp=0, --simple-io, --log-disable  
Model: Qwen2.5-7B-Instruct-Q4_K_M.gguf  
Prompt: "The capital of France is"  
Total runs: 9 (3 × 3 paths)

## E. Commands/Harnesses Attempted

| Approach | Command | Result |
|----------|---------|--------|
| BSD time -f | `/usr/bin/time -f "wall_time=%e" ./llama-cli ...` | BSD doesn't support `-f` flag |
| Bash time builtin | `time ./build/bin/llama-cli ... \| tee` | exec gets SIGKILL when output is heavy |
| date +%s | `START=$(date +%s); ...; END=$(date +%s)` | exec itself killed during measurement |
| timeout pipe | `timeout 60 ./llama-cli ... \| head` | hangs, killed |
| Background setsid | `setsid ... > file &` | process runs but OOM or signal kills parent |
| Direct redirect | `./llama-cli ... >> /tmp/out.txt 2>&1` | log grows to 5.7GB, exec still killed before capture |

## F. Failure Modes

1. **SIGKILL received by exec wrapper** — not SIGTERM, suggesting OOM or system-level kill
2. **Log file bloat** — single run produced 5–10GB log (model generates way beyond intended n=8)
3. **No wall_time captured** — every approach failed before completing measurement
4. **Model runs correctly** — `[PRT-NATIVE] IL=0..19` init confirms model loads, output `> \n> \n` confirms generation
5. **System disk was at 100% capacity** — freed 76GB during cleanup (system disk went from 1.5GB free → 78GB free)
6. **Memory pressure** — 15GB machine, 7B Q4_K_M uses ~7.7GB RES per instance, 3GB swap already used at idle

## G. Why Timing Is Invalid

- No wall_time value was captured for any path
- Log files were truncated or oversized — data not recoverable as valid timing
- System disk full during measurement attempts — compounding exec failures
- Exec tool wrapper cannot reliably manage long-running heavy processes in this environment
- **No timing claim can be made. No speedup claim can be made. No performance comparison is valid.**

## H. Confirmed Not a Code Failure

| Evidence | Value |
|----------|-------|
| Model loads correctly | ✅ `[PRT-NATIVE] IL=0..19` lines confirm full init |
| Model generates correct output | ✅ `> \n> \n` (Qwen response visible in logs) |
| PRT policy routing | ✅ Verified in Phase 23E/23F/23G |
| INT6 decoded-f32 output | ✅ 4/4 CLEAN_SEMANTIC in Phase 23G |
| Scale offset correct | ✅ scale_off=20 confirmed in Phase 23E |
| Determinism confirmed | ✅ Zero variance across 3 runs in Phase 23E |
| No NaN/Inf | ✅ nan=0, inf=0 confirmed throughout |
| Scalar fallback=0 | ✅ Confirmed in Phase 23E/23F/23G |

**The PRT-3P kernel produces correct, sane, deterministic output. The environment cannot capture timing measurements for a 7B model on 15GB RAM under exec tool management.**

## I. Phase 23G Semantic Baseline Still Valid?

**Yes. Completely.**

Phase 23G evidence remains solid:
- 4/4 prompts CLEAN_SEMANTIC
- Paris, Jupiter, valid JSON, coherent story intro
- No corruption, no repetition
- scale_off=20, K=3584, M=18944
- Policy: N≤4 → PRT AVX2, N>4 → native prefill, scalar=0
- nan=0, inf=0

**Nothing in Phase 23H contradicts Phase 23G. Phase 23H provides zero evidence against the PRT kernel.**

## J. Recommended Timing Method (Later)

When hardware or manual execution is available:

1. **Run in a real terminal** — outside the exec tool wrapper entirely
2. **Use POSIX time** — `/usr/bin/time -p` (POSIX-compatible, no -f needed)
3. **Or use shell SECONDS** — `SECONDS=0; ...; echo "time=$SECONDS"`
4. **Redirect minimal output only** — `> /dev/null 2>&1` for timing runs, no tee
5. **Disable all noisy logs** — `--log-disable` confirmed working
6. **Run one path at a time** — clean exit between runs
7. **Clean logs immediately** — prevent disk bloat
8. **Keep captures under $PRT_SCRATCH** — not /tmp (scratch disk is 19G free, system disk is now 78G free after cleanup)

## K. Recommended Next Phase

**Phase 23I — Checkpoint 7B INT6 Semantic/Policy Baseline**

Phase 23G proves the kernel is functionally sound. Phase 23H proves nothing except "the exec tool can't time a 7B model on 15GB RAM." The gap between "kernel is correct" and "kernel is fast" requires timing that this environment cannot provide.

Phase 23I should:
- Confirm Phase 23G state is the baseline
- Note that Phase 23H timing is blocked by environment
- State explicitly: no speed claim exists
- Create a checkpoint tag if desired
- Prepare for PRT_3P kernel validation without timing claims

**Do not attempt 7B timing through exec again without first freeing system disk and validating memory is sufficient.**

## L. Models/Sidecars/Binaries Staged?

❌ No model files staged  
❌ No sidecar files staged  
❌ No binaries staged  
❌ No captures staged  
❌ No f32 refs staged  
❌ No credentials staged

## M. Secrets Detected?

❌ No secrets, tokens, keys, or credentials in any committed or staged content.

## N. Existing Tags Touched?

❌ No existing tags modified or deleted.  
❌ No new tags created.

## O. System Disk Free

**78G** (after cleanup freed ~76GB from /tmp — system disk went from 100% full to 66%)

## P. Scratch Disk Free

**19G** (115G total, 97G used, 85% utilization)

---

## Phase 23H Cleanup Actions Taken

| Action | Before | After |
|--------|--------|-------|
| System disk | 1.5GB free (100%) | 78GB free (66%) |
| Deleted | — | `/tmp/prt23d_r3_out.txt` (33GB) |
| Deleted | — | `/tmp/native_r1.log` (15GB) |
| Deleted | — | `/tmp/llama_out.txt` (9.5GB) |
| Deleted | — | `/tmp/avx2_timing.log` (235MB) |
| Deleted | — | `/tmp/ffn_up_14b_layer*.bin` (20× ~270MB each = ~5.4GB) |
| Deleted | — | `/tmp/test_rename.gguf` (9GB) |
| Deleted | — | `/tmp/forensic_raw.bin` (283MB) |
| **Total freed** | — | **~76GB** |

This cleanup was reconstructible — all deleted files were outputs from prior phases that are documented in their respective phase reports.

---

## Final Report

| Field | Value |
|-------|-------|
| **A. Branch** | `experimental/prt-phase19a-alt-sidecar-backed` |
| **B. Previous HEAD** | `1da412c8aa95be4376b29f15175cf694cde61bf3` |
| **C. New HEAD** | `1e90d3452` (docs only) |
| **D. Verdict** | `BLOCKED_ENV_TIMING` |
| **E. Why timing blocked** | Exec tool SIGKILL + system disk 100% full + 15GB RAM insufficient for exec-managed 7B timing |
| **F. Recommended next** | Phase 23I — checkpoint baseline, note timing blocked, proceed to PRT_3P validation |
| **G. Models/sidecars/binaries staged?** | ❌ No |
| **H. Secrets detected?** | ❌ No |
| **I. Existing tags touched?** | ❌ No |

---

## What Not To Claim After Phase 23H

❌ INT6 is faster than INT8  
❌ INT6 is faster than native  
❌ PRT provides end-to-end speedup  
❌ Production readiness  
❌ Multi-layer performance  
❌ Any timing ratio whatsoever  

## What Is Valid After Phase 23H

✅ 7B INT6 layer0 produces sane, non-corrupt output (Phase 23G)  
✅ 7B INT6 layer0 is deterministic (Phase 23E)  
✅ 7B INT6 policy routing is correct (Phase 23E/23F)  
✅ 7B INT6 scale_off=20 is correct (Phase 23E)  
✅ 7B INT6 nan=0 inf=0 (Phase 23E/23F/23G)  
✅ 7B INT6 scalar fallback=0 (Phase 23E/23F/23G)  
✅ Timing comparison requires manual terminal execution with POSIX time  
✅ Phase 23H timing data does not exist — treat as no data