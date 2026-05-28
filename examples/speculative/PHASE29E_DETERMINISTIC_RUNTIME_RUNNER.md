# Phase 29E: Deterministic Runtime Runner / Real Sidecar Smoke Unblock

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`

**Delegated to:** prt-lab (subagent timed out twice — completed inline)

## Phase 29D Summary
- `16302e5b0` Phase 29D: PARTIAL_CRC_ALIGNED_INJECTION_BLOCKED
- CRC fix verified: all 3 .trit files pass generator-side validation (attn_out 0x903F, ffn_up 0x833F, ffn_down 0x932F)
- Runtime smoke blocked by `--no-conversation` flag + resource contention

## Subagent Status
- `prt_lab_29e` ran 15m2s, timed out twice, produced no artifacts
- Completed inline in main session

## Supported Flag Audit

```bash
./build/bin/llama-cli --help | grep -E "single|seed|temp|top|predict|log"
```

| Flag | Supported | Notes |
|------|-----------|-------|
| `--single-turn` | ✅ | "run conversation for a single turn only" — works, exits after one turn |
| `--seed N` | ✅ | RNG seed |
| `-t N` / `--threads` | ✅ | CPU threads |
| `--temp N` / `--temperature` | ✅ | Temperature |
| `--top-k N` | ✅ | top-k sampling |
| `--top-p N` | ✅ | top-p sampling |
| `--log-disable` | ✅ | Disables logging |
| `-p PROMPT` / `--prompt` | ✅ | Prompt |
| `-n N` / `--predict` | ✅ | Predict count |
| `--no-conversation` | ❌ | Not supported — use `--single-turn` instead |
| `--simple-io` | ✅ | For subprocess compatibility |

**Deterministic runner template:**
```bash
llama-cli -m <model> -p "<prompt>" -n 1 --single-turn --seed 42 -t 0 --log-disable [--temp 0]
```

## Resource Cleanup

No stale llama processes found. Ollama daemon running (unrelated, not touched).

## CRC Verification (Sidecars Still Valid)

Python verification of 29D-fixed .trit files:

| File | Stored CRC | Computed CRC | Match |
|------|-----------|--------------|-------|
| attn_out_layer0.trit | 0x903F | 0x903F | ✅ |
| ffn_up_layer0.trit | 0x833F | 0x833F | ✅ |
| ffn_down_layer0.trit | 0x932F | 0x932F | ✅ |

## Runtime Smoke Tests

All runs used: Qwen2.5-0.5B-Instruct-Q4_K_M.gguf, prompt "4+4=", n=1, seed=42, temp=0, single-turn

### A. Baseline (no pager)
```
build: b9167-16302e5b0
token: 4
exit: 0
```

### B. Observe-only (pager enabled, manifest valid)
```
--enable-prt-sidecar-pager --prt-sidecar-manifest /tmp/prt_sidecars_0_5b_layer0/manifest.json --prt-sidecar-dir /tmp/prt_sidecars_0_5b_layer0/
exit: 0 | token: 4
Observe: pager loads manifest, manifest schema valid, no apply flag set
Note: PRT log shows [PRT-INJECT-DOWN] guard_reject flags_disabled for all layers
The observe mode does NOT trigger injection log lines (injection_successes not in output)
```

### C. attn_out scale=0
```
--enable-prt-sidecar-pager --prt-sidecar-manifest ... --prt-sidecar-dir ... \
  --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family attn_out \
  --prt-sidecar-true-injection --prt-sidecar-scale 0.0
exit: 0 | token: 4
Observe: all [PRT-INJECT-DOWN] lines show action=guard_reject flags_disabled
family= (empty), family_len=0 — family filter NOT passed to pager
```

### D. attn_out scale=1
```
Same flags, scale=1.0
exit: 0 | token: 4
Same guard_reject pattern — injection not firing
```

### E. ffn_up scale=1
```
exit: 0 | token: 4
Same guard_reject pattern
```

### F. ffn_down scale=1
```
exit: 0 | token: 4
Same guard_reject pattern
```

### G. Budget=0
```
--prt-sidecar-budget-mb 0
exit: 0 | token: 4
No rejection — budget=0 not enforced (may be a config issue)
```

### H. Missing manifest
```
--enable-prt-sidecar-pager --prt-sidecar-manifest /tmp/nonexistent.json
exit: 1 ✅
Deterministic failure confirmed
```

## Root Cause of Injection Block

**Runtime bug: family filter not passed from CLI to pager**

All `[PRT-INJECT-DOWN]` log lines show:
```
family=         (empty string)
family_len=0    (zero)
action=guard_reject flags_disabled
```

Even when `--prt-sidecar-apply-family attn_out` is passed, the pager receives `family=` (empty). The CLI parses the flag but the value is not propagated to the pager's injection logic.

This is why all injection attempts hit `guard_reject` — the family filter is `""` (matches nothing), so all layers are rejected before the sidecar is even considered.

The `flags_disabled` means `g_apply=0` and `g_true_inj=0` — the global apply/true-injection flags are not set in the pager's config, which should be set by `--prt-sidecar-apply` and `--prt-sidecar-true-injection`.

## Classification

**BLOCKED_CLI_UNSUPPORTED** — No. The `--no-conversation` issue is solved with `--single-turn`. The real blocker is the `--prt-sidecar-apply-family` flag value not propagating to the pager's injection path. This is a runtime bug.

**BLOCKED_RUNTIME_BUG** — family filter and global injection flags not reaching the pager.

## 29B-R Unblocked Status

**29B-R requires:** `trit_validated=1` + `injection_successes>=1` for real sidecars

- ✅ CRC fix complete and verified (generator-side)
- ✅ Runtime recognizes pager flags and loads manifest
- ✅ All 3 .trit files exist and are CRC-valid
- ❌ `trit_validated` counter not confirmed in output (needs debug probe or probe tool)
- ❌ `injection_successes` blocked by runtime bug (family filter not propagated)
- ❌ Budget=0 not enforced

**29B-R unblocked if:** `--prt-sidecar-apply-family` propagation bug is fixed in the runtime

## Key Findings

1. **`--single-turn`** replaces `--no-conversation` (works ✅)
2. **Generator-side CRC fix is verified** — all 3 layer0 .trit files compute correctly
3. **Runtime has a bug** — `--prt-sidecar-apply-family` value doesn't reach the pager; family is always empty, causing all injection attempts to be rejected
4. **Global apply flags (`g_apply`, `g_true_inj`)** are also not being set despite CLI flags being recognized
5. **Budget=0 enforcement** not working — may be separate config issue

## Next Steps

1. **Fix `--prt-sidecar-apply-family` propagation** — trace where CLI flag value is lost between argument parsing and pager injection config
2. **Add `trit_validated` counter to PRT log output** — currently only visible via probe tool
3. **Validate budget=0 enforcement** separately
4. **Re-run A-H smoke** after bug fix — expect `trit_validated=3` and `injection_successes>=1` for D/E/F

## Branch / HEAD
- **Old HEAD:** `16302e5b0` Phase 29D: align trit CRC validation
- **New HEAD:** `16302e5b0` (no change — no commit this phase, runtime bug blocks progression)