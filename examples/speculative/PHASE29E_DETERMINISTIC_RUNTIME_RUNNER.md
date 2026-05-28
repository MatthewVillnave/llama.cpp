# Phase 29E: Deterministic Runtime Runner / Real Sidecar Smoke Unblock

## Branch & State
- **Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
- **Old HEAD:** `16302e5b0` (Phase 29D)
- **New HEAD:** `2a8f333d0` (Phase 30C) — committed after 29E work
- **Classification:** `PARTIAL_RUNNER_FOUND_INJECTION_BLOCKED`

## Phase 29D Classification
`PARTIAL_CRC_ALIGNED_INJECTION_BLOCKED` — CRC validated offline but runtime smoke blocked.

## What Was Fixed vs Blocked

### 29D Fixes (VERIFIED):
- .trit CRC/header generation compatible with runtime (32-byte header, CRC over bytes 0-29)
- Manifest schema aligned to Phase28Y runtime parser format
- CRC validation passes offline for all 3 layer0 sidecars:
  - `attn_out_layer0.trit`: CRC=0x903F ✅
  - `ffn_up_layer0.trit`: CRC=0x833F ✅  
  - `ffn_down_layer0.trit`: CRC=0x932F ✅

### Remaining Blocker: CLI FLAG PROPAGATION FAILURE

Even with all flags `--prt-sidecar-apply`, `--prt-sidecar-true-injection`, `--prt-sidecar-scale 1.0`, 
the globals `g_prt_sidecar_apply_enabled` and `g_prt_sidecar_true_injection_enabled` remain FALSE.

**Evidence:** Every `[PRT-INJECT-DOWN-DEBUG]` log entry shows:
```
g_true_inj=0 g_apply=0 family= family_len=0
```
Output token is identical across baseline, observe, and all injection attempts: `2+2=4` (token id=17).

---

## Supported Runner Flags (llama-cli b9168)

| Required Flag | Supported | Notes |
|---|---|---|
| `--no-conversation` | ❌ NOT SUPPORTED | Pre-existing, llama-cli bd45130d7 |
| `--single-turn` | ✅ Supported | `-st` / `--single-turn` |
| `--seed` | ✅ Supported | deterministic sampling |
| `--temp 0` | ✅ Supported | zero randomness |
| `--top-k 1` | ✅ Supported | greedy |
| `--top-p` | ✅ Supported | nucleus sampling |
| `--log-disable` | ✅ Supported | suppress chat template noise |
| `-n` (predict count) | ✅ Supported | tokens to generate |
| `-p` (prompt) | ✅ Supported | prompt string |
| `-t 0` (threads) | ✅ Supported | use all threads |
| `--enable-prt-sidecar-pager` | ✅ Supported | enable pager |
| `--prt-sidecar-dir` | ✅ Supported | sidecar directory |
| `--prt-sidecar-manifest` | ✅ Supported | manifest path |
| `--prt-sidecar-apply` | ✅ Supported | but NOT propagating |
| `--prt-sidecar-apply-family` | ✅ Supported | family filter |
| `--prt-sidecar-apply-layer` | ✅ Supported | layer filter |
| `--prt-sidecar-true-injection` | ✅ Supported | but NOT propagating |
| `--prt-sidecar-scale` | ✅ Supported | scale factor |
| `--prt-sidecar-checksum` | ✅ Supported | CRC validation |
| `--prt-log-level` | ✅ Supported | debug/summary/quiet |

## Final Deterministic Command Template

```bash
./build/bin/llama-cli \
  -m /home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "2+2=" -n 6 -t 0 --seed 42 --single-turn \
  --temp 0 --top-k 1 --log-disable \
  [--enable-prt-sidecar-pager --prt-sidecar-dir /tmp/prt_sidecars_0_5b_layer0 \
   --prt-sidecar-manifest /tmp/prt_sidecars_0_5b_layer0/manifest.json \
   [--prt-sidecar-apply] [--prt-sidecar-true-injection] [--prt-sidecar-scale 1.0]]
```

---

## Smoke Test Results (A-H)

| Test | Description | Expected | Actual | Status |
|---|---|---|---|---|
| A | Baseline (no pager) | `2+2=4` | `2+2=4` token_id=17 | ✅ PASS |
| B | Observe-only (pager, no apply) | `2+2=4` | `2+2=4` token_id=17 | ✅ PASS |
| C | attn_out scale=0 | `2+2=4` no-op | `2+2=4` token_id=17 | ✅ PASS (no-op confirmed) |
| D | attn_out scale=1 | injection ≥1 | `2+2=4` token_id=17 | ❌ BLOCKED |
| E | ffn_up scale=1 | injection ≥1 | `2+2=4` token_id=17 | ❌ BLOCKED |
| F | ffn_down scale=1 | injection ≥1 | `2+2=4` token_id=17 | ❌ BLOCKED |
| G | budget=0 | reject | `2+2=4` token_id=17 | ❌ BLOCKED |
| H | Missing manifest | fail | `error: --enable-prt-sidecar-pager requires --prt-sidecar-manifest` | ✅ PASS |

**Key observation:** Output is byte-for-byte identical across ALL tests including baseline. 
The token sequence `17, 10, 17, 28, 19, 151645` is stable — proving NO injection is occurring.

**Root cause:** `g_prt_sidecar_apply_enabled` and `g_prt_sidecar_true_injection_enabled` are never 
set to TRUE despite --prt-sidecar-apply and --prt-sidecar-true-injection flags being accepted.

---

## trit_validated / injection_successes

| Metric | Value | Evidence |
|---|---|---|
| trit_validated | NULL | No PRT-PAGER init messages; no trit_validated counter output |
| injection_successes | 0 | g_true_inj=0 g_apply=0 on all 24 layers |
| sidecar_math_influenced_output | 0 | Output identical to baseline |

---

## Resource Cleanup

```bash
$ ps aux | grep -E 'llama|phase29|prt' | grep -v grep
ollama  3091  0.0  0.1 2677780 23224 ?  Ssl  May27  0:05 /usr/local/bin/ollama serve
```

**Result:** Only ollama daemon running — no stale llama-cli or phase29 processes.
No cleanup needed.

---

## Classification Breakdown

| Classification | Status |
|---|---|
| `PASS_RUNTIME_SMOKE_UNBLOCKED` | ❌ NO — injection confirmed blocked |
| `PARTIAL_RUNNER_FOUND_INJECTION_BLOCKED` | ✅ YES — runner found, injection blocked |
| `BLOCKED_CLI_UNSUPPORTED` | ✅ YES — --no-conversation not supported (pre-existing) |
| `BLOCKED_RESOURCE_CONTENTION` | ❌ NO — no resource contention detected |

**Raw classification:** `PARTIAL_RUNNER_FOUND_INJECTION_BLOCKED + BLOCKED_CLI_UNSUPPORTED`

---

## Phase 29B-R Unblock Status

**29B-R is NOT unblocked.**

Despite Phase 29F fixing the std::string→char[64] ABI propagation, and Phase 29E finding 
a supported deterministic runner, the actual injection pipeline is broken.

**Key evidence:**
1. `llama_set_prt_flags()` is called from cli.cpp (verified via objdump)
2. The function writes to globals (verified via disassembly showing mov to g_prt_sidecar_apply_enabled address)
3. But `g_prt_sidecar_apply_enabled` remains 0 when graph code reads it
4. This suggests either: (a) wrong copy of globals being modified, or (b) the values are reset after set_prt_flags() is called

---

## Next Steps (to unblock 29B-R)

1. **Verify llama_set_prt_flags() actually writes to the right globals** — add debug print in the function itself
2. **Check if ctx_cli.ctx_server.get_llama_context() is modifying globals** during model load
3. **Add a test that reads back the globals immediately after llama_set_prt_flags()** to confirm they were set
4. **Check if --enable-prt-sidecar-pager sets g_prt_pager_enabled before llama_set_prt_flags() is called**

---

## Artifacts

- `examples/speculative/PHASE29E_DETERMINISTIC_RUNTIME_RUNNER.md` — this document
- `examples/speculative/results/phase29e_deterministic_runtime_runner.json` — structured results

