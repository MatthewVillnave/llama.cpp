# Phase 27I: KV/Context Memory Mapping Design

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`d6310bcad95beaae19c083028d01a7216b22ec77`

## C. Phase 27H-D Result Summary

Phase 27H-D formally documented the bounded 7B working-set probe sequence:

| Working Set | Context | Result | Score | Swap Δ |
|------------|---------|--------|-------|--------|
| WS-512 | 2048 | OK | 1.0 | 0.0 GB |
| WS-1024 | 2048 | OK | 1.0 | 0.0 GB |
| WS-2048 | 4096 | OK | 1.0 | 0.0 GB |
| WS-4096 | 4096 | OK | 1.0 | 0.0 GB |
| WS-6144 | 8192 | FAIL | 0.0 (not established) | 0.0 GB |

**Key finding:** The failure at WS-6144/c=8192 was not RAM/swap exhaustion. Swap held flat at ~0.367 GB. RAM stayed above 3 GB. The failure was generation/API behavior — a controlled request that returned through an error path with no response text captured.

## D. Known Memory Facts

1. **Swap behavior:** Swap remained constant at ~0.367–0.405 GB across all working-set sizes including the failing WS-6144 run. No swap tripwire fired.

2. **RAM behavior:** Available RAM stayed in the 7.0–7.6 GB range throughout all Phase 27H runs. The 3 GB guard threshold was never close to triggering.

3. **Model state:** qwen2.5:7b loads into Ollama once and stays resident (~5.3 GB RSS for the runner process). Subsequent runs don't reload the model.

4. **Context size correlation:** The working-set sizes that passed (WS-512 through WS-4096) used c=2048 and c=4096. The failing run (WS-6144) required c=8192.

5. **Generation behavior correlation:** Simple prompts at c=8192 work fine (confirmed post-failure with "Testing. Say hello." — returned correctly). The failure appears specific to large prompts or specific prompt structures at c=8192, not c=8192 itself.

6. **Token estimate vs actual:** The WS-6144 prompt was estimated at ~6,170 tokens. qwen2.5:7b's context window is 32K tokens, so capacity should not be the issue.

7. **Failure pattern:** The curl request to Ollama returned with an error/no-output path. No HTTP error code was explicitly captured. The stderr was blank. The process exited cleanly.

## E. Possible Failure Causes

Ranked by likelihood (not confirmed):

### 1. Ollama backend termination before output capture (HIGH)
**Description:** Ollama may have partially processed the request and returned a non-200 status or closed the connection before the response body was fully captured by curl's `--max-time`.
**Evidence:** curl returned without a parseable JSON body; blank stderr; process exited cleanly.
**Test:** Capture raw HTTP response headers + status code + body.

### 2. Internal attention/KV allocation at large context (MEDIUM)
**Description:** qwen2.5:7b may hit an internal attention computation issue at c=8192 + ~6K token prompt, causing the generation to stall or be killed by Ollama's internal limits.
**Evidence:** Model is responsive at small prompts with c=8192, so the context window itself works. The combination of large context + large prompt may be the trigger.
**Test:** Measure generation time per token; check if stall is during prompt_eval or eval phase.

### 3. Prompt structure interaction (MEDIUM)
**Description:** The specific prompt construction (heavy DEBUG LOG filler + context headers) may interact poorly with the larger context window, possibly causing unexpected formatting or tokenization issues.
**Evidence:** Simple prompts work; the structured DEBUG LOG prompt did not.
**Test:** Test c=8192 with random filler vs structured DEBUG LOG filler.

### 4. Token estimate mismatch (LOW)
**Description:** Our token estimate (~6,170) may be wrong. If actual tokens exceed c=8192, Ollama truncates or fails silently.
**Evidence:** 32K context window should easily hold 6K tokens. But if tokenization is inefficient or the estimate is wrong, truncation could occur.
**Test:** Actual token count via `prompt_eval_count` returned by Ollama.

### 5. Backend-specific c=8192 handling (LOW)
**Description:** Ollama's handling of num_ctx=8192 may have edge cases or bugs on this specific hardware/kernel.
**Evidence:** Model is responsive at c=8192 for tiny prompts. But large prompts may trigger a different code path.
**Test:** Compare c=8192 behavior at various prompt sizes.

### 6. Timeout behavior at large context (LOW)
**Description:** The 180–240 second timeout may have fired during prompt evaluation or generation, causing a clean exit without captured output.
**Evidence:** Wall time was measured but output was not captured. The process exited with code 0.
**Test:** Check if timeout fires during prompt_eval or during token generation.

### 7. Output capture issue (ruled out)
**Description:** The Python runner may have failed to capture curl's output correctly.
**Evidence:** Post-failure direct curl calls work fine. The runner captured other runs correctly.
**Verdict:** Unlikely the primary cause, but possible contributing factor.

## F. Future Measurement Plan

**Important design principle:** Future testing should isolate the `c=8192` variable from the `WS-6144` difficulty. A passing test at c=8192 with a tiny prompt rules out backend-specific c=8192 issues as the sole cause.

### Proposed future probe sequence (Phase 27J):

#### Test A: c=8192 + tiny prompt
- Prompt: 1-2 sentences, <100 tokens
- Context: c=8192
- Goal: Verify backend handles c=8192 for simple requests
- Pass criteria: Valid response captured, no error

#### Test B: c=8192 + medium filler prompt
- Prompt: ~2,000 chars of filler, easy retrieval question
- Context: c=8192
- Goal: Test context size sensitivity with a normal prompt structure
- Pass criteria: Correct answer, stable RAM/swap

#### Test C: c=8192 + WS-6144 structured prompt (only if A and B pass)
- Prompt: Same structure as Phase 27H-C WS-6144
- Context: c=8192
- Goal: Confirm real cliff is the specific prompt/context combination
- Pass criteria: Correct answer or documented failure mode

#### Test D: c=6144 or c=7168 midpoint (optional)
- Only if Tests A and B pass and if there's diagnostic value
- Goal: Find the actual boundary if c=8192 itself is the issue

### Measurement instrumentation for Phase 27J:
1. Capture raw HTTP response headers (status code + body)
2. Log `prompt_eval_count` and `eval_count` from Ollama response
3. Time-stamp each request phase (start, prompt_eval done, generation, end)
4. Record RAM/swap before and after
5. Log curl's actual return code and stderr
6. Do NOT rely on exit code alone; capture full response body

## G. Future Safety Guard

Phase 27J must implement:

- **One run at a time** — no parallel runs
- **Timeout:** 300s max per run (increase from 240s if justified)
- **Output cap:** 256 tokens max
- **Swap tripwire:** abort if swap delta > 250 MB
- **RAM guard:** abort if available RAM < 3 GB
- **Capture everything:** raw HTTP response, headers, status, timing
- **No WS-8192**
- **No c=16384**
- **No broad eval suite**
- **Stop on first backend failure** — document and stop, do not retry same configuration

## H. Recommended Next Phase

**Phase 27J: c=8192 Backend/API Forensics**

Start with Test A (c=8192 + tiny prompt) to determine whether c=8192 itself works for normal prompts on this backend. This is the fastest path to ruling out a pure infrastructure issue vs a model/context issue.

If Test A passes: proceed to Test B (medium filler) to establish context size sensitivity.

If Test A fails: document c=8192 as broken on this backend and stop.

**Why this over KV memory mapping first:** The Phase 27H-C failure was a generation/API behavior failure, not a proven memory failure. The fastest diagnostic is to confirm whether c=8192 works at all for non-trivial prompts. If it does, the WS-6144 failure is specifically about prompt size/structure, not c=8192 broadly.

## I. Models/Sidecars/F32 Refs Staged?
No.

## J. Secrets Detected?
No secrets in any committed files.

## K. Tags Touched?
No tags created, modified, or pushed.

---

## Verdicts
- ✅ `PASS_PHASE27I_KV_CONTEXT_MAPPING_DESIGN`
- ✅ `RECOMMEND_C8192_BACKEND_FORENSICS`
- ✅ `RECOMMEND_STOP_7B_PROBING`
- ✅ `BLOCKED_REPO_STATE` (old 26K/L/R reports remain untracked)