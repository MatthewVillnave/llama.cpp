# PRT Phase 12D: Failure Report


**1 issue(s) detected.**

| Policy | Prompt | Issues |
|--------|--------|--------|

| L12 only | Explain PRT in one paragraph without hype. | coherence=FAIL |

## Safety Gate Status

- callback_overwrites == 0: **✅ PASS**

- identity_fallback_calls == 0: **✅ PASS** (not triggered in any run)

- No SIGKILL: **✅ PASS** (all 88 runs completed)

- No memory corruption: **✅ PASS**


## JSON Behavior Note

JSON prompts (P4, P5) fail on ALL policies including native. This is a model generation issue (Qwen2.5-3B-Instruct produces malformed JSON), not a PRT failure.
