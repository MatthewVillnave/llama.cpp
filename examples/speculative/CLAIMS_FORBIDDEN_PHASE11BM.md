# Claims Forbidden: Phase 11BM

**Version:** 1.0
**Date:** 2026-05-02
**Scope:** Route A + L12+L15 Fallback Policy

---

## Forbidden Claims

### Production Claims
- ~~"production-ready"~~
- ~~"production-viable"~~
- ~~"production-safe"~~
- ~~"ready for deployment"~~

**Why:** Limited prompt testing, JSON/structured not fully validated, hardware constraints on test machine.

### Universal Speedup Claims
- ~~"universal 2x speedup"~~
- ~~"consistent 2x speedup across all prompts"~~
- ~~"2x speedup guaranteed"~~

**Why:** Observed speedup is ~1.85x, varies by prompt. Not tested on all prompt types.

### Full Suite Claims
- ~~"full broader suite passed"~~
- ~~"all prompts preserve native output"~~
- ~~"universally safe"~~
- ~~"no edge cases remain"~~

**Why:** Broader suite was resource-constrained. JSON/structured prompts not fully tested. Only 4 narrative + 1 factual + 1 code prompt validated.

### Pure Route A Claims
- ~~"pure all36 Route A is safe"~~
- ~~"Route A all36 works on all prompts"~~
- ~~"Route A without fallback is production-viable"~~

**Why:** Pure Route A all36 FAILS on "Once upon a time in a" — produces "far away land" instead of "small village". Fallback is REQUIRED.

### Complete Validation Claims
- ~~"JSON/structured fully validated"~~
- ~~"all structured outputs are valid JSON"~~
- ~~"markdown tables are syntactically correct"~~

**Why:** JSON/structured prompts SIGKILL at n>30 due to memory constraints. Not fully tested.

### Callback Path Claims
- ~~"callback path is safe"~~
- ~~"callback modes recommended"~~
- ~~"callback correction is production-viable"~~

**Why:** Callback modes achieve correctness by overwriting PRT output with native. Destroys speed path. Forbidden in RC policy.

### Exact Match Claims
- ~~"exact native output on all prompts"~~
- ~~"byte-for-byte identical to native"~~
- ~~"always produces native-equivalent output"~~

**Why:** Only confirmed on specific tested prompts. "distant galaxy" produces different (but coherent) continuation.

---

## Forbidden Phrasing

Do NOT use:
- "production-ready"
- "universally safe"
- "2x speedup" (use "~1.85x")
- "all prompts" (use "tested prompts")
- "fully validated" (use "partially validated" or "on tested prompts")
- "no known issues" (use "no issues on tested prompts")

---

## Policy Reminder

This is a **release-candidate experimental branch** with a **production-candidate fallback policy**.

Claim only what was tested. Be specific about scope. Do not overgeneralize.

---

*End of Forbidden Claims*
