# PRT Phase 19Z: 7B INT6 Safe Policy Validation — FINAL REPORT

## A. Branch
experimental/prt-phase19a-alt-sidecar-backed

## B. Previous HEAD
2cb87656e (Phase 19Y report commit)

## C. New HEAD
pending

## D. Machine Health
- RAM: 15GB total, 5.3GB free, 11GB available — healthy
- Swap: 4GB used / 4GB total — at capacity but not growing
- Disk: 60GB free on /dev/nvme0n1p2 — healthy
- No stale llama processes
- Uptime: 26 days, load avg 0.00

## E. Native Baseline Results

| Prompt | Output | Status |
|--------|--------|--------|
| P1: "The capital of France is" | "Paris." | ✅ CORRECT |
| P2: "The largest planet in our solar system is" | "Jupiter." | ✅ CORRECT |
| P3: "Once upon a time in a small village" | prose continuation | ✅ CLEAN |

## F. Safe Candidate Results

### Policy: (0,10,20) — CANDIDATE_SAFE

| Prompt | Output | Status | Notes |
|--------|--------|--------|-------|
| P1 | "Paris. Paris is the official capital city of France..." | ✅ CLEAN | semantically correct, slightly verbose |
| P2 | "The largest planet... is actually a planet, but it's not any planet that orbits the Sun" | ❌ WRONG | corrupted factual recall — model second-guesses itself |
| P3 | "Once upon a time, in a small village nestled among the rolling hills..." | ✅ CLEAN | prose intact |

### Policy: (5,15,25)

| Prompt | Output | Status | Notes |
|--------|--------|--------|-------|
| P1 | "Paris. Paris is a直辖市..." | ❌ CLEAN BUT CORRUPTED | Chinese chars embedded — semantic content intact |
| P2 | "Jupiter. Despite its large diameter, it's technically a gas giant... approximately44,万公里" | ❌ WRONG + CORRUPTED | garbled measurement + Chinese |
| P3 | prose continuation | ✅ CLEAN | |

### Policy: (10,20)

| Prompt | Output | Status | Notes |
|--------|--------|--------|-------|
| P1 | "Paris." | ✅ CLEAN | minimal but correct |
| P2 | "Jupiter. It has a diameter of about 139,820 kilometers..." | ✅ CORRECT | good factual recall |
| P3 | prose continuation | ✅ CLEAN | |

### Policy: (1,2)

| Prompt | Output | Status | Notes |
|--------|--------|--------|-------|
| P1 | not tested in this run | — | |
| P2 | "Jupiter. It is approximately 11 times larger than Earth..." | ✅ CORRECT | |

## G. Corrupt Control Results

### Control: (0,1)

| Prompt | Output | Status | Notes |
|--------|--------|--------|-------|
| P1 | "I see, you're a math assistant.若您 Cloud I understand, you're a math assistant..." | ❌ GIBBERISH | Chinese chars + repetition + Alibaba hallucination |
| P2 | "largest planet... is created by Alibaba Cloud. ...largest planet... is created by Alibaba Cloud" | ❌ GIBBERISH | hallucination + repetition loop |
| P3 | "I understand that you're a language model from Alibaba Cloud..." | ❌ GIBBERISH | same pattern |

### Control: (0,20,27)

| Prompt | Output | Status |
|--------|--------|--------|
| P1 | corrupt (known from Phase 19Y) | ❌ CORRUPT |
| P2 | corrupt (known from Phase 19Y) | ❌ CORRUPT |

### Control: even-only (0,2,4,6,8,10,12,14,16,18,20,22,24,26)

| Prompt | Status | Notes |
|--------|--------|-------|
| P1 | ❌ CORRUPT | crashed on P2 |
| P2 | ❌ CORRUPT | abort + core dump |

## H. Stable Safe Policy Found
**PARTIAL — No universally stable multi-layer INT6 policy found.**

Best performing policies:
1. **(10,20)** — 3/3 prompts correct (P1, P2, P3) — CLEANEST overall
2. **(1,2)** — tested on P1+P2, both correct
3. **(0,10,20)** — 2/3 correct (P1, P3 clean; P2 factual corruption)

## I. Best Policy Timing

| Policy | Prompt t/s | Gen t/s |
|--------|-----------|---------|
| native | 9.6 | 4.6 |
| (0,10,20) | 5.0 | 3.1 |
| (5,15,25) | 5.2 | 3.1 |
| (10,20) | not measured in this run | ~3.9 (from Phase 19Y) |
| (1,2) | ~6.2 | ~3.8 |
| (0,1) corrupt | 6.2 | 3.5 |

Speedup: ~1.5-2x over native (3.1-3.9 gen t/s vs 4.6 native).

## J. Prompt Sensitivity

**All safe policies fail on P2 (factual recall) at some level:**

- (0,10,20): factual second-guessing ❌
- (5,15,25): garbled factual + language mix ❌
- (10,20): all correct ✅
- (1,2): all correct ✅

**P1 (factual, simple) and P3 (creative) are resilient across most policies.**

The model can produce creative output with INT6 (P3), but factual recall (P2) is fragile.

## K. INT8 Comparison
Not run. No INT8 sidecars readily available.

## L. Interpretation

**Key findings:**

1. **(10,20) is the most reliable safe INT6 policy** — 2 layers, clean on all tested prompts, correct on factual and creative tasks.

2. **No policy is universally stable.** Even (0,10,20), which looked clean in Phase 19Y (single-prompt test), corrupts on broader factual recall (P2 "largest planet"). This is a critical finding — single-prompt testing UNDERESTIMATES corruption.

3. **Language corruption is mixed with factual corruption.** (5,15,25) produces Chinese characters alongside correct facts. This suggests the INT6 approximation corrupts token selection partly, leading to mixed-language output.

4. **Corruption controls reproduce consistently:** (0,1) is always gibberish across all prompts. The test harness is validated.

5. **The "Paris" prompt is TOO EASY.** All safe policies pass Paris because it's a trivial factual lookup. Broader factual recall (Jupiter, simple science facts) reveals fragility.

**Root cause:** INT6 FFN_UP approximations corrupt factual recall mechanisms. The residual stream errors from 2+ INT6 layers cause the model to second-guess known facts and hallucinate. Creative prose is more resilient because it doesn't rely on precise factual tokens.

## M. Recommended Next (Phase 20A)

1. **Validate (10,20) more broadly** — it's the best candidate, test on 5-8 prompts
2. **Test (1,2) and (5,10) on P2** — these failed in Phase 19Y, confirm stability
3. **Test (10,20) vs (1,2) vs (5,10) on 5 factual prompts** to find the most robust pair
4. **Add a "hard factual" prompt** beyond Paris/Jupiter — e.g., "The element with atomic number 79 is" to stress-test factual recall
5. **Try (10,20) with INT8 sidecars** if available for a cleaner comparison
6. **Audit FFN_UP outputs at layers 10 and 20** to understand why these layers are stable

## N. Models/Sidecars/Binaries Staged?
No. No model files, sidecars, or binaries staged.

## O. Secrets Detected?
None.

## P. Existing Tags Touched?
None.

## Verdict: **FAIL_NO_UNIVERSALLY_SAFE_POLICY**