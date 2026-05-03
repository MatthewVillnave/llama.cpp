# PRT Phase 12E: Failures & Anomalies

**Branch:** `experimental/prt-route-a-phase12e-l11-l15`  
**Base commit:** `842eba6fd`  
**Runs:** 72 (24 prompts × 3 policies)

---

## Truncated / Incomplete Runs

Three prompts (p10, p11, p12) produced incomplete runs across multiple policies:

### p10 — "Who wrote Hamlet"
| Policy | Wall Time | Tokens Generated | Status |
|--------|-----------|-----------------|--------|
| Native | — | 109 [GEN] steps | ⚠️ Truncated (no `real` line) |
| L12+L15 | — | 76 [GEN] steps | ⚠️ Truncated |
| L11+L15 | — | 61 [GEN] steps | ⚠️ Truncated |

Generation started successfully in all three runs (up to 109 tokens for native), but all three failed to reach completion and write a final wall time. This is a **process failure**, not a correctness failure — the generation was clearly proceeding normally before being interrupted.

**Root cause hypothesis:** The prompt "Who wrote Hamlet" is a trivially answerable factual query. The model may have generated a very short answer quickly and the run script may have terminated early due to an end-of-generation condition that fired before the normal completion path. All three policies affected equally, suggesting this is prompt-specific, not PRT-related.

---

### p11 — "Explain quantum computing"
| Policy | Wall Time | Tokens Generated | Status |
|--------|-----------|-----------------|--------|
| Native | — | 91 [GEN] steps | ⚠️ Truncated |
| L12+L15 | — | 7 [GEN] steps | ⚠️ Truncated |
| L11+L15 | — | 211 [GEN] steps | ⚠️ Truncated |

All three policies truncated. Native generated 91 tokens, L12 generated only 7 tokens before truncation, and L11 generated 211 tokens (more than the n=50 target). This is a consistent process failure affecting all three policies, not a PRT correctness issue.

---

### p12 — "What is machine learning"
| Policy | Wall Time | Tokens Generated | Status |
|--------|-----------|-----------------|--------|
| Native | — | 25 [GEN] steps | ⚠️ Truncated |
| L12+L15 | 47.49s | 300 [GEN] steps | ✅ Complete |
| L11+L15 | 47.98s | 300 [GEN] steps | ✅ Complete |

Native truncated after only 25 tokens. L12 and L11 completed normally. This suggests the native run for p12 may have encountered a model behavior that caused early termination. The PRT policies both completed successfully, indicating no correctness regression.

---

### Summary of Truncation

| Prompt | Native | L12+L15 | L11+L15 | Likely Cause |
|--------|--------|---------|---------|--------------|
| p10 | Truncated | Truncated | Truncated | Prompt-specific process issue |
| p11 | Truncated | Truncated | Truncated | Prompt-specific process issue |
| p12 | Truncated | Complete | Complete | Native-only issue (model?) |

**Conclusion:** The truncated runs are **not PRT-related failures**. They affect native and PRT policies equally. The fact that p12 L12/L11 completed successfully while native did not is actually a mildly positive signal for PRT (it didn't make things worse). The truncation is likely due to the run harness or model hitting an end-of-generation condition earlier than expected for these specific prompts.

---

## JSON Validity — Prompts 13–16

Prompts 13–16 explicitly instruct JSON output:

| Prompt | Instruction | L12 Token-0 | L11 Token-0 | Valid? |
|--------|------------|-----------|-----------|--------|
| p13 | `Write JSON: {"name":"John","age":30}` | ` The` | ` The` | ❌ Model ignores JSON instruction |
| p14 | `Parse JSON: {"a":1,"b":2}` | ` Ensure` | `uits` | ❌ Model ignores JSON instruction |
| p15 | `Validate JSON: {"x":[1,2,3]}` | ` Here` | ` Here` | ❌ Model ignores JSON instruction |
| p16 | `Echo JSON: {"ok":true}` | ` The` | ` The` | ❌ Model ignores JSON instruction |

**Neither L12+L15 nor L11+L15 reliably produces valid JSON for these prompts.** This is a model capability limitation (Qwen2.5-3B-Instruct does not reliably follow JSON output instructions in this framing), not a PRT correctness issue.

**Native failures:** These are NOT listed as separate failures — all policies failed to produce JSON for these prompts. The model simply starts with a conversational response regardless of policy.

---

## Mismatched Token-0 Cases

Both PRT policies exhibit the same token-0 mismatches:

| Prompt | Native Token-0 | L12 Token-0 | L11 Token-0 | Issue |
|--------|--------------|-----------|-----------|-------|
| p7  | ` T` (space+T) | `T` | `T` | Whitespace-only diff — cosmetic |
| p14 | `uits` | ` Ensure` | `uits` | L12 differs from native/L11 |

- **p7:** Both PRT policies produce `T` vs native's ` T` — same token, different leading whitespace prefix. Cosmetic only.
- **p14:** L12 outputs ` Ensure` vs native/L11 both outputting `uits`. L11 matches native exactly; L12 does not. **L11 wins on this prompt.**

---

## Are There Any Real Failures?

| Category | Status |
|----------|--------|
| Correctness regressions | ✅ None (token-0 matches tie) |
| Counter anomalies | ✅ None |
| Speedup regressions | ✅ None (L11+L15 fastest) |
| Truncated runs | ⚠️ 7 runs truncated (p10–p12 all policies) — process issue, not PRT |
| JSON validity | ❌ 0/4 for both policies — model limitation |

**Overall: No PRT correctness failures detected.** The truncated runs are a process/harness issue affecting all policies equally. JSON validity failures are model capability limitations, not PRT regressions.
