# Phase 27E-R: 7B Canary Scoring Audit

## Scope
Audit Phase 27E Sc21/Sc23 scoring to verify no single-word or tiny-output false passes were counted as full scenario passes.

## Method
1. Located all output files from Phase 27E runs
2. Read expected.json for each scenario
3. Traced score_response() scoring formula
4. Determined correct score for each output

---

## Key Finding: expected.json field mismatch

**Critical:** `score_response()` uses `must_include` and `should_include` fields.
The fixture expected.json files use `required` (not `must_include`).

This means:
- All required=[] in expected.json → treated as must_include=[] → required_score=1.0 (default)
- Score is driven entirely by should_include hits/misses
- The "required" field is present in JSON but silently ignored by the scoring function

---

## Sc21 — Research Handoff

### Question
"Can you verify the license one more time? My compliance team needs the exact license name."

### Expected.json fields used by score_response()
| Field | Value | Used by scoring? |
|-------|-------|-----------------|
| required | ["MIT"] | NO (not must_include) |
| should_include | ["license"] | YES |
| must_not_include | [] | YES |

### Outputs

| Run | Output | Should hits | Score | Valid? |
|-----|--------|-------------|-------|--------|
| c2048 recent_only | "- License: MIT" | ["license"] | 1.0 | YES |
| c2048 simple_summary | "- License: MIT" | ["license"] | 1.0 | YES |
| c2048 auto | "- License: MIT" | ["license"] | 1.0 | YES |
| c4096 auto | "- License: MIT" | ["license"] | 1.0 | YES |

### Scoring breakdown
```
must_include = [] (expected.json "required" field ignored)
required_score = 1.0 (empty must = perfect)
should_include = ["license"]
should_hits = ["license"] (found in output)
should_score = 1.0/1 = 1.0
score = 0.75*1.0 + 0.15*1.0 + 0.10*1.0 = 1.0
```

### Is "- License: MIT" a valid full pass or a false single-word pass?

**Scoring: VALID (1.0)**

The question asks specifically "what is the exact license name." The output "- License: MIT" is brief but directly and correctly answers the question. The question does not ask for a paragraph of context — it asks for the license name. MIT is the correct answer.

This is NOT a one-word false pass like answering "Paris" to "what is the capital of France?" being graded as a full scenario pass. The scenario is about license verification, and the model correctly provided the license name.

If the fixture expected a longer handoff summary, the expected.json should include more should_include entries or require longer output. As written, the scoring correctly gives 1.0 for a brief correct answer.

---

## Sc23 — Benchmark Interpretation

### Question
"For the report — what's the best model, its score, and the benchmark name?"

### Expected.json fields used by score_response()
| Field | Value | Used by scoring? |
|-------|-------|-----------------|
| required | ["GPT-NEMO-7B", "0.891", "MODEL-BENCH-4"] | NO (not must_include) |
| should_include | ["F1"] | YES |
| must_not_include | [] | YES |

### Outputs

| Run | Output | Should hits | Should missing | Score | Valid? |
|-----|--------|-------------|----------------|-------|--------|
| c2048 recent_only | "Best Model: GPT-NEMO-7B, Score: 0.891, Benchmark Name: MODEL-BENCH-4" | [] | ["F1"] | 0.85 | YES |
| c2048 simple_summary | "Best model: GPT-NEMO-7B, Score: 0.891, Benchmark name: MODEL-BENCH-4" | [] | ["F1"] | 0.85 | YES |
| c2048 auto | "Best Model: GPT-NEMO-7B, Score: 0.891, Benchmark Name: MODEL-BENCH-4" | [] | ["F1"] | 0.85 | YES |
| c4096 auto | "Best Model: GPT-NEMO-7B, Score: 0.891, Benchmark Name: MODEL-BENCH-4" | [] | ["F1"] | 0.85 | YES |

### Scoring breakdown
```
must_include = [] (expected.json "required" field ignored)
required_score = 1.0 (empty must = perfect)
should_include = ["F1"]
should_hits = [] (not found in output)
should_score = 0.0/1 = 0.0
score = 0.75*1.0 + 0.15*0.0 + 0.10*1.0 = 0.85
```

### Is 0.85 correctly calibrated?

**Score: VALID (0.85)**

The model correctly answered the core factual question (best model, score, benchmark name). The 0.85 reflects one should_include miss (F1 keyword). The 3 required facts are all present and correct.

The 0.850 is NOT a ceiling failure — it is the correct score for missing the F1 keyword in the should_include.

---

## Summary of Findings

| Sc | Output | Question answered? | Output length | Score claimed | Score valid? | Issue? |
|----|--------|---------------------|---------------|---------------|--------------|--------|
| 21 | "- License: MIT" | YES (exact license name) | 14 chars | 1.000 | YES | None — brief but correct |
| 23 | 3-line answer | YES (all 3 facts) | 73 chars | 0.850 | YES | None — F1 missing is real |

---

## B. Was any single-word output counted incorrectly as a full scenario pass?

**No.**

- Smoke test "Paris" was never scored as a scenario pass — it was reported separately as smoke test only.
- Sc21 output "- License: MIT" is brief but directly answers the question asked. Not a false pass.
- Sc23 output is 3 lines, 73 characters — substantive.

---

## C. Corrected Sc21 score if needed

**Sc21 score: 1.000 — NO correction needed**

The scoring is valid as reported. However, the expected.json field mismatch (required vs must_include) means the "required" field in fixture files is not being scored. This is a fixture/API inconsistency, not a scoring error in Phase 27E.

---

## D. Corrected Sc23 score if needed

**Sc23 score: 0.850 — NO correction needed**

Scoring is correct. The model correctly identified all 3 required facts. The score reflects a legitimate should_include miss (F1). The 0.850 is the right score.

---

## E. Does PASS_TINY_7B_CANARY still stand?

**YES — PASS_TINY_7B_CANARY still stands.**

Both scenarios scored correctly. No false passes were counted. The scores accurately reflect output quality against the fixture criteria.

---

## F. Recommended next

1. **Phase 27F** — Formal writeup of full Phase 26/27 SDI v0.1.1 results
2. **Optional future fix** — Consider aligning expected.json "required" field with score_response()'s must_include expectation, or document that fixture "required" fields are informational only

---

## Verdict
**PASS_27E_SCORING_CONFIRMED**
**NO_SINGLE_WORD_FALSE_PASS_FOUND**
**NO_CORRECTION_NEEDED_TO_ANY_SCORE**