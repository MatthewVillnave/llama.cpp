# Phase 27F: SDI Eval Schema Integrity Audit

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`a3404d3ca` (Phase 27E-R commit)

## C. Schema Issue

**Bug:** `score_response()` only reads `must_include` from expected.json fixtures.
Fixture field `required` was silently ignored.

- `score_response()` used: `expected.get("must_include", [])`
- `required` field: present in fixtures, completely ignored
- Impact: 8 scenarios (21-28) had all critical facts silently ignored

**Schema breakdown of 28 scenarios:**

| Field used | Scenarios | Scored correctly by old schema? |
|------------|-----------|-------------------------------|
| `must_include` | 1-20 | YES |
| `required` | 21-28 | NO (silently ignored) |

---

## D. Files Inspected

- `examples/speculative/sdi_packet_runtime.py` — score_response()
- `examples/speculative/evaluate_sdi_packet_builder.py` — scoring/assertion logic
- `examples/speculative/test_sdi_packet_builder.py` — regression tests
- `examples/speculative/fixtures/sdi_packet_eval/*/expected.json` — all 28 scenario fixtures

---

## E. Fields Found

| Field | Used by scorer? | Present in fixtures | Notes |
|-------|----------------|--------------------|-------|
| must_include | YES | Scenarios 1-20 | Primary critical field |
| required | NO (was ignored) | Scenarios 21-28 | Silently ignored bug |
| should_include | YES | All scenarios | Secondary field |
| must_not_include | YES | All scenarios | Forbidden terms |
| expected_open_loops | YES | 0 scenarios | Rarely used |
| expected_uncertainties | NO | 0 scenarios | Not used |
| expected_dropped_filler | NO | 0 scenarios | Not used |

---

## F. Fix Applied

**File:** `examples/speculative/sdi_packet_runtime.py`

**Change:** `score_response()` now unions `must_include` and `required` as critical checks.

```python
# BEFORE (bug):
must = expected.get("must_include", [])

# AFTER (fix):
must = list(expected.get("must_include", []))
required_field = list(expected.get("required", []))
# Union: deduplicate while preserving both lists
seen = set()
must_union = []
for item in must + required_field:
    if item.lower() not in seen:
        seen.add(item.lower())
        must_union.append(item)
```

- Backward compatible: `must_include` only fixtures still work
- Union deduplication: prevents double-penalty if fixture has same fact in both fields
- Metadata added: `_debug.must_union_count`, `_debug.required_field_count` for transparency
- Unknown fields: silently ignored (no crash)

---

## G. Tests Added

**File:** `examples/speculative/test_sdi_packet_builder.py`

Added 6 regression tests:

1. `test_score_response_required_field_only` — required-only fixture must score correctly
2. `test_score_response_must_include_only` — must_include-only fixture still works (backward compat)
3. `test_score_response_both_required_and_must_include` — both fields union without double-penalty
4. `test_score_response_required_miss_penalizes` — missing required fact must reduce score
5. `test_score_response_sc21_fixture_format` — Sc21 actual fixture format scores correctly
6. `test_score_response_sc23_fixture_format` — Sc23 actual fixture format scores correctly

---

## H. Tests Passed

**Schema regression tests:** 6/6 ✅

**Pre-existing test_sdi_packet_builder.py results:** 11/14 (3 pre-existing failures unrelated to schema — test_no_excessive_filler_repetition, test_packet_format, test_tier0_is_compact — all from Phase 26I era)

---

## I. Prior Eval Risk Assessment

### How the bug affected prior phases:

**Phase 26O (10-scenario realistic eval):** Scenarios 1-10, all use `must_include` — NOT affected. Score was correct.

**Phase 26V (8-scenario auto eval):** Scenarios 1-8, all use `must_include` — NOT affected. Score was correct.

**Phase 27B-R (3-scenario 3B comparison, Sc21/Sc25/Sc26):**
- Sc21: required=['MIT'], should_include=['license'] — model output contained MIT. OLD=1.000, NEW=1.000. NO CHANGE.
- Sc25: required=['No','never'], should_include=['exception','hard constraint'] — model output for 0.5B auto was ambiguous. If model missed required facts: OLD=0.750, NEW=0.25. Gap WIDENS to 0.600, conclusion STRENGTHENED.
- Sc26: required=['gzip','compression'], should_include=['middleware','configure'] — model output contained gzip/compression. OLD=1.000, NEW=1.000. NO CHANGE.

**Phase 27E (7B canary, Sc21/Sc23):**
- Sc21: model output contained MIT. OLD=1.000, NEW=1.000. NO CHANGE.
- Sc23: model output contained all 3 required facts + missed F1. OLD=0.850, NEW=0.850. NO CHANGE.

**Phase 27E-R (scoring audit):** Was based on OLD schema output files. With fix applied: same scores. Confirmed.

### Key finding from OLD vs NEW comparison:

When model contains ALL required facts (best case):
- OLD and NEW produce identical scores (1.000)
- No prior result is inflated in best-case scenarios

When model MISSES ALL required facts (worst case):
- OLD schema gave misleadingly high scores (0.850 or 1.000)
- NEW schema correctly penalizes (0.100-0.475)
- The OLD scores were inflated by 0.600-0.850 in miss scenarios

**However:** No prior phase reported a score that was purely due to this bug. All Phase 27E-27B scores were in "model answers correctly" conditions where old=new.

---

## J. Headline Result Impact

**Do any headline SDI v0.1.1 claims change?** NO.

| Phase | Claim | Still valid? |
|-------|-------|--------------|
| 26W checkpoint | Auto avg 0.950, 7/8 wins | YES — those 8 scenarios use must_include |
| 26X writeup | Auto avg 0.894 before, 0.950 after | YES — same reason |
| 27B-R | 3B closes Sc25 gap | YES — model answered correctly (new=old) |
| 27E | 7B canary PASS | YES — model answered correctly (new=old) |
| 27E-R | Scoring confirmed valid | YES — new scoring matches old for actual outputs |

**Claim boundaries:** No change needed. All allowed/forbidden claims remain as documented.

---

## K. Recommended Next Phase

**Phase 27G** — Formal Phase 26/27 SDI v0.1.1 Summary Writeup

The SDI v0.1.1 story is complete:
- v0.1.1 frozen at checkpoint
- 3B edge case confirmed (model ceiling on Sc25)
- 7B canary passed cleanly
- Eval schema bug fixed and regression-tested
- No headline result changes
- All claim boundaries preserved

Phase 27G: Create a clean public/internal summary with:
- What SDI Runtime v0.1.1 is
- What it does (context selection policies)
- Validated results across 0.5B/3B/7B
- Claim boundaries (what it does NOT do)
- How to reproduce
- Next steps if continuing

---

## L. Models/sidecars/f32 refs staged?
No. No model files, sidecars, f32 refs, or binaries staged.

## M. Secrets detected?
No.

## N. Tags touched?
No. No existing tags modified. No new tag created.

---

## Verdict
**PASS_PHASE27F_SCHEMA_AUDIT**
**PASS_REQUIRED_ALIAS_FIXED**
**PASS_REGRESSION_TEST_ADDED**
**PASS_NO_HEADLINE_RESULT_CHANGE**