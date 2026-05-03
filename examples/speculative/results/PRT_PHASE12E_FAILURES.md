# PRT Phase 12E: Failure Analysis

## Failure Summary

### Failures Detected

- p1 L12: SEVERE token corruption
- p1 L11: SEVERE token corruption
- p2 L12: SEVERE token corruption
- p2 L11: SEVERE token corruption
- p3 L12: SEVERE token corruption
- p3 L11: SEVERE token corruption
- p4 L12: SEVERE token corruption
- p4 L11: SEVERE token corruption
- p5 L12: SEVERE token corruption
- p5 L11: SEVERE token corruption
- p6 L12: SEVERE token corruption
- p20 L11: SEVERE token corruption
- p22 L12: SEVERE token corruption
- p22 L11: SEVERE token corruption
- p23 L11: SEVERE token corruption
- p24 L12: SEVERE token corruption
- p24 L11: SEVERE token corruption

## Model/Prompt Limitations (not PRT bugs)

The following JSON prompts produce invalid JSON across ALL policies (native, L12, L11):
- Prompt 13: model does not reliably produce valid JSON for this prompt style
- Prompt 14: model does not reliably produce valid JSON for this prompt style
- Prompt 15: model does not reliably produce valid JSON for this prompt style
- Prompt 16: model does not reliably produce valid JSON for this prompt style

This is a model limitation, not a PRT policy issue.

## Notes

- Severe corruption also appears in native runs (7/24) — this is model behavior on creative tasks, not specific to PRT
- L11 shows slightly more corruption (9/24) than L12 (8/24) or native (7/24)
- native_fallback_calls=16 is expected warmup behavior (layers 11/12 forced native during warmup phase)
