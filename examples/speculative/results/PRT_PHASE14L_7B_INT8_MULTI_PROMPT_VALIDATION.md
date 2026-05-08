# PRT Phase 14L: 7B INT8 Multi-Prompt Validation

## Executive Summary

| Field | Value |
|-------|-------|
| **Branch** | `experimental/prt-phase14a-packed-sidecars` |
| **Previous HEAD** | `257a1b18c` |
| **Native completed** | 4/4 ✅ |
| **INT8 PRT completed** | 4/4 ✅ |
| **Clean outputs** | 4/4 ✅ |
| **Semantic matches** | 4/4 ✅ |
| **Quality degradations** | 0 |
| **Verdict** | **PASS_7B_INT8_4PROMPT_VALIDATION** |

## Results Summary

| # | Prompt | Native Output | INT8 PRT Output | Match | Quality |
|---|--------|--------------|-----------------|-------|---------|
| 1 | "The capital of France is" | "The capital of France is Paris." | "The capital of France is Paris." | ✅ Exact | ✅ |
| 2 | "Write a Python function that reverses a list." | Code with slicing approach | Code with slicing approach | ✅ Semantic | ✅ |
| 3 | "Return JSON with keys name and status." | Valid JSON `{"name":..., "status":...}` | Valid JSON `{"name":..., "status":...}` | ✅ Exact | ✅ |
| 4 | "Explain CPU inference in one sentence." | CPU process definition | CPU process definition | ✅ Exact | ✅ |

## Timing Summary

| Prompt | Native Wall (s) | Native t/s | INT8 Wall (s) | INT8 t/s | Ratio |
|--------|----------------|------------|--------------|----------|-------|
| 1 | 4.305 | 9.6 | 4.839 | 9.7 | 0.89× |
| 2 | 8.304 | 8.5 | 8.807 | 8.6 | 0.94× |
| 3 | 8.253 | 8.7 | 8.677 | 8.7 | 0.95× |
| 4 | 6.303 | 8.9 | 7.002 | 8.7 | 0.90× |
| **Avg** | **6.791** | **8.9** | **7.331** | **8.9** | **0.93×** |

*Ratio = INT8/Native wall time. Informational only — no broad speedup claim.*

## PRT_SHAPE_DETAIL (captured from INT8 runs)

```
n_layer=28 hidden=3584 ffn=18944 sidecar_rows=18944 sidecar_cols=3584
runtime_M=18944 runtime_N=3584 format=int8
```

All 28 sidecars loaded in all 4 runs.

## Fallback Behavior

- `--prt-force-native 11,15` applied in all INT8 runs
- Layers 11 and 15 use native FFN_UP (expected behavior)
- All other 26 layers use INT8 PRT sidecars

## Output Details

### Prompt 1: "The capital of France is"
- **Native**: "The capital of France is Paris."
- **INT8 PRT**: "The capital of France is Paris."
- **Exact match**: ✅ (identical token-by-token)

### Prompt 2: "Write a Python function that reverses a list."
- **Native**: "Certainly! Below is a Python function that reverses a list. This function uses slicing, which is a concise and efficient way to reverse a list in Python.\n\n```python\ndef reverse_list(input_list..."
- **INT8 PRT**: "Certainly! Below is a Python function that reverses a list. This function uses slicing, which is a concise and efficient way to reverse a list in Python.\n\n```python\ndef reverse_list(input_list..."
- **Code plausible**: ✅ (valid Python, correct syntax)
- **Semantic match**: ✅

### Prompt 3: "Return JSON with keys name and status."
- **Native**: `{"name": "Example", "status": "Active"}` (with markdown wrapping)
- **INT8 PRT**: `{"name": "Example", "status": "Active"}` (with markdown wrapping)
- **JSON valid**: ✅ Both produce valid JSON block
- **Exact match**: ✅

### Prompt 4: "Explain CPU inference in one sentence."
- **Native**: "CPU inference refers to the process of using a CPU to perform the prediction or decision-making tasks of a machine learning model."
- **INT8 PRT**: "CPU inference refers to the process of using a CPU to perform the prediction or decision-making tasks of a machine learning model."
- **Exact match**: ✅ (identical)

## Quality Assessment

- **Repeatition/collapse**: None observed
- **Path fragments**: Contained in stderr log but not in output text
- **Debug contamination**: None in stdout
- **Quality degradations**: 0 across all 4 prompts

## Verdict

**PASS_7B_INT8_4PROMPT_VALIDATION** ✅

All pass criteria met:
- ✅ Native completed 4/4
- ✅ INT8 PRT completed 4/4
- ✅ Clean outputs 4/4
- ✅ Semantic matches 4/4
- ✅ Quality degradations 0
- ✅ JSON valid for prompt 3
- ✅ Code plausible for prompt 2
- ✅ Sidecars loaded 28/28 (all runs)
- ✅ PRT_SHAPE_DETAIL correct
- ✅ Fallback limited to layers 11 and 15 only

## Recommended Next

- Phase 14M: 7B INT8 repeatability (10-run timing benchmark, same as 14G format)
- Do NOT claim production readiness yet

## Safety

- **Models/sidecars staged?** NO
- **Secrets detected?** NO
- **Tags touched?** NO