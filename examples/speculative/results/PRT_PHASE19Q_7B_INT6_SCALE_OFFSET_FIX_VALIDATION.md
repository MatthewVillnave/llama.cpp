# Phase 19Q: 7B INT6 Scale Offset Fix Validation

## Summary
Scale offset fix verified correct at file/loader level, but **separate INT6 kernel bug** causes corruption.

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## HEAD
`149d7d844` (pre-fix), working-tree has scale_off fix applied

## Scale Offset Fix
- **Location**: `tools/cli/cli.cpp` line 760
- **Old**: `size_t scale_off = 16;` (hardcoded)
- **New**: `size_t scale_off = (M == 4864 && K == 896) ? 16 : 20;` (schema-aware)
- **7B**: Uses 20-byte header schema (scale_off=20)
- **0.5B**: Uses 16-byte header schema (scale_off=16)

## Step 1: File/Loader Sanity
| Check | 7B | 0.5B |
|-------|-----|-------|
| M | 18944 | 4864 |
| K | 3584 | 896 |
| scale_off used | 20 | 16 |
| scale[0] value | 0.00216875 ✅ | 0.00195288 ✅ |
| payload bounds | valid | valid |
| Status | PASS | PASS |

**Verdict**: PASS

## Step 2: Tiny 7B Canary
- **Native** (no PRT): "The capital of France is Paris." ✅
- **INT6 with force-native ALL 28**: "The capital of France is Paris." ✅
- **INT6 with force-native 11,15**: Gibberish ("evenORIZentifier...") ❌

| Criteria | Result |
|----------|--------|
| native clean | YES |
| INT6 clean | NO |
| INT6 semantic match: "Paris" | **NO** |
| 28/28 sidecars loaded | YES |
| force-native 11/15 applied | YES |
| format=int6 logged | YES |
| PRT compute hits > 0 | YES (834) |
| scale_0 correct | YES (0.002169) |
| no crash | YES |
| timing captured | YES |

**Verdict**: FAIL_7B_AFTER_SCALE_FIX

## Root Cause Analysis

### Scale Offset Bug (FIXED)
- 7B PRT6 sidecar file has 20-byte header (magic + ver + M + K + reserved)
- Reserved field at bytes 16-20 is zero for 7B
- scale[0] starts at byte 20
- Old loader: `scale_off=16` → read reserved (0.0) as scale[0]
- New loader: `scale_off=schema-aware` → reads correct scale[0]

**This fix is correct and verified by runtime audit.**

### Remaining Bug (NEW)
- Scale values load correctly (scale_0=0.002169 verified)
- But INT6 compute output is still corrupted
- Any INT6 layer causes corruption
- ALL layers native → clean output
- This points to a **separate bug in INT6 kernel**

## Timing
- Native: 9.7 t/s
- INT6 scalar (format=2): 1.1 t/s (expected - scalar path)

## Conclusion
**scale_offset fix verified correct but separate INT6 kernel bug remains.**

## Verdict: FAIL_7B_AFTER_SCALE_FIX

## Recommended Next
1. Debug INT6 compute kernel (format=2 path in prt_graph_replace.h)
2. Compare Phase 15B-G (working) vs current INT6 kernel code
3. Check for regression between phases

## Models/Sidecars/Binaries
- No models staged
- Sidecars: existing at `/tmp/prt_sidecars_7b_int6_phase15b_packed/`
- Binary: rebuilt with fix in working-tree

## Secrets
- None detected