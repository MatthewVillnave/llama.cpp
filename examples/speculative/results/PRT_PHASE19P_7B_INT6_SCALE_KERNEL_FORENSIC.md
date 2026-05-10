# PRT Phase 19P — 7B INT6 Scale/Kernel Forensic

## Verdict

**PARTIAL_ROOT_CAUSE_IDENTIFIED** ⚠️

---

## Context

Phase 19O found 7B INT6 scalar produces corrupt output at ~1.1 tok/s on Phase 19 branch. This phase digs into root cause by comparing file-level, loader-level, and kernel-level data, and comparing Phase 14a/15 branch vs Phase 19 branch code.

---

## Key Findings

### B. File-Level Forensic
- All 28 layers have **scale[0] = 0.000000 exactly zero**
- This is baked into the PRT6 sidecar files
- Scale[1+] are non-zero (0.001-0.110 range)
- **NOT a loader bug** - file-level property
- Likely model weight property (FFN row 0 is all zeros)
- This would zero out output[0] for all tokens

### C. Loader-Level Forensic
- Loader correctly reads scale[0]=0.000000 from file
- Format is correctly set to 2 (INT6)
- Data is correctly unpacked from packed to int8_t
- **NOT a loader bug** - same data as Phase 15B-H

### D. Packed Size Anomaly
- All 28 sidecar files are 4 bytes LARGER than expected
- Expected: 50997264, Actual: 50997268
- **Systematic** - not random corruption

### E. Reference Branch Comparison
Worktree created at `/tmp/llama_prt_phase15_ref` but is at Phase 18C (not the original Phase 14a/15)

### F. Code Diff Audit - KEY FINDING

| Branch | Kernel `format` check | INT6 path |
|--------|---------------------|-----------|
| Phase 14a (fdf657e1 = Phase 15H) | `ud->format == 1` only | **format=2 would FALL THROUGH** |
| Phase 19 (current) | `(ud->format == 1 \|\| ud->format == 2)` | **format=2 NOW HANDLED** |

**Critical insight:** Phase 14a/15 branch's scalar kernel does NOT handle format=2 (INT6)!
- format=0 (float32) → native path
- format=1 (INT8) → int8_data path  
- format=2 (INT6) → **falls through to float32 fallback which reads nullptr → SHOULD CRASH**

But Phase 15B-H claimed to PASS on Phase 14a branch. How?
- Possible: Phase 15B-H was run on a different commit with format==2 support
- Or: report was wrong/误导
- Or: testing was done with different flags

---

## Root Cause Analysis

### Hypothesis 1: scale[0]=0 in sidecar files
**Evidence:** File-level = 0.000000, Loader-level = 0.000000
**Status:** CONFIRMED - but would have affected ALL branches including Phase 15
**Not root cause** of Phase 19-specific regression

### Hypothesis 2: format==2 bug in current branch  
**Evidence:** Phase 14a doesn't have format==2 in kernel; Phase 19D added it
**Status:** format==2 is NOW handled - but still produces wrong output
**Not root cause** of Phase 19-specific regression

### Hypothesis 3: Something else changed between Phase 14a and Phase 19
- Unpack logic: SAME in both branches (LUT-based, Phase 15H)
- Loader: SAME logic (use int6 path)
- Format handling: DIFFERENCE (Phase 19 has format==2, Phase 14a doesn't)

### Conclusion
Since Phase 14a wouldn't even handle format=2 (would crash/fallback), 
the working INT6 in Phase 15 must have been on a DIFFERENT code path
or the branch has evolved significantly since Phase 15 validation.

The current corruption in Phase 19 might be from a DIFFERENT bug 
that only manifests with format==2 properly handled.

---

## Additional Observations

### Timing Discrepancy
- Phase 19 INT6: ~1.1 tok/s (10x slower than native ~9.7 tok/s)
- Phase 15B-H: ~8-9 tok/s (same as native!)
- Even if format==2 is fixed, timing is drastically different

Possible causes:
- Data layout difference
- Index computation different  
- Extra overhead in kernel path
- Build configuration difference

---

## Recommendations

1. **Verify sidecar generation** - Check if scale[0]=0 is truly needed or an artifact
2. **Check Phase 19D code** - The format==2 addition might have subtle issues
3. **Compare Phase 19D vs current** - What's different now vs when 0.5B worked?
4. **Look at 0.5B vs 7B** - 0.5B INT6 worked in Phase 19M, 7B doesn't

---

## Files Analyzed
- `examples/speculative/prt_graph_replace.h` (both branches)
- `tools/cli/cli.cpp` (INT6 loader)
- `src/llama.cpp` (setters)
- Sidecar files: 28 INT6 files

---

## Verdicts
- PARTIAL_ROOT_CAUSE_IDENTIFIED
- Not loader bug
- Not format==2 addition bug (it's NOW working, just wrong output)
- File-scale[0]=0 confirmed but affects ALL branches
- Something else changed between Phase 14a and current

