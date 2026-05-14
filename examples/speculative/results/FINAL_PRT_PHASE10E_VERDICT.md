# FINAL_PRT_PHASE10E_VERDICT.md

## Phase 10E Final Classification

| Phase | Status | Verdict |
|-------|--------|---------|
| 10E-0 | ✅ PASS | Baseline llama.cpp inference functional |
| 10E-1 | ✅ PASS | Single-layer PRT substitution works |
| 10E-2 | ✅ PASS | Custom op smoke test passes |
| 10E-3 | ✅ PASS | PRT custom op registered and called |
| 10E-4 | ✅ PASS | Multi-layer PRT substitution works |
| 10E-5 | ✅ PASS | Per-layer sidecar lookup functional |
| 10E-6 | ⚠️ PARTIAL | All-layer sidecar loading functional; quality smoke test inconclusive |
| 10E-6S | ⚠️ PARTIAL | Full-layer sidecar construction passed; end-to-end generation blocked |
| 10E-7S | ❌ FAIL | ggml custom-op integration corruption — output contains path string fragments |
| 10E-8 | 🚫 BLOCKED | NOT ALLOWED — quality canary blocked by 10E-7S failure |
| 10F | 🚫 BLOCKED | NOT ALLOWED — broader benchmark blocked by 10E-7S failure |

## Root Cause (10E-7S)

**ggml_map_custom2 integration bug** — when sidecar data is loaded, the ggml computation graph corrupts the matmul output tensor before the custom op runs. Even a pure identity copy (src0→dst) inside the custom op produces garbage output.

**Evidence:**
- Sidecar file: CLEAN (no path strings, correct size, finite values)
- Loader pointers: CLEAN (all within allocated range)
- Standalone PRT compute: CLEAN (produces correct output outside llama.cpp)
- Custom op identity copy: CORRECT floats inside the op
- Final decoded token: CORRUPT (contains "ffn_up_layer35_prt.bin" path fragments)

## Key Achievements

1. ✅ Built PRT_3P standalone sidecar kernel (offline, validated)
2. ✅ Built full-layer sidecar construction pipeline
3. ✅ Demonstrated llama.cpp graph interception and substitution
4. ✅ All-layer sidecar lookup/substitution infrastructure reached the graph
5. ✅ Confirmed corruption is NOT in: sidecar format, loader, standalone PRT math, or custom op code

## What Remains Blocked

- ❌ End-to-end PRT active generation (ggml_map_custom2 memory corruption)
- ❌ Quality/speed benchmarks
- ❌ Production integration

## Classification

**ARCHIVED / BLOCKED** — Do not continue active generation tests.