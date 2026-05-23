# Phase 28BD: Pager View → Decoded Residual Matmul Bridge

## Verdict: ✅ PASS

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## Goal
Prove that a `.trit` residual sidecar loaded through the manifest → pager → `prt_get_residual_view()` path produces the same decoded residual matmul parity as the direct file path.

## Test Harness
`phase28bd_pager_matmul_bridge.cpp`

Build:
```bash
cd /home/matthew-villnave/llama.cpp
g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL \
    -I. -Iggml/include -Iinclude \
    -c examples/speculative/prt_sidecar_pager.cpp -o /tmp/prt_pager.o
g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL \
    -I. -Iggml/include -Iinclude \
    -c examples/speculative/prt_trit_decode.cpp -o /tmp/prt_decode.o
g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL \
    -I. -Iggml/include -Iinclude \
    examples/speculative/phase28bd_pager_matmul_bridge.cpp \
    /tmp/prt_pager.o /tmp/prt_decode.o \
    -o /tmp/phase28bd_pager_matmul_bridge
```

Run: `/tmp/phase28bd_pager_matmul_bridge`

## Results

### Case: row_split
| Metric | Value |
|--------|-------|
| K×M | 96×48 |
| N | 4 |
| block_grid | 3×1 |
| `direct_vs_pager_R_max_abs_err` | **0.0** |
| `direct_vs_pager_R_rmse` | **0.0** |
| `pager_view_is_null` | false |
| `pager_view_size_gt_0` | true |
| pass | ✅ |

### Case: col_split
| Metric | Value |
|--------|-------|
| K×M | 32×144 |
| N | 4 |
| block_grid | 1×3 |
| `direct_vs_pager_R_max_abs_err` | **0.0** |
| `direct_vs_pager_R_rmse` | **0.0** |
| `pager_view_is_null` | false |
| `pager_view_size_gt_0` | true |
| pass | ✅ |

### Case: row_col_split
| Metric | Value |
|--------|-------|
| K×M | 96×144 |
| N | 6 |
| block_grid | 3×3 |
| `direct_vs_pager_R_max_abs_err` | **0.0** |
| `direct_vs_pager_R_rmse` | **0.0** |
| `pager_view_is_null` | false |
| `pager_view_size_gt_0` | true |
| pass | ✅ |

### Case: awkward_edge
| Metric | Value |
|--------|-------|
| K×M | 70×101 |
| N | 2 |
| block_grid | 3×3 partial |
| `direct_vs_pager_R_max_abs_err` | **0.0** |
| `direct_vs_pager_R_rmse` | **0.0** |
| `pager_view_is_null` | false |
| `pager_view_size_gt_0` | true |
| pass | ✅ |

## Negative Tests
- **missing_sidecar**: `prt_init_pager("/nonexistent")` → returns `false` ✅
- **bad_tensor_family**: `prt_get_residual_view(0, "nonexistent_family")` → returns `is_null=true, reason=not_found` ✅

## Conclusion
All 4 positive cases and both negative tests pass. The `.trit` bytes delivered by `prt_get_residual_view()` via the pager manifest path decode to **exact zero error** parity with the ground truth direct-file decode. The bridge is proven.

## Commit
```
Phase 28BD: add pager-view decoded residual matmul bridge
```