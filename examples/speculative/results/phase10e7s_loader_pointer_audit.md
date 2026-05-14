# Phase 10E-7S: Loader Pointer Audit

## Loader Pointer Audit Results

Simulated the exact harness loading pattern outside llama.cpp.

### Path String
- Path used: `/tmp/prt_sidecars/ffn_up_layer0_prt.bin`
- Path address: `0x7ffd79285c60` (stack, not in heap)
- Path length: 39 bytes

### Allocation
- Allocated: `0x7a9218bff010` to `0x7a921e1ff010`
- Size: 90,177,536 bytes
- Bytes read: 90,177,536 (exact match)

### Pointer Validation
| Pointer | Address | In Range |
|---------|---------|----------|
| plane0 (offset 0) | `0x7a9218bff010` | YES ✓ |

All weight/plane pointers are within the allocated sidecar range. No pointer points to stack or path string memory.

### Path String Contamination Check
Searched entire allocation for path strings ("ffn_up", "prt_side", "/tmp/").
**Result: NONE FOUND**

## Verdict
**Loader pointers are VALID.** Sidecar loading does not introduce path string contamination. The path string is on the stack (separate from allocation). Plane pointers are correctly within the allocated region.

The corruption is NOT in the loader. It must be in the PRT compute integration inside the custom op.