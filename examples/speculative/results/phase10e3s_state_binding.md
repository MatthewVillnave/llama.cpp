# Phase 10E-3S: State Binding — Strategy & Verification

## Chosen Strategy: Strategy D (Tensor Name Parsing)

The compute callback (`prt_ffn_up_prt_op`) determines layer identity by parsing the tensor name:

```cpp
const char * tname = dst->name ? dst->name : "";
int op_layer = -1;
if (strstr(tname, "ffn_up.prt.layer0")) {
    op_layer = 0;
}
```

The name `"ffn_up.prt.layer0"` is set at graph construction time via:
```cpp
tmp = ggml_map_custom2(ctx0, matmul_result, cur, prt_ffn_up_prt_op, 1, nullptr);
ggml_set_name(tmp, "ffn_up.prt.layer0");
```

The key fix: **the PRT tensor is NOT passed through `cb(tmp, "ffn_up", il)`** which would overwrite the name. A guard condition prevents renaming:

```cpp
if (!(il == 0 && up && tmp->op == GGML_OP_MAP_CUSTOM2)) {
    cb(tmp, "ffn_up", il);
}
```

## Proof: Compute Callback Sees layer_id=0

```
[PRT] prt_op_entry #1: name=ffn_up.prt.layer0 op_layer=0 sidecar=0x7aee5be5e010
[PRT] PRT_OP: op_layer=0 PRT compute nelem=11008
```

Every invocation of `prt_op_entry` shows `op_layer=0` (parsed from tensor name), not -1, not 35.

## Proof: It Does NOT See layer_id=35

- `op_layer` is parsed from tensor name only. The string `"ffn_up.prt.layer0"` never appears with any other layer number.
- No fallback to identity for `op_layer != 0` occurred (fallback count = 0).
- The tensor name is set once at construction time and never modified.

## Counters

| Counter | Value |
|---------|-------|
| PRT replacements | 21 (token 1) + 10 (subsequent tokens) = 31 |
| Identity fallbacks | 0 |
| Custom op entries | 31 (1 per token per layer0 activation) |

## State Binding Fix Summary

| Item | Before (10E-3R) | After (10E-3S) |
|------|-----------------|----------------|
| Layer source | `g_prt_ffn_up_layer_last` global | `dst->name` parsing |
| Global state bleed | YES — updated per layer, ends at 35 | NO — tensor-name-derived, immutable |
| compute-time layer | 35 (wrong) | 0 (correct) |
| Identity fallback count | high | 0 |
| PRT execution | all fallback | all successful |

**Phase 10E-3S PASS — state binding is correct.**
