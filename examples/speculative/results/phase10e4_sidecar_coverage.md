# Phase 10E-4: Sidecar Coverage

## Model Layer Count

**36** — Qwen2.5-3B-Instruct-Q4_K_M.gguf (`qwen2.block_count=36`)

## Sidecar Count Before

**28** — layers 0–27 only (layers 28–35 were missing)

## Sidecar Count After

**36** — layers 0–35, all built from `blk.{layer}.ffn_up.weight` Q4_K weights via `llama-prt-ffn-up-extract`

## Sidecar Build Method

```bash
for layer in $(seq 28 35); do
  ./build/bin/llama-prt-ffn-up-extract -t "blk.${layer}.ffn_up.weight" \
    -o "/tmp/ffn_up_layer${layer}_float.bin"
  # Convert to |W| (element-wise absolute value) → PRT sidecar
  python3 -c "struct.pack('22544384f', *(abs(v) for v in struct.unpack('22544384f', f.read())))"
  rm /tmp/ffn_up_layer${layer}_float.bin
done
```

Each sidecar: 90,177,536 bytes = 2048 × 11008 × float32

## Sidecar Verification

| Layer | Size | Status |
|-------|------|--------|
| 0–27 | 90,177,536 bytes | ✅ pre-existing |
| 28–35 | 90,177,536 bytes | ✅ newly built |

All 36 sidecars: correct size (90,177,536 bytes), correct format (float32, all non-negative).

## Layer-to-Sidecar Map

```json
{
  "0":  "/tmp/prt_sidecars/ffn_up_layer0_prt.bin",
  "1":  "/tmp/prt_sidecars/ffn_up_layer1_prt.bin",
  ...
  "35": "/tmp/prt_sidecars/ffn_up_layer35_prt.bin"
}
```

All 36 layers mapped. No layer uses another layer's sidecar. No missing layers.
