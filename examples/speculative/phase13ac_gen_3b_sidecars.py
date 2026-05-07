#!/usr/bin/env python3
"""
PRT Phase 13AC: Generate Qwen2.5-3B PRT sidecars
Extracts blk.{L}.ffn_up.weight from Qwen2.5-3B-Instruct-Q4_K_M.gguf
and writes float32 sidecar files compatible with the fixed PRT runtime.
"""
import struct, os, json, sys
import numpy as np

MODEL_PATH = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
OUT_DIR = "/tmp/prt_sidecars_3b"
N_LAYERS = 36
HIDDEN = 2048
FFN = 11008
EXPECTED_FLOATS = HIDDEN * FFN  # 22,532,096
EXPECTED_BYTES = EXPECTED_FLOATS * 4  # 90,177,536

# Q4_K_M constants
Q4_K_M_BLOCK_SIZE = 128
Q4_K_M_NBITS = 4
Q4_K_M_TYPE = 12

def read_gguf_metadata(path):
    """Read GGUF header and metadata."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f"Not a GGUF file: {magic}"
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        
        # Parse metadata entries
        metadata = {}
        for _ in range(metadata_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8', errors='replace')
            typ = struct.unpack('<I', f.read(4))[0]
            val_len = struct.unpack('<Q', f.read(8))[0]
            val_data = f.read(val_len)
            metadata[key] = (typ, val_data)
        
        return {
            'version': version,
            'tensor_count': tensor_count,
            'metadata': metadata
        }

def dequantize_q4_k_m(tensor_data, ne0, ne1):
    """
    Dequantize Q4_K_M tensor to float32.
    ne0 = first dimension (columns, e.g. FFN)
    ne1 = second dimension (rows, e.g. hidden)
    
    Layout: [ne1, ne0/16, 16, 2] in memory = [rows, block_count, block_size, quantizers]
    Q4_K_M uses block size 128, with scales and quants per block.
    """
    n_elements = ne0 * ne1
    result = np.zeros(n_elements, dtype=np.float32)
    
    # Q4_K_M block: 128 elements per block, 4-bit quantization
    # Per block: 128 * 0.5 bytes = 64 bytes of quants + 2*4 bytes (scale, offset) = 72 bytes
    block_size = 128  # elements per block
    block_bytes = 64   # quantized data bytes per block (128 * 4bits / 8)
    n_blocks = (n_elements + block_size - 1) // block_size
    
    # We need to know how the data is laid out
    # Q4_K_M: per block, first 4 bytes = scale (float16), next 4 bytes = offset (float16 or zero)
    # Then 64 bytes = 128 4-bit values packed
    scale_offset = 8  # bytes for scale + offset at start of each block
    
    offset = 0
    for b in range(n_blocks):
        block_start = b * block_size
        block_end = min(block_start + block_size, n_elements)
        block_count = block_end - block_start
        
        # Read scale and offset (float16)
        scale = struct.unpack_from('<e', tensor_data, offset)[0]
        offset += 2
        offset_val = struct.unpack_from('<e', tensor_data, offset)[0]
        offset += 2
        
        # Read packed 4-bit quants (64 bytes for 128 values)
        qdata = tensor_data[offset:offset+block_bytes]
        offset += block_bytes
        
        # Unpack 4-bit values: each byte has 2 values
        unpacked = []
        for byte in qdata[:block_count // 2]:
            unpacked.append((byte >> 4) - 8)   # high nibble - 8 offset
            unpacked.append((byte & 0x0F) - 8) # low nibble - 8 offset
        
        # Dequantize: result = scale * (unpacked + offset_val)
        for i, q in enumerate(unpacked[:block_count]):
            val = scale * (q + offset_val)
            result[block_start + i] = val
    
    return result

def main():
    print("=== PRT Phase 13AC: 3B Sidecar Generation ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUT_DIR}")
    print(f"Expected shape: {FFN} x {HIDDEN} ({EXPECTED_FLOATS} floats, {EXPECTED_BYTES} bytes)")
    
    # Check model file
    model_size = os.path.getsize(MODEL_PATH)
    print(f"Model size: {model_size/1024/1024:.1f} MB")
    
    # Read GGUF header
    info = read_gguf_metadata(MODEL_PATH)
    print(f"GGUF v{info['version']}, {info['tensor_count']} tensors, {len(info['metadata'])} metadata entries")
    
    # Confirm architecture
    arch_key = 'general.architecture'
    arch = info['metadata'].get(arch_key, (0, b'?'))[1].decode('utf-8', errors='replace')
    print(f"Architecture: {arch}")
    
    block_count = info['metadata'].get('qwen2.block_count', (0, b'\x24'))[1]
    n_layers = struct.unpack('<Q', block_count[:8])[0]
    print(f"Block/layer count: {n_layers}")
    
    emb = info['metadata'].get('qwen2.embedding_length', (0, b'\x00\x08'))[1]
    hidden = struct.unpack('<Q', emb[:8])[0]
    print(f"Embedding/hidden: {hidden}")
    
    ffn_len = info['metadata'].get('qwen2.feed_forward_length', (0, b'\x00'))[1]
    ffn_dim = struct.unpack('<Q', ffn_len[:8])[0]
    print(f"FFN dimension: {ffn_dim}")
    
    # Validate dimensions
    if hidden != HIDDEN or ffn_dim != FFN or n_layers != N_LAYERS:
        print(f"\nERROR: DIMENSION_MISMATCH")
        print(f"  Expected: hidden={HIDDEN}, ffn={FFN}, layers={N_LAYERS}")
        print(f"  Got: hidden={hidden}, ffn={ffn_dim}, layers={n_layers}")
        sys.exit(1)
    
    print("\nDimensions confirmed: 36 layers, 2048 hidden, 11008 FFN ✅")
    
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Now read tensor data directly from the GGUF file
    # We need to read the tensor data section
    with open(MODEL_PATH, 'rb') as f:
        # Skip header + metadata
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        
        # Skip metadata
        for _ in range(metadata_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)
            typ = struct.unpack('<I', f.read(4))[0]
            val_len = struct.unpack('<Q', f.read(8))[0]
            f.read(val_len)
        
        # Now read tensor metadata to find offsets
        tensor_index = []
        for i in range(tensor_count):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8', errors='replace')
            n_dim = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dim)]
            typ = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            
            if 'ffn_up' in name.lower() and 'weight' in name.lower():
                tensor_index.append({
                    'name': name,
                    'dims': dims,  # [ne0=FFN, ne1=hidden]
                    'type': typ,
                    'offset': offset
                })
        
        print(f"\nFound {len(tensor_index)} ffn_up.weight tensors")
        
        # Extract each tensor
        all_stats = {}
        for idx, t in enumerate(sorted(tensor_index, key=lambda x: x['name'])):
            layer = int(t['name'].split('.')[1])
            ne0, ne1 = t['dims']  # ne0=FFN(11008), ne1=hidden(2048)
            
            # Calculate size in file (Q4_K_M quantized)
            # Q4_K_M: (ne0 * ne1 * 4bits) / 8 + block metadata
            block_count_calc = ((ne0 * ne1) + 127) // 128
            qdata_size = block_count_calc * 72  # 128 elems + 2 scales + quants per block
            
            print(f"\nLayer {layer:2d}: {t['name']}")
            print(f"  Shape: {ne0} x {ne1} = {ne0*ne1} elements")
            print(f"  Type: {t['type']} (Q4_K_M={Q4_K_M_TYPE})")
            print(f"  Offset: {t['offset']}, Q4_K size: ~{qdata_size/1024:.0f} KB")
            
            # Read quantized data
            f.seek(t['offset'])
            qdata = f.read(qdata_size)
            
            # Dequantize to float32
            print(f"  Dequantizing {len(qdata)} bytes of Q4_K_M...", end=" ", flush=True)
            f32_data = dequantize_q4_k_m(qdata, ne0, ne1)
            
            # Stats
            finite = np.isfinite(f32_data).sum()
            zero_frac = (f32_data == 0).sum() / len(f32_data)
            neg_frac = (f32_data < 0).sum() / len(f32_data)
            stats = {
                'layer': layer,
                'size_bytes': EXPECTED_BYTES,
                'q4_size_bytes': len(qdata),
                'checksum': float(f32_data.sum()),
                'sum_abs': float(np.abs(f32_data).sum()),
                'min': float(f32_data.min()),
                'max': float(f32_data.max()),
                'mean': float(f32_data.mean()),
                'std': float(f32_data.std()),
                'zero_frac': float(zero_frac),
                'neg_frac': float(neg_frac),
                'finite': int(finite),
            }
            
            print(f"min={stats['min']:.4f} max={stats['max']:.4f} mean={stats['mean']:.6f} finite={finite}/{len(f32_data)}")
            print(f"  zero={zero_frac:.4f} neg={neg_frac:.4f} sum_abs={stats['sum_abs']:.1f}")
            
            # Write sidecar
            out_path = os.path.join(OUT_DIR, f"ffn_up_layer{layer}_prt.bin")
            with open(out_path, 'wb') as out:
                out.write(f32_data.astype(np.float32).tobytes())
            
            print(f"  Wrote: {out_path}")
            all_stats[layer] = stats
        
        print(f"\n\nGenerated {len(all_stats)}/{N_LAYERS} sidecars")
        
        # Save stats
        stats_path = os.path.join(OUT_DIR, "stats.json")
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"Stats: {stats_path}")
        
        # Validate
        missing = [l for l in range(N_LAYERS) if l not in all_stats]
        if missing:
            print(f"\nWARNING: Missing layers: {missing}")
        
        return 0 if not missing else 1

if __name__ == '__main__':
    sys.exit(main())