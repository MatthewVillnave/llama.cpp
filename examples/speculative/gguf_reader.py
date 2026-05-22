#!/usr/bin/env python3
"""
GGUF v3 Metadata Reader for Phase 28P
Correctly reads GGUF v3 format (tensor names are inline strings, no separate name table).
"""
import struct
import sys

def read_gguf_metadata(path):
    """Read GGUF v3 header and tensor metadata."""
    with open(path, 'rb') as f:
        data = f.read()
    
    file_size = len(data)
    
    magic = struct.unpack('<I', data[0:4])[0]
    if magic != 0x46554747:
        raise ValueError(f"Not a GGUF file: magic={hex(magic)}")
    
    version = struct.unpack('<I', data[4:8])[0]
    n_tensors = struct.unpack('<Q', data[8:16])[0]
    n_kv = struct.unpack('<Q', data[16:24])[0]
    
    print(f"GGUF v{version}, tensors={n_tensors}, n_kv={n_kv}, file_size={file_size}", file=sys.stderr)
    
    offset = 24  # end of header
    
    # Read KV pairs (metadata)
    for _ in range(n_kv):
        # Read key string (uint64 length + bytes, padded to 8)
        key_len = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        key = data[offset:offset+key_len].decode('utf-8', errors='replace')
        offset += key_len
        offset += (8 - (key_len % 8)) % 8
        
        # Read value type
        val_type = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # Skip value based on gguf_type
        if val_type == 0:   # UINT8
            offset += 1
        elif val_type == 1:   # INT8
            offset += 1
        elif val_type == 2:   # UINT16
            offset += 2
        elif val_type == 3:   # INT16
            offset += 2
        elif val_type == 4:   # UINT32
            offset += 4
        elif val_type == 5:   # INT32
            offset += 4
        elif val_type == 6:   # FLOAT32
            offset += 4
        elif val_type == 7:   # BOOL
            offset += 1
        elif val_type == 8:   # STRING
            str_len = struct.unpack('<Q', data[offset:offset+8])[0]
            offset += 8 + str_len
            offset += (8 - (str_len % 8)) % 8
        elif val_type == 9:   # ARRAY
            offset += 8 + 4  # element count + array type
        elif val_type == 10:  # UINT64
            offset += 8
        elif val_type == 11:  # INT64
            offset += 8
        elif val_type == 12:  # FLOAT64
            offset += 8
        else:
            print(f"Unknown KV type {val_type} at offset {offset-4}", file=sys.stderr)
            break
    
    kv_end = offset
    print(f"KV pairs ended at offset {kv_end}", file=sys.stderr)
    
    # Read alignment field (uint64)
    alignment = struct.unpack('<Q', data[offset:offset+8])[0]
    offset += 8
    
    # Data starts at first tensor offset aligned to alignment
    data_offset = offset  # we'll compute actual from first tensor
    
    # Read tensor info table
    tensors = []
    for i in range(n_tensors):
        if offset + 20 > file_size:
            print(f"Tensor {i}: ran out of file", file=sys.stderr)
            break
        
        # Read tensor name (inline string: uint64 len + bytes, padded to 8)
        name_len = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        name = data[offset:offset+name_len].decode('utf-8', errors='replace')
        offset += name_len
        offset += (8 - (name_len % 8)) % 8
        
        # Read n_dims (uint32)
        n_dims = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # Read dims (n_dims × uint64)
        dims = []
        for _ in range(n_dims):
            d = struct.unpack('<Q', data[offset:offset+8])[0]
            dims.append(d)
            offset += 8
        
        # Read tensor type (uint32)
        tensor_type = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # Read offset (uint64, relative to start of tensor data section)
        tensor_offset = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        tensors.append({
            'index': i,
            'name': name,
            'type': tensor_type,
            'dims': dims,
            'offset': tensor_offset,
        })
    
    # Find the minimum tensor offset to determine data section start
    if tensors:
        min_offset = min(t['offset'] for t in tensors)
        # The GGUF spec says data_offset from gguf_get_data_offset() includes alignment
        # but tensor offsets are relative to the tensor data area (after alignment field)
        actual_data_start = offset + min_offset  # offset after reading all tensor infos + min tensor offset
        # Actually we need the base of the tensor data area which is where the first tensor data begins
        # The alignment field at kv_end tells us where tensor data starts
        # But tensor offsets are relative to tensor data base
        # So actual tensor file offset = tensor info table end + tensor_offset
        data_base = offset  # This is right after the alignment field
        for t in tensors:
            t['file_offset'] = data_base + t['offset']
    
    return data, tensors


def main():
    if len(sys.argv) < 2:
        print("Usage: gguf_reader.py <gguf_file> [tensor_name_pattern]", file=sys.stderr)
        sys.exit(1)
    
    path = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) > 2 else "ffn_up"
    
    try:
        data, tensors = read_gguf_metadata(path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    DTYPE_NAMES = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q5_0", 4: "Q8_0",
        5: "Q2_0", 6: "Q3_0", 7: "Q4_1", 8: "Q5_1", 9: "Q1_0",
        10: "Q4_2", 11: "Q6_K", 12: "Q8_K", 13: "IQ2_XXS",
        14: "IQ2_XS", 15: "IQ3_XXS", 16: "IQ1_S", 17: "IQ4_NL",
        18: "IQ3_S", 19: "IQ2_S", 20: "IQ4_XS", 21: "I8",
        254: "BF16", 255: "F64"
    }
    
    print(f"\nTotal tensors: {len(tensors)}", file=sys.stderr)
    
    # Find ffn_up tensors
    ffn_up_tensors = [t for t in tensors if 'ffn_up' in t['name'] and 'weight' in t['name']]
    
    print(f"\nffn_up.weight tensors ({len(ffn_up_tensors)}):", file=sys.stderr)
    for i, t in enumerate(ffn_up_tensors[:5]):
        dtype_name = DTYPE_NAMES.get(t['type'], f"DTYPE_{t['type']}")
        dims_str = "×".join(str(d) for d in t['dims'])
        print(f"  [{i}] {t['name']} | {dtype_name} | [{dims_str}] | file_offset={t.get('file_offset','?')}", file=sys.stderr)
    if len(ffn_up_tensors) > 5:
        print(f"  ... and {len(ffn_up_tensors)-5} more", file=sys.stderr)
    
    # Show matching tensors
    matching = [t for t in tensors if pattern in t['name']]
    print(f"\nTensors matching '{pattern}' ({len(matching)}):", file=sys.stderr)
    for t in matching[:20]:
        dtype_name = DTYPE_NAMES.get(t['type'], f"DTYPE_{t['type']}")
        dims_str = "×".join(str(d) for d in t['dims'])
        print(f"  [{t['index']}] {t['name']} | {dtype_name} | [{dims_str}]", file=sys.stderr)


if __name__ == "__main__":
    main()