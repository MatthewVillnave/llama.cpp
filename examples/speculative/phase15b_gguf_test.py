#!/usr/bin/env python3
"""
Simple 7B GGUF extractor - reads tensor data directly using known offsets.
Uses the same approach as the 3B script but with corrected 7B parameters.
"""
import struct, os, json, sys, time, hashlib
import numpy as np

MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
FRESH = "/tmp/prt_sidecars_7b_int8_phase15b_fixed"
AUDIT = "/tmp/prt_phase15b_b_extraction_audit"
FFN, HIDDEN = 18944, 3584
N_LAYERS = 28
QK_K = 256
BPS = 144

def read_gguf_header(path):
    """Read simple header."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        alignment = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]
    return version, n_tensors, alignment, metadata_count

def skip_metadata(path, metadata_count):
    """Skip metadata section by reading each entry."""
    with open(path, 'rb') as f:
        f.read(32)  # header
        
        for _ in range(metadata_count):
            # key_len
            b = f.read(1)
            if not b:
                break
            key_len = b[0]
            if key_len & 0x80:
                more = (f.read(1)[0] << 7)
                key_len = (key_len & 0x7F) | more
            
            f.read(key_len)  # key name
            f.read(4)  # type
            
            # value_len (varint)
            b = f.read(1)
            if not b:
                break
            val_len = b[0]
            if val_len & 0x80:
                more = (f.read(1)[0] << 7)
                val_len = (val_len & 0x7F) | more
            
            f.read(val_len)  # value
    
    return f.tell()

def read_tensor_index(path, ffn_up_only=True):
    """Read all tensor index entries."""
    tensors = {}
    
    version, n_tensors, alignment, metadata_count = read_gguf_header(path)
    print(f"GGUF v{version}, {n_tensors} tensors, {metadata_count} metadata entries")
    
    # Skip metadata
    skip_offset = skip_metadata(path, metadata_count)
    print(f"Metadata ends at offset {skip_offset}")
    
    with open(path, 'rb') as f:
        f.seek(skip_offset)
        
        for i in range(n_tensors):
            name_len_byte = f.read(1)
            if not name_len_byte:
                break
            
            name_len = name_len_byte[0]
            if name_len & 0x80:
                name_len = (name_len & 0x7F) | ((f.read(1)[0]) << 7)
            
            name = f.read(name_len).decode('utf-8', errors='replace')
            
            n_dim = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dim)]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            
            if ffn_up_only and 'ffn_up' in name and 'weight' in name:
                tensors[name] = {'dims': dims, 'dtype': dtype, 'offset': offset}
    
    return tensors

def dequant_q4_k(data, ne0, ne1):
    """Dequantize Q4_K (256-element blocks, 144 bytes each)."""
    n_elements = ne0 * ne1
    result = np.zeros(n_elements, dtype=np.float32)
    n_blocks = n_elements // QK_K
    
    pos = 0
    for b in range(n_blocks):
        d = struct.unpack_from('<e', data, pos)[0]
        dmin = struct.unpack_from('<e', data, pos + 2)[0]
        
        scales = memoryview(data)[pos + 4:pos + 20]
        qdata = memoryview(data)[pos + 20:pos + 144]
        
        qp = 0
        for j in range(QK_K // 64):
            s0 = scales[j * 2]
            s1 = scales[j * 2 + 1]
            sc0 = s0 & 0xF
            m0 = (s0 >> 4) & 0xF
            sc1 = s1 & 0xF
            m1 = (s1 >> 4) & 0xF
            d1 = d * sc0
            m1v = dmin * m0
            d2 = d * sc1
            m2v = dmin * m1
            
            base_qp = j * 128
            
            for l in range(32):
                q = qdata[base_qp + l] & 0xF
                result[base_qp + l * 2] = d1 * ((q + 8)) - m1v
                result[base_qp + l * 2 + 1] = d1 * ((qdata[base_qp + l] >> 4) + 8) - m1v
            
            for l in range(32, 64):
                q = qdata[base_qp + l] & 0xF
                result[base_qp + (l-32) * 2 + 128] = d2 * ((q + 8)) - m2v
                result[base_qp + (l-32) * 2 + 129] = d2 * ((qdata[base_qp + l] >> 4) + 8) - m2v
        
        pos += BPS
    
    return result

def main():
    os.makedirs(FRESH, exist_ok=True)
    os.makedirs(AUDIT, exist_ok=True)
    
    print("=== Testing GGUF extraction ===")
    
    try:
        tensors = read_tensor_index(MODEL)
        print(f"Found {len(tensors)} ffn_up tensors")
        
        for name, info in list(tensors.items())[:3]:
            print(f"  {name}: offset={info['offset']}, shape={info['dims']}, dtype={info['dtype']}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())