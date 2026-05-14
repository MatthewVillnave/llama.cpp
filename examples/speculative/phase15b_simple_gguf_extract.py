#!/usr/bin/env python3
"""
Minimal GGUF ffn_up extractor using memory mapping and fixed parameters.
Bypasses metadata parsing and reads tensor index directly.
"""
import struct, os, sys, hashlib, time
import numpy as np
import mmap

MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
FRESH = "/tmp/prt_sidecars_7b_int8_phase15b_fixed"
AUDIT = "/tmp/prt_phase15b_b_extraction_audit"
FFN, HIDDEN = 18944, 3584
N_LAYERS = 28
QK_K = 256
BPS = 144

def read_varint_le(f):
    """Readlittle-endian varint (7 bits per byte)."""
    result = 0
    shift = 0
    while True:
        b = f.read(1)
        if not b:
            return None
        b = b[0]
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result
        shift += 7

def find_tensor_offsets(path):
    """Find ffn_up.weight tensor offsets by scanning tensor index."""
    print(f"Scanning {path} for ffn_up tensors...")
    
    with open(path, 'rb') as f:
        f.read(32)  # Skip header
        
        # Read metadata entries
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        print(f"Metadata entries: {metadata_count}")
        
        # Skip metadata section
        for _ in range(metadata_count):
            key_len = read_varint_le(f)
            if key_len:
                f.read(key_len)
                f.read(4)  # type
                val_len = read_varint_le(f)
                if val_len:
                    f.read(val_len)
        
        # Now at tensor index
        print(f"Offset after metadata: {f.tell()}")
        
        tensors = {}
        while True:
            pos = f.tell()
            name_len_byte = f.read(1)
            if not name_len_byte:
                break
            name_len = name_len_byte[0]
            if name_len & 0x80:
                # continued varint
                name_len = (name_len & 0x7F) | ((f.read(1)[0]) << 7)
            
            name = f.read(name_len).decode('utf-8', errors='replace')
            if not name:
                break
                
            n_dim = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dim)]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            
            if 'ffn_up' in name and 'weight' in name and 'block' in name:
                tensors[name] = {'dims': dims, 'dtype': dtype, 'offset': offset}
                print(f"  {name}: {offset}, shape {dims}")
                
        print(f"Found {len(tensors)} ffn_up.weight tensors")
        return tensors

def dequant_block(data):
    """Dequantize one Q4_K block (144 bytes → 256 float32)."""
    W = np.empty(QK_K, dtype=np.float32)
    d = struct.unpack_from('<e', data, 0)[0]
    dmin = struct.unpack_from('<e', data, 2)[0]
    
    for j in range(4):
        s = struct.unpack_from('<B', data, 4 + j)[0]
        sc0 = s & 0xF
        m0 = (s >> 4) & 0xF
        d1 = d * sc0
        d2 = dmin * m0
        
        base = j * 32
        for k in range(32):
            # low nibble
            b = data[20 + base + k]
            W[base + k * 2] = d1 * (b & 0xF) if (b & 0xF) <= 7 else (-d1 * ((b & 0xF) - 16)
            # high nibble
            W[base + k * 2 + 1] = d1 * ((b >> 4) & 0xF)
    
    return W

def extract_layer_mm(path, layer_idx):
    """Extract one layer using mmap."""
    tensor_name = f"blk.{layer_idx}.ffn_up.weight"
    tensors = find_tensor_offsets(path)
    
    if tensor_name not in tensors:
        raise ValueError(f"Tensor {tensor_name} not found")
    
    info = tensors[tensor_name]
    offset = info['offset']
    ne0, ne1 = info['dims'][:2]
    
    n_elements = ne0 * ne1
    n_blocks = n_elements // QK_K
    
    with open(path, 'rb') as f:
        f.seek(offset)
        data = f.read(n_blocks * BPS)
    
    W_rows = []
    for i in range(n_blocks):
        block = dequant_block(data[i * BPS:(i+1) * BPS])
        W_rows.append(block)
    
    return np.stack(W_rows, axis=0)

def main():
    os.makedirs(FRESH, exist_ok=True)
    os.makedirs(AUDIT, exist_ok=True)
    
    print("=== GGUF Extraction + INT8 Parity (via mmapped scan) ===")
    
    try:
        tensors = find_tensor_offsets(MODEL)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    if not tensors:
        print("No tensors found")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())