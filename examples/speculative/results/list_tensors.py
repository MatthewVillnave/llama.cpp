#!/usr/bin/env python3
"""List tensors from GGUF model file."""
import struct
import sys
import json

def read_string(f):
    length = struct.unpack('I', f.read(4))[0]
    if length == 0 or length >> 31:
        # varint
        if length >> 31 == 1:
            length = length & 0x7FFFFFFF
        else:
            return ""
    return f.read(length).decode('utf-8', errors='replace')

def read_string(f):
    length = struct.unpack('I', f.read(4))[0]
    return f.read(length).decode('utf-8', errors='replace')

def list_tensors(gguf_path):
    with open(gguf_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('I', f.read(4))[0]
        num_tensors = struct.unpack('Q', f.read(8))[0]
        alignment = struct.unpack('Q', f.read(8))[0]
        
        tensors = []
        for i in range(num_tensors):
            try:
                name = read_string(f)
            except:
                name = f"tensor_{i}"
            
            n_dims = struct.unpack('I', f.read(4))[0]
            shape = []
            for d in range(n_dims):
                dim = struct.unpack('Q', f.read(8))[0]
                shape.append(dim)
            
            # Pad to 8 dimensions
            while len(shape) < 8:
                shape.append(1)
            
            dtype = struct.unpack('I', f.read(4))[0]
            offset = struct.unpack('Q', f.read(8))[0]
            
            # GGUF dtype names
            dtype_map = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q5_0', 4: 'Q8_0', 
                         5: 'Q2_0', 6: 'Q3_0', 7: 'Q4_1', 8: 'Q5_1', 9: 'Q1_0',
                         10: 'Q4_2', 11: 'Q6_K', 12: 'Q8_K', 13: 'IQ2_XXS', 14: 'IQ2_XS',
                         15: 'IQ3_XXS', 16: 'IQ1_S', 17: 'IQ4_NL', 18: 'IQ3_S', 19: 'IQ2_S',
                         20: 'IQ4_XS', 21: 'I8', 22: 'Q4_0_4_4', 23: 'Q4_0_4_8', 24: 'Q4_0_2_4',
                         25: 'Q4_0_2_8', 26: 'Q2_0_4_4', 27: 'Q2_0_4_8', 28: 'Q2_0_2_4', 29: 'Q2_0_2_8',
                         30: 'Q4_K_4_4', 31: 'Q4_K_4_8', 32: 'Q4_K_2_4', 33: 'Q4_K_2_8',
                         34: 'Q6_K_4_4', 35: 'Q6_K_4_8', 36: 'Q6_K_2_4', 37: 'Q6_K_2_8',
                         255: 'F64', 254: 'BF16'}
            
            tensors.append({
                'name': name,
                'shape': list(shape),
                'dtype': dtype_map.get(dtype, f'UNKNOWN_{dtype}'),
                'layer': None
            })
        
        return tensors

if __name__ == '__main__':
    model_path = '/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf'
    tensors = list_tensors(model_path)
    
    # Categorize
    ffn_up_tensors = [t for t in tensors if 'ffn_up' in t['name']]
    all_ffn = [t for t in tensors if 'ffn' in t['name'].lower()]
    
    print(f"Total tensors: {len(tensors)}")
    print(f"ffn_up tensors: {len(ffn_up_tensors)}")
    print(f"All ffn tensors: {len(all_ffn)}")
    print()
    print("=== FFN_UP Tensors ===")
    for t in ffn_up_tensors:
        print(f"  {t['name']} shape={t['shape']} dtype={t['dtype']}")
    
    # Output to file
    with open('/home/matthew-villnave/llama.cpp/examples/speculative/results/phase10a_direct_01_tensor_inventory.md', 'w') as f:
        f.write("# Phase 10A Tensor Inventory\n\n")
        f.write(f"**Timestamp:** 2026-04-30 08:40 EDT\n\n")
        f.write(f"**Total tensors:** {len(tensors)}\n\n")
        f.write(f"**ffn_up tensors found:** {len(ffn_up_tensors)}\n\n")
        f.write("## ffn_up Tensors\n\n")
        for t in ffn_up_tensors:
            f.write(f"- `{t['name']}` shape={t['shape']} dtype={t['dtype']}\n")
        f.write("\n## All ffn Tensors\n\n")
        for t in all_ffn:
            f.write(f"- `{t['name']}` shape={t['shape']} dtype={t['dtype']}\n")
    
    with open('/home/matthew-villnave/llama.cpp/examples/speculative/results/phase10a_direct_01_tensor_inventory.json', 'w') as f:
        json.dump({
            'total_tensors': len(tensors),
            'ffn_up_count': len(ffn_up_tensors),
            'ffn_up_tensors': ffn_up_tensors,
            'all_ffn_tensors': all_ffn
        }, f, indent=2)
    
    print(f"\nInventory written to results/")