#!/usr/bin/env python3
"""Extract ffn_up tensor metadata from GGUF using raw binary parsing."""
import struct
import sys

def read_string(f):
    length = struct.unpack('I', f.read(4))[0]
    if length >> 31:
        length = length & 0x7FFFFFFF
    if length == 0:
        return ""
    data = f.read(length)
    if len(data) < length:
        return ""
    return data.decode('utf-8', errors='replace')

path = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
print(f"=== PRT Phase 10A: GGUF Tensor Extraction (Python) ===")
print(f"File: {path}\n")

with open(path, 'rb') as f:
    magic, version = struct.unpack('II', f.read(8))
    num_tensors, alignment, data_offset = struct.unpack('QQQ', f.read(24))
    
    print(f"GGUF: magic=0x{magic:x} version={version}")
    print(f"Tensors: {num_tensors}, Alignment: {alignment}, Data offset: {data_offset}")
    print()
    
    ffn_up_count = 0
    for i in range(num_tensors):
        try:
            name = read_string(f)
            n_dims = struct.unpack('I', f.read(4))[0]
            dims = list(struct.unpack(f'{n_dims}Q', f.read(8 * n_dims)))
            dtype = struct.unpack('I', f.read(4))[0]
            offset = struct.unpack('Q', f.read(8))[0]
            
            if 'ffn_up' in name and 'weight' in name:
                ffn_up_count += 1
                dtype_map = {0:'F32',1:'F16',2:'Q4_0',3:'Q5_0',4:'Q8_0',5:'Q2_0',6:'Q3_0',7:'Q4_1',8:'Q5_1',9:'Q1_0',10:'Q4_2',11:'Q6_K',12:'Q8_K',13:'IQ2_XXS',14:'IQ2_XS',15:'IQ3_XXS',16:'IQ1_S',17:'IQ4_NL',18:'IQ3_S',19:'IQ2_S',20:'IQ4_XS',21:'I8',254:'BF16',255:'F64'}
                dt = dtype_map.get(dtype, f'UNK_{dtype}')
                size = dims[0] * dims[1]
                if dt.startswith('Q'):
                    size = size // 2  # approximate
                if ffn_up_count <= 5:
                    print(f"FFN_UP[{ffn_up_count}]: idx={i} name={name}")
                    print(f"  shape={dims} type={dt} offset={offset}")
                    
        except Exception as e:
            break
    
    print(f"\n=== RESULT ===")
    print(f"ffn_up tensors found: {ffn_up_count}")