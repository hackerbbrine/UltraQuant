#!/usr/bin/env python
"""Anchor: llama.cpp's ACTUAL reconstruction NMSE on a tensor (the real bar to beat).
Dequantize the same tensor from each quant GGUF and compare to the Q8 reference."""
import numpy as np, sys
from gguf import GGUFReader
import gguf

def find(r, n):
    for t in r.tensors:
        if t.name == n: return t
    raise KeyError(n)

def deq_q8(t):
    rows, brow = t.data.shape; nb = brow // 34
    d = t.data.reshape(rows, nb, 34)
    sc = d[:, :, :2].copy().view(np.float16).astype(np.float32)
    qs = d[:, :, 2:].view(np.int8).astype(np.float32)
    return (sc * qs).reshape(rows, nb * 32)

def deq_any(t):
    """Use gguf's dequantize for arbitrary types (IQ2, Q4_K, ...)."""
    return gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)

name = sys.argv[1] if len(sys.argv) > 1 else "blk.16.ffn_gate.weight"
ref = deq_q8(find(GGUFReader(r"models\Llama-3.1-8B-Q8_0.gguf"), name)).ravel()
print(f"tensor {name}  ref(Q8) std {ref.std():.5f}")
for tag, path in [("Q4_K_M", r"models\8b\q4km.gguf"),
                  ("IQ2_M",  r"models\8b\iq2m.gguf"),
                  ("IQ2_XS", r"models\8b\iq2xs.gguf"),
                  ("IQ2_XXS",r"models\8b\iq2xxs.gguf"),
                  ("IQ1_M",  r"models\8b\iq1m.gguf")]:
    try:
        w = deq_any(find(GGUFReader(path), name)).ravel()[:ref.size]
        nm = float(np.sum((ref - w)**2) / np.sum(ref**2))
        print(f"  {tag:8s} actual NMSE vs Q8 = {nm:.5f}")
    except Exception as e:
        print(f"  {tag:8s} ERR {e}")
