#!/usr/bin/env python
"""Mixed-allocation model: start from the 2bpw E8 model, re-quantize ONLY attn_k/attn_v
at high precision (E8 step 0.22 ~= 4bpw). V/K are ~4% of params -> avg stays ~2.08 bpw,
matching IQ2's 2.06.  This adds my assigned extension: importance-based bit allocation
on top of incoherence+lattice. Tests whether it now beats IQ2 end-to-end."""
import numpy as np, shutil, time, os
from gguf import GGUFReader
from qlab import load_WH, rht_rows, irht_rows, q_e8_fast
from buildq import enc_q8

BASE = r"models\8b_sim\q8_e8.gguf"        # bulk already at 2bpw E8
OUT  = r"models\8b_sim\q8_e8mix.gguf"
VK_STEP = 0.22

def quant_e8_step(W, H, step):
    H = np.maximum(H, H.max()*1e-3); sH = np.sqrt(H)[None, :]
    A = W*sH; in0 = A.shape[1]
    Ah = irht_rows(q_e8_fast(rht_rows(A), fixed_step=step), in0)
    return (Ah/sH).astype(np.float32)

shutil.copyfile(BASE, OUT)
r = GGUFReader(BASE)
vk = [t for t in r.tensors if t.name.endswith(".weight") and (".attn_k." in t.name or ".attn_v." in t.name)]
print(f"re-quantizing {len(vk)} V/K tensors at step {VK_STEP} (~4bpw)", flush=True)
fh = open(OUT, "r+b"); t0 = time.time()
for i, t in enumerate(vk):
    W, H = load_WH(t.name)
    enc = enc_q8(quant_e8_step(W, H, VK_STEP))
    assert enc.nbytes == t.n_bytes
    fh.seek(int(t.data_offset)); fh.write(enc.tobytes())
fh.close()
print(f"[done] {len(vk)} tensors in {time.time()-t0:.0f}s -> {OUT}", flush=True)
