#!/usr/bin/env python
"""Protect arbitrary tensor types at higher bits on the trellis base (9.37), via E8 at the matching
step. Usage: python buildalloc.py OUT.gguf down:3 attn_q:3 attn_output:3
Reports the protected set; measure ppl separately with gpu_bench."""
import numpy as np, shutil, time, sys
from gguf import GGUFReader
from qlab import load_WH, rht_rows, irht_rows, q_e8_fast
from buildq import enc_q8

BASE = r"models\8b_sim\q8_trellis.gguf"
args = sys.argv[1:]
LAYERS = None
if "--layers" in args:
    i = args.index("--layers"); LAYERS = set(int(x) for x in args[i+1].split(",")); del args[i:i+2]
OUT  = args[0]
PROT = dict((s.split(":")[0], int(s.split(":")[1])) for s in args[1:])   # type->bits
STEP = {3: 0.48, 4: 0.25, 5: 0.16, 6: 0.11}
SUF  = {"down":"ffn_down","q":"attn_q","o":"attn_output","gate":"ffn_gate","up":"ffn_up",
        "k":"attn_k","v":"attn_v","attn_q":"attn_q","attn_output":"attn_output"}

def quant_step(W, H, step):
    H = np.maximum(H, H.max()*1e-3); sH = np.sqrt(H)[None, :]
    Ah = irht_rows(q_e8_fast(rht_rows(W*sH), fixed_step=step), W.shape[1])
    return (Ah/sH).astype(np.float32)

shutil.copyfile(BASE, OUT)
r = GGUFReader(BASE); tens = {t.name: t for t in r.tensors}
print(f"protecting {PROT} on trellis base", flush=True)
fh = open(OUT, "r+b"); t0 = time.time()
for typ, bits in PROT.items():
    suf = SUF[typ]; step = STEP[bits]
    for t in [t for t in r.tensors if t.name.endswith(f".{suf}.weight")]:
        if LAYERS is not None and int(t.name.split(".")[1]) not in LAYERS:
            continue
        W, H = load_WH(t.name)
        enc = enc_q8(quant_step(W, H, step))
        assert enc.nbytes == t.n_bytes
        fh.seek(int(t.data_offset)); fh.write(enc.tobytes())
fh.close()
print(f"[done] {time.time()-t0:.0f}s -> {OUT}", flush=True)
