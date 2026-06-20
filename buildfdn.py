#!/usr/bin/env python
"""Allocation sweep: on the e8mix winner (10.04), ALSO protect ffn_down (bump 2bpw -> ~3bpw via
finer E8 step). ffn_down is 26.9% of params so this RAISES avg bitrate to ~2.35 -- report ppl AND
the bitrate cost. Tests whether extra ffn_down precision helps (diagnosis says I already beat IQ2
on ffn_down at 2bpw, so expect small gains)."""
import numpy as np, shutil, time, sys
from gguf import GGUFReader
from qlab import load_WH, rht_rows, irht_rows, q_e8_fast
from buildq import enc_q8

BASE = sys.argv[1] if len(sys.argv) > 1 else r"models\8b_sim\q8_e8mix.gguf"
OUT  = sys.argv[2] if len(sys.argv) > 2 else r"models\8b_sim\q8_e8mix_fdn.gguf"
FDN_STEP = 0.48   # ~3 bpw

def quant_step(W, H, step):
    H = np.maximum(H, H.max()*1e-3); sH = np.sqrt(H)[None, :]
    A = W*sH; in0 = A.shape[1]
    Ah = irht_rows(q_e8_fast(rht_rows(A), fixed_step=step), in0)
    return (Ah/sH).astype(np.float32)

shutil.copyfile(BASE, OUT)
r = GGUFReader(BASE)
fdn = [t for t in r.tensors if t.name.endswith(".ffn_down.weight")]
print(f"protecting {len(fdn)} ffn_down tensors at step {FDN_STEP} (~3bpw)", flush=True)
fh = open(OUT, "r+b"); t0 = time.time()
for t in fdn:
    W, H = load_WH(t.name)
    enc = enc_q8(quant_step(W, H, FDN_STEP))
    assert enc.nbytes == t.n_bytes
    fh.seek(int(t.data_offset)); fh.write(enc.tobytes())
fh.close()
print(f"[done] {len(fdn)} ffn_down in {time.time()-t0:.0f}s -> {OUT}", flush=True)
