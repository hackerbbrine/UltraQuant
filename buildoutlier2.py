#!/usr/bin/env python
"""Sparse outlier overlay on the trellis model (9.37): restore the top-p% highest-output-error weights
(|W - W_trellis| * sqrt(H)) to full precision. Targets the bulk 2-bit error directly for ~0.3bpw/1%.
Reports ppl; bitrate overhead ~ p% * (16 + ~13 idx) bits/weight."""
import numpy as np, shutil, time, sys
from gguf import GGUFReader
from qlab import load_WH, _reader, _find, deq_q8
from buildq import enc_q8
import model as M

P = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
BASE = r"models\8b_sim\q8_trellis.gguf"
OUT  = rf"models\8b_sim\q8_trellis_out{P}.gguf"
BULK = ("attn_q","attn_output","ffn_gate","ffn_up","ffn_down")

shutil.copyfile(BASE, OUT)
rt = GGUFReader(BASE); tt = {t.name: t for t in rt.tensors}
bulk = [t for t in rt.tensors if t.name.endswith(".weight") and any(f".{k}.weight" in t.name for k in BULK)]
print(f"outlier overlay p={P}% on {len(bulk)} bulk tensors of trellis model", flush=True)
fh = open(OUT, "r+b"); t0=time.time(); kept=0; tot=0
for i, t in enumerate(bulk):
    W, H = load_WH(t.name)                      # original W + imatrix diag H
    Wh = deq_q8(tt[t.name]).astype(np.float64)  # trellis reconstruction
    sH = np.sqrt(np.maximum(H, H.max()*1e-3))
    score = np.abs(W - Wh) * sH[None, :]
    thr = np.percentile(score, 100 - P)
    mask = score >= thr
    Wh[mask] = W[mask]
    kept += int(mask.sum()); tot += W.size
    enc = enc_q8(Wh.astype(np.float32))
    assert enc.nbytes == t.n_bytes
    fh.seek(int(t.data_offset)); fh.write(enc.tobytes())
    if i % 32 == 0: print(f"  [{i+1}/{len(bulk)}] ({time.time()-t0:.0f}s)", flush=True)
fh.close()
ovh = (kept/tot)*(16+13)
print(f"[done] kept {100*kept/tot:.2f}% outliers -> +{ovh:.3f} bpw. {OUT}", flush=True)
