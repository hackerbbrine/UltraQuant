#!/usr/bin/env python
"""Trellis-bulk model: start from e8mix (winner, 10.04), overwrite the 5 BULK linear types
with QTIP trellis (L=12, R=2, seg=64, scale_group=256, fixed ~2.06 bpw). Keep V/K @ E8 ~4bpw.
RESUMABLE: checkpoints completed tensors to a .done file; re-running skips them."""
import numpy as np, shutil, time, os
from gguf import GGUFReader
from qlab import load_WH, rht_rows, irht_rows
from buildq import enc_q8
from trellis import trellis_quant_fast

BASE = r"models\8b_sim\q8_e8mix.gguf"
OUT  = r"models\8b_sim\q8_trellis.gguf"
DONE = OUT + ".done"
BULK = ("attn_q", "attn_output", "ffn_gate", "ffn_up", "ffn_down")

def quant_trellis(W, H, L=12):
    H = np.maximum(H, H.max()*1e-3); sH = np.sqrt(H)[None, :]
    A = W*sH; in0 = A.shape[1]
    Ah = irht_rows(trellis_quant_fast(rht_rows(A), L=L, R=2, seg=64, scale_group=256, mem_gb=4.0), in0)
    return (Ah/sH).astype(np.float32)

done = set()
if os.path.exists(OUT) and os.path.exists(DONE):
    done = set(open(DONE).read().split())
    print(f"RESUME: {len(done)} tensors already done", flush=True)
else:
    print(f"[copy] {BASE} -> {OUT}", flush=True); shutil.copyfile(BASE, OUT)
    open(DONE, "w").close()

r = GGUFReader(BASE)
bulk = [t for t in r.tensors if t.name.endswith(".weight") and any(f".{k}.weight" in t.name for k in BULK)]
todo = [t for t in bulk if t.name not in done]
print(f"trellis-quantizing {len(todo)}/{len(bulk)} bulk tensors (L=12 seg=64)", flush=True)
fh = open(OUT, "r+b"); t0 = time.time()
for i, t in enumerate(todo):
    W, H = load_WH(t.name)
    enc = enc_q8(quant_trellis(W, H))
    assert enc.nbytes == t.n_bytes
    fh.seek(int(t.data_offset)); fh.write(enc.tobytes()); fh.flush()
    with open(DONE, "a") as d: d.write(t.name + "\n")
    if i % 16 == 0:
        print(f"  [{i+1}/{len(todo)}] {t.name} ({time.time()-t0:.0f}s)", flush=True)
fh.close()
print(f"[done] {len(todo)} tensors in {time.time()-t0:.0f}s -> {OUT}", flush=True)
