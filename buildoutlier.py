#!/usr/bin/env python
"""Sparse outlier overlay (SpQR-style) on the e8mix winner. Keep the top-p% weights by
OUTPUT-error contribution (|W-W_hat|*sqrt(H)) at full precision; rest stays 2-bit E8.
Directly exploits the measured extreme-outlier channels (kurtosis up to 2157).
Overhead ~ p% * (16 + index) bits. Reports ppl + bitrate cost."""
import numpy as np, shutil, time, sys
from gguf import GGUFReader
from qlab import load_WH, rht_rows, irht_rows, q_e8_fast
from buildq import enc_q8

P = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0     # percent kept at full precision
BASE = r"models\8b_sim\q8_e8mix.gguf"
OUT  = rf"models\8b_sim\q8_e8mix_out{P}.gguf"
BULK = ("attn_q", "attn_output", "ffn_gate", "ffn_up", "ffn_down")

def recon_e8(W, H):
    H = np.maximum(H, H.max()*1e-3); sH = np.sqrt(H)[None, :]
    A = W*sH; in0 = A.shape[1]
    return (irht_rows(q_e8_fast(rht_rows(A), 2.0, calib_key='global'), in0)/sH).astype(np.float64)

shutil.copyfile(BASE, OUT)
r = GGUFReader(BASE)
bulk = [t for t in r.tensors if t.name.endswith(".weight") and any(f".{k}.weight" in t.name for k in BULK)]
print(f"outlier overlay p={P}% on {len(bulk)} bulk tensors", flush=True)
fh = open(OUT, "r+b"); t0 = time.time(); kept = 0; tot = 0
for i, t in enumerate(bulk):
    W, H = load_WH(t.name); sH = np.sqrt(np.maximum(H, H.max()*1e-3))
    Wh = recon_e8(W, H)
    score = np.abs(W - Wh) * sH[None, :]
    thr = np.percentile(score, 100 - P)
    mask = score >= thr
    Wh[mask] = W[mask]                                    # restore full precision on outliers
    kept += int(mask.sum()); tot += W.size
    enc = enc_q8(Wh.astype(np.float32))
    assert enc.nbytes == t.n_bytes
    fh.seek(int(t.data_offset)); fh.write(enc.tobytes())
    if i % 32 == 0: print(f"  [{i+1}/{len(bulk)}] ({time.time()-t0:.0f}s)", flush=True)
fh.close()
ovh = (kept/tot) * (16 + 12) / 1.0                        # ~16b value + ~12b index per outlier, /weight
print(f"[done] kept {100*kept/tot:.2f}% outliers -> +{ovh:.3f} bpw overhead. {OUT}", flush=True)
