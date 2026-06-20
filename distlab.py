#!/usr/bin/env python
"""Test the assumptions behind the '2-bit floor'. Two questions:
 1. Are weights actually Gaussian i.i.d.? Measure distribution at every scale.
 2. Is the quantization ERROR low-rank (esp. the OUTPUT-relevant error E.diag(sqrt H))?
    If a few directions dominate, a cheap low-rank correction can erase most output error.
"""
import numpy as np
from qlab import load_WH, rht_rows, irht_rows, q_e8_fast, deq_iq

def kurt(x):
    x = x - x.mean(); v = x.var() + 1e-12
    return float((x**4).mean()/v**2)

def q2bit_recon(W, H):
    """incoherence+E8 ~2bpw reconstruction in weight space (the e8mix bulk method)."""
    H = np.maximum(H, H.max()*1e-3); sH = np.sqrt(H)[None, :]
    A = W*sH; in0 = A.shape[1]
    Ah = irht_rows(q_e8_fast(rht_rows(A), 2.0, calib_key='g'), in0)
    return (Ah/sH).astype(np.float64)

for nm in ['blk.16.attn_q.weight', 'blk.16.ffn_gate.weight', 'blk.16.ffn_down.weight', 'blk.16.attn_v.weight']:
    W, H = load_WH(nm)
    print(f"\n=== {nm}  shape {W.shape} ===", flush=True)
    # --- distribution at scales ---
    col_k = np.array([kurt(W[:, j]) for j in np.random.RandomState(0).randint(0, W.shape[1], 300)])
    col_v = W.var(0)
    print(f"  kurtosis: whole={kurt(W.ravel()):.2f}  per-col median={np.median(col_k):.2f} "
          f"p95={np.percentile(col_k,95):.2f} max={col_k.max():.2f}  (Gaussian=3)", flush=True)
    print(f"  per-col variance spread: max/median={col_v.max()/np.median(col_v):.1f}  "
          f"top-1% cols hold {100*np.sort(col_v)[::-1][:max(1,W.shape[1]//100)].sum()/col_v.sum():.0f}% of variance", flush=True)
    # --- quantization error rank ---
    sH = np.sqrt(np.maximum(H, H.max()*1e-3))
    Wh = q2bit_recon(W, H)
    E = W - Wh                                   # weight-space error
    Ew = E * sH[None, :]                          # OUTPUT-relevant error (Hessian-weighted)
    for tag, M in [("raw E", E), ("weighted E.sqrtH", Ew)]:
        s = np.linalg.svd(M, compute_uv=False)
        e = np.cumsum(s**2)/np.sum(s**2)
        full = len(s)
        r50 = int(np.searchsorted(e, 0.5))+1
        r90 = int(np.searchsorted(e, 0.9))+1
        print(f"  {tag}: rank50%={r50}/{full} ({100*r50/full:.0f}%)  rank90%={r90}/{full} ({100*r90/full:.0f}%)", flush=True)
