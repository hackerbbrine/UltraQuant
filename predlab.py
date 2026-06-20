#!/usr/bin/env python
"""Predictor / redundancy lab (techniques #2 neural predictor, #5 cross-layer sharing).

Tests whether weights are predictable from:
  (a) neighbors  -> lag-1 autocorrelation (the 'video-codec delta' premise)
  (b) low rank   -> SVD energy spectrum (predict W from a small basis)
  (c) other layers -> cross-layer correlation + shared-base + per-layer delta
A predictor only compresses if the residual has materially lower energy than W itself.
"""
import numpy as np, sys
from gguf import GGUFReader

Q8 = r"models\Llama-3.1-8B-Q8_0.gguf"

def find(r, n):
    for t in r.tensors:
        if t.name == n: return t
    raise KeyError(n)

def deq_q8(t):
    rows, brow = t.data.shape; nb = brow // 34
    d = t.data.reshape(rows, nb, 34)
    scales = d[:, :, :2].copy().view(np.float16).astype(np.float32)
    qs = d[:, :, 2:].view(np.int8).astype(np.float32)
    return (scales * qs).reshape(rows, nb * 32)

def autocorr(W):
    # lag-1 along rows and cols, averaged
    r = np.mean([np.corrcoef(W[i, :-1], W[i, 1:])[0, 1] for i in np.random.randint(0, W.shape[0], 200)])
    c = np.mean([np.corrcoef(W[:-1, j], W[1:, j])[0, 1] for j in np.random.randint(0, W.shape[1], 200)])
    return r, c

def main():
    r = GGUFReader(Q8)
    base = sys.argv[1] if len(sys.argv) > 1 else "ffn_gate"
    W = deq_q8(find(r, f"blk.16.{base}.weight"))
    print(f"tensor blk.16.{base}  shape {W.shape}")

    print("\n== (a) neighbor predictability (video-codec premise) ==")
    ar, ac = autocorr(W)
    print(f"  lag-1 autocorr  row={ar:+.4f}  col={ac:+.4f}   (|rho|~0 => neighbors carry ~no info)")

    print("\n== (b) low-rank structure (SVD) ==")
    s = np.linalg.svd(W, compute_uv=False)
    e = np.cumsum(s**2) / np.sum(s**2)
    m, n = W.shape; full = min(m, n)
    for frac in (0.90, 0.95, 0.99):
        k = int(np.searchsorted(e, frac)) + 1
        store = k * (m + n); orig = m * n
        print(f"  {int(frac*100)}% energy at rank {k}/{full}  -> store {store/orig:.2f}x of full "
              f"({'compress' if store<orig else 'NO compress'})")

    print("\n== (c) cross-layer redundancy / weight sharing (technique #5) ==")
    Ws = [deq_q8(find(r, f"blk.{L}.{base}.weight")) for L in range(0, 32, 4)]
    mean = np.mean(Ws, axis=0)
    # correlation of adjacent layers
    L0 = deq_q8(find(r, f"blk.15.{base}.weight")); L1 = deq_q8(find(r, f"blk.16.{base}.weight"))
    cc = np.corrcoef(L0.ravel()[::97], L1.ravel()[::97])[0, 1]
    print(f"  corr(blk15, blk16) = {cc:+.4f}")
    # shared base + delta: energy of delta vs original
    for L in (0, 8, 16, 24):
        WL = deq_q8(find(r, f"blk.{L}.{base}.weight"))
        delta = WL - mean
        print(f"  blk.{L}: ||W-mean||^2/||W||^2 = {np.sum(delta**2)/np.sum(WL**2):.3f}  "
              f"(<1 => sharing a base helps; ~1 => layers ~independent)")

if __name__ == "__main__":
    main()
