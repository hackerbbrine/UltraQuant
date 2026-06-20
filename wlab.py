#!/usr/bin/env python
"""Weight rate-distortion lab on REAL extracted weights (no format constraints).

Tests, on actual Llama-3.1-8B tensors (dequantized from Q8_0 ~ ground truth):
  #1 codebook/dictionary: uniform scalar vs k-means scalar (optimal non-uniform) vs
     vector-quant (exploit neighbor correlation) vs Hadamard-rotated (incoherence).
  #2 predictor: mean / low-rank(SVD) / cross-layer prediction -> residual compressibility.
Metric: NMSE = ||W - W_hat||^2 / ||W||^2  at a given rate (bits/weight). Lower = better.
Anchor: llama.cpp's ACTUAL IQ2_XXS reconstruction NMSE on the same tensor (the bar to beat).
"""
import numpy as np, sys, time
from gguf import GGUFReader
from scipy.cluster.vq import kmeans2, vq

Q8 = r"models\Llama-3.1-8B-Q8_0.gguf"

def find(reader, name):
    for t in reader.tensors:
        if t.name == name:
            return t
    raise KeyError(name)

def deq_q8(t):
    """Dequantize a Q8_0 gguf tensor -> float32 2D (rows, cols)."""
    rows, brow = t.data.shape          # brow = nblocks*34
    nb = brow // 34
    d = t.data.reshape(rows, nb, 34)
    scales = d[:, :, :2].copy().view(np.float16).astype(np.float32)   # (rows,nb,1)
    qs = d[:, :, 2:].view(np.int8).astype(np.float32)                 # (rows,nb,32)
    w = scales * qs
    return w.reshape(rows, nb * 32)

def nmse(W, Wh):
    return float(np.sum((W - Wh)**2) / np.sum(W**2))

# ---------- scalar quantizers ----------
def uniform_q(W, bits, blocksize=32):
    """Per-block symmetric absmax uniform quant (what llama.cpp K-quants approximate)."""
    n = W.size
    pad = (-n) % blocksize
    x = np.concatenate([W.ravel(), np.zeros(pad, W.dtype)]).reshape(-1, blocksize)
    amax = np.max(np.abs(x), axis=1, keepdims=True) + 1e-12
    lvl = (1 << (bits - 1)) - 1
    q = np.round(x / amax * lvl)
    xh = q / lvl * amax
    return xh.ravel()[:n].reshape(W.shape)

def _nmse_sample(x, xh):
    return float(np.sum((x - xh)**2) / np.sum(x**2))

def kmeans_scalar_nmse(W, bits, samp_n=1_000_000):
    """NMSE on a representative subsample (distribution is what matters)."""
    K = 1 << bits
    flat = W.ravel().astype(np.float64)
    s = flat[np.random.RandomState(0).randint(0, flat.size, min(samp_n, flat.size))]
    cb, _ = kmeans2(s[:300_000].reshape(-1, 1), K, minit='++', seed=0)
    idx, _ = vq(s.reshape(-1, 1), cb)
    return _nmse_sample(s, cb[idx, 0])

def vq_nmse(W, dim, bits_per_vec, samp_n=1_000_000):
    """Vector quant NMSE on subsample. rate = bits_per_vec/dim bits/weight. K capped <=4096."""
    K = 1 << bits_per_vec
    flat = W.ravel()
    pad = (-flat.size) % dim
    x = np.concatenate([flat, np.zeros(pad, flat.dtype)]).reshape(-1, dim).astype(np.float64)
    sel = np.random.RandomState(0).randint(0, x.shape[0], min(samp_n // dim, x.shape[0]))
    xs = x[sel]
    cb, _ = kmeans2(xs[:200_000], K, minit='++', seed=0)
    idx, _ = vq(xs, cb)
    return _nmse_sample(xs, cb[idx]), bits_per_vec / dim

def hadamard_q(W, bits, bs=32):
    """Random-sign + Hadamard rotate each block (incoherence), uniform quant, invert."""
    H = hadamard(bs) / np.sqrt(bs)
    n = W.size; pad = (-n) % bs
    x = np.concatenate([W.ravel(), np.zeros(pad, W.dtype)]).reshape(-1, bs).astype(np.float64)
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=bs)
    xr = (x * signs) @ H.T
    amax = np.max(np.abs(xr), axis=1, keepdims=True) + 1e-12
    lvl = (1 << (bits - 1)) - 1
    q = np.round(xr / amax * lvl) / lvl * amax
    xh = (q @ H) * signs
    return xh.ravel()[:n].reshape(W.shape).astype(np.float32)

def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H

def main():
    r = GGUFReader(Q8)
    name = sys.argv[1] if len(sys.argv) > 1 else "blk.16.ffn_gate.weight"
    W = deq_q8(find(r, name))
    print(f"tensor {name}  shape {W.shape}  {W.size/1e6:.1f}M weights  "
          f"std {W.std():.4f}  kurtosis {float(((W-W.mean())**4).mean()/W.var()**2):.2f}")

    print("\n== #1 codebook rate-distortion (NMSE vs bits/weight, lower=better) ==", flush=True)
    for bits in (2, 3, 4):
        u = nmse(W, uniform_q(W, bits))
        k = kmeans_scalar_nmse(W, bits)
        print(f"  {bits}-bit: uniform NMSE={u:.5f}  kmeans NMSE={k:.5f}  "
              f"(kmeans {100*(u-k)/u:+.1f}% vs uniform)", flush=True)
    print("  vector-quant at matched rates (exploit neighbor correlation):", flush=True)
    for dim, bpv in ((2, 4), (4, 8), (2, 6), (2, 8)):   # rates: 2, 2, 3, 4 (K<=256, fast)
        nm, rate = vq_nmse(W, dim, bpv)
        print(f"    dim={dim} {bpv}b/vec = {rate:.2f} b/w : NMSE={nm:.5f}", flush=True)
    print("  Hadamard-rotated (incoherence) vs uniform:", flush=True)
    for bits in (2, 3, 4):
        h = nmse(W, hadamard_q(W, bits)); u = nmse(W, uniform_q(W, bits))
        print(f"    {bits}-bit: uniform={u:.5f}  hadamard={h:.5f}  ({100*(u-h)/u:+.1f}%)", flush=True)

if __name__ == "__main__":
    main()
