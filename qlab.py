#!/usr/bin/env python
"""Advanced quantization lab (QuIP#/E8/QTIP/AQLM frontier) on REAL extracted weights.

FAIR metric (the whole point): imatrix-weighted NMSE = the GPTQ/imatrix objective.
  Define A = W * sqrt(H) over input columns (H_j = mean activation energy of channel j).
  Then unweighted_NMSE(A, A_hat) == imatrix-weighted NMSE(W, W_hat).  This is what perplexity
  cares about, and what llama.cpp IQ2 optimizes -> apples-to-apples.
Incoherence = orthogonal rotation V on the input axis of A (preserves ||.||, Gaussianizes).

All methods quantize A (or rotated A) unweighted; we report NMSE in that space and compare to
llama.cpp IQ2_XXS at equal bits/weight.
"""
import numpy as np, sys, time
from gguf import GGUFReader
import gguf
from scipy.spatial import cKDTree

Q8 = r"models\Llama-3.1-8B-Q8_0.gguf"
IM = r"models\8b\imatrix.dat"

# ---------- IO ----------
def _find(r, n):
    for t in r.tensors:
        if t.name == n: return t
    raise KeyError(n)

def deq_q8(t):
    rows, brow = t.data.shape; nb = brow // 34
    d = t.data.reshape(rows, nb, 34)
    sc = d[:, :, :2].copy().view(np.float16).astype(np.float32)
    qs = d[:, :, 2:].view(np.int8).astype(np.float32)
    return (sc * qs).reshape(rows, nb * 32)

_RD = {}
def _reader(path):
    if path not in _RD:
        _RD[path] = GGUFReader(path)
    return _RD[path]

def load_WH(name):
    W = deq_q8(_find(_reader(Q8), name))              # (out, in)
    ri = _reader(IM)
    s2 = np.asarray(_find(ri, name + ".in_sum2").data, dtype=np.float64)
    cnt = float(np.asarray(_find(ri, name + ".counts").data)[0])
    H = (s2 / cnt).astype(np.float64)                 # (in,) mean x^2
    return W.astype(np.float64), H

def deq_iq(path, name):
    t = _find(_reader(path), name)
    return gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float64)

# ---------- metric ----------
def wnmse(A, Ah):
    return float(np.sum((A - Ah)**2) / np.sum(A**2))

# ---------- incoherence: fast Walsh-Hadamard on last axis (n power of 2) ----------
def fwht(x):
    x = x.copy(); n = x.shape[-1]; h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2*h), 2, h)
        a = x[..., 0, :]; b = x[..., 1, :]
        x = np.concatenate([a + b, a - b], axis=-1).reshape(*x.shape[:-3], n)
        h *= 2
    return x / np.sqrt(2)  # per stage; total /sqrt(n) overall handled below

def _next_pow2(n):
    p = 1
    while p < n: p *= 2
    return p

def rht_rows(A, seed=0):
    """Randomized Hadamard Transform along input axis (cols). Pads in-dim to next 2^k.
    Orthogonal -> invertible: A_rot = rht_rows(A); A ~= irht_rows(A_rot)[:, :in]."""
    out, n0 = A.shape
    n = _next_pow2(n0)
    if n != n0:
        A = np.concatenate([A, np.zeros((out, n - n0))], axis=1)
    d = np.random.RandomState(seed).choice([-1.0, 1.0], size=n)
    x = (A * d).copy(); h = 1
    while h < n:
        x = x.reshape(out, n // (2*h), 2, h)
        a = x[:, :, 0, :]; b = x[:, :, 1, :]
        x = np.concatenate([a + b, a - b], axis=2).reshape(out, n)
        h *= 2
    return x / np.sqrt(n)

def irht_rows(X, n0, seed=0):
    """Inverse of rht_rows; returns (out, n0)."""
    out, n = X.shape
    x = X.copy() * np.sqrt(n); h = 1
    while h < n:                      # Hadamard is symmetric & its own inverse (up to scale)
        x = x.reshape(out, n // (2*h), 2, h)
        a = x[:, :, 0, :]; b = x[:, :, 1, :]
        x = np.concatenate([a + b, a - b], axis=2).reshape(out, n)
        h *= 2
    d = np.random.RandomState(seed).choice([-1.0, 1.0], size=n)
    x = (x / n) * d
    return x[:, :n0]

# ---------- quantizers (operate on a matrix, per-row scaling) ----------
def q_uniform(A, bpw, bs=32):
    n = A.shape[1]
    x = A.reshape(-1, bs)
    amax = np.max(np.abs(x), axis=1, keepdims=True) + 1e-12
    lvl = (1 << (int(round(bpw)) - 1)) - 1
    xh = np.round(x / amax * lvl) / lvl * amax
    return xh.reshape(A.shape)

# ---------- E8 lattice ----------
_E8_CACHE = {}
def e8_codebook(K):
    if K in _E8_CACHE: return _E8_CACHE[K]
    rng = range(-3, 4)
    grids = np.array(np.meshgrid(*[rng]*8, indexing='ij')).reshape(8, -1).T.astype(np.float64)
    d8 = grids[(grids.sum(1) % 2 == 0)]                         # D8: even sum
    half = grids + 0.5
    d8h = half[((half - 0.5).sum(1).astype(int) % 2 == 0)]      # D8 + 1/2 coset
    pts = np.concatenate([d8, d8h], 0)
    norm = (pts**2).sum(1)
    order = np.argsort(norm)
    cb = pts[order[:K]]
    cb = cb / np.sqrt((cb**2).sum(1).mean() / 8)               # normalize ~unit var/dim
    _E8_CACHE[K] = cb
    return cb

_E8_TREE = {}
def _e8_tree(K):
    if K not in _E8_TREE:
        _E8_TREE[K] = cKDTree(e8_codebook(K))
    return _E8_TREE[K]

# ---- fast closed-form E8 lattice decode (Conway-Sloane), O(8)/vector, vectorized ----
def _d8_decode(Y):
    f = np.round(Y)
    s = f.sum(1).astype(np.int64)
    odd = (s % 2) != 0
    if odd.any():
        err = Y[odd] - f[odd]
        k = np.argmax(np.abs(err), axis=1)
        rows = np.nonzero(odd)[0]
        f[rows, k] += np.sign(err[np.arange(len(rows)), k])
    return f

def e8_decode(Y):
    """Nearest E8 point (even coordinate system: Z^8 even-sum  U  (Z+1/2)^8 even-sum)."""
    a = _d8_decode(Y)
    b = _d8_decode(Y - 0.5) + 0.5
    da = ((Y - a)**2).sum(1); db = ((Y - b)**2).sum(1)
    return np.where((db < da)[:, None], b, a)

def _entropy_bpw(idx_pts):
    # entropy of used lattice points / 8 dims = bits/weight (ideal entropy coding)
    key = np.ascontiguousarray((idx_pts * 2).astype(np.int64)).view([('', np.int64)] * 8).ravel()
    _, cnt = np.unique(key, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum()) / 8.0

_E8_SCALE = {}
def q_e8_fast(A, target_bpw=2.0, group=256, calib_key="g", fixed_step=None):
    """Fast E8 lattice quant via closed-form decode. Per-group RMS scale; one global step
    multiplier calibrated so index entropy ~= target_bpw (entropy-coded rate).
    fixed_step overrides calibration (for high-rate tensors entropy can't measure)."""
    flat = A.ravel().astype(np.float64); pad = (-flat.size) % group
    fp = np.concatenate([flat, np.zeros(pad)])
    g = fp.reshape(-1, group)
    rms = np.sqrt((g**2).mean(1, keepdims=True)) + 1e-12
    gn = (g / rms).reshape(-1, 8)
    if fixed_step is not None:
        rec = (e8_decode(gn / fixed_step) * fixed_step).reshape(-1, group) * rms
        return rec.ravel()[:A.size].reshape(A.shape)
    if calib_key not in _E8_SCALE:
        ns = min(600_000, gn.shape[0])
        sub = gn[np.random.RandomState(0).randint(0, gn.shape[0], ns)]   # large -> avoids entropy saturation
        lo, hi = 0.3, 10.0
        for _ in range(18):
            step = (lo + hi) / 2
            bpw = _entropy_bpw(e8_decode(sub / step))   # entropy of integer lattice indices
            if bpw > target_bpw: lo = step      # too fine -> larger step
            else: hi = step
        _E8_SCALE[calib_key] = (lo + hi) / 2
    step = _E8_SCALE[calib_key]
    rec = (e8_decode(gn / step) * step).reshape(-1, group) * rms
    return rec.ravel()[:A.size].reshape(A.shape)

def q_e8(A, bpw, group_scale=256):
    """Quantize 8D subvectors to a finite E8 codebook of size 2^(bpw*8). Per-group scale."""
    K = 1 << int(round(bpw * 8))
    cb = e8_codebook(K)
    tree = _e8_tree(K)
    flat = A.ravel().copy()
    pad = (-flat.size) % group_scale
    flat = np.concatenate([flat, np.zeros(pad)])
    g = flat.reshape(-1, group_scale)
    s = np.sqrt((g**2).mean(1, keepdims=True)) + 1e-12          # per-group scale
    gn = (g / s).reshape(-1, 8)
    _, idx = tree.query(gn, workers=-1)
    rec = (cb[idx].reshape(-1, group_scale) * s).ravel()[:A.size]
    return rec.reshape(A.shape)

def q_kmeans_vq(A, bpw, dim=4, fit_n=200_000):
    from scipy.cluster.vq import kmeans2, vq
    K = 1 << int(round(bpw * dim))
    flat = A.ravel(); pad = (-flat.size) % dim
    x = np.concatenate([flat, np.zeros(pad)]).reshape(-1, dim)
    sel = np.random.RandomState(0).randint(0, x.shape[0], min(fit_n, x.shape[0]))
    cb, _ = kmeans2(x[sel], K, minit='++', seed=0, missing='raise')
    _, idx = cKDTree(cb).query(x, workers=-1)
    return cb[idx].reshape(-1)[:A.size].reshape(A.shape)

def run(name, bpw=2.0):
    W, H = load_WH(name)
    sH = np.sqrt(H)[None, :]
    A = W * sH                                                  # importance-whitened
    print(f"\n=== {name}  shape {W.shape}  bpw target {bpw} ===", flush=True)
    print(f"  importance dynamic range H max/median = {H.max()/np.median(H):.1f}", flush=True)

    # IQ2 anchor (weighted)
    for tag, path in [("IQ2_XXS", r"models\8b\iq2xxs.gguf"), ("IQ2_XS", r"models\8b\iq2xs.gguf")]:
        Wq = deq_iq(path, name)[:W.shape[0], :W.shape[1]]
        print(f"  ANCHOR {tag:8s} weighted-NMSE = {wnmse(A, Wq*sH):.5f}", flush=True)

    Arot = rht_rows(A)
    print("  -- my methods (weighted-NMSE @ ~{:.1f} bpw) --".format(bpw), flush=True)
    print(f"  uniform                : {wnmse(A, q_uniform(A, bpw)):.5f}", flush=True)
    print(f"  kmeans-VQ dim4         : {wnmse(A, q_kmeans_vq(A, bpw, 4)):.5f}", flush=True)
    print(f"  E8 lattice             : {wnmse(A, q_e8(A, bpw)):.5f}", flush=True)
    print(f"  incoherence+uniform    : {wnmse(Arot, q_uniform(Arot, bpw)):.5f}", flush=True)
    print(f"  incoherence+E8 (QuIP#) : {wnmse(Arot, q_e8(Arot, bpw)):.5f}", flush=True)

if __name__ == "__main__":
    nm = sys.argv[1] if len(sys.argv) > 1 else "blk.16.ffn_gate.weight"
    bpw = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    run(nm, bpw)
