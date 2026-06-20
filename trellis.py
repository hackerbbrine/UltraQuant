#!/usr/bin/env python
"""QTIP-style bitshift trellis-coded quantization, GPU-batched Viterbi.

Each weight emits R bits; the state = last L bits of the bitstream (sliding window), so
weight N's code influences weight N+1 (memory). Codeword(state) = hashed Gaussian (implicit
codebook, no storage). Viterbi finds the globally min-distortion bit path. FIXED rate = R bpw
(+ tiny per-group scale) -> directly comparable to IQ2, no entropy coding needed.

Reference: Tseng et al., "QTIP: Quantization with Trellises and Incoherence Processing" (2024).
"""
import numpy as np, torch
from scipy.stats import norm

def codebook(L, seed=0):
    s = np.arange(1 << L, dtype=np.uint64)
    h = (s * np.uint64(2654435761) + np.uint64(0x9E3779B97F4A7C15))
    h ^= (h >> np.uint64(29)); h *= np.uint64(0xBF58476D1CE4E5B9); h ^= (h >> np.uint64(32))
    u = ((h & np.uint64(0xFFFFFFFF)).astype(np.float64) + 0.5) / 2**32
    return norm.ppf(u).astype(np.float32)            # (2^L,) ~ N(0,1)

def pred_table(L, R):
    S = 1 << L; mask = S - 1
    s = np.arange(S)[:, None]
    k = np.arange(1 << R)[None, :]
    return ((s >> R) | (k << (L - R))) & mask        # (2^L, 2^R) predecessor states

def trellis_quant(A, L=10, R=2, group=256, chunk_rows=512, device="cuda"):
    """Quantize matrix A (out,in) row-wise. Returns reconstruction, fixed rate ~R bpw."""
    cw = torch.tensor(codebook(L), device=device)            # (S,)
    pred = torch.tensor(pred_table(L, R), device=device, dtype=torch.long)  # (S,2^R)
    S = 1 << L
    out = np.empty_like(A, dtype=np.float32)
    nrow, T = A.shape
    for r0 in range(0, nrow, chunk_rows):
        blk = A[r0:r0+chunk_rows].astype(np.float32)
        b, _ = blk.shape
        # per-group RMS normalize along the row
        Tp = ((T + group - 1) // group) * group
        pad = np.zeros((b, Tp), np.float32); pad[:, :T] = blk
        g = pad.reshape(b, Tp // group, group)
        sc = np.sqrt((g**2).mean(2, keepdims=True)) + 1e-12
        xn = (g / sc).reshape(b, Tp)
        xt = torch.tensor(xn, device=device)
        cost = (xt[:, 0:1] - cw[None, :])**2                 # (b,S)
        bp = torch.empty((b, Tp, S), dtype=torch.uint8, device=device)
        for t in range(1, Tp):
            cand = cost[:, pred]                             # (b,S,2^R)
            mn, k = cand.min(dim=2)                          # (b,S)
            cost = (xt[:, t:t+1] - cw[None, :])**2 + mn
            bp[:, t, :] = k.to(torch.uint8)
        states = torch.empty((b, Tp), dtype=torch.long, device=device)
        states[:, Tp-1] = cost.argmin(1)
        ar = torch.arange(b, device=device)
        for t in range(Tp-1, 0, -1):
            k = bp[ar, t, states[:, t]].long()
            states[:, t-1] = pred[states[:, t], k]
        rec = (cw[states].reshape(b, Tp//group, group) * torch.tensor(sc, device=device)).reshape(b, Tp)
        out[r0:r0+b] = rec[:, :T].cpu().numpy()
        del bp, cost, states
    return out

def trellis_quant_fast(A, L=12, R=2, seg=64, scale_group=256, mem_gb=3.0, device="cuda"):
    """Segmented + max-batched Viterbi. Viterbi resets every `seg` weights (small seg = fewer GPU
    launches = fast). Scale is per `scale_group` weights (decoupled from seg) -> fair ~0.0625 bpw
    overhead at group=256. Fixed rate = R bpw + scale. seg>>L so memory loss at resets is tiny."""
    cw = torch.tensor(codebook(L), device=device)
    pred = torch.tensor(pred_table(L, R), device=device, dtype=torch.long)
    S = 1 << L
    nrow, ncol = A.shape
    flat = A.astype(np.float32).ravel()
    Ntot = flat.size
    pad = (-Ntot) % scale_group
    fpad = np.concatenate([flat, np.zeros(pad, np.float32)])
    gsc = fpad.reshape(-1, scale_group)
    sc_g = np.sqrt((gsc**2).mean(1, keepdims=True)) + 1e-12          # per scale_group
    xnorm = (gsc / sc_g).ravel()                                     # normalized flat
    fp = xnorm.reshape(-1, seg)                                      # Viterbi segments
    sc = None
    xn = fp
    Nseg = xn.shape[0]
    chunk = max(1, int(mem_gb * 1e9 / (seg * S)))          # segments per chunk (backptr bound)
    rec = np.empty_like(xn)
    ar_cache = {}
    for c0 in range(0, Nseg, chunk):
        xt = torch.tensor(xn[c0:c0+chunk], device=device)  # (b, seg)
        b = xt.shape[0]
        nT = 1 << R; Slow = S >> R
        cw2 = cw[None, :]
        cost = (xt[:, 0:1] - cw2)**2                        # (b,S)
        bpq = torch.empty((b, seg, Slow), dtype=torch.uint8, device=device)  # argmin-k per q
        for t in range(1, seg):
            m, k = cost.view(b, nT, Slow).min(dim=1)        # (b,Slow): min over predecessor high bits
            bpq[:, t, :] = k.to(torch.uint8)
            prev = m.repeat_interleave(nT, dim=1)           # (b,S): out[s]=m[s>>R]
            cost = (xt[:, t:t+1] - cw2)**2 + prev
        st = torch.empty((b, seg), dtype=torch.long, device=device)
        st[:, seg-1] = cost.argmin(1)
        if b not in ar_cache: ar_cache[b] = torch.arange(b, device=device)
        ar = ar_cache[b]
        for t in range(seg-1, 0, -1):
            q = st[:, t] >> R
            k = bpq[ar, t, q].long()
            st[:, t-1] = k * Slow + q                        # predecessor state = k*Slow + q
        rec[c0:c0+chunk] = cw[st].cpu().numpy()
        del bpq, cost, st
    rec_flat = rec.ravel()                                  # normalized
    rec_flat = (rec_flat.reshape(-1, scale_group) * sc_g).ravel()   # un-normalize per scale_group
    return rec_flat[:Ntot].reshape(nrow, ncol).astype(np.float32)

def _selftest():
    rng = np.random.RandomState(0)
    X = rng.randn(2000, 1024).astype(np.float32)             # unit Gaussian
    bound = 2.0**(-2*2)                                       # D(R=2) = 2^-4 = 0.0625
    for L in (8, 10, 12):
        Xh = trellis_quant(X, L=L, R=2, group=256, chunk_rows=2000)
        nmse = float(np.sum((X-Xh)**2)/np.sum(X**2))
        print(f"  L={L} R=2: NMSE={nmse:.5f}  (Gaussian R-D bound={bound:.4f}, memoryless~0.118)")

if __name__ == "__main__":
    print("self-test on unit Gaussian (trellis should beat memoryless ~0.118, approach 0.0625):")
    _selftest()
