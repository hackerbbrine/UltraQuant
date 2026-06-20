#!/usr/bin/env python
"""RVQ / AQLM-style learned codebooks on incoherence-rotated, importance-whitened weights.

Extensions tested (my own, not in the papers as a combo):
  - Residual Vector Quant (stacked codebooks, EnCodec/SoundStream style) applied to LLM weights
  - ON TOP of QuIP#-style incoherence rotation
  - codebooks fine-tuned (GPU, Adam) to minimize the importance-weighted reconstruction error
Everything runs in the whitened+rotated space, so unweighted NMSE here == imatrix-weighted NMSE.
"""
import numpy as np, sys, time
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from qlab import load_WH, rht_rows, deq_iq

def rvq(A, dim, M, bits, iters_ft=0, fit_n=200_000):
    """M-stage residual VQ. rate = M*bits/dim bpw. Optional torch fine-tune of codebooks."""
    K = 1 << bits
    flat = A.ravel(); pad = (-flat.size) % dim
    X = np.concatenate([flat, np.zeros(pad)]).reshape(-1, dim).astype(np.float64)
    sel = np.random.RandomState(0).randint(0, X.shape[0], min(fit_n, X.shape[0]))
    cbs, idxs = [], []
    R = X.copy()
    for m in range(M):
        cb, _ = kmeans2(R[sel], K, minit='++', seed=m, missing='raise')
        _, idx = cKDTree(cb).query(R, workers=-1)
        cbs.append(cb); idxs.append(idx)
        R = R - cb[idx]
    rate = M * bits / dim
    if iters_ft > 0:
        cbs = _finetune(X, cbs, idxs, iters_ft)
    rec = sum(cbs[m][idxs[m]] for m in range(M))
    nmse = float(np.sum((X - rec)**2) / np.sum(X**2))
    return nmse, rate

def _finetune(X, cbs, idxs, iters):
    import torch
    d = 'cuda'
    Xt = torch.tensor(X, device=d)
    C = [torch.tensor(cb, device=d, requires_grad=True) for cb in cbs]
    I = [torch.tensor(idx, device=d) for idx in idxs]
    opt = torch.optim.Adam(C, lr=1e-3)
    for it in range(iters):
        opt.zero_grad()
        rec = sum(C[m][I[m]] for m in range(len(C)))
        loss = ((Xt - rec)**2).mean()
        loss.backward(); opt.step()
    return [c.detach().cpu().numpy() for c in C]

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "blk.16.ffn_gate.weight"
    W, H = load_WH(name)
    sH = np.sqrt(np.maximum(H, H.max()*1e-3))[None, :]
    A = W * sH
    Arot = rht_rows(A)
    # IQ2 anchor (weighted)
    Wq = deq_iq(r"models\8b\iq2xxs.gguf", name)[:W.shape[0], :W.shape[1]]
    anc = float(np.sum(((W-Wq)*sH)**2) / np.sum((W*sH)**2))
    print(f"=== {name}  IQ2_XXS anchor (weighted) = {anc:.5f} ===", flush=True)
    # RVQ configs all ~2.0 bpw, on incoherence-rotated A
    for dim, M, bits in [(8,2,8),(8,4,4),(4,1,8),(8,1,16)]:
        nm, rate = rvq(Arot, dim, M, bits)
        tag = f"RVQ dim{dim} M{M} {bits}b"
        print(f"  {tag:20s} {rate:.2f}bpw  NMSE={nm:.5f}", flush=True)
    # best config + fine-tune
    nm0, rate = rvq(Arot, 8, 2, 8)
    nmft, _ = rvq(Arot, 8, 2, 8, iters_ft=300)
    print(f"  RVQ dim8 M2 8b +finetune  {rate:.2f}bpw  {nm0:.5f} -> {nmft:.5f}", flush=True)

if __name__ == "__main__":
    main()
