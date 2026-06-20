#!/usr/bin/env python
"""INVENTION TEST: Hessian-eigenbasis (KLT) transform coding + rate-distortion water-filling.

Not incoherence/lattice/trellis/GPTQ. Idea: rotate W into the eigenbasis of the input Hessian
H=XX^T (the KLT, optimal decorrelator), then allocate VARIABLE bits per eigendirection by
water-filling against the OUTPUT-error rate-distortion objective (output err = sum_j lambda_j *
||col_j err||^2). Directions inputs never visit (tiny lambda) get 0 bits. Stable (orthogonal U,
no H^-.5). Compares output-NMSE vs diagonal-uniform baseline at equal avg bits.
"""
import numpy as np, torch, sys
import model as M, gptq

def capture_H(layer=0, which="ffn", ncal=24):
    m = M.load_weights(unpermute=True)
    calib = [torch.tensor(np.load("data/wiki_train_tokens.npy")[i*512:(i+1)*512]) for i in range(ncal)]
    Hs = [m["tok"][tc].to(M.DEV) for tc in calib]; pos=[torch.arange(len(tc),device=M.DEV) for tc in calib]
    ld = {k:(v.to(M.DEV) if torch.is_tensor(v) else v) for k,v in m["layers"][layer].items()}
    key = {"attn":0,"wo":1,"ffn":2,"down":3}[which]
    H=None; nt=0
    with torch.no_grad():
        for ci in range(len(Hs)):
            outs = gptq.layer_io(ld, Hs[ci], pos[ci]); x = outs[1+key].float()
            H = x.T@x if H is None else H + x.T@x; nt += x.shape[0]
    Wname = {"attn":"attn_q","wo":"attn_output","ffn":"ffn_gate","down":"ffn_down"}[which]
    W = ld[{"attn":"wq","wo":"wo","ffn":"gate","down":"down"}[which]].detach().double().cpu().numpy()
    return W, (H/nt).double().cpu().numpy()

def uniform_q_cols(Wc, bits_per_col):
    """Quantize each COLUMN to its own integer bit count (per-col absmax). 0 bits -> drop."""
    out = np.zeros_like(Wc)
    for j, b in enumerate(bits_per_col):
        if b <= 0:
            continue
        col = Wc[:, j]; amax = np.abs(col).max() + 1e-12
        lvl = (1 << (b-1)) - 1 if b > 1 else 1
        out[:, j] = np.round(col/amax*lvl)/lvl*amax
    return out

def waterfill(energy, total_bits, bmax=8):
    """Allocate integer bits per direction maximizing -sum energy_j 2^-2b_j, sum b_j = total_bits."""
    n = len(energy); b = np.zeros(n, int)
    # greedy: repeatedly give 1 bit to the direction with largest marginal distortion reduction
    import heapq
    def gain(j, bb): return energy[j]*(2.0**(-2*bb) - 2.0**(-2*(bb+1)))
    h = [(-gain(j,0), j) for j in range(n)]; heapq.heapify(h)
    for _ in range(int(total_bits)):
        g, j = heapq.heappop(h)
        b[j] += 1
        if b[j] < bmax: heapq.heappush(h, (-gain(j,b[j]), j))
    return b

def outnmse(W, Wh, H):
    E = W - Wh
    return float(np.trace(E @ H @ E.T) / np.trace(W @ H @ W.T))

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "ffn"
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    bpw = 2
    W, H = capture_H(layer, which)
    out, inn = W.shape
    print(f"\n=== layer {layer} {which}  W{W.shape}  bpw={bpw} ===", flush=True)
    Hd = H + np.eye(inn)*1e-3*np.trace(H)/inn
    # baseline: diagonal allocation = per-channel uniform with bits by H-diagonal (water-fill on diag)
    diagE = np.diag(Hd) * (W**2).mean(0)
    bdiag = waterfill(diagE, bpw*inn)
    Wd = uniform_q_cols(W, bdiag)
    print(f"  diagonal-basis water-fill : outNMSE = {outnmse(W, Wd, Hd):.5f}  (bits used {bdiag.sum()}/{bpw*inn})", flush=True)
    # KLT: eigenbasis transform + water-fill
    lam, U = np.linalg.eigh(Hd)                    # ascending
    Wp = W @ U                                     # rotate to eigenbasis (out, in)
    eigE = lam * (Wp**2).mean(0)                   # per-direction output energy
    bklt = waterfill(eigE, bpw*inn)
    Wp_q = uniform_q_cols(Wp, bklt)
    Wk = Wp_q @ U.T                                # rotate back
    print(f"  KLT eigenbasis water-fill : outNMSE = {outnmse(W, Wk, Hd):.5f}  (bits used {bklt.sum()}/{bpw*inn}, "
          f"{(bklt==0).sum()} dirs dropped)", flush=True)
    print(f"  eigenvalue spread: lam_max/lam_med = {lam.max()/np.median(lam):.1f}; "
          f"top-10% dirs hold {100*np.sort(eigE)[::-1][:inn//10].sum()/eigE.sum():.0f}% of output energy", flush=True)

if __name__ == "__main__":
    main()
