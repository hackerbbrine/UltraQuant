#!/usr/bin/env python
"""MAKE-OR-BREAK for KLT invention: can ONE shared eigenbasis serve all layers?
If yes, U amortizes to ~0.07 bpw (negligible) and KLT transform-coding is viable.
Capture ffn-input Hessian per layer; quantize each layer's gate with (a) its OWN eigenbasis and
(b) a SHARED eigenbasis (from layer 0, and from the mean Hessian). Compare output NMSE."""
import numpy as np, torch
import model as M, gptq
from klt import uniform_q_cols, waterfill, outnmse

LAYERS = list(range(0, 32, 4)); BPW = 2
m = M.load_weights(unpermute=True)
calib = [torch.tensor(np.load("data/wiki_train_tokens.npy")[i*512:(i+1)*512]) for i in range(24)]
Hs = [m["tok"][tc].to(M.DEV) for tc in calib]; pos=[torch.arange(len(tc),device=M.DEV) for tc in calib]

# one forward through all layers, accumulating ffn-input Hessian per layer + grabbing gate weights
Hcap = {}; Wcap = {}
with torch.no_grad():
    for li in range(32):
        ld = {k:(v.to(M.DEV) if torch.is_tensor(v) else v) for k,v in m["layers"][li].items()}
        if li in LAYERS:
            H=None; nt=0
            for ci in range(len(Hs)):
                outs = gptq.layer_io(ld, Hs[ci], pos[ci]); x = outs[3].float()  # x_ffn
                H = x.T@x if H is None else H+x.T@x; nt += x.shape[0]
            Hcap[li] = (H/nt).double().cpu().numpy()
            Wcap[li] = ld["gate"].detach().double().cpu().numpy()
        for ci in range(len(Hs)):
            Hs[ci], *_ = gptq.layer_io(ld, Hs[ci], pos[ci])
        del ld; torch.cuda.empty_cache()

inn = 4096
def Ufrom(H):
    Hd = H + np.eye(inn)*1e-3*np.trace(H)/inn
    lam,U = np.linalg.eigh(Hd); return U, Hd
U0,_ = Ufrom(Hcap[0])
Umean,_ = Ufrom(np.mean([Hcap[l] for l in LAYERS],0))

def klt_q(W, U, Herr):
    Wp = W@U; eigE = np.maximum(np.diag(U.T@Herr@U),0)*(Wp**2).mean(0)
    b = waterfill(eigE, BPW*inn); return (uniform_q_cols(Wp,b))@U.T

print(f"{'layer':6} {'own-U':>8} {'sharedU0':>9} {'sharedUmean':>11}", flush=True)
for li in LAYERS:
    W=Wcap[li]; _,Hd=Ufrom(Hcap[li])
    Uown,_=Ufrom(Hcap[li])
    e_own = outnmse(W, klt_q(W,Uown,Hd), Hd)
    e_u0  = outnmse(W, klt_q(W,U0,Hd),   Hd)
    e_um  = outnmse(W, klt_q(W,Umean,Hd),Hd)
    print(f"{li:6} {e_own:8.4f} {e_u0:9.4f} {e_um:11.4f}", flush=True)
# subspace overlap of top-512 eigvecs between layer 0 and others
def topspace(H,k=512):
    _,U=Ufrom(H); return U[:,-k:]
P0=topspace(Hcap[0])
print("\ntop-512 eigenspace overlap with layer 0 (1.0=identical):", flush=True)
for li in LAYERS:
    Pl=topspace(Hcap[li]); ov=np.linalg.norm(P0.T@Pl,'fro')**2/512
    print(f"  layer {li}: {ov:.3f}", flush=True)
