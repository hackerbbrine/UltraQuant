#!/usr/bin/env python
"""Single-tensor diagnostic: does full-H whitening reduce or EXPLODE output error vs diagonal?
Measures real ||W_hat x - W x|| on captured activations for layer-0 ffn gate."""
import numpy as np, torch
import model as M
import gptq
import pplcheck

m = M.load_weights(unpermute=True)
calib = [torch.tensor(np.load("data/wiki_train_tokens.npy")[i*512:(i+1)*512]) for i in range(8)]
# capture layer-0 ffn input (x_ffn) and its Hessian
ld = {k: (v.to(M.DEV) if torch.is_tensor(v) else v) for k, v in m["layers"][0].items()}
Hs = [m["tok"][tc].to(M.DEV) for tc in calib]
pos = [torch.arange(len(tc), device=M.DEV) for tc in calib]
X = []; Hfull = None; ntok = 0
with torch.no_grad():
    for ci in range(len(Hs)):
        _, xa, xw, xf, xd = gptq.layer_io(ld, Hs[ci], pos[ci])
        X.append(xf.float()); xf32 = xf.float()
        Hfull = xf32.T @ xf32 if Hfull is None else Hfull + xf32.T @ xf32
        ntok += xf.shape[0]
Hfull = (Hfull/ntok).cpu().numpy()
Xc = torch.cat(X, 0)                                  # (Ntok, 4096)
W = ld["gate"]                                        # (14336, 4096)
out_ref = (Xc @ W.T.float())
def outerr(Wh):
    return float(((Xc @ Wh.T.float() - out_ref)**2).sum() / (out_ref**2).sum())

Hdiag = np.diag(Hfull)
Wd = gptq.quant_wt(W, Hdiag, R=2, L=10, whiten=None)
Wf = gptq.quant_wt(W, Hdiag, R=2, L=10, whiten=gptq.whiten_mats(Hfull))
print(f"diagonal-H : output NMSE = {outerr(Wd):.5f}")
print(f"full-H     : output NMSE = {outerr(Wf):.5f}")
print(f"H condition number (after damping): {np.linalg.cond(Hfull + np.eye(4096)*0.01*np.trace(Hfull)/4096):.1f}")
