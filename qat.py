#!/usr/bin/env python
"""QAT: AQLM-style codebook+scale fine-tuning with FIXED trellis assignments.
Step 1 (this file): expose trellis states, build a DIFFERENTIABLE torch reconstruction, and VALIDATE
it matches the non-differentiable trellis output. Then per-layer optimize (codebook, scales) to
minimize true output error.  cw is per-tensor learnable (4096 vals ~ 0.001 bpw, negligible).
"""
import numpy as np, torch
from trellis import codebook, pred_table
DEV = "cuda"

def trellis_encode(A, L=12, R=2, seg=64, scale_group=256):
    """Replicates trellis_quant_fast but RETURNS (states, sc_g, cw, meta) instead of reconstruction."""
    cw = torch.tensor(codebook(L), device=DEV); pred = torch.tensor(pred_table(L,R),device=DEV,dtype=torch.long)
    S=1<<L; nrow,ncol=A.shape; flat=A.astype(np.float32).ravel(); Ntot=flat.size
    pad=(-Ntot)%scale_group; fpad=np.concatenate([flat,np.zeros(pad,np.float32)])
    gsc=fpad.reshape(-1,scale_group); sc_g=np.sqrt((gsc**2).mean(1,keepdims=True))+1e-12
    xnorm=(gsc/sc_g).ravel(); fp=xnorm.reshape(-1,seg); Nseg=fp.shape[0]
    chunk=max(1,int(3.0e9/(seg*S))); nT=1<<R; Slow=S>>R
    states=np.empty((Nseg,seg),np.int32); ar_cache={}
    for c0 in range(0,Nseg,chunk):
        xt=torch.tensor(fp[c0:c0+chunk],device=DEV); b=xt.shape[0]
        cost=(xt[:,0:1]-cw[None,:])**2; bpq=torch.empty((b,seg,Slow),dtype=torch.uint8,device=DEV)
        for t in range(1,seg):
            m,k=cost.view(b,nT,Slow).min(dim=1); bpq[:,t,:]=k.to(torch.uint8)
            cost=(xt[:,t:t+1]-cw[None,:])**2 + m.repeat_interleave(nT,dim=1)
        st=torch.empty((b,seg),dtype=torch.long,device=DEV); st[:,seg-1]=cost.argmin(1)
        if b not in ar_cache: ar_cache[b]=torch.arange(b,device=DEV)
        ar=ar_cache[b]
        for t in range(seg-1,0,-1):
            q=st[:,t]>>R; st[:,t-1]=bpq[ar,t,q].long()*Slow+q
        states[c0:c0+chunk]=st.cpu().numpy(); del bpq,cost,st
    meta=dict(nrow=nrow,ncol=ncol,Ntot=Ntot,seg=seg,scale_group=scale_group,pad=pad,Nseg=Nseg)
    return states, sc_g.astype(np.float32), cw.cpu().numpy(), meta

def fwht_torch(x):
    """Differentiable fast Walsh-Hadamard along last axis (n=2^k), normalized by 1/sqrt(n)."""
    n=x.shape[-1]; orig=x.shape; h=1
    while h<n:
        x=x.reshape(-1, n//(2*h), 2, h); a=x[:,:,0,:]; b=x[:,:,1,:]
        x=torch.cat([a+b,a-b],dim=2).reshape(*orig[:-1],n); h*=2
    return x/np.sqrt(n)

def irht_torch(Xrot, in0, seed=0):
    """Differentiable inverse of qlab.rht_rows: Xrot (out, n=2^k>=in0) -> (out, in0).
    qlab.irht = fwht_torch(X) * d  (derived: raw_FWHT(X*sqrt n)/n * d = fwht_torch(X)*d)."""
    out,n=Xrot.shape
    x=fwht_torch(Xrot)
    d=torch.tensor(np.random.RandomState(seed).choice([-1.0,1.0],size=n),device=DEV,dtype=Xrot.dtype)
    return (x*d)[:, :in0]

def recon_torch(states, cw, sc_g, meta, sH, in0):
    """Differentiable Ŵ from fixed states + learnable (cw, sc_g). Returns (out,in)."""
    st=torch.tensor(states,device=DEV,dtype=torch.long)
    rec=cw[st]                                          # (Nseg, seg) normalized, diff in cw
    rec=rec.reshape(-1)[:meta['Nseg']*meta['seg']]
    rec=(rec.reshape(-1, meta['scale_group'])*sc_g).reshape(-1)[:meta['Ntot']]
    A_rot=rec.reshape(meta['nrow'], meta['ncol'])       # = rotated-whitened reconstruction (padded in-dim? no)
    A_hat=irht_torch(A_rot, in0)
    return A_hat / sH

def qat_tensor(W, Hdiag, X, steps=30, lr=2e-3, tune_scales=True, Xval=None):
    """Fine-tune per-tensor codebook (+optionally scales), fixed states, to minimize output error.
    Returns (Wh, train_l0, train_l1, val_l0, val_l1) where val_* use held-out Xval if given."""
    from qlab import rht_rows
    sH = np.sqrt(np.maximum(Hdiag, Hdiag.max()*1e-3))[None, :]
    A = rht_rows(W*sH).astype(np.float32); in0 = W.shape[1]
    states, sc_g, cwn, meta = trellis_encode(A)
    cw = torch.tensor(cwn, device=DEV, requires_grad=True)
    scg = torch.tensor(sc_g, device=DEV, requires_grad=tune_scales)
    sH_t = torch.tensor(sH, device=DEV, dtype=torch.float32)
    Wt = torch.tensor(W, device=DEV, dtype=torch.float32)
    target = (X @ Wt.T)
    tval = (Xval @ Wt.T) if Xval is not None else None
    params = [cw] + ([scg] if tune_scales else [])
    opt = torch.optim.Adam(params, lr=lr)
    def losses():
        Wh = recon_torch(states, cw, scg, meta, sH_t, in0)
        tr = ((X @ Wh.T - target)**2).mean().item()
        vl = ((Xval @ Wh.T - tval)**2).mean().item() if Xval is not None else float('nan')
        return tr, vl
    l0, v0 = losses()
    for _ in range(steps):
        opt.zero_grad()
        Wh = recon_torch(states, cw, scg, meta, sH_t, in0)
        loss = ((X @ Wh.T - target)**2).mean()
        loss.backward(); opt.step()
    l1, v1 = losses()
    Wh = recon_torch(states, cw, scg, meta, sH_t, in0).detach()
    return Wh.cpu().numpy().astype(np.float32), l0, l1, v0, v1

if __name__ == "__main__":
    import sys, model as M, gptq, pplcheck
    from qlab import load_WH
    from buildq import enc_q8
    from gguf import GGUFReader
    import shutil
    TYPES = ("attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down")
    KEY = {"attn_q":("wq",0),"attn_k":("wk",0),"attn_v":("wv",0),"attn_output":("wo",1),
           "ffn_gate":("gate",2),"ffn_up":("up",2),"ffn_down":("down",3)}
    BASE = r"models\8b_sim\q8_trellis.gguf"; OUT = r"models\8b_sim\q8_trellis_qat.gguf"
    NCAL = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    print(f"QAT: ncal={NCAL} steps={STEPS}", flush=True)
    shutil.copyfile(BASE, OUT)
    m = M.load_weights(unpermute=True)
    calib = [torch.tensor(np.load("data/wiki_train_tokens.npy")[i*512:(i+1)*512]) for i in range(NCAL)]
    Hs = [m["tok"][tc].to(DEV) for tc in calib]; pos=[torch.arange(len(tc),device=DEV) for tc in calib]
    r = GGUFReader(BASE); tt = {t.name: t for t in r.tensors}
    fh = open(OUT, "r+b"); import time; t0=time.time()
    for li in range(32):
        ld = {k:(v.to(DEV) if torch.is_tensor(v) else v) for k,v in m["layers"][li].items()}
        # capture clean inputs (4 distinct) for this layer
        X = [[] for _ in range(4)]
        with torch.no_grad():
            for ci in range(len(Hs)):
                o = gptq.layer_io(ld, Hs[ci], pos[ci])
                for j in range(4): X[j].append(o[1+j].float())
        Xc = [torch.cat(x,0) for x in X]
        for typ in TYPES:
            key, inp = KEY[typ]
            W, Hd = load_WH(f"blk.{li}.{typ}.weight")
            Wh, l0, l1, _, _ = qat_tensor(W, Hd, Xc[inp], steps=STEPS, lr=3e-4, tune_scales=True)
            enc = enc_q8(Wh); t = tt[f"blk.{li}.{typ}.weight"]
            assert enc.nbytes == t.n_bytes
            fh.seek(int(t.data_offset)); fh.write(enc.tobytes())
            m["layers"][li][key] = torch.tensor(Wh, dtype=torch.float16)   # use tuned for propagation
        # re-propagate with tuned weights
        ldt = {k:(v.to(DEV) if torch.is_tensor(v) else v) for k,v in m["layers"][li].items()}
        with torch.no_grad():
            for ci in range(len(Hs)): Hs[ci],*_ = gptq.layer_io(ldt, Hs[ci], pos[ci])
        del ld, ldt; torch.cuda.empty_cache()
        if li % 4 == 0: print(f"  layer {li} done ({time.time()-t0:.0f}s)  last-tensor loss {l0:.4e}->{l1:.4e}", flush=True)
    fh.close()
    print(f"[done] QAT model -> {OUT} ({time.time()-t0:.0f}s)", flush=True)
