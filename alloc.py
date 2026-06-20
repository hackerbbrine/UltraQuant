#!/usr/bin/env python
"""Real bit-allocation search vs actual perplexity (not proxy NMSE). Quantize each tensor type at a
chosen bit-rate (trellis), non-sequential (clean-model diagonal Hessian). Test several allocations at
matched ~budget; report ppl AND measured avg bpw so wins must be on the rate-distortion frontier."""
import numpy as np, torch, sys, copy
import model as M, gptq, pplcheck

# param fraction per type (Llama-8B linear)
FRAC = {"wq":0.077,"wk":0.019,"wv":0.019,"wo":0.077,"gate":0.269,"up":0.269,"down":0.269}
IN   = {"wq":0,"wk":0,"wv":0,"wo":1,"gate":2,"up":2,"down":3}   # which captured input
NAME = {"wq":"attn_q","wk":"attn_k","wv":"attn_v","wo":"attn_output","gate":"ffn_gate","up":"ffn_up","down":"ffn_down"}

ALLOCS = {
 "A_current(V/K@4)":      {"wq":2,"wk":4,"wv":4,"wo":2,"gate":2,"up":2,"down":2},
 "B_+down@3,gate/up@1":   {"wq":2,"wk":4,"wv":4,"wo":2,"gate":1,"up":1,"down":3},
 "C_+down@3,gate/up@2":   {"wq":2,"wk":4,"wv":4,"wo":2,"gate":2,"up":2,"down":3},
 "D_down@3,V/K@3,g/u@1.x":{"wq":2,"wk":3,"wv":3,"wo":2,"gate":1,"up":2,"down":3},
}

def avg_bpw(a): return sum(FRAC[t]*a[t] for t in FRAC)/sum(FRAC.values())

print("loading + capturing clean diagonal Hessians (one forward)...", flush=True)
m0 = M.load_weights(unpermute=True)
calib = [torch.tensor(np.load("data/wiki_train_tokens.npy")[i*512:(i+1)*512]) for i in range(16)]
Hs = [m0["tok"][tc].to(M.DEV) for tc in calib]; pos=[torch.arange(len(tc),device=M.DEV) for tc in calib]
diagH = {}   # (layer, inputidx) -> diag vector
with torch.no_grad():
    for li in range(32):
        ld = {k:(v.to(M.DEV) if torch.is_tensor(v) else v) for k,v in m0["layers"][li].items()}
        outs = [None]*4; acc=[None]*4
        for ci in range(len(Hs)):
            o = gptq.layer_io(ld, Hs[ci], pos[ci]);
            for j in range(4):
                xf=o[1+j].float(); g=(xf*xf).sum(0); acc[j]=g if acc[j] is None else acc[j]+g
        nt=sum(len(p) for p in pos)
        for j in range(4): diagH[(li,j)] = (acc[j]/nt).cpu().numpy()
        for ci in range(len(Hs)): Hs[ci],*_ = gptq.layer_io(ld, Hs[ci], pos[ci])
        del ld; torch.cuda.empty_cache()
del Hs; torch.cuda.empty_cache()
eval_chunks = pplcheck.get_tokens(40, 512)

def run_alloc(alloc):
    m = M.load_weights(unpermute=True)
    for li in range(32):
        for t in FRAC:
            H = diagH[(li, IN[t])]
            m["layers"][li][t] = gptq.quant_wt(m["layers"][li][t].to(M.DEV), H, R=alloc[t], L=8).to("cpu")
    ph, pf, nt = M.perplexity(m, eval_chunks)
    del m; torch.cuda.empty_cache()
    return ph

for name, a in ALLOCS.items():
    ph = run_alloc(a)
    print(f"  {name:24} bpw={avg_bpw(a):.3f}  ppl={ph:.4f}", flush=True)
