#!/usr/bin/env python
"""Closed-form per-output-channel correction on the trellis-quantized model (9.37).
For each corrected matrix, scale each output channel j by a_j = <O_hat_j, O_j>/<O_hat_j,O_hat_j>
(O from original W, O_hat from quantized) -> minimizes per-channel output MSE. Output-aware, ~0 bits.
Propagated layer-by-layer through the quantized model. Measures torch perplexity (second-half)."""
import numpy as np, torch, sys
import model as M
import gptq, pplcheck

TRELLIS = r"models\8b_sim\q8_trellis.gguf"
_VALID = {"wq","wk","wv","wo","gate","up","down"}
CORR = (set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {"wo", "down"}) & _VALID
print(f"correcting matrices: {sorted(CORR)}", flush=True)

mq = M.load_weights(unpermute=True, path=TRELLIS)     # quantized (trellis) model
ro = __import__("gguf").GGUFReader(M.GG)               # original Q8
og = {t.name: t for t in ro.tensors}
def orig(name, nh=None):
    w = M._deq_q8(og[name])
    if nh: w = M._unpermute(w, nh)
    return torch.tensor(w, dtype=torch.float16, device=M.DEV)

calib = [torch.tensor(np.load("data/wiki_train_tokens.npy")[i*512:(i+1)*512]) for i in range(16)]
Hs = [mq["tok"][tc].to(M.DEV) for tc in calib]
pos = [torch.arange(len(tc), device=M.DEV) for tc in calib]

NAME = {"wq":("attn_q",M.NH),"wk":("attn_k",M.NKV),"wv":("attn_v",M.NKV),"wo":("attn_output",None),
        "gate":("ffn_gate",None),"up":("ffn_up",None),"down":("ffn_down",None)}
IN = {"wq":"x_attn","wk":"x_attn","wv":"x_attn","wo":"x_wo","gate":"x_ffn","up":"x_ffn","down":"x_down"}

@torch.no_grad()
def correct_layer(li, ldg):
    ins = {n: [] for n in ("x_attn","x_wo","x_ffn","x_down")}
    for ci in range(len(Hs)):
        _, xa, xw, xf, xd = gptq.layer_io(ldg, Hs[ci], pos[ci])
        ins["x_attn"].append(xa); ins["x_wo"].append(xw); ins["x_ffn"].append(xf); ins["x_down"].append(xd)
    cat = {k: torch.cat(v, 0).float() for k, v in ins.items()}
    for wk_ in CORR:
        nm, nh = NAME[wk_]; x = cat[IN[wk_]]
        Wq = ldg[wk_].float(); Wo = orig(f"blk.{li}.{nm}.weight", nh).float()
        Ohat = x @ Wq.T; O = x @ Wo.T                       # (T, out)
        a = (Ohat * O).sum(0) / ((Ohat * Ohat).sum(0) + 1e-8)
        a = a.clamp(0.5, 2.0)                                # guard pathological channels
        ldg[wk_] = (ldg[wk_].float() * a[:, None]).to(torch.float16)

for li, ld in enumerate(mq["layers"]):
    ldg = {k: (v.to(M.DEV) if torch.is_tensor(v) else v) for k, v in ld.items()}
    correct_layer(li, ldg)
    for ci in range(len(Hs)):
        Hs[ci], *_ = gptq.layer_io(ldg, Hs[ci], pos[ci])
    mq["layers"][li] = {k: (v.to("cpu", torch.float16) if torch.is_tensor(v) else v) for k, v in ldg.items()}
    del ldg; torch.cuda.empty_cache()

eval_chunks = pplcheck.get_tokens(40, 512)
ph, pf, nt = M.perplexity(mq, eval_chunks)
print(f"CORRECTED ({sorted(CORR)}) PPL: half={ph:.4f} full={pf:.4f}  (trellis baseline 9.37, Q8 6.96)", flush=True)
