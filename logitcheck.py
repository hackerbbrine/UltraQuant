#!/usr/bin/env python
"""Definitive forward-correctness test: compare torch logits to llama.cpp logits on the same
tokens. If they agree (high top-1 match, low KL), the forward is correct and any perplexity gap
is methodology. If not, the forward has a bug."""
import numpy as np, torch, sys
import model as M

N = 48
toks = np.load("data/wiki_tokens.npy")[:N].tolist()

# llama.cpp reference logits
from llama_cpp import Llama
llm = Llama(model_path=M.GG, n_ctx=128, logits_all=True, verbose=False)
llm.reset(); llm.eval(toks)
ref = np.array(llm.scores[:N])                      # (N, vocab)

unperm = "noperm" not in sys.argv
m = M.load_weights(unpermute=unperm)
tc = [torch.tensor(toks, dtype=torch.long)]
H = m["tok"][tc[0]].to(M.DEV)
pos = torch.arange(N, device=M.DEV)
with torch.no_grad():
    for ld in m["layers"]:
        ldg = {k: (v.to(M.DEV) if torch.is_tensor(v) else v) for k, v in ld.items()}
        H = M.layer_forward(ldg, H, pos)
    h = M.rmsnorm(H, m["onorm"].to(M.DEV))
    mine = (h @ m["head"].to(M.DEV).T).float().cpu().numpy()   # (N, vocab)

print(f"unpermute={unperm}")
agree = 0; kls = []
for i in range(2, N):
    a, b = ref[i], mine[i]
    if np.argmax(a) == np.argmax(b): agree += 1
    pa = np.exp(a - a.max()); pa /= pa.sum()
    pb = np.exp(b - b.max()); pb /= pb.sum()
    kls.append(float(np.sum(pa * np.log((pa+1e-9)/(pb+1e-9)))))
print(f"top-1 argmax agreement: {agree}/{N-2}   mean KL(ref||mine): {np.mean(kls):.4f}")
print(f"logit corr (pos 10): {np.corrcoef(ref[10], mine[10])[0,1]:.4f}")
