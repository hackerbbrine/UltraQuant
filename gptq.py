#!/usr/bin/env python
"""Sequential GPTQ-style quantization with CROSS-LAYER error propagation, on the torch model.

Quantize layer-by-layer on the PROGRESSIVELY-quantized model: capture each layer's real input
Hessian from the current (already-quantized-upstream) activations, quantize the layer, then push
the calibration data through the QUANTIZED layer so its error propagates to downstream layers.
This is new vs all prior per-tensor-independent quantization. Quantizer = incoherence + QTIP trellis,
with full-Hessian or diagonal whitening. Measures torch perplexity (second-half, matches llama.cpp).
"""
import numpy as np, torch, time, sys
import model as M
from qlab import rht_rows, irht_rows
from trellis import trellis_quant_fast

DEV = M.DEV

def whiten_mats(full_H):
    """Precompute H^.5 and H^-.5 once per distinct Hessian (eigendecomp)."""
    Hd = full_H + np.eye(full_H.shape[0]) * (1e-2 * np.trace(full_H)/full_H.shape[0])
    ev, U = np.linalg.eigh(Hd)
    ev = np.clip(ev, 1e-8, None)
    return (U * np.sqrt(ev)) @ U.T, (U * (1.0/np.sqrt(ev))) @ U.T

def quant_wt(W_gpu, Hdiag, R=2, L=10, whiten=None):
    """Quantize one weight (out,in). Diagonal whitening by default; whiten=(Hh,Hinv) for full-H
    (minimizes tr(E H E^T) = output error, capturing off-diagonal Hessian)."""
    W = W_gpu.detach().cpu().double().numpy()
    if whiten is None:
        sH = np.sqrt(np.maximum(Hdiag, Hdiag.max()*1e-3))[None, :]
        A = W * sH
        Ah = irht_rows(trellis_quant_fast(rht_rows(A), L=L, R=R, seg=64, scale_group=256, mem_gb=3.0), W.shape[1])
        Wh = Ah / sH
    else:
        Hh, Hinv = whiten
        A = W @ Hh
        Ah = irht_rows(trellis_quant_fast(rht_rows(A), L=L, R=R, seg=64, scale_group=256, mem_gb=3.0), W.shape[1])
        Wh = Ah @ Hinv
    return torch.tensor(Wh, dtype=torch.float16, device=DEV)

@torch.no_grad()
def layer_io(ld, h, pos):
    """Run a layer, returning output and the 4 linear-input activations for Hessian capture."""
    T = h.shape[0]
    x_attn = M.rmsnorm(h, ld["an"])
    q = (x_attn @ ld["wq"].T).view(T, M.NH, M.HD).transpose(0, 1)
    k = (x_attn @ ld["wk"].T).view(T, M.NKV, M.HD).transpose(0, 1)
    v = (x_attn @ ld["wv"].T).view(T, M.NKV, M.HD).transpose(0, 1)
    q = M.rope(q, pos); k = M.rope(k, pos)
    rep = M.NH // M.NKV; k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
    att = (q.float() @ k.float().transpose(-1, -2)) / np.sqrt(M.HD)
    att = (att + torch.full((T, T), float("-inf"), device=DEV).triu(1)).softmax(-1)
    x_wo = (att @ v.float()).to(torch.float16).transpose(0, 1).reshape(T, M.NE)
    h = h + x_wo @ ld["wo"].T
    x_ffn = M.rmsnorm(h, ld["fn"])
    x_down = (torch.nn.functional.silu((x_ffn @ ld["gate"].T).float()).to(torch.float16) * (x_ffn @ ld["up"].T))
    h = h + x_down @ ld["down"].T
    return h, x_attn, x_wo, x_ffn, x_down

@torch.no_grad()
def gptq_quantize(m, calib, full_H=False, vk_R=4, bulk_R=2):
    """Sequential quantize with error propagation. calib: list of token LongTensors."""
    Hs = [m["tok"][tc].to(DEV) for tc in calib]
    pos = [torch.arange(len(tc), device=DEV) for tc in calib]
    t0 = time.time()
    for li, ld in enumerate(m["layers"]):
        ldg = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in ld.items()}
        # accumulate Hessians (diag always; full optionally) from current activations
        acc = {n: None for n in ("attn", "wo", "ffn", "down")}
        for ci in range(len(Hs)):
            _, xa, xw, xf, xd = layer_io(ldg, Hs[ci], pos[ci])
            for n, x in (("attn", xa), ("wo", xw), ("ffn", xf), ("down", xd)):
                xf32 = x.float()
                g = (xf32.T @ xf32) if (full_H and n != "down") else (xf32 * xf32).sum(0)
                acc[n] = g if acc[n] is None else acc[n] + g
        ntok = sum(len(p) for p in pos)
        def Hd(n):
            a = acc[n]
            return (a.diagonal() if a.dim() == 2 else a).cpu().numpy() / ntok
        # full-H whitening for 4096-dim inputs (attn,wo,ffn); diagonal for down (14336, too big)
        W_ = {}
        if full_H:
            for n in ("attn", "wo", "ffn"):
                W_[n] = whiten_mats(acc[n].cpu().numpy() / ntok)
        # quantize: q/k/v share attn-input; wo; gate/up share ffn-input; down
        ldg["wq"] = quant_wt(ldg["wq"], Hd("attn"), R=bulk_R, whiten=W_.get("attn"))
        ldg["wk"] = quant_wt(ldg["wk"], Hd("attn"), R=vk_R,  whiten=W_.get("attn"))
        ldg["wv"] = quant_wt(ldg["wv"], Hd("attn"), R=vk_R,  whiten=W_.get("attn"))
        ldg["wo"] = quant_wt(ldg["wo"], Hd("wo"),   R=bulk_R, whiten=W_.get("wo"))
        ldg["gate"] = quant_wt(ldg["gate"], Hd("ffn"), R=bulk_R, whiten=W_.get("ffn"))
        ldg["up"]   = quant_wt(ldg["up"],   Hd("ffn"), R=bulk_R, whiten=W_.get("ffn"))
        ldg["down"] = quant_wt(ldg["down"], Hd("down"), R=bulk_R, whiten=None)
        # propagate quantized layer forward
        for ci in range(len(Hs)):
            Hs[ci], *_ = layer_io(ldg, Hs[ci], pos[ci])
        m["layers"][li] = {k: (v.detach().to("cpu", torch.float16) if torch.is_tensor(v) else v) for k, v in ldg.items()}
        del ldg, acc; torch.cuda.empty_cache()
        print(f"  layer {li} done ({time.time()-t0:.0f}s)", flush=True)
    return m

if __name__ == "__main__":
    full_H = "full" in sys.argv
    print(f"loading model; full_H={full_H}", flush=True)
    m = M.load_weights(unpermute=True)
    # calib = wiki.train tokens; eval = wiki.test tokens (disjoint)
    import pplcheck
    calib_toks = np.load("data/wiki_train_tokens.npy") if __import__("os").path.exists("data/wiki_train_tokens.npy") else None
    if calib_toks is None:
        from llama_cpp import Llama
        llm = Llama(model_path=M.GG, vocab_only=True, verbose=False)
        txt = open("data/wiki.train.raw", encoding="utf-8").read()[:400000]
        calib_toks = np.array(llm.tokenize(txt.encode("utf-8")), dtype=np.int64)
        np.save("data/wiki_train_tokens.npy", calib_toks)
    calib = [torch.tensor(calib_toks[i*512:(i+1)*512]) for i in range(12)]
    eval_chunks = pplcheck.get_tokens(40, 512)
    print(f"calib {len(calib)} chunks, eval {len(eval_chunks)} chunks", flush=True)
    m = gptq_quantize(m, calib, full_H=full_H)
    ph, pf, nt = M.perplexity(m, eval_chunks)
    print(f"GPTQ-seq quantized PPL: half={ph:.4f} full={pf:.4f}  (Q8=6.96, non-seq trellis=9.37)", flush=True)
