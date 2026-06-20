#!/usr/bin/env python
"""Minimal torch Llama-3.1-8B that loads dequantized GGUF weights. Enables OUTPUT-error-aware
quantization: capture per-layer activations -> real Hessians (X X^T, the full off-diagonal that
the diagonal imatrix lacks), GPTQ/LDLQ error feedback, end-to-end tuning, and torch perplexity.

Processes the whole dataset layer-by-layer (GPTQ-style): move layer L to GPU, push ALL sequences
through it (optionally capturing the input Hessian / applying a quantizer), advance, free.
"""
import numpy as np, torch
from gguf import GGUFReader

GG = r"models\Llama-3.1-8B-Q8_0.gguf"
NL, NE, NF, NH, NKV, HD = 32, 4096, 14336, 32, 8, 128
ROPE_THETA, EPS, VOCAB = 500000.0, 1e-5, 128256
DEV = "cuda"

def _deq_q8(t):
    rows, brow = t.data.shape; nb = brow // 34
    d = t.data.reshape(rows, nb, 34)
    sc = d[:, :, :2].copy().view(np.float16).astype(np.float32)
    qs = d[:, :, 2:].view(np.int8).astype(np.float32)
    return (sc * qs).reshape(rows, nb * 32)

def _unpermute(w, nh):
    # invert llama.cpp's GGUF q/k permutation -> HF rotate_half layout
    return w.reshape(nh, w.shape[0]//nh//2, 2, w.shape[1]).swapaxes(1, 2).reshape(w.shape)

def load_weights(unpermute=True, path=None):
    r = GGUFReader(path or GG)
    g = {t.name: t for t in r.tensors}
    def W(n): return torch.tensor(_deq_q8(g[n]), dtype=torch.float16)
    m = {"tok": W("token_embd.weight"),
         "onorm": torch.tensor(np.asarray(g["output_norm.weight"].data), dtype=torch.float32),
         "head": W("output.weight") if "output.weight" in g else W("token_embd.weight")}
    layers = []
    for i in range(NL):
        p = f"blk.{i}."
        wq = _deq_q8(g[p+"attn_q.weight"]); wk = _deq_q8(g[p+"attn_k.weight"])
        if unpermute:
            wq = _unpermute(wq, NH); wk = _unpermute(wk, NKV)
        layers.append({
            "an": torch.tensor(np.asarray(g[p+"attn_norm.weight"].data), dtype=torch.float32),
            "fn": torch.tensor(np.asarray(g[p+"ffn_norm.weight"].data), dtype=torch.float32),
            "wq": torch.tensor(wq, dtype=torch.float16), "wk": torch.tensor(wk, dtype=torch.float16),
            "wv": W(p+"attn_v.weight"), "wo": W(p+"attn_output.weight"),
            "gate": W(p+"ffn_gate.weight"), "up": W(p+"ffn_up.weight"), "down": W(p+"ffn_down.weight")})
    m["layers"] = layers
    return m

def rmsnorm(x, w, eps=EPS):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * w).to(torch.float16)

def rope(x, pos):  # x: (..., T, HD) HF rotate_half
    half = HD // 2
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, half, device=x.device).float() / half))
    ang = pos[:, None].float() * freqs[None, :]              # (T, half)
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None]        # (1,T,HD)
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None]
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], -1)
    return (x.float()*cos + rot.float()*sin).to(x.dtype)

@torch.no_grad()
def layer_forward(ld, h, pos):
    T = h.shape[0]
    x = rmsnorm(h, ld["an"].to(DEV))
    q = (x @ ld["wq"].T).view(T, NH, HD).transpose(0, 1)     # (NH,T,HD)
    k = (x @ ld["wk"].T).view(T, NKV, HD).transpose(0, 1)
    v = (x @ ld["wv"].T).view(T, NKV, HD).transpose(0, 1)
    q = rope(q, pos); k = rope(k, pos)
    rep = NH // NKV
    k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
    att = (q.float() @ k.float().transpose(-1, -2)) / np.sqrt(HD)
    mask = torch.full((T, T), float("-inf"), device=DEV).triu(1)
    att = (att + mask).softmax(-1)
    o = (att @ v.float()).to(torch.float16).transpose(0, 1).reshape(T, NE)
    h = h + o @ ld["wo"].T
    x = rmsnorm(h, ld["fn"].to(DEV))
    h = h + (torch.nn.functional.silu((x @ ld["gate"].T).float()).to(torch.float16) * (x @ ld["up"].T)) @ ld["down"].T
    return h

@torch.no_grad()
def perplexity(m, token_chunks, quant=None):
    """token_chunks: list of 1D LongTensors (each <= ctx). Layer-by-layer over all chunks."""
    H = [m["tok"][tc].to(DEV) for tc in token_chunks]        # embeds per chunk on GPU
    poss = [torch.arange(len(tc), device=DEV) for tc in token_chunks]
    for li, ld in enumerate(m["layers"]):
        ldg = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in ld.items()}
        if quant is not None:
            quant(ldg, li)
        for ci in range(len(H)):
            H[ci] = layer_forward(ldg, H[ci], poss[ci])
        del ldg; torch.cuda.empty_cache()
    onorm = m["onorm"].to(DEV); head = m["head"].to(DEV)
    nll_full, nt_full, nll_half, nt_half = 0.0, 0, 0.0, 0
    for ci, tc in enumerate(token_chunks):
        h = rmsnorm(H[ci], onorm)
        logits = (h @ head.T).float()
        lp = torch.log_softmax(logits[:-1], -1)
        tgt = tc[1:].to(DEV)
        per = -lp[torch.arange(len(tgt), device=DEV), tgt]    # per-token NLL
        nll_full += per.sum().item(); nt_full += len(tgt)
        half = len(tc) // 2                                   # llama.cpp scores 2nd half (>=half ctx)
        nll_half += per[half-1:].sum().item(); nt_half += len(per[half-1:])
    del head; torch.cuda.empty_cache()
    return float(np.exp(nll_half/nt_half)), float(np.exp(nll_full/nt_full)), nt_half
