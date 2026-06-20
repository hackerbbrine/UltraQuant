#!/usr/bin/env python
"""Build controlled simulated-quantization models in a Q8 container.

Two models from the SAME Q8 base, differing ONLY in how the 7 linear tensor types
(attn_q/k/v/output, ffn_gate/up/down) are quantized -> isolates the METHOD at ~2bpw:
  A: incoherence + E8 lattice (QuIP#-core), importance-whitened
  B: llama.cpp's actual IQ2_XXS reconstruction (the bar)
Everything else (embeddings, norms, output) = original Q8 in both -> controlled.
Quality is measured by perplexity; storage stays Q8 (we measure quality-at-bitrate, not file size).
"""
import numpy as np, shutil, sys, time, os
from gguf import GGUFReader
import gguf
from qlab import load_WH, rht_rows, irht_rows, q_e8, q_e8_fast, deq_q8, _find

Q8 = r"models\Llama-3.1-8B-Q8_0.gguf"
IQ2 = r"models\8b\iq2xxs.gguf"
OUTDIR = r"models\8b_sim"
LIN = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")

def enc_q8(W):
    rows, cols = W.shape; nb = cols // 32
    x = W.reshape(rows, nb, 32).astype(np.float32)
    sc = np.max(np.abs(x), 2, keepdims=True) / 127.0
    sc = np.where(sc == 0, 1e-9, sc)
    q = np.round(x / sc).clip(-127, 127).astype(np.int8)
    out = np.zeros((rows, nb, 34), np.uint8)
    out[:, :, :2] = sc.squeeze(-1).astype(np.float16).view(np.uint8).reshape(rows, nb, 2)
    out[:, :, 2:] = q.view(np.uint8)
    return out.reshape(rows, nb * 34)

def quant_e8(W, H, bpw=2.0):
    H = np.maximum(H, H.max() * 1e-3)
    sH = np.sqrt(H)[None, :]
    A = W * sH
    in0 = A.shape[1]
    Arot = rht_rows(A)
    Ah_rot = q_e8_fast(Arot, bpw, calib_key="global")   # fast closed-form E8, global step ~2bpw
    Ah = irht_rows(Ah_rot, in0)
    return (Ah / sH).astype(np.float32)

def is_lin(name):
    return name.endswith(".weight") and any(f".{k}.weight" in name for k in LIN)

def build(method, layers=None):
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"q8_{method}.gguf")
    print(f"[copy] {Q8} -> {out}", flush=True); shutil.copyfile(Q8, out)
    r = GGUFReader(Q8)
    iq = GGUFReader(IQ2) if method == "iq2" else None
    tens = [t for t in r.tensors if is_lin(t.name)]
    if layers is not None:
        tens = [t for t in tens if int(t.name.split('.')[1]) < layers]
    fh = open(out, "r+b")
    t0 = time.time()
    for i, t in enumerate(tens):
        W, H = load_WH(t.name)
        if method == "e8":
            Wh = quant_e8(W, H, 2.0)
        elif method == "iq2":
            Wh = gguf.quants.dequantize(_find(iq, t.name).data, _find(iq, t.name).tensor_type).astype(np.float32)
        enc = enc_q8(Wh)
        assert enc.nbytes == t.n_bytes, f"size mismatch {t.name}: {enc.nbytes} vs {t.n_bytes}"
        fh.seek(int(t.data_offset)); fh.write(enc.tobytes())
        if i % 28 == 0:
            print(f"  [{i+1}/{len(tens)}] {t.name}  ({time.time()-t0:.0f}s)", flush=True)
    fh.close()
    print(f"[done] {method}: {len(tens)} tensors in {time.time()-t0:.0f}s -> {out}", flush=True)

if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "e8"
    layers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    build(method, layers)
