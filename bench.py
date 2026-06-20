#!/usr/bin/env python
"""UltraQuant benchmark harness.

Measures, for a GGUF model, three things on THIS machine:
  - perplexity  (sliding-window NLL over a fixed corpus)
  - VRAM used   (device-level free-memory delta around model load, via ROCm/HIP)
  - speed       (prompt-eval + generation tok/s)

Model can be given as an Ollama name (resolved to its GGUF blob) or a direct path.
"""
import argparse, json, os, sys, time, glob
import numpy as np

# ---- ROCm device VRAM probe (works for llama.cpp's allocations too: device-level) ----
def vram_free_bytes():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.mem_get_info()[0]
    except Exception:
        pass
    return None

# ---- resolve an ollama model name -> gguf blob path ----
def resolve_model(name_or_path):
    if os.path.exists(name_or_path):
        return name_or_path
    root = os.path.join(os.path.expanduser("~"), ".ollama", "models")
    # name like "llama3.2:latest" or "llama3.3:70b"
    if ":" in name_or_path:
        model, tag = name_or_path.split(":", 1)
    else:
        model, tag = name_or_path, "latest"
    man = os.path.join(root, "manifests", "registry.ollama.ai", "library", model, tag)
    if not os.path.exists(man):
        # try any registry namespace
        cands = glob.glob(os.path.join(root, "manifests", "**", model, tag), recursive=True)
        man = cands[0] if cands else None
    if not man or not os.path.exists(man):
        raise FileNotFoundError(f"no ollama manifest for {name_or_path}")
    with open(man, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    digest = None
    for layer in manifest.get("layers", []):
        if layer.get("mediaType", "").endswith("model"):
            digest = layer["digest"]
            break
    if not digest:
        raise RuntimeError(f"no model layer in manifest {man}")
    blob = os.path.join(root, "blobs", digest.replace(":", "-"))
    if not os.path.exists(blob):
        raise FileNotFoundError(f"blob missing: {blob}")
    return blob

# ---- corpus ----
def load_corpus(path, max_chars):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    return txt[:max_chars] if max_chars else txt

# ---- perplexity ----
def perplexity(llm, token_ids, n_ctx, max_chunks):
    from scipy.special import logsumexp
    n_vocab = llm.n_vocab()
    chunks = [token_ids[i:i+n_ctx] for i in range(0, len(token_ids), n_ctx)]
    chunks = [c for c in chunks if len(c) >= 2]
    if max_chunks:
        chunks = chunks[:max_chunks]
    total_nll, total_tok = 0.0, 0
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        llm.reset()
        llm.eval(chunk)
        scores = np.asarray(llm.scores)              # (n_ctx, n_vocab)
        if scores.ndim == 1:
            scores = scores.reshape(-1, n_vocab)
        m = len(chunk)
        logits = scores[:m-1].astype(np.float64)     # predict token i+1 from pos i
        targets = np.asarray(chunk[1:m])
        lse = logsumexp(logits, axis=1)
        tgt_logits = logits[np.arange(m-1), targets]
        nll = -(tgt_logits - lse)
        total_nll += float(nll.sum())
        total_tok += (m - 1)
        if ci % 5 == 0:
            cur = np.exp(total_nll / max(total_tok, 1))
            print(f"  chunk {ci+1}/{len(chunks)}  running ppl={cur:.4f}", flush=True)
    dt = time.time() - t0
    ppl = float(np.exp(total_nll / max(total_tok, 1)))
    return ppl, total_tok, dt

# ---- speed ----
def measure_speed(llm, prompt, gen_tokens):
    llm.reset()
    t0 = time.time()
    out = llm.create_completion(prompt, max_tokens=gen_tokens, temperature=0.0)
    dt = time.time() - t0
    n = out["usage"]["completion_tokens"]
    return n / dt if dt > 0 else 0.0, n, dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngl", type=int, default=-1, help="n_gpu_layers (-1 = all)")
    ap.add_argument("--nctx", type=int, default=512)
    ap.add_argument("--max-chunks", type=int, default=20)
    ap.add_argument("--corpus", default="data/wiki.test.raw")
    ap.add_argument("--max-chars", type=int, default=600000)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--kv-type", default=None, help="q8_0|q4_0 for KV cache quant")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    path = resolve_model(args.model)
    print(f"[model] {args.model} -> {path}", flush=True)
    print(f"[model] file size: {os.path.getsize(path)/1024**3:.2f} GB", flush=True)

    free0 = vram_free_bytes()

    from llama_cpp import Llama
    kw = dict(model_path=path, n_ctx=args.nctx, n_gpu_layers=args.ngl,
              logits_all=True, verbose=False)
    if args.kv_type:
        import llama_cpp as lc
        tmap = {"q8_0": lc.GGML_TYPE_Q8_0, "q4_0": lc.GGML_TYPE_Q4_0,
                "f16": lc.GGML_TYPE_F16}
        kw["type_k"] = tmap[args.kv_type]; kw["type_v"] = tmap[args.kv_type]
        kw["flash_attn"] = True

    t_load = time.time()
    llm = Llama(**kw)
    load_s = time.time() - t_load
    free1 = vram_free_bytes()
    vram_mb = (free0 - free1) / 1024**2 if (free0 and free1) else float("nan")
    print(f"[load] {load_s:.1f}s   VRAM used: {vram_mb:.0f} MB", flush=True)

    corpus = load_corpus(args.corpus, args.max_chars)
    tokens = llm.tokenize(corpus.encode("utf-8"))
    print(f"[corpus] {len(corpus)} chars -> {len(tokens)} tokens", flush=True)

    ppl, ntok, ppl_s = perplexity(llm, tokens, args.nctx, args.max_chunks)
    print(f"[ppl] {ppl:.4f} over {ntok} tokens in {ppl_s:.1f}s", flush=True)

    spd, gn, gs = measure_speed(llm, "The history of artificial intelligence began", args.gen_tokens)
    print(f"[speed] {spd:.2f} tok/s ({gn} tok in {gs:.1f}s)", flush=True)

    rec = dict(tag=args.tag or args.model, model=args.model, path=path,
               file_gb=round(os.path.getsize(path)/1024**3, 3),
               ngl=args.ngl, nctx=args.nctx, kv_type=args.kv_type,
               vram_mb=round(vram_mb, 1), load_s=round(load_s, 1),
               ppl=round(ppl, 4), ppl_tokens=ntok,
               tok_s=round(spd, 2))
    os.makedirs("results", exist_ok=True)
    with open("results/bench.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print("[done]", json.dumps(rec), flush=True)

if __name__ == "__main__":
    main()
