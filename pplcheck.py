#!/usr/bin/env python
"""Validate the torch Llama forward: perplexity on Q8 weights must be ~7.0 (matches llama.cpp).
Tokenize with llama-cpp-python (same tokenizer as the reference) for comparability."""
import numpy as np, torch, sys, os
import model as M

def get_tokens(nchunks=40, ctx=512):
    cache = "data/wiki_tokens.npy"
    if os.path.exists(cache):
        toks = np.load(cache)
    else:
        from llama_cpp import Llama
        llm = Llama(model_path=M.GG, vocab_only=True, verbose=False)
        txt = open("data/wiki.test.raw", encoding="utf-8").read()[:600000]
        toks = np.array(llm.tokenize(txt.encode("utf-8")), dtype=np.int64)
        np.save(cache, toks)
    chunks = [torch.tensor(toks[i*ctx:(i+1)*ctx]) for i in range(nchunks)]
    return [c for c in chunks if len(c) >= 2]

if __name__ == "__main__":
    unperm = "noperm" not in sys.argv
    print(f"loading weights (unpermute q/k = {unperm}) ...", flush=True)
    m = M.load_weights(unpermute=unperm)
    chunks = get_tokens()
    print(f"{len(chunks)} chunks; running torch perplexity ...", flush=True)
    ppl_half, ppl_full, ntok = M.perplexity(m, chunks)
    print(f"TORCH PPL  half(2nd)={ppl_half:.4f}  full={ppl_full:.4f}   (llama.cpp Q8 ref = 7.006)", flush=True)
