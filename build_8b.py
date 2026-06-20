#!/usr/bin/env python
"""Build the 8B quant variants for the controlled UltraQuant experiment.

Reference + source: Q8_0 (near-lossless). All variants requantized from it with
the SAME imatrix (built from wiki.train, disjoint from the wiki.test eval set).

Thesis under test: importance-aware, budget-constrained per-tensor mixed precision
beats uniform quantization at equal model size.

Custom mix rationale (Llama-3.1-8B: FFN ~80% of params, ffn_down most sensitive,
attention sensitive but cheap, lm_head directly produces logits):
  - output (lm_head): Q6_K        - keep, directly drives logits
  - token_embd:       Q4_K        - large, accessed by lookup, tolerant
  - attn q/k/v/o:     IQ4_XS      - sensitive but only ~20% of params -> cheap to protect
  - ffn_down:         IQ3_XXS     - most sensitive FFN tensor -> protect
  - ffn_gate/up:      IQ2_XXS/IQ1 - the compressible bulk
"""
import os, sys, json
from quantize import build_imatrix, quantize

ROOT = r"C:\Users\Tony Stark\Documents\UltraQuant"
SRC  = os.path.join(ROOT, "models", "Llama-3.1-8B-Q8_0.gguf")
OUT  = os.path.join(ROOT, "models", "8b")
IMAT = os.path.join(ROOT, "models", "8b", "imatrix.dat")
CAL  = os.path.join(ROOT, "data", "wiki.train.raw")
os.makedirs(OUT, exist_ok=True)

def p(name): return os.path.join(OUT, name)

# uniform baselines: (label, type, use_imatrix)
UNIFORM = [
    ("q4km",   "Q4_K_M",  False),   # brief baseline
    ("q3km",   "Q3_K_M",  False),
    ("iq3xxs", "IQ3_XXS", True),
    ("iq2m",   "IQ2_M",   True),
    ("iq2xxs", "IQ2_XXS", True),
    ("iq1m",   "IQ1_M",   True),
]

# custom mixes: (label, base_type, tensor_type_overrides[], output_type, embd_type)
# base_type sets the default; overrides refine specific tensors.
CUSTOM = [
    # ~IQ2_M size, but bits steered toward sensitive tensors
    ("uqmix_a", "IQ2_XXS",
     ["attn_q=iq4_xs","attn_k=iq4_xs","attn_v=iq4_xs","attn_output=iq4_xs",
      "ffn_down=iq3_xxs"], "Q6_K", "Q4_K"),
    # ~IQ1_M size, aggressive but protect attention + ffn_down + lm_head
    ("uqmix_b", "IQ1_M",
     ["attn_v=iq4_xs","attn_k=iq4_xs","ffn_down=iq2_xxs"], "Q6_K", "Q5_K"),
]

def main():
    if not os.path.exists(SRC):
        print("source not ready:", SRC); sys.exit(1)
    build_imatrix(SRC, CAL, IMAT, ngl=99, chunks=80)
    built = []
    for label, qt, use_im in UNIFORM:
        out = p(f"{label}.gguf")
        if os.path.exists(out):
            print("skip", out)
        else:
            quantize(SRC, out, qt, imatrix=(IMAT if use_im else None))
        built.append((label, out))
    for label, base, tts, ot, et in CUSTOM:
        out = p(f"{label}.gguf")
        if os.path.exists(out):
            print("skip", out)
        else:
            quantize(SRC, out, base, imatrix=IMAT, tensor_types=tts,
                     output_type=ot, embd_type=et)
        built.append((label, out))
    print("\n=== sizes ===")
    for label, out in built:
        if os.path.exists(out):
            print(f"  {label:10s} {os.path.getsize(out)/1024**3:6.3f} GB  {out}")

if __name__ == "__main__":
    main()
