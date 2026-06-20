# RESUME — pick up here after power cycle

## Where we are
Best 2-bit result: **9.37 perplexity** (trellis + incoherence + V/K allocation), vs production
llama.cpp IQ2_XXS 11.83 -> **-21%**. Q8 (lossless) = 6.96. Target was 7.0.

Full trajectory (controlled, second-half ppl, ~2.1 bpw):
  IQ2_XXS 11.83 -> E8+alloc 10.04 -> trellis+alloc 9.37  (best)

## Verdict on 7.0 (settled, measured)
7.0 = lossless. Not reachable at true 2-bit: error is high-rank, weights near-Gaussian, cross-layer
corr ~0, full-Hessian helps only ~7% and is fragile. Realistic absolute floor with ALL levers ~8.3-8.5.
Details in FINDINGS.md (sections dated 2026-06-17) and RESULTS.md section 8.

## The open decision (what to do next)
1. Run end-to-end QAT / codebook fine-tuning (AQLM-style) — honest expected landing ~8.3-8.6, NOT 7.0.
   Hours of GPU. Would be the strongest possible number.
2. Lock in 9.37 and write up properly.
3. Fix full-H per-tensor (no sequential propagation) + L=12 for a marginal ~9.1 (low value).

## Pipeline state (all validated, on disk)
- torch Llama-8B (model.py): logits match llama.cpp 0.9999; torch-Q8 ppl 6.96 (second-half). WORKS.
- gptq.py: sequential quantize loop. Diagonal works (10.46 @ L10). full_H=True EXPLODES via propagation
  (do NOT use sequential full-H as-is). To use full-H: per-tensor only, no propagation.
- trellis.py: QTIP bitshift trellis + batched Viterbi (optimized: reshape-min, no gather). L=12 best.
- buildtrellis.py / buildmix.py / buildq.py: GGUF simulated-quant builders (resumable via .done files).
- Quantized models in models/8b_sim/: q8_trellis.gguf (9.37, BEST), q8_e8mix.gguf (10.04), q8_iq2.gguf (11.83 baseline).

## To re-validate after reboot (sanity)
  python pplcheck.py        # expect torch-Q8 half=6.96
  python gpu_bench.py --model models\8b_sim\q8_trellis.gguf --label trellis --ngl 99 --chunks 40 --no-bench  # expect ~9.37

## Notes
- torch model loads ~16GB weights to CPU each run (~40s); per-layer to GPU during forward.
- Tokens cached: data/wiki_tokens.npy (eval), data/wiki_train_tokens.npy (calib).
- A leftover "zombie" python (stuck bwtest in GPU driver) sometimes spins a core; harmless, dies on reboot.
