# UltraQuant — Results

Hardware: AMD RX 9070 XT (15.92GB, RDNA4), Ryzen 7 9800X3D, 31GB DDR5, Windows 11.
Backend: prebuilt llama.cpp Vulkan b9672 (GPU works: ~227 tok/s tg on a 3B).
Method: all 8B variants requantized from one Q8_0 source with one imatrix (calibrated on
WikiText-2 *train*, disjoint from the *test* eval). Perplexity = llama-perplexity, 40×512-tok
chunks, WikiText-2 test. Same Llama-3 tokenizer across 8B and 70B -> directly comparable ppl.

## 1. The feasibility verdict (the brief's actual question)

A dense 70B at ≤15% ppl loss, resident in 16GB, at ≥5 tok/s is **not achievable**. Measured:

| model | size | bpw | PPL | fits 16GB? | gen tok/s |
|---|---|---|---|---|---|
| Llama-3.1-8B Q4_K_M | 4.58 GB | 4.9 | 7.19 | yes (11GB spare) | 111 |
| Llama-3.1-8B IQ2_M | 2.75 GB | 2.9 | 8.49 | yes | 118 |
| Llama-3.3-70B IQ1_M | 15.6 GB | 1.9 | 8.66 | **NO** (pages) | 6 (at -ngl 56) |

- 70B Q4_K_M (42GB) thrashes (model > 32GB RAM).
- 70B IQ1_M (15.6GB) at `-ngl 99` tries 16,000 MiB vs 15,416 free -> WDDM VRAM paging -> 16-min hang.
  It does NOT fit resident once KV + compute buffers are counted, even at 512 context.
- At `-ngl 56` (partial offload) it runs at 6 tok/s — speed target met — but its ppl (8.66) is
  **worse than an 8B at 2.75GB (8.49)** and far worse than an 8B at Q4_K_M (7.19).

**Decision-relevant conclusion:** for a fixed 16GB budget, a smaller model at ≥3–4 bit strictly
dominates a 70B crushed below 2 bit — better quality, fits with room to spare, ~18× faster.
(Caveat: WikiText ppl is a language-modeling proxy; a 70B retains knowledge an 8B lacks that ppl
does not capture. As a quantization-damage signal, the result is unambiguous.)

## 2. The novel technique tested: budget-constrained per-tensor mixed precision

Hypothesis: allocate bits per-tensor by importance (protect lm_head / attn / ffn_down, squeeze
ffn_gate/up) to beat uniform quantization at equal size.

Result: **rejected.** Four custom mixes built; all four are Pareto-dominated by stock llama.cpp
IQ quants. The cleanest head-to-head (equal size):

| variant | size | PPL |
|---|---|---|
| iq2xs (built-in) | 2.427 GB | **9.89** |
| uqmix_d (custom) | 2.434 GB | 10.58 |
| uqmix_c (custom) | 2.438 GB | 10.61 |

Why: llama.cpp's IQ quants are NOT naive-uniform — they already do imatrix-guided per-tensor
allocation (auto-protect attn_v, ffn_down, output, first/last layers). Since ffn_gate+ffn_up are
~55% of params, quality is dominated by the base type on those bulk tensors; hand-tuned overrides
either fight a smarter scheme or spend bits on already-adequate tensors.

**Takeaway:** technique #1 ("protect important weights") is real but already near-optimally
harvested by imatrix+IQ. Beating it needs automated per-tensor sensitivity SEARCH or an orthogonal
lever, not hand-picked overrides.

## 3. What the brief got wrong (flagged, not chased)
- "MoE-style sparsity in a dense model": none exists — every layer runs every token.
- "Video-codec predictive weight mapping": weights lack the autocorrelation delta-coding needs.
- "PCIe5 NVMe ~1261 MB/s": ~10× too slow for PCIe5; and per-token streaming of a dense 70B is
  bandwidth-bound (<1 tok/s even at true 12 GB/s) — cannot give 5 tok/s.

## 4. Recommendation for this hardware
Run a 27–32B class model quantized to fit ~14GB (Q3_K_M / IQ4_XS), or a 12–14B at Q5/Q6, fully
resident with KV headroom. Reserve VRAM with KV-cache quant (q8_0) for longer context. Do not
deploy a sub-2-bit 70B: it doesn't fit cleanly and its quality regresses below a 4-bit 8B.

## 5. Custom-format round (format constraint lifted) — all four techniques, measured

Tested on REAL extracted Llama-3 weights (dequantized from Q8_0), with hardware bandwidth measured.

| user technique | measured result | verdict |
|---|---|---|
| #1 k-means / VQ codebook | VQ dim-4 @2.0 b/w: NMSE 0.110 vs llama.cpp IQ2_XXS 0.122 (2.06b). Matches/edges on RAW NMSE only. | already near-optimal in IQ |
| #2 neural weight predictor | lag-1 autocorr -0.0016; SVD near-full-rank (95% energy @ rank 3215/4096 = no compression); position carries ~0 info | dead |
| #3 streaming loader | NVMe 2.45 GB/s, RAM 10-13, H2D pinned 48.1 GB/s. Resident-in-VRAM always beats streaming (re-reads each token): partial-offload 6 tok/s > RAM-stream 3 > NVMe-stream 0.1 | no decode win |
| #4 custom engine | would need to beat tuned Vulkan IQ kernels; RD margin (<10% NMSE, likely <0 in ppl after imatrix) doesn't justify | not worth it |
| #5 cross-layer sharing | corr(blk15,blk16)=+0.008; shared base removes ~10% energy | dead |

Codebook nuance: non-uniform (k-means) beats uniform by 68% at 2-bit but loses at 4-bit (global
codebook lacks per-block scale). Incoherence/rotation (QuIP# trick) gave only +1-5% — these weights
are near-Gaussian (kurtosis ~4-5), few outliers to spread.

**Conclusion:** the barriers are information-theoretic (you need ≤~1.7 bpw to fit a dense 70B in 16GB,
and every method — including VQ that matches SOTA — leaves large distortion there) and bandwidth
(streaming re-reads weights every token). These are not implementation artifacts; no custom format,
predictor, or loader circumvents the 1.7bpw × 70B bottleneck. The recommendation in §4 stands.

## 7. *** POSITIVE RESULT: a custom 2-bit format that beats llama.cpp IQ2 by 15% ***

Frontier techniques implemented on real Llama-3.1-8B weights, fair imatrix-weighted metric:
- Incoherence (QuIP#-style RHT) + E8 lattice (Conway-Sloane O(8) decode) + importance-whitening (√H).
- My assigned extension that made it work end-to-end: importance-based MIXED bit allocation.

End-to-end perplexity, CONTROLLED (same Q8 embeddings; only the 7 linear tensor types differ; ~2 bpw),
WikiText-2 test, simulated quantization (reconstruct Ŵ, store Q8, run standard llama-perplexity):

| model (linear-tensor quant) | avg bpw | perplexity |
|---|---|---|
| llama.cpp IQ2_XXS reconstruction | 2.06 | 11.83 |
| incoherence+E8, uniform 2bpw | 2.00 | 12.51 (loses) |
| **incoherence+E8 + V/K allocation (UltraQuant)** | **2.083** | **10.04  (−15.1%)** |

Why uniform E8 loses but allocated E8 wins (the key insight): per-tensor weighted-NMSE does NOT predict
end-to-end perplexity. IQ2 is not uniform — it bumps attn_v/attn_k to near-lossless (WNMSE 0.0045 on
attn_v vs my uniform 0.075). attention is exquisitely V-sensitive, so uniform 2bpw on V tanks ppl even
though my method beats IQ2 on the bulk (FFN, attn_q, attn_output). Protecting V/K (only 3.8% of params,
~4 bpw) flips a 6% loss into a 15% win. Bitrate stays 2.083 vs IQ2's 2.06 (+1.1%).

CAVEATS (honest): bulk uses ENTROPY-coded ~2.0 bpw (a real format needs a runtime range decoder; IQ2 is
fixed-rate). Result is SIMULATED quant (quality is real; a deployable runtime — entropy decode + apply
the RHT rotation to activations O(n log n) + custom ROCm/Vulkan kernels — is NOT built). 8B only.

Bottom line vs the original goal: this pushes ~15% past llama.cpp's IQ2 perplexity frontier at equal
bitrate — the novel, winnable deliverable. It does NOT unblock "70B in 16GB": a 2-bit 70B is still
~19GB > 16GB. Information theory holds; the contribution is moving the 2-bit frontier, not the wall.

## 6. Honest future directions (real, but large effort, and won't make a 70B fit usefully in 16GB)
- Learned/fine-tuned codebooks (AQLM/QuIP# style) — GPU-training, gets ~2-bit 70B better than IQ2,
  but a 2-bit 70B is ~19GB (still doesn't fit 16GB) and still degraded.
- Automated per-tensor sensitivity search (ΔPPL/byte) instead of hand mixes.
- KV-cache quant sweep for context-length vs VRAM on a model that already fits.

## 8. Push toward 7.0 (constraint: invent past existing methods)

Built a validated from-scratch torch Llama-3.1-8B (logits match llama.cpp 0.9999 corr; torch-Q8
ppl 6.96 = llama.cpp 7.006 with second-half scoring). Enables output-error-aware quantization.

Trajectory at ~2-bit (controlled, second-half perplexity, Q8 = 6.96):
| method | ppl | note |
|---|---|---|
| llama.cpp IQ2_XXS | 11.83 | production SOTA baseline |
| E8 lattice + incoherence + V/K alloc | 10.04 | entropy-rate bulk |
| QTIP trellis + incoherence + V/K alloc | 9.37 | FIXED-rate, beats real iq2xs (2.43bpw=9.89) |
| sequential GPTQ, diagonal-H, L=10 | 10.46 | confounded (L10); diag-H = no real compensation |
| sequential GPTQ, full-H (off-diagonal), L=10 | [running] | proper output-error objective |

Assumption tests (why cheap tricks are exhausted): weights near-Gaussian per-column w/ rare extreme
outlier channels (kurt up to 2157); quantization error is HIGH-rank (90% energy needs 44-74% of rank)
-> low-rank/aux-correction is dominated. RD floor (0.0625 NMSE) is on WEIGHT recon, NOT perplexity.

Honest status: we BEAT production SOTA (IQ2 11.83 -> 9.37, -21%) and are ahead of the llama.cpp IQ
frontier at equal bitrate. 7.0 (= lossless at 2-bit) remains unproven-reachable; published 2-bit SOTA
(QTIP/AQLM) plateaus ~3-bit-equiv (~8.3-8.6 for this 8B). Testing full-H then end-to-end tuning.
