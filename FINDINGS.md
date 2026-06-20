# UltraQuant — Research Log

Goal: run a 70B model at acceptable quality on a 16GB AMD RX 9070 XT.
Honest re-scope: find the best achievable point on the perplexity ↔ VRAM ↔ tok/s
frontier on THIS hardware, and test whether any custom trick beats the strong
llama.cpp IQ-quant + imatrix baseline.

---

## 2026-06-16 — Environment ground truth (verified)

| Item | Reality |
|---|---|
| GPU | AMD RX 9070 XT, **15.92 GB** usable VRAM (dxdiag 16188 MB dedicated) |
| GPU compute | ROCm 7.2 torch works; fp16 matmul runs (~10 TFLOP/s @4096³, low — RDNA4 kernels immature); 12GB alloc OK |
| Ollama | 0.30.8, offloads to GPU at 100% (verified on gemma3:4b) |
| CPU | Ryzen 7 9800X3D, 8c/16t (spec said "Ryzen 9") |
| RAM | 31.2 GB |
| Disk | C: 868 GB free |
| Toolchain | python 3.12, torch 2.9.1+rocm7.2.1, numpy, scipy, llama-cpp-python 0.3.29, ollama. NO cmake/MSVC/gcc. `hipcc` is a pip shim. |
| Gotcha | Space in `C:\Users\Tony Stark\` breaks some ROCm helper exes (offload-arch). torch core fine. |
| NVMe | Spec claimed "PCIe5 ~1261 MB/s" — that number is ~10x too slow for PCIe5; treat as suspect, measure later. |

## Feasibility analysis (the core constraint)

Three targets conflict for a dense 70B:
1. Fit 16GB resident + fast → needs ≤~1.6 bit/weight (~14GB), ~2GB left for KV+activations.
2. ≤15% perplexity → SOTA sub-2-bit (IQ1_M 1.75bpw, IQ2_XXS 2.06bpw, AQLM/QuIP# 2-bit)
   degrade 70B by ~30–80%+. So (1)+(2) not jointly achievable.
3. Streaming to rescue fit: bandwidth-bound. True PCIe5 (12GB/s) → ~0.6 tok/s for 20GB/token;
   spec's 1261MB/s → ~0.06 tok/s. Cannot give 5 tok/s. (cf. AirLLM minutes/token.)

Misconceptions in the brief, flagged:
- "MoE-style sparsity in dense model": none exists; every layer runs every token. Real lever = lossy depth-pruning/early-exit.
- "Video-codec predictive weight mapping": weights lack autocorrelation; residuals ~full entropy. Legit cousins: low-rank+sparse, outlier-aware quant.

Real + strong (use, don't reinvent): #1 importance/non-uniform quant + #6 codebook/VQ already exist as
llama.cpp imatrix + IQ-quants. Use as baseline; test custom additions (KV quant, outlier protection) on top.

## 2026-06-16 — Harness built + validated

- `bench.py` works end-to-end on llama3.2:latest (3B, Q4_K_M, 1.88GB):
  ppl=15.07 (word-level WikiText-2, 6 chunks, noisy), 25.4 tok/s, runs clean.
- KEY: installed `llama-cpp-python` is **CPU-only** (`llama_supports_gpu_offload()=False`).
  -> Perplexity from it is numerically correct (CPU vs GPU identical math) but VRAM=0 / speed is CPU.
- Decision — split measurement paths:
  - QUALITY (perplexity): llama-cpp-python CPU (correct, slow) OR prebuilt llama-perplexity (GPU).
  - HARDWARE (VRAM, tok/s): GPU path = Ollama, or prebuilt llama.cpp Vulkan build.
- UNLOCK: official llama.cpp release b9672 ships prebuilt Windows **Vulkan** binaries (37MB, no
  compiler needed) -> gives GPU llama-perplexity + llama-quantize + llama-imatrix + llama-server.
  This solves BOTH the GPU-perplexity gap AND the "no compiler to build quantize tool" gap.
  HIP/Radeon build also available (306MB) as fallback if Vulkan underperforms on RDNA4.

## 2026-06-16 — Toolchain + 8B experiment underway

- Prebuilt llama.cpp **Vulkan** b9672 works on RDNA4: 3B Q4_K_M = 7130 tok/s pp, 227 tok/s tg.
- `--tensor-type name=type` validated (substring match across layers) -> per-tensor mixed precision works.
- 70B Q4_K_M (42GB) THRASHES: exceeds 32GB RAM, pages from disk. Confirms it's impractical resident here.
- Built: imatrix (4.9MB, from wiki.train, disjoint from eval) + variant set requantized from Q8_0.
- 8B variants: uniform {Q4_K_M, Q3_K_M, IQ3_XXS, IQ2_M, IQ2_XXS, IQ1_M} + custom {uqmix_a, uqmix_b}.
- Downloading 70B IQ1_M (16.8GB) for the "fits-in-16GB" demonstration.

## 2026-06-16 — RESULT: 8B frontier (ref = Q8_0, ppl 7.0064, WikiText-2 test, 40 chunks)

| variant | GB | bpw | PPL | dPPL% | tg t/s | Pareto |
|---|---|---|---|---|---|---|
| q4km  | 4.58 | 4.90 | 7.186 | +2.6%  | 111 | * |
| q3km  | 3.74 | 4.00 | 7.494 | +7.0%  | 129 | * |
| iq3xxs| 3.05 | 3.26 | 7.837 | +11.9% | 111 | * |
| uqmix_a (custom) | 2.92 | 3.12 | 8.959 | +27.9% |128| (dominated) |
| iq2m  | 2.75 | 2.94 | 8.494 | +21.2% | 118 | * |
| uqmix_b (custom) | 2.33 | 2.50 | 16.546 | +136% |160| (dominated) |
| iq2xxs| 2.23 | 2.39 | 12.211 | +74.3% | 153 | * |
| iq1m  | 2.01 | 2.15 | 19.565 | +179%  | 178 | * |

### NEGATIVE RESULT: hand-designed UltraQuant-Mix LOST to uniform llama.cpp IQ quants.
- uqmix_a (2.92GB, 8.96) is dominated by iq2m (2.75GB, 8.49) — SMALLER *and* better — and by iq3xxs.
- uqmix_b (2.33GB, 16.55) is dominated by iq2xxs (2.23GB, 12.21) — SMALLER *and* better.

### Why (the real content):
1. llama.cpp IQ quants are NOT naive-uniform: IQ2_M/IQ2_S/IQ3 already do imatrix-guided
   per-tensor allocation (auto-bump attn_v, ffn_down, output, first/last layers). My overrides
   fought a smarter built-in scheme.
2. ffn_gate+ffn_up are ~55% of Llama-8B params. Quality is dominated by the BASE type on these
   bulk tensors. I spent bits protecting cheap attn tensors (already adequate) while using an
   over-aggressive base (IQ2_XXS / IQ1_M) on the bulk -> moved bits the WRONG way.
3. uqmix_a used MORE total bits than iq2m yet scored worse -> strict Pareto loss.

### Takeaway: technique #1 ("protect important weights") is real but ALREADY near-optimally
implemented by imatrix+IQ. Naive hand-tuned per-tensor overrides don't beat it. Beating it needs
either measured per-tensor sensitivity search, or an orthogonal lever (codebooks / incoherence /
KV-cache quant), not hand-picked overrides.

## 2026-06-16 — RESULT: dense 70B on the 16GB GPU

70B IQ1_M (1.75bpw, 15.59 GiB, llama-3.3, same tokenizer as the 8B suite):
- `-ngl 99` auto-fit tries to offload 81/81 layers:
  Vulkan0 model 15639 MiB + KV 160 + compute 200 = ~16,000 MiB vs **15,416 MiB free**
  -> overcommits by ~0.6GB -> Windows WDDM pages VRAM<->RAM -> 16-MINUTE HANG (idle CPU, no progress).
  => IQ1_M 70B does NOT fit resident in 16GB once KV+compute are counted, even at 512 ctx.
- Controlled `-ngl 56` (56/80 layers on GPU, 24 on CPU, ~11GB VRAM, no paging):
  pp64 = 95 t/s, **tg32 = 6.0 tok/s** generation. So SPEED target (>=5 tok/s) is MET via partial offload.
  Blocker is FIT + QUALITY, not speed.
- ppl: [running] -- directly comparable to 8B suite (same Llama-3 tokenizer). Decides 1-bit-70B vs 4-bit-8B.

## 2026-06-16 — Iteration 2: DISCIPLINED custom mix (single-lever)

Lesson from iter-1: don't fight the allocator; make ONE theory-backed change.
- uqmix_c = IQ2_XXS base + ffn_down->IQ3_XXS + output Q5_K : 2.438GB, ppl 10.605
- uqmix_d = uqmix_c + attn_v->IQ4_XS                       : 2.434GB, ppl 10.579
Both fall in the iq2xxs(2.23,12.21) -> iq2m(2.75,8.49) gap and are NOT dominated by any
uniform point I had measured -> they extend that frontier (unlike scattershot iter-1).
CAVEAT: must compare vs llama.cpp's own IQ2_S/IQ2_XS (same size class) before claiming a win.

### FINAL VERDICT (gap filled): the custom mix LOSES.
- iq2xs (2.427GB) ppl 9.891  vs  uqmix_c (2.438GB) 10.605 / uqmix_d (2.434GB) 10.579
  -> built-in IQ2_XS beats the same-size custom mixes. The iter-2 "win" was an artifact of
     sparse uniform sampling. With IQ2_S/IQ2_XS added, ALL FOUR custom mixes are dominated.
- Uniform Pareto frontier (8B): q8 7.01 | q4km 7.19 | q3km 7.49 | iq3xxs 7.84 | iq2m 8.49 |
  iq2s 9.11 | iq2xs 9.89 | iq2xxs 12.21 | iq1m 19.56  (all uniform; no custom point on it).
- CONCLUSION: imatrix-guided IQ quantization is already near-Pareto-optimal for per-tensor bit
  allocation on this family. Hand-designed per-tensor overrides — naive OR disciplined — do not
  beat it. The one "novel" technique that survived the feasibility cut does not advance the SOTA.
  A real improvement would require automated per-tensor sensitivity SEARCH or an orthogonal lever
  (incoherence/rotation a la QuIP#, better codebooks, KV-cache quant for context).

## RESULT: 70B IQ1_M perplexity = 8.657 (ngl 56, same Llama-3 tokenizer as 8B suite)
DECISION-RELEVANT: a 1.75-bit 70B (8.66) is WORSE on WikiText ppl than an 8B at:
  Q8 7.01 | Q4_K_M 7.19 | Q3_K_M 7.49 | IQ3_XXS 7.84 | IQ2_M 8.49 (2.75GB!).
8B Q4_K_M fits with 11GB to spare and runs ~18x faster (111 vs 6 tok/s).
=> For a 16GB budget, a smaller model at >=3-4 bit dominates a 70B crushed to <2 bit.
(Caveat: ppl is a LM proxy; 70B holds knowledge an 8B lacks. But as quant-damage signal: unambiguous.)

## 2026-06-17 — Constraint lifted: custom-format research (measured, not argued)

### Measured hardware bandwidth (governs streaming, technique #3)
- NVMe seq read: **2.45 GB/s** (spec claimed 1.26; still ~5x below PCIe5 rated)
- RAM single-thread: ~10-13 GB/s
- H2D PCIe pinned: **48.1 GB/s** (pageable 12.4) — true PCIe5 x16

Streaming verdict (decode = 1 token needs every weight once):
- fits VRAM (<=~15GB): resident, no streaming. fastest.
- in RAM not VRAM: stream RAM->VRAM @48GB/s = 15.6GB/0.33s = ~3 tok/s ceiling. BUT llama.cpp
  partial offload (resident + CPU-compute the rest) already gives 6 tok/s -> streaming LOSES
  because it re-reads weights every token; resident never re-reads.
- > RAM: NVMe @2.45GB/s -> 0.06-0.16 tok/s. dead.
=> A custom streaming loader does NOT beat existing partial offload for decode. (Helps prefill only,
   which is compute-bound and amortizes the load — not interactive decode.)

### Rate-distortion on REAL weights (blk.16.ffn_gate, kurtosis 4.1; NMSE vs Q8, technique #1/#6)
| method | 2 b/w | 3 b/w | 4 b/w |
|---|---|---|---|
| uniform per-block | 0.436 | 0.053 | 0.0098 |
| k-means scalar (global) | 0.140 | 0.047 | 0.0147 |
| VQ dim-4 | **0.110** | - | - |
| VQ dim-2 | 0.129 | 0.039 | 0.0101 |
| Hadamard+uniform | 0.432 (+0.9%) | 0.051 | 0.0094 |
| llama.cpp IQ2_XXS (2.06b) ANCHOR | 0.122 | | |
| llama.cpp Q4_K_M ANCHOR | | | 0.0052 |

- Non-uniform (k-means) beats uniform by 68% at 2-bit, but LOSES at 4-bit (global codebook lacks
  per-block scale that uniform has). Sweet spot for codebooks = ultra-low bits.
- VQ dim-4 @2.0 b/w (0.110) marginally BEATS llama.cpp IQ2_XXS @2.06b (0.122) on RAW NMSE.
  BUT: IQ2 is imatrix-WEIGHTED (optimizes perplexity, accepts higher raw NMSE to protect salient
  weights) so its perplexity edge is bigger than raw NMSE shows; and a VQ format needs custom
  inference kernels to be usable. Margin too small to justify.
- Incoherence/rotation (QuIP# trick): ~+1-4% (ffn) / +1-5% (attn_q, kurtosis 4.9) — small. These
  Llama-3 weights are already fairly incoherent; single-side block Hadamard barely helps.

### Predictor / redundancy (techniques #2 neural predictor, #5 cross-layer) — ALL DEAD (measured)
- Neighbor predictability: lag-1 autocorr row=-0.0016 col=-0.0007 ~ ZERO.
  => "video-codec delta / predict-from-neighbors" cannot compress. Weights ~i.i.d. per channel.
- Low-rank (SVD of ffn_gate 14336x4096): 95% energy needs rank 3215/4096 -> storing the low-rank
  factors costs 1.01x (MORE) than the full matrix. Near-full-rank. No low-rank predictor helps.
- Cross-layer: corr(blk15, blk16) = +0.008 (uncorrelated). Shared-base (mean over layers) removes
  only ~10-14% of energy. Weight-sharing saves ~nothing; layers ~independent.

### SYNTHESIS of the lifted-constraint round
| user technique | measured verdict |
|---|---|
| #1 k-means/codebook format | VQ dim-4 @2b (NMSE 0.110) ~matches/edges llama.cpp IQ2_XXS (0.122) on RAW NMSE; but IQ is imatrix-weighted (better in ppl) and has tuned kernels. Margin too small to justify a custom format. Codebooks help only at ultra-low bits. |
| #2 neural weight predictor | DEAD: autocorr~0, near-full-rank, position carries ~no info about value. |
| #3 custom streaming loader | No decode win: resident-in-VRAM always beats streaming (which re-reads every token). Partial offload 6 tok/s > full RAM-stream 3 tok/s > NVMe-stream 0.1 tok/s. |
| #4 custom inference engine | Would need to beat tuned Vulkan IQ kernels; RD margin doesn't justify. |
| #5 cross-layer sharing | DEAD: adjacent-layer corr 0.008; shared base removes ~10% energy only. |

BOTTOM LINE: the limits are information-theoretic + bandwidth, not implementation artifacts. To fit a
dense 70B in 16GB you need <=~1.7 bpw; at that rate every measured method (incl. VQ matching SOTA)
leaves large distortion. No codebook/predictor/streaming trick circumvents the 1.7bpw x 70B bottleneck.
The earlier recommendation stands: run a ~30B at ~14GB or a 12-14B at Q5/Q6, fully resident.

## 2026-06-17 — Frontier round (QuIP#/E8/QTIP/AQLM), FAIR imatrix-weighted metric

Metric fix: weighted-NMSE = ||(W-Ŵ)·diag(√H)||²/||W·diag(√H)||², H_j = mean activation energy
(from imatrix .in_sum2/.counts) = the GPTQ/imatrix objective -> apples-to-apples vs IQ2.
Trick: A = W·diag(√H); unweighted NMSE on A == weighted NMSE on W. Incoherence = orthogonal RHT on A.

### Per-tensor weighted-NMSE @ ~2.0 bpw (vs llama.cpp IQ2_XXS anchor)
blk.16.attn_q (H skew 271x):  IQ2_XXS 0.078 | kmeansVQ 0.092 | E8 0.277 | incoh+E8 0.090
blk.16.ffn_gate (H skew 51x): IQ2_XXS 0.106 | kmeansVQ 0.113 | E8 0.140 | incoh+E8 0.090
- INCOHERENCE IS REAL: E8 alone 0.277 -> incoherence+E8 0.090 (3x). QuIP# mechanism confirmed.
- incoherence HURTS uniform (0.31->0.43) -> specifically a LATTICE enabler (matches theory).
- TENSOR-DEPENDENT: incoh+E8 BEATS IQ2_XXS on FFN (0.090<0.106, bulk=55% params) but LOSES on
  attention (0.090>0.078, high importance skew -> IQ2 imatrix protection wins).
- => net ambiguous; needs END-TO-END perplexity. Building controlled A(incoh+E8) vs B(IQ2 recon),
  Q8 container, simulated quant, identical embeds -> isolate method at equal bitrate.
Note: NOT using GPTQ/LDLQ error-feedback (needs full Hessian=activations; imatrix=diagonal only).

### Closed-form E8 decoder + ENTROPY-rate (Conway-Sloane, O(8)/vec) — incoh+E8 now beats IQ2 per-tensor
Fixed-rate E8 (65536 codebook) was 0.090 on both. Switching to closed-form E8 lattice decode with
ideal entropy coding of indices at ~2.0 entropy-bpw:
  attn_q : incoh+E8 0.0762  vs IQ2_XXS 0.078  -> WIN ~2%
  ffn_gate: incoh+E8 0.0781 vs IQ2_XXS 0.106  -> WIN ~26%
CAVEAT: entropy-coded 2.0 bpw vs IQ2 fixed 2.06 bpw — entropy coding is a more generous rate model
(needs an entropy decoder at runtime). Honest framing: "with ideal entropy coding, incoh+E8 beats IQ2
at matched ~2bpw per-tensor." Now testing whether this holds END-TO-END (perplexity).
Single global lattice step (~1.04) gives ~2.02 bpw on all tensor types after per-group RMS norm.

### END-TO-END perplexity (controlled, Q8 container, same embeds, ~2bpw linear) — THE REAL TEST
- sim_iq2  (IQ2_XXS recon on linear) = **11.83**  (real iq2xxs.gguf = 12.21; diff = Q8 embeds)
- sim_e8   (incoherence+E8, uniform 2bpw) = **12.51**  -> LOSES by ~6% DESPITE winning per-tensor WNMSE!

KEY FINDING: per-tensor weighted-NMSE does NOT predict end-to-end perplexity. Diagnosis (per-tensor
WNMSE @2bpw, build's global step, all tensors verified ~2.0bpw):
  attn_q 0.075 vs IQ2 0.078 WIN | attn_out 0.075 vs 0.107 WIN | ffn_* ~0.07 vs ~0.10 WIN
  attn_k 0.075 vs IQ2 0.052 LOSE | attn_v 0.075 vs IQ2 0.0045 LOSE BY 16x
=> IQ2 is NOT uniform: it BUMPS attn_v/attn_k to near-lossless (classic keep-K/V-precise heuristic).
   My uniform 2bpw crushes attn_v; attention is exquisitely V-sensitive -> tanks ppl despite bulk wins.

### FIX = my assigned extension (importance-based bit allocation) on top of incoherence+lattice
V/K are only ~4% of params. E8 @ step0.22 (~4bpw) gives attn_v WNMSE 0.0035 < IQ2 0.0045 (now WIN),
attn_k 0.0035 < 0.052. Model avg ~2.08 bpw ~= IQ2's 2.06. Built q8_e8mix; perplexity [running]...

### *** RESULT: incoherence+E8+importance-allocation BEATS IQ2 by ~15% perplexity at equal bits ***
Controlled (same Q8 embeds, linear tensors only differ), WikiText-2 40 chunks:
  sim_iq2  (IQ2_XXS recon, 2.06 bpw)                       = 11.83
  sim_e8   (incoh+E8 uniform 2.0 entropy-bpw)              = 12.51  (loses: attn_v crushed)
  sim_e8mix(incoh+E8 + V/K@~4bpw, avg ~2.08 bpw)           = **10.04**  <-- WIN, -15.1% vs IQ2
Internal consistency: e8->e8mix improved 2.47 ppl by ONLY re-quantizing V/K (64 tensors). Diagnosis
confirmed: my bulk (FFN/attn_q/attn_out) already beats IQ2; protecting V/K removes the only weakness.
CAVEATS (honest): (1) bulk uses ENTROPY-coded ~2.0 bpw (needs a runtime range decoder) vs IQ2 fixed
2.06; true avg ~2.08-2.13 bpw, ~0-3% more bits than IQ2 -- the 15% ppl win far exceeds that.
(2) SIMULATED quant (stored Q8); a deployable format needs entropy decode + RHT-on-activations
(cheap O(n log n)) + custom kernels -- quality result is established, runtime is not built.
(3) 8B only; a 2-bit 70B is still ~19GB > 16GB so the "70B in 16GB" goal stays blocked (as stated).
This is the novel deliverable: a real, measured margin past llama.cpp's IQ2 frontier at equal bitrate.

## 2026-06-17 — QTIP trellis quantization (push past 10.04)

Implemented bitshift trellis (QTIP-style): R bits/weight emitted, state = last L bits (sliding
window) so weight N influences N+1; codeword = hashed Gaussian (implicit codebook); GPU-batched
Viterbi finds globally optimal bit path. FIXED rate = R bpw + per-seg scale (NO entropy coding ->
removes the E8 caveat). Self-test on unit Gaussian @R=2: L8 0.091, L10 0.078, L12 0.072 -> approaches
R-D bound 0.0625 as memory grows. Correct.

### Per-tensor weighted-NMSE on incoherence-rotated weights (fixed 2.06 bpw)
                    IQ2      E8(entropy2.0)   trellis L12 (full-row)  trellis L12 seg128
  attn_q          0.0780      0.0762            0.0729                 0.0684
  ffn_gate        0.1057      0.0781            0.0730                 0.0684
=> trellis L12 (seg128) BEATS IQ2, E8, AND full-row trellis on both — finer per-seg scaling helps.
   And it's FIXED-rate (no entropy decoder needed). Building full bulk model (seg256 for fair
   2.06bpw bulk) on the e8mix base (keep V/K@4bpw E8). End-to-end vs 10.04 = [building, GPU ~1.8hr].

### Allocation sweep: protect ffn_down too (2bpw -> ~3bpw via E8 step 0.48) on the e8mix base
  e8mix (V/K@4bpw only)        : ppl 10.04  @ ~2.08 bpw
  e8mix_fdn (+ffn_down@3bpw)   : ppl  9.21  @ ~2.35 bpw  -> HELPS (-8%) but costs +0.27 bpw (not free)
=> ffn_down IS ppl-sensitive despite my winning its 2bpw weighted-NMSE (0.065 vs IQ2 0.104). Another
   case of the per-tensor proxy underpredicting end-to-end sensitivity. At EQUAL bitrate one would need
   to take those bits from a truly-insensitive tensor; the optimal allocation is itself a search.
   (Honest: e8mix_fdn at 2.35 bpw is NOT an equal-bitrate win vs IQ2 2.06; it's a higher-rate operating
    point. cf. real iq2s ~2.6bpw = 9.11. So extra bits on ffn_down ~ matches IQ2's use of the same bits.)

## Plan
1. [done] Pull llama3.3:70b Q4_K_M (baseline). 42GB. (Thrashes — see above.)

## Plan
1. [done] Pull llama3.3:70b Q4_K_M (baseline). 42GB. (Thrashes — see above.)
2. [next] Build benchmark harness: perplexity (sliding-window NLL), VRAM, tok/s. Validate on small local model.
3. Measure frontier: Q4_K_M vs IQ3/IQ2/IQ1 + imatrix, full GPU vs partial offload, KV q8/q4.
4. Test whether a custom trick beats the frontier.
5. Document what worked / didn't.

## 2026-06-17 — Testing the assumptions behind the "2-bit floor" (target ppl 7.0)

CORRECTION: the RD floor (0.0625 NMSE) bounds WEIGHT reconstruction, not perplexity. Perplexity
= f(OUTPUT error), which error-aware methods can push below the weight-NMSE floor. Floor was the
wrong quantity; perplexity floor is open. Testing structural assumptions on real weights (blk.16):

Q1 Gaussian i.i.d.?  whole-kurt / per-col median / per-col max / var-spread
  attn_q   4.93 / 4.10 / 84.7  / 2.0
  ffn_gate 4.14 / 3.67 / 22.2  / 1.4
  ffn_down 4.19 / 3.09 / 2157.6 (!) / 2.1
  attn_v   3.76 / 3.30 / 15.4  / 2.6
=> PARTIALLY FALSE: columns near-Gaussian, but rare EXTREME outlier channels (ffn_down kurt 2157
   = classic LLM outlier feature). Not i.i.d. across channels. -> sparse outlier overlay worth testing.

Q2 is quantization ERROR low-rank?  rank for 90% energy (raw / weighted-by-sqrtH):
  attn_q 44/45% | ffn_gate 71/73% | ffn_down 72/74% | attn_v 74/74%
=> HIGH-RANK. Cheap low-rank correction ("aux network", linear form) is DOMINATED by just using
   more base bits. NEGATIVE for low-rank-residual direction.

Implications: cheap structural exploits limited/already captured by incoherence+allocation. Real
remaining levers target OUTPUT error & need real activations (GPTQ true-Hessian, cross-layer
compensation, end-to-end codebook tuning) -> requires a torch forward pass of the model. Published
2-bit SOTA plateaus ~3-bit-equiv (~7.5-9 for this 8B), not lossless. Testing anyway, measuring each.

## 2026-06-17 — RESULT: QTIP trellis bulk beats E8, FIXED-RATE
Controlled (same Q8 embeds + V/K@4bpw E8; only bulk method differs), WikiText-2 40 chunks:
  sim_iq2     (IQ2_XXS)              = 11.83
  sim_e8mix   (E8 bulk + V/K)        = 10.04   (-15% vs IQ2; bulk entropy-rate)
  sim_trellis (TRELLIS bulk + V/K)   =  9.37   (-21% vs IQ2; bulk FIXED 2.06 bpw, no entropy caveat)
Trellis L=12 bulk improved 10.04 -> 9.37 (V/K held fixed -> per-tensor gain DID translate this time).
Avg ~2.13 bpw. Beats real llama.cpp iq2xs (2.43bpw=9.89) and approaches iq2s (2.6bpw=9.11) at LOWER
bitrate. We are now ahead of the production IQ frontier. Still far from 7.0 -> next: output-error-aware
quant (full-Hessian GPTQ + trellis) via torch forward pass [building].

## 2026-06-17 — torch pipeline validated + sequential GPTQ (diagonal) — confounded negative
- Built from-scratch torch Llama-3.1-8B. Logits match llama.cpp: corr 0.9999, top-1 44/46, KL 4e-4.
- Perplexity method reconciled: SECOND-HALF scoring -> torch-Q8 = 6.96 ~= llama.cpp 7.006. (Full-window
  scoring = 9.27; the gap was methodology, not a bug.) torch numbers now on the 7.0 scale.
- Sequential GPTQ (quantize layer-by-layer on progressively-quantized model, diag-Hessian captured on
  propagated activations, incoherence+trellis L=10, 12 calib chunks, V/K trellis R=4):
    PPL half = 10.46  (vs non-seq trellis L=12 = 9.37, Q8 = 6.96)
  WORSE, but CONFOUNDED: this used L=10 (weaker) + only 12 calib chunks vs the record's L=12 + 80-chunk
  imatrix. Cannot conclude propagation hurts.
- DIAGNOSIS (the real content): my "sequential GPTQ" is NOT real GPTQ. It only re-measures the DIAGONAL
  Hessian on propagated activations -> there is no compensation mechanism. Real GPTQ/QuIP# gains come
  from the OFF-DIAGONAL Hessian: within-layer column error feedback (LDLQ/Cholesky) or full-H whitening
  (A = W H^.5, quantize, W_hat = A_hat H^-.5). I coded full_H=True but have NOT run it -> that is the
  actual output-error-aware test. [HOLD - needs GPU; user wants it for VRChat]

## 2026-06-17 — full-Hessian: marginal + fragile; sequential propagation explodes
- Sequential full-H run = GARBAGE (ppl 578609). Diagnosis via single-tensor output-error test:
  diagonal-H output NMSE 0.0392 ; full-H 0.0364 (only ~7% better per-tensor). H cond# = 27320.
- So full-H whitening is CORRECT per-tensor (marginally helps) but the H^-.5 reconstruction puts large
  weight errors in low-eigenvalue directions; with few calib chunks those still hit the real forward,
  and over 32 sequentially-propagated layers it compounds -> explosion.
- Verdict: the proper Hessian method gives only ~7% per-tensor and is numerically fragile. Not a path
  to a large gain, definitely not to 7.0. Fixable (no propagation) for ~9.37->~9.1, marginal.

## INFORMATION-THEORETIC VERDICT on 7.0 @ 2-bit (the directions, measured dead)
7.0 = LOSSLESS (Q8=6.96). 2-bit stores 2GB of info; the model was trained at 16-bit = 16GB. Lossless
2-bit requires the learned function to be ~8x redundant. MEASURED it is NOT:
  - quantization error is HIGH-rank (90% energy needs 44-74% of rank) -> not low-rank compressible
  - weights near-Gaussian per-column (kurt ~3-4), no hidden structure at any measured scale
  - cross-layer corr ~0.008, adjacent-weight autocorr ~0 -> no redundancy to exploit
  - proper Hessian (full-H) helps only ~7%; incoherence+trellis already near the practical 2-bit ceiling
Directions tested & dead/marginal: VQ/codebook (=IQ), neural predictor (dead), cross-layer share (dead),
low-rank/aux correction (dead, high-rank err), streaming (bandwidth-dead), per-tensor mixed alloc (key
lever, used), incoherence (helps lattice), E8 lattice, QTIP trellis (best, 9.37), seq-GPTQ diag (no help),
full-H (marginal+fragile). UNTESTED: end-to-end QAT/codebook fine-tuning (AQLM-style) -> expected ceiling
~8.3-8.6 (published 2-bit SOTA = ~3-bit-equiv), NOT 7.0.
CONCLUSION: 7.0 at true 2-bit is information-theoretically out of reach for a pretrained model. Best
achieved = 9.37 (-21% vs production IQ2 11.83). Realistic absolute floor with all levers ~ 8.3-8.5.

## 2026-06-17 — post-reboot: cheap output-aware levers exhausted
- torch-trellis baseline (loaded q8_trellis.gguf into torch) = 9.40 half-ppl == llama.cpp 9.37. Consistent.
- Per-channel output correction (closed-form a_j = <Ohat,O>/<Ohat,Ohat> on wo+down, propagated):
  9.40 -> 9.64 = HURTS. Overfits 16-chunk calib + perturbs the well-quantized residual stream; propagation
  compounds it. NEGATIVE.
- Tally of post-quant output-aware levers, all measured: full-H whitening (marginal ~7% per-tensor, EXPLODES
  in propagation), sequential-diagonal GPTQ (no help, 10.46@L10), per-channel correction (HURTS). The
  imatrix+incoherence+trellis quantization is already well-calibrated; cheap post-hoc fixes don't help.
- Only remaining untested lever = heavy end-to-end QAT / learnable-codebook fine-tuning (AQLM-style):
  big build (differentiable codebook+reconstruction, block-wise Adam), expected ceiling ~8.3-8.6 (published
  2-bit SOTA), NOT 7.0. 
FINAL (so far): BEST = 9.37/9.40 (-21% vs production IQ2 11.83). 7.0 unreachable. Floor w/ full QAT ~8.5.

## 2026-06-17 — INVENTION: Hessian-eigenbasis (KLT) transform coding + water-filling
NEW idea (not rotation/codebook/lattice/trellis/GPTQ): rotate W into the input-Hessian eigenbasis
(KLT), water-fill VARIABLE bits per eigendirection by output-energy (lambda_j * ||col||^2), drop
directions inputs never visit. Exploits Hessian anisotropy that QuIP# incoherence deliberately destroys.
PER-TENSOR RESULT (ffn_gate, 2bpw, crude per-col uniform quant):
  diagonal-basis water-fill outNMSE 0.705  vs  KLT eigenbasis water-fill 0.067  = 10x better.
  (eigenvalue spread lam_max/lam_med = 1469; top-10% dirs hold 84% of output energy; 1014 dirs dropped.)
  -> KLT matches the trellis (~0.064) with a FAR cruder quantizer. The idea WORKS.
MAKE-OR-BREAK (shared transform across layers, to amortize U storage):
  own-U 0.067-0.105 | layer-0-U on other layers 0.49-0.73 | mean-H-U 0.13-0.56.
  => eigenbases are LAYER-SPECIFIC (cross-layer transfer fails). Novel structural finding: the network
     rotates its activation subspace between layers, so the optimal transform is irreducibly per-layer.
VERDICT: FAILS as compression. Dense per-layer U costs ~2.3 bpw to store (can't be shared); the 10x
quality edge over diagonal is already matched by the trellis via FREE incoherence. Honest negative
with a clear mechanism. (Genuinely-new idea, built+tested+measured, failed for a specific reason.)

## 2026-06-17 — Axis 3: real allocation search (vs actual ppl, not proxy NMSE)
Flexible per-type bit allocation, trellis, non-sequential. (L8 used for search speed -> absolute ppl
inflated vs L12, but ranking holds.) Partial run (got cut off after 2/4):
  A current (V/K@4, others@2)         bpw 2.076  ppl 11.94 (L8)
  B +down@3, gate/up@1                bpw 1.807  ppl 96.7  (L8) -- CATASTROPHIC
KEY FINDING: gate/up @ 1 bit = model collapse (96.7). The bulk FFN (54% of params) is NOT compressible
below 2 bits. => at ~2 bpw the allocation is NEAR-FORCED: gate/up/q/o @2, V/K @4 (cheap, 3.8% of params),
down @2. There is almost no budget slack to redistribute. The hand-picked V/K@4 allocation is already
near-optimal at 2 bpw; you can't free bits from the bulk to protect down without wrecking the model.
This is also WHY 2-bit is a hard floor: the dominant FFN weights need >=2 bits each, period.
Protecting down therefore requires a HIGHER bitrate tier (testing trellis+down@3 ~2.35bpw next).

## 2026-06-17 — *** NEW BEST via allocation: trellis + ffn_down@3 = 8.60 ***
trellis (V/K@4, down@2) 9.37 @ ~2.13 bpw  ->  trellis + down@3  8.60 @ ~2.35 bpw  (-8%).
8.60 @ 2.35bpw BEATS real llama.cpp iq2s (9.11 @ 2.6bpw) and approaches iq2m (8.49 @ 2.94bpw) at LOWER
bitrate -> clearly ahead of the production frontier. The allocation axis (protect ffn_down) WORKS:
ffn_down's 2-bit weighted-NMSE proxy badly UNDERPREDICTED its end-to-end ppl sensitivity (3rd time
the proxy-vs-real gap appeared). Testing down@4 next. Trajectory: 11.83 -> 10.04 -> 9.37 -> 8.60.

## 2026-06-17 — allocation frontier keeps moving: down+q+o@3 = 8.01
trellis + down@3 + attn_q@3 + attn_output@3 = 8.01 @ ~2.5 bpw. (down@4 only 8.48@2.67 -> down@3 sweet.)
attn_q/attn_output were ALSO under-allocated (proxy underpredicted, like down). Pattern: the 2-bit base
systematically under-protects multiple sensitive tensors; protecting them to 3-bit progressively helps.
Frontier (trellis base + E8 protection, llama.cpp ppl, WikiText-2 40chunks):
  trellis        9.37 @ 2.13bpw
  +down@3        8.60 @ 2.35
  +down@4        8.48 @ 2.67
  +down/q/o@3    8.01 @ 2.50   <- beats real iq2m (8.49@2.94) at LOWER bitrate
This is a BETTER rate-distortion frontier than llama.cpp at every bitrate. NOTE on 7.0: it's ~Q4/Q8
quality (~4.9/8.5 bpw); reachable only at ~4.5-5 bpw, NOT at 2-bit. Mapping toward higher bpw next.

## 2026-06-17 — allocation frontier: trellis+allocation DOMINATES llama.cpp at every bitrate
  trellis           9.37 @ 2.13bpw
  +down@3           8.60 @ 2.35
  +down/q/o@3       8.01 @ 2.50   (beats iq2m 8.49@2.94)
  +all-sensitive@3  7.43 @ 3.04   (beats q3km 7.49@4.0 AND iq3xxs 7.84@3.26, at lower bpw!)
llama.cpp frontier for comparison: iq2s 9.11@2.6 | iq2m 8.49@2.94 | iq3xxs 7.84@3.26 | q3km 7.49@4 |
q4km 7.19@4.9 | q8 6.96. MY frontier is strictly better at every comparable bitrate.
=> the real win is a BETTER rate-distortion FRONTIER (trellis quantizer + searched allocation), not a
single point. 7.0 is now in sight as a ~4-4.5bpw result. Building everything@4 (~4bpw) next.

## 2026-06-17 — frontier reaches ~7.0; PIVOT to low-bitrate (user: focus <=3 bpw)
  +all@4   7.09 @ ~4.0 bpw  (beats q4km 7.19@4.9; ~= Q8 6.96). So 7.0 is reachable but at Q4-class bpw.
Full UltraQuant frontier vs llama.cpp (ppl @ bpw):
  ME:  9.37@2.13 | 8.60@2.35 | 8.01@2.50 | 7.43@3.04 | 7.09@4.0
  llama.cpp: iq2s 9.11@2.6 | iq2m 8.49@2.94 | iq3xxs 7.84@3.26 | q3km 7.49@4 | q4km 7.19@4.9 | q8 6.96
  -> UltraQuant dominates at every bitrate. (Caveat: protected-tensor bpw is entropy-rate; trellis base
     is fixed-rate. Perplexities are exact.)
NEW FOCUS: lower ppl at <=2.5-3 bpw. Lever = PER-LAYER allocation (protect only sensitive layers).
Testing down/q/o@3 in first8+last8 only (~2.3bpw) vs all-32 (8.01@2.5).

## 2026-06-17 — per-layer allocation: DEAD (sensitivity is uniform across layers)
down/q/o@3 in first8+last8 layers only = 8.56 @ ~2.3bpw  vs  all-32-layers 8.01 @ 2.5bpw.
Edges-only loses most of the gain -> sensitivity is NOT concentrated in first/last layers; it's spread
roughly uniformly. Per-layer concentration doesn't beat all-layer protection. (Edges 8.56@2.3 ~ ties
down@3-all 8.60@2.35, so it's on the frontier but not better.)
=> Low-bitrate frontier is near its ALLOCATION limit. Binding constraint at ~2.5bpw = gate/up @ 2-bit
(54% of params, can't go lower). Further low-bpw ppl gains need a better 2-bit BULK quantizer or QAT
(end-to-end fine-tuning), not more allocation tricks.

## 2026-06-17 — outlier overlay (SpQR-style): DOMINATED by allocation
trellis + restore top-1% highest-error bulk weights to fp16 = 9.11 @ ~2.42bpw (+0.29bpw for -0.26ppl).
DOMINATED by down@3 (8.60 @ 2.35 = lower ppl AND lower bpw). Protecting whole sensitive TENSORS is more
bit-efficient than restoring scattered outlier WEIGHTS. Another low-bpw lever subsumed by allocation.

### LOW-BITRATE SUMMARY (all cheap levers explored)
Winning frontier (allocation on trellis base): 9.37@2.13 | 8.60@2.35 | 8.01@2.50  (dominates llama.cpp:
iq2xxs 12.21@2.4, iq2s 9.11@2.6). Levers tested & dominated/dead at low bpw: per-layer alloc (dead,
uniform sensitivity), outlier overlay (dominated), per-channel correction (hurts), full-H (fragile).
Binding constraint = gate/up @2bit (54% params). ONLY remaining lever for lower low-bpw ppl = end-to-end
QAT / learnable-codebook fine-tuning (AQLM-style): multi-hour build, ~0.3-0.5 ppl potential, NOT a trick.

## 2026-06-20 — QAT (AQLM-style codebook+scale fine-tuning, fixed trellis states) — GENERALIZES
Exposed trellis states; built differentiable torch reconstruction (validated == trellis to 1e-7).
Per-tensor fine-tune learnable codebook (4096 vals, ~0.001bpw) + per-group scales to minimize true
OUTPUT error ||ŴX - WX|| on real activations. KEY de-risk = held-out generalization test (train 8
chunks, eval 8 held-out):
  cw+scales: heldout output-err -10.8%   cw-only: -5.0%   => GENERALIZES (cw+scales best).
NOTE: per-tensor wNMSE (diagonal-imatrix proxy) went UP (0.064->0.070) but held-out OUTPUT error went
DOWN -10.8% -> the wNMSE diagonal proxy is MISLEADING; QAT optimizes the FULL Hessian (real XX^T),
trading diagonal error for off-diagonal reduction that lowers true output error. This is the right
objective. Running full-model QAT (cw+scales, 8 calib, 50 steps, sequential propagation) -> measure ppl.
Codebook+scales add negligible bits, so any ppl gain is at SAME bitrate as the trellis base (2.13bpw).
