# UltraQuant

**Pushing LLM weight quantization below the production rate–distortion frontier on a 16 GB consumer GPU.**

UltraQuant is a research codebase that explores sub-4-bit quantization of LLM weights and produces a
**perplexity ↔ bits-per-weight frontier that beats llama.cpp's IQ/K quants at every bitrate tested** on
Llama-3.1-8B. It combines incoherence processing, an E8 lattice / QTIP-style trellis quantizer, a
searched per-tensor bit allocation, and AQLM-style codebook fine-tuning — implemented from scratch and
validated against a from-scratch PyTorch forward pass that matches llama.cpp logits to 0.9999 correlation.

> **Honesty first:** every perplexity number here is *measured* end-to-end through the real inference
> engine. The bitrates for fine-protected tensors are entropy-coded estimates (see [Caveats](#caveats)).
> This is a research result on an 8B testbed, not a shipped inference runtime.

---

## Headline result

WikiText-2 perplexity (second-half scoring, matches `llama-perplexity`) vs. bits/weight. Lower-left is better.
Lossless reference: Q8_0 = **6.96**.

| bits/weight | **UltraQuant** | llama.cpp (IQ/K) |
|---|---|---|
| ~2.1 | **9.37** | iq2xxs 12.21 |
| ~2.5 | **8.01** | iq2s 9.11 (@2.6) |
| ~2.9–3.0 | **7.43** (@3.0) | iq2m 8.49 / iq3xxs 7.84 (@3.26) |
| ~4.0 | **7.09** | q3km 7.49 / q4km 7.19 (@4.9) |

UltraQuant's curve sits strictly below llama.cpp's: e.g. **8.01 ppl @ ~2.5 bpw beats their iq2m
(8.49 @ 2.94)** and **7.43 @ 3 bpw beats their q3km (7.49 @ 4 bpw)** — both at *lower* bitrate.

The single biggest lever was a **real bit-allocation search** (not hand-tuned): the 2-bit base
systematically under-protects `ffn_down`, `attn_q`, and `attn_output`; restoring them to ~3-bit drops
perplexity 9.37 → 8.01 at ~2.5 bpw.

## Quick start

```bash
pip install -r requirements.txt
python ultraquant.py            # interactive TUI: quantize / benchmark / view frontier
python ultraquant.py --frontier # just print the rate-distortion table
```

The TUI (`rich`-based) detects source GGUFs, offers quality presets mapped to the validated frontier
points below, runs the pipeline with live progress, and can measure perplexity. See
[Setup](#setup) for the external artifacts it needs (model + llama.cpp binaries).

## Key ideas

- **Incoherence processing** — randomized Hadamard rotation before quantization, to Gaussianize weights
  and spread outliers (QuIP#-style). Confirmed to dramatically help the lattice quantizer.
- **E8 lattice + QTIP trellis** — quantize to the densest 8-D lattice (closed-form Conway–Sloane decode)
  and a bitshift trellis with Viterbi decoding (fixed-rate, no entropy coding needed).
- **Importance-aware, searched bit allocation** — protect the tensors that actually move perplexity,
  found by searching against *real* end-to-end perplexity rather than a reconstruction-error proxy.
- **AQLM-style codebook fine-tuning** — fix the trellis assignments, make the per-tensor codebook +
  scales learnable, and fine-tune them to minimize true output error (validated to generalize on
  held-out data). Adds ~0.001 bpw, so gains come at the *same* bitrate.

See **[FINDINGS.md](FINDINGS.md)** for the full timestamped research log and **[RESULTS.md](RESULTS.md)**
for the consolidated results write-up.

## Caveats

- **Simulated quantization.** Quality is measured by reconstructing `Ŵ` and running it through the real
  engine — the *perplexity is exact*. There is no deployable runtime (a real one needs entropy/lattice
  decode + the rotation applied to activations + custom ROCm/Vulkan kernels).
- **Entropy-coded bitrates for protected tensors.** The trellis base is fixed-rate; the E8-protected
  tensors use entropy-rate estimates. A fixed-rate deployment would be somewhat higher bpw.
- **8B testbed.** All numbers are Llama-3.1-8B. The original "run a dense 70B in 16 GB at <15% loss" goal
  is shown to be **information-theoretically blocked** at 2-bit (the bulk FFN can't go below ~2 bits
  without collapse); see FINDINGS. 7.0 perplexity is reachable, but only at ~Q4-class (~4–5) bpw.

## Hardware

Developed on: AMD RX 9070 XT (16 GB, RDNA4, ROCm 7.2 / Vulkan), Ryzen 7 9800X3D, 32 GB DDR5, Windows 11.
GPU work runs through prebuilt llama.cpp Vulkan binaries and PyTorch-ROCm.

## Setup

```bash
pip install -r requirements.txt           # see file for the ROCm torch note
```

Then provide the external artifacts (git-ignored — see `.gitignore`):

- `tools/vulkan/` — prebuilt llama.cpp Vulkan binaries (release **b9672+**): `llama-perplexity`,
  `llama-bench`, `llama-quantize`, `llama-imatrix`, `llama-cli`.
- `models/Llama-3.1-8B-Q8_0.gguf` — near-lossless source/reference (e.g. from bartowski on HF).
- `data/` — WikiText-2 train (calibration) and test (eval); the scripts auto-fetch and cache tokens.

## Reproduce

```bash
# 1. Build the IQ-quant baselines + the imatrix
python build_8b.py
# 2. Build UltraQuant variants (trellis bulk + searched allocation)
python buildtrellis.py
python buildalloc.py models/8b_sim/q8_ultraquant.gguf down:3 attn_q:3 attn_output:3
# 3. Measure perplexity (GPU, llama.cpp)
python gpu_bench.py --model models/8b_sim/q8_ultraquant.gguf --label uq --ngl 99 --chunks 40 --no-bench
# 4. Pareto analysis
python analyze.py 8b q8
# 5. (optional) AQLM-style codebook fine-tuning
python qat.py 8 50
```

## Repository layout

Core modules (reusable):

| file | purpose |
|---|---|
| `model.py` | from-scratch PyTorch Llama-3.1-8B forward (logits match llama.cpp 0.9999) |
| `qlab.py` | quantization primitives: incoherence/RHT, E8 lattice, imatrix-weighted metric |
| `trellis.py` | QTIP-style bitshift trellis + GPU-batched Viterbi |
| `gptq.py` | sequential quantization, Hessian capture, `quant_wt` |
| `qat.py` | AQLM-style codebook+scale fine-tuning (fixed states, differentiable reconstruct) |
| `quantize.py` | thin wrapper over llama.cpp `quantize` / `imatrix` |

Builders (produce quantized GGUFs): `build_8b.py`, `buildq.py`, `buildtrellis.py`, `buildmix.py`,
`buildalloc.py`, `buildfdn.py`, `buildoutlier*.py`.

Evaluators: `gpu_bench.py` (ppl/speed/VRAM), `bench.py` (CPU ppl), `analyze.py` (Pareto),
`pplcheck.py` / `logitcheck.py` (torch-vs-llama.cpp validation).

Research labs (one-off analyses, kept for reproducibility): `wlab.py` (rate–distortion),
`klt.py` / `klt_shared.py` (KLT transform-coding *invention attempt*), `distlab.py` (weight
distribution + error rank), `predlab.py` (predictor/redundancy), `anchor.py`, `bitrate.py`,
`bwtest.py`, `vram_probe.py`, `correct.py`, `fhcheck.py`, and the `check*.py` probes.

## What didn't work (negative results — also data)

Documented in FINDINGS.md with numbers: hand-picked per-tensor mixes (dominated by IQ); per-channel
correction (overfits/hurts); low-rank/auxiliary-network correction (error is high-rank); cross-layer
weight sharing (corr ≈ 0); neural weight prediction (autocorrelation ≈ 0); per-token NVMe streaming
(bandwidth-bound); and a **KLT (Hessian-eigenbasis) transform-coding invention** that works per-tensor
(10× better than diagonal) but fails as a scheme because the optimal transform is irreducibly
per-layer (cross-layer eigenbasis overlap ≈ 0) and too large to amortize.

## License

[Apache 2.0](LICENSE).
