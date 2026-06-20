#!/usr/bin/env python
"""UltraQuant GPU benchmark driver (uses prebuilt llama.cpp Vulkan tools).

For a given GGUF:
  - perplexity  via llama-perplexity (GPU)            -> ppl +/- err
  - speed       via llama-bench (GPU)                 -> pp t/s, tg t/s
  - VRAM        parsed from llama.cpp load-tensor logs -> per-device buffer MB

Appends one JSON record per run to results/frontier.jsonl.
"""
import argparse, json, os, re, subprocess, sys

TOOLS = r"C:\Users\Tony Stark\Documents\UltraQuant\tools\vulkan"
PPL_EXE   = os.path.join(TOOLS, "llama-perplexity.exe")
BENCH_EXE = os.path.join(TOOLS, "llama-bench.exe")

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (p.stdout or "") + "\n" + (p.stderr or "")

def parse_vram(log):
    """Sum Vulkan0 buffers (model + KV + compute) = VRAM actually used on dGPU."""
    dev = {}
    # e.g. "load_tensors:      Vulkan0 model buffer size = 12345.67 MiB"
    for m in re.finditer(r"(Vulkan0|Vulkan1|CPU|CPU_Mapped)\s+(?:model|KV|compute|.*?)\s*buffer size\s*=\s*([\d.]+)\s*MiB", log):
        dev.setdefault(m.group(1), 0.0)
        dev[m.group(1)] += float(m.group(2))
    off = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers", log)
    return dev, (off.groups() if off else None)

def parse_ppl(log):
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", log)
    if m:
        return float(m.group(1)), float(m.group(2))
    # fallback: last "[NN]X.XXXX," running value
    vals = re.findall(r"PPL\s*=\s*([\d.]+)", log)
    return (float(vals[-1]), None) if vals else (None, None)

def parse_bench(log):
    pp = tg = None
    for line in log.splitlines():
        if "|" not in line:
            continue
        if re.search(r"\bpp\d+\b", line):
            mm = re.findall(r"([\d.]+)\s*(?:±|\+/-)", line)
            if mm: pp = float(mm[-1])
        if re.search(r"\btg\d+\b", line):
            mm = re.findall(r"([\d.]+)\s*(?:±|\+/-)", line)
            if mm: tg = float(mm[-1])
    return pp, tg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--nctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--corpus", default=r"data\wiki.test.raw")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--no-ppl", action="store_true")
    args = ap.parse_args()

    size_gb = round(os.path.getsize(args.model)/1024**3, 3)
    rec = dict(label=args.label, model=os.path.basename(args.model),
               file_gb=size_gb, ngl=args.ngl, nctx=args.nctx)

    if not args.no_ppl:
        print(f"[ppl] {args.label} ...", flush=True)
        log = run([PPL_EXE, "-m", args.model, "-f", args.corpus, "-ngl", str(args.ngl),
                   "-dev", "Vulkan0", "-c", str(args.nctx), "--chunks", str(args.chunks)])
        ppl, err = parse_ppl(log)
        dev, off = parse_vram(log)
        vram = round(dev.get("Vulkan0", float("nan")), 1)
        rec.update(ppl=ppl, ppl_err=err, vram_mb=vram,
                   vram_breakdown={k: round(v,1) for k,v in dev.items()},
                   offload=("/".join(off) if off else None))
        print(f"  ppl={ppl} +/-{err}  VRAM(Vulkan0)={vram}MB  offload={rec['offload']}", flush=True)

    if not args.no_bench:
        print(f"[bench] {args.label} ...", flush=True)
        log = run([BENCH_EXE, "-m", args.model, "-ngl", str(args.ngl),
                   "-dev", "Vulkan0", "-p", "256", "-n", "64"])
        pp, tg = parse_bench(log)
        rec.update(pp_tok_s=pp, tg_tok_s=tg)
        print(f"  pp={pp} t/s  tg={tg} t/s", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(r"results\frontier.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print("[rec]", json.dumps(rec), flush=True)

if __name__ == "__main__":
    main()
