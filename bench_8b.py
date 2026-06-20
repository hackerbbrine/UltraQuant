#!/usr/bin/env python
"""Benchmark every 8B variant (+ Q8 reference) -> results/frontier.jsonl."""
import os, glob, subprocess, sys

ROOT = r"C:\Users\Tony Stark\Documents\UltraQuant"
SRC  = os.path.join(ROOT, "models", "Llama-3.1-8B-Q8_0.gguf")
OUTD = os.path.join(ROOT, "models", "8b")

def bench(path, label, ngl=99, chunks=40):
    cmd = [sys.executable, os.path.join(ROOT, "gpu_bench.py"),
           "--model", path, "--label", label, "--ngl", str(ngl), "--chunks", str(chunks)]
    print("\n>>>", label, flush=True)
    subprocess.run(cmd)

def main():
    bench(SRC, "q8")
    for f in sorted(glob.glob(os.path.join(OUTD, "*.gguf"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem == "imatrix":
            continue
        bench(f, stem)

if __name__ == "__main__":
    main()
