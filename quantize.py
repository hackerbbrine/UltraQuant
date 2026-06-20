#!/usr/bin/env python
"""UltraQuant quantization engine — wraps prebuilt llama.cpp tools.

  - imatrix(): build importance matrix from a calibration corpus (GPU)
  - quantize(): produce a GGUF at a target type, with optional per-tensor
                overrides (--tensor-type) for budget-constrained mixed precision
  - dry_run sizing to hit an exact VRAM budget

Recipes (uniform + custom 'UltraQuant-Mix') are defined in RECIPES below.
"""
import os, re, subprocess, sys, argparse

TOOLS = r"C:\Users\Tony Stark\Documents\UltraQuant\tools\vulkan"
QUANT = os.path.join(TOOLS, "llama-quantize.exe")
IMAT  = os.path.join(TOOLS, "llama-imatrix.exe")

def sh(cmd, capture=True):
    print(">", " ".join(os.path.basename(c) if c.endswith('.exe') else c for c in cmd), flush=True)
    if capture:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return (p.stdout or "") + "\n" + (p.stderr or ""), p.returncode
    else:
        p = subprocess.run(cmd)
        return "", p.returncode

def build_imatrix(src, corpus, out, ngl=99, chunks=80):
    if os.path.exists(out):
        print(f"[imatrix] exists: {out}")
        return out
    log, rc = sh([IMAT, "-m", src, "-f", corpus, "-o", out,
                  "-ngl", str(ngl), "-dev", "Vulkan0", "--chunks", str(chunks)])
    if rc != 0 or not os.path.exists(out):
        print(log[-2000:]); raise RuntimeError("imatrix failed")
    print(f"[imatrix] wrote {out}")
    return out

def quantize(src, out, qtype, imatrix=None, tensor_types=None,
             output_type=None, embd_type=None, dry_run=False, allow_requant=True):
    cmd = [QUANT]
    if allow_requant: cmd.append("--allow-requantize")
    if imatrix: cmd += ["--imatrix", imatrix]
    if output_type: cmd += ["--output-tensor-type", output_type]
    if embd_type: cmd += ["--token-embedding-type", embd_type]
    for tt in (tensor_types or []):
        cmd += ["--tensor-type", tt]
    if dry_run: cmd.append("--dry-run")
    cmd += [src, out, qtype]
    log, rc = sh(cmd, capture=True)
    if dry_run:
        # parse predicted size if present
        m = re.search(r"=\s*([\d.]+)\s*([GM])iB", log)
        print(log[-1500:])
        return log
    if rc != 0 or not os.path.exists(out):
        print(log[-3000:]); raise RuntimeError(f"quantize failed: {qtype}")
    gb = os.path.getsize(out)/1024**3
    print(f"[quant] {qtype} -> {out}  {gb:.3f} GB")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--imatrix", default=None)
    ap.add_argument("--tensor-type", action="append", default=[])
    ap.add_argument("--output-type", default=None)
    ap.add_argument("--embd-type", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    quantize(a.src, a.out, a.type, imatrix=a.imatrix, tensor_types=a.tensor_type,
             output_type=a.output_type, embd_type=a.embd_type, dry_run=a.dry_run)
