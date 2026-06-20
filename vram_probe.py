#!/usr/bin/env python
"""Measure actual peak dGPU VRAM used by a llama.cpp model load+run.

Device-level (hipMemGetInfo via torch) -> build-independent, no log parsing.
Launches llama-cli holding the model, samples free VRAM, peak = idle_free - min_free.
"""
import subprocess, sys, time, argparse, os
import torch

TOOLS = r"C:\Users\Tony Stark\Documents\UltraQuant\tools\vulkan"
CLI = os.path.join(TOOLS, "llama-cli.exe")

def free_mb():
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0] / 1024**2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--nctx", type=int, default=512)
    ap.add_argument("--gen", type=int, default=300)
    ap.add_argument("--max-wait", type=float, default=180.0)
    a = ap.parse_args()

    # warm torch ctx, then idle baseline
    _ = torch.zeros(1, device="cuda")
    time.sleep(0.5)
    idle = free_mb()
    print(f"idle free VRAM: {idle:.0f} MB", flush=True)

    proc = subprocess.Popen(
        [CLI, "-m", a.model, "-ngl", str(a.ngl), "-dev", "Vulkan0",
         "-c", str(a.nctx), "-n", str(a.gen), "-p", "The history of computing is",
         "-no-cnv", "--no-warmup"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    min_free = idle
    t0 = time.time()
    while time.time() - t0 < a.max_wait:
        if proc.poll() is not None:
            break
        f = free_mb()
        if f < min_free:
            min_free = f
        time.sleep(0.3)
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=10)
        except Exception: proc.kill()

    used = idle - min_free
    print(f"min free VRAM: {min_free:.0f} MB", flush=True)
    print(f"PEAK VRAM USED by model: {used:.0f} MB  ({used/1024:.2f} GB)", flush=True)
    fits = used < (15.92*1024)
    print(f"fits in 16GB VRAM (resident): {'YES' if fits else 'NO'}", flush=True)

if __name__ == "__main__":
    main()
