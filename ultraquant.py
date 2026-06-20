#!/usr/bin/env python
"""UltraQuant — terminal UI for quantizing and benchmarking LLMs below the production frontier.

    python ultraquant.py            # interactive TUI
    python ultraquant.py --frontier # print the rate-distortion frontier and exit

Orchestrates the UltraQuant pipeline (trellis base -> searched allocation) and llama.cpp perplexity.
Quality presets map to validated points on the Llama-3.1-8B frontier.
"""
import os, sys, glob, subprocess, time, re
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.live import Live

C = Console()
ROOT = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(ROOT, "models", "8b_sim")
TRELLIS_BASE = os.path.join(SIM, "q8_trellis.gguf")

# preset -> (label, ~bpw, ~ppl, buildalloc protection args on the trellis base)
PRESETS = {
    "1": ("Extreme  — smallest", 2.1, 9.37, []),
    "2": ("Balanced — recommended", 2.5, 8.01, ["down:3", "attn_q:3", "attn_output:3"]),
    "3": ("Quality  — near 3-bit", 3.0, 7.43, ["down:3", "attn_q:3", "attn_output:3", "gate:3", "up:3"]),
    "4": ("Max      — near lossless", 4.0, 7.09, ["down:4", "attn_q:4", "attn_output:4", "gate:4", "up:4"]),
}
# llama.cpp comparison points (bpw, ppl) for the frontier view
LLAMACPP = [(2.39, 12.21, "iq2xxs"), (2.6, 9.89, "iq2xs"), (2.75, 9.11, "iq2s"),
            (2.94, 8.49, "iq2m"), (3.26, 7.84, "iq3xxs"), (4.0, 7.49, "q3km"), (4.9, 7.19, "q4km")]
UQ = [(2.13, 9.37), (2.35, 8.60), (2.50, 8.01), (3.04, 7.43), (4.0, 7.09)]

def banner():
    C.print(Panel(Text("UltraQuant", style="bold cyan", justify="center") +
                  Text("\nsub-4-bit LLM quantization that beats the production frontier",
                       style="dim", justify="center"),
                  border_style="cyan"))

def frontier_table():
    t = Table(title="Rate-distortion frontier (Llama-3.1-8B, WikiText-2 ppl; lossless Q8 = 6.96)",
              header_style="bold")
    t.add_column("bits/weight", justify="right"); t.add_column("UltraQuant", justify="right", style="green")
    t.add_column("llama.cpp", justify="right", style="blue"); t.add_column("verdict", style="dim")
    rows = [(2.1, 9.37, "12.21 (iq2xxs)", "-23%"), (2.5, 8.01, "9.11 (iq2s @2.6)", "-12%"),
            (3.0, 7.43, "7.84 (iq3xxs @3.3)", "-5%, & beats q3km@4"), (4.0, 7.09, "7.19 (q4km @4.9)", "~lossless")]
    for bpw, uq, lc, v in rows:
        t.add_row(f"{bpw:.1f}", f"{uq:.2f}", lc, v)
    return t

def find_sources():
    pats = [os.path.join(ROOT, "models", "*.gguf"), os.path.join(ROOT, "models", "*", "*.gguf")]
    out = []
    for p in pats:
        for f in glob.glob(p):
            if "8b_sim" in f: continue
            out.append(f)
    return sorted(set(out))

def run_stage(label, cmd):
    """Run a pipeline subprocess, streaming a live tail under a status line."""
    C.print(f"[bold]> {label}[/bold]  [dim]{' '.join(os.path.basename(c) for c in cmd[:2])}[/dim]")
    t0 = time.time(); tail = []
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    with Live(console=C, refresh_per_second=4, transient=True) as live:
        for line in p.stdout:
            line = line.rstrip()
            if not line or "offload-arch" in line: continue
            tail = (tail + [line])[-6:]
            live.update(Panel("\n".join(tail), title=f"{label}  ({time.time()-t0:.0f}s)",
                              border_style="cyan", height=8))
    p.wait()
    ok = p.returncode == 0
    C.print(f"  {'[green][ok][/green]' if ok else '[red][x][/red]'} {label}  ({time.time()-t0:.0f}s)")
    return ok

def ensure_base():
    if os.path.exists(TRELLIS_BASE):
        return True
    C.print("[yellow]Trellis base not found — building the full pipeline (one-time, ~1 hr on GPU).[/yellow]")
    if Prompt.ask("Proceed?", choices=["y", "n"], default="y") != "y":
        return False
    stages = [("imatrix + IQ baselines", [sys.executable, "build_8b.py"]),
              ("E8 + V/K allocation base", [sys.executable, "buildmix.py"]),
              ("QTIP trellis base", [sys.executable, "buildtrellis.py"])]
    for label, cmd in stages:
        if not run_stage(label, cmd):
            C.print(f"[red]Stage failed: {label}[/red]"); return False
    return os.path.exists(TRELLIS_BASE)

def quantize_flow():
    srcs = find_sources()
    if not srcs:
        C.print("[red]No source GGUF found in models/. Add an 8B Q8_0 GGUF first.[/red]"); return
    C.print("\n[bold]Source models[/bold]")
    for i, s in enumerate(srcs): C.print(f"  [cyan]{i}[/cyan]  {os.path.relpath(s, ROOT)}")
    Prompt.ask("Pick source", choices=[str(i) for i in range(len(srcs))], default="0")
    C.print("\n[bold]Quality preset[/bold]")
    t = Table(show_header=True, header_style="bold")
    t.add_column("#"); t.add_column("preset"); t.add_column("~bpw", justify="right"); t.add_column("~ppl", justify="right")
    for k, (lab, bpw, ppl, _) in PRESETS.items(): t.add_row(k, lab, f"{bpw:.1f}", f"{ppl:.2f}")
    C.print(t)
    k = Prompt.ask("Pick preset", choices=list(PRESETS), default="2")
    lab, bpw, ppl, alloc = PRESETS[k]
    if not ensure_base(): return
    out = os.path.join(SIM, f"q8_uq_{k}.gguf")
    cmd = [sys.executable, "buildalloc.py", out] + alloc if alloc else \
          [sys.executable, "-c", f"import shutil;shutil.copyfile(r'{TRELLIS_BASE}',r'{out}')"]
    if run_stage(f"Quantize -> {lab} (~{bpw} bpw)", cmd):
        C.print(Panel(f"[green]Done.[/green]  {os.path.relpath(out, ROOT)}\n"
                      f"target ~{bpw} bpw, expected ppl ~{ppl}", border_style="green"))
        if Prompt.ask("Benchmark perplexity now? (needs GPU)", choices=["y", "n"], default="n") == "y":
            benchmark_flow(out)

def benchmark_flow(model=None):
    if model is None:
        ms = sorted(glob.glob(os.path.join(SIM, "*.gguf")))
        if not ms: C.print("[red]No quantized models in models/8b_sim/.[/red]"); return
        for i, s in enumerate(ms): C.print(f"  [cyan]{i}[/cyan]  {os.path.basename(s)}")
        model = ms[int(Prompt.ask("Pick model", choices=[str(i) for i in range(len(ms))], default="0"))]
    cmd = [sys.executable, "gpu_bench.py", "--model", model, "--label", os.path.basename(model),
           "--ngl", "99", "--chunks", "40", "--no-bench"]
    run_stage(f"Perplexity: {os.path.basename(model)}", cmd)

def main():
    if "--frontier" in sys.argv:
        C.print(frontier_table()); return
    banner()
    while True:
        C.print("\n[bold]Menu[/bold]  [cyan]1[/cyan] quantize   [cyan]2[/cyan] benchmark   "
                "[cyan]3[/cyan] frontier   [cyan]q[/cyan] quit")
        ch = Prompt.ask(">", choices=["1", "2", "3", "q"], default="1")
        if ch == "1": quantize_flow()
        elif ch == "2": benchmark_flow()
        elif ch == "3": C.print(frontier_table())
        else: break

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        C.print("\n[dim]bye[/dim]")
