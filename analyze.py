#!/usr/bin/env python
"""Read results/frontier.jsonl, compute the perplexity<->size<->speed frontier,
mark Pareto-optimal points, and write results/summary.md."""
import json, os, sys

PARAMS = {"8b": 8.03e9, "70b": 70.6e9}

def load(path=r"results\frontier.jsonl"):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs

def bpw(file_gb, family):
    return file_gb * 1024**3 * 8 / PARAMS[family]

def main():
    family = sys.argv[1] if len(sys.argv) > 1 else "8b"
    ref_label = sys.argv[2] if len(sys.argv) > 2 else "q8"
    recs = [r for r in load() if r.get("ppl")]
    # dedupe by label, keep last
    by = {}
    for r in recs:
        by[r["label"]] = r
    recs = list(by.values())
    ref = by.get(ref_label)
    ref_ppl = ref["ppl"] if ref else min(r["ppl"] for r in recs)

    rows = []
    for r in recs:
        b = bpw(r["file_gb"], family)
        inc = 100*(r["ppl"]-ref_ppl)/ref_ppl
        rows.append(dict(label=r["label"], gb=r["file_gb"], bpw=b, ppl=r["ppl"],
                         inc=inc, tg=r.get("tg_tok_s"), vram=r.get("vram_mb"),
                         fits=(r.get("vram_mb") or 1e9) < 15500))
    rows.sort(key=lambda x: x["gb"])

    # Pareto: minimize (gb, ppl)
    pareto = []
    best_ppl = 1e9
    for r in sorted(rows, key=lambda x: x["gb"]):
        if r["ppl"] < best_ppl - 1e-9:
            pareto.append(r["label"]); best_ppl = r["ppl"]

    lines = ["# UltraQuant frontier — %s (ref=%s, ppl=%.4f)\n" % (family, ref_label, ref_ppl),
             "| variant | GB | bpw | PPL | dPPL%% | tg tok/s | VRAM MB | <16GB | Pareto |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %.3f | %.2f | %.4f | %+.1f%% | %s | %s | %s | %s |" % (
            r["label"], r["gb"], r["bpw"], r["ppl"], r["inc"],
            ("%.1f"%r["tg"]) if r["tg"] else "-",
            ("%.0f"%r["vram"]) if r["vram"] else "-",
            "yes" if r["fits"] else "no",
            "*" if r["label"] in pareto else ""))
    out = "\n".join(lines) + "\n"
    os.makedirs("results", exist_ok=True)
    with open(r"results\summary.md", "w", encoding="utf-8") as f:
        f.write(out)
    print(out)

if __name__ == "__main__":
    main()
