#!/usr/bin/env python3
"""92-latency.py — median / MAD for the paired-alternating -c N runs (reports/timings.md rules).

usage: 92-latency.py board/latency-fc.tsv [board/latency-dec.tsv ...]

Per product: n, median ms/inf, MAD, min, max. Then the paired ratio per round against p0 --
alternating buys us an apples-to-apples ratio inside each round, which is the only cross-run
comparison timings.md allows.
"""
import sys, statistics as st

def load(path):
    rows = []
    with open(path) as f:
        f.readline()
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 4 or p[3] in ("", "NA"):
                continue
            rows.append(dict(round=int(p[0]), product=p[1], total_ms=float(p[2]), per=float(p[3])))
    return rows

for path in sys.argv[1:]:
    rows = load(path)
    print("== %s (n=%d) ==" % (path, len(rows)))
    if not rows:
        print("  no rows\n")
        continue
    prods = sorted({r["product"] for r in rows}, key=lambda s: s[-1])
    by = {}
    for r in rows:
        by.setdefault(r["product"], []).append(r)
    med = {}
    print("  %-12s %3s %10s %8s %8s %8s" % ("product", "n", "median_ms", "MAD", "min", "max"))
    for p in prods:
        v = [r["per"] for r in by[p]]
        m = st.median(v)
        med[p] = m
        print("  %-12s %3d %10.4f %8.4f %8.4f %8.4f" % (
            p, len(v), m, st.median([abs(x - m) for x in v]), min(v), max(v)))
    base = med[prods[0]]
    print("  median ratios vs %s: %s" % (prods[0], "  ".join("%s=%.3f" % (p, med[p] / base) for p in prods)))
    rounds = sorted({r["round"] for r in rows})
    print("  per-round ratio vs %s:" % prods[0])
    for r in rounds:
        b = [x["per"] for x in by[prods[0]] if x["round"] == r]
        if not b:
            continue
        parts = []
        for p in prods[1:]:
            v = [x["per"] for x in by[p] if x["round"] == r]
            if v:
                parts.append("%s=%.3f" % (p, v[0] / b[0]))
        print("    round%d base=%.4f  %s" % (r, b[0], "  ".join(parts)))
    print()
