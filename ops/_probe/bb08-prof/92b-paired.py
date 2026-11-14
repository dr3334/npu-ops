#!/usr/bin/env python3
"""92b-paired.py — the paired statistic 92-latency.py does not print.

usage: 92b-paired.py board/08a/latency-fc.tsv [board/08a/latency-dec.tsv ...]

92-latency.py gives per-product medians and ratio-of-medians.  Ratio-of-medians mixes
"drift between rounds" into the answer, so here the ratio is formed INSIDE each round
(p<k> and p0 measured seconds apart in the same round) and only then summarised:

  median of the per-round ratios, its MAD, how many rounds went the "slower" way,
  and the exact two-sided sign test on that count.

Sign test is the honest floor: with n rounds it cannot resolve anything smaller than
the round-to-round scatter, which is exactly the claim we want to make or refuse.
"""
import sys, statistics as st
from math import comb

def load(path):
    rows = []
    with open(path) as f:
        f.readline()
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 4 or p[3] in ("", "NA"):
                continue
            rows.append(dict(round=int(p[0]), product=p[1], per=float(p[3])))
    return rows

def sign_p(k, n):
    """exact two-sided sign test, p=0.5"""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)

for path in sys.argv[1:]:
    rows = load(path)
    prods = sorted({r["product"] for r in rows}, key=lambda s: s[-1])
    rounds = sorted({r["round"] for r in rows})
    val = {(r["round"], r["product"]): r["per"] for r in rows}
    base = prods[0]
    print("== %s — paired inside each round vs %s (n=%d rounds) ==" % (path, base, len(rounds)))
    print("  %-10s %6s %8s %8s %8s %5s %7s" % ("product", "rounds", "med_ratio", "MAD", "min", "slower", "sign_p"))
    for p in prods[1:]:
        ratios = []
        for r in rounds:
            b, v = val.get((r, base)), val.get((r, p))
            if b and v:
                ratios.append(v / b)
        if not ratios:
            continue
        m = st.median(ratios)
        mad = st.median([abs(x - m) for x in ratios])
        slower = sum(1 for x in ratios if x > 1)
        print("  %-10s %6d %8.4f %8.4f %8.4f %5d %7.3f" % (
            p, len(ratios), m, mad, min(ratios), slower, sign_p(slower, len(ratios))))
    print()
