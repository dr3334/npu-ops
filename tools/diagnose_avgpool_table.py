#!/usr/bin/env python3
"""Full contingency table for average-pool rounding: sign x fractional part of
sum/count  ->  device_value - trunc_toward_zero(sum/count).

Reading the table tells us the exact rule without guessing (AGENTS.md R10).
"""
import json
import os
import sys
from collections import Counter
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_convpool import pool_loop


def trunc_tz(fr):
    """Truncate a Fraction toward zero."""
    n, d = fr.numerator, fr.denominator
    q = abs(n) // d
    return q if n >= 0 else -q


def table(d):
    meta = json.load(open(os.path.join(d, "model.json")))
    shp = meta["shape"]
    ins = tuple(meta["input"]["shape"])
    outs = tuple(meta["output"]["shape"])
    x = np.fromfile(os.path.join(d, "golden", "input.bin"), np.int8).reshape(ins)
    dev = np.fromfile(os.path.join(d, "out_device.bin"), np.int8).reshape(outs)[0]
    tot, cnt = pool_loop(x[0], shp["FH"], shp["stride"], shp["padding"], "avg")
    tot = np.array(tot, np.int64)
    cnt = np.array(cnt, np.int64)
    hist = Counter()
    for idx in np.ndindex(tot.shape):
        s, c, dv = int(tot[idx]), int(cnt[idx]), int(dev[idx])
        q = Fraction(s, c)
        hist[(s >= 0, float(q - trunc_tz(q)), c, dv - trunc_tz(q))] += 1
    print(f"== {os.path.basename(d)}")
    print("   sign+  frac  count  device-trunc   n")
    for (pos, f, c, delta), n in sorted(hist.items(), key=lambda kv: (not kv[0][0], kv[0][1], kv[0][2])):
        print(f"   {'+' if pos else '-'}     {f:.4f}  {c:5d}   {delta:+d}          {n}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        table(d)
