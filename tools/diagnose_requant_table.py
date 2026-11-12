#!/usr/bin/env python3
"""Empirical table: device_value - floor(exact) as a function of the fractional
part of the exact requant, for one op dir. Used to *infer* the device rounding
rule instead of guessing it (AGENTS.md R10)."""
import json
import os
import sys
from collections import Counter
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_convpool import conv_loop, frac


def table(opdir):
    meta = json.load(open(os.path.join(opdir, "model.json")))
    shp, q = meta["shape"], meta["quant"]
    ins = tuple(meta["input"]["shape"])
    outs = tuple(meta["output"]["shape"])
    x = np.fromfile(os.path.join(opdir, "golden", "input.bin"), np.int8).reshape(ins)
    dev = np.fromfile(os.path.join(opdir, "out_device.bin"), np.int8).reshape(outs)
    w = np.fromfile(os.path.join(opdir, "golden", "filter.bin"), np.int8).reshape(
        tuple(meta["filter_shape"]))
    bp = os.path.join(opdir, "golden", "bias.bin")
    bias = np.fromfile(bp, np.int32) if os.path.exists(bp) else None
    dw = meta["op"] == "DEPTHWISE_CONV_2D"
    acc = np.array(conv_loop(x[0], w, bias, shp["stride"], shp["dilation"],
                             shp["padding"], dw, shp.get("mult", 1)))[None]
    m = frac(q["input"]) * frac(q["weights"]) / frac(q["output"])
    print(f"== {os.path.basename(opdir)}  mult={m}")
    hist = Counter()
    for a, d in zip(acc.reshape(-1), dev.reshape(-1)):
        e = Fraction(int(a)) * m
        fl = int(np.floor(e))
        fracpart = e - fl
        hist[(round(float(fracpart), 4), int(d) - fl)] += 1
    for (fp, delta), n in sorted(hist.items()):
        print(f"   frac={fp:.4f}  device-floor={delta:+d}   n={n}")
    return 0


if __name__ == "__main__":
    for d in sys.argv[1:]:
        table(d)
