#!/usr/bin/env python3
"""Explain every ±1 mismatch: recompute the int64 accumulator for the failing
positions and print acc, exact rational quotient, golden and device value, so the
device's requant rounding rule can be inferred from data instead of guessed."""
import json
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_convpool import conv_loop, frac, half_away  # same math, second path


def explain(opdir):
    meta = json.load(open(os.path.join(opdir, "model.json")))
    shp = meta["shape"]
    q = meta["quant"]
    ins = tuple(meta["input"]["shape"])
    outs = tuple(meta["output"]["shape"])
    x = np.fromfile(os.path.join(opdir, "golden", "input.bin"), np.int8).reshape(ins)
    exp = np.fromfile(os.path.join(opdir, "golden", "expected.bin"), np.int8).reshape(outs)
    dev = np.fromfile(os.path.join(opdir, "out_device.bin"), np.int8).reshape(outs)
    w = np.fromfile(os.path.join(opdir, "golden", "filter.bin"), np.int8).reshape(
        tuple(meta["filter_shape"]))
    bp = os.path.join(opdir, "golden", "bias.bin")
    bias = np.fromfile(bp, np.int32) if os.path.exists(bp) else None
    depthwise = meta["op"] == "DEPTHWISE_CONV_2D"
    acc = np.array(conv_loop(x[0], w, bias, shp["stride"], shp["dilation"],
                             shp["padding"], depthwise,
                             shp.get("mult", 1)))[None]
    m = frac(q["input"]) * frac(q["weights"]) / frac(q["output"])
    print(f"== {os.path.basename(opdir)}  multiplier={m}  "
          f"({float(m):.9g})  act={shp['activation']}")
    bad = np.argwhere(exp != dev)
    print(f"   mismatches: {len(bad)}")
    for i in map(tuple, bad[:12]):
        a = int(acc[i])
        exact = Fraction(a) * m
        rule_away = half_away(exact)
        rule_floor = int(np.floor(exact))
        rule_half_up = int(np.floor(exact + Fraction(1, 2)))
        print(f"   at {i}: acc={a:8d}  acc*m={float(exact):+.6f}  "
              f"golden={int(exp[i]):4d} device={int(dev[i]):4d} | "
              f"half_away={rule_away} floor={rule_floor} half_up={rule_half_up}")
    # global test of each rule
    rules = {
        "half_away": lambda e: half_away(e),
        "floor": lambda e: int(np.floor(e)),
        "half_up": lambda e: int(np.floor(e + Fraction(1, 2))),
        "half_even": lambda e: int(round(float(e))),
    }
    flat = acc.reshape(-1)
    dflat = dev.reshape(-1)
    for name, fn in rules.items():
        pred = np.array([max(-128, min(127, fn(Fraction(int(a)) * m))) for a in flat], np.int64)
        nb = int((pred != dflat).sum())
        print(f"   rule {name:10s}: mismatch {nb}/{flat.size}")
    return 0


if __name__ == "__main__":
    for d in sys.argv[1:]:
        explain(d)
