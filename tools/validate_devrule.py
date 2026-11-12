#!/usr/bin/env python3
"""Validate the DERIVED device requant model against device dumps that were
recorded *before* the model existed.

The three generation-1 ops were built with the naive rational golden and their
out_device.bin is on disk.  Here we take their int64 accumulator (from the
independent loop implementation) and push it through the two-step fixed-point
rule

    out = RoundingDivideByPOT( round_half_up(acc * M0 / 2^31), shift )

with (M0, shift) from TFLite's QuantizeMultiplier of the scale ratio, and count
mismatches against the device.  0 mismatches over ~1.5k outputs of three
different multipliers (1/1024, 1/2048, 1/3) means the rule explains the device,
not just the sample.

    /usr/bin/python3 tools/validate_devrule.py ops/<name> [...]
"""
import json
import os
import sys
from collections import Counter
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_convpool import conv_loop, rdpot, rhm  # integer building blocks only


def quantize_multiplier(num, den):
    m = Fraction(int(num), int(den))
    shift = 0
    while m < Fraction(1, 2):
        m *= 2
        shift += 1
    return int(np.floor(m * (1 << 31) + Fraction(1, 2))), shift


def rational(acc, num, den):
    """Naive reference: one rounding step, exact rational multiplier."""
    m = Fraction(int(num), int(den))
    out = []
    for a in acc:
        e = Fraction(int(a)) * m
        out.append((2 * e.numerator + e.denominator) // (2 * e.denominator)
                   if e >= 0 else
                   -((-2 * e.numerator + e.denominator) // (2 * e.denominator)))
    return np.array(out, np.int64)


def analyse(opdir):
    meta = json.load(open(os.path.join(opdir, "model.json")))
    shp, q = meta["shape"], meta["quant"]
    ins = tuple(meta["input"]["shape"])
    outs = tuple(meta["output"]["shape"])
    x = np.fromfile(os.path.join(opdir, "golden", "input.bin"), np.int8).reshape(ins)
    dev = np.fromfile(os.path.join(opdir, "out_device.bin"), np.int8).reshape(outs).reshape(-1)
    w = np.fromfile(os.path.join(opdir, "golden", "filter.bin"), np.int8).reshape(
        tuple(meta["filter_shape"]))
    bp = os.path.join(opdir, "golden", "bias.bin")
    bias = np.fromfile(bp, np.int32) if os.path.exists(bp) else None
    dw = meta["op"] == "DEPTHWISE_CONV_2D"
    acc = np.array(conv_loop(x[0], w, bias, shp["stride"], shp["dilation"],
                             shp["padding"], dw, shp.get("mult", 1))).reshape(-1)

    # multiplier = (in_scale * w_scale) / out_scale, as an exact rational
    r = Fraction(float(q["input"])) * Fraction(float(q["weights"])) / Fraction(float(q["output"]))
    # power-of-two case: r.denominator is a power of two and r.numerator == 1
    num, den = r.numerator, r.denominator
    m0, shift = quantize_multiplier(num, den)
    dev_pred = rdpot(rhm(acc, m0), shift)
    rat_pred = rational(acc, num, den)

    # clamp like the device does (int8 + activation)
    act = shp.get("activation", "NONE")

    def post(pred):
        p = np.clip(pred, -128, 127)
        if act in ("RELU", "RELU6"):
            p = np.maximum(p, 0)
        if act == "RELU6":
            p = np.minimum(p, int(np.floor(6.0 / float(q["output"]))))
        return p

    for pred, tag in ((dev_pred, "dev-rule"), (rat_pred, "rational")):
        nb = int((post(pred) != dev).sum())
        print(f"{os.path.basename(opdir):34s} mult={num}/{den} (M0={m0}, shift={shift})"
              f"  {tag:9s}: mismatch {nb:3d}/{dev.size}")

    # if the dev rule misses, print where (should not happen; kept as a guard)
    p = post(dev_pred)
    bad = np.argwhere(p != dev).ravel()[:6]
    for i in bad:
        print(f"   at {i}: acc={int(acc[i])} dev={int(dev[i])} devrule={int(p[i])}")
    return int((post(dev_pred) != dev).sum())


if __name__ == "__main__":
    tot = 0
    for d in sys.argv[1:]:
        tot += analyse(d)
    print(f"\ntotal dev-rule mismatches: {tot}")
    sys.exit(0 if tot == 0 else 1)
