#!/usr/bin/env python3
"""Find the exact rounding threshold the device uses: for a *power-of-two*
multiplier 1/2^k the golden remainder acc mod 2^k is known exactly, so the set of
remainders that round up locates the device's rounding constant."""
import json
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_convpool import conv_loop, frac


def analyse(opdir, k):
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
                             shp["padding"], dw, shp.get("mult", 1))).reshape(-1)
    d = dev.reshape(-1).astype(np.int64)
    print(f"== {os.path.basename(opdir)}  mult=1/2^{k}  acc range "
          f"[{acc.min()},{acc.max()}]")
    up, down, exactn = [], [], 0
    for a, dv in zip(acc, d):
        a = int(a)
        fl = a >> k                       # arithmetic floor == floor(a/2^k)
        rem = a - (fl << k)               # 0..2^k-1
        frac_ = Fraction(rem, 1 << k)
        delta = int(dv) - fl
        if frac_ == 0:
            exactn += 1
            if delta != 0:
                print(f"   !! exact multiple but delta={delta} (acc={a})")
            continue
        (up if delta == 1 else down).append((float(frac_), a, int(dv)))
    print(f"   exact multiples: {exactn}")
    print(f"   rounded UP   n={len(up)}   frac range "
          f"[{min((u[0] for u in up), default=0):.4f}, {max((u[0] for u in up), default=0):.4f}]")
    print(f"   rounded DOWN n={len(down)} frac range "
          f"[{min((u[0] for u in down), default=0):.4f}, {max((u[0] for u in down), default=0):.4f}]")
    # smallest frac rounding up, largest rounding down
    if up:
        print(f"   min frac up   = {min(u[0] for u in up):.6f}")
    if down:
        print(f"   max frac down = {max(u[0] for u in down):.6f}")
    return 0


if __name__ == "__main__":
    analyse(sys.argv[1], int(sys.argv[2]))
