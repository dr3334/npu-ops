#!/usr/bin/env python3
"""Which rounding rule does the MDLA use for AVERAGE_POOL_2D?

The conv rule (two-step fixed-point, tools/validate_devrule.py) does NOT hold for
average pooling: the dev-rule golden is off by ±1 on ~23% / ~11% of the outputs.
This script takes the device dump, recomputes the window sum and the valid-element
count with the independent loop implementation, and scores candidate rules:

  rational    : round-half-away(sum / count)            (exact rational)
  two_step    : RoundingDivideByPOT(round_half_up(s*M0/2^31), shift)   (TFLite)
  floor       : floor(sum / count)
  half_up     : floor((sum + count/2) / count)
  half_even   : round-half-to-even(sum / count)
  pad_div     : round-half-away(sum / filter_area)      (padding counted)

Usage: /usr/bin/python3 tools/diagnose_avgpool.py ops/<avgpool op> [...]
"""
import json
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_convpool import pool_loop, rdpot, rhm


def quantize_multiplier(num, den):
    m = Fraction(int(num), int(den))
    shift = 0
    while m < Fraction(1, 2):
        m *= 2
        shift += 1
    return int(np.floor(m * (1 << 31) + Fraction(1, 2))), shift


def half_away(fr):
    n, d = fr.numerator, fr.denominator
    return (2 * n + d) // (2 * d) if n >= 0 else -((-2 * n + d) // (2 * d))


def rules(s, c, area):
    out = {}
    out["rational"] = half_away(Fraction(s, c))
    m0, sh = quantize_multiplier(1, c)
    out["two_step"] = int(rdpot(rhm(np.array([s], np.int64), m0), sh)[0])
    out["floor"] = s // c
    out["half_up"] = (s * 2 + c) // (2 * c) if s >= 0 else -((-2 * s + c) // (2 * c))
    out["half_even"] = int(round(float(Fraction(s, c))))
    out["pad_div"] = half_away(Fraction(s, area))
    return out


def main(dirs):
    for d in dirs:
        meta = json.load(open(os.path.join(d, "model.json")))
        shp = meta["shape"]
        ins = tuple(meta["input"]["shape"])
        outs = tuple(meta["output"]["shape"])
        x = np.fromfile(os.path.join(d, "golden", "input.bin"), np.int8).reshape(ins)
        dev = np.fromfile(os.path.join(d, "out_device.bin"), np.int8).reshape(outs)[0]
        tot, cnt = pool_loop(x[0], shp["FH"], shp["stride"], shp["padding"], "avg")
        tot = np.array(tot, np.int64)
        cnt = np.array(cnt, np.int64)
        area = shp["FH"] * shp["FW"]
        names = list(rules(0, 2, area).keys())
        score = {k: 0 for k in names}
        detail = []
        for idx in np.ndindex(tot.shape):
            s, c = int(tot[idx]), int(cnt[idx])
            r = rules(s, c, area)
            dv = int(dev[idx])
            for k in names:
                if r[k] != dv:
                    score[k] += 1
            if r["rational"] != dv:
                detail.append((idx, s, c, dv, r))
        print(f"== {os.path.basename(d)}  area={area}  counts={sorted(set(cnt.ravel().tolist()))}")
        for k in names:
            print(f"   {k:9s}: mismatch {score[k]:4d}/{tot.size}")
        for idx, s, c, dv, r in detail[:6]:
            print(f"   at {idx}: sum={s:5d} cnt={c} device={dv:4d} "
                  f"rational={r['rational']:4d} two_step={r['two_step']:4d} "
                  f"floor={r['floor']:4d} pad_div={r['pad_div']:4d}")


if __name__ == "__main__":
    main(sys.argv[1:])
