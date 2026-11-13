#!/usr/bin/env python3
"""Compare a device output dump against the golden expected output -- 16-bit aware.

Ticket 04d has int16 activation tensors, which tools/compare.py's DT map does not
know ({"int8","int32","float32","float16"} only; it raises KeyError on "int16").
Per AGENTS 8 the public tool is not modified from a ticket: this file is the
ticket-04d companion, same output contract, same PASS/FAIL meaning, and it prints
the extra code-difference statistics the 16-bit tier needs (distribution, not just
max).  Registered tolerance is still model.json["int8_tol"] in CODES.

    /usr/bin/python3 tools/compare_i16.py ops/<name> [--device out_device.bin]
"""
import argparse
import json
import os
import sys

import numpy as np

DT = {"int8": np.int8, "int16": np.int16, "int32": np.int32,
      "float32": np.float32, "float16": np.float16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("opdir")
    ap.add_argument("--device", default="out_device.bin")
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=None)
    args = ap.parse_args()

    d = args.opdir
    meta = json.load(open(os.path.join(d, "model.json")))
    otype = meta["output"]["type"]
    if otype not in DT:
        print("FAIL  %s: unknown output type %r" % (d, otype))
        return 2
    dt = DT[otype]
    exp_path = os.path.join(d, "golden", "expected.bin")
    dev_path = os.path.join(d, args.device)
    if not os.path.exists(dev_path):
        print("FAIL  %s: device output missing (%s)" % (d, args.device))
        return 2

    exp = np.fromfile(exp_path, dtype=dt)
    dev = np.fromfile(dev_path, dtype=dt)
    shape = tuple(meta["output"]["shape"])
    print("op        : %s %s  dtype=%s" % (meta["op"], meta["shape"], meta["dtype"]))
    print("expected  : %d vals   device: %d vals   logical shape=%s"
          % (exp.size, dev.size, shape))
    if exp.size != dev.size:
        print("FAIL  size mismatch (%d vs %d)" % (exp.size, dev.size))
        return 1
    exp = exp.reshape(shape)
    dev = dev.reshape(shape)

    if np.issubdtype(dt, np.integer):
        tol = int(meta.get("int8_tol", 0))
        neq = exp.astype(np.int64) - dev.astype(np.int64)
        a = np.abs(neq)
        n = exp.size
        bad = int((a != 0).sum())
        print("exact     : %s   mismatched=%d/%d" % ("YES" if bad == 0 else "NO", bad, n))
        print("contract  : int8_tol=%d  max_abs_diff=%d  fraction_off=%.4f%%"
              % (tol, int(a.max()) if n else 0, 100.0 * bad / n))
        print("codes     : exact=%d  <=1=%d  <=2=%d  <=3=%d  <=5=%d  <=10=%d"
              % tuple([int((a <= k).sum()) for k in (0, 1, 2, 3, 5, 10)]))
        print("distribution: mean=%.2f median=%.0f p99=%.0f max=%d"
              % (a.mean(), np.median(a), np.percentile(a, 99), int(a.max())))
        ok = int(a.max()) <= tol
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    e = exp.astype(np.float64)
    dev_f = dev.astype(np.float64)
    default = 2e-2 if otype == "float16" else 1e-5
    atol = float(meta.get("atol", default)) if args.atol is None else args.atol
    rtol = float(meta.get("rtol", default)) if args.rtol is None else args.rtol
    diff = np.abs(e - dev_f)
    rel = float(np.linalg.norm(e - dev_f) / max(np.linalg.norm(e), 1e-12))
    print("max_abs   : %.3e" % float(diff.max()))
    print("rel_l2    : %.3e" % rel)
    print("tol       : atol=%g rtol=%g" % (atol, rtol))
    ok = bool(np.all(diff <= atol + rtol * np.abs(e)))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
