#!/usr/bin/env python3
"""Second, deliberately slow golden for the conv/pool probes.

`tools/build_convpool.py` computes the reference with vectorised numpy (index
planes gathered per (kh,kw) tap + int64 matmul).  This file re-derives the same
expected.bin with plain Python nested loops, so an indexing / padding / striding
mistake in the fast golden shows up as a VERIFY-FAIL before any device time is
spent.

The requant rule is NOT re-derived here: it is read from model.json
("requant": {rule, num, den, m0, shift, relu6_code}) and applied in explicit
integer steps, so this file shares no derivation code with the builder.

    /usr/bin/python3 tools/verify_convpool.py ops/<name> [...]

Exit code 0 = every listed op's expected.bin is reproduced by the loop reference.
"""
import json
import os
import sys
from fractions import Fraction

import numpy as np


def pad_split(in_size, filt, stride, dil, padding):
    eff = (filt - 1) * dil + 1
    if padding == "SAME":
        out = (in_size + stride - 1) // stride
        total = max((out - 1) * stride + eff - in_size, 0)
        return out, total // 2
    return (in_size - eff) // stride + 1, 0


def half_away(frac):
    """Nearest integer to a Fraction, ties away from zero."""
    n, d = frac.numerator, frac.denominator
    return (2 * n + d) // (2 * d) if n >= 0 else -((-2 * n + d) // (2 * d))


def rhm(x, m0):
    """round-half-up(x*m0 / 2^31), i.e. gemmlowp's high-multiply."""
    return (np.asarray(x, np.int64) * np.int64(m0) + (1 << 30)) >> 31


def rdpot(x, shift):
    """round-half-away-from-zero(x / 2^shift)."""
    x = np.asarray(x, np.int64)
    if shift == 0:
        return x
    mask = (1 << shift) - 1
    thr = np.int64(mask >> 1) + (x < 0).astype(np.int64)
    return (x >> shift) + ((x & mask) > thr).astype(np.int64)


def conv_loop(x, w, bias, stride, dil, padding, depthwise, mult):
    H, W, C = x.shape
    FH, FW = w.shape[1], w.shape[2]
    OC = w.shape[3] if depthwise else w.shape[0]
    OH, pt = pad_split(H, FH, stride, dil, padding)
    OW, pl = pad_split(W, FW, stride, dil, padding)
    out = [[[0] * OC for _ in range(OW)] for _ in range(OH)]
    for oh in range(OH):
        for ow in range(OW):
            for oc in range(OC):
                acc = 0
                for kh in range(FH):
                    ih = oh * stride + kh * dil - pt
                    if not (0 <= ih < H):
                        continue
                    for kw in range(FW):
                        iw = ow * stride + kw * dil - pl
                        if not (0 <= iw < W):
                            continue
                        if depthwise:
                            acc += int(x[ih][iw][oc // mult]) * int(w[0][kh][kw][oc])
                        else:
                            for ic in range(C):
                                acc += int(x[ih][iw][ic]) * int(w[oc][kh][kw][ic])
                if bias is not None:
                    acc += int(bias[oc])
                out[oh][ow][oc] = acc
    return out


def pool_loop(x, f, stride, padding, mode):
    """Returns per-output max, or (sum, valid_count) for average pooling."""
    H, W, C = x.shape
    OH, pt = pad_split(H, f, stride, 1, padding)
    OW, pl = pad_split(W, f, stride, 1, padding)
    out = [[[0] * C for _ in range(OW)] for _ in range(OH)]
    cnt = [[[0] * C for _ in range(OW)] for _ in range(OH)]
    for oh in range(OH):
        for ow in range(OW):
            for c in range(C):
                vals = []
                for kh in range(f):
                    ih = oh * stride + kh - pt
                    for kw in range(f):
                        iw = ow * stride + kw - pl
                        if 0 <= ih < H and 0 <= iw < W:
                            vals.append(int(x[ih][iw][c]))
                        elif mode == "max":
                            vals.append(-128)      # int8 dtype minimum, TFLite rule
                if mode == "max":
                    out[oh][ow][c] = max(vals)
                else:
                    out[oh][ow][c] = sum(vals)
                    cnt[oh][ow][c] = len(vals)
    return (out, None) if mode == "max" else (out, cnt)


def frac(v):
    return Fraction(float(v)).limit_denominator(1 << 40)


def check(opdir):
    meta = json.load(open(os.path.join(opdir, "model.json")))
    shp = meta["shape"]
    ins = tuple(meta["input"]["shape"])
    outs = tuple(meta["output"]["shape"])
    x = np.fromfile(os.path.join(opdir, "golden", "input.bin"), np.int8).reshape(ins)
    exp = np.fromfile(os.path.join(opdir, "golden", "expected.bin"), np.int8).reshape(outs)
    op = meta["op"]
    act = shp.get("activation", "NONE")
    q = meta["quant"]
    req = meta["requant"]
    tol = int(meta.get("int8_tol", 0))
    name = os.path.basename(opdir.rstrip("/"))

    if op in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
        mode = "max" if op == "MAX_POOL_2D" else "avg"
        tot, cnt = pool_loop(x[0], shp["FH"], shp["stride"], shp["padding"], mode)
        if mode == "max":
            o = np.array(tot, np.int64)[None]
        else:
            # round to nearest, ties toward +inf: floor((2*sum + count) / (2*count))
            s = np.array(tot, np.int64)
            c = np.array(cnt, np.int64)
            o = ((2 * s + c) // (2 * c))[None]
    else:
        depthwise = op == "DEPTHWISE_CONV_2D"
        fs = tuple(meta["filter_shape"])
        w = np.fromfile(os.path.join(opdir, "golden", "filter.bin"), np.int8).reshape(fs)
        bp = os.path.join(opdir, "golden", "bias.bin")
        bias = np.fromfile(bp, np.int32) if os.path.exists(bp) else None
        mult = shp.get("mult", 1) if depthwise else 1
        acc = np.array(conv_loop(x[0], w, bias, shp["stride"], shp["dilation"],
                                 shp["padding"], depthwise, mult), np.int64)
        if req["rule"] == "dev":
            conv = lambda a: rdpot(rhm(int(a), req["m0"]), req["shift"])   # noqa: E731
        else:
            m = Fraction(req["num"], req["den"])
            conv = lambda a: half_away(Fraction(int(a)) * m)               # noqa: E731
        o = np.vectorize(conv, otypes=[np.int64])(acc)[None]
        if act in ("RELU", "RELU6"):
            o = np.maximum(o, 0)
        if act == "RELU6":
            o = np.minimum(o, req["relu6_code"])

    o = np.clip(o, -128, 127).astype(np.int8)
    assert o.shape == exp.shape, (o.shape, exp.shape)
    d = np.abs(exp.astype(np.int64) - o.astype(np.int64))
    bad = int((d > tol).sum())
    print(f"{name:36s} loop-vs-vector golden: max|diff|={int(d.max()) if d.size else 0}"
          f"  over_tol={bad}  {'OK' if bad == 0 else 'VERIFY-FAIL'}")
    return bad == 0


def main():
    dirs = sys.argv[1:]
    if not dirs:
        raise SystemExit(__doc__)
    ok = all(check(d) for d in dirs)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
