#!/usr/bin/env python3
"""E-108 — elementwise BINARY operators: MUL / SUB / MAXIMUM / MINIMUM (int8, same shape).

    /usr/bin/python3 tools/build_binop.py --op MUL     --m 1 --k 64  --out ops/mul_1x64_int8
    /usr/bin/python3 tools/build_binop.py --op SUB     --m 8 --k 128 --out ops/sub_8x128_int8
    /usr/bin/python3 tools/build_binop.py --op MAXIMUM --m 1 --k 64  --out ops/max_1x64_int8
    /usr/bin/python3 tools/build_binop.py --op MINIMUM --m 1 --k 64  --out ops/min_1x64_int8

Same pipeline as build_add.py (single-op TFLite -> ncc-tflite -> neuronrt -> compare.py).

Quantisation contract (why it is exact, i.e. int8_tol = 0):
  * SUB / MAXIMUM / MINIMUM: out_scale == in_scale == 1/127, zero point 0.
    these ops are exact integer ops (a-b, max(a,b), min(a,b)) in code space, no requant.
  * MUL: out_scale == in_a_scale * in_b_scale (== 1/127**2), so TFLite's
    real_multiplier = sa*sb/so == 1.0 exactly and the kernel is a pure integer
    multiply + clamp: out = clip(a*b, -128, 127). If out_scale were 1/127 the
    kernel would need a fixed-point requant by 1/127 and rounding would become an
    implementation detail (then int8_tol would have to be >= 1).
    Input codes are drawn from a narrow range (-12..12) so that a*b mostly stays
    inside int8 and only the tails saturate -> the test covers the linear region
    AND the clamp edges at once.

Reference implementation is independent of the device: computed with numpy int64
and clipped to int8 here, never taken from a device output (AGENTS.md R3).
"""
import argparse
import json
import os

import numpy as np
import flatbuffers

import build_ops as bo

SEED = 20261112

# op -> TFLite BuiltinOperator / BuiltinOptions + how to build the options table
OPS = {
    "MUL":     dict(builtin="MUL",     options="MulOptions"),
    "SUB":     dict(builtin="SUB",     options="SubOptions"),
    "MAXIMUM": dict(builtin="MAXIMUM", options="MaximumMinimumOptions"),
    "MINIMUM": dict(builtin="MINIMUM", options="MaximumMinimumOptions"),
}


def make_options(b, kind):
    S = bo.S
    if kind in ("MulOptions", "SubOptions"):
        start = getattr(S, kind + "Start")
        start(b)
        # NONE fused activation; ncc would otherwise fold ReLU/ReLU6 into the elementwise layer
        getattr(S, kind + "AddFusedActivationFunction")(b, 0)
        end = getattr(S, kind + "End")(b)
    elif kind == "MaximumMinimumOptions":
        S.MaximumMinimumOptionsStart(b)
        end = S.MaximumMinimumOptionsEnd(b)
    else:
        raise SystemExit(f"unknown options {kind}")
    return getattr(bo.BuiltinOptions, kind), end


def make_data(op, m, k, rng):
    """Return (a, b, expected, sa, sb, so, note) as int8 arrays.

    Random codes first, then a fixed set of injected edge cases (flat indices 0..7)
    so the test always covers clamp edges / ties / sign combinations instead of
    hoping that random sampling hits them.
    """
    lo, hi = (-12, 12) if op == "MUL" else (-100, 100)
    a = rng.integers(lo, hi + 1, size=(m, k)).astype(np.int8)
    b = rng.integers(lo, hi + 1, size=(m, k)).astype(np.int8)

    af, bf = a.ravel(), b.ravel()
    if op == "MUL":
        # (a,b) pairs chosen to hit +clamp, -clamp, zero, and unit cases
        pairs = [(12, 12), (-12, 12), (12, -12), (-11, 12), (0, -12), (11, -11),
                 (1, 127 % 127), (-1, 12)]
        sa = sb = 1.0 / 127.0
        so = sa * sb                      # real_multiplier == 1 -> exact integer path
        note = ("codes in [-12,12]; out_scale = sa*sb so TFLite real_multiplier == 1 "
                "(no requant rounding); clamp edges injected at flat idx 0..7")
    else:
        pairs = [(100, -100), (-100, 100), (-100, -100), (100, 100),
                 (7, 7), (-7, -7), (0, -100), (-100, 0)]
        sa = sb = so = 1.0 / 127.0
        note = ("full-range codes in [-100,100]; out_scale == in_scale, no requant; "
                "edge/tie pairs injected at flat idx 0..7")
    for i, (pa, pb) in enumerate(pairs[: af.size]):
        af[i] = pa
        bf[i] = pb
    a, b = af.reshape(m, k), bf.reshape(m, k)

    ai, bi = a.astype(np.int64), b.astype(np.int64)
    if op == "MUL":
        y = ai * bi
    elif op == "SUB":
        y = ai - bi
    elif op == "MAXIMUM":
        y = np.maximum(ai, bi)
    elif op == "MINIMUM":
        y = np.minimum(ai, bi)
    else:
        raise SystemExit(f"unknown op {op}")
    expected = np.clip(y, -128, 127).astype(np.int8)
    return a, b, expected, sa, sb, so, note


def main(op, m, k, out_dir):
    spec = OPS[op]
    rng = np.random.default_rng(SEED)
    a, b, expected, sa, sb, so, note = make_data(op, m, k, rng)

    blob_b = flatbuffers.Builder(2048)
    buffers = [bo.buf_empty(blob_b), bo.buf_empty(blob_b)]
    t_a = bo.tensor(blob_b, "a", [m, k], bo.I8, 0, bo.quant_tensor(blob_b, [sa], [0]))
    t_b = bo.tensor(blob_b, "b", [m, k], bo.I8, 0, bo.quant_tensor(blob_b, [sb], [0]))
    t_o = bo.tensor(blob_b, "out", [m, k], bo.I8, 1, bo.quant_tensor(blob_b, [so], [0]))

    opt_type, opt = make_options(blob_b, spec["options"])
    op_builtin = getattr(bo.BuiltinOperator, spec["builtin"])
    op_tbl = bo.operator(blob_b, 0, [0, 1], [2], opt_type, opt)
    blob = bo.finish(blob_b, buffers, [bo.opcode(blob_b, op_builtin)],
                     [t_a, t_b, t_o], [op_tbl], [0, 1], [2],
                     f"{spec['builtin'].lower()} {m}x{k} int8")

    os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
    with open(os.path.join(out_dir, "model.tflite"), "wb") as f:
        f.write(blob)
    a.tofile(os.path.join(out_dir, "golden", "input.bin"))
    b.tofile(os.path.join(out_dir, "golden", "input1.bin"))
    expected.tofile(os.path.join(out_dir, "golden", "expected.bin"))

    meta = {
        "op": spec["builtin"],
        "shape": {"M": m, "K": k},
        "dtype": "int8",
        "inputs": [
            {"name": "a", "shape": [m, k], "type": "int8"},
            {"name": "b", "shape": [m, k], "type": "int8"},
        ],
        "output": {"name": "out", "shape": [m, k], "type": "int8"},
        "quant": {"input": sa, "input1": sb, "output": so},
        "int8_tol": 0,
        "reference": "int64_exact_clip",
        "quant_note": note,
        "tflite_options": spec["options"],
        "seed": SEED,
    }
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    n_sat = int((np.abs(expected.astype(np.int16)) >= 127).sum())  # clamp-edge coverage
    print(f"{out_dir}: {spec['builtin']} {m}x{k} int8  tflite={len(blob)}B  "
          f"clamped_edges={n_sat}/{expected.size}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=sorted(OPS))
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.op, args.m, args.k, args.out)
