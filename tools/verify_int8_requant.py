#!/usr/bin/env python3
"""Board-free replay: does the int8 requant contract explain FC / BMM device dumps?

For every op dir given this script

  * reads model.tflite  -- operator type, operand shapes, quantisation scales,
                           BatchMatMul adj flags (no reliance on model.json);
  * rebuilds the INT64 accumulator from the runtime inputs (golden/input.bin,
    golden/input1.bin, golden/weights.bin) plus the weight buffer baked into the
    .tflite byte stream;
  * applies three candidate requant rules and counts mismatches against
    out_device.bin:

      A  what tools/build_ops.py:204 and tools/build_bmm.py:122 ship today
         -- one float64 np.round of the exact scale ratio;
      B  the contract, M0 taken from an INTEGER multiplier (the declared ratio as
         a short rational, 1/32 -> M0=2**30, shift=4):
             RoundingDivideByPOT(round_half_up(acc*M0/2**31), shift)
      C  the same two-step form but with M0 from TFLite's QuantizeMultiplier
         applied to the float64 ratio of the *stored float32* scales -- the
         "reverse-engineer M0 from the scale product" variant (M0=1073741840);

  * reports A against the shipped golden/expected.bin, which proves expected.bin
    really is rule A (and therefore that B is a change, not a no-op).

Zero board time, zero writes: the op dir is only read.

    /usr/bin/python3 tools/verify_int8_requant.py ops/<name> [ops/<name> ...]
    /usr/bin/python3 tools/verify_int8_requant.py --all-int8

Exit code 0 = rule B is bit-exact on every op that has a device dump.
"""
import argparse
import json
import os
import sys
from fractions import Fraction

import numpy as np
import tflite as S
from tflite.Model import Model as MModel

DT = {S.TensorType.INT8: np.int8, S.TensorType.INT32: np.int32,
      S.TensorType.INT64: np.int64, S.TensorType.FLOAT16: np.float16,
      S.TensorType.FLOAT32: np.float32}

OP_FC = 9
OP_BMM = 126


# ------------------------------------------------------------------ the rules
def _normalise(ratio):
    """ratio == q * 2**exponent with q in [1/2, 1)  ->  (M0 = round(q*2**31), exponent).

    Same normalisation as TFLite's QuantizeMultiplier(); the returned exponent keeps
    TFLite's sign convention (+ = left shift, - = right shift, see
    MultiplyByQuantizedMultiplier below).
    """
    q, exp = np.frexp(float(ratio))
    q_fixed = int(round(float(q) * (1 << 31)))
    if q_fixed == (1 << 31):              # round() can land on 2**31
        q_fixed //= 2
        exp += 1
    return q_fixed, int(exp)


def quantize_multiplier_rational(ratio, max_den=1 << 20):
    """Integer multiplier: the declared ratio as a short exact rational, THEN normalise.

    The probe ratios are 1 and 1/32, both exactly representable, so this lands on a
    power of two (1 -> M0=2**30,shift=+1 ; 1/32 -> M0=2**30,shift=-4).  This is the
    "M0 取整乘子" half of the contract: M0 is *not* reverse-engineered from float32
    scale products.
    """
    m = Fraction(ratio).limit_denominator(max_den)
    shift = 0
    while m < Fraction(1, 2):
        m *= 2
        shift += 1
    while m >= 1:
        m /= 2
        shift -= 1
    return int(np.floor(m * (1 << 31) + Fraction(1, 2))), -shift


def quantize_multiplier_float32(ratio):
    """TFLite QuantizeMultiplier() on the float64 ratio of the *stored float32* scales."""
    return _normalise(ratio) if ratio else (0, 0)


def rhm(x, m0):
    """gemmlowp SaturatingRoundingDoublingHighMul: round-half-up(x*m0 / 2**31)."""
    x = np.asarray(x, np.int64)
    return (x * np.int64(m0) + np.int64(1 << 30)) >> 31


def rdpot(x, shift):
    """gemmlowp RoundingDivideByPOT: round-half-away-from-zero(x / 2**shift), shift >= 0."""
    x = np.asarray(x, np.int64)
    if shift <= 0:
        return x
    mask = (1 << shift) - 1
    thr = np.int64(mask >> 1) + (x < 0).astype(np.int64)
    return (x >> shift) + ((x & mask) > thr).astype(np.int64)


def two_step(acc, m0, exp):
    """TFLite MultiplyByQuantizedMultiplier(x, M0, exp) == the contract's two roundings.

    left_shift = max(exp,0) applied BEFORE the high-multiply, right_shift = max(-exp,0)
    applied after -- that ordering is what makes an exact multiplier of 1 the identity.
    """
    left = exp if exp > 0 else 0
    right = 0 if exp > 0 else -exp
    x = np.asarray(acc, np.int64) << left
    return np.clip(rdpot(rhm(x, m0), right), -128, 127).astype(np.int8)


def one_round(acc, ratio):
    """Rule A: what the two builders ship today."""
    return np.clip(np.round(acc.astype(np.float64) * ratio), -128, 127).astype(np.int8)


# -------------------------------------------------------------- tflite reading
def _tensor_vals(m, sg, i):
    """(array or None, dtype) for tensor i; None = runtime input (empty buffer)."""
    t = sg.Tensors(i)
    n = int(np.prod([t.Shape(j) for j in range(t.ShapeLength())])) if t.ShapeLength() else 0
    dt = DT[t.Type()]
    raw = bytes(m.Buffers(t.Buffer()).DataAsNumpy())
    if n == 0:
        return np.zeros(0, dt), dt
    if len(raw) < n * np.dtype(dt).itemsize:
        return None, dt
    return np.frombuffer(raw, dtype=dt, count=n), dt


def _scales(sg, i):
    q = sg.Tensors(i).Quantization()
    if q is None or q.ScaleLength() == 0:
        return [1.0]
    return [q.Scale(j) for j in range(q.ScaleLength())]


def _name(sg, i):
    return bytes(sg.Tensors(i).Name()).decode()


def load_runtime_inputs(opdir, sg, tensor_idx):
    """Runtime value for a subgraph input tensor, in e2e.sh's push order."""
    order = [sg.Inputs(i) for i in range(sg.InputsLength())]
    pos = order.index(tensor_idx)
    if pos == 0:
        f = "input.bin"
    elif pos == 1 and os.path.exists(os.path.join(opdir, "golden", "weights.bin")):
        f = "weights.bin"
    else:
        f = "input%d.bin" % pos
    p = os.path.join(opdir, "golden", f)
    if not os.path.exists(p):
        raise SystemExit("missing runtime input %s" % p)
    return np.fromfile(p, dtype=np.int8)


def analyse(opdir):
    meta = json.load(open(os.path.join(opdir, "model.json")))
    m = MModel.GetRootAsModel(open(os.path.join(opdir, "model.tflite"), "rb").read(), 0)
    sg = m.Subgraphs(0)
    if sg.OperatorsLength() != 1:
        raise SystemExit("not a single-operator model (%d ops)" % sg.OperatorsLength())
    op = sg.Operators(0)
    code = m.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    ins = [op.Inputs(i) for i in range(op.InputsLength())]
    out_i = op.Outputs(0)

    def val(i):
        v, _ = _tensor_vals(m, sg, i)
        return load_runtime_inputs(opdir, sg, i) if v is None else v

    def shape(i):
        t = sg.Tensors(i)
        return tuple(t.Shape(j) for j in range(t.ShapeLength()))

    dev = np.fromfile(os.path.join(opdir, "out_device.bin"), dtype=np.int8)
    exp_p = os.path.join(opdir, "golden", "expected.bin")
    exp = np.fromfile(exp_p, dtype=np.int8) if os.path.exists(exp_p) else None

    if code == OP_FC:
        x_i, w_i = ins[0], ins[1]
        b_i = ins[2] if len(ins) > 2 and ins[2] >= 0 else None
        x, w = val(x_i), val(w_i)
        K = shape(x_i)[-1]
        M = int(x.size // K)
        N = int(w.size // K)
        acc = x.astype(np.int64).reshape(M, K) @ w.astype(np.int64).reshape(N, K).T
        if b_i is not None:
            acc = acc + val(b_i).astype(np.int64)[None, :]
        ratio = (_scales(sg, x_i)[0] * _scales(sg, w_i)[0]) / _scales(sg, out_i)[0]
        kind = "FC"
    elif code == OP_BMM:
        a_i, b_i = ins[0], ins[1]
        a, b = val(a_i), val(b_i)
        opt = S.BatchMatMulOptions()
        opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
        adj_y = bool(opt.AdjY())
        A = a.astype(np.int64).reshape((-1,) + shape(a_i)[-2:])
        B = b.astype(np.int64).reshape((-1,) + shape(b_i)[-2:])
        acc = (np.einsum("btq,bsq->bts", A, B) if adj_y
               else np.einsum("btq,bqd->btd", A, B))
        ratio = (_scales(sg, a_i)[0] * _scales(sg, b_i)[0]) / _scales(sg, out_i)[0]
        kind = "BMM(adj_y=%d)" % adj_y
    else:
        raise SystemExit("unsupported op code %d" % code)

    accf = acc.reshape(-1)
    if dev.size != accf.size:
        raise SystemExit("dump size %d != acc size %d" % (dev.size, accf.size))

    m0i, shi = quantize_multiplier_rational(ratio)
    m0f, shf = quantize_multiplier_float32(ratio)
    ya = one_round(accf, ratio)
    yb = two_step(accf, m0i, shi)
    yc = two_step(accf, m0f, shf)

    def n_bad(y):
        return int((y.astype(np.int64) != dev.astype(np.int64)).sum())

    row = {
        "op": os.path.basename(opdir.rstrip("/")),
        "kind": kind,
        "n": int(dev.size),
        "ratio": ratio,
        "m0_int": m0i, "shift_int": shi,
        "m0_f32": m0f, "shift_f32": shf,
        "A": n_bad(ya), "B": n_bad(yb), "C": n_bad(yc),
        "golden_is_A": (int((ya != exp).sum()) if exp is not None else None),
        "golden_vs_dev": (int((exp.astype(np.int64) != dev.astype(np.int64)).sum())
                          if exp is not None else None),
        "int8_tol": meta.get("int8_tol"),
        "codes": meta.get("codes"),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("opdirs", nargs="*")
    ap.add_argument("--all-int8", action="store_true",
                    help="every ops/* dir whose model.json says int8 and that has out_device.bin")
    ap.add_argument("--csv", help="also write the rows here")
    args = ap.parse_args()

    dirs = list(args.opdirs)
    if args.all_int8:
        for d in sorted(os.listdir("ops")):
            p = os.path.join("ops", d)
            mj = os.path.join(p, "model.json")
            if not os.path.exists(mj) or not os.path.exists(os.path.join(p, "out_device.bin")):
                continue
            try:
                meta = json.load(open(mj))
            except Exception:
                continue
            if meta.get("dtype") == "int8":
                dirs.append(p)

    rows, skipped = [], []
    for d in dirs:
        try:
            rows.append(analyse(d))
        except SystemExit as e:
            skipped.append((os.path.basename(d.rstrip("/")), str(e)))
        except Exception as e:
            skipped.append((os.path.basename(d.rstrip("/")), "%s: %s" % (type(e).__name__, e)))

    print("%-32s %-13s %7s %-8s %-13s %6s %6s %6s %6s" % (
        "op", "kind", "N", "ratio", "(M0,shift)", "A", "B", "C", "A==golden"))
    tot = {"N": 0, "A": 0, "B": 0, "C": 0}
    for r in rows:
        print("%-32s %-13s %7d %-8s %-13s %6d %6d %6d %6s" % (
            r["op"], r["kind"], r["n"], ("%.6g" % r["ratio"]),
            "(%d,%d)" % (r["m0_int"], r["shift_int"]),
            r["A"], r["B"], r["C"], r["golden_is_A"]))
        for k in tot:
            tot[k] += r[k] if k != "N" else r["n"]
    print("---- ops=%d  values=%d   A(float64 one-round)=%d  B(M0 integer)=%d  "
          "C(M0 from float32 scales)=%d" % (len(rows), tot["N"], tot["A"], tot["B"], tot["C"]))
    if skipped:
        print("---- skipped %d dirs (not FC/BMM or missing inputs):" % len(skipped))
        for n, why in skipped:
            print("     %-40s %s" % (n, why))
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("csv ->", args.csv)
    return 0 if all(r["B"] == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
