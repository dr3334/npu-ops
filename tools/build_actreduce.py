#!/usr/bin/env python3
"""Activation / reduction single-op probes for MDLA (MT6897, mdla5.0).

Scope (docs/AGENT-BRIEF.md, branch feat/actreduce):
    RELU, LOGISTIC, TANH, SOFTMAX, MEAN  (+ SUM if it compiles)

Method rules from AGENTS.md:
  R3  the golden is computed HERE, from the *stored* input values, in float64 / int64.
      It never goes through ncc, the TFLite interpreter, or the device runtime, so a
      golden/device agreement is evidence and not a shared bug.
  R7  compile / device-run / numerics / device-state are reported separately
      (this script only produces the artifacts; tools/compare.py does the numerics).
  R1  the numeric contract is registered in SPECS **before** the board run and copied
      verbatim into model.json, so the tolerance cannot be retro-fitted to the result.
      Rationale for each number is in the model.json "tolerance_rationale" field and in
      docs/OP-actreduce.md §1.

Quantisation follows the TFLite op spec (so the .tflite is a faithful TFLite program,
not something bent to the compiler):
  * int8 RELU            : in/out scale s, zp 0           -> max(q,0)   (bit-exact)
  * int8 LOGISTIC        : out scale 1/256, zp -128       (TFLite spec)
  * int8 TANH            : out scale 1/128, zp 0          (TFLite spec)
  * int8 SOFTMAX         : out scale 1/256, zp -128       (TFLite spec)
  * int8 MEAN / SUM      : out scale == in scale, zp 0    -> exact integer reduction

Usage (WSL python that has the `tflite` flatbuffer schema package):
    /usr/bin/python3 tools/build_actreduce.py --list
    /usr/bin/python3 tools/build_actreduce.py relu_1x64_fp16 --out ops/relu_1x64_fp16
"""
import argparse
import json
import os

import flatbuffers
import numpy as np

import build_ops as bo

SEED = 20261112          # same deterministic seed family as the rest of the repo

F32 = bo.F32
F16 = bo.F16
I8 = bo.I8
I32 = bo.I32
BO = bo.S.BuiltinOperator
BOPT = bo.S.BuiltinOptions
# the generated flatbuffer module exposes TensorType as plain ints, not an Enum
TYPE_NAME = {F32: "float32", F16: "float16", I8: "int8", I32: "int32"}


# --------------------------------------------------------------- spec table
# atol/rtol for float ops, int8_tol for quantised ops.  Registered before running.
SPECS = {
    # ---- RELU: the control experiment (expected to work, expected to be exact) -------
    "relu_1x64_fp16": dict(
        op="RELU", shape=[1, 64], dtype="fp16",
        atol=0.0, rtol=0.0,                     # relu is exact in fp16: max(x,0)
        prediction="compiles to an MDLA elementwise/activation layer; bit-exact",
    ),
    "relu_1x64_int8": dict(
        op="RELU", shape=[1, 64], dtype="int8",
        int8_tol=0,                             # same scale in/out, zp 0 -> max(q,0)
        prediction="compiles; bit-exact int8 clamp at 0",
    ),
    # ---- LOGISTIC -------------------------------------------------------------------
    "logistic_1x64_fp16": dict(
        op="LOGISTIC", shape=[1, 64], dtype="fp16",
        atol=2e-3, rtol=1e-2,                   # ~4 ULP at 0.5 (fp16 u=4.9e-4)
        prediction="LUT/activation path on MDLA; fp16 accuracy a few ULP",
    ),
    "logistic_1x64_int8": dict(
        op="LOGISTIC", shape=[1, 64], dtype="int8", in_scale=1 / 16.0,
        int8_tol=2,                             # fixed-point LUT vs exact sigmoid
        prediction="int8 logistic via fixed-point LUT; a couple of codes of error",
    ),
    # ---- TANH -----------------------------------------------------------------------
    "tanh_1x64_fp16": dict(
        op="TANH", shape=[1, 64], dtype="fp16",
        atol=2e-3, rtol=1e-2,
        prediction="LUT/activation path; fp16 accuracy a few ULP",
    ),
    "tanh_1x64_int8": dict(
        op="TANH", shape=[1, 64], dtype="int8", in_scale=1 / 16.0,
        int8_tol=2,
        prediction="int8 tanh via fixed-point LUT; a couple of codes of error",
    ),
    # ---- SOFTMAX (the open question: needed by attention) ---------------------------
    "softmax_1x8x16_fp16": dict(
        op="SOFTMAX", shape=[1, 8, 16], dtype="fp16", beta=1.0,
        atol=2e-3, rtol=1e-2,                   # ~4 ULP at 0.5
        prediction="either native MDLA softmax, or 'Bridging is disallowed' (CPU-only)",
    ),
    "softmax_1x8x16_int8": dict(
        op="SOFTMAX", shape=[1, 8, 16], dtype="int8", beta=1.0, in_scale=1 / 16.0,
        int8_tol=2,
        prediction="if fp16 compiles, int8 may still bridge (requant nodes)",
    ),
    # 供自建提交器做 attention 链路用：T=16 的方阵（每行 16 B，满足 16 字节规则）
    "softmax_1x16x16_int8": dict(
        op="SOFTMAX", shape=[1, 16, 16], dtype="int8", beta=1.0, in_scale=1 / 16.0,
        int8_tol=2,
        prediction="same lowering as [1,8,16]; needed as an asset for the attention chain",
    ),
    # ---- boundary extension (AGENTS.md §3.3: find the edge, not one sample) ---------
    "softmax_1x1x64_fp16": dict(
        op="SOFTMAX", shape=[1, 1, 64], dtype="fp16", beta=1.0,
        atol=2e-3, rtol=1e-2,
        prediction="single-head, L=64; expect the same SoftMaxLayer",
    ),
    "softmax_1x12x128_fp16_beta0125": dict(
        op="SOFTMAX", shape=[1, 12, 128], dtype="fp16", beta=0.125,
        atol=2e-3, rtol=1e-2,
        prediction="attention shape with 1/sqrt(d) style beta: does ncc honour beta?",
    ),
    "softmax_1x4x64_int8": dict(
        op="SOFTMAX", shape=[1, 4, 64], dtype="int8", beta=1.0, in_scale=1 / 16.0,
        int8_tol=2,
        prediction="int8 attention-sized softmax; expect exact codes like the 16-wide case",
    ),
    "relu_1x13_fp16": dict(
        op="RELU", shape=[1, 13], dtype="fp16",           # width not a multiple of 8
        atol=0.0, rtol=0.0,
        prediction="alignment probe: elementwise width 13 may bridge or pad",
    ),
    "mean_1x8x16_axis1_fp16": dict(
        op="MEAN", shape=[1, 8, 16], dtype="fp16", axis=[1], keep_dims=True,
        atol=2e-3, rtol=1e-2,
        prediction="reduction over a non-last axis; MeanLayer may or may not accept it",
    ),
    # ---- MEAN -----------------------------------------------------------------------
    "mean_1x8x16_fp16": dict(
        op="MEAN", shape=[1, 8, 16], dtype="fp16", axis=[2], keep_dims=False,
        atol=2e-3, rtol=1e-2,
        prediction="reduction over the last axis - bridge risk is high",
    ),
    "mean_4x16_int8": dict(
        op="MEAN", shape=[4, 16], dtype="int8", axis=[1], keep_dims=True,
        int8_tol=1,                             # mean -> requant rounding boundary
        prediction="if it compiles, require exact sum then round(mean(q))",
    ),
    # ---- SUM (bonus: same reduction family) -----------------------------------------
    "sum_1x8x16_fp16": dict(
        op="SUM", shape=[1, 8, 16], dtype="fp16", axis=[2], keep_dims=False,
        atol=0.1, rtol=1e-2,                    # |sum|<=64 -> fp16 ULP 0.0625, 2 ULP ok
        prediction="SUM is reducible to MEAN*16 on most backends; or it bridges",
    ),
    "sum_4x16_int8": dict(
        op="SUM", shape=[4, 16], dtype="int8", axis=[1], keep_dims=True, small_codes=True,
        int8_tol=0,                             # out scale == in scale -> exact clip
        prediction="exact integer reduce, if the backend has a sum primitive",
    ),
}


# --------------------------------------------------------------- golden math
def _fp16_round(x):
    """Round float64 -> float16 (storage type).  Golden stays independent of the device."""
    return np.asarray(x, np.float64).astype(np.float16)


def _set_specials(x, vals):
    """Pin a few known-interesting values (both tails + an exact 0) into the sample.

    Guarded: a 1-element tensor (used by the width scan) has no room for three.
    """
    if x.size >= len(vals):
        x.flat[:len(vals)] = vals


def _sigmoid(x):
    # numerically stable, float64, written out here so it does not depend on any
    # library the device side might share
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def golden(cfg, rng):
    """Return (x_store, expected_store, quant_meta, in_ttype, out_ttype, extra)."""
    op, dtype, shape = cfg["op"], cfg["dtype"], cfg["shape"]

    if op == "RELU":
        if dtype == "fp16":
            x = _fp16_round(rng.uniform(-6.0, 6.0, size=shape))
            # put both signs and an exact 0.0 in the sample on purpose
            _set_specials(x, (-6.0, 0.0, 6.0))
            y = np.maximum(x.astype(np.float64), 0.0)
            return x, _fp16_round(y), {}, F16, F16, {}
        s = 1.0 / 127.0
        q = rng.integers(-128, 128, size=shape).astype(np.int8)
        _set_specials(q, (np.int8(-128), np.int8(127)))
        y = np.clip(np.maximum(q.astype(np.int64), 0), -128, 127).astype(np.int8)
        qm = {"input": s, "output": s, "input_zp": 0, "output_zp": 0}
        return q, y, qm, I8, I8, {}

    if op == "LOGISTIC":
        if dtype == "fp16":
            x = _fp16_round(rng.uniform(-8.0, 8.0, size=shape))
            _set_specials(x, (-8.0, 0.0, 8.0))
            y = _fp16_round(_sigmoid(x.astype(np.float64)))
            return x, y, {}, F16, F16, {}
        s = cfg["in_scale"]
        q = rng.integers(-128, 128, size=shape).astype(np.int8)
        real = q.astype(np.float64) * s
        code = np.round(_sigmoid(real) * 256.0) - 128.0
        y = np.clip(code, -128, 127).astype(np.int8)
        qm = {"input": s, "output": 1 / 256.0, "input_zp": 0, "output_zp": -128}
        return q, y, qm, I8, I8, {}

    if op == "TANH":
        if dtype == "fp16":
            x = _fp16_round(rng.uniform(-6.0, 6.0, size=shape))
            _set_specials(x, (-6.0, 0.0, 6.0))
            y = _fp16_round(np.tanh(x.astype(np.float64)))
            return x, y, {}, F16, F16, {}
        s = cfg["in_scale"]
        q = rng.integers(-128, 128, size=shape).astype(np.int8)
        real = q.astype(np.float64) * s
        y = np.clip(np.round(np.tanh(real) * 128.0), -128, 127).astype(np.int8)
        qm = {"input": s, "output": 1 / 128.0, "input_zp": 0, "output_zp": 0}
        return q, y, qm, I8, I8, {}

    if op == "SOFTMAX":
        beta = float(cfg.get("beta", 1.0))
        if dtype == "fp16":
            x = _fp16_round(rng.uniform(-4.0, 4.0, size=shape))
            xd = x.astype(np.float64) * beta
            xd = xd - xd.max(axis=-1, keepdims=True)          # log-sum-exp shift
            e = np.exp(xd)
            y = _fp16_round(e / e.sum(axis=-1, keepdims=True))
            return x, y, {}, F16, F16, {"beta": beta}
        s = cfg["in_scale"]
        q = rng.integers(-128, 128, size=shape).astype(np.int8)
        xd = q.astype(np.float64) * s * beta
        xd = xd - xd.max(axis=-1, keepdims=True)
        e = np.exp(xd)
        p = e / e.sum(axis=-1, keepdims=True)
        code = np.round(p * 256.0) - 128.0
        y = np.clip(code, -128, 127).astype(np.int8)
        qm = {"input": s, "output": 1 / 256.0, "input_zp": 0, "output_zp": -128}
        return q, y, qm, I8, I8, {"beta": beta}

    if op in ("MEAN", "SUM"):
        axis = tuple(cfg["axis"])
        if dtype == "fp16":
            x = _fp16_round(rng.uniform(-4.0, 4.0, size=shape))
            acc = x.astype(np.float64).sum(axis=axis, keepdims=cfg["keep_dims"])
            if op == "MEAN":
                acc = acc / float(np.prod([shape[a] for a in axis]))
            return x, _fp16_round(acc), {}, F16, F16, {"axis": list(axis),
                                                       "keep_dims": cfg["keep_dims"]}
        s = 1.0 / 127.0
        if cfg.get("small_codes"):
            q = rng.integers(-7, 8, size=shape).astype(np.int8)   # |sum| <= 112
        else:
            q = rng.integers(-100, 101, size=shape).astype(np.int8)
        acc = q.astype(np.int64).sum(axis=axis, keepdims=cfg["keep_dims"])
        if op == "MEAN":
            n = float(np.prod([shape[a] for a in axis]))
            acc = np.round(acc.astype(np.float64) / n).astype(np.int64)
        y = np.clip(acc, -128, 127).astype(np.int8)
        qm = {"input": s, "output": s, "input_zp": 0, "output_zp": 0}
        return q, y, qm, I8, I8, {"axis": list(axis), "keep_dims": cfg["keep_dims"]}

    raise SystemExit(f"no golden for {op}")


# --------------------------------------------------------------- flatbuffer
def build_blob(cfg, x, y, qm, in_type, out_type, extra):
    op = cfg["op"]
    b = flatbuffers.Builder(4096)
    buffers = [bo.buf_empty(b)]
    tensors = []

    in_q = out_q = None
    if cfg["dtype"] == "int8":
        in_q = bo.quant_tensor(b, [qm["input"]], [qm["input_zp"]])
        out_q = bo.quant_tensor(b, [qm["output"]], [qm["output_zp"]])

    t_in = bo.tensor(b, "input", cfg["shape"], in_type, 0, in_q)
    tensors.append(t_in)                                   # tensor 0

    consts = []                                            # (name, shape, type, data)
    if op in ("MEAN", "SUM"):
        consts.append(("axis", [len(cfg["axis"])], I32,
                       np.asarray(cfg["axis"], np.int32)))
    for nm, shp, tt, arr in consts:
        buffers.append(bo.buf_data(b, arr))
        tensors.append(bo.tensor(b, nm, shp, tt, len(buffers) - 1))

    buffers.append(bo.buf_empty(b))                        # activation output buffer
    out_shape = list(np.asarray(y).shape)
    tensors.append(bo.tensor(b, "output", out_shape, out_type, len(buffers) - 1, out_q))
    out_idx = len(tensors) - 1

    S = bo.S
    opt = None
    opt_type = None
    if op == "SOFTMAX":
        S.SoftmaxOptionsStart(b)
        S.SoftmaxOptionsAddBeta(b, float(cfg.get("beta", 1.0)))
        opt = S.SoftmaxOptionsEnd(b)
        opt_type = BOPT.SoftmaxOptions
    elif op in ("MEAN", "SUM"):
        S.ReducerOptionsStart(b)
        S.ReducerOptionsAddKeepDims(b, bool(cfg["keep_dims"]))
        opt = S.ReducerOptionsEnd(b)
        opt_type = BOPT.ReducerOptions

    # op inputs vs *model* inputs are different lists:
    #   the reducer axis is a constant tensor (it carries a buffer) and must NOT be a
    #   subgraph input -- AGENTS.md §7.3: a model input with a data buffer confuses ncc
    #   (observed: ".dla declares 2 input handles, the 2nd is an unused input" and the
    #   runtime then refuses: "The number of input files should be 2" -> "Cannot get
    #   input[1] size").  This cost one debugging round; keep the comment.
    op_inputs = [0] + list(range(1, len(tensors) - 1))
    model_inputs = [0]
    node = bo.operator(b, 0, op_inputs, [out_idx], opt_type, opt)
    blob = bo.finish(b, buffers, [bo.opcode(b, getattr(BO, op))], tensors, [node],
                     model_inputs, [out_idx],
                     f"{op} {'x'.join(map(str, cfg['shape']))} {cfg['dtype']}")
    return blob, out_shape


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.list:
        for k in SPECS:
            print(k)
        return
    if args.spec not in SPECS:
        raise SystemExit(f"unknown spec {args.spec}; use --list")

    cfg = SPECS[args.spec]
    rng = np.random.default_rng(SEED)
    x, y, qm, in_type, out_type, extra = golden(cfg, rng)
    blob, out_shape = build_blob(cfg, x, y, qm, in_type, out_type, extra)

    out = args.out or f"ops/{args.spec}/model.tflite"
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(os.path.join(d, "golden"), exist_ok=True)
    with open(out, "wb") as f:
        f.write(blob)
    x.tofile(os.path.join(d, "golden", "input.bin"))
    y.tofile(os.path.join(d, "golden", "expected.bin"))

    meta = {
        "op": cfg["op"],
        # FC models name the math dims M/K/N; a single-op elementwise/reduction model has
        # no such names, so use positional d0/d1/... (a dict of repeated names would
        # silently collapse to the last value for 3-D shapes).
        "shape": {f"d{i}": int(d) for i, d in enumerate(cfg["shape"])},
        "op_shape": [int(d) for d in cfg["shape"]],
        "dtype": cfg["dtype"],
        "input": {"name": "input", "shape": cfg["shape"],
                  "type": TYPE_NAME[in_type], "bytes": int(np.asarray(x).nbytes)},
        "output": {"name": "output", "shape": out_shape,
                   "type": TYPE_NAME[out_type], "bytes": int(np.asarray(y).nbytes)},
        "quant": qm,
        "reference": "float64" if cfg["dtype"] == "fp16" else "int64_exact",
        "seed": SEED,
        "prediction": cfg["prediction"],
        "generator": "tools/build_actreduce.py",
    }
    meta.update(extra)
    if cfg["dtype"] == "int8":
        meta["int8_tol"] = int(cfg["int8_tol"])
        meta["tolerance_rationale"] = {
            0: "bit-exact contract: same scale in/out, zp 0, no requant rounding",
            1: "one requant/rounding boundary code (see reports/S2 §2.2)",
            2: "fixed-point LUT vs exact transcendental: <=2 codes of 256",
        }[int(cfg["int8_tol"])]
    else:
        meta["atol"] = float(cfg["atol"])
        meta["rtol"] = float(cfg["rtol"])
        meta["tolerance_rationale"] = (
            "fp16 storage: atol is a small multiple of the fp16 ULP at the op's "
            "output magnitude; exactness is additionally reported as max_abs")
    meta["tflite"] = os.path.basename(out)
    meta["tflite_bytes"] = len(blob)
    with open(os.path.join(d, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{out}  {len(blob)} B  in{list(np.asarray(x).shape)} -> out{out_shape} "
          f"dtype={cfg['dtype']}  op={cfg['op']}")


if __name__ == "__main__":
    main()
