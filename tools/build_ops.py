#!/usr/bin/env python3
"""Build single-operator TFLite probe models (FULLY_CONNECTED first), together with
model.json (IO contract) and golden/ (input.bin + expected.bin) for device-side 对拍.

Run with the interpreter that has the `tflite` schema package:
    /usr/bin/python3 tools/build_ops.py --list
    /usr/bin/python3 tools/build_ops.py fc --m 1 --k 64 --n 64 --dtype int8 \
        --out ops/fc_1x64x64_int8/model.tflite

Layout follows TFLite FULLY_CONNECTED:
    input  [M,K]  ->  weights [N,K]  (+ bias [N])  ->  output [M,N]
    out[m,n] = sum_k in[m,k]*W[n,k] + bias[n]
"""
import argparse
import json
import os

import flatbuffers
import numpy as np
import tflite as S
from tflite import (
    BuiltinOperator, BuiltinOptions, FullyConnectedOptions, Model, Operator,
    OperatorCode, QuantizationParameters, SubGraph, Tensor, TensorType,
)

F32 = TensorType.FLOAT32
F16 = TensorType.FLOAT16
I8 = TensorType.INT8
I32 = TensorType.INT32

SEED = 20261112


# --------------------------------------------------------------- flatbuffers
def buf_empty(b):
    S.BufferStart(b)
    return S.BufferEnd(b)


def buf_data(b, arr):
    v = b.CreateByteVector(np.ascontiguousarray(arr).tobytes())
    S.BufferStart(b)
    S.BufferAddData(b, v)
    return S.BufferEnd(b)


def quant_tensor(b, scales, zps, axis=0):
    sv = b.CreateNumpyVector(np.asarray(scales, np.float32))
    zv = b.CreateNumpyVector(np.asarray(zps, np.int64))
    S.QuantizationParametersStart(b)
    S.QuantizationParametersAddScale(b, sv)
    S.QuantizationParametersAddZeroPoint(b, zv)
    S.QuantizationParametersAddQuantizedDimension(b, axis)
    return S.QuantizationParametersEnd(b)


def tensor(b, name, shape, ttype, buffer_idx, q=None):
    n = b.CreateString(name)
    shp = b.CreateNumpyVector(np.asarray(shape, np.int32))
    S.TensorStart(b)
    S.TensorAddName(b, n)
    if q is not None:
        S.TensorAddQuantization(b, q)
    S.TensorAddShape(b, shp)
    S.TensorAddShapeSignature(b, shp)
    S.TensorAddType(b, ttype)
    S.TensorAddBuffer(b, buffer_idx)
    S.TensorAddHasRank(b, True)
    return S.TensorEnd(b)


def opcode(b, builtin, version=1):
    S.OperatorCodeStart(b)
    S.OperatorCodeAddDeprecatedBuiltinCode(b, builtin if builtin < 127 else 127)
    S.OperatorCodeAddBuiltinCode(b, builtin)
    S.OperatorCodeAddVersion(b, version)
    return S.OperatorCodeEnd(b)


def operator(b, oc, ins, outs, opt_type=None, opt=None):
    iv = b.CreateNumpyVector(np.asarray(ins, np.int32))
    ov = b.CreateNumpyVector(np.asarray(outs, np.int32))
    S.OperatorStart(b)
    S.OperatorAddOpcodeIndex(b, oc)
    S.OperatorAddInputs(b, iv)
    S.OperatorAddOutputs(b, ov)
    if opt is not None:
        S.OperatorAddBuiltinOptionsType(b, opt_type)
        S.OperatorAddBuiltinOptions(b, opt)
    return S.OperatorEnd(b)


def finish(b, buffers, opcodes, tensors, operators, inputs, outputs, desc):
    def vec(objs):
        v = b.StartVector(4, len(objs), 4)
        for o in reversed(objs):
            b.PrependUOffsetTRelative(o)
        return b.EndVector()

    tv, ov, cv, bv = vec(tensors), vec(operators), vec(opcodes), vec(buffers)
    iv = b.CreateNumpyVector(np.asarray(inputs, np.int32))
    og = b.CreateNumpyVector(np.asarray(outputs, np.int32))
    nm = b.CreateString("main")
    S.SubGraphStart(b)
    S.SubGraphAddTensors(b, tv)
    S.SubGraphAddInputs(b, iv)
    S.SubGraphAddOutputs(b, og)
    S.SubGraphAddOperators(b, ov)
    S.SubGraphAddName(b, nm)
    sg = S.SubGraphEnd(b)
    sgv = b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(sg)
    sgv = b.EndVector()
    d = b.CreateString(desc)
    S.ModelStart(b)
    S.ModelAddVersion(b, 3)
    S.ModelAddOperatorCodes(b, cv)
    S.ModelAddSubgraphs(b, sgv)
    S.ModelAddDescription(b, d)
    S.ModelAddBuffers(b, bv)
    m = S.ModelEnd(b)
    b.Finish(m, file_identifier=b"TFL3")
    return bytes(b.Output())


# ------------------------------------------------------------------- builder
def build_fc(m, k, n, dtype, codes="full"):
    """Return (tflite_bytes, model_json, input_array, expected_array).

    codes="full"  : full-range int8 activations/weights, int8 requant output
                    (device requant rounding is NOT bit-identical to exact float
                    division -> contract is |diff| <= int8_tol, see reports/S2-*.md)
    codes="small" : codes in {-1,0,1}, no bias, output scale == input*weight scale,
                    so out == exact INT64 accumulator with no rounding at all.
    """
    rng = np.random.default_rng(SEED)

    # deterministic, non-degenerate data
    a_f32 = rng.uniform(-1.0, 1.0, size=(m, k)).astype(np.float32)
    w_f32 = rng.uniform(-1.0, 1.0, size=(n, k)).astype(np.float32)
    b_f32 = rng.uniform(-0.5, 0.5, size=(n,)).astype(np.float32)
    ref = a_f32 @ w_f32.T + b_f32  # fp32 reference (float64 accumulate below)

    b = flatbuffers.Builder(4096)
    buffers = [buf_empty(b)]
    tensors = []
    opcodes = [opcode(b, BuiltinOperator.FULLY_CONNECTED)]

    # buffer 0 is the empty placeholder: model input / activations must NOT carry data
    if dtype == "fp32":
        t_in = tensor(b, "input", [m, k], F32, 0)

        w_buf = buf_data(b, w_f32)
        buffers.append(w_buf)
        t_w = tensor(b, "weights", [n, k], F32, 1)

        b1_buf = buf_data(b, b_f32)
        buffers.append(b1_buf)
        t_b = tensor(b, "bias", [n], F32, 2)

        buffers.append(buf_empty(b))
        t_out = tensor(b, "output", [m, n], F32, 3)

        input_arr = a_f32
        expected = ref
        int8_tol = 0
        in_type, out_type = "float32", "float32"
        scales = {"input": None, "weights": None, "bias": None, "output": None}

    elif dtype == "fp16":
        a16, w16, b16 = a_f32.astype(np.float16), w_f32.astype(np.float16), b_f32.astype(np.float16)
        t_in = tensor(b, "input", [m, k], F16, 0)
        buffers.append(buf_data(b, w16))
        t_w = tensor(b, "weights", [n, k], F16, 1)
        buffers.append(buf_data(b, b16))
        t_b = tensor(b, "bias", [n], F16, 2)
        buffers.append(buf_empty(b))
        t_out = tensor(b, "output", [m, n], F16, 3)
        acc = a16.astype(np.float32) @ w16.astype(np.float32).T
        input_arr, expected = a16, (acc + b16.astype(np.float32)).astype(np.float16)
        int8_tol = 0
        in_type = out_type = "float16"
        scales = {"input": None, "weights": None, "bias": None, "output": None}

    elif dtype == "int8":
        a_scale = 1.0 / 127.0
        w_scale = 1.0 / 127.0
        b_scale = a_scale * w_scale
        if codes == "small":
            # {-1,0,1} codes, zero bias, out scale == acc scale => out == exact INT64 acc
            o_scale = b_scale
            a_q = rng.integers(-1, 2, size=(m, k)).astype(np.int8)
            w_q = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
            b_q = np.zeros(n, dtype=np.int32)
            acc = a_q.astype(np.int64) @ w_q.astype(np.int64).T
            expected = np.clip(acc, -128, 127).astype(np.int8)
            int8_tol = 0
        else:
            o_scale = a_scale * w_scale * 32.0    # requant multiplier 1/32
            a_q = np.clip(np.round(a_f32 / a_scale), -127, 127).astype(np.int8)
            w_q = np.clip(np.round(w_f32 / w_scale), -127, 127).astype(np.int8)
            b_q = np.round(b_f32 / b_scale).astype(np.int32)
            acc = a_q.astype(np.int64) @ w_q.astype(np.int64).T
            y = np.round((acc + b_q[None, :].astype(np.int64)) * (b_scale / o_scale))
            expected = np.clip(y, -128, 127).astype(np.int8)
            int8_tol = 1                          # requant rounding boundary, see report

        t_in = tensor(b, "input", [m, k], I8, 0, quant_tensor(b, [a_scale], [0]))
        buffers.append(buf_data(b, w_q))          # buffer 1: weights
        t_w = tensor(b, "weights", [n, k], I8, 1, quant_tensor(b, [w_scale], [0]))
        buffers.append(buf_data(b, b_q))          # buffer 2: bias
        t_b = tensor(b, "bias", [n], I32, 2, quant_tensor(b, [b_scale], [0]))
        buffers.append(buf_empty(b))              # buffer 3: output placeholder
        t_out = tensor(b, "output", [m, n], I8, 3, quant_tensor(b, [o_scale], [0]))

        input_arr = a_q
        in_type, out_type = "int8", "int8"
        scales = {"input": a_scale, "weights": w_scale,
                  "bias": b_scale, "output": o_scale}
    else:
        raise SystemExit(f"unsupported dtype {dtype}")

    tensors.extend([t_in, t_w, t_b, t_out])

    S.FullyConnectedOptionsStart(b)
    S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)  # NONE
    S.FullyConnectedOptionsAddWeightsFormat(b, 0)            # DEFAULT [N,K]
    S.FullyConnectedOptionsAddKeepNumDims(b, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
    opt = S.FullyConnectedOptionsEnd(b)

    op = operator(b, 0, [0, 1, 2], [3], BuiltinOptions.FullyConnectedOptions, opt)
    blob = finish(b, buffers, opcodes, tensors, [op], [0], [3],
                  f"fc {m}x{k}x{n} {dtype}")

    meta = {
        "op": "FULLY_CONNECTED",
        "shape": {"M": m, "K": k, "N": n},
        "dtype": dtype,
        "input": {"name": "input", "shape": [m, k], "type": in_type,
                  "bytes": int(input_arr.nbytes)},
        "output": {"name": "output", "shape": [m, n], "type": out_type,
                   "bytes": int(expected.nbytes)},
        "quant": scales,
        "weights_shape": [n, k],
        "reference": ("int64_exact" if (dtype == "int8" and codes == "small")
                      else "int64_exact_requant" if dtype == "int8" else "float32"),
        "codes": codes,
        "int8_tol": int8_tol,
        "seed": SEED,
    }
    return blob, meta, input_arr, expected


# ---------------------------------------------------------------------- main
SPECS = {
    "fc_1x64x64_int8":   dict(op="fc", m=1, k=64, n=64, dtype="int8"),
    "fc_1x64x64_int8_exact": dict(op="fc", m=1, k=64, n=64, dtype="int8", codes="small"),
    "fc_1x64x64_fp32":   dict(op="fc", m=1, k=64, n=64, dtype="fp32"),
    "fc_1x64x64_fp16":   dict(op="fc", m=1, k=64, n=64, dtype="fp16"),
    "fc_8x128x128_fp16": dict(op="fc", m=8, k=128, n=128, dtype="fp16"),
    "fc_1x128x128_int8": dict(op="fc", m=1, k=128, n=128, dtype="int8"),
    "fc_8x128x128_int8": dict(op="fc", m=8, k=128, n=128, dtype="int8"),
    "fc_8x128x696_int8": dict(op="fc", m=8, k=128, n=696, dtype="int8"),
    "fc_1x128x21704_int8": dict(op="fc", m=1, k=128, n=21704, dtype="int8"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", help="output model.tflite path (dir gets model.json+golden/)")
    ap.add_argument("--m", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--codes", choices=["full", "small"], default="full")
    args = ap.parse_args()

    if args.list:
        for name in SPECS:
            print(name)
        return

    if args.spec in SPECS:
        cfg = dict(SPECS[args.spec])
    elif args.spec == "fc" and args.m and args.k and args.n and args.dtype:
        cfg = dict(op="fc", m=args.m, k=args.k, n=args.n, dtype=args.dtype)
    else:
        raise SystemExit("unknown spec; use --list or: fc --m .. --k .. --n .. --dtype ..")

    if cfg["op"] != "fc":
        raise SystemExit("only fc implemented so far")

    cfg.setdefault("codes", args.codes)
    blob, meta, x, y = build_fc(cfg["m"], cfg["k"], cfg["n"], cfg["dtype"], cfg.get("codes", "full"))

    out = args.out or f"ops/{args.spec or 'fc'}/model.tflite"
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(os.path.join(d, "golden"), exist_ok=True)
    with open(out, "wb") as f:
        f.write(blob)
    x.tofile(os.path.join(d, "golden", "input.bin"))
    y.tofile(os.path.join(d, "golden", "expected.bin"))
    meta["tflite"] = os.path.basename(out)
    meta["tflite_bytes"] = len(blob)
    with open(os.path.join(d, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{out}  {len(blob)} B  input{x.shape} output{y.shape} dtype={cfg['dtype']}")


if __name__ == "__main__":
    main()
