#!/usr/bin/env python3
"""Build an int8 FULLY_CONNECTED model whose WEIGHTS are a runtime input (dynamic B).

This is the S4 (docs/05 E2) probe: if ncc-tflite compiles it, the weight tensor is not
baked into the .dla and one compiled component can serve different weights -> L3.

    /usr/bin/python3 tools/build_dynb.py --m 1 --k 64 --n 64 --out ops/fc_dynb_1x64x64_int8

Writes: model.tflite, model.json, golden/{input.bin,weights.bin,weights2.bin,expected.bin,expected2.bin}
"""
import argparse
import json
import os

import numpy as np

import build_ops as bo

SEED = 20261112


def run(out_dir, m=1, k=64, n=64, tol=1):
    rng = np.random.default_rng(SEED)
    s = 1.0 / 127.0
    o_scale = s * s * 32.0

    a_q = np.clip(np.round(rng.uniform(-1, 1, (m, k)) / s), -127, 127).astype(np.int8)
    w1 = np.clip(np.round(rng.uniform(-1, 1, (n, k)) / s), -127, 127).astype(np.int8)
    w2 = np.clip(np.round(rng.uniform(-1, 1, (n, k)) / s), -127, 127).astype(np.int8)
    b_q = np.zeros(n, dtype=np.int32)          # zero bias: isolate the weight path

    def expect(w):
        acc = a_q.astype(np.int64) @ w.astype(np.int64).T
        y = np.round(acc * (s * s) / o_scale)
        return np.clip(y, -128, 127).astype(np.int8)

    b = bo.flatbuffers.Builder(4096)
    buffers = [bo.buf_empty(b)]                # 0: placeholder for IO tensors
    buffers.append(bo.buf_data(b, b_q))        # 1: bias (constant)
    buffers.append(bo.buf_empty(b))            # 2: output placeholder

    t_in = bo.tensor(b, "input", [m, k], bo.I8, 0, bo.quant_tensor(b, [s], [0]))
    t_w = bo.tensor(b, "weights", [n, k], bo.I8, 0, bo.quant_tensor(b, [s], [0]))
    t_b = bo.tensor(b, "bias", [n], bo.I32, 1, bo.quant_tensor(b, [s * s], [0]))
    t_out = bo.tensor(b, "output", [m, n], bo.I8, 2, bo.quant_tensor(b, [o_scale], [0]))

    S = bo.S
    S.FullyConnectedOptionsStart(b)
    S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)
    S.FullyConnectedOptionsAddWeightsFormat(b, 0)
    S.FullyConnectedOptionsAddKeepNumDims(b, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
    opt = S.FullyConnectedOptionsEnd(b)

    op = bo.operator(b, 0, [0, 1, 2], [3], bo.BuiltinOptions.FullyConnectedOptions, opt)
    opcodes = [bo.opcode(b, bo.BuiltinOperator.FULLY_CONNECTED)]
    blob = bo.finish(b, buffers, opcodes, [t_in, t_w, t_b, t_out], [op],
                     [0, 1], [3], f"fc dynB {m}x{k}x{n} int8")

    os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
    open(os.path.join(out_dir, "model.tflite"), "wb").write(blob)
    a_q.tofile(os.path.join(out_dir, "golden", "input.bin"))
    w1.tofile(os.path.join(out_dir, "golden", "weights.bin"))
    w2.tofile(os.path.join(out_dir, "golden", "weights2.bin"))
    expect(w1).tofile(os.path.join(out_dir, "golden", "expected.bin"))
    expect(w2).tofile(os.path.join(out_dir, "golden", "expected2.bin"))

    meta = {
        "op": "FULLY_CONNECTED",
        "shape": {"M": m, "K": k, "N": n},
        "dtype": "int8",
        "dynamic_weights": True,
        "inputs": [
            {"name": "input", "shape": [m, k], "type": "int8"},
            {"name": "weights", "shape": [n, k], "type": "int8"},
        ],
        "output": {"name": "output", "shape": [m, n], "type": "int8"},
        "quant": {"input": s, "weights": s, "bias": s * s, "output": o_scale},
        "int8_tol": tol,
        "seed": SEED,
        "tflite_bytes": len(blob),
    }
    json.dump(meta, open(os.path.join(out_dir, "model.json"), "w"), indent=2)
    print(f"{out_dir}/model.tflite {len(blob)} B  inputs=2 (A,W)  out {expect(w1)[:4].tolist()}...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    run(a.out, a.m, a.k, a.n)
