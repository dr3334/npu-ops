#!/usr/bin/env python3
"""E-107 — second operator: ADD (elementwise int8), same pipeline as FULLY_CONNECTED.

    /usr/bin/python3 tools/build_add.py --out ops/add_1x64_int8 --k 64

Writes model.tflite, model.json and golden/{input.bin,input1.bin,expected.bin}.
Contract: out = clip(qa + qb, -128..127) with identical scales, zero points 0.
"""
import argparse
import json
import os

import numpy as np

import build_ops as bo

SEED = 20261112


def main(out_dir, k_dim, m):
    s = 1.0 / 127.0
    rng = np.random.default_rng(SEED)
    a = rng.integers(-100, 101, size=(m, k_dim)).astype(np.int8)
    b = rng.integers(-100, 101, size=(m, k_dim)).astype(np.int8)
    expected = np.clip(a.astype(np.int64) + b.astype(np.int64), -128, 127).astype(np.int8)

    blob_b = bo.flatbuffers.Builder(2048)
    buffers = [bo.buf_empty(blob_b), bo.buf_empty(blob_b)]
    t_a = bo.tensor(blob_b, "a", [m, k_dim], bo.I8, 0, bo.quant_tensor(blob_b, [s], [0]))
    t_b = bo.tensor(blob_b, "b", [m, k_dim], bo.I8, 0, bo.quant_tensor(blob_b, [s], [0]))
    t_o = bo.tensor(blob_b, "out", [m, k_dim], bo.I8, 1, bo.quant_tensor(blob_b, [s], [0]))

    S = bo.S
    S.AddOptionsStart(blob_b)
    S.AddOptionsAddFusedActivationFunction(blob_b, 0)
    opt = S.AddOptionsEnd(blob_b)

    op = bo.operator(blob_b, 0, [0, 1], [2], bo.BuiltinOptions.AddOptions, opt)
    blob = bo.finish(blob_b, buffers, [bo.opcode(blob_b, bo.BuiltinOperator.ADD)],
                     [t_a, t_b, t_o], [op], [0, 1], [2], f"add {m}x{k_dim} int8")

    d = out_dir
    os.makedirs(os.path.join(d, "golden"), exist_ok=True)
    open(os.path.join(d, "model.tflite"), "wb").write(blob)
    a.tofile(os.path.join(d, "golden", "input.bin"))
    b.tofile(os.path.join(d, "golden", "input1.bin"))
    expected.tofile(os.path.join(d, "golden", "expected.bin"))

    meta = {
        "op": "ADD",
        "shape": {"M": m, "K": k_dim},
        "dtype": "int8",
        "inputs": [
            {"name": "a", "shape": [m, k_dim], "type": "int8"},
            {"name": "b", "shape": [m, k_dim], "type": "int8"},
        ],
        "output": {"name": "out", "shape": [m, k_dim], "type": "int8"},
        "quant": {"input": s, "output": s},
        "int8_tol": 0,
        "reference": "int64_exact",
        "seed": SEED,
    }
    json.dump(meta, open(os.path.join(d, "model.json"), "w"), indent=2)
    print(f"{d}: tflite={len(blob)}B  out=a+b clipped, {m}x{k_dim}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ops/add_1x64_int8")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--m", type=int, default=1)
    a = ap.parse_args()
    main(a.out, a.k, a.m)
