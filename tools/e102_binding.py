#!/usr/bin/env python3
"""E-102 — weight binding vs address relocation (predictions written before the run).

Question 1 (binding): does output column n depend only on weight row n?
  W[n,k] = 1 only at (n0,k0), A all ones  =>  out[:, n0] == 1 and every other column == 0.
  Two different (n0,k0) pairs (5,3) and (40,17) exercise both a small and a large index.

Question 2 (relocation): identical model+data run in two fresh processes (system picks
  different dma-buf IOVAs each time) must give bit-identical output.

Predictions:
  H1 content-only binding : each op matches its expected pattern exactly; the two runs
                            of the same op are bit-identical even though the MAP IOVAs differ.
  H2 address leak         : output shifts / corrupts when the IOVA differs between runs,
                            or the wrong column lights up.

    /usr/bin/python3 tools/e102_binding.py --out ops/e102_col5  --n0 5  --k0 3
    /usr/bin/python3 tools/e102_binding.py --out ops/e102_col40 --n0 40 --k0 17
"""
import argparse
import json
import os

import numpy as np

import build_ops as bo

SEED = 20261112
M, K, N = 8, 64, 64


def main(out_dir, n0, k0):
    s = 1.0 / 127.0
    o_scale = s * s                       # unity requant: output == raw accumulator

    a = np.ones((M, K), dtype=np.int8)    # all-ones activations
    w = np.zeros((N, K), dtype=np.int8)
    w[n0, k0] = 1                          # single non-zero weight

    acc = a.astype(np.int64) @ w.astype(np.int64).T
    expected = np.clip(acc, -128, 127).astype(np.int8)   # column n0 == 1, others 0

    b = bo.flatbuffers.Builder(4096)
    buffers = [bo.buf_empty(b), bo.buf_data(b, np.zeros(N, dtype=np.int32)), bo.buf_empty(b)]

    t_in = bo.tensor(b, "input", [M, K], bo.I8, 0, bo.quant_tensor(b, [s], [0]))
    t_w = bo.tensor(b, "weights", [N, K], bo.I8, 0, bo.quant_tensor(b, [s], [0]))
    t_b = bo.tensor(b, "bias", [N], bo.I32, 1, bo.quant_tensor(b, [o_scale], [0]))
    t_out = bo.tensor(b, "output", [M, N], bo.I8, 2, bo.quant_tensor(b, [o_scale], [0]))

    S = bo.S
    S.FullyConnectedOptionsStart(b)
    S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)
    S.FullyConnectedOptionsAddWeightsFormat(b, 0)
    S.FullyConnectedOptionsAddKeepNumDims(b, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
    opt = S.FullyConnectedOptionsEnd(b)

    op = bo.operator(b, 0, [0, 1, 2], [3], bo.BuiltinOptions.FullyConnectedOptions, opt)
    blob = bo.finish(b, buffers, [bo.opcode(b, bo.BuiltinOperator.FULLY_CONNECTED)],
                     [t_in, t_w, t_b, t_out], [op], [0, 1], [3],
                     f"e102 col{n0} dynB {M}x{K}x{N} int8")

    os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
    open(os.path.join(out_dir, "model.tflite"), "wb").write(blob)
    a.tofile(os.path.join(out_dir, "golden", "input.bin"))
    w.tofile(os.path.join(out_dir, "golden", "weights.bin"))
    expected.tofile(os.path.join(out_dir, "golden", "expected.bin"))

    meta = {
        "op": "FULLY_CONNECTED (weight-binding probe, E-102)",
        "shape": {"M": M, "K": K, "N": N},
        "dtype": "int8",
        "dynamic_weights": True,
        "inputs": [
            {"name": "input", "shape": [M, K], "type": "int8"},
            {"name": "weights", "shape": [N, K], "type": "int8"},
        ],
        "output": {"name": "output", "shape": [M, N], "type": "int8"},
        "quant": {"input": s, "weights": s, "bias": o_scale, "output": o_scale},
        "int8_tol": 0,
        "weight_single": {"n0": n0, "k0": k0},
        "hypotheses": {
            "H1": "content-only: column n0 == 1, all other columns == 0, runs reproducible",
            "H2": "address leak: wrong column / shifted output when IOVA differs",
        },
        "reference": "int64_exact",
        "seed": SEED,
        "tflite_bytes": len(blob),
    }
    json.dump(meta, open(os.path.join(out_dir, "model.json"), "w"), indent=2)
    print(f"{out_dir}: W[{n0},{k0}]=1 -> expected column {n0} == 1, others 0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n0", type=int, required=True)
    ap.add_argument("--k0", type=int, required=True)
    a = ap.parse_args()
    main(a.out, a.n0, a.k0)
