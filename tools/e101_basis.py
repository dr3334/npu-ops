#!/usr/bin/env python3
"""E-101 — basis-vector layout probe (predictions written BEFORE the run).

Question: for out[m,n] = sum_k A[m,k]*W[n,k], does the device read K in plain order?

Design (dynamic-B model, so we can change W and A at runtime without recompiling):
  * W[n,k] = k            (each weight row carries the K index itself)
  * A row r = e_{j_r}     (one-hot at position j_r, eight different j in one run)
  * output scale == input*weight scale (multiplier 1), zero bias -> out must be the raw INT accumulator

  => out[r, :] must be the constant j_r for every n.

Competing hypotheses and their predictions:
  H1 plain row-major K reading  : out[r,:] == j_r for ALL j in {0,1,15,16,31,32,47,63}
  H2 K is tiled/permuted        : out[r,:] == perm(j_r); rows whose j is inside the first tile are
                                  correct, later tiles map to a different index
  H3 weights are repacked with
     an index/offset error       : out[r,:] == j_r + c (constant shift) or == some other column,
                                  but the same for all j (no permutation)

Any of those three is decidable from the 8 rows in one run.

    /usr/bin/python3 tools/e101_basis.py --out ops/e101_basis_dynb
"""
import argparse
import json
import os

import numpy as np

import build_ops as bo

SEED = 20261112
DEFAULT_J = [0, 1, 15, 16, 31, 32, 47, 63]
M, K, N = 8, 64, 64


def main(out_dir, j_list=None, k_dim=64):
    global M, K, N
    J = j_list if j_list else DEFAULT_J
    M, K, N = len(J), k_dim, 64
    rng = np.random.default_rng(SEED)          # kept for future variants; unused here
    s = 1.0 / 127.0
    o_scale = s * s                            # multiplier 1 -> raw accumulator, no requant

    a = np.zeros((M, K), dtype=np.int8)
    for r, j in enumerate(J):
        a[r, j] = 1

    w = np.zeros((N, K), dtype=np.int8)
    for k in range(K):
        w[:, k] = k                            # W[n,k] = k

    acc = a.astype(np.int64) @ w.astype(np.int64).T      # exact INT64 reference
    expected = np.clip(acc, -128, 127).astype(np.int8)   # == j_r broadcast over n

    b = bo.flatbuffers.Builder(4096)
    buffers = [bo.buf_empty(b)]
    buffers.append(bo.buf_empty(b))                      # 1: bias placeholder (zero bias below)
    buffers.append(bo.buf_empty(b))                      # 2: output placeholder
    bias = np.zeros(N, dtype=np.int32)
    buffers[1] = bo.buf_data(b, bias)

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
                     f"e101 basis dynB {M}x{K}x{N} int8")

    os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
    open(os.path.join(out_dir, "model.tflite"), "wb").write(blob)
    a.tofile(os.path.join(out_dir, "golden", "input.bin"))
    w.tofile(os.path.join(out_dir, "golden", "weights.bin"))
    expected.tofile(os.path.join(out_dir, "golden", "expected.bin"))

    meta = {
        "op": "FULLY_CONNECTED (basis-vector layout probe, E-101)",
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
        "j_positions": J,
        "hypotheses": {
            "H1": "plain K order: every row == j",
            "H2": "K tiled/permuted: rows beyond tile 0 map to another index",
            "H3": "weight repack offset: row == j + c, same for all j",
        },
        "reference": "int64_exact",
        "seed": SEED,
        "tflite_bytes": len(blob),
    }
    json.dump(meta, open(os.path.join(out_dir, "model.json"), "w"), indent=2)
    print(f"{out_dir}: tflite={len(blob)}B  J={J}  expected row r = j_r")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ops/e101_basis_dynb")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--j", default="")
    a = ap.parse_args()
    jl = [int(x) for x in a.j.split(",")] if a.j else None
    main(a.out, jl, a.k)
