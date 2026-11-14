#!/usr/bin/env python3
"""Softmax [1,16,16] int8 — one model, one purpose: give the self-built submitter a softmax
asset for the attention chain (T=16 so each row is 16 B and the 16-byte boundary rule holds).

    /usr/bin/python3 tools/build_softmax_t16.py --out ops/softmax_1x16x16_int8
"""
import argparse
import json
import os

import numpy as np

import build_ops as bo

T = 16
S_IN = 1.0 / 16.0     # scores are QK^T int8 codes; a plausible scale for |s| <= 8
S_OUT = 1.0 / 256.0





def main(out_dir):
    b = bo.flatbuffers.Builder(1 << 16)
    S = bo.S
    buf0 = bo.buf_empty(b)
    buf1 = bo.buf_empty(b)
    q_in = bo.quant_tensor(b, [S_IN], [0])
    q_out = bo.quant_tensor(b, [S_OUT], [-128])   # TFLite int8 softmax: zp = -128
    t_in = bo.tensor(b, "input", [1, T, T], bo.I8, 0, q_in)
    t_out = bo.tensor(b, "output", [1, T, T], bo.I8, 1, q_out)

    oc = bo.opcode(b, bo.BuiltinOperator.SOFTMAX)
    S.SoftmaxOptionsStart(b)
    S.SoftmaxOptionsAddBeta(b, 1.0)
    opt = S.SoftmaxOptionsEnd(b)
    op = bo.operator(b, oc, [0], [1], bo.BuiltinOptions.SoftmaxOptions, opt)

    blob = bo.finish(b, [buf0, buf1], [oc], [t_in, t_out], [op], [0], [1],
                     "softmax [1,16,16] int8")

    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "model.tflite"), "wb").write(blob)
    rng = np.random.default_rng(7)
    x = rng.integers(-40, 41, size=(1, T, T), dtype=np.int8)   # |code| <= 40 -> |s| <= 2.5
    x.tofile(os.path.join(out_dir, "golden_input.bin"))
    # independent reference in float64, then quantised with the model's output scale
    s = x.astype(np.float64) * S_IN
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    p = e / e.sum(axis=-1, keepdims=True)
    ref = np.clip(np.round(p / S_OUT) - 128, -128, 127).astype(np.int8)
    os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
    x.tofile(os.path.join(out_dir, "golden", "input.bin"))
    ref.tofile(os.path.join(out_dir, "golden", "expected.bin"))
    json.dump({"op": "SOFTMAX", "shape": {"T": T}, "dtype": "int8",
               "quant": {"in_scale": S_IN, "out_scale": S_OUT},
               "inputs": [{"name": "input", "shape": [1, T, T], "type": "int8"}],
               "output": {"name": "output", "shape": [1, T, T], "type": "int8"},
               "int8_tol": 1},
              open(os.path.join(out_dir, "model.json"), "w"), indent=2)
    print(f"{out_dir}: softmax [1,{T},{T}] int8, {len(blob)} B")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ops/softmax_1x16x16_int8")
    main(ap.parse_args().out)
