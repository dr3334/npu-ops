#!/usr/bin/env python3
"""Compile-only probes for the LM head dimension limit (ticket 08).

The monolithic FULLY_CONNECTED (N=151936) was rejected by ncc with

    MDLA: Dimension should be <= 65535. Operand: 1 got <151936 x 896>.
    MDLA: Dimension should be <= 65535. Result : 0 got <8 x 151936>.

so the question is not "does it fit in memory" but "which tensors may carry a
dimension > 65535". Each probe below isolates one hypothesis; all are compile-only
(no golden, no board):

  fc_65535        FC 8x896x65535        boundary accept?  (117 MB weights)
  fc_65536        FC 8x896x65536        boundary reject?
  fc_37984        FC 8x896x37984        the chunk size the fallback needs
  concat_151936   CONCAT(axis=1) 4x[8,37984] -> [8,151936]
  concat_75968    CONCAT(axis=1) 2x[8,37984] -> [8,75968]
  concat_65535    CONCAT(axis=1) 2x[8,32768] -> [8,65535]
  concat_65536    CONCAT(axis=1) 2x[8,32768] -> [8,65536]
  reshape_151936  RESHAPE [32,37984] -> [8,151936]
  reshape_big3d   RESHAPE [32,37984] -> [8,4,37984]   (control: all dims <= 65535)

Usage (WSL):
    /usr/bin/python3 ops/lm_head_t8_fp16/probe/make_probes.py --outdir ops/lm_head_t8_fp16/probe
"""
import argparse
import os
import sys

import numpy as np
import tflite as S

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "tools")))
import build_ops as bo

F16, I32 = bo.F16, bo.I32
M, K = 8, 896


def save(outdir, name, blob):
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "model.tflite")
    with open(p, "wb") as f:
        f.write(blob)
    print("%-16s %10d B  %s" % (name, len(blob), p))


def fc_probe(n):
    b = bo.flatbuffers.Builder(1 << 16) if hasattr(bo, "flatbuffers") else None
    import flatbuffers
    b = flatbuffers.Builder(max(1 << 16, n * K * 2 + (1 << 20)))
    buffers = [bo.buf_empty(b)]
    tensors = []
    opcodes = [bo.opcode(b, bo.BuiltinOperator.FULLY_CONNECTED)]
    t_in = bo.tensor(b, "input", [M, K], F16, 0)
    w = np.zeros((n, K), dtype=np.float16)          # values irrelevant: compile-only probe
    buffers.append(bo.buf_data(b, w))
    t_w = bo.tensor(b, "weights", [n, K], F16, 1)
    buffers.append(bo.buf_data(b, np.zeros(n, dtype=np.float16)))
    t_b = bo.tensor(b, "bias", [n], F16, 2)
    buffers.append(bo.buf_empty(b))
    t_out = bo.tensor(b, "output", [M, n], F16, 3)
    tensors.extend([t_in, t_w, t_b, t_out])
    S.FullyConnectedOptionsStart(b)
    S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)
    S.FullyConnectedOptionsAddWeightsFormat(b, 0)
    S.FullyConnectedOptionsAddKeepNumDims(b, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
    opt = S.FullyConnectedOptionsEnd(b)
    op = bo.operator(b, 0, [0, 1, 2], [3], bo.BuiltinOptions.FullyConnectedOptions, opt)
    return bo.finish(b, buffers, opcodes, tensors, [op], [0], [3],
                     "probe fc %dx%dx%d %s" % (M, K, n, "fp16"))


def concat_probe(parts, n_each):
    import flatbuffers
    b = flatbuffers.Builder(1 << 16)
    buffers = [bo.buf_empty(b)]
    tensors = []
    opcodes = [bo.opcode(b, bo.BuiltinOperator.CONCATENATION)]
    ins = []
    for i in range(parts):
        tensors.append(bo.tensor(b, "in%d" % i, [M, n_each], F16, 0))
        ins.append(len(tensors) - 1)
    buffers.append(bo.buf_empty(b))
    t_out = bo.tensor(b, "out", [M, parts * n_each], F16, 1)
    tensors.append(t_out)
    S.ConcatenationOptionsStart(b)
    S.ConcatenationOptionsAddAxis(b, 1)
    S.ConcatenationOptionsAddFusedActivationFunction(b, 0)
    opt = S.ConcatenationOptionsEnd(b)
    op = bo.operator(b, 0, ins, [len(tensors) - 1],
                     bo.BuiltinOptions.ConcatenationOptions, opt)
    return bo.finish(b, buffers, opcodes, tensors, [op], ins, [len(tensors) - 1],
                     "probe concat %d x [%d,%d]" % (parts, M, n_each))


def reshape_probe(in_shape, out_shape):
    import flatbuffers
    b = flatbuffers.Builder(1 << 16)
    buffers = [bo.buf_empty(b)]
    tensors = []
    opcodes = [bo.opcode(b, bo.BuiltinOperator.RESHAPE)]
    tensors.append(bo.tensor(b, "input", list(in_shape), F16, 0))
    nsv = b.CreateNumpyVector(np.asarray(out_shape, np.int32))       # before the table
    buffers.append(bo.buf_data(b, np.asarray(out_shape, np.int32)))
    tensors.append(bo.tensor(b, "new_shape", [len(out_shape)], I32, 1))
    buffers.append(bo.buf_empty(b))
    tensors.append(bo.tensor(b, "output", list(out_shape), F16, 2))
    S.ReshapeOptionsStart(b)
    S.ReshapeOptionsAddNewShape(b, nsv)
    opt = S.ReshapeOptionsEnd(b)
    op = bo.operator(b, 0, [0, 1], [2], bo.BuiltinOptions.ReshapeOptions, opt)
    return bo.finish(b, buffers, opcodes, tensors, [op], [0], [2],
                     "probe reshape %s->%s" % (in_shape, out_shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join("ops", "lm_head_t8_fp16", "probe"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    for n in (65535, 65536, 37984):
        save(a.outdir, "fc_%d" % n, fc_probe(n))
    save(a.outdir, "concat_151936", concat_probe(4, 37984))
    save(a.outdir, "concat_75968", concat_probe(2, 37984))
    save(a.outdir, "concat_65535", concat_probe(2, 32768))
    save(a.outdir, "concat_65536", concat_probe(2, 32768))
    save(a.outdir, "reshape_151936", reshape_probe([32, 37984], [8, 151936]))
    save(a.outdir, "reshape_big3d", reshape_probe([32, 37984], [8, 4, 37984]))


if __name__ == "__main__":
    main()
