#!/usr/bin/env python3
"""graph 规格 -> model.tflite + model.json + golden/{a,b,expected}.bin

由 Graph.compile() 调起（WSL 的 /usr/bin/python3）：

    /usr/bin/python3 -m npu_lib.build_model --spec <spec.json> --out <dir>

本文件**不**做数值参考：golden 由 golden.py（int64 + quant_contract）算，flatbuffer 只承载
形状与量化参数（规则 R3：参考实现独立于生成器）。
"""
import argparse
import json
import os
import sys

import numpy as np

from . import _tools
from . import golden as golden_mod

_tools.ensure()
import flatbuffers          # noqa: E402
import build_ops as bo      # noqa: E402  (公共工具，只 import)

S = bo.S
OPS = {
    "add": ("ADD", "AddOptions"),
    "sub": ("SUB", "SubOptions"),
    "mul": ("MUL", "MulOptions"),
}


def _options(b, kind):
    start = getattr(S, kind + "Start")
    add = getattr(S, kind + "AddFusedActivationFunction")
    end = getattr(S, kind + "End")
    start(b)
    add(b, 0)                      # NONE：否则 ncc 会把 ReLU/ReLU6 折进逐元素层
    return getattr(bo.BuiltinOptions, kind), end(b)


def code_inputs(op, shape, seed):
    """确定性的 int8 输入码 + 注入的边缘用例（钳位/进位边界）。"""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    if op == "mul":
        lo, hi = -12, 12           # a*b 大多留在 int8 内，只看尾巴饱和
        pairs = [(12, 12), (-12, 12), (12, -12), (-11, 12), (0, -12), (11, -11), (1, 1), (-1, 12)]
    else:
        lo, hi = -100, 100
        pairs = [(100, -100), (-100, 100), (-100, -100), (100, 100),
                 (7, 7), (-7, -7), (0, -100), (-100, 0)]
    a = rng.integers(lo, hi + 1, size=n).astype(np.int8)
    b = rng.integers(lo, hi + 1, size=n).astype(np.int8)
    for i, (pa, pb) in enumerate(pairs[:n]):
        a[i] = pa
        b[i] = pb
    return a, b


def build(spec, outdir):
    op = spec["op"]
    if op == "fc":
        return build_fc(spec, outdir)
    shape = tuple(spec["shape"])
    n = int(np.prod(shape))
    sa = float(spec["in_scale"])
    sb = float(spec["in1_scale"])
    so = float(spec["out_scale"])
    seed = int(spec.get("seed", 20261113))
    builtin, opt_kind = OPS[op]

    a, b = code_inputs(op, shape, seed)
    expected, acc, ratio = golden_mod.reference(op, a, b, sa, sb, so)

    bb = flatbuffers.Builder(4096)
    buffers = [bo.buf_empty(bb), bo.buf_empty(bb)]
    t_a = bo.tensor(bb, spec["in_name"], list(shape), bo.I8, 0, bo.quant_tensor(bb, [sa], [0]))
    t_b = bo.tensor(bb, spec["in1_name"], list(shape), bo.I8, 0, bo.quant_tensor(bb, [sb], [0]))
    t_o = bo.tensor(bb, spec["out_name"], list(shape), bo.I8, 1, bo.quant_tensor(bb, [so], [0]))
    opt_type, opt = _options(bb, opt_kind)
    op_tbl = bo.operator(bb, 0, [0, 1], [2], opt_type, opt)
    blob = bo.finish(bb, buffers, [bo.opcode(bb, getattr(bo.BuiltinOperator, builtin))],
                     [t_a, t_b, t_o], [op_tbl], [0, 1], [2],
                     "npu_lib %s %s int8" % (builtin, "x".join(str(d) for d in shape)))

    os.makedirs(os.path.join(outdir, "golden"), exist_ok=True)
    with open(os.path.join(outdir, "model.tflite"), "wb") as f:
        f.write(blob)
    a.tofile(os.path.join(outdir, "golden", spec["in_name"] + ".bin"))
    b.tofile(os.path.join(outdir, "golden", spec["in1_name"] + ".bin"))
    expected.tofile(os.path.join(outdir, "golden", "expected.bin"))

    meta = {
        "name": spec["name"],
        "op": builtin,
        "shape": list(shape),
        "dtype": "int8",
        "inputs": [
            {"name": spec["in_name"], "shape": list(shape), "type": "int8", "scale": sa},
            {"name": spec["in1_name"], "shape": list(shape), "type": "int8", "scale": sb},
        ],
        "output": {"name": spec["out_name"], "shape": list(shape), "type": "int8", "scale": so},
        "requant_ratio": ratio,
        "requant_note": (
            "ratio == 1 by construction (out_scale == in_scale for ADD/SUB, "
            "out_scale == sa*sb for MUL): the device kernel is a pure integer op + clamp, "
            "so int8_tol = 0 and the golden is bit-exact. Requant != 1 for elementwise has "
            "no board evidence yet (see docs/OWN-SUBMITTER-LIB.md)."),
        "int8_tol": 0,
        "reference": "int64_acc + tools/quant_contract.requant_int8",
        "seed": seed,
        "tflite_bytes": len(blob),
    }
    with open(os.path.join(outdir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _fc_codes(m, k, n, seed):
    """确定性的满量程 int8 码（激活 ±127、权重 ±127）。bias 全 0：隔离权重路径。"""
    rng = np.random.default_rng(seed)
    a = rng.integers(-127, 128, size=(m, k)).astype(np.int8)
    w = rng.integers(-127, 128, size=(n, k)).astype(np.int8)
    return a, w


def build_fc(spec, outdir):
    """FULLY_CONNECTED，**权重是第二输入**（动态权重）。

    形状：x[M,K] × w[N,K] -> out[M,N]（TFLite 的 FC 布局：out[m,n] = sum_k x[m,k]*w[n,k]）。
    为什么不用常量权重：常量区在自建提交器上只有恰好落在抓取那次的 IOVA 才对
    （docs/ISSUE-const-fc.md 的 3/11），动态权重有板端证据（docs/OWN-SUBMITTER.md §6）。
    golden 走 golden.reference（int64 累加 + quant_contract 两步舍入），与 flatbuffer 无关。
    """
    sh = spec["shapes"]
    a_shape = tuple(int(x) for x in sh["a"])
    w_shape = tuple(int(x) for x in sh["b"])
    o_shape = tuple(int(x) for x in sh["out"])
    m, k = a_shape
    n, k2 = w_shape
    if k2 != k or o_shape != (m, n):
        raise ValueError("fc: shapes disagree a=%r w=%r out=%r" % (a_shape, w_shape, o_shape))
    sa = float(spec["in_scale"])
    sw = float(spec["in1_scale"])
    so = float(spec["out_scale"])
    seed = int(spec.get("seed", 20261113))

    a, w = _fc_codes(m, k, n, seed)
    bias = np.zeros(n, dtype=np.int32)
    expected, acc, ratio = golden_mod.reference("fc", a, w, sa, sw, so)

    bb = flatbuffers.Builder(4096)
    buffers = [bo.buf_empty(bb)]                 # 0: IO 占位（输入张量不得带数据）
    buffers.append(bo.buf_data(bb, bias))        # 1: bias 常量（全 0）
    buffers.append(bo.buf_empty(bb))             # 2: 输出占位
    t_a = bo.tensor(bb, spec["in_name"], list(a_shape), bo.I8, 0, bo.quant_tensor(bb, [sa], [0]))
    t_w = bo.tensor(bb, spec["in1_name"], list(w_shape), bo.I8, 0, bo.quant_tensor(bb, [sw], [0]))
    t_b = bo.tensor(bb, spec["out_name"] + "_bias", [n], bo.I32, 1,
                    bo.quant_tensor(bb, [sa * sw], [0]))
    t_o = bo.tensor(bb, spec["out_name"], list(o_shape), bo.I8, 2, bo.quant_tensor(bb, [so], [0]))
    S.FullyConnectedOptionsStart(bb)
    S.FullyConnectedOptionsAddFusedActivationFunction(bb, 0)
    S.FullyConnectedOptionsAddWeightsFormat(bb, 0)
    S.FullyConnectedOptionsAddKeepNumDims(bb, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(bb, False)
    opt = S.FullyConnectedOptionsEnd(bb)
    op_tbl = bo.operator(bb, 0, [0, 1, 2], [3], bo.BuiltinOptions.FullyConnectedOptions, opt)
    blob = bo.finish(bb, buffers, [bo.opcode(bb, bo.BuiltinOperator.FULLY_CONNECTED)],
                     [t_a, t_w, t_b, t_o], [op_tbl], [0, 1], [3],
                     "npu_lib FC %dx%dx%d int8 dynW" % (m, k, n))

    os.makedirs(os.path.join(outdir, "golden"), exist_ok=True)
    with open(os.path.join(outdir, "model.tflite"), "wb") as f:
        f.write(blob)
    a.tofile(os.path.join(outdir, "golden", spec["in_name"] + ".bin"))
    w.tofile(os.path.join(outdir, "golden", spec["in1_name"] + ".bin"))
    expected.tofile(os.path.join(outdir, "golden", "expected.bin"))

    meta = {
        "name": spec["name"],
        "op": "FULLY_CONNECTED",
        "dynamic_weights": True,
        "shapes": {"a": list(a_shape), "b": list(w_shape), "out": list(o_shape)},
        "dtype": "int8",
        "inputs": [
            {"name": spec["in_name"], "shape": list(a_shape), "type": "int8", "scale": sa},
            {"name": spec["in1_name"], "shape": list(w_shape), "type": "int8", "scale": sw},
        ],
        "output": {"name": spec["out_name"], "shape": list(o_shape), "type": "int8",
                   "scale": so},
        "requant_ratio": ratio,
        "requant_note": ("s_out = s_a*s_w*32 by construction of the caller, so the device "
                         "multiplier is 1/32; golden = int64 acc + quant_contract"),
        "int8_tol": 0,
        "reference": "int64_acc(a @ w.T) + tools/quant_contract.requant_int8",
        "seed": seed,
        "tflite_bytes": len(blob),
    }
    with open(os.path.join(outdir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    spec = json.load(open(args.spec))
    meta = build(spec, args.out)
    print(json.dumps({"ok": True, "model": os.path.join(args.out, "model.tflite"),
                      "shape": meta.get("shape") or meta.get("shapes"), "op": meta["op"],
                      "requant_ratio": meta["requant_ratio"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
