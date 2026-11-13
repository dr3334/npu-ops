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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    spec = json.load(open(args.spec))
    meta = build(spec, args.out)
    print(json.dumps({"ok": True, "model": os.path.join(args.out, "model.tflite"),
                      "shape": meta["shape"], "op": meta["op"],
                      "requant_ratio": meta["requant_ratio"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
