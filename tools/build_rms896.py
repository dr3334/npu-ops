#!/usr/bin/env python3
"""D=896 RMSNorm —— 三种均值实现对照（Qwen2.5-0.5B 的真实 hidden 宽度）。

A: MEAN 路径（我们现有）：MUL -> MEAN(axis=-1) -> ADD eps -> RSQRT -> MUL -> MUL(gamma)
B: FC tile 路径（bestkojima/mtk-npug-handoff 的 tiled_statistics）：
   MUL -> FC[32,896](W=1/896) -> [T,32] -> CONCAT x28 -> [T,896] -> RSQRT -> MUL -> MUL(gamma)
C: 扩张 FC（他们报告在 D2048 上触发 APUSYS I/O 错误的做法，这里作为负对照）：
   MUL -> FC[32,896] -> [T,32] -> FC[896,32] -> [T,896] -> RSQRT -> MUL -> MUL(gamma)

    /usr/bin/python3 tools/build_rms896.py --variant A|B|C --T 8 --out ops/<name>
"""
import argparse
import json
import os

import numpy as np

import build_ops as bo

D = 896
EPS = 1e-5


def main(variant, T, out_dir):
    rng = np.random.default_rng(20261113)
    x = rng.normal(0, 0.6, (T, D)).astype(np.float32)
    gamma = rng.uniform(0.8, 1.2, (D,)).astype(np.float32)

    b = bo.flatbuffers.Builder(1 << 22)
    S = bo.S
    F16, I32 = bo.F16, bo.I32

    buffers = [bo.buf_empty(b)]
    tensors, opcodes, operators, idx = [], [], [], {}

    def const_buf(arr):
        """只登记 buffer，返回 buffer 索引（不是 tensor 索引——之前混用导致图坏了）。"""
        buffers.append(bo.buf_data(b, np.asarray(arr)))
        return len(buffers) - 1

    def const(arr, ttype=F16, shape=None, name=None):
        bi = const_buf(arr)
        a = np.asarray(arr)
        tensors.append(bo.tensor(b, name or f"c{bi}", list(shape if shape else a.shape), ttype, bi))
        return len(tensors) - 1

    def act(name, shape):
        tensors.append(bo.tensor(b, name, list(shape), F16, 0))
        idx[name] = len(tensors) - 1
        return idx[name]

    def oc(kind, version=1):
        opcodes.append(bo.opcode(b, kind, version))
        return len(opcodes) - 1

    OC_FC = oc(bo.BuiltinOperator.FULLY_CONNECTED)
    OC_MUL = oc(bo.BuiltinOperator.MUL)
    OC_ADD = oc(bo.BuiltinOperator.ADD)
    OC_MEAN = oc(bo.BuiltinOperator.MEAN)
    OC_RSQRT = oc(bo.BuiltinOperator.RSQRT)
    OC_CONCAT = oc(bo.BuiltinOperator.CONCATENATION)

    act("x", (T, D))

    def mul(a, bb, out, shape):
        t = act(out, shape)
        S.MulOptionsStart(b); S.MulOptionsAddFusedActivationFunction(b, 0); o = S.MulOptionsEnd(b)
        operators.append(bo.operator(b, OC_MUL, [idx[a], idx[bb]], [t], bo.BuiltinOptions.MulOptions, o))
        return t

    def fc(inp, out, w, n, k):
        t_w = bo.tensor(b, f"W_{out}", [n, k], F16, const_buf(np.asarray(w, np.float16)))
        tensors.append(t_w)
        ti_w = len(tensors) - 1
        # TFLite FULLY_CONNECTED 必须有第三个输入 bias（可以为空张量）：之前漏了它，
        # importer 直接拒（"Fail to process the given model"）。
        t_b = bo.tensor(b, f"B_{out}", [n], F16, const_buf(np.zeros(n, np.float16)))
        tensors.append(t_b)
        ti_b = len(tensors) - 1
        t = act(out, (T, n))
        S.FullyConnectedOptionsStart(b)
        S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)
        S.FullyConnectedOptionsAddWeightsFormat(b, 0)
        S.FullyConnectedOptionsAddKeepNumDims(b, True)
        S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
        o = S.FullyConnectedOptionsEnd(b)
        operators.append(bo.operator(b, OC_FC, [idx[inp], ti_w, ti_b], [t],
                                    bo.BuiltinOptions.FullyConnectedOptions, o))
        return t

    # x^2
    mul("x", "x", "sq", (T, D))

    if variant == "A0":   # A 去掉 eps（诊断：ADD 报 type 不兼容）
        t_m = act("mean", (T, 1))
        S.ReducerOptionsStart(b); S.ReducerOptionsAddKeepDims(b, True); ro = S.ReducerOptionsEnd(b)
        ax = const(np.asarray([1], np.int32), I32, [1])
        operators.append(bo.operator(b, OC_MEAN, [idx["sq"], ax], [t_m], bo.BuiltinOptions.ReducerOptions, ro))
        idx["mean"] = len(tensors) - 1
        t_r = act("rsqrt", (T, 1))
        operators.append(bo.operator(b, OC_RSQRT, [idx["mean"]], [t_r]))
        idx["rsqrt"] = len(tensors) - 1
    elif variant == "B0":  # 只到统计 tile：[T,32]
        tile_w = np.full((32, D), np.float32(1.0 / D), np.float32)
        fc("sq", "tile", tile_w, 32, D)
    elif variant == "A":
        t_m = act("mean", (T, 1))
        S.ReducerOptionsStart(b); S.ReducerOptionsAddKeepDims(b, True); ro = S.ReducerOptionsEnd(b)
        ax = const(np.asarray([1], np.int32), I32, [1])
        operators.append(bo.operator(b, OC_MEAN, [idx["sq"], ax], [t_m], bo.BuiltinOptions.ReducerOptions, ro))
        idx["mean"] = len(tensors) - 1
        t_e = act("mean_eps", (T, 1))
        eps = const(np.full((T, 1), EPS, np.float16))
        S.AddOptionsStart(b); S.AddOptionsAddFusedActivationFunction(b, 0); ao = S.AddOptionsEnd(b)
        operators.append(bo.operator(b, OC_ADD, [idx["mean"], eps], [t_e], bo.BuiltinOptions.AddOptions, ao))
        idx["mean_eps"] = len(tensors) - 1
        t_r = act("rsqrt", (T, 1))
        operators.append(bo.operator(b, OC_RSQRT, [idx["mean_eps"]], [t_r]))
        idx["rsqrt"] = len(tensors) - 1
    else:
        tile_w = np.full((32, D), np.float32(1.0 / D), np.float32)
        fc("sq", "tile", tile_w, 32, D)
        if variant == "B":
            copies = D // 32            # 896 = 28 * 32，正好整除
            t_c = act("tile_exp", (T, D))
            S.ConcatenationOptionsStart(b)
            S.ConcatenationOptionsAddAxis(b, 1)
            S.ConcatenationOptionsAddFusedActivationFunction(b, 0)
            co = S.ConcatenationOptionsEnd(b)
            operators.append(bo.operator(b, OC_CONCAT, [idx["tile"]] * copies, [t_c],
                                        bo.BuiltinOptions.ConcatenationOptions, co))
            idx["tile_exp"] = len(tensors) - 1
            t_r = act("rsqrt", (T, D))
            operators.append(bo.operator(b, OC_RSQRT, [idx["tile_exp"]], [t_r]))
            idx["rsqrt"] = len(tensors) - 1
        else:  # C: 扩张 FC（负对照）
            exp_w = np.full((D, 32), np.float32(1.0), np.float32)
            fc("tile", "tile_exp", exp_w, D, 32)
            t_r = act("rsqrt", (T, D))
            operators.append(bo.operator(b, OC_RSQRT, [idx["tile_exp"]], [t_r]))
            idx["rsqrt"] = len(tensors) - 1

    if variant in ("B0",):
        blob = bo.finish(b, buffers, opcodes, tensors, operators, [idx["x"]], [idx["tile"]],
                         "bisect: FC statistics tile only")
        os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
        open(os.path.join(out_dir, "model.tflite"), "wb").write(blob)
        x.astype(np.float16).tofile(os.path.join(out_dir, "golden", "input.bin"))
        ((x * x).mean(axis=-1, keepdims=True) * np.ones((1, 32))).astype(np.float16).tofile(
            os.path.join(out_dir, "golden", "expected.bin"))
        json.dump({"op": "RMS896_B0", "shape": {"T": T, "D": D}, "dtype": "float16",
                   "atol": 2e-2, "rtol": 2e-2,
                   "inputs": [{"name": "x", "shape": [T, D], "type": "float16"}],
                   "output": {"name": "tile", "shape": [T, 32], "type": "float16"}},
                  open(os.path.join(out_dir, "model.json"), "w"), indent=2)
        print(f"{out_dir}: B0 tile-only {len(blob)} B")
        return

    mul("x", "rsqrt", "scaled", (T, D))
    gam = const(np.asarray(gamma, np.float16), F16, [D])
    t_o = act("out", (T, D))
    S.MulOptionsStart(b); S.MulOptionsAddFusedActivationFunction(b, 0); mo = S.MulOptionsEnd(b)
    operators.append(bo.operator(b, OC_MUL, [idx["scaled"], gam], [t_o], bo.BuiltinOptions.MulOptions, mo))
    idx["out"] = len(tensors) - 1

    blob = bo.finish(b, buffers, opcodes, tensors, operators, [idx["x"]], [idx["out"]],
                     f"rmsnorm D={D} variant {variant} T={T} fp16")

    # 独立参考（float32，按 fp16 舍入）
    r = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + (0.0 if variant == "A0" else EPS))
    ref = (x * r * gamma).astype(np.float16)

    os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
    open(os.path.join(out_dir, "model.tflite"), "wb").write(blob)
    x.astype(np.float16).tofile(os.path.join(out_dir, "golden", "input.bin"))
    ref.tofile(os.path.join(out_dir, "golden", "expected.bin"))
    json.dump({"op": f"RMSNORM_D896_{variant}", "shape": {"T": T, "D": D}, "dtype": "float16",
               "atol": 2e-2, "rtol": 2e-2,
               "inputs": [{"name": "x", "shape": [T, D], "type": "float16"}],
               "output": {"name": "out", "shape": [T, D], "type": "float16"},
               "note": f"variant {variant}: A=MEAN, B=FC-tile+CONCAT, C=expansion FC (negative control)",
               "tflite_bytes": len(blob)},
              open(os.path.join(out_dir, "model.json"), "w"), indent=2)
    print(f"{out_dir}: variant {variant}, tflite {len(blob)} B, T={T} D={D}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "A0", "B", "B0", "C"])
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.variant, a.T, a.out)
