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
        """常量张量。常量**不登记进 idx**（只有 act() 登记），所以不可能有名字指到常量上。"""
        bi = const_buf(arr)
        a = np.asarray(arr)
        return push_t(bo.tensor(b, name or f"c{bi}", list(shape if shape else a.shape), ttype, bi))

    def push_t(t):
        """唯一一处 `len(tensors) - 1`：把张量塞进表并返回它的索引。"""
        tensors.append(t)
        return len(tensors) - 1

    def act(name, shape):
        t = push_t(bo.tensor(b, name, list(shape), F16, 0))
        if name in idx:
            # 同名张量重复登记会静默改掉已有索引，随后所有"按名字取输入"的算子都会拿错张量。
            raise KeyError(f"张量名 {name!r} 已登记（索引 {idx[name]}），拒绝覆盖")
        idx[name] = t
        return t

    def op(kind, ins, outs, options_type=None, options=None):
        """建一条 operator。ins/outs 写张量名（查 idx）或写张量索引（const()/act() 的返回值）。

        旧代码在 act()/const() 之后手写 `idx[name] = len(tensors) - 1`：只要中间又追加了
        一个张量，这个名字就指到那个新张量上——D=896 变体 A 的 RSQRT 吃到 int32 axis、
        ADD 吃到 eps 常量，都是这一处造成的。现在索引只来自 act()/const() 的返回值。
        """
        def resolve(v):
            return idx[v] if isinstance(v, str) else int(v)
        operators.append(bo.operator(b, kind, [resolve(v) for v in ins],
                                     [resolve(v) for v in outs], options_type, options))

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
        op(OC_MUL, [a, bb], [out], bo.BuiltinOptions.MulOptions, o)
        return t

    def fc(inp, out, w, n, k):
        ti_w = push_t(bo.tensor(b, f"W_{out}", [n, k], F16, const_buf(np.asarray(w, np.float16))))
        # TFLite FULLY_CONNECTED 必须有第三个输入 bias（可以为空张量）：之前漏了它，
        # importer 直接拒（"Fail to process the given model"）。
        ti_b = push_t(bo.tensor(b, f"B_{out}", [n], F16, const_buf(np.zeros(n, np.float16))))
        t = act(out, (T, n))
        S.FullyConnectedOptionsStart(b)
        S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)
        S.FullyConnectedOptionsAddWeightsFormat(b, 0)
        S.FullyConnectedOptionsAddKeepNumDims(b, True)
        S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
        o = S.FullyConnectedOptionsEnd(b)
        op(OC_FC, [inp, ti_w, ti_b], [out], bo.BuiltinOptions.FullyConnectedOptions, o)
        return t

    # x^2
    mul("x", "x", "sq", (T, D))

    if variant == "A0":   # A 去掉 eps（诊断：ADD 报 type 不兼容）
        act("mean", (T, 1))
        S.ReducerOptionsStart(b); S.ReducerOptionsAddKeepDims(b, True); ro = S.ReducerOptionsEnd(b)
        ax = const(np.asarray([1], np.int32), I32, [1])
        op(OC_MEAN, ["sq", ax], ["mean"], bo.BuiltinOptions.ReducerOptions, ro)
        act("rsqrt", (T, 1))
        op(OC_RSQRT, ["mean"], ["rsqrt"])
    elif variant == "B0":  # 只到统计 tile：[T,32]
        tile_w = np.full((32, D), np.float32(1.0 / D), np.float32)
        fc("sq", "tile", tile_w, 32, D)
    elif variant == "A":
        act("mean", (T, 1))
        S.ReducerOptionsStart(b); S.ReducerOptionsAddKeepDims(b, True); ro = S.ReducerOptionsEnd(b)
        ax = const(np.asarray([1], np.int32), I32, [1])
        op(OC_MEAN, ["sq", ax], ["mean"], bo.BuiltinOptions.ReducerOptions, ro)
        act("mean_eps", (T, 1))
        eps = const(np.full((T, 1), EPS, np.float16))
        S.AddOptionsStart(b); S.AddOptionsAddFusedActivationFunction(b, 0); ao = S.AddOptionsEnd(b)
        op(OC_ADD, ["mean", eps], ["mean_eps"], bo.BuiltinOptions.AddOptions, ao)
        act("rsqrt", (T, 1))
        op(OC_RSQRT, ["mean_eps"], ["rsqrt"])
    else:
        tile_w = np.full((32, D), np.float32(1.0 / D), np.float32)
        fc("sq", "tile", tile_w, 32, D)
        if variant == "B":
            copies = D // 32            # 896 = 28 * 32，正好整除
            act("tile_exp", (T, D))
            S.ConcatenationOptionsStart(b)
            S.ConcatenationOptionsAddAxis(b, 1)
            S.ConcatenationOptionsAddFusedActivationFunction(b, 0)
            co = S.ConcatenationOptionsEnd(b)
            op(OC_CONCAT, ["tile"] * copies, ["tile_exp"], bo.BuiltinOptions.ConcatenationOptions, co)
            act("rsqrt", (T, D))
            op(OC_RSQRT, ["tile_exp"], ["rsqrt"])
        else:  # C: 扩张 FC（负对照）
            exp_w = np.full((D, 32), np.float32(1.0), np.float32)
            fc("tile", "tile_exp", exp_w, D, 32)
            act("rsqrt", (T, D))
            op(OC_RSQRT, ["tile_exp"], ["rsqrt"])

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
    act("out", (T, D))
    S.MulOptionsStart(b); S.MulOptionsAddFusedActivationFunction(b, 0); mo = S.MulOptionsEnd(b)
    op(OC_MUL, ["scaled", gam], ["out"], bo.BuiltinOptions.MulOptions, mo)

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
