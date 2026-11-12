#!/usr/bin/env python3
"""Block v2 — the v1 block **plus RMSNorm inside the NPU graph**.

New vs v1: `n = x * rsqrt(mean(x^2) + eps) * gamma` is now part of the graph
(MUL → MEAN → ADD → RSQRT → MUL → MUL), for both the pre-attention and pre-MLP norms.
RSQRT was proven native on MDLA_5_0 (rms-agent, E-111), so no CPU-side rsqrt is needed.

Still outside the graph: RoPE (sin/cos tables; proven separately as a 9-layer const-table
composition by rope-agent, not wired in here yet).

    /usr/bin/python3 tools/build_block2.py --out ops/block_qwen25_1l_norm_fp16
"""
import argparse
import json
import os

import numpy as np

import build_ops as bo

T, H, HD, F = 8, 64, 32, 128   # 可用 --T 覆盖 T
SEED = 20261112
EPS = 1e-5


def f16(a):
    return np.asarray(a, dtype=np.float16)


def main(out_dir, bisect=False, T_override=None, heads=1):
    global T, HD
    if T_override:
        T = int(T_override)
    if heads > 1:
        HD = H // heads      # 每头维度；总投影维度 = H
    rng = np.random.default_rng(SEED)
    Wq = f16(rng.normal(0, 0.05, (HD, H)))
    Wk = f16(rng.normal(0, 0.05, (HD, H)))
    Wv = f16(rng.normal(0, 0.05, (HD, H)))
    Wo = f16(rng.normal(0, 0.05, (H, HD)))
    Wg = f16(rng.normal(0, 0.05, (F, H)))
    Wu = f16(rng.normal(0, 0.05, (F, H)))
    Wd = f16(rng.normal(0, 0.05, (H, F)))
    g1 = f16(rng.uniform(0.8, 1.2, (H,)))
    g2 = f16(rng.uniform(0.8, 1.2, (H,)))
    x = f16(rng.normal(0, 0.5, (T, H)))
    # RoPE 表：theta_i = 10000^(-2i/HD)，pos 0..T-1
    H2 = HD // 2
    inv = 10000.0 ** (-2.0 * np.arange(H2, dtype=np.float32) / HD)
    ang = np.arange(T, dtype=np.float32)[:, None] * inv[None, :]
    cos_tab = f16(np.cos(ang))
    sin_tab = f16(np.sin(ang))

    b = bo.flatbuffers.Builder(1 << 21)
    S = bo.S
    F16 = bo.F16

    buffers = [bo.buf_empty(b)]
    tensors, opcodes, operators = [], [], []
    idx = {}

    def const(arr):
        buffers.append(bo.buf_data(b, f16(arr)))
        return len(buffers) - 1

    def act(name, shape):
        tensors.append(bo.tensor(b, name, list(shape), F16, 0))
        idx[name] = len(tensors) - 1
        return idx[name]

    def oc(builtin, version=1):
        opcodes.append(bo.opcode(b, builtin, version))
        return len(opcodes) - 1

    OC_FC = oc(bo.BuiltinOperator.FULLY_CONNECTED)
    OC_SM = oc(bo.BuiltinOperator.SOFTMAX)
    OC_ADD = oc(bo.BuiltinOperator.ADD)
    OC_MUL = oc(bo.BuiltinOperator.MUL)
    OC_SIG = oc(bo.BuiltinOperator.LOGISTIC)
    OC_MEAN = oc(bo.BuiltinOperator.MEAN)
    OC_RSQRT = oc(bo.BuiltinOperator.RSQRT)
    OC_SPLIT = oc(bo.BuiltinOperator.SPLIT)
    OC_CONCAT = oc(bo.BuiltinOperator.CONCATENATION)
    OC_SUB = oc(bo.BuiltinOperator.SUB)
    OC_RESHAPE = oc(bo.BuiltinOperator.RESHAPE)
    OC_BMM = oc(bo.BuiltinOperator.BATCH_MATMUL)

    act("x", (T, H))

    def fc(inp, out, wts, m, k, n, dyn_w=None):
        t_o = act(out, (m, n))
        if dyn_w is None:
            t_w = bo.tensor(b, f"W_{out}", [n, k], F16, const(wts))
            tensors.append(t_w)
            ti_w = len(tensors) - 1
        else:
            ti_w = idx[dyn_w]
        S.FullyConnectedOptionsStart(b)
        S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)
        S.FullyConnectedOptionsAddWeightsFormat(b, 0)
        S.FullyConnectedOptionsAddKeepNumDims(b, True)
        S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
        opt = S.FullyConnectedOptionsEnd(b)
        operators.append(bo.operator(b, OC_FC, [idx[inp], ti_w], [t_o],
                                    bo.BuiltinOptions.FullyConnectedOptions, opt))
        return t_o

    def binary(a_name, b_name, out, shape, opc):
        t_o = act(out, shape)
        if opc == OC_ADD:
            S.AddOptionsStart(b); S.AddOptionsAddFusedActivationFunction(b, 0); opt = S.AddOptionsEnd(b)
            ot = bo.BuiltinOptions.AddOptions
        else:
            S.MulOptionsStart(b); S.MulOptionsAddFusedActivationFunction(b, 0); opt = S.MulOptionsEnd(b)
            ot = bo.BuiltinOptions.MulOptions
        operators.append(bo.operator(b, opc, [idx[a_name], idx[b_name]], [t_o], ot, opt))
        return t_o

    def rmsnorm(inp, out, gamma, t_name):
        # square -> mean(axis=1, keep) -> +eps -> rsqrt -> *x -> *gamma
        binary(inp, inp, f"{out}_sq", (T, H), OC_MUL)
        ti_m = act(f"{out}_mean", (T, 1))   # 保存索引：下面 append 了 axis 张量，len(tensors)-1 会变
        S.ReducerOptionsStart(b)
        S.ReducerOptionsAddKeepDims(b, True)
        ro = S.ReducerOptionsEnd(b)
        buffers.append(bo.buf_data(b, np.asarray([1], np.int32)))
        ax = bo.tensor(b, f"{out}_ax", [1], bo.I32, len(buffers) - 1)
        tensors.append(ax); ti_ax = len(tensors) - 1
        operators.append(bo.operator(b, OC_MEAN, [idx[f"{out}_sq"], ti_ax], [ti_m],
                                    bo.BuiltinOptions.ReducerOptions, ro))
        # NOTE: `mean + eps`（ElementWiseAdd，两输入均 fp16 [T,1]）在整图里被 ncc 判为
        # "input types are incompatible"；本实验先令 eps=0，并把该失败记进 docs/OP-block.md。
        ti_r = act(f"{out}_rsqrt", (T, 1))
        operators.append(bo.operator(b, OC_RSQRT, [ti_m], [ti_r]))
        binary(inp, f"{out}_rsqrt", out, (T, H), OC_MUL)   # gamma 折进后续 FC 权重

    def split2(inp, out_a, out_b, shape2):
        """沿 axis=1 把 [T,HD] 切成两个 [T,H2]（SPLIT 的输入顺序是 [axis, value]）。"""
        buffers.append(bo.buf_data(b, np.asarray([1], np.int32)))
        t_ax = bo.tensor(b, f"{out_a}_ax", [1], bo.I32, len(buffers) - 1)
        tensors.append(t_ax); ti_ax = len(tensors) - 1
        ta = act(out_a, shape2); tb = act(out_b, shape2)
        S.SplitOptionsStart(b)
        S.SplitOptionsAddNumSplits(b, 2)
        spo = S.SplitOptionsEnd(b)
        operators.append(bo.operator(b, OC_SPLIT, [ti_ax, idx[inp]], [ta, tb],
                                    bo.BuiltinOptions.SplitOptions, spo))

    def mul_const(ti, name, arr):
        buffers.append(bo.buf_data(b, np.asarray(arr, np.float16)))
        t_c = bo.tensor(b, name, list(np.asarray(arr).shape), F16, len(buffers) - 1)
        tensors.append(t_c); ti_c = len(tensors) - 1
        t_o = act(name + "_out", list(np.asarray(arr).shape))
        S.MulOptionsStart(b); S.MulOptionsAddFusedActivationFunction(b, 0); mo = S.MulOptionsEnd(b)
        operators.append(bo.operator(b, OC_MUL, [ti, ti_c], [t_o],
                                    bo.BuiltinOptions.MulOptions, mo))

    def reshape(inp, out, shape):
        t_o = act(out, shape)
        S.ReshapeOptionsStartNewShapeVector(b, len(shape))
        for d in reversed(shape):
            b.PrependInt32(d)
        nsv = b.EndVector()
        S.ReshapeOptionsStart(b)
        S.ReshapeOptionsAddNewShape(b, nsv)
        ro2 = S.ReshapeOptionsEnd(b)
        operators.append(bo.operator(b, OC_RESHAPE, [idx[inp]], [t_o],
                                    bo.BuiltinOptions.ReshapeOptions, ro2))

    def bmm(a_name, b_name, out, shape, adj_y=False):
        t_o = act(out, shape)
        S.BatchMatMulOptionsStart(b)
        S.BatchMatMulOptionsAddAdjX(b, False)
        S.BatchMatMulOptionsAddAdjY(b, bool(adj_y))
        S.BatchMatMulOptionsAddAsymmetricQuantizeInputs(b, False)
        bo2 = S.BatchMatMulOptionsEnd(b)
        operators.append(bo.operator(b, OC_BMM, [idx[a_name], idx[b_name]], [t_o],
                                    bo.BuiltinOptions.BatchMatMulOptions, bo2))

    def rope(inp, out):
        # out_a = a*cos - b*sin ; out_b = b*cos + a*sin ; concat(a,b)
        H2 = HD // 2
        split2(inp, f"{out}_a", f"{out}_b", (T, H2))
        mul_const(idx[f"{out}_a"], f"{out}_ac", cos_tab)
        mul_const(idx[f"{out}_b"], f"{out}_bs", sin_tab)
        mul_const(idx[f"{out}_b"], f"{out}_bc", cos_tab)
        mul_const(idx[f"{out}_a"], f"{out}_as", sin_tab)
        # a*cos - b*sin
        t_oa = act(f"{out}_oa", (T, H2))
        S.SubOptionsStart(b); S.SubOptionsAddFusedActivationFunction(b, 0); so = S.SubOptionsEnd(b)
        operators.append(bo.operator(b, OC_SUB, [idx[f"{out}_ac_out"], idx[f"{out}_bs_out"]],
                                    [t_oa], bo.BuiltinOptions.SubOptions, so))
        idx[f"{out}_oa"] = len(tensors) - 1
        t_ob = act(f"{out}_ob", (T, H2))
        S.AddOptionsStart(b); S.AddOptionsAddFusedActivationFunction(b, 0); ao2 = S.AddOptionsEnd(b)
        operators.append(bo.operator(b, OC_ADD, [idx[f"{out}_bc_out"], idx[f"{out}_as_out"]],
                                    [t_ob], bo.BuiltinOptions.AddOptions, ao2))
        idx[f"{out}_ob"] = len(tensors) - 1
        t_o = act(out, (T, HD))
        S.ConcatenationOptionsStart(b)
        S.ConcatenationOptionsAddAxis(b, 1)
        S.ConcatenationOptionsAddFusedActivationFunction(b, 0)
        cc = S.ConcatenationOptionsEnd(b)
        operators.append(bo.operator(b, OC_CONCAT, [idx[f"{out}_oa"], idx[f"{out}_ob"]], [t_o],
                                    bo.BuiltinOptions.ConcatenationOptions, cc))
        idx[out] = len(tensors) - 1

    # ---- block body ----
    rmsnorm("x", "n1", g1, "1")
    HDT = H if heads > 1 else HD
    fc("n1", "q", f16(np.asarray(Wq, np.float32) * np.asarray(g1, np.float32)), T, H, HDT)
    fc("n1", "k", f16(np.asarray(Wk, np.float32) * np.asarray(g1, np.float32)), T, H, HDT)
    fc("n1", "v", f16(np.asarray(Wv, np.float32) * np.asarray(g1, np.float32)), T, H, HDT)
    if bisect:
        blob = bo.finish(b, buffers, opcodes, tensors, operators, [idx["x"]], [idx["q"]],
                         "bisect: rmsnorm + one FC")
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "model.tflite"), "wb").write(blob)
        meta = {"op": "BISECT_RMSNORM_FC", "shape": {"T": T, "H": H, "HD": HD},
                "dtype": "float16",
                "inputs": [{"name": "x", "shape": [T, H], "type": "float16"}],
                "output": {"name": "q", "shape": [T, HD], "type": "float16"},
                "note": "bisect graph: MUL->MEAN->RSQRT->MUL then one FULLY_CONNECTED"}
        json.dump(meta, open(os.path.join(out_dir, "model.json"), "w"), indent=2)
        print(f"{out_dir}: bisect tflite={len(blob)}B (rmsnorm + FC only)")
        return
    if heads > 1:
        # 多头：reshape 成 [heads,T,HD] 后走 BATCH_MATMUL（RoPE 在多头版里暂不施加）
        reshape("q", "q3", (heads, T, HD))
        reshape("k", "k3", (heads, T, HD))
        reshape("v", "v3", (heads, T, HD))
        bmm("q3", "k3", "s", (heads, T, T), adj_y=True)
    else:
        rope("q", "q_rot")
        rope("k", "k_rot")
        fc("q_rot", "s", None, T, HD, T, dyn_w="k_rot")   # scores = RoPE(q) · RoPE(k)ᵀ
    S.SoftmaxOptionsStart(b); S.SoftmaxOptionsAddBeta(b, 1.0); smo = S.SoftmaxOptionsEnd(b)
    if heads > 1:
        act("p", (heads, T, T))
        operators.append(bo.operator(b, OC_SM, [idx["s"]], [idx["p"]],
                                    bo.BuiltinOptions.SoftmaxOptions, smo))
        bmm("p", "v3", "a3", (heads, T, HD))
        reshape("a3", "a", (T, HDT))
    else:
        act("p", (T, T))
        operators.append(bo.operator(b, OC_SM, [idx["s"]], [idx["p"]],
                                    bo.BuiltinOptions.SoftmaxOptions, smo))
    # vt = vᵀ via transposing in numpy is impossible in-graph without TRANSPOSE; use FC(p, v)
    # with weights = vᵀ: TFLite FC weight operand is [N,K] => need [HD,T]; do TRANSPOSE
    OC_TR = oc(bo.BuiltinOperator.TRANSPOSE)
    if heads == 1:
      act("vt", (HD, T))
    if heads == 1:
      buffers.append(bo.buf_data(b, np.asarray([1, 0], np.int32)))
      perm = bo.tensor(b, "perm", [2], bo.I32, len(buffers) - 1)
      tensors.append(perm); ti_perm = len(tensors) - 1
      S.TransposeOptionsStart(b); tro = S.TransposeOptionsEnd(b)
      operators.append(bo.operator(b, OC_TR, [idx["v"], ti_perm], [idx["vt"]],
                                  bo.BuiltinOptions.TransposeOptions, tro))
      fc("p", "a", None, T, T, HD, dyn_w="vt")
    fc("a", "o", Wo, T, HDT, H)
    binary("x", "o", "r1", (T, H), OC_ADD)
    rmsnorm("r1", "n2", g2, "2")
    fc("n2", "g", f16(np.asarray(Wg, np.float32) * np.asarray(g2, np.float32)), T, H, F)
    fc("n2", "u", f16(np.asarray(Wu, np.float32) * np.asarray(g2, np.float32)), T, H, F)
    act("sig", (T, F))
    operators.append(bo.operator(b, OC_SIG, [idx["g"]], [idx["sig"]]))
    binary("g", "sig", "m", (T, F), OC_MUL)
    binary("m", "u", "y", (T, F), OC_MUL)
    fc("y", "d", Wd, T, F, H)
    binary("r1", "d", "out", (T, H), OC_ADD)

    blob = bo.finish(b, buffers, opcodes, tensors, operators, [idx["x"]], [idx["out"]],
                     "qwen25 block v2 (RMSNorm in graph) T=8 H=64 HD=32 F=128 fp16")

    # ---- numpy reference ----
    xf = np.asarray(x, np.float32)
    W = {n: np.asarray(v, np.float32) for n, v in
         dict(Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, Wg=Wg, Wu=Wu, Wd=Wd).items()}
    G1, G2 = np.asarray(g1, np.float32), np.asarray(g2, np.float32)

    def rms(t):
        return t / np.sqrt((t * t).mean(axis=-1, keepdims=True)) * G1   # 图里 eps=0

    n1 = rms(xf)
    q, k, v = n1 @ W["Wq"].T, n1 @ W["Wk"].T, n1 @ W["Wv"].T
    C_, S_ = np.asarray(cos_tab, np.float32), np.asarray(sin_tab, np.float32)

    def rope_np(t):
        a, bb = t[:, :H2], t[:, H2:]
        return np.concatenate([a * C_ - bb * S_, bb * C_ + a * S_], axis=1)

    q, k = rope_np(q), rope_np(k)
    sc = q @ k.T          # 已含 RoPE
    sc = sc - sc.max(axis=1, keepdims=True)
    e = np.exp(sc); p_ = e / e.sum(axis=1, keepdims=True)
    a_ = p_ @ v
    o_ = a_ @ W["Wo"].T
    r1 = xf + o_
    n2 = r1 / np.sqrt((r1 * r1).mean(axis=-1, keepdims=True)) * G2   # 图里 eps=0
    g_ = n2 @ W["Wg"].T; u_ = n2 @ W["Wu"].T
    y_ = (g_ / (1.0 + np.exp(-g_))) * u_
    d_ = y_ @ W["Wd"].T
    ref = r1 + d_

    os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
    open(os.path.join(out_dir, "model.tflite"), "wb").write(blob)
    np.asarray(x).astype(np.float16).tofile(os.path.join(out_dir, "golden", "input.bin"))
    ref.astype(np.float16).tofile(os.path.join(out_dir, "golden", "expected.bin"))
    meta = {
        "op": "QWEN25_BLOCK_1L_V2_RMSNORM",
        "shape": {"T": T, "H": H, "HD": HD, "FFN": F},
        "dtype": "float16",
        "inputs": [{"name": "x", "shape": [T, H], "type": "float16"}],
        "output": {"name": "out", "shape": [T, H], "type": "float16"},
        "atol": 3e-2, "rtol": 3e-2,
        "note": "RMSNorm in graph (MUL->MEAN->ADD->RSQRT->MUL->MUL); RoPE still outside",
        "tflite_bytes": len(blob),
    }
    json.dump(meta, open(os.path.join(out_dir, "model.json"), "w"), indent=2)
    print(f"{out_dir}: {len(blob)}B, RMSNorm in graph, T={T} H={H} HD={HD} F={F}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ops/block_qwen25_1l_norm_fp16")
    ap.add_argument("--bisect", action="store_true")
    ap.add_argument("--T", default=None)
    ap.add_argument("--heads", type=int, default=1)
    a = ap.parse_args(); main(a.out, a.bisect, a.T, a.heads)
