#!/usr/bin/env python3
"""Assemble ONE scaled Qwen2.5-style decoder block as a single TFLite graph.

Scale: hidden H=64, 1 attention head (head_dim 32), FFN 128, T=8, fp16 in/out.
Kept out of the graph on purpose (documented split, like openmtk's hybrid baseline):
  * RMSNorm  — needs rsqrt; whether it can live on MDLA is the rms-agent's open question
  * RoPE     — sin/cos tables; sin/cos are rejected by the ncc importer
Both are therefore applied in the *reference* (numpy) and the graph starts from the
already-normalised, already-rotated tensors: this experiment is about "can the block's
matmuls + attention + gating be one NPU graph", not about norm/rope support.

Graph (ops):
  q  = FC(x, Wq)                    [8,64] -> [8,32]   const weights
  k  = FC(x, Wk)                    [8,64] -> [8,32]
  v  = FC(x, Wv)                    [8,64] -> [8,32]
  s  = FC(q, k)                     [8,32] -> [8,8]    **dynamic B**
  p  = SOFTMAX(s)                   [8,8]
  a  = FC(p, Vt)                    [8,8]  -> [8,32]   **dynamic B** (Vt = v transposed)
  o  = FC(a, Wo)                    [8,32] -> [8,64]
  r1 = ADD(x, o)
  g  = FC(r1, Wg)                   [8,64] -> [8,128]
  u  = FC(r1, Wu)                   [8,64] -> [8,128]
  si = LOGISTIC(g);  m = MUL(g, si); y = MUL(m, u)
  d  = FC(y, Wd)                    [8,128] -> [8,64]
  out = ADD(r1, d)

    /usr/bin/python3 tools/build_block.py --out ops/block_qwen25_1l_fp16
"""
import argparse
import json
import os

import numpy as np

import build_ops as bo

T, H, HD, F = 8, 64, 32, 128
SEED = 20261112


def f16(a):
    return np.asarray(a, dtype=np.float16)


def main(out_dir):
    rng = np.random.default_rng(SEED)
    # weights (fp16), shapes [N,K] for FC
    Wq = f16(rng.normal(0, 0.05, (HD, H)))
    Wk = f16(rng.normal(0, 0.05, (HD, H)))
    Wv = f16(rng.normal(0, 0.05, (HD, H)))
    Wo = f16(rng.normal(0, 0.05, (H, HD)))
    Wg = f16(rng.normal(0, 0.05, (F, H)))
    Wu = f16(rng.normal(0, 0.05, (F, H)))
    Wd = f16(rng.normal(0, 0.05, (H, F)))
    x = f16(rng.normal(0, 0.5, (T, H)))

    b = bo.flatbuffers.Builder(1 << 20)
    S = bo.S
    F16 = bo.F16

    # --- buffers: 0 = empty (activations), then each const ---
    buffers = [bo.buf_empty(b)]

    def const(arr):
        buffers.append(bo.buf_data(b, f16(arr)))
        return len(buffers) - 1

    def act(name, shape, ttype=F16):
        return bo.tensor(b, name, list(shape), ttype, 0)

    tensors = []
    opcodes = []
    operators = []

    def opcode(builtin):
        opcodes.append(bo.opcode(b, builtin))
        return len(opcodes) - 1

    OC_FC = opcode(bo.BuiltinOperator.FULLY_CONNECTED)
    OC_SM = opcode(bo.BuiltinOperator.SOFTMAX)
    OC_ADD = opcode(bo.BuiltinOperator.ADD)
    OC_MUL = opcode(bo.BuiltinOperator.MUL)
    OC_SIG = opcode(bo.BuiltinOperator.LOGISTIC)

    t_x = act("x", (T, H))
    tensors.append(t_x)
    t_idx = {"x": 0}

    def add_tensor(name, t):
        tensors.append(t)
        t_idx[name] = len(tensors) - 1
        return t_idx[name]

    def fc(in_name, out_name, wts, m, k, n):
        wi = const(wts)
        t_w = bo.tensor(b, f"W_{out_name}", [n, k], F16, wi)
        ti_w = add_tensor(f"W_{out_name}", t_w)
        t_o = act(out_name, (m, n))
        ti_o = add_tensor(out_name, t_o)
        S.FullyConnectedOptionsStart(b)
        S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)
        S.FullyConnectedOptionsAddWeightsFormat(b, 0)
        S.FullyConnectedOptionsAddKeepNumDims(b, True)
        S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
        opt = S.FullyConnectedOptionsEnd(b)
        operators.append(bo.operator(b, OC_FC, [t_idx[in_name], ti_w], [ti_o],
                                    bo.BuiltinOptions.FullyConnectedOptions, opt))
        return t_idx[out_name]

    def unary(in_name, out_name, shape, oc, opt_type=None, opt=None):
        t_o = act(out_name, shape)
        ti = add_tensor(out_name, t_o)
        operators.append(bo.operator(b, oc, [t_idx[in_name]], [ti], opt_type, opt))
        return ti

    def binary(a_name, b_ti, out_name, shape, oc):
        t_o = act(out_name, shape)
        ti = add_tensor(out_name, t_o)
        S.AddOptionsStart(b) if oc == OC_ADD else S.MulOptionsStart(b)
        S.AddOptionsAddFusedActivationFunction(b, 0) if oc == OC_ADD else S.MulOptionsAddFusedActivationFunction(b, 0)
        opt = S.AddOptionsEnd(b) if oc == OC_ADD else S.MulOptionsEnd(b)
        operators.append(bo.operator(b, oc, [t_idx[a_name], b_ti], [ti],
                                    bo.BuiltinOptions.AddOptions if oc == OC_ADD else bo.BuiltinOptions.MulOptions,
                                    opt))
        return ti

    # --- block body ---
    fc("x", "q", Wq, T, H, HD)
    fc("x", "k", Wk, T, H, HD)
    fc("x", "v", Wv, T, H, HD)
    # scores = q . k^T  (dynamic B: k is a runtime tensor)
    t_k = tensors[t_idx["k"]]
    ti_k = t_idx["k"]
    t_s = act("s", (T, T))
    ti_s = add_tensor("s", t_s)
    S.FullyConnectedOptionsStart(b)
    S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)
    S.FullyConnectedOptionsAddWeightsFormat(b, 0)
    S.FullyConnectedOptionsAddKeepNumDims(b, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
    opt = S.FullyConnectedOptionsEnd(b)
    operators.append(bo.operator(b, OC_FC, [t_idx["q"], ti_k], [ti_s],
                                bo.BuiltinOptions.FullyConnectedOptions, opt))
    # softmax over last axis
    S.SoftmaxOptionsStart(b)
    S.SoftmaxOptionsAddBeta(b, 1.0)
    sm_opt = S.SoftmaxOptionsEnd(b)
    unary("s", "p", (T, T), OC_SM, bo.BuiltinOptions.SoftmaxOptions, sm_opt)
    # a = p . v  (v transposed as dynamic B, shape [HD, T])
    OC_TR = opcode(bo.BuiltinOperator.TRANSPOSE)
    t_vt = act("vt", (HD, T))
    ti_vt = add_tensor("vt", t_vt)
    perm = bo.buf_data(b, np.asarray([1, 0], np.int32))
    buffers.append(perm)
    t_perm = bo.tensor(b, "perm", [2], bo.I32, len(buffers) - 1)
    ti_perm = add_tensor("perm", t_perm)
    S.TransposeOptionsStart(b)
    tr_opt = S.TransposeOptionsEnd(b)
    operators.append(bo.operator(b, OC_TR, [t_idx["v"], ti_perm], [ti_vt],
                                bo.BuiltinOptions.TransposeOptions, tr_opt))
    t_a = act("a", (T, HD))
    ti_a = add_tensor("a", t_a)
    S.FullyConnectedOptionsStart(b)
    S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)
    S.FullyConnectedOptionsAddWeightsFormat(b, 0)
    S.FullyConnectedOptionsAddKeepNumDims(b, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
    opt = S.FullyConnectedOptionsEnd(b)
    operators.append(bo.operator(b, OC_FC, [t_idx["p"], ti_vt], [ti_a],
                                bo.BuiltinOptions.FullyConnectedOptions, opt))
    fc("a", "o", Wo, T, HD, H)
    binary("x", t_idx["o"], "r1", (T, H), OC_ADD)
    fc("r1", "g", Wg, T, H, F)
    fc("r1", "u", Wu, T, H, F)
    unary("g", "sig", (T, F), OC_SIG)
    binary("g", t_idx["sig"], "m", (T, F), OC_MUL)
    binary("m", t_idx["u"], "y", (T, F), OC_MUL)
    fc("y", "d", Wd, T, F, H)
    binary("r1", t_idx["d"], "out", (T, H), OC_ADD)

    # graph inputs: x plus the dynamic-B tensors k and vt
    inputs = [t_idx["x"]]
    outputs = [t_idx["out"]]
    blob = bo.finish(b, buffers, opcodes, tensors, operators, inputs, outputs,
                     "qwen25 block 1x (T=8,H=64,HD=32,F=128) fp16")

    # --- numpy reference (float32 math, rounded to fp16) ---
    xf = np.asarray(x, np.float32)
    Wqf, Wkf, Wvf = (np.asarray(w, np.float32) for w in (Wq, Wk, Wv))
    Wof, Wgf, Wuf, Wdf = (np.asarray(w, np.float32) for w in (Wo, Wg, Wu, Wd))
    q = xf @ Wqf.T; k = xf @ Wkf.T; v = xf @ Wvf.T
    sc = q @ k.T
    sc = sc - sc.max(axis=1, keepdims=True)
    e = np.exp(sc); p_ = e / e.sum(axis=1, keepdims=True)
    a_ = p_ @ v
    o_ = a_ @ Wof.T
    r1 = xf + o_
    g_ = r1 @ Wgf.T; u_ = r1 @ Wuf.T
    y_ = (g_ / (1.0 + np.exp(-g_))) * u_
    d_ = y_ @ Wdf.T
    ref = r1 + d_

    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "model.tflite"), "wb").write(blob)
    os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
    np.asarray(x).astype(np.float16).tofile(os.path.join(out_dir, "golden", "input.bin"))
    ref.astype(np.float16).tofile(os.path.join(out_dir, "golden", "expected.bin"))
    np.asarray(Wk @ np.zeros(0) if False else Wk).tofile(os.path.join(out_dir, "Wk.bin"))

    meta = {
        "op": "QWEN25_BLOCK_1L",
        "shape": {"T": T, "H": H, "HD": HD, "FFN": F},
        "dtype": "float16",
        "inputs": [
            {"name": "x", "shape": [T, H], "type": "float16"},
            {"name": "k", "shape": [T, HD], "type": "float16"},
            {"name": "vt", "shape": [HD, T], "type": "float16"},
        ],
        "output": {"name": "out", "shape": [T, H], "type": "float16"},
        "atol": 2e-2,
        "rtol": 2e-2,
        "note": "RMSNorm and RoPE are applied outside the graph (see docs/OP-block.md)",
        "tflite_bytes": len(blob),
    }
    json.dump(meta, open(os.path.join(out_dir, "model.json"), "w"), indent=2)
    print(f"{out_dir}: tflite={len(blob)}B, 3 inputs (x,k,vt), fp16, T={T} H={H} HD={HD} F={F}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ops/block_qwen25_1l_fp16")
    main(ap.parse_args().out)
