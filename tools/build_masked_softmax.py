#!/usr/bin/env python3
"""Ticket 05 — causal mask + native SoftMax, Qwen2.5-0.5B attention shape (fp16).

    scores [14,8,8] --MUL(1/8 const)--> ADD(+ mask) --> SOFTMAX(beta=1) --> [14,8,8]

Why this exact shape (the only one the compiler leaves us):

  * the推理路径里没有 EXP / LOG 之类的一元逐元素算子可用（importer 阶段就被拒，见
    docs/OP-rope.md §"SIN/COS 被拒"）。所以"手写 exp 链"不存在，掩码只能
    (a) 以加法进入 softmax 之前，让**原生 SoftMax 层**自己完成 exp+归一化。
  * 原生 SOFTMAX 已被证实落到 MDLA_5_0（docs/OP-MATRIX §1，beta 生效），所以方案唯一形态
    就是 MUL -> ADD -> SOFTMAX，掩码只能是一个**加在分数上的有限负常数**。

Mask value (why -4096, and why it is added AFTER the 1/8 scale):

  scores come from the冻结夹具（ticket 02）真实统计量下的 layer0 attention：
      |q.k| max = 4110.7    scaled(|s|/8) max = 513.8
  掩码位置的输出要**确定性地**为 0，只需要
      (masked value) - (row max) <= T0
  T0 = ln(2^-25) = -17.33 if the device keeps exp() in fp16, and
  T0 = -104 if it keeps exp() in fp32 (exp underflows to 0 below -103.3) —— 因此取更严的 -104，
  再留 4x 余量：M >= 2*S_scaled + 4*104 = 2*513.8 + 416 = 1444。
  取 2^12 = 4096（幂次：fp16 精确可表示，没有表示误差），余量 29.5x，且 |v|max = M + S = 4610 << 65504。
  **不用 -inf**：fp16 的 -inf 进 ADD/normalize 会出 NaN（未定义行为）。
  **不用 -10000**：它比下界大 7x，白吃掉 2.4x 的 fp16 动态范围预算，且没有推导支撑。

  顺序（必须在 scale 之后）：若掩码加在 scale 之前，同一个 M 会被再除 8，要压过**未缩放**的
  |q.k| = 4110.7，所需 M 从 1444 抬到 2*4110.7 + 8*104 = 9054（6.3x），而且 fp16 中间量
  |s+M| 会到 2 万量级。post-scale 让掩码与"缩放后"的分数比大小，量级差 8 倍。

Independent golden (R3): tools/compare.py 读的 golden/expected.bin 完全由 numpy/float64 从
**落盘后的 fp16 输入字节**重算，绝不调用本脚本写 flatbuffer 的那段代码。

Usage:
    QWEN25_WEIGHTS_DIR='<fixture>' /usr/bin/python3 tools/build_masked_softmax.py \
        --out ops/masked_softmax_t8_fp16
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

import build_ops as bo

T, H, HKV, HD, D = 8, 14, 2, 64, 896     # T=8, 14 query heads, 2 KV heads, head_dim=64
G = H // HKV                              # 7 q-heads per kv-head
SCALE = 0.125                             # 1 / sqrt(head_dim) = 1/8, exact power of two
MASK_MAG = 4096.0                         # 2^12, derivation in the module docstring
MASK_MAG_ALT = 8192.0                     # 2^13, 只用于"掩码幅度不敏感"的对照实验
SEED = 20261113

CAUSAL = np.tril(np.ones((T, T), dtype=bool))      # row i may attend j <= i


# ------------------------------------------------------------------ input data
def fixture_scores(layer):
    """Raw q.k for one layer of the frozen fixture, [H,T,T] fp16 (before the 1/8 scale).

    x 取 RMSNorm 的输出：单位 RMS 的随机方向 * 该层 input_layernorm 的实测增益
    （layer0 gamma mean=0.034 std=0.094 -> attention 输入 RMS~0.10，与真实模型一致）。
    RoPE 不在本票内：它是正交旋转，不改变 |q|/|k|，因此不影响这里要定的**量级**。
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import qwen25_weights as qw

    Wq = qw.get_fp32(layer + ".attn.q_proj.weight")
    bq = qw.get_fp32(layer + ".attn.q_proj.bias")
    Wk = qw.get_fp32(layer + ".attn.k_proj.weight")
    bk = qw.get_fp32(layer + ".attn.k_proj.bias")
    gam = qw.get_fp32(layer + ".input_layernorm.weight")

    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((T, D))
    x = x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True)) * gam
    q = (x @ Wq.T + bq).reshape(T, H, HD).transpose(1, 0, 2)      # [H,T,HD]
    k = (x @ Wk.T + bk).reshape(T, HKV, HD).transpose(1, 0, 2)    # [HKV,T,HD]
    kk = np.repeat(k, G, axis=0)                                   # KV replication 2 -> 14
    raw = np.einsum("htd,hjd->htj", q, kk)                         # [H,T,T]
    return raw.astype(np.float16), dict(
        layer=layer, hidden_rms=float(np.sqrt((x ** 2).mean())),
        bq_std=float(bq.std()), bk_std=float(bk.std()),
        raw_absmax=float(np.abs(raw).max()),
        scaled_absmax=float(np.abs(raw).max()) * SCALE,
        per_head_offset_min=float(min(raw[h].mean() for h in range(H)) * SCALE),
        per_head_offset_max=float(max(raw[h].mean() for h in range(H)) * SCALE),
    )


def make_mask(mag):
    """Causal additive mask: 0 where j<=i (visible), -mag elsewhere."""
    m = np.zeros((H, T, T), np.float16)
    m[:, ~CAUSAL] = np.float16(-mag)
    return m


# --------------------------------------------------------------- golden (R3)
def softmax_ref(scores16, mask16):
    """Independent float64 reference: softmax(scores*SCALE + mask), last axis.

    Canonical form p = exp(v)/sum(exp(v)) —— **不做 max 减法**，这样参考实现不预设设备内部
    用哪种稳定化策略（|v|<=~500 在 float64 里 exp 既不上溢也不下溢）。随后与 max-减法版本
    交叉校验（见 crosscheck()）。
    """
    s = scores16.astype(np.float64) * SCALE          # *2^-3 在 fp64 里精确
    v = s + mask16.astype(np.float64)
    e = np.exp(v)
    return e / e.sum(axis=-1, keepdims=True)         # float64; 落盘时才舍入到 fp16


def crosscheck(scores16, mask16):
    s = scores16.astype(np.float64) * SCALE
    v = s + mask16.astype(np.float64)
    vm = v - v.max(axis=-1, keepdims=True)
    e = np.exp(vm)
    p = e / e.sum(axis=-1, keepdims=True)
    return float(np.abs(p - softmax_ref(scores16, mask16)).max())


# --------------------------------------------------------------------- builder
def build_model(const_mask=None, desc="masked softmax"):
    """MUL(scale) -> ADD(mask) -> SOFTMAX(beta=1).

    const_mask=None -> mask is the **second runtime input** (lets one .dla be run with
    different masks: real causal / double magnitude / all zeros).
    const_mask=arr -> mask baked as a constant (the production form for ticket 09).
    """
    b = bo.flatbuffers.Builder(1 << 16)
    S = bo.S
    F16 = bo.F16

    buffers = [bo.buf_empty(b)]
    tensors, opcodes, operators, idx = [], [], [], {}

    def const(arr, name):
        a = np.ascontiguousarray(np.asarray(arr, np.float16))
        buffers.append(bo.buf_data(b, a))
        tensors.append(bo.tensor(b, name, list(a.shape), F16, len(buffers) - 1))
        return len(tensors) - 1

    def act(name, shape):
        tensors.append(bo.tensor(b, name, list(shape), F16, 0))
        idx[name] = len(tensors) - 1
        return idx[name]

    def oc(kind, version=1):
        opcodes.append(bo.opcode(b, kind, version))
        return len(opcodes) - 1

    OC_MUL = oc(bo.BuiltinOperator.MUL)
    OC_ADD = oc(bo.BuiltinOperator.ADD)
    OC_SM = oc(bo.BuiltinOperator.SOFTMAX)

    act("scores", (H, T, T))
    if const_mask is None:
        act("mask", (H, T, T))

    t_scale = const(np.full((H, T, T), SCALE), "scale")
    act("scaled", (H, T, T))
    S.MulOptionsStart(b)
    S.MulOptionsAddFusedActivationFunction(b, 0)
    operators.append(bo.operator(b, OC_MUL, [idx["scores"], t_scale], [idx["scaled"]],
                                 bo.BuiltinOptions.MulOptions, S.MulOptionsEnd(b)))

    if const_mask is None:
        t_mask = idx["mask"]
    else:
        t_mask = const(const_mask, "mask_const")
    act("masked", (H, T, T))
    S.AddOptionsStart(b)
    S.AddOptionsAddFusedActivationFunction(b, 0)
    operators.append(bo.operator(b, OC_ADD, [idx["scaled"], t_mask], [idx["masked"]],
                                 bo.BuiltinOptions.AddOptions, S.AddOptionsEnd(b)))

    act("probs", (H, T, T))
    S.SoftmaxOptionsStart(b)
    S.SoftmaxOptionsAddBeta(b, 1.0)
    operators.append(bo.operator(b, OC_SM, [idx["masked"]], [idx["probs"]],
                                 bo.BuiltinOptions.SoftmaxOptions, S.SoftmaxOptionsEnd(b)))

    inputs = [idx["scores"]] if const_mask is not None else [idx["scores"], idx["mask"]]
    return bo.finish(b, buffers, opcodes, tensors, operators, inputs, [idx["probs"]], desc)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ops/masked_softmax_t8_fp16")
    args = ap.parse_args()
    out = args.out
    gdir = os.path.join(out, "golden")
    os.makedirs(gdir, exist_ok=True)

    # ---- inputs (fixture-derived, deterministic) -------------------------
    s0, st0 = fixture_scores("layer0")
    s1, st1 = fixture_scores("layer1")
    mask = make_mask(MASK_MAG)
    mask_alt = make_mask(MASK_MAG_ALT)
    mask_zero = np.zeros((H, T, T), np.float16)

    # ---- models ---------------------------------------------------------
    blob = build_model(None, "masked softmax T=8 H=14 fp16 (mask = 2nd input)")
    blob_const = build_model(mask, "masked softmax T=8 H=14 fp16 (mask = const)")
    open(os.path.join(out, "model.tflite"), "wb").write(blob)
    open(os.path.join(out, "model_constmask.tflite"), "wb").write(blob_const)

    # ---- golden files ---------------------------------------------------
    files = {
        "input.bin": s0,                 # layer0 raw q.k  (primary)
        "input1.bin": mask,              # causal mask -4096 (primary)
        "expected.bin": softmax_ref(s0, mask),
        "input1_mag2.bin": mask_alt,     # same support, 2x magnitude
        "expected_mag2.bin": softmax_ref(s0, mask_alt),
        "input1_zero.bin": mask_zero,    # no mask at all
        "expected_zero.bin": softmax_ref(s0, mask_zero),
        "input_l1.bin": s1,              # layer1 scores (smaller per-head offset)
        "input1_l1.bin": mask,
        "expected_l1.bin": softmax_ref(s1, mask),
    }
    for name, arr in files.items():
        np.ascontiguousarray(arr).astype(np.float16).tofile(os.path.join(gdir, name))
    files = {n: a.astype(np.float16) for n, a in files.items()}

    exp = files["expected.bin"]
    exp_l1 = files["expected_l1.bin"]
    assert exp[~CAUSAL[None, :, :].repeat(H, 0)].max() == 0, "masked positions must be golden 0"
    assert exp_l1[~CAUSAL[None, :, :].repeat(H, 0)].max() == 0

    meta = {
        "op": "MUL+ADD+SOFTMAX (masked softmax)",
        "shape": {"H": H, "T": T, "KVH": HKV, "HD": HD, "D": D},
        "op_shape": [H, T, T],
        "dtype": "float16",
        "graph": "scores -MUL(1/8)-> -ADD(mask)-> -SOFTMAX(beta=1)-> probs",
        "scale": SCALE,
        "scale_exact": "1/8 = 2^-3, power of two -> the MUL is exact in fp16 (no rounding)",
        "mask_value": -MASK_MAG,
        "mask_value_alt": -MASK_MAG_ALT,
        "mask_order": "added AFTER the 1/8 scale (see docs/OP-masked-softmax.md §2)",
        "input": {"name": "scores", "shape": [H, T, T], "type": "float16",
                  "bytes": int(s0.nbytes), "means": "raw q.k (before 1/sqrt(head_dim))"},
        "input1": {"name": "mask", "shape": [H, T, T], "type": "float16",
                   "bytes": int(mask.nbytes), "means": "additive causal mask, 0 / -4096"},
        "output": {"name": "probs", "shape": [H, T, T], "type": "float16",
                   "bytes": int(exp.nbytes)},
        "reference": "float64 canonical softmax exp(v)/sum(exp(v)), v = fp16_input*1/8 + mask",
        "seed": SEED,
        "input_source": "frozen fixture (ticket 02) q/k projections of the named layer; "
                        "x = unit-RMS random direction * measured input_layernorm gamma",
        "score_stats": {"layer0": st0, "layer1": st1},
        "atol": 1e-3,
        "rtol": 5e-3,
        "tolerance_rationale":
            "registered BEFORE the board run. fp16 output magnitude ~1/(i+1) in [0.125,1]; "
            "1 ULP at 0.125 is 1.22e-4, and a device/CPU 1-ULP difference on 1/7 is 8.5e-4 "
            "relative. atol=1e-3 = 8 ULP at 0.125 (must also cover the masked positions whose "
            "expected value is exactly 0); rtol=5e-3 = 5.9x the relative 1-ULP of 0.125. "
            "Both are far below the repo default 2e-2 and above the measured noise floor (see "
            "docs/OP-masked-softmax.md §6).",
        "prediction": [
            "P1 compile: EXIT=0, single Execution Step on MDLA_5_0, layers = "
            "ElementWiseMulLayer + ElementWiseAddLayer + SoftMaxLayer, no bridge.",
            "P2 masked positions: device output is EXACTLY 0.0 (all fp16 bits zero).",
            "P3 mask magnitude -4096 vs -8192: the two board dumps are BIT-IDENTICAL "
            "(finite mask saturated => behaves like the ideal -inf mask).",
            "P4 zero mask => plain softmax over all 8 entries; masked-run/zero-run ratio at "
            "visible positions is the per-row constant S_all/S_visible.",
            "P5 noise floor (same .dla, same inputs, 2 runs) == 0 (deterministic hardware).",
            "P6 layer0 scores have |s|max=513.8 with per-head offsets +-500; if the device "
            "does NOT subtract the row max, exp(468) overflows fp16 -> NaN (this run answers it).",
        ],
        "experiments": [
            {"name": "primary", "input": "golden/input.bin", "input1": "golden/input1.bin",
             "expected": "golden/expected.bin", "compare": "compare.txt",
             "tests": "causal masked softmax, layer0 (largest measured dynamic range)"},
            {"name": "noise", "input": "golden/input.bin", "input1": "golden/input1.bin",
             "expected": "golden/expected.bin", "compare": "compare_noise.txt",
             "tests": "noise floor: same .dla + same inputs, second run"},
            {"name": "mag2", "input": "golden/input.bin", "input1": "golden/input1_mag2.bin",
             "expected": "golden/expected_mag2.bin", "compare": "compare_mag2.txt",
             "tests": "mask magnitude -4096 vs -8192 -> dump must be bit-identical to primary"},
            {"name": "zero", "input": "golden/input.bin", "input1": "golden/input1_zero.bin",
             "expected": "golden/expected_zero.bin", "compare": "compare_zero.txt",
             "tests": "mask == 0 -> plain softmax (未掩码版本), control for judgement 2"},
            {"name": "layer1", "input": "golden/input_l1.bin", "input1": "golden/input1_l1.bin",
             "expected": "golden/expected_l1.bin", "compare": "compare_l1.txt",
             "tests": "second regime: per-head offsets only +-45, non-degenerate probabilities"},
        ],
        "tflite": "model.tflite",
        "tflite_bytes": len(blob),
        "tflite_constmask": "model_constmask.tflite",
        "tflite_constmask_bytes": len(blob_const),
        "sha256": {name: sha256_file(os.path.join(out, "model" + suf))
                   for name, suf in ((".tflite", ".tflite"),
                                     ("_constmask.tflite", "_constmask.tflite"))},
    }
    meta["golden_sha256"] = {n: sha256_file(os.path.join(gdir, n)) for n in files}
    with open(os.path.join(out, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("%s: model.tflite %d B, model_constmask.tflite %d B" % (out, len(blob), len(blob_const)))
    print("golden: %d files" % len(files))
    for nm, st in (("layer0", st0), ("layer1", st1)):
        print("  %s: raw |q.k|max=%.1f  scaled |s|max=%.1f  per-head offset [%+.1f,%+.1f]  "
              "x_rms=%.4f bq_std=%.2f bk_std=%.2f" %
              (nm, st["raw_absmax"], st["scaled_absmax"], st["per_head_offset_min"],
               st["per_head_offset_max"], st["hidden_rms"], st["bq_std"], st["bk_std"]))
    print("  reference cross-check (canonical vs max-subtracted, float64): "
          "%.3e / %.3e" % (crosscheck(s0, mask), crosscheck(s0, mask_zero)))
    vis = np.broadcast_to(CAUSAL[None], (H, T, T))
    v = s0.astype(np.float64) * SCALE + mask.astype(np.float64)
    rowmax = np.where(vis, v, -np.inf).max(axis=-1, keepdims=True)
    gap = float((v[~vis] - np.broadcast_to(rowmax, (H, T, T))[~vis]).max())
    print("  masked value: min=%.1f rev-limits |v|max=%.1f (fp16 max 65504) ; value@visible "
          "min=%.1f max=%.1f" % (v.min(), np.abs(v).max(), v[vis].min(), v[vis].max()))
    print("  worst-case (masked - rowmax_visible) = %+.1f  ->  slack vs fp32-exp-underflow "
          "threshold (-104): %.1fx ; vs fp16 (-17.4): %.0fx" % (gap, -gap / 104.0, -gap / 17.4))
    print("  golden: masked out == 0 exactly on %d/%d masked positions" %
          (int((files["expected.bin"][~vis] == 0).sum()), int((~vis).sum())))
    print("  golden visible range: [%.6f, %.6f]" % (files["expected.bin"][vis].min(),
                                                   files["expected.bin"][vis].max()))
    sall = np.exp(s0.astype(np.float64) * SCALE).sum(axis=-1)
    svis = np.where(vis, np.exp(s0.astype(np.float64) * SCALE), 0.0).sum(axis=-1)
    print("  predicted primary/zero per-row ratio S_all/S_vis: [%.6f, %.6f]" %
          ((sall / svis).min(), (sall / svis).max()))


if __name__ == "__main__":
    main()