#!/usr/bin/env python3
"""Build single-operator TFLite probe models for the CONV / POOL family
(CONV_2D, DEPTHWISE_CONV_2D, MAX_POOL_2D, AVERAGE_POOL_2D) plus model.json and
golden/ (input.bin + expected.bin) for device-side 对拍.

Independent numpy golden (AGENTS.md R3): this file does NOT share any packing /
indexing / quantization code with ncc-tflite or the neuron runtime. The conv is
expressed as an explicit int64 accumulate over (kh, kw) taps with an
independently computed SAME/VALID padding rule, so a layout mistake on either
side shows up as a mismatch instead of cancelling out.

Layout follows TFLite:
    CONV_2D            input [N,H,W,C]   filter [OC,FH,FW,C]   bias [OC]  -> [N,OH,OW,OC]
    DEPTHWISE_CONV_2D  input [N,H,W,C]   filter [1,FH,FW,OC]   bias [OC]  -> [N,OH,OW,OC]
    MAX/AVG_POOL_2D    input [N,H,W,C]                                     -> [N,OH,OW,C]

Quantization schemes (see docs/OP-convpool.md, "three tiers"):
  "small"  : input/weight codes in {-1,0,1}, bias 0, output scale == in*weight
             scale  =>  out == clip(exact INT64 accumulator).  int8_tol = 0.
  "pow2"   : full int8 codes, output scale chosen so the requant multiplier is an
             exact power of two, output range picked from the actual accumulator
             range so nothing saturates.  int8_tol = 0.
  "div3"   : small-range codes, output scale = 3*in*weight  =>  multiplier 1/3,
             a non power-of-two requant that exercises the rounding rule.
             int8_tol = 1 (rounding boundary is a documented contract).

Usage:
    /usr/bin/python3 tools/build_convpool.py --list
    /usr/bin/python3 tools/build_convpool.py conv_1x8x8x4_f3_int8 \
        --out ops/conv_1x8x8x4_f3_int8/model.tflite
"""
import argparse
import json
import os
from fractions import Fraction

import flatbuffers
import numpy as np
import tflite as S
from tflite import (
    ActivationFunctionType, BuiltinOperator, BuiltinOptions, Padding, Model,
    Operator, OperatorCode, Tensor, TensorType,
)

F32 = TensorType.FLOAT32
F16 = TensorType.FLOAT16
I8 = TensorType.INT8
I32 = TensorType.INT32

SEED = 20261112
PAD = {"SAME": Padding.SAME, "VALID": Padding.VALID}
ACT = {
    "NONE": ActivationFunctionType.NONE,
    "RELU": ActivationFunctionType.RELU,
    "RELU6": ActivationFunctionType.RELU6,
}


# --------------------------------------------------------------- flatbuffers
def buf_empty(b):
    S.BufferStart(b)
    return S.BufferEnd(b)


def buf_data(b, arr):
    v = b.CreateByteVector(np.ascontiguousarray(arr).tobytes())
    S.BufferStart(b)
    S.BufferAddData(b, v)
    return S.BufferEnd(b)


def quant_tensor(b, scales, zps, axis=0):
    sv = b.CreateNumpyVector(np.asarray(scales, np.float32))
    zv = b.CreateNumpyVector(np.asarray(zps, np.int64))
    S.QuantizationParametersStart(b)
    S.QuantizationParametersAddScale(b, sv)
    S.QuantizationParametersAddZeroPoint(b, zv)
    S.QuantizationParametersAddQuantizedDimension(b, axis)
    return S.QuantizationParametersEnd(b)


def tensor(b, name, shape, ttype, buffer_idx, q=None):
    n = b.CreateString(name)
    shp = b.CreateNumpyVector(np.asarray(shape, np.int32))
    S.TensorStart(b)
    S.TensorAddName(b, n)
    if q is not None:
        S.TensorAddQuantization(b, q)
    S.TensorAddShape(b, shp)
    S.TensorAddShapeSignature(b, shp)
    S.TensorAddType(b, ttype)
    S.TensorAddBuffer(b, buffer_idx)
    S.TensorAddHasRank(b, True)
    return S.TensorEnd(b)


def opcode(b, builtin, version=1):
    S.OperatorCodeStart(b)
    S.OperatorCodeAddDeprecatedBuiltinCode(b, builtin if builtin < 127 else 127)
    S.OperatorCodeAddBuiltinCode(b, builtin)
    S.OperatorCodeAddVersion(b, version)
    return S.OperatorCodeEnd(b)


def operator(b, oc, ins, outs, opt_type=None, opt=None):
    iv = b.CreateNumpyVector(np.asarray(ins, np.int32))
    ov = b.CreateNumpyVector(np.asarray(outs, np.int32))
    S.OperatorStart(b)
    S.OperatorAddOpcodeIndex(b, oc)
    S.OperatorAddInputs(b, iv)
    S.OperatorAddOutputs(b, ov)
    if opt is not None:
        S.OperatorAddBuiltinOptionsType(b, opt_type)
        S.OperatorAddBuiltinOptions(b, opt)
    return S.OperatorEnd(b)


def finish(b, buffers, opcodes, tensors, operators, inputs, outputs, desc):
    def vec(objs):
        v = b.StartVector(4, len(objs), 4)
        for o in reversed(objs):
            b.PrependUOffsetTRelative(o)
        return b.EndVector()

    tv, ov, cv, bv = vec(tensors), vec(operators), vec(opcodes), vec(buffers)
    iv = b.CreateNumpyVector(np.asarray(inputs, np.int32))
    og = b.CreateNumpyVector(np.asarray(outputs, np.int32))
    nm = b.CreateString("main")
    S.SubGraphStart(b)
    S.SubGraphAddTensors(b, tv)
    S.SubGraphAddInputs(b, iv)
    S.SubGraphAddOutputs(b, og)
    S.SubGraphAddOperators(b, ov)
    S.SubGraphAddName(b, nm)
    sg = S.SubGraphEnd(b)
    sgv = b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(sg)
    sgv = b.EndVector()
    d = b.CreateString(desc)
    S.ModelStart(b)
    S.ModelAddVersion(b, 3)
    S.ModelAddOperatorCodes(b, cv)
    S.ModelAddSubgraphs(b, sgv)
    S.ModelAddDescription(b, d)
    S.ModelAddBuffers(b, bv)
    m = S.ModelEnd(b)
    b.Finish(m, file_identifier=b"TFL3")
    return bytes(b.Output())


# ------------------------------------------------------------ geometry rules
def out_size(in_size, f, s, dil, padding):
    """Output extent + (before, after) padding, computed independently of ncc.

    SAME  : out = ceil(in/s); pad split with the extra pixel at the end (TFLite).
    VALID : out = floor((in - eff)/s) + 1, no padding.
    """
    eff = (f - 1) * dil + 1
    if padding == "SAME":
        out = -(-in_size // s)                 # ceil division
        total = max((out - 1) * s + eff - in_size, 0)
        return out, total // 2, total - total // 2
    out = (in_size - eff) // s + 1
    return out, 0, 0


def requant_div(acc, num, den):
    """(acc * num/den) with round-half-away-from-zero, pure int64 arithmetic.

    Integer-only on purpose: a float multiply here would make the golden's
    rounding depend on float64 representation of the scale ratio.
    This is the *mathematically correct* requant. It is NOT what the MDLA does
    for a general multiplier -- see requant_dev, and docs/OP-convpool.md for the
    device dumps that falsified it (2/512 and 167/512 mismatches).
    """
    acc = np.asarray(acc, np.int64)
    pos = np.where(acc >= 0, (acc * num + den // 2) // den,
                   -((-acc * num + den // 2) // den))
    return pos.astype(np.int64)


def quantize_multiplier(num, den):
    """TFLite/gemmlowp fixed-point form of the scale ratio num/den (< 1).

    ratio = q * 2^-shift with q in [0.5, 1)  ->  M0 = round(q * 2^31)
    (this is TFLite's QuantizeMultiplier; frexp gives the same q/exponent).
    """
    m = Fraction(int(num), int(den))
    shift = 0
    while m < Fraction(1, 2):
        m *= 2
        shift += 1
    m0 = int(np.floor(m * (1 << 31) + Fraction(1, 2)))
    return m0, shift


def rhm(x, m0):
    """gemmlowp SaturatingRoundingDoublingHighMul on the (unreached) int32 range:
    round(x*m0 / 2^31) with the +2^30 nudge, i.e. round-half-UP, then >>31."""
    return (x * m0 + (1 << 30)) >> 31


def rdpot(x, shift):
    """gemmlowp RoundingDivideByPOT: round-half-away-from-zero of x / 2^shift."""
    if shift == 0:
        return x
    mask = (1 << shift) - 1
    rem = x & mask
    thr = (mask >> 1) + np.where(x < 0, 1, 0)
    return (x >> shift) + np.where(rem > thr, 1, 0)


def avg_divice(sumv, cnt):
    """AVERAGE_POOL_2D division rule, inferred from device dumps.

    floor((2*sum + count) / (2*count)) == "round to nearest, ties toward +inf".
    Evidence (tools/diagnose_avgpool_table.py, ops/avgpool_1x8x8x16_*): every
    fractional part > 1/2 rounds away from zero, < 1/2 truncates, exactly 1/2
    rounds UP for positive sums (-38.5 -> -38 for negative ones).  A single
    rounding on the exact ratio, NOT the conv two-step fixed-point rule -- the
    counts 4 / 6 / 9 mix in one tensor rules the two-step model out (it misses
    non-tie values by ±1).
    """
    sumv = np.asarray(sumv, np.int64)
    cnt = np.asarray(cnt, np.int64)
    return (2 * sumv + cnt) // (2 * cnt)


def requant_dev(acc, num, den):
    """Device-accurate requant: MultiplyByQuantizedMultiplier(acc, M0, shift).

    Two rounding steps (this is where ±1 vs the exact rational comes from):
      step 1: round-half-up(acc * M0 / 2^31)     -- integer granularity, loses
              the sub-LSB information of the product
      step 2: round-half-away(x / 2^shift)
    For M0 = 2^31, shift = 0 (multiplier exactly 1) this is the identity, which
    is why the {-1,0,1}-codes tier is bit-exact on both this and the rational
    reference.
    """
    m0, shift = quantize_multiplier(num, den)
    acc = np.asarray(acc, np.int64)
    return rdpot(rhm(acc, m0), shift).astype(np.int64)


def clip_i8(x):
    return np.clip(x, -128, 127).astype(np.int8)


def apply_act(x, act):
    if act == "RELU":
        return np.maximum(x, 0)
    if act == "RELU6":
        return np.minimum(np.maximum(x, 0), 6)
    return x


# ------------------------------------------------------------------ references
def conv_ref(x, w, bias, stride, dil, padding, depthwise=False):
    """int64 accumulator reference.

    x  : [H,W,C] activations (int8 codes)
    w  : [OC,FH,FW,C] for CONV_2D, [1,FH,FW,C*mult] for DEPTHWISE_CONV_2D
         (TFLite keeps the depthwise filter transposed: OHWI with O=1)
    The (kh,kw) tap loop is written out explicitly and the padded rows/cols are
    gathered by index, so a SAME/VALID padding or stride mistake cannot hide.
    """
    H, W, C = x.shape
    FH, FW = w.shape[1], w.shape[2]
    OC = w.shape[3] if depthwise else w.shape[0]
    mult = OC // C if depthwise else 1
    OH, pt, pb = out_size(H, FH, stride, dil, padding)
    OW, pl, pr = out_size(W, FW, stride, dil, padding)
    xp = np.zeros((H + pt + pb, W + pl + pr, C), np.int64)
    xp[pt:pt + H, pl:pl + W] = x.astype(np.int64)
    rows = [kh * dil + np.arange(OH) * stride for kh in range(FH)]
    cols = [kw * dil + np.arange(OW) * stride for kw in range(FW)]
    acc = np.zeros((OH, OW, OC), np.int64)
    for kh in range(FH):
        for kw in range(FW):
            patch = xp[np.ix_(rows[kh], cols[kw])]          # [OH,OW,C]
            if depthwise:
                # out channel oc reads input channel oc // mult
                xr = np.repeat(patch, mult, axis=2)          # [OH,OW,OC]
                acc += xr * w[0, kh, kw][None, None, :]
            else:
                acc += (patch.reshape(-1, C) @ w[:, kh, kw, :].T).reshape(OH, OW, OC)
    if bias is not None:
        acc += bias.astype(np.int64)[None, None, :]
    return acc, (OH, OW)


def pool_windows(x, f, s, padding, mode):
    """Sum/max over pooling windows, plus the per-element valid-element count.

    max : SAME padding contributes -128 (TFLite uses the dtype minimum).
    avg : padding contributes nothing; the divisor is the *valid* element count
          (TFLite excludes padding from the average).  The rounding of sum/count
          is applied by the caller so that it can follow the device's fixed-point
          rule rather than a made-up one.
    """
    H, W, C = x.shape
    OH, pt, pb = out_size(H, f, s, 1, padding)
    OW, pl, pr = out_size(W, f, s, 1, padding)
    fill = -128 if mode == "max" else 0
    xp = np.full((H + pt + pb, W + pl + pr, C), fill, np.int64)
    xp[pt:pt + H, pl:pl + W] = x.astype(np.int64)
    valid = np.zeros(xp.shape[:2] + (C,), np.int64)
    valid[pt:pt + H, pl:pl + W] = 1
    rows = [kh + np.arange(OH) * s for kh in range(f)]
    cols = [kw + np.arange(OW) * s for kw in range(f)]
    stack = np.stack([xp[np.ix_(rows[kh], cols[kw])] for kh in range(f)
                      for kw in range(f)])                       # [f*f,OH,OW,C]
    cnt = np.stack([valid[np.ix_(rows[kh], cols[kw])] for kh in range(f)
                    for kw in range(f)]).sum(axis=0)             # [OH,OW,C]
    if mode == "max":
        return stack.max(axis=0), None
    return stack.sum(axis=0), cnt


# ---------------------------------------------------------------------- build
def build(kind, cfg):
    rng = np.random.default_rng(SEED)
    H, W, C = cfg["h"], cfg["w"], cfg["c"]
    fh, fw = cfg["fh"], cfg["fw"]
    s, dil, pad, act = cfg["s"], cfg.get("dil", 1), cfg["pad"], cfg.get("act", "NONE")
    codes = cfg["codes"]
    A = 1.0 / 128.0                     # input scale, power of two
    Wq = 1.0 / 128.0                    # weight scale, power of two

    if kind == "conv":
        OC = cfg["oc"]
        filt_shape = [OC, fh, fw, C]
    elif kind == "dwconv":
        OC = C * cfg.get("mult", 1)
        filt_shape = [1, fh, fw, OC]
    else:
        OC = C
        filt_shape = None

    bias_on = cfg.get("bias", True) and filt_shape is not None

    # ---------------- activations / weights / bias -------------------------
    if filt_shape is None:
        x_q = rng.integers(-128, 128, size=(H, W, C)).astype(np.int8)
        w_q = b_q = None
        acc = None
    elif codes == "small":
        x_q = rng.integers(-1, 2, size=(H, W, C)).astype(np.int8)
        w_q = rng.integers(-1, 2, size=filt_shape).astype(np.int8)
        b_q = np.zeros(OC, np.int32)
        acc, _ = conv_ref(x_q, w_q, b_q, s, dil, pad, kind == "dwconv")
    elif codes == "pow2":
        # full int8 codes, but the requant multiplier is 1/2^k so the requant is
        # a bit-exact shift; output range is derived from the accumulator range
        # so the int8 output never saturates (saturation would hide errors).
        # cfg["wmax"] / cfg["shift"] let a spec pin the weight code range and
        # the shift when the *output range* itself is what is being probed
        # (e.g. RELU6 needs 6.0 to be representable in int8).
        x_q = rng.integers(-127, 128, size=(H, W, C)).astype(np.int8)
        wmax = int(cfg.get("wmax", 127))
        w_q = rng.integers(-wmax, wmax + 1, size=filt_shape).astype(np.int8)
        b_q = (rng.integers(-1000, 1001, size=OC)).astype(np.int32) if bias_on else np.zeros(OC, np.int32)
        if bias_on and cfg.get("bias_off"):
            # constant offset (in bias codes = 1/(input_scale*weight_scale) units)
            b_q = (b_q.astype(np.int64) + int(cfg["bias_off"])).astype(np.int32)
        acc, _ = conv_ref(x_q, w_q, b_q, s, dil, pad, kind == "dwconv")
        if cfg.get("shift") is not None:
            shift = int(cfg["shift"])
        else:
            shift = max(int(np.ceil(np.log2(max(abs(int(acc.max())), abs(int(acc.min())), 1) / 127.0))), 0)
        cfg["_shift"] = shift
        cfg["_num"], cfg["_den"] = 1, 1 << shift
        cfg["_oscale"] = A * Wq * (1 << shift)
    elif codes in ("div3", "div"):
        # non power-of-two requant multiplier 1/den -> exercises the rounding rule.
        # codes are kept small so the int8 output never saturates.
        den = 3 if codes == "div3" else int(cfg.get("div", 3))
        x_q = rng.integers(-3, 4, size=(H, W, C)).astype(np.int8)
        w_q = rng.integers(-3, 4, size=filt_shape).astype(np.int8)
        b_q = np.zeros(OC, np.int32)
        acc, _ = conv_ref(x_q, w_q, b_q, s, dil, pad, kind == "dwconv")
        cfg["_num"], cfg["_den"] = 1, den
        cfg["_oscale"] = A * Wq * den
    else:
        raise SystemExit("bad codes")

    # ---------------- output tensor + golden -------------------------------
    if filt_shape is None:
        mode = "max" if kind == "maxpool" else "avg"
        tot, cnt = pool_windows(x_q, fh, s, pad, mode)
        if mode == "max":
            out = tot
            req = {"rule": "none", "divisor": None}
            tol = 0
            ref_name = "int64_exact_max"
        else:
            # divide by the valid-element count, per element (counts differ at
            # the borders when SAME padding is in play)
            out = avg_divice(tot, cnt)
            req = {"rule": "round_half_up", "divisor": "valid_count",
                   "counts": sorted(int(c) for c in np.unique(cnt))}
            tol = int(cfg.get("avg_tol", 0))
            ref_name = "int64_avg_round_half_up_over_valid_count"
        out_i8 = clip_i8(np.asarray(out, np.int64))
        in_scale = out_scale = 1.0 / 128.0
        meta_quant = {"input": in_scale, "weights": None, "bias": None,
                      "output": out_scale}
    else:
        if codes == "small":
            num, den, oscale = 1, 1, A * Wq
        else:
            num, den, oscale = cfg["_num"], cfg["_den"], cfg["_oscale"]
        rule = cfg.get("requant", "dev")
        if rule == "dev":
            q = requant_dev(acc, num, den)
            m0, sh = quantize_multiplier(num, den)
            req = {"rule": "dev", "num": int(num), "den": int(den),
                   "m0": int(m0), "shift": int(sh)}
        else:
            q = requant_div(acc, num, den)
            req = {"rule": "rational", "num": int(num), "den": int(den)}
        # activation is applied in real units; for int8 with zp=0 that is the
        # same as clamping the quantized value at act/oscale.
        if act == "RELU":
            q = np.maximum(q, 0)
            tol = 0
        elif act == "RELU6":
            hi = int(np.floor(6.0 / oscale))
            q = np.minimum(np.maximum(q, 0), hi)
            req["relu6_code"] = hi
            tol = 0
        else:
            # "rational" references are the deliberately-naive goldens kept as
            # failure evidence: they miss the device's two-step requant by ±1.
            tol = 0 if (codes == "small" or rule == "dev") else 1
        out_i8 = clip_i8(q)
        in_scale, out_scale = A, oscale
        meta_quant = {"input": in_scale, "weights": Wq if codes != "small" else A,
                      "bias": None if not bias_on else in_scale * Wq, "output": out_scale}
        ref_name = ("int64_exact" if codes == "small"
                    else "int64_" + rule + "rule_" + ("pow2" if codes == "pow2" else f"div{den}"))

    OH, OW = out_i8.shape[0], out_i8.shape[1]

    # ------------------------------ flatbuffer ----------------------------
    b = flatbuffers.Builder(65536)
    buffers = [buf_empty(b)]
    tensors = []
    if kind == "conv":
        bi = BuiltinOperator.CONV_2D
        otype = BuiltinOptions.Conv2DOptions
    elif kind == "dwconv":
        bi = BuiltinOperator.DEPTHWISE_CONV_2D
        otype = BuiltinOptions.DepthwiseConv2DOptions
    elif kind == "maxpool":
        bi = BuiltinOperator.MAX_POOL_2D
        otype = BuiltinOptions.Pool2DOptions
    else:
        bi = BuiltinOperator.AVERAGE_POOL_2D
        otype = BuiltinOptions.Pool2DOptions
    opcodes = [opcode(b, bi)]

    # tensor order (== operator input indices) must be: input, [filter], [bias], output
    t_in = tensor(b, "input", [1, H, W, C], I8, 0,
                  quant_tensor(b, [in_scale], [0]))
    tensors.append(t_in)
    ins = [0]
    if filt_shape is not None:
        buffers.append(buf_data(b, w_q))                 # buffer 1: filter
        t_w = tensor(b, "filter", filt_shape, I8, 1,
                     quant_tensor(b, [Wq], [0]))
        tensors.append(t_w)                              # index 1
        ins.append(1)
        if bias_on:
            buffers.append(buf_data(b, b_q))             # buffer 2: bias
            t_b = tensor(b, "bias", [OC], I32, 2,
                         quant_tensor(b, [in_scale * Wq], [0]))
            tensors.append(t_b)                          # index 2
            ins.append(2)
    buffers.append(buf_empty(b))
    out_idx = len(buffers) - 1
    out_tensor_idx = len(tensors)
    t_out = tensor(b, "output", [1, OH, OW, OC], I8, out_idx,
                   quant_tensor(b, [out_scale], [0]))

    if kind == "conv":
        S.Conv2DOptionsStart(b)
        S.Conv2DOptionsAddPadding(b, PAD[pad])
        S.Conv2DOptionsAddStrideW(b, s)
        S.Conv2DOptionsAddStrideH(b, s)
        S.Conv2DOptionsAddFusedActivationFunction(b, ACT[act])
        S.Conv2DOptionsAddDilationWFactor(b, dil)
        S.Conv2DOptionsAddDilationHFactor(b, dil)
        opt = S.Conv2DOptionsEnd(b)
    elif kind == "dwconv":
        S.DepthwiseConv2DOptionsStart(b)
        S.DepthwiseConv2DOptionsAddPadding(b, PAD[pad])
        S.DepthwiseConv2DOptionsAddStrideW(b, s)
        S.DepthwiseConv2DOptionsAddStrideH(b, s)
        S.DepthwiseConv2DOptionsAddDepthMultiplier(b, cfg.get("mult", 1))
        S.DepthwiseConv2DOptionsAddFusedActivationFunction(b, ACT[act])
        S.DepthwiseConv2DOptionsAddDilationWFactor(b, dil)
        S.DepthwiseConv2DOptionsAddDilationHFactor(b, dil)
        opt = S.DepthwiseConv2DOptionsEnd(b)
    else:
        S.Pool2DOptionsStart(b)
        S.Pool2DOptionsAddPadding(b, PAD[pad])
        S.Pool2DOptionsAddStrideW(b, s)
        S.Pool2DOptionsAddStrideH(b, s)
        S.Pool2DOptionsAddFilterWidth(b, fw)
        S.Pool2DOptionsAddFilterHeight(b, fh)
        S.Pool2DOptionsAddFusedActivationFunction(b, ACT[act])
        opt = S.Pool2DOptionsEnd(b)

    tensors.append(t_out)
    op = operator(b, 0, ins, [out_tensor_idx], otype, opt)
    blob = finish(b, buffers, opcodes, tensors, [op], [0], [out_tensor_idx],
                  f"{kind} {H}x{W}x{C} f{fh}x{fw} s{s} {pad} {codes}")

    op_name = {"conv": "CONV_2D", "dwconv": "DEPTHWISE_CONV_2D",
               "maxpool": "MAX_POOL_2D", "avgpool": "AVERAGE_POOL_2D"}[kind]
    meta = {
        "op": op_name,
        "shape": {"N": 1, "H": H, "W": W, "C": C, "OH": OH, "OW": OW, "OC": OC,
                  "FH": fh, "FW": fw, "stride": s, "dilation": dil,
                  "mult": int(cfg.get("mult", 1)),
                  "padding": pad, "activation": act},
        "dtype": "int8",
        "input": {"name": "input", "shape": [1, H, W, C], "type": "int8",
                  "bytes": int(x_q.nbytes)},
        "output": {"name": "output", "shape": [1, OH, OW, OC], "type": "int8",
                   "bytes": int(out_i8.nbytes)},
        "quant": meta_quant,
        "filter_shape": filt_shape,
        "bias": bool(bias_on),
        "reference": ref_name,
        "requant": req,
        "codes": codes,
        "int8_tol": int(tol),
        "seed": SEED,
    }
    consts = {"filter": w_q, "bias": b_q if bias_on else None}
    return blob, meta, x_q, out_i8, consts


# ---------------------------------------------------------------------- main
def C(**kw):
    d = dict(kind="conv", s=1, pad="SAME", dil=1, act="NONE", codes="pow2")
    d.update(kw)
    return d


SPECS = {
    # =====================================================================
    # Generation 1 -- "rational" golden (round-half-away of acc*scale_ratio).
    # Kept on purpose: the device dumps of these ops are the evidence that the
    # naive requant is off by ±1 (2/512, 1/512, 167/512).  Do not "fix" them.
    # =====================================================================
    # exact tier: codes {-1,0,1}, multiplier 1 -> pure layout/integer check
    "conv_1x8x8x4_f3_int8":     C(h=8, w=8, c=4, oc=8, fh=3, fw=3, codes="small"),
    # full-range tier: real int8 codes, power-of-two requant multiplier
    "conv_1x8x8x4_f3_full_int8": C(h=8, w=8, c=4, oc=8, fh=3, fw=3, requant="rational"),
    # 1x1 conv = matmul-shaped conv (fallback path for matmul)
    "conv_1x8x8x4_f1x1_int8":   C(h=8, w=8, c=4, oc=8, fh=1, fw=1),
    # fused activation: does ncc keep RELU6 inside the conv layer?
    #   With oscale=0.125 (shift 11) the int8 codomain covers 15.875 real units,
    #   so 6.0 sits at code 48 and is *distinguishable* from int8 saturation.
    #   wmax=127 + a +40000 bias offset push part of the pre-activations past
    #   code 48 => the golden contains clipped 48s (9 of 512), which is the
    #   decisive evidence that RELU6 was applied, not just "no negatives".
    "conv_1x8x8x4_f3_relu6_int8": C(h=8, w=8, c=4, oc=8, fh=3, fw=3, act="RELU6",
                                    wmax=127, shift=11, bias_off=40000,
                                    requant="rational"),
    # non-power-of-two requant (1/3): closest look at the rounding rule
    "conv_1x8x8x4_f3_div3_int8": C(h=8, w=8, c=4, oc=8, fh=3, fw=3, codes="div3",
                                   requant="rational"),
    # stride 2 + VALID at C=8 and dilation at C=4: compile-boundary evidence
    # (need an eDPA relayout, rejected by --disallow-bridge)
    "conv_1x16x16x8_f3s2v_int8": C(h=16, w=16, c=8, oc=8, fh=3, fw=3, s=2, pad="VALID"),
    "conv_1x8x8x4_f3d2_int8":   C(h=8, w=8, c=4, oc=8, fh=3, fw=3, dil=2),
    # =====================================================================
    # Generation 2 -- golden uses the *derived* device requant model
    # (QuantizeMultiplier + SaturatingRoundingDoublingHighMul + RoundingDivideByPOT,
    #  see requant_dev).  Prediction recorded before the run: bit-exact.
    # =====================================================================
    "conv_1x8x8x4_f3_full_devrule_int8":  C(h=8, w=8, c=4, oc=8, fh=3, fw=3),
    "conv_1x8x8x4_f3_relu6_devrule_int8": C(h=8, w=8, c=4, oc=8, fh=3, fw=3,
                                            act="RELU6", wmax=127, shift=11,
                                            bias_off=40000),
    # in-sample: 1/3 was one of the multipliers used to derive the rule
    "conv_1x8x8x4_f3_div3_devrule_int8":  C(h=8, w=8, c=4, oc=8, fh=3, fw=3,
                                            codes="div", div=3),
    # OUT-OF-SAMPLE: 1/5 and 1/6 were never used to derive the rule.
    # Prediction (written before the run): 0/512 mismatches for both.
    "conv_1x8x8x4_f3_div5_devrule_int8":  C(h=8, w=8, c=4, oc=8, fh=3, fw=3,
                                            codes="div", div=5),
    "conv_1x8x8x4_f3_div6_devrule_int8":  C(h=8, w=8, c=4, oc=8, fh=3, fw=3,
                                            codes="div", div=6),
    # --- DEPTHWISE_CONV_2D ----------------------------------------------
    "dwconv_1x8x8x4_f3_int8":   C(kind="dwconv", h=8, w=8, c=4, fh=3, fw=3),
    "dwconv_1x8x8x4_f3m2_int8": C(kind="dwconv", h=8, w=8, c=4, fh=3, fw=3, mult=2),
    # --- POOL ------------------------------------------------------------
    # C=4 / C=8 pool probes are kept as *compile-boundary evidence*: ncc maps them
    # to MaxPool2DLayer on MDLA but then needs an eDPA relayout step, which
    # --disallow-bridge rejects (see reports/convpool_scan.tsv).  C=16 and up map
    # to a single native MDLA step, so the numeric probes use C=16.
    "maxpool_1x8x8x4_2x2s2_int8":      C(kind="maxpool", h=8, w=8, c=4, fh=2, fw=2, s=2, pad="VALID"),
    "maxpool_1x8x8x4_2x2s2_same_int8": C(kind="maxpool", h=8, w=8, c=4, fh=2, fw=2, s=2, pad="SAME"),
    "maxpool_1x8x8x4_3x3s1_same_int8": C(kind="maxpool", h=8, w=8, c=4, fh=3, fw=3, s=1, pad="SAME"),
    "avgpool_1x8x8x4_2x2s2_int8":      C(kind="avgpool", h=8, w=8, c=4, fh=2, fw=2, s=2, pad="VALID", avg_tol=1),
    "avgpool_1x8x8x4_2x2s2_same_int8": C(kind="avgpool", h=8, w=8, c=4, fh=2, fw=2, s=2, pad="SAME", avg_tol=1),
    "avgpool_1x8x8x4_3x3s1_same_int8": C(kind="avgpool", h=8, w=8, c=4, fh=3, fw=3, s=1, pad="SAME", avg_tol=1),
    # C=16 pool probes (compile natively; these carry the numeric result).
    # 2x2 s2 SAME on 8x8 has no padding at all (out 4x4 tiles 8x8 exactly), so
    # the SAME-padding semantics are probed by the 3x3 s1 SAME variants:
    #   max: padding = -128 must never win the max  -> identical to VALID rule
    #   avg: if the divisor were padding-included (9/6) instead of valid-count
    #        (6/4), the borders would be off by ~1.5x, not by ±1.
    "maxpool_1x8x8x16_2x2s2_int8":      C(kind="maxpool", h=8, w=8, c=16, fh=2, fw=2, s=2, pad="VALID"),
    "maxpool_1x8x8x16_3x3s1_same_int8": C(kind="maxpool", h=8, w=8, c=16, fh=3, fw=3, s=1, pad="SAME"),
    "avgpool_1x8x8x16_2x2s2_int8":      C(kind="avgpool", h=8, w=8, c=16, fh=2, fw=2, s=2, pad="VALID"),
    "avgpool_1x8x8x16_2x2s2_same_int8": C(kind="avgpool", h=8, w=8, c=16, fh=2, fw=2, s=2, pad="SAME"),
    "avgpool_1x8x8x16_3x3s1_same_int8": C(kind="avgpool", h=8, w=8, c=16, fh=3, fw=3, s=1, pad="SAME"),
    # OUT-OF-SAMPLE for the avg-pool division rule: 2x2 s1 SAME gives window
    # counts {4, 2, 1} (edges and corners), i.e. counts that were NOT part of the
    # data the rule was inferred from (4/6/9).  Prediction before the run:
    # bit-exact, tol 0.
    "avgpool_1x8x8x16_2x2s1_same_int8": C(kind="avgpool", h=8, w=8, c=16, fh=2, fw=2, s=1, pad="SAME"),
    # C=16 conv probes for stride / dilation (C=4..8 need an eDPA relayout)
    "conv_1x16x16x16_f3s2v_int8": C(h=16, w=16, c=16, oc=16, fh=3, fw=3, s=2, pad="VALID"),
    "conv_1x8x8x16_f3d2_int8":    C(h=8, w=8, c=16, oc=16, fh=3, fw=3, dil=2),
    "dwconv_1x8x8x16_f3m2_int8":  C(kind="dwconv", h=8, w=8, c=16, fh=3, fw=3, mult=2),
    # --- BONUS: 64x64x64 matmul expressed as a 1x1 conv ------------------
    # input [1,64,1,64] (H=64 rows, W=1, C=64) x filter [64,1,1,64] -> [1,64,1,64]
    "conv_matmul_1x64x64_int8": C(h=64, w=1, c=64, oc=64, fh=1, fw=1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.list:
        for k in SPECS:
            print(k)
        return
    if args.spec not in SPECS:
        raise SystemExit(f"unknown spec {args.spec!r}; see --list")

    cfg = dict(SPECS[args.spec])
    kind = cfg.pop("kind")
    blob, meta, x, y, consts = build(kind, cfg)

    out = args.out or f"ops/{args.spec}/model.tflite"
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(os.path.join(d, "golden"), exist_ok=True)
    with open(out, "wb") as f:
        f.write(blob)
    x.tofile(os.path.join(d, "golden", "input.bin"))
    y.tofile(os.path.join(d, "golden", "expected.bin"))
    # filter/bias are constants inside the .tflite (the device never sees them as
    # runtime inputs); exporting them lets tools/verify_convpool.py re-derive the
    # golden with a second, loop-based implementation (AGENTS.md R3).
    if consts.get("filter") is not None:
        consts["filter"].tofile(os.path.join(d, "golden", "filter.bin"))
    if consts.get("bias") is not None:
        consts["bias"].tofile(os.path.join(d, "golden", "bias.bin"))
    meta["tflite"] = os.path.basename(out)
    meta["tflite_bytes"] = len(blob)
    with open(os.path.join(d, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{out}  {len(blob)} B  in{x.shape} out{y.shape} "
          f"tol={meta['int8_tol']} ref={meta['reference']}")


if __name__ == "__main__":
    main()
