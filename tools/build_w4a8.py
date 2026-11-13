#!/usr/bin/env python3
"""4W8A / 8W8A / 4W16A / 8W16A (int4|int8 weights, int8|int16 activations).

Tickets 04 (4W8A), 04c (8W8A, --w-bits 8) and 04d (4W16A, --a-bits 16).  ADR 0005.
Two independent knobs, one code path:

  --w-bits 4|8    weight format: int4 packed 2 codes/byte, or plain int8, both
                  symmetric per-output-channel (s[n] = max|W[n,:]|/qmax).
  --a-bits 8|16   activation format: int8 per-tensor (s = max|v|/127) or int16
                  per-tensor (s = max|v|/32767), bias still int32 per-channel,
                  requant still tools/quant_contract.py (two integer roundings).

Everything below reads the two globals W_BITS / A_BITS, so the four tiers share the
graph, the seam and the golden construction; only the code domains change.  Hard
constraints encoded here (each re-confirmed in docs/OP-w4a8-layer.md):

  * int4 weights are packed the TFLite way: 2 codes/byte, element 2i in the LOW
    nibble, 2i+1 in the HIGH nibble (probe-w4a8 board run: the opposite order
    misses 63/64 values).
  * per-channel weight quantisation requires the bias to be per-channel int32
    (ncc: "Bias shape should be per channel quant"), and the OUTPUT may not be
    per-channel ("Output cannot be per-channel quantization").
  * FULLY_CONNECTED op versions {1,3,4,5,6,7,9} are accepted; canonical v13 is
    rejected at load time, v2/v8/v10/v11 at backend planning time.
  * the 16-byte row rule still applies with int8 activations: a DRAM-boundary
    tensor whose innermost row is <16 B (or not a multiple of 16 B) bridges.
    p in P*V at T=8 is [H,8,8] = 8 B rows -> "Bridging is disallowed but gap is
    detected" (ops/bmm_pv_t8_int8).  Hence T=16 here.
  * the int8 requant contract is tools/quant_contract.py (two integer roundings);
    per-channel multipliers are NOT covered by that ticket (tensor-level only),
    so every op here registers int8_tol >= 1: ADR 0005 criterion is
    |delta| <= 1 code and cosine >= 0.9999, no bit-exactness promise.

Usage
    /usr/bin/python3 tools/build_w4a8.py --list
    /usr/bin/python3 tools/build_w4a8.py fc_proj --out ops/w4a8_fc_proj
    /usr/bin/python3 tools/build_w4a8.py layer   --out ops/decoder_layer_t16_w4a8
    /usr/bin/python3 tools/build_w4a8.py layer --w-bits 8 --out ops/decoder_layer_t16_w8a8
    /usr/bin/python3 tools/build_w4a8.py --weights-report     # per-matrix W error, 4 vs 8 bit
"""
import argparse
import json
import os
import sys

import flatbuffers
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ops as bo          # flatbuffer toolbox (read-only use, AGENTS 8)
import quant_contract as qc     # the int8 requant contract
import qwen25_weights as qw     # frozen fixture loader

I4 = bo.S.TensorType.INT4
I8 = bo.I8
I16 = bo.S.TensorType.INT16
I32 = bo.I32

# Ticket 04c: the weight knob.  4 -> int4 packed 2 codes/byte (ticket 04);
# 8 -> plain int8 codes, per-channel.
# Ticket 04d: the activation knob.  8 -> int8 per-tensor (tickets 04/04c);
# 16 -> int16 per-tensor, bias still int32 per-channel (ticket 04d).
# Both are set by argparse before any spec is built; every tensor type, every
# scale and every requant ratio below follows them, so all four tiers share one
# code path.
W_BITS = 4
A_BITS = 8

# How the masked softmax crosses dtypes in the LAYER graph (ticket 04d asks
# whether int16 activations still need the fp16 seam that int8 needed):
#   "f16"    -- int16 scores -> DEQUANTIZE fp16 -> x1/sqrt(d) -> +mask -> SOFTMAX
#               -> QUANTIZE int16   (the ticket 04 seam, dtype-parameterised)
#   "native" -- int16 scores -> +int16 mask (own scale) -> SOFTMAX int16
SEAM = "f16"


FORM = "w4a8"      # "w4a8" | "w8a8" | "w4a16" | "w8a16"; --w-bits / --a-bits


def wq_max(bits=None):
    """Largest magnitude a symmetric weight code may take at this tier."""
    return 7 if (bits or W_BITS) == 4 else 127


def aq_max():
    """Largest magnitude a symmetric ACTIVATION code may take (127 or 32767)."""
    return 127 if A_BITS == 8 else 32767


def a_dtype():
    return np.int8 if A_BITS == 8 else np.int16


def a_name():
    """model.json / graph-description name of the activation storage type."""
    return "int%d" % A_BITS


def a_type():
    """Flatbuffer TensorType of every activation tensor in this tier."""
    return I8 if A_BITS == 8 else I16


def clip_codes(x):
    """Clip integer codes into this tier's storage domain and cast to it."""
    lo = -128 if A_BITS == 8 else -32768
    hi = 127 if A_BITS == 8 else 32767
    return np.clip(x, lo, hi).astype(a_dtype())


def form_tag():
    """Tier tag used in graph descriptions / default op dirs / model.json."""
    return FORM


def set_tier(w_bits, a_bits=8):
    global W_BITS, A_BITS, FORM
    W_BITS = int(w_bits)
    A_BITS = int(a_bits)
    FORM = "w%da%d" % (W_BITS, A_BITS)

T = 16
D = 896
H = 14
KV = 2
HD = 64
HALF = HD // 2
INTER = 4864
KVR = H // KV
THETA = 1e6
LAYER = 0
EPS = 1e-6
SCALE = 1.0 / 8.0
MASK = -4096.0
TOKEN_IDS = [9707, 11, 1879, 828, 11780, 13, 220, 16]   # + arithmetic extras for T>8

ADR_TOL = ("ADR 0005: the 4-bit tier does not promise bit-exactness; the operator "
           "criterion is |delta| <= 1 code and cosine >= 0.9999. Per-channel "
           "multipliers are outside reports/Q02-int8-bitexact.md (tensor-level only), "
           "so int8_tol=1.")

TOL8 = ("ADR 0005 (2026-11-13 revision): the operator criterion applies to 8-bit and "
        "wider -- |delta| <= 1 code AND cosine >= 0.9999. 8W8A uses int8 weights, "
        "per-channel s[n]=max|W[n,:]|/127, int32 per-channel bias, and the two-step "
        "requant of tools/quant_contract.py; int8_tol=1 is registered here (the ADR "
        "allowance), while the ticket expects bit-exactness -- tools/compare.py prints "
        "the exact counts, so 0 codes is reported as 0/14336, never inferred from PASS.")


TOL16 = ("ADR 0005 (2026-11-13, second revision) + ticket 04d: this is the 16-bit "
         "ACTIVATION tier (4W16A). Operator criterion: device == quantised golden, "
         "|delta| <= int8_tol codes, AND |cos(device) - cos(golden)| <= 0.003 (device "
         "fidelity); quality is judged by cosine vs the independent fp64 chain. "
         "int8_tol=1 is registered before the first board run for components; the "
         "layer registers 2 for the int8 RMSNorm/int8 seam chain (see LAYER_TOL) or 1 "
         "when the graph is fully int16 (registered before the run, no adjustment "
         "afterwards).")


def tol_text():
    if A_BITS == 16:
        return TOL16
    return TOL8 if W_BITS == 8 else ADR_TOL


# --------------------------------------------------------------- quantisation math
def pack_i4(codes):
    """TFLite int4 layout: element 2i -> low nibble, 2i+1 -> high nibble."""
    f = np.asarray(codes, np.int64).ravel() & 0xF
    if f.size % 2:
        f = np.append(f, 0)
    return ((f[1::2] << 4) | f[0::2]).astype(np.uint8)


def s8(x):
    """Symmetric per-tensor activation quantisation at the active width: (codes, scale).

    Named s8 for historical reasons (tickets 04/04c); at --a-bits 16 it returns
    int16 codes in [-32767, 32767] with s = max|v|/32767.
    """
    x = np.asarray(x, np.float64)
    m = float(np.abs(x).max())
    q = aq_max()
    scale = m / float(q) if m > 0 else 1.0
    return np.clip(np.round(x / scale), -q, q).astype(a_dtype()), scale


def s16(x):
    """Explicit int16 activation codes regardless of the tier (probes only)."""
    x = np.asarray(x, np.float64)
    m = float(np.abs(x).max())
    scale = m / 32767.0 if m > 0 else 1.0
    return np.clip(np.round(x / scale), -32767, 32767).astype(np.int16), scale


def sw_weights(W, bits=None):
    """Symmetric per-output-channel weights for the active tier.

    codes[n,k] in [-qmax,qmax] with qmax = 7 (int4) or 127 (int8),
    scales[n] = max|W[n,:]| / qmax.  Only the mirror point and the divisor change
    between the two tiers -- the packing below is the only other difference.

    Note (ticket 04c): the division runs on the float64 scale, which is what
    sw4() did for the FC component in ticket 04; the old inline copy in
    q_fc_tables() divided by the float32 scale instead.  Both tiers now use ONE
    rule, so the 4-bit layer/SwiGLU golden moves by ~1e-4 in cosine relative to
    ticket 04 (see docs/OP-8w8a-layer.md 10) -- the measured device numbers in
    this ticket were all produced with this single rule.
    """
    W = np.asarray(W, np.float64)
    qmax = wq_max(bits)
    sc = np.maximum(np.abs(W).max(axis=1) / float(qmax), 1e-12)
    q = np.clip(np.round(W / sc[:, None]), -qmax, qmax).astype(np.int8)
    return q, sc.astype(np.float32)


def weight_bytes(codes):
    """Bytes this weight tensor occupies on disk: int4 packs 2 codes/byte."""
    codes = np.asarray(codes)
    return int(pack_i4(codes).size) if W_BITS == 4 else int(codes.nbytes)


def ratio_groups(ratios):
    """[(shift, [(channel, m0), ...])] -- group output channels by power-of-two shift."""
    g = {}
    for n, r in enumerate(np.asarray(ratios, np.float64).ravel()):
        m0, sh = qc.quantize_multiplier(r)
        g.setdefault(sh, []).append((n, m0))
    return sorted(g.items())


def requant_pc(acc, ratios):
    """Per-output-channel two-step requant (gemmlowp); acc [M,N], ratios [N]."""
    acc = np.asarray(acc, np.int64)
    out = np.zeros_like(acc)
    for sh, items in ratio_groups(ratios):
        for n, m0 in items:
            out[:, n] = qc.multiply_by_quantized_multiplier(acc[:, n], m0, sh)
    return out


def dequant(codes, scale):
    return np.asarray(codes, np.float64) * np.asarray(scale, np.float64)


def cosine(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ------------------------------------------------------------------------- fixture
def fixture():
    g = lambda n: qw.get("layer%d.%s" % (LAYER, n))
    return {
        "g1": g("input_layernorm.weight"),
        "Wq": g("attn.q_proj.weight"), "bq": g("attn.q_proj.bias"),
        "Wk": g("attn.k_proj.weight"), "bk": g("attn.k_proj.bias"),
        "Wv": g("attn.v_proj.weight"), "bv": g("attn.v_proj.bias"),
        "Wo": g("attn.o_proj.weight"),
        "g2": g("post_attention_layernorm.weight"),
        "Wg": g("mlp.gate_proj.weight"), "Wu": g("mlp.up_proj.weight"),
        "Wd": g("mlp.down_proj.weight"),
    }


def token_ids(t):
    vocab = int(qw.get("embed.weight").shape[0])
    ids = list(TOKEN_IDS[:t])
    i = 0
    while len(ids) < t:
        c = (41 + 1103 * i) % vocab
        i += 1
        if c not in ids:
            ids.append(c)
    return ids


def input_hidden(t):
    emb = qw.get("embed.weight")
    return np.ascontiguousarray(emb[np.asarray(token_ids(t))].astype(np.float32))


def rmsnorm_f32(x, gamma, eps=EPS):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * gamma


# ------------------------------------------------------------------- graph writer
class GQ:
    """Flatbuffer writer: int8 activations throughout, int4 FC weights."""

    def __init__(self):
        self.b = flatbuffers.Builder(1 << 20)
        self.buffers = [bo.buf_empty(self.b)]
        self.tensors, self.opcodes, self.ops = [], [], []
        self.idx = {}

    def _t(self, name, shape, ttype, buf, scales=None, zps=None, axis=0):
        q = None
        if scales is not None:
            if np.isscalar(scales) or np.asarray(scales).ndim == 0:
                scales, zps = [float(scales)], [int(zps or 0)]
            q = bo.quant_tensor(self.b, list(np.asarray(scales, np.float32)),
                                list(np.asarray(zps, np.int64)), axis=axis)
        self.tensors.append(bo.tensor(self.b, name, list(shape), ttype, buf, q))
        self.idx[name] = len(self.tensors) - 1
        return self.idx[name]

    def const_i8(self, name, codes, scale, zp=0):
        """Constant activation in the ACTIVE activation format (int8 or int16)."""
        a = np.ascontiguousarray(np.asarray(codes, a_dtype()))
        self.buffers.append(bo.buf_data(self.b, a))
        return self._t(name, a.shape, a_type(), len(self.buffers) - 1, scale, zp)

    def const_i32(self, name, vals):
        a = np.ascontiguousarray(np.asarray(vals, np.int32))
        self.buffers.append(bo.buf_data(self.b, a))
        return self._t(name, a.shape, I32, len(self.buffers) - 1)

    def const_i4(self, name, codes, scales, axis=0):
        """Weight tensor in the ACTIVE tier format, per-channel scales on axis 0.

        int4: packed 2 codes/byte (TFLite layout), tensor type INT4.
        int8: raw int8 codes, tensor type INT8 -- the stock int8 FC weight form.
        """
        shp = list(np.asarray(codes).shape)
        zps = [0] * len(np.atleast_1d(scales))
        if W_BITS == 4:
            self.buffers.append(bo.buf_data(self.b, pack_i4(codes)))
            return self._t(name, shp, I4, len(self.buffers) - 1,
                           list(np.asarray(scales, np.float32)), zps, axis=axis)
        a = np.ascontiguousarray(np.asarray(codes, np.int8))
        self.buffers.append(bo.buf_data(self.b, a))
        return self._t(name, shp, I8, len(self.buffers) - 1,
                       list(np.asarray(scales, np.float32)), zps, axis=axis)

    def act(self, name, shape, scale, zp=0, ttype=None):
        return self._t(name, shape, ttype or a_type(), 0, scale, zp)

    def _op(self, builtin, ins, outs, opt_type=None, opt=None, version=1):
        self.opcodes.append(bo.opcode(self.b, builtin, version=version))
        self.ops.append(bo.operator(self.b, len(self.opcodes) - 1, ins, outs, opt_type, opt))

    def fc_i4(self, inp, name, w_codes, w_scales, bias_q, s_a, out_scale, n, version=9):
        """FULLY_CONNECTED with per-channel weights of the ACTIVE tier + int32 bias."""
        tw = self.const_i4(name + "_W", w_codes, w_scales, axis=0)
        self.buffers.append(bo.buf_data(self.b, np.asarray(bias_q, np.int32)))
        tb = self._t(name + "_b", [len(bias_q)], I32, len(self.buffers) - 1,
                     (np.asarray(w_scales, np.float64) * s_a).tolist(),
                     [0] * len(bias_q), axis=0)
        t = self.act(name, (T, n), out_scale)
        S = bo.S
        S.FullyConnectedOptionsStart(self.b)
        S.FullyConnectedOptionsAddFusedActivationFunction(self.b, 0)
        S.FullyConnectedOptionsAddWeightsFormat(self.b, 0)
        S.FullyConnectedOptionsAddKeepNumDims(self.b, True)
        S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(self.b, False)
        opt = S.FullyConnectedOptionsEnd(self.b)
        self._op(bo.BuiltinOperator.FULLY_CONNECTED, [inp, tw, tb], [t],
                 bo.BuiltinOptions.FullyConnectedOptions, opt, version=version)
        return t

    def finish(self, inputs, outputs, desc):
        return bo.finish(self.b, self.buffers, self.opcodes, self.tensors, self.ops,
                         inputs, outputs, desc)


def w4a8_fc_tables(a_f64, W_f64, b_f64):
    """Quantised operands + the per-channel requant tables for one int8-activation FC."""
    a_i8, s_a = s8(a_f64)
    w_q, s_w = sw_weights(W_f64)
    bs = s_a * s_w.astype(np.float64)
    b_q = np.round(b_f64 / bs).astype(np.int32)
    acc = a_i8.astype(np.int64) @ w_q.astype(np.int64).T + b_q[None, :]
    o_ref = (a_i8.astype(np.float64) * s_a) @ (w_q.astype(np.float64) * s_w[:, None]).T + b_f64
    s_o = float(np.abs(o_ref).max()) / float(aq_max())
    ratios = bs / s_o
    return dict(a_i8=a_i8, s_a=s_a, w_q=w_q, s_w=s_w, b_q=b_q, bs=bs,
                s_o=s_o, ratios=ratios, acc=acc, o_ref=o_ref)


# ------------------------------------------------------------------- components
def build_fc_proj(P):
    """q_proj: n1 [T,896] int8 -> q [T,896] int8, per-channel weights at the active tier."""
    x = input_hidden(T).astype(np.float64)
    n1 = rmsnorm_f32(x, P["g1"].astype(np.float64))
    tb = w4a8_fc_tables(n1, P["Wq"].astype(np.float64), P["bq"].astype(np.float64))
    out_q = clip_codes(requant_pc(tb["acc"], tb["ratios"]))

    g = GQ()
    ta = g.act("input", (T, D), tb["s_a"])
    t = g.fc_i4(ta, "out", tb["w_q"], tb["s_w"], tb["b_q"], tb["s_a"], tb["s_o"], D)
    blob = g.finish([ta], [t], form_tag() + " q_proj T=%d" % T)
    meta = dict(op="FULLY_CONNECTED", form=FORM, shape={"T": T, "K": D, "N": D},
                dtype=a_name(), weights_dtype="int%d" % W_BITS, ops_version=9,
                input={"name": "input", "shape": [T, D], "type": a_name()},
                output={"name": "out", "shape": [T, D], "type": a_name()},
                quant={"input": tb["s_a"], "output": tb["s_o"],
                       "weights_pc_min": float(tb["s_w"].min()),
                       "weights_pc_max": float(tb["s_w"].max()),
                       "weights_bytes": weight_bytes(tb["w_q"]),
                       "weights_elems": int(tb["w_q"].size)},
                int8_tol=1, tolerance_rationale=tol_text())
    extra = dict(fp32=(n1 @ np.asarray(P["Wq"], np.float64).T
                       + np.asarray(P["bq"], np.float64)))
    return blob, meta, tb["a_i8"], out_q, extra


# --------------------------------------------------------------------- probes
def build_probe_add_scales(P):
    """Does ncc accept ADD of two int8 tensors with DIFFERENT scales (no requant op)?"""
    rng = np.random.default_rng(7)
    a = rng.integers(-100, 100, size=(T, 64)).astype(np.int8)
    b = rng.integers(-100, 100, size=(T, 64)).astype(np.int8)
    s_a, s_b, s_o = 1.0 / 64, 1.0 / 128, 1.0 / 64
    g = GQ()
    ta = g.act("a", (T, 64), s_a)
    tb = g.act("b", (T, 64), s_b)
    t = g.act("out", (T, 64), s_o)
    S = bo.S
    S.AddOptionsStart(g.b)
    S.AddOptionsAddFusedActivationFunction(g.b, 0)
    g._op(bo.BuiltinOperator.ADD, [ta, tb], [t], bo.BuiltinOptions.AddOptions,
          S.AddOptionsEnd(g.b))
    blob = g.finish([ta, tb], [t], "probe int8 add different scales")
    real = a.astype(np.float64) * s_a + b.astype(np.float64) * s_b
    out = np.clip(np.round(real / s_o), -128, 127).astype(np.int8)
    meta = dict(op="ADD", form="probe", shape={"T": T, "K": 64}, dtype="int8",
                input={"name": "a", "shape": [T, 64], "type": "int8"},
                output={"name": "out", "shape": [T, 64], "type": "int8"},
                quant={"a": s_a, "b": s_b, "out": s_o}, int8_tol=1,
                tolerance_rationale=tol_text(), dynamic_inputs=2)
    return blob, meta, a, out, dict(second=b)


def build_probe_quantize(P):
    """Does an explicit int8 -> int8 QUANTIZE (requant) compile as a native layer?"""
    rng = np.random.default_rng(11)
    a = rng.integers(-120, 120, size=(T, 64)).astype(np.int8)
    s_a, s_o = 1.0 / 32, 1.0 / 8
    g = GQ()
    ta = g.act("input", (T, 64), s_a)
    t = g.act("out", (T, 64), s_o)
    g._op(bo.BuiltinOperator.QUANTIZE, [ta], [t])
    blob = g.finish([ta], [t], "probe int8 quantize requant 1/32 -> 1/8")
    ratio = s_a / s_o
    m0, sh = qc.quantize_multiplier(ratio)
    out = np.clip(qc.multiply_by_quantized_multiplier(a.astype(np.int64), m0, sh),
                  -128, 127).astype(np.int8)
    meta = dict(op="QUANTIZE", form="probe", shape={"T": T, "K": 64}, dtype="int8",
                input={"name": "input", "shape": [T, 64], "type": "int8"},
                output={"name": "out", "shape": [T, 64], "type": "int8"},
                quant={"input": s_a, "output": s_o}, int8_tol=1,
                tolerance_rationale=tol_text())
    return blob, meta, a, out, {}


# ------------------------------------------------------- fp16 seam (DEQUANTIZE)
class TF:
    """A float16 tensor in the graph (used only for the masked-softmax seam)."""
    __slots__ = ("i", "v")

    def __init__(self, i, v):
        self.i, self.v = i, np.asarray(v, np.float64)


def _add_f16_methods():
    def const_f16(self, name, arr):
        a = np.ascontiguousarray(np.asarray(arr, np.float16))
        self.buffers.append(bo.buf_data(self.b, a))
        return self._t(name, a.shape, bo.F16, len(self.buffers) - 1)

    def act_f16(self, name, shape):
        return self._t(name, shape, bo.F16, 0)

    def f16_dequant(self, a, name):
        """int8 -> fp16 DEQUANTIZE: one native layer, no bridge (ADR 0005 mixed graph)."""
        t = self.act_f16(name, a.c.shape)
        self._op(bo.BuiltinOperator.DEQUANTIZE, [a.i], [t])
        # the device stores the result as fp16 -> the golden rounds there too
        return TF(t, np.asarray(a.real, np.float16).astype(np.float64))

    def f16_add(self, a, b_idx, name, b_val):
        t = self.act_f16(name, a.v.shape)
        S = bo.S
        S.AddOptionsStart(self.b)
        S.AddOptionsAddFusedActivationFunction(self.b, 0)
        self._op(bo.BuiltinOperator.ADD, [a.i, b_idx], [t], bo.BuiltinOptions.AddOptions,
                 S.AddOptionsEnd(self.b))
        return TF(t, np.asarray(a.v + b_val, np.float16).astype(np.float64))

    def f16_mul_scalar(self, a, f, name):
        c = np.full(a.v.shape, np.float16(f), np.float16)
        tc = self.const_f16(name + "_c", c)
        t = self.act_f16(name, a.v.shape)
        S = bo.S
        S.MulOptionsStart(self.b)
        S.MulOptionsAddFusedActivationFunction(self.b, 0)
        self._op(bo.BuiltinOperator.MUL, [a.i, tc], [t], bo.BuiltinOptions.MulOptions,
                 S.MulOptionsEnd(self.b))
        return TF(t, np.asarray(a.v * c.astype(np.float64), np.float16).astype(np.float64))

    def f16_softmax(self, a, name):
        t = self.act_f16(name, a.v.shape)
        S = bo.S
        S.SoftmaxOptionsStart(self.b)
        S.SoftmaxOptionsAddBeta(self.b, 1.0)
        self._op(bo.BuiltinOperator.SOFTMAX, [a.i], [t], bo.BuiltinOptions.SoftmaxOptions,
                 S.SoftmaxOptionsEnd(self.b))
        return TF(t, np.asarray(softmax_last(a.v), np.float16).astype(np.float64))

    def f16_quantize(self, a, scale, name):
        codes = clip_codes(np.round(a.v / scale))
        t = self.act(name, a.v.shape, scale)
        self._op(bo.BuiltinOperator.QUANTIZE, [a.i], [t])
        return TS(t, codes, scale)

    for f in (const_f16, act_f16, f16_dequant, f16_mul_scalar, f16_add, f16_softmax,
              f16_quantize):
        setattr(GQ, f.__name__, f)


_add_f16_methods()


# ------------------------------------------------------------ fp32 reference chain
def rope_tables(heads):
    inv = THETA ** (-2.0 * np.arange(HALF) / HD)
    ang = np.outer(np.arange(T), inv)
    c = np.broadcast_to(np.cos(ang)[:, None, :], (T, heads, HALF))
    s = np.broadcast_to(np.sin(ang)[:, None, :], (T, heads, HALF))
    return np.ascontiguousarray(c.copy()), np.ascontiguousarray(s.copy())


def causal_mask():
    m = np.zeros((T, T))
    m[np.triu_indices(T, k=1)] = MASK
    return np.ascontiguousarray(np.broadcast_to(m, (H, T, T)).copy())


def softmax_last(v):
    m = v - v.max(axis=-1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(axis=-1, keepdims=True)


def chain_f32(P):
    """Independent float64 chain of the whole layer: the ACCURACY reference.

    Written from the model definition, not from the flatbuffer code (AGENTS R3).
    """
    x = input_hidden(T).astype(np.float64)
    out = {"x": x}
    n1 = rmsnorm_f32(x, P["g1"].astype(np.float64))
    out["n1"] = n1

    def proj(W, b, heads):
        return (n1 @ np.asarray(W, np.float64).T + np.asarray(b, np.float64)).reshape(T, heads, HD)

    def rope(y3, heads):
        c, s = rope_tables(heads)
        a, bb = y3[..., :HALF], y3[..., HALF:]
        C, S = c, s     # rope_tables already carries the head axis
        return np.concatenate([a * C - bb * S, bb * C + a * S], axis=-1)

    q3 = rope(proj(P["Wq"], P["bq"], H), H)
    k3 = rope(proj(P["Wk"], P["bk"], KV), KV)
    v3 = proj(P["Wv"], P["bv"], KV)
    out.update(q3=q3, k3=k3, v3=v3)
    k14 = np.repeat(k3, KVR, axis=1)
    v14 = np.repeat(v3, KVR, axis=1)
    out.update(k14=k14, v14=v14)
    qt, kt, vt = (np.transpose(z, (1, 0, 2)) for z in (q3, k14, v14))
    scores = qt @ np.transpose(kt, (0, 2, 1))
    out["scores"] = scores
    scaled = scores * SCALE
    masked = scaled + causal_mask()
    probs = softmax_last(masked)
    out["probs"] = probs
    a3 = probs @ vt
    attn = np.transpose(a3, (1, 0, 2)).reshape(T, D)
    out["attn"] = attn
    o = attn @ np.asarray(P["Wo"], np.float64).T
    r1 = o + x
    out["r1"] = r1
    n2 = rmsnorm_f32(r1, P["g2"].astype(np.float64))
    out["n2"] = n2
    gate = n2 @ np.asarray(P["Wg"], np.float64).T
    up = n2 @ np.asarray(P["Wu"], np.float64).T
    sig = 1.0 / (1.0 + np.exp(-gate))
    y = gate * sig * up
    out["mlp_y"] = y
    down = y @ np.asarray(P["Wd"], np.float64).T
    out["down"] = down
    out["out"] = down + r1
    return out


# --------------------------------------------------------------- quantised tensor
class TS:
    """A quantised tensor: flatbuffer index + int8 codes + per-tensor scale/zp."""
    __slots__ = ("i", "c", "s", "zp")

    def __init__(self, i, c, s, zp=0):
        self.i, self.c, self.s, self.zp = i, c, s, zp

    @property
    def real(self):
        return (self.c.astype(np.float64) - self.zp) * self.s


def clip_sym(real):
    """Codes of a value in [0,1] stored in the SYMMETRIC 1/aq_max domain (P*V operand)."""
    q = aq_max()
    return np.clip(np.round(np.asarray(real, np.float64) * q), -q, q).astype(a_dtype())


def lsm_scale():
    """LOGISTIC / SOFTMAX output scale, TFLite's convention for the active width."""
    return 1.0 / 256.0 if A_BITS == 8 else 1.0 / 65536.0


def lsm_zp():
    """LOGISTIC / SOFTMAX output zero point, TFLite's convention for the active width."""
    return -128 if A_BITS == 8 else -32768


def lsm_codes(real):
    """Fixed-point codes of a [0,1]-valued op output (logistic/softmax) at this width."""
    return clip_codes(np.round(np.asarray(real, np.float64) / lsm_scale()) + lsm_zp())


def requant_scalar(acc, ratio):
    m0, sh = qc.quantize_multiplier(ratio)
    return qc.multiply_by_quantized_multiplier(np.asarray(acc, np.int64), m0, sh)


def pick_scale(real):
    """Per-tensor symmetric scale for a real-valued array, in this tier's code domain."""
    m = float(np.abs(np.asarray(real, np.float64)).max())
    return (m / float(aq_max())) if m > 0 else 1.0


class QL(GQ):
    """GQ plus int8 op helpers: each call computes the device-model codes AND emits
    the flatbuffer operator, so the golden and the graph cannot drift apart."""

    def mk_in(self, name, shape, codes, scale, zp=0):
        return TS(self.act(name, shape, scale, zp), np.asarray(codes, a_dtype()), scale, zp)

    def q_requant(self, a, so, name, zp_o=0, factor=1.0):
        """Re-quantise into a new scale; factor scales the REAL value on the way (the
        1/sqrt(d) of the softmax input is folded here, not into a separate MUL)."""
        codes = clip_codes(requant_scalar(a.c.astype(np.int64) - a.zp, a.s * factor / so) + zp_o)
        t = self.act(name, a.c.shape, so, zp_o)
        self._op(bo.BuiltinOperator.QUANTIZE, [a.i], [t])
        return TS(t, codes, so, zp_o)

    def q_mul(self, a, b, so, name):
        acc = (a.c.astype(np.int64) - a.zp) * (b.c.astype(np.int64) - b.zp)
        codes = clip_codes(requant_scalar(acc, a.s * b.s / so))
        t = self.act(name, codes.shape, so)
        S = bo.S
        S.MulOptionsStart(self.b)
        S.MulOptionsAddFusedActivationFunction(self.b, 0)
        self._op(bo.BuiltinOperator.MUL, [a.i, b.i], [t], bo.BuiltinOptions.MulOptions,
                 S.MulOptionsEnd(self.b))
        return TS(t, codes, so)

    def q_addsub(self, a, b, so, name, minus=False):
        """Different scales allowed (ncc inserts the requant; probes/_w4a8_probe_add)."""
        real = a.real - b.real if minus else a.real + b.real
        codes = clip_codes(np.round(real / so))
        t = self.act(name, codes.shape, so)
        S = bo.S
        if minus:
            S.SubOptionsStart(self.b)
            S.SubOptionsAddFusedActivationFunction(self.b, 0)
            self._op(bo.BuiltinOperator.SUB, [a.i, b.i], [t], bo.BuiltinOptions.SubOptions,
                     S.SubOptionsEnd(self.b))
        else:
            S.AddOptionsStart(self.b)
            S.AddOptionsAddFusedActivationFunction(self.b, 0)
            self._op(bo.BuiltinOperator.ADD, [a.i, b.i], [t], bo.BuiltinOptions.AddOptions,
                     S.AddOptionsEnd(self.b))
        return TS(t, codes, so)

    def q_add_const(self, a, const_codes, so, name):
        """ADD a constant tensor that shares a's scale (exact code addition)."""
        tc = self.const_i8(name + "_c", const_codes, a.s, a.zp)
        codes = clip_codes(a.c.astype(np.int64) + np.asarray(const_codes, np.int64))
        t = self.act(name, a.c.shape, so)
        S = bo.S
        S.AddOptionsStart(self.b)
        S.AddOptionsAddFusedActivationFunction(self.b, 0)
        self._op(bo.BuiltinOperator.ADD, [a.i, tc], [t], bo.BuiltinOptions.AddOptions,
                 S.AddOptionsEnd(self.b))
        return TS(t, codes, so)

    def q_mean(self, a, so, name):
        real = a.real.mean(axis=1, keepdims=True)
        codes = clip_codes(np.round(real / so))
        ta = self.const_i32(name + "_axis", [1])
        t = self.act(name, codes.shape, so)
        S = bo.S
        S.ReducerOptionsStart(self.b)
        S.ReducerOptionsAddKeepDims(self.b, True)
        self._op(bo.BuiltinOperator.MEAN, [a.i, ta], [t], bo.BuiltinOptions.ReducerOptions,
                 S.ReducerOptionsEnd(self.b))
        return TS(t, codes, so)

    def q_rsqrt(self, a, so, name):
        real = 1.0 / np.sqrt(a.real)
        codes = clip_codes(np.round(real / so))
        t = self.act(name, codes.shape, so)
        self._op(bo.BuiltinOperator.RSQRT, [a.i], [t])
        return TS(t, codes, so)

    def q_logistic(self, a, name):
        real = 1.0 / (1.0 + np.exp(-a.real))
        codes = lsm_codes(real)
        t = self.act(name, codes.shape, lsm_scale(), lsm_zp())
        self._op(bo.BuiltinOperator.LOGISTIC, [a.i], [t])
        return TS(t, codes, lsm_scale(), lsm_zp())

    def q_softmax(self, a, name, scale=None, zp=None):
        """SOFTMAX with the tier's fixed-point output convention (overridable)."""
        real = softmax_last(a.real)
        sc = lsm_scale() if scale is None else scale
        z = lsm_zp() if zp is None else zp
        codes = clip_codes(np.round(real / sc) + z)
        t = self.act(name, codes.shape, sc, z)
        S = bo.S
        S.SoftmaxOptionsStart(self.b)
        S.SoftmaxOptionsAddBeta(self.b, 1.0)
        self._op(bo.BuiltinOperator.SOFTMAX, [a.i], [t], bo.BuiltinOptions.SoftmaxOptions,
                 S.SoftmaxOptionsEnd(self.b))
        return TS(t, codes, sc, z)

    def q_bmm(self, a, b, so, name, adj_y=False):
        av, bv = a.c.astype(np.int64) - a.zp, b.c.astype(np.int64) - b.zp
        acc = av @ np.swapaxes(bv, -1, -2) if adj_y else av @ bv
        codes = clip_codes(requant_scalar(acc, a.s * b.s / so))
        t = self.act(name, codes.shape, so)
        S = bo.S
        S.BatchMatMulOptionsStart(self.b)
        S.BatchMatMulOptionsAddAdjX(self.b, False)
        S.BatchMatMulOptionsAddAdjY(self.b, bool(adj_y))
        S.BatchMatMulOptionsAddAsymmetricQuantizeInputs(self.b, False)
        self._op(bo.BuiltinOperator.BATCH_MATMUL, [a.i, b.i], [t],
                 bo.BuiltinOptions.BatchMatMulOptions, S.BatchMatMulOptionsEnd(self.b))
        return TS(t, codes, so)

    def q_shape(self, a, name, shape):           # RESHAPE
        t = self.act(name, shape, a.s, a.zp)
        tsh = self.const_i32(name + "_shape", list(shape))
        nsv = self.b.CreateNumpyVector(np.asarray(shape, np.int32))
        S = bo.S
        S.ReshapeOptionsStart(self.b)
        S.ReshapeOptionsAddNewShape(self.b, nsv)
        self._op(bo.BuiltinOperator.RESHAPE, [a.i, tsh], [t],
                 bo.BuiltinOptions.ReshapeOptions, S.ReshapeOptionsEnd(self.b))
        return TS(t, a.c.reshape(shape), a.s, a.zp)

    def q_transpose(self, a, name, perm):
        shape = [a.c.shape[p] for p in perm]
        t = self.act(name, shape, a.s, a.zp)
        tp = self.const_i32(name + "_perm", list(perm))
        S = bo.S
        S.TransposeOptionsStart(self.b)
        self._op(bo.BuiltinOperator.TRANSPOSE, [a.i, tp], [t],
                 bo.BuiltinOptions.TransposeOptions, S.TransposeOptionsEnd(self.b))
        return TS(t, np.transpose(a.c, perm), a.s, a.zp)

    def q_split(self, a, name, axis, num, out_shape):
        ta = self.const_i32(name + "_axis", [axis])
        ts = [self.act(name + str(k), out_shape, a.s, a.zp) for k in range(num)]
        S = bo.S
        S.SplitOptionsStart(self.b)
        S.SplitOptionsAddNumSplits(self.b, num)
        self._op(bo.BuiltinOperator.SPLIT, [ta, a.i], ts,
                 bo.BuiltinOptions.SplitOptions, S.SplitOptionsEnd(self.b))
        parts = np.split(a.c, num, axis=axis)
        return [TS(ts[k], np.ascontiguousarray(parts[k]), a.s, a.zp) for k in range(num)]

    def q_concat(self, ins, name, axis, out_shape):
        t = self.act(name, out_shape, ins[0].s, ins[0].zp)
        S = bo.S
        S.ConcatenationOptionsStart(self.b)
        S.ConcatenationOptionsAddAxis(self.b, axis)
        S.ConcatenationOptionsAddFusedActivationFunction(self.b, 0)
        self._op(bo.BuiltinOperator.CONCATENATION, [z.i for z in ins], [t],
                 bo.BuiltinOptions.ConcatenationOptions, S.ConcatenationOptionsEnd(self.b))
        return TS(t, np.concatenate([z.c for z in ins], axis=axis), ins[0].s, ins[0].zp)

    def q_fc(self, a, W_i4, s_w, b_q, so, n, name, version=9):
        tw = self.const_i4(name + "_W", W_i4, s_w, axis=0)
        self.buffers.append(bo.buf_data(self.b, np.asarray(b_q, np.int32)))
        tb = self._t(name + "_b", [len(b_q)], I32, len(self.buffers) - 1,
                     (np.asarray(s_w, np.float64) * a.s).tolist(), [0] * len(b_q), axis=0)
        acc = a.c.astype(np.int64) @ W_i4.astype(np.int64).T + np.asarray(b_q, np.int64)[None, :]
        ratios = (a.s * np.asarray(s_w, np.float64)) / so
        codes = clip_codes(requant_pc(acc, ratios))
        t = self.act(name, (a.c.shape[0], n), so)
        S = bo.S
        S.FullyConnectedOptionsStart(self.b)
        S.FullyConnectedOptionsAddFusedActivationFunction(self.b, 0)
        S.FullyConnectedOptionsAddWeightsFormat(self.b, 0)
        S.FullyConnectedOptionsAddKeepNumDims(self.b, True)
        S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(self.b, False)
        opt = S.FullyConnectedOptionsEnd(self.b)
        self._op(bo.BuiltinOperator.FULLY_CONNECTED, [a.i, tw, tb], [t],
                 bo.BuiltinOptions.FullyConnectedOptions, opt, version=version)
        return TS(t, codes, so)


def q_fc_tables(a_codes, a_scale, W_f64, b_f64):
    """Per-channel weights of the ACTIVE tier + int32 per-channel bias for one FC."""
    W = np.asarray(W_f64, np.float64)
    wq, s_w = sw_weights(W)
    bs = a_scale * s_w.astype(np.float64)
    b_q = np.round(np.asarray(b_f64, np.float64) / bs).astype(np.int32)
    o_ref = (a_codes.astype(np.float64) * a_scale) @ (wq.astype(np.float64) * s_w[:, None]).T \
        + np.asarray(b_f64, np.float64)
    return wq, s_w, b_q, pick_scale(o_ref), o_ref


# ----------------------------------------------------------------- components
def _to_symmetric(g, p, name):
    """Rescale a [0,1] fixed-point tensor (zp = -2^k) into the symmetric 1/aq_max domain.

    MUL by the exact constant 1.0: the device requants into the declared output scale,
    which is the same two-step path every other MUL in this graph uses.
    """
    one = np.full(p.c.shape, 1, a_dtype())      # code 1, scale 1.0: ratio ~0.5, no
    t = TS(g.const_i8(name + "_one", one, 1.0), one, 1.0)   # left shift
    return g.q_mul(p, t, 1.0 / float(aq_max()), name)


def _seam_native(g, scores, sym):
    """int16 masked softmax in three (or four) native layers, no fp16 seam.

    Order matters: the causal mask (-4096) is only representable in the RAW score
    domain (codes -4096/s_s = -28500 of +-32767); after the 1/sqrt(d) the same mask
    would need 8x the code range.  So: mask first (exact code add), then multiply by
    1/sqrt(HD) = 1/8 (the MUL requant is bit-exact on device: ops/_w4a16_probe/p16_mul),
    then SOFTMAX.  sym=True declares the SOFTMAX output symmetric (1/aq_max, zp 0) so
    P*V can consume it directly; sym=False uses TFLite's (1/65536, zp -32768) and pays
    one exact rescale MUL (ops/_w4a16_probe/p16_mul_one).
    """
    # the causal pattern, as int16 codes in the raw score domain (-4096 / s_s = -28500)
    mask_codes = np.round(causal_mask() / scores.s).astype(np.int64)
    masked = g.q_add_const(scores, mask_codes, scores.s, "masked")
    # code 1 with scale 1/8 (NOT code 4096 with scale 1/32768): the MUL accumulator
    # then stays at int16 magnitude and the ratio is ~1.34, i.e. a right shift.  With
    # full-scale codes the ratio becomes 3.3e-4, shift=+12 and the requant wraps int64
    # (ops/_w4a16_probe/p16_native_seam was the first casualty of this).
    # code 1 with scale 1/8: the MUL accumulator stays at int16 magnitude and the
    # requant ratio lands near 1 (a right shift), which is also what keeps the golden
    # and the device on the same side of every rounding boundary.
    e = np.full(masked.c.shape, 1, a_dtype())
    ec = TS(g.const_i8("eighth", e, 0.125), e, 0.125)
    scaled = g.q_mul(masked, ec, pick_scale(masked.real / 8.0), "scaled")
    if sym:
        p = g.q_softmax(scaled, "probs", scale=1.0 / float(aq_max()), zp=0)
        return p, p
    p = g.q_softmax(scaled, "probs")
    return p, _to_symmetric(g, p, "p_sym")


def _rmsnorm5(g, a, gamma, tag):
    """int8 RMSNorm, 5 layers: MUL(x,x) MEAN RSQRT MUL(x,r) MUL(gamma).

    The eps ADD is gone: at int8 resolution round(eps/scale) == 0 for every scale
    that keeps the data in range, so ncc constant-folds it (docs/OP-rmsnorm.md 2.5).
    Registered consequence: the int8 chain is x/sqrt(mean(x^2)), i.e. eps = 0.
    """
    sq_scale = pick_scale(a.real * a.real)
    sq = g.q_mul(a, a, sq_scale, tag + "_sq")
    mean = g.q_mean(sq, pick_scale(sq.real.mean(axis=1, keepdims=True)), tag + "_mean")
    rs = g.q_rsqrt(mean, pick_scale(1.0 / np.sqrt(np.maximum(mean.real, 1e-30))), tag + "_rsqrt")
    sc = g.q_mul(a, rs, pick_scale(a.real * rs.real), tag + "_scaled")
    gc, gs = s8(gamma)
    ty = TS(g.const_i8(tag + "_gamma", gc, gs), gc, gs)
    return g.q_mul(sc, ty, pick_scale(sc.real * (gc.astype(np.float64) * gs)), tag + "_norm")


def build_rmsnorm(P):
    ch = chain_f32(P)
    x_c, s_x = s8(ch["x"])
    g = QL()
    a = g.mk_in("input", (T, D), x_c, s_x)
    y = _rmsnorm5(g, a, P["g1"], "n1")
    blob = g.finish([a.i], [y.i], form_tag() + " int8 rmsnorm T=%d" % T)
    meta = dict(op="RMSNORM", form=FORM, shape={"T": T, "D": D}, dtype=a_name(),
                layers_expected=5,
                eps_register="dropped (constant-folded at int8 resolution), see ADR 0005",
                input={"name": "input", "shape": [T, D], "type": a_name()},
                output={"name": "out", "shape": [T, D], "type": a_name()},
                quant={"input": s_x, "output": y.s}, int8_tol=1,
                tolerance_rationale=tol_text())
    return blob, meta, x_c, y.c, dict(fp32=ch["n1"])


def build_swiglu(P):
    ch = chain_f32(P)
    n2 = ch["n2"]
    a_c, s_a = s8(n2)
    g = QL()
    a = g.mk_in("input", (T, D), a_c, s_a)
    wg, swg, bqg, s_gate, _ = q_fc_tables(a_c, s_a, P["Wg"], np.zeros(INTER))
    gate = g.q_fc(a, wg, swg, bqg, s_gate, INTER, "gate")
    wu, swu, bqu, s_up, _ = q_fc_tables(a_c, s_a, P["Wu"], np.zeros(INTER))
    up = g.q_fc(a, wu, swu, bqu, s_up, INTER, "up")
    sig = g.q_logistic(gate, "sig")
    s1 = g.q_mul(gate, sig, pick_scale(gate.real * sig.real), "silu")
    y = g.q_mul(s1, up, pick_scale(s1.real * up.real), "mlp_y")
    wd, swd, bqd, s_down, _ = q_fc_tables(y.c, y.s, P["Wd"], np.zeros(D))
    down = g.q_fc(y, wd, swd, bqd, s_down, D, "down")
    blob = g.finish([a.i], [down.i], form_tag() + " swiglu T=%d" % T)
    meta = dict(op="SWIGLU", form=FORM, shape={"T": T, "D": D, "I": INTER},
                dtype=a_name(), weights_dtype="int%d" % W_BITS,
                input={"name": "input", "shape": [T, D], "type": a_name()},
                output={"name": "out", "shape": [T, D], "type": a_name()},
                quant={"input": s_a, "gate": s_gate, "up": s_up, "mlp_y": y.s,
                       "output": down.s}, int8_tol=1, tolerance_rationale=tol_text())
    return blob, meta, a_c, down.c, dict(fp32=ch["down"], y=y.c, s_y=y.s)


def build_bmm_qk(P):
    ch = chain_f32(P)
    qt = np.transpose(ch["q3"], (1, 0, 2))
    kt = np.transpose(ch["k14"], (1, 0, 2))
    q_c, s_q = s8(qt)
    k_c, s_k = s8(kt)
    s_s = pick_scale(ch["scores"])
    g = QL()
    a = g.mk_in("a", (H, T, HD), q_c, s_q)
    b = g.mk_in("b", (H, T, HD), k_c, s_k)
    sc = g.q_bmm(a, b, s_s, "scores", adj_y=True)
    blob = g.finish([a.i, b.i], [sc.i], form_tag() + " int8 bmm qk T=%d" % T)
    meta = dict(op="BATCH_MATMUL", form=FORM, matmul="Q.K^T",
                shape={"H": H, "T": T, "HD": HD, "adj_y": 1}, dtype=a_name(),
                inputs=[{"name": "a", "shape": [H, T, HD], "type": a_name()},
                        {"name": "b", "shape": [H, T, HD], "type": a_name()}],
                output={"name": "scores", "shape": [H, T, T], "type": a_name()},
                quant={"a": s_q, "b": s_k, "output": s_s}, int8_tol=1,
                tolerance_rationale=tol_text())
    return blob, meta, q_c, sc.c, dict(second=k_c, fp32=ch["scores"])


def build_bmm_pv(P):
    ch = chain_f32(P)
    p = ch["probs"]
    vt = np.transpose(ch["v14"], (1, 0, 2))
    p_c = clip_sym(p)   # symmetric 1/aq_max domain
    v_c, s_v = s8(vt)
    a3 = np.transpose(ch["probs"], (0, 1, 2)) @ vt
    s_o = pick_scale(a3)
    g = QL()
    a = g.mk_in("a", (H, T, T), p_c, 1.0 / float(aq_max()))
    b = g.mk_in("b", (H, T, HD), v_c, s_v)
    o = g.q_bmm(a, b, s_o, "attn")
    blob = g.finish([a.i, b.i], [o.i], form_tag() + " int8 bmm pv T=%d" % T)
    meta = dict(op="BATCH_MATMUL", form=FORM, matmul="P.V",
                shape={"H": H, "T": T, "HD": HD}, dtype=a_name(),
                inputs=[{"name": "a", "shape": [H, T, T], "type": a_name()},
                        {"name": "b", "shape": [H, T, HD], "type": a_name()}],
                output={"name": "attn", "shape": [H, T, HD], "type": a_name()},
                quant={"a": 1.0 / float(aq_max()), "b": s_v, "output": s_o}, int8_tol=1,
                tolerance_rationale=tol_text())
    return blob, meta, p_c, o.c, dict(second=v_c, fp32=a3)


# ------------------------------------------------------------------ whole layer
STAGES = ["n1", "q3", "scores", "probs", "attn", "r1", "n2", "mlp_y", "down", "out"]
STAGE_KEY = {"scores": "scores", "probs": "probs", "attn": "attn", "r1": "r1",
             "n2": "n2", "mlp_y": "mlp_y", "down": "down", "out": "out",
             "q3": "q3", "n1": "n1"}


def build_layer(P, upto="out"):
    ch = chain_f32(P)
    g = QL()
    x_c, s_x = s8(ch["x"])
    x = g.mk_in("x", (T, D), x_c, s_x)

    n1 = _rmsnorm5(g, x, P["g1"], "n1")
    if upto == "n1":
        return finish_layer(g, x, n1, ch, "n1")

    def qkv(W, b, heads, tag, rope=True):
        w, sw, bq, so, _ = q_fc_tables(n1.c, n1.s, W, b)
        y = g.q_fc(n1, w, sw, bq, so, heads * HD, tag + "_fc")
        y3 = g.q_shape(y, tag + "_3d", (T, heads, HD))
        if not rope:
            return y3
        c, s = rope_tables(heads)
        cc, sc = s8(c)
        ss, sg = s8(s)
        tcc = TS(g.const_i8(tag + "_cos", cc, sc), cc, sc)
        tss = TS(g.const_i8(tag + "_sin", ss, sg), ss, sg)
        lo_c, hi_c = g.q_split(y3, tag + "_split", 2, 2, (T, heads, HALF))
        p1 = g.q_mul(lo_c, tcc, pick_scale(lo_c.real * c), tag + "_ac")
        p2 = g.q_mul(hi_c, tss, pick_scale(hi_c.real * s), tag + "_bs")
        lo = g.q_addsub(p1, p2, pick_scale(lo_c.real * c - hi_c.real * s), tag + "_lo", minus=True)
        p3 = g.q_mul(hi_c, tcc, pick_scale(hi_c.real * c), tag + "_bc")
        p4 = g.q_mul(lo_c, tss, pick_scale(lo_c.real * s), tag + "_as")
        hi = g.q_addsub(p3, p4, pick_scale(hi_c.real * c + lo_c.real * s), tag + "_hi")
        return g.q_concat([lo, hi], tag + "_rope", 2, (T, heads, HD))

    q3 = qkv(P["Wq"], P["bq"], H, "q")
    if upto == "q3":
        return finish_layer(g, x, q3, ch, "q3")
    k3 = qkv(P["Wk"], P["bk"], KV, "k")
    wv, swv, bqv, s_vfc, _ = q_fc_tables(n1.c, n1.s, P["Wv"], P["bv"])
    v_fc = g.q_fc(n1, wv, swv, bqv, s_vfc, KV * HD, "v_fc")
    v3 = g.q_shape(v_fc, "v_3d", (T, KV, HD))

    def kv14(t3, tag):
        k0, k1 = g.q_split(t3, tag + "_split2", 1, 2, (T, 1, HD))
        return g.q_concat([k0] * KVR + [k1] * KVR, tag + "14", 1, (T, H, HD))

    k14 = kv14(k3, "k")
    v14 = kv14(v3, "v")
    qt = g.q_transpose(q3, "qt", [1, 0, 2])
    kt = g.q_transpose(k14, "kt", [1, 0, 2])
    vt = g.q_transpose(v14, "vt", [1, 0, 2])
    scores = g.q_bmm(qt, kt, pick_scale(ch["scores"]), "scores", adj_y=True)
    if upto == "scores":
        return finish_layer(g, x, scores, ch, "scores")
    # The masked softmax seam.  Ticket 04 (int8 activations) had to cross into fp16:
    #  1. the causal mask (-4096) is not representable in an int8 score domain (codes span
    #     +-127 * s, s = 37 here), so an int8 mask ADD cannot be exact;
    #  2. an int8 SOFTMAX over these scores does not compile at all (ncc: "Large scale
    #     quant Softmax may have accuracy issue", EDPA unsupported -> no execution plan).
    # Ticket 04d asks whether int16 activations still need that seam; SEAM decides and
    # both branches are built from the same golden machinery:
    #  "f16"    : DEQUANTIZE int16->fp16, x1/sqrt(d), +mask(-4096), SOFTMAX, QUANTIZE back
    #             (the ticket 04 graph, dtype-parameterised -- 2 extra layers per seam).
    #  "native" : the int16 score domain has 256x the int8 resolution, so the SAME -4096
    #             mask is representable in codes (-4096 / 0.1437 = -28500) and can be added
    #             as an exact constant; SOFTMAX stays int16.  No fp16 seam at all.
    #  "native_sym": as native, but the SOFTMAX output is declared symmetric (1/aq_max, zp 0)
    #             so P*V takes it without a rescale MUL.
    if SEAM in ("native", "native_sym"):
        p, p_sym = _seam_native(g, scores, SEAM == "native_sym")
        if upto == "probs":
            return finish_layer(g, x, p, ch, "probs")
    else:
        f16 = g.f16_dequant(scores, "scores_f16")
        f16 = g.f16_mul_scalar(f16, SCALE, "scaled")
        tmask = g.const_f16("mask", causal_mask())
        masked = g.f16_add(f16, tmask, "masked", causal_mask())
        p = g.f16_softmax(masked, "probs")
        if upto == "probs":
            return finish_layer(g, x, p, ch, "probs")
        p_sym = g.f16_quantize(p, 1.0 / float(aq_max()), "p_sym")
    a3 = g.q_bmm(p_sym, vt, pick_scale(np.transpose(ch["probs"], (0, 1, 2)) @ np.transpose(ch["v14"], (1, 0, 2))),
                 "a3")
    at = g.q_transpose(a3, "at", [1, 0, 2])
    attn = g.q_shape(at, "attn", (T, D))
    if upto == "attn":
        return finish_layer(g, x, attn, ch, "attn")
    wo, swo, bqo, s_o, _ = q_fc_tables(attn.c, attn.s, P["Wo"], np.zeros(D))
    o = g.q_fc(attn, wo, swo, bqo, s_o, D, "o_proj")
    r1 = g.q_addsub(o, x, pick_scale(ch["r1"]), "r1")
    if upto == "r1":
        return finish_layer(g, x, r1, ch, "r1")
    n2 = _rmsnorm5(g, r1, P["g2"], "n2")
    if upto == "n2":
        return finish_layer(g, x, n2, ch, "n2")
    wg, swg, bqg, s_gate, _ = q_fc_tables(n2.c, n2.s, P["Wg"], np.zeros(INTER))
    gate = g.q_fc(n2, wg, swg, bqg, s_gate, INTER, "gate")
    wu, swu, bqu, s_up, _ = q_fc_tables(n2.c, n2.s, P["Wu"], np.zeros(INTER))
    up = g.q_fc(n2, wu, swu, bqu, s_up, INTER, "up")
    sig = g.q_logistic(gate, "sig")
    s1 = g.q_mul(gate, sig, pick_scale(gate.real * sig.real), "silu")
    y = g.q_mul(s1, up, pick_scale(s1.real * up.real), "mlp_y")
    if upto == "mlp_y":
        return finish_layer(g, x, y, ch, "mlp_y")
    wd, swd, bqd, s_down, _ = q_fc_tables(y.c, y.s, P["Wd"], np.zeros(D))
    down = g.q_fc(y, wd, swd, bqd, s_down, D, "down")
    if upto == "down":
        return finish_layer(g, x, down, ch, "down")
    out = g.q_addsub(down, r1, pick_scale(ch["out"]), "out")
    return finish_layer(g, x, out, ch, "out")


def finish_layer(g, x, t, ch, stage):
    f16_out = isinstance(t, TF)
    blob = g.finish([x.i], [t.i], "decoder layer qwen2.5-0.5b T=%d %s (int%d W / int8 A)"
                     % (T, FORM.upper(), W_BITS))
    if f16_out:
        shape, codes, s_o = list(t.v.shape), np.asarray(t.v, np.float16), None
        otype = "float16"
    else:
        shape, codes, s_o = list(t.c.shape), t.c, t.s
        otype = a_name()
    meta = dict(op="DECODER_LAYER", form=FORM, stage=stage,
                shape=dict(T=T, D=D, H=H, KV=KV, HD=HD, I=INTER),
                dtype=a_name(), weights_dtype="int%d" % W_BITS,
                input={"name": "x", "shape": [T, D], "type": a_name()},
                output={"name": "out", "shape": shape, "type": otype},
                quant={"input": x.s, "output": s_o}, int8_tol=2,
                tolerance_rationale=LAYER_TOL)
    return blob, meta, x.c, codes, dict(fp32=ch[stage].astype(np.float64))


LAYER_TOL = ("Register-before-run, before the first board run of this graph. Code level: "
             "(unchanged from ticket 04 for both weight tiers: this layer contains the same "
             "int8 RMSNorm chain and the same fp16<->int8 seam either way) "
             "int8_tol=2 because the layer contains the int8 RMSNorm chain (its RSQRT is a "
             "fixed-point LUT: docs/OP-rmsnorm.md 2.3 registers 4 codes for that chain, and "
             "the component run here measured max 2 codes on 6% of the values) plus one "
             "fp16<->int8 seam. Quality level: ADR 0005 cosine >= 0.9999 -- registered as "
             "the criterion, NOT adjusted afterwards.")


# --------------------------------------------------------------- device-side diag
REF_KEY = {"fc_proj": "q_proj_pre_rope", "rmsnorm": "n1", "swiglu": "down",
           "bmm_qk": "scores", "bmm_pv": "a3"}


def fp32_ref(key, P):
    ch = chain_f32(P)
    if key == "q_proj_pre_rope":
        return (ch["n1"] @ np.asarray(P["Wq"], np.float64).T + np.asarray(P["bq"], np.float64))
    if key == "a3":
        return np.transpose(ch["probs"], (0, 1, 2)) @ np.transpose(ch["v14"], (1, 0, 2))
    return ch[key]


def diag(opdir):
    """Compare a board dump against the INDEPENDENT float64 chain.

    This is the accuracy half of the two-layer criterion (ADR 0005); the code-level
    device-vs-golden half is tools/compare.py's job.
    """
    meta = json.load(open(os.path.join(opdir, "model.json")))
    dev = np.fromfile(os.path.join(opdir, "out_device.bin"),
                      {"int8": np.int8, "int16": np.int16}[meta["output"]["type"]])
    shape = tuple(meta["output"]["shape"])
    dev = dev.reshape(shape)
    s = float(meta["quant"]["output"])
    key = meta.get("fp32_ref") or REF_KEY.get(opdir.rstrip("/").replace("\\", "/").split("/")[-1].split("_")[-1], "out")
    ref = fp32_ref(key, fixture())
    dq = dev.astype(np.float64) * s
    d = dq - ref.reshape(shape)
    cos = cosine(dq, ref)
    rel = float(np.linalg.norm(d) / max(np.linalg.norm(ref), 1e-30))
    print("op            : %s stage=%s" % (meta["op"], meta.get("stage", "-")))
    print("fp32 ref      : %s  max|ref| %.4g" % (key, np.abs(ref).max()))
    print("1 code (real) : %.4e" % s)
    print("max_abs       : %.4e  = %.1f codes" % (np.abs(d).max(), np.abs(d).max() / s))
    print("rel_l2        : %.4e" % rel)
    print("cosine        : %.7f   (ADR 0005 operator criterion: >= 0.9999)"
          % cos)
    return 0


def build_probe_dequant(P):
    """Does DEQUANTIZE int8->fp16 actually apply the declared scale on MDLA?"""
    rng = np.random.default_rng(13)
    a = rng.integers(-120, 120, size=(T, 64)).astype(np.int8)
    s = 1.0 / 32.0
    g = QL()
    ta = g.mk_in("input", (T, 64), a, s)
    tf = g.f16_dequant(ta, "out")
    blob = g.finish([ta.i], [tf.i], "probe int8->fp16 dequantize")
    meta = dict(op="DEQUANTIZE", form="probe", shape={"T": T, "K": 64}, dtype="int8",
                input={"name": "input", "shape": [T, 64], "type": "int8"},
                output={"name": "out", "shape": [T, 64], "type": "float16"},
                quant={"input": s, "output": None}, int8_tol=2,
                tolerance_rationale=tol_text())
    return blob, meta, a, np.asarray(tf.v, np.float16), dict(fp32=a.astype(np.float64) * s)


def build_probe_dequant_add(P):
    """DEQUANTIZE -> fp16 ADD of a constant: is the first operand kept?"""
    rng = np.random.default_rng(17)
    a = rng.integers(-120, 120, size=(T, 64)).astype(np.int8)
    s = 1.0 / 32.0
    c = np.full((T, 64), -4096.0, np.float16)
    c[:, :2] = 0.0
    g = QL()
    ta = g.mk_in("input", (T, 64), a, s)
    tf = g.f16_dequant(ta, "deq")
    tadd = g.f16_add(tf, g.const_f16("cmask", c), "out", c)
    blob = g.finish([ta.i], [tadd.i], "probe dequantize + fp16 add const")
    meta = dict(op="ADD", form="probe", shape={"T": T, "K": 64}, dtype="int8",
                input={"name": "input", "shape": [T, 64], "type": "int8"},
                output={"name": "out", "shape": [T, 64], "type": "float16"},
                quant={"input": s, "output": None}, int8_tol=2,
                tolerance_rationale=tol_text())
    return blob, meta, a, np.asarray(tadd.v, np.float16), dict(fp32=a.astype(np.float64) * s + c.astype(np.float64))


def _seam_probe(s_out):
    P = fixture()
    ch = chain_f32(P)
    mask = causal_mask()
    c8, s8_ = s8(ch["scores"])
    g = QL()
    ta = g.mk_in("scores", (H, T, T), c8, s8_)
    sc = g.q_requant(ta, s_out, "scaled", factor=SCALE)
    f16 = g.f16_dequant(sc, "deq")
    tmask = g.const_f16("mask", mask)
    masked = g.f16_add(f16, tmask, "masked", mask)
    p = g.f16_softmax(masked, "out")
    blob = g.finish([ta.i], [p.i], "probe int8 score -> requant -> fp16 mask+softmax")
    meta = dict(op="SEAM", form="probe", shape={"H": H, "T": T}, dtype="int8",
                input={"name": "scores", "shape": [H, T, T], "type": "int8"},
                output={"name": "out", "shape": [H, T, T], "type": "float16"},
                quant={"input": s8_, "output": None, "scaled_scale": s_out},
                int8_tol=2, tolerance_rationale=tol_text())
    return blob, meta, c8, np.asarray(p.v, np.float16), dict(fp32=softmax_last(sc.real + mask))


def build_probe_seam(P):
    return _seam_probe(pick_scale(chain_f32(P)["scores"] * SCALE))


def build_probe_seam_alt(P):
    return _seam_probe(5.0)


def _seam_noreq(shape2d):
    P = fixture()
    ch = chain_f32(P)
    mask = causal_mask()
    s_out = pick_scale(ch["scores"] * SCALE)
    codes = np.clip(np.round(ch["scores"] * SCALE / s_out), -127, 127).astype(np.int8)
    g = QL()
    if shape2d:
        mk = mask[0]
        ta = g.mk_in("scores", (T, T), codes[0], s_out)
        f16 = g.f16_dequant(ta, "deq")
        tmask = g.const_f16("mask", mk)
        masked = g.f16_add(f16, tmask, "masked", mk)
        p = g.f16_softmax(masked, "out")
        ref = softmax_last(codes[0].astype(np.float64) * s_out + mk)
        shape = (T, T)
    else:
        ta = g.mk_in("scores", (H, T, T), codes, s_out)
        f16 = g.f16_dequant(ta, "deq")
        tmask = g.const_f16("mask", mask)
        masked = g.f16_add(f16, tmask, "masked", mask)
        p = g.f16_softmax(masked, "out")
        ref = softmax_last(codes.astype(np.float64) * s_out + mask)
        shape = (H, T, T)
    blob = g.finish([ta.i], [p.i], "probe dequant(no requant) -> fp16 mask+softmax")
    meta = dict(op="SEAM2", form="probe", shape={"shape": list(shape)}, dtype="int8",
                input={"name": "scores", "shape": list(shape), "type": "int8"},
                output={"name": "out", "shape": list(shape), "type": "float16"},
                quant={"input": s_out, "output": None}, int8_tol=2,
                tolerance_rationale=tol_text())
    return blob, meta, codes[0] if shape2d else codes, np.asarray(p.v, np.float16), dict(fp32=ref)


def build_probe_seam_noreq(P):
    return _seam_noreq(False)


def build_probe_seam_noreq_2d(P):
    return _seam_noreq(True)


def build_probe_pv_t8(P):
    """The 16-byte row rule with int8 activations: P*V at T=8 has [H,8,8] rows = 8 B.

    Expected to FAIL compilation ("Bridging is disallowed but gap is detected between
    Input and MDLA_5_0"); T=16 (16 B rows) compiles.  Same failure as ops/bmm_pv_t8_int8,
    reproduced here with this ticket's builder and shape.
    """
    rng = np.random.default_rng(21)
    a = rng.integers(-120, 120, size=(H, 8, 8)).astype(np.int8)
    b = rng.integers(-120, 120, size=(H, 8, HD)).astype(np.int8)
    s_a, s_b, s_o = 1.0 / 127, 1.0 / 127, 1.0 / 127 * 127 / 127
    g = QL()
    ta = g.mk_in("a", (H, 8, 8), a, s_a)
    tb = g.mk_in("b", (H, 8, HD), b, s_b)
    o = g.q_bmm(ta, tb, 1.0 / 127.0, "attn")
    blob = g.finish([ta.i, tb.i], [o.i], "probe P*V T=8 int8 (8 B rows)")
    meta = dict(op="BATCH_MATMUL", form="probe", matmul="P.V", shape={"H": H, "T": 8, "HD": HD},
                dtype="int8",
                inputs=[{"name": "a", "shape": [H, 8, 8], "type": "int8"},
                        {"name": "b", "shape": [H, 8, HD], "type": "int8"}],
                output={"name": "attn", "shape": [H, 8, HD], "type": "int8"},
                quant={"a": s_a, "b": s_b, "output": 1.0 / 127.0}, int8_tol=1,
                tolerance_rationale=tol_text(), expected="COMPILE_FAIL (16 B row rule)")
    return blob, meta, a, o.c, dict(second=b)



# ------------------------------------------------------- int16 acceptance probes
# Ticket 04d: before building a whole 4W16A layer, which operators does ncc accept
# with int16 activations, and does the device compute them the way the quantised
# golden says?  Each probe is a small graph with a real (non-trivial) golden.
def _need16():
    if A_BITS != 16:
        raise SystemExit("p16_* probes need --a-bits 16")


def _pmeta(op, xname, xshape, yshape, s_in, s_out, note="", tol=1, ytype="int16",
           extra=None):
    m = dict(op=op, form="probe16", shape={"in": list(xshape), "out": list(yshape)},
             dtype="int16", input={"name": xname, "shape": list(xshape), "type": "int16"},
             output={"name": "out", "shape": list(yshape), "type": ytype},
             quant={"input": s_in, "output": s_out}, int8_tol=tol,
             tolerance_rationale=TOL16, note=note)
    if extra:
        m.update(extra)
    return m


def build_p16_fc(P):
    """FULLY_CONNECTED with int16 activations, per-channel int4/int8 weights, int32 bias."""
    _need16()
    rng = np.random.default_rng(31)
    M, K, N = T, 64, 64
    a_i, s_a = s16(rng.standard_normal((M, K)) * 3.0)
    W = rng.standard_normal((N, K)) * 0.05
    b = rng.standard_normal(N) * 0.1
    wq, s_w = sw_weights(W)
    bs = s_a * s_w.astype(np.float64)
    b_q = np.round(b / bs).astype(np.int32)
    o_ref = (a_i.astype(np.float64) * s_a) @ (wq.astype(np.float64) * s_w[:, None]).T + b
    s_o = pick_scale(o_ref)
    g = QL()
    ta = g.mk_in("input", (M, K), a_i, s_a)
    out = g.q_fc(ta, wq, s_w, b_q, s_o, N, "out", version=9)
    blob = g.finish([ta.i], [out.i], "probe int16 FC v9 int%d W" % W_BITS)
    return blob, _pmeta("FULLY_CONNECTED", "input", (M, K), (M, N), s_a, s_o,
                        note="op version 9, bias int32 per-channel"), a_i, out.c, \
        dict(fp32=o_ref)


def build_p16_bmm_pv(P):
    """BATCH_MATMUL P*V shape with int16 activations: [2,16,16] x [2,16,64]."""
    _need16()
    rng = np.random.default_rng(32)
    a_i, s_a = s16(rng.random((2, T, T)))
    b_i, s_b = s16(rng.standard_normal((2, T, HD)))
    s_o = pick_scale(np.einsum("hij,hjk->hik", a_i.astype(np.float64) * s_a,
                               b_i.astype(np.float64) * s_b))
    g = QL()
    ta = g.mk_in("a", (2, T, T), a_i, s_a)
    tb = g.mk_in("b", (2, T, HD), b_i, s_b)
    out = g.q_bmm(ta, tb, s_o, "out")
    blob = g.finish([ta.i, tb.i], [out.i], "probe int16 BMM P*V")
    return blob, _pmeta("BATCH_MATMUL", "a", (2, T, T), (2, T, HD), s_a, s_o,
                        note="P*V shape", extra=dict(second=dict(name="b",
                        shape=[2, T, HD], scale=s_b))), a_i, out.c, \
        dict(second=b_i, fp32=None)


def build_p16_bmm_qk(P):
    """BATCH_MATMUL Q*K^T shape with int16 activations: [2,16,64] x [2,16,64] adj_y."""
    _need16()
    rng = np.random.default_rng(33)
    a_i, s_a = s16(rng.standard_normal((2, T, HD)))
    b_i, s_b = s16(rng.standard_normal((2, T, HD)))
    prod = np.einsum("hid,hjd->hij", a_i.astype(np.float64) * s_a,
                     b_i.astype(np.float64) * s_b)
    s_o = pick_scale(prod)
    g = QL()
    ta = g.mk_in("a", (2, T, HD), a_i, s_a)
    tb = g.mk_in("b", (2, T, HD), b_i, s_b)
    out = g.q_bmm(ta, tb, s_o, "out", adj_y=True)
    blob = g.finish([ta.i, tb.i], [out.i], "probe int16 BMM Q*K^T adj_y")
    return blob, _pmeta("BATCH_MATMUL", "a", (2, T, HD), (2, T, T), s_a, s_o,
                        note="Q*K^T shape, adj_y", extra=dict(second=dict(name="b",
                        shape=[2, T, HD], scale=s_b))), a_i, out.c, \
        dict(second=b_i, fp32=None)


def build_p16_mul(P):
    """MUL of two int16 tensors into int16 (RMSNorm / RoPE / SwiGLU elementwise)."""
    _need16()
    rng = np.random.default_rng(34)
    a_i, s_a = s16(rng.standard_normal((T, 64)))
    b_i, s_b = s16(rng.standard_normal((T, 64)))
    real = a_i.astype(np.float64) * s_a * (b_i.astype(np.float64) * s_b)
    g = QL()
    ta = g.mk_in("a", (T, 64), a_i, s_a)
    tb = g.mk_in("b", (T, 64), b_i, s_b)
    out = g.q_mul(ta, tb, pick_scale(real), "out")
    blob = g.finish([ta.i, tb.i], [out.i], "probe int16 MUL")
    return blob, _pmeta("MUL", "a", (T, 64), (T, 64), s_a, out.s,
                        extra=dict(second=dict(name="b", shape=[T, 64], scale=s_b))), \
        a_i, out.c, dict(second=b_i, fp32=real)


def _p16_addsub(P, minus):
    _need16()
    rng = np.random.default_rng(35 if minus else 36)
    a_i, s_a = s16(rng.standard_normal((T, 64)))
    b_i, s_b = s16(rng.standard_normal((T, 64)) * 2.0)      # deliberately != s_a
    real = (a_i.astype(np.float64) * s_a - b_i.astype(np.float64) * s_b) if minus else \
           (a_i.astype(np.float64) * s_a + b_i.astype(np.float64) * s_b)
    g = QL()
    ta = g.mk_in("a", (T, 64), a_i, s_a)
    tb = g.mk_in("b", (T, 64), b_i, s_b)
    out = g.q_addsub(ta, tb, pick_scale(real), "out", minus=minus)
    blob = g.finish([ta.i, tb.i], [out.i], "probe int16 %s (different scales)"
                    % ("SUB" if minus else "ADD"))
    return blob, _pmeta("SUB" if minus else "ADD", "a", (T, 64), (T, 64), s_a, out.s,
                        note="operands have different per-tensor scales",
                        extra=dict(second=dict(name="b", shape=[T, 64], scale=s_b))), \
        a_i, out.c, dict(second=b_i, fp32=real)


def build_p16_add(P):
    return _p16_addsub(P, False)


def build_p16_sub(P):
    return _p16_addsub(P, True)


def build_p16_mean(P):
    """MEAN(axis=1) of int16 (RMSNorm chain: the second layer)."""
    _need16()
    rng = np.random.default_rng(37)
    a_i, s_a = s16(rng.random((T, 64)))
    real = (a_i.astype(np.float64) * s_a).mean(axis=1, keepdims=True)
    g = QL()
    ta = g.mk_in("input", (T, 64), a_i, s_a)
    out = g.q_mean(ta, pick_scale(real), "out")
    blob = g.finish([ta.i], [out.i], "probe int16 MEAN axis=1")
    return blob, _pmeta("MEAN", "input", (T, 64), (T, 1), s_a, out.s), a_i, out.c, \
        dict(fp32=real)


def build_p16_rsqrt(P):
    """RSQRT of int16 (RMSNorm chain: the third layer)."""
    _need16()
    rng = np.random.default_rng(38)
    a_i, s_a = s16(rng.random((T, 64)) + 0.05)
    real = 1.0 / np.sqrt(a_i.astype(np.float64) * s_a)
    g = QL()
    ta = g.mk_in("input", (T, 64), a_i, s_a)
    out = g.q_rsqrt(ta, pick_scale(real), "out")
    blob = g.finish([ta.i], [out.i], "probe int16 RSQRT")
    return blob, _pmeta("RSQRT", "input", (T, 64), (T, 64), s_a, out.s), a_i, out.c, \
        dict(fp32=real)


def build_p16_logistic(P):
    """LOGISTIC of int16 with the int16 fixed-point output convention (1/65536, zp -32768)."""
    _need16()
    rng = np.random.default_rng(39)
    a_i, s_a = s16(rng.standard_normal((T, 64)) * 4.0)
    real = 1.0 / (1.0 + np.exp(-(a_i.astype(np.float64) * s_a)))
    g = QL()
    ta = g.mk_in("input", (T, 64), a_i, s_a)
    out = g.q_logistic(ta, "out")
    blob = g.finish([ta.i], [out.i], "probe int16 LOGISTIC")
    return blob, _pmeta("LOGISTIC", "input", (T, 64), (T, 64), s_a, out.s,
                        note="output scale 1/65536 zp -32768"), a_i, out.c, \
        dict(fp32=real)


def _p16_native_seam(sym):
    """The native int16 seam exactly as the layer builds it (mask -> x1/8 -> softmax)."""
    P = fixture()
    ch = chain_f32(P)
    c16, s_s = s16(ch["scores"])
    g = QL()
    ta = g.mk_in("scores", (H, T, T), c16, s_s)
    p, p_sym = _seam_native(g, ta, sym)
    blob = g.finish([ta.i], [p_sym.i], "probe int16 native seam (%s)"
                    % ("sym" if sym else "tflite"))
    return blob, _pmeta("SEAM_NATIVE", "scores", (H, T, T), (H, T, T), s_s, p_sym.s,
                        note="ADD(mask -4096 in raw domain) -> MUL(1/8) -> SOFTMAX int16 "
                             + ("(1/32767 zp 0)" if sym else "(1/65536 zp -32768) -> x1.0")),         c16, p_sym.c, dict(fp32=ch["probs"])


def build_p16_native_seam(P):
    return _p16_native_seam(True)


def build_p16_native_seam_tflite(P):
    return _p16_native_seam(False)


def build_p16_shapeops(P):
    """RESHAPE / SPLIT / CONCAT / TRANSPOSE with int16 activations."""
    _need16()
    rng = np.random.default_rng(41)
    a_i, s_a = s16(rng.standard_normal((T, 64)))
    g = QL()
    ta = g.mk_in("input", (T, 64), a_i, s_a)
    y = g.q_shape(ta, "r3d", (T, 2, 32))
    p0, p1 = g.q_split(y, "sp", 2, 2, (T, 2, 16))
    cc = g.q_concat([p0, p1], "cc", 2, (T, 2, 32))
    tt = g.q_transpose(cc, "tt", [1, 0, 2])
    out = g.q_shape(tt, "out", (2, T, 32))
    blob = g.finish([ta.i], [out.i], "probe int16 reshape/split/concat/transpose")
    return blob, _pmeta("SHAPEOPS", "input", (T, 64), (2, T, 32), s_a, s_a,
                        note="RESHAPE->SPLIT->CONCAT->TRANSPOSE->RESHAPE"), a_i, out.c, \
        dict(fp32=None)


def build_p16_seam(P):
    """The ticket 04 fp16 seam, dtype-parameterised: int16 -> fp16 -> ... -> int16."""
    _need16()
    P_ = fixture()
    ch = chain_f32(P_)
    mask = causal_mask()
    c16, s_s = s16(ch["scores"])
    g = QL()
    ta = g.mk_in("scores", (H, T, T), c16, s_s)
    f = g.f16_dequant(ta, "deq")
    f = g.f16_mul_scalar(f, SCALE, "scaled")
    tm = g.const_f16("mask", mask)
    f = g.f16_add(f, tm, "masked", mask)
    p = g.f16_softmax(f, "probs")
    out = g.f16_quantize(p, 1.0 / 32767.0, "out")
    blob = g.finish([ta.i], [out.i], "probe int16 -> fp16 softmax -> int16")
    return blob, _pmeta("SEAM16", "scores", (H, T, T), (H, T, T), s_s, out.s,
                        note="DEQUANTIZE->MUL->ADD->SOFTMAX->QUANTIZE (int16 both ends)"), \
        c16, out.c, dict(fp32=None)


def build_p16_dequant(P):
    """DEQUANTIZE int16 -> fp16 alone (first layer of the 04 seam)."""
    _need16()
    rng = np.random.default_rng(43)
    a_i, s_a = s16(rng.standard_normal((T, 64)) * 100.0)
    g = QL()
    ta = g.mk_in("input", (T, 64), a_i, s_a)
    f = g.f16_dequant(ta, "out")
    blob = g.finish([ta.i], [f.i], "probe int16 -> fp16 DEQUANTIZE")
    return blob, _pmeta("DEQUANTIZE", "input", (T, 64), (T, 64), s_a, None,
                        ytype="float16"), a_i, np.asarray(f.v, np.float16), \
        dict(fp32=a_i.astype(np.float64) * s_a)


def build_p16_mul_one(P):
    """MUL by the constant 1.0 with a new output scale: the symmetric rescale of probs."""
    _need16()
    rng = np.random.default_rng(44)
    p_real = rng.random((T, 64))
    c_i = np.clip(np.round(p_real * 65536.0) - 32768, -32768, 32767).astype(np.int16)
    s_p = 1.0 / 65536.0
    g = QL()
    ta = g.mk_in("p", (T, 64), c_i, s_p, -32768)
    one = np.full((T, 64), 32767, np.int16)
    t1 = TS(g.const_i8("one", one, 1.0 / 32767.0), one, 1.0 / 32767.0)
    out = g.q_mul(ta, t1, 1.0 / 32767.0, "out")
    blob = g.finish([ta.i], [out.i], "probe int16 MUL by 1.0 (rescale to sym domain)")
    return blob, _pmeta("MUL", "p", (T, 64), (T, 64), s_p, out.s,
                        note="1/65536 zp -32768 -> 1/32767 zp 0"), c_i, out.c, \
        dict(fp32=p_real)



SPECS = {"fc_proj": build_fc_proj, "rmsnorm": build_rmsnorm, "swiglu": build_swiglu,
         "bmm_qk": build_bmm_qk, "bmm_pv": build_bmm_pv, "layer": build_layer,
         "probe_add_scales": build_probe_add_scales, "probe_quantize": build_probe_quantize,
         "probe_dequant": build_probe_dequant, "probe_dequant_add": build_probe_dequant_add,
         "probe_seam": build_probe_seam, "probe_seam_alt": build_probe_seam_alt,
         "probe_seam_noreq": build_probe_seam_noreq,
         "probe_seam_noreq_2d": build_probe_seam_noreq_2d,
         "probe_pv_t8": build_probe_pv_t8,
         # ticket 04d: int16 activation acceptance probes
         "p16_fc": build_p16_fc, "p16_bmm_pv": build_p16_bmm_pv,
         "p16_bmm_qk": build_p16_bmm_qk, "p16_mul": build_p16_mul,
         "p16_add": build_p16_add, "p16_sub": build_p16_sub,
         "p16_mean": build_p16_mean, "p16_rsqrt": build_p16_rsqrt,
         "p16_logistic": build_p16_logistic,
         "p16_softmax": build_p16_native_seam,
         "p16_native_seam": build_p16_native_seam,
         "p16_native_seam_tflite": build_p16_native_seam_tflite,
         "p16_shapeops": build_p16_shapeops, "p16_seam": build_p16_seam,
         "p16_dequant": build_p16_dequant, "p16_mul_one": build_p16_mul_one}


def wq_chain():
    """Weights quantised at the active tier, activations EXACT (float64 chain).

    The attribution half of ticket 04c: chain_f32() is the independent reference,
    so running it on dequantised weights isolates the weight error from the int8
    activation error without touching the graph.  Compare with the layer golden's
    cosine (which carries both) to see which side the remaining error sits on.
    """
    P = fixture()
    Pq = dict(P)
    for name in ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd"):
        Wf = np.asarray(P[name], np.float64)
        q, s = sw_weights(Wf)
        Pq[name] = q.astype(np.float64) * s[:, None].astype(np.float64)
    for bits in (4, 8):
        set_tier(bits, 8)
        Pq = dict(P)
        for name in ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd"):
            Wf = np.asarray(P[name], np.float64)
            q, s = sw_weights(Wf, bits)
            Pq[name] = q.astype(np.float64) * s[:, None].astype(np.float64)
        exact = chain_f32(P)
        wq = chain_f32(Pq)
        print("weights-only int%d: cosine(out)=%.7f  rel_l2(out)=%.4f%%  "
              "cosine(down)=%.7f"
              % (bits, cosine(wq["out"], exact["out"]),
                 100.0 * float(np.linalg.norm(wq["out"] - exact["out"])
                               / np.linalg.norm(exact["out"])),
                 cosine(wq["down"], exact["down"])))
    return 0


def weight_error_table():
    """Per-matrix weight quantisation error, int4 vs int8 (ticket 04c item 3).

    Same per-output-channel symmetric scheme in both rows; the only difference is
    the code range (-8..7 vs -127..127), i.e. exactly what --w-bits switches.
    """
    P = fixture()
    mats = [("Wq", P["Wq"]), ("Wk", P["Wk"]), ("Wv", P["Wv"]), ("Wo", P["Wo"]),
            ("Wg", P["Wg"]), ("Wu", P["Wu"]), ("Wd", P["Wd"])]
    print("matrix      bits  codes      rel_l2      cosine    max_abs   params")
    for bits in (4, 8):
        for name, Wm in mats:
            Wf = np.asarray(Wm, np.float64)
            q, s = sw_weights(Wf, bits)
            dq = q.astype(np.float64) * s[:, None].astype(np.float64)
            rel = float(np.linalg.norm(dq - Wf) / np.linalg.norm(Wf))
            print("%-10s  %2d  %d..%d  %8.4f%%  %9.7f  %8.4f  %d"
                  % (name, bits, -wq_max(bits), wq_max(bits), 100.0 * rel,
                     cosine(dq, Wf), float(np.abs(dq - Wf).max()), Wf.size))
    ch = chain_f32(P)
    a8, s_a = s8(ch["n1"])
    print("(activation n1, per-tensor int8: rel_l2 %.4f%%)"
          % (100.0 * float(np.linalg.norm(a8 * s_a - ch["n1"]) / np.linalg.norm(ch["n1"]))))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--upto", default="out", choices=STAGES)
    ap.add_argument("--w-bits", type=int, default=4, choices=(4, 8),
                    help="weight precision tier: 4 = ticket 04 (int4 packed), "
                         "8 = ticket 04c (int8 per-channel)")
    ap.add_argument("--a-bits", type=int, default=8, choices=(8, 16),
                    help="activation precision tier: 8 = tickets 04/04c (int8 "
                         "per-tensor), 16 = ticket 04d (int16 per-tensor)")
    ap.add_argument("--seam", default="f16", choices=("f16", "native", "native_sym"),
                    help="how the LAYER crosses the masked softmax: f16 = the ticket 04 "
                         "fp16 seam; native/native_sym = int16 masked SOFTMAX")
    ap.add_argument("--weights-report", action="store_true",
                    help="print the per-matrix weight quantisation error, both tiers")
    ap.add_argument("--wq-chain", action="store_true",
                    help="cosine of the fp64 chain run on quantised weights only")
    ap.add_argument("--diag", action="store_true", help="compare out_device.bin to the fp32 chain")
    a = ap.parse_args()
    global SEAM
    SEAM = a.seam
    set_tier(a.w_bits, a.a_bits)
    if a.weights_report:
        return weight_error_table()
    if a.wq_chain:
        return wq_chain()
    if a.diag:
        return diag(a.out or ("ops/%s_" % FORM + (a.spec or "layer")))
    if a.list or not a.spec:
        print(" ".join(SPECS))
        return 0
    if a.spec not in SPECS:
        raise SystemExit("unknown spec %s" % a.spec)
    fn = SPECS[a.spec]
    blob, meta, x, y, extra = fn(fixture(), a.upto) if a.spec == "layer" else fn(fixture())
    d = a.out or ("ops/%s_" % FORM + a.spec)
    os.makedirs(os.path.join(d, "golden"), exist_ok=True)
    open(os.path.join(d, "model.tflite"), "wb").write(blob)
    x.tofile(os.path.join(d, "golden", "input.bin"))
    y.tofile(os.path.join(d, "golden", "expected.bin"))
    if extra.get("second") is not None:
        extra["second"].tofile(os.path.join(d, "golden", "input1.bin"))
    meta["tflite"] = "model.tflite"
    meta["tflite_bytes"] = len(blob)
    meta["precision_tier"] = dict(
        weights="int%d, symmetric per-output-channel" % W_BITS,
        activations=("int%d, symmetric per-tensor (s = max|v|/%d)"
                     % (A_BITS, aq_max())),
        bias="int32 per-channel (s[n] = s_a * s_w[n])",
        requant="tools/quant_contract.py: two integer roundings, per-channel multipliers",
        builder="tools/build_w4a8.py --w-bits %d --a-bits %d --seam %s"
                % (W_BITS, A_BITS, SEAM))
    if a.spec == "layer":
        meta["fp32_ref"] = STAGE_KEY[a.upto]
    elif a.spec in REF_KEY:
        meta["fp32_ref"] = REF_KEY[a.spec]
    meta["quality_criterion"] = dict(
        cosine_min=0.9999, source="ADR 0005 (operator-level)",
        code_criterion="device vs quantised golden: |delta| <= int8_tol codes")
    if extra.get("fp32") is not None:
        ref = np.asarray(extra["fp32"], np.float64)
        if meta["quant"]["output"] is None:
            dq = np.asarray(y, np.float64)
        else:
            dq = dequant(y, meta["quant"]["output"])
        rel = float(np.linalg.norm(dq - ref) / max(np.linalg.norm(ref), 1e-30))
        meta["accuracy_vs_fp32"] = dict(
            reference="independent float64 chain (tools/build_w4a8.py chain_f32)",
            cosine=cosine(dq, ref), rel_l2=rel,
            max_abs=float(np.abs(dq - ref).max()),
            max_abs_over_maxabs_out=float(np.abs(dq - ref).max() / max(np.abs(ref).max(), 1e-30)),
            out_scale=meta["quant"]["output"],
            one_code_real=(float(meta["quant"]["output"])
                           if meta["quant"]["output"] is not None else None),
            note="int8 quantisation error of the whole 4W8A path, no device involved")
        print("  vs fp32 chain: cosine=%.7f rel_l2=%.4e max_abs=%.4e"
              % (meta["accuracy_vs_fp32"]["cosine"], rel, meta["accuracy_vs_fp32"]["max_abs"]))
    json.dump(meta, open(os.path.join(d, "model.json"), "w"), indent=2)
    print("%s  %d B  in%s out%s" % (d, len(blob), x.shape, y.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
