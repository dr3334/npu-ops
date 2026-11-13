#!/usr/bin/env python3
"""Ticket 10 -- two Qwen2.5-0.5B decoder layers chained on the board, then the tied LM head.

Three independent graphs, each one its own ops/<name>/ directory (one .tflite, one .dla,
one compare.txt, one registered tolerance):

    ops/two_layer_l0_t8_fp16/      x[8,896]  --decoder layer 0 (layer0.* weights)--> h1[8,896]
    ops/two_layer_l1_t8_fp16/      h1[8,896] --decoder layer 1 (layer1.* weights)--> h2[8,896]
    ops/two_layer_logits_t8_fp16/  h2[8,896] --tied LM head (4xFC + CONCAT)-------> logits[8,151936]

WHY three graphs and not one
---------------------------
A .dla does not expose intermediate tensors, so "which layer went wrong" is only answerable
if every layer boundary is a graph boundary (spec, Implementation Decisions: the whole spec
uses one seam -- ops/<name>/ + one e2e run).  The chain therefore lives on the *board*:
tools/chain_two_layer.sh copies layer 0's out_device.bin over layer 1's golden/input.bin and
layer 1's out_device.bin over the logits graph's golden/input.bin, so the second layer sees
what the first layer actually produced, not what numpy thinks it should have produced
(NOTES 10.4 item 5: "板端输出喂下一层" is done by overwriting golden/input.bin).

WHY the second/third graph is still compared against the *CPU chain*
--------------------------------------------------------------------
Each graph's registered golden/expected.bin is the CPU chain value (h1_cpu, h2_cpu,
logits_cpu) computed from the same frozen fixture and the same sin/cos tables.  Once the
board has handed over its own h1, the difference "device h2 vs h2_cpu" contains two things:
propagation (device h1 != cpu h1) and this layer's own error.  --diag therefore also computes
a second reference that starts from the *actual stored input* (i.e. the board's h1), which
isolates the layer's own error.  Both numbers go into docs/OP-two-layer.md.

Tolerance is registered in model.json BEFORE the first board run (NOTES 4 / 10.1):

    floor = max(floor_self, floor_chain)          # device-independent, numpy only
    atol = rtol = max(4 * 2^-11 * max|out|, 4 * floor)

    floor_self  : this layer's own fp16-rounding floor (numpy chain with a fp16 rounding at
                  every tensor boundary vs the single-rounding golden -- same construction
                  ticket 09 used to justify its 9.03e-3)
    floor_chain : how much this layer's CPU reference moves when it is fed the *fp16-rounded
                  previous-layer output* instead of the exact one.  That is the size of the
                  propagation effect the board handoff necessarily introduces, estimated
                  without any device data.

top-1 / argmax is NOT a criterion (ADR 0003): under frozen random weights it carries no
semantics and one fp16 ULP flips near-ties.  It stays frozen until the real-weight stage.
"""
import argparse
import json
import os
import sys

import flatbuffers
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ops as bo          # flatbuffer toolbox (read-only use, AGENTS 8)
import qwen25_weights as qw     # frozen fixture loader (sha256-verified on load)

T = 8
D = 896
H = 14                  # query heads
KV = 2                  # KV heads
HD = 64                 # head_dim
HALF = HD // 2
INTER = 4864            # SwiGLU intermediate width
THETA = 1e6
POS0 = 0
EPS = 1e-6              # transformers rms_norm_eps
SCALE = 1.0 / 8.0       # 1/sqrt(64), exact in fp16
MASK = -4096.0          # 2^12; interval from docs/OP-masked-softmax.md 3.4
KVR = H // KV           # 7

VOCAB = 151936
CHUNKS = 4              # N > 65535 is refused by MDLA for FULLY_CONNECTED (spec C3 / ticket 08)
N_CHUNK = VOCAB // CHUNKS

TOKEN_IDS = [9707, 11, 1879, 828, 11780, 13, 220, 16]   # same list as tickets 03/06/09

FP16_ULP_REL = 2.0 ** -11
TOL_K = 4.0

GRAPHS = {
    "l0": {"op": "TWO_LAYER_L0_T8_FP16", "layer": 0,
           "out": "ops/two_layer_l0_t8_fp16", "output": "h1"},
    "l1": {"op": "TWO_LAYER_L1_T8_FP16", "layer": 1,
           "out": "ops/two_layer_l1_t8_fp16", "output": "h2"},
    "logits": {"op": "TWO_LAYER_LOGITS_T8_FP16", "layer": None,
               "out": "ops/two_layer_logits_t8_fp16", "output": "logits"},
}

STAGES = ["n1", "q3", "k14", "v14", "scores", "probs", "attn", "r1", "n2", "mlp_y", "down", "out"]


# ------------------------------------------------------------------ fixture data
def fixture(layer):
    g = lambda n: qw.get("layer%d.%s" % (layer, n))
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


def embed_weight():
    w = qw.get("embed.weight")
    if qw.get("lm_head.weight") is not w:
        raise SystemExit("FIXTURE BUG: lm_head.weight is not the same object as embed.weight")
    if w.dtype != np.float16 or list(w.shape) != [VOCAB, D]:
        raise SystemExit("embed.weight %s %s != [%d,%d] fp16" % (w.shape, w.dtype, VOCAB, D))
    return w


def input_hidden():
    """Layer-0 input: the raw embed rows for the fixed token ids, fp16 (ticket 09's boundary)."""
    emb = qw.get("embed.weight")
    return np.ascontiguousarray(emb[np.asarray(TOKEN_IDS)].astype(np.float16))


def tables(heads, p0=POS0):
    inv = THETA ** (-2.0 * np.arange(HALF) / HD)
    ang = np.outer(p0 + np.arange(T), inv)
    c = np.broadcast_to(np.cos(ang)[:, None, :], (T, heads, HALF))
    s = np.broadcast_to(np.sin(ang)[:, None, :], (T, heads, HALF))
    return (np.ascontiguousarray(c).astype(np.float16),
            np.ascontiguousarray(s).astype(np.float16))


def causal_mask():
    m = np.zeros((T, T), np.float16)
    m[np.triu_indices(T, k=1)] = np.float16(MASK)
    return np.ascontiguousarray(np.broadcast_to(m, (H, T, T)).copy())


# ------------------------------------------------------------------------ maths
def _rmsnorm(x, g, store):
    x = store(x)
    sq = store(x * x)
    mean = store(sq.mean(axis=-1, keepdims=True))
    mean_eps = store(mean + store(np.full_like(mean, EPS)))
    rsq = store(1.0 / np.sqrt(mean_eps))
    scaled = store(x * rsq)
    return store(scaled * store(g.astype(x.dtype)))


def _rope(y, c16, s16, store):
    dt = y.dtype
    y3 = store(y).reshape(T, -1, HD)
    a, b = y3[..., :HALF], y3[..., HALF:]
    C = np.repeat(c16.astype(dt)[:, :1, :], y3.shape[1], axis=1)
    S = np.repeat(s16.astype(dt)[:, :1, :], y3.shape[1], axis=1)
    lo = store(store(a * C) - store(b * S))
    hi = store(store(b * C) + store(a * S))
    return store(np.concatenate([lo, hi], axis=-1)).reshape(T, y3.shape[1], HD)


def _softmax(v, store):
    m = store(v - store(v.max(axis=-1, keepdims=True)))
    e = store(np.exp(m.astype(np.float64)).astype(v.dtype))
    return store(e / store(e.sum(axis=-1, keepdims=True)))


def forward(x16, P, dtype=np.float32, round_fp16=False, stage="out"):
    """One decoder layer in numpy -- no flatbuffer, no ncc, no device (AGENTS R3).

    The maths is identical to ticket 09's forward() (same conventions: split-half RoPE,
    h//7 KV replication, mask after the 1/8 scaling, eps as a [T,1] constant), re-typed
    here so this ticket's golden and this ticket's flatbuffer writer share no code.
    """
    def st(a):
        a = np.asarray(a, dtype)
        return np.asarray(a, np.float16).astype(dtype) if round_fp16 else a

    x = st(x16.astype(dtype))
    out = {"x": x}
    n1 = _rmsnorm(x, P["g1"].astype(dtype), st)
    out["n1"] = n1
    if stage == "n1":
        return out

    def proj(W, b, heads):
        y = n1 @ W.astype(dtype).T + b.astype(dtype)
        return st(y).reshape(T, heads, HD)

    q3 = _rope(proj(P["Wq"], P["bq"], H), *tables(H), store=st)
    out["q3"] = q3
    k3 = _rope(st(n1 @ P["Wk"].astype(dtype).T + P["bk"].astype(dtype)).reshape(T, KV, HD),
               *tables(KV), store=st)
    v3 = st(st(n1 @ P["Wv"].astype(dtype).T + P["bv"].astype(dtype)).reshape(T, KV, HD))
    k14 = np.repeat(k3, KVR, axis=1)
    v14 = np.repeat(v3, KVR, axis=1)
    out["k14"], out["v14"] = k14, v14
    if stage in ("q3", "k14", "v14"):
        return out

    qt, kt, vt = (np.transpose(z, (1, 0, 2)) for z in (q3, k14, v14))
    scores = st(qt @ np.transpose(kt, (0, 2, 1)))
    out["scores"] = scores
    if stage == "scores":
        return out
    scaled = st(scores * np.asarray(SCALE, dtype))
    masked = st(scaled + causal_mask().astype(dtype))
    p = _softmax(masked, st)
    out["probs"] = p
    if stage == "probs":
        return out
    a3 = st(p @ vt)
    attn = st(np.transpose(a3, (1, 0, 2)).reshape(T, D))
    out["attn"] = attn
    if stage == "attn":
        return out
    o = st(attn @ P["Wo"].astype(dtype).T)                 # o_proj: no bias
    r1 = st(st(o) + st(x))
    out["r1"] = r1
    if stage == "r1":
        return out
    n2 = _rmsnorm(r1, P["g2"].astype(dtype), st)
    out["n2"] = n2
    if stage == "n2":
        return out
    gate = st(n2 @ P["Wg"].astype(dtype).T)
    up = st(n2 @ P["Wu"].astype(dtype).T)
    sig = st(1.0 / (1.0 + np.exp(-gate.astype(np.float64))).astype(dtype))
    y = st(st(st(gate * sig)) * up)
    out["mlp_y"] = y
    if stage == "mlp_y":
        return out
    down = st(y @ P["Wd"].astype(dtype).T)
    out["down"] = down
    if stage == "down":
        return out
    res = st(st(down) + st(r1))
    out["out"] = res
    return out


def logits_ref(h16, w_embed=None, chunk=N_CHUNK):
    """Tied LM head reference: fp16(h_f32 @ W_f32^T), computed in vocab chunks (RSS bound)."""
    w = embed_weight() if w_embed is None else w_embed
    hf = np.asarray(h16, np.float32)
    exp = np.empty((T, VOCAB), dtype=np.float16)
    for v0 in range(0, VOCAB, chunk):
        v1 = min(v0 + chunk, VOCAB)
        wc = np.asarray(w[v0:v1], np.float32)
        exp[:, v0:v1] = (hf @ wc.T).astype(np.float16)
    return exp


def chain_refs():
    """The CPU chain x -> h1 -> h2 (fp16 boundaries), exact float32 inside each layer."""
    P0, P1 = fixture(0), fixture(1)
    x16 = input_hidden()
    h1 = np.asarray(forward(x16, P0, np.float32, False)["out"], np.float16)
    h2 = np.asarray(forward(h1, P1, np.float32, False)["out"], np.float16)
    hs1 = np.asarray(forward(x16, P0, np.float32, True)["out"], np.float16)    # device sem.
    hs2 = np.asarray(forward(hs1, P1, np.float32, True)["out"], np.float16)
    return {"x": x16, "h1": h1, "h2": h2, "h1_sim": hs1, "h2_sim": hs2,
            "P0": P0, "P1": P1}


# ------------------------------------------------------------ flatbuffer builder
class G:
    """Accumulator with unique tensor names; buffer 0 is the activation placeholder."""

    def __init__(self, initial_size=1 << 21):
        self.b = flatbuffers.Builder(initial_size)
        self.buffers = [bo.buf_empty(self.b)]
        self.tensors, self.opcodes, self.ops = [], [], []

    def _t(self, name, shape, ttype, buf):
        self.tensors.append(bo.tensor(self.b, name, list(shape), ttype, buf))
        return len(self.tensors) - 1

    def const(self, name, arr):
        a = np.ascontiguousarray(arr)
        ttype = bo.I32 if a.dtype == np.int32 else bo.F16
        if a.dtype not in (np.int32, np.float16):
            a = a.astype(np.float16)
        self.buffers.append(bo.buf_data(self.b, a))
        return self._t(name, a.shape, ttype, len(self.buffers) - 1)

    def act(self, name, shape):
        return self._t(name, shape, bo.F16, 0)

    def op(self, builtin, ins, outs, opt_type=None, opt=None):
        self.opcodes.append(bo.opcode(self.b, builtin))
        self.ops.append(bo.operator(self.b, len(self.opcodes) - 1, ins, outs, opt_type, opt))
        return self.ops[-1]

    def finish(self, inputs, outputs, desc):
        return bo.finish(self.b, self.buffers, self.opcodes, self.tensors, self.ops,
                         inputs, outputs, desc)

    # ---- helpers (each returns the index of its output tensor) -------------------
    def fc(self, inp, name, w, bias, n):
        tw = self.const(name + "_W", np.asarray(w, np.float16))
        tb = self.const(name + "_b", np.asarray(bias, np.float16))
        t = self.act(name, (T, n))
        S = bo.S
        S.FullyConnectedOptionsStart(self.b)
        S.FullyConnectedOptionsAddFusedActivationFunction(self.b, 0)      # NONE
        S.FullyConnectedOptionsAddWeightsFormat(self.b, 0)                # DEFAULT [N,K]
        S.FullyConnectedOptionsAddKeepNumDims(self.b, True)
        S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(self.b, False)
        self.op(bo.BuiltinOperator.FULLY_CONNECTED, [inp, tw, tb], [t],
                bo.BuiltinOptions.FullyConnectedOptions, S.FullyConnectedOptionsEnd(self.b))
        return t

    def mul(self, a, name, b=None, shape=(T, D)):
        S = bo.S
        if b is None:
            b = self.const(name + "_c", np.full(shape, 1.0, np.float16))
        t = self.act(name, shape)
        S.MulOptionsStart(self.b)
        S.MulOptionsAddFusedActivationFunction(self.b, 0)
        self.op(bo.BuiltinOperator.MUL, [a, b], [t], bo.BuiltinOptions.MulOptions,
                S.MulOptionsEnd(self.b))
        return t

    def add(self, a, b, name, shape=(T, D)):
        S = bo.S
        t = self.act(name, shape)
        S.AddOptionsStart(self.b)
        S.AddOptionsAddFusedActivationFunction(self.b, 0)
        self.op(bo.BuiltinOperator.ADD, [a, b], [t], bo.BuiltinOptions.AddOptions,
                S.AddOptionsEnd(self.b))
        return t

    def mean_last(self, inp, name):
        S = bo.S
        ax = self.const(name + "_axis", np.asarray([1], np.int32))
        t = self.act(name, (T, 1))
        S.ReducerOptionsStart(self.b)
        S.ReducerOptionsAddKeepDims(self.b, True)
        self.op(bo.BuiltinOperator.MEAN, [inp, ax], [t], bo.BuiltinOptions.ReducerOptions,
                S.ReducerOptionsEnd(self.b))
        return t

    def rsqrt(self, inp, name):
        t = self.act(name, (T, 1))
        self.op(bo.BuiltinOperator.RSQRT, [inp], [t])
        return t

    def reshape(self, inp, name, shape, src_len):
        S = bo.S
        tsh = self.const(name + "_shape", np.asarray(shape, np.int32))
        t = self.act(name, shape)
        nsv = self.b.CreateNumpyVector(np.asarray(shape, np.int32))   # order-preserving (NOTES 5.4)
        S.ReshapeOptionsStart(self.b)
        S.ReshapeOptionsAddNewShape(self.b, nsv)
        self.op(bo.BuiltinOperator.RESHAPE, [inp, tsh], [t],
                bo.BuiltinOptions.ReshapeOptions, S.ReshapeOptionsEnd(self.b))
        return t

    def transpose(self, inp, name, perm, in_shape):
        S = bo.S
        tperm = self.const(name + "_perm", np.asarray(perm, np.int32))
        t = self.act(name, [in_shape[p] for p in perm])
        S.TransposeOptionsStart(self.b)
        self.op(bo.BuiltinOperator.TRANSPOSE, [inp, tperm], [t],
                bo.BuiltinOptions.TransposeOptions, S.TransposeOptionsEnd(self.b))
        return t

    def split(self, inp, name, axis, num, out_shape):
        S = bo.S
        ta = self.const(name + "_axis", np.asarray([axis], np.int32))
        outs = [self.act("%s%d" % (name, i), out_shape) for i in range(num)]
        S.SplitOptionsStart(self.b)
        S.SplitOptionsAddNumSplits(self.b, num)
        self.op(bo.BuiltinOperator.SPLIT, [ta, inp], outs,
                bo.BuiltinOptions.SplitOptions, S.SplitOptionsEnd(self.b))
        return outs

    def concat(self, ins, name, axis, out_shape):
        S = bo.S
        t = self.act(name, out_shape)
        S.ConcatenationOptionsStart(self.b)
        S.ConcatenationOptionsAddAxis(self.b, axis)
        S.ConcatenationOptionsAddFusedActivationFunction(self.b, 0)
        self.op(bo.BuiltinOperator.CONCATENATION, ins, [t],
                bo.BuiltinOptions.ConcatenationOptions, S.ConcatenationOptionsEnd(self.b))
        return t

    def bmm(self, a, b, name, shape, adj_y=False):
        S = bo.S
        t = self.act(name, shape)
        S.BatchMatMulOptionsStart(self.b)
        S.BatchMatMulOptionsAddAdjX(self.b, False)
        S.BatchMatMulOptionsAddAdjY(self.b, bool(adj_y))
        S.BatchMatMulOptionsAddAsymmetricQuantizeInputs(self.b, False)
        self.op(bo.BuiltinOperator.BATCH_MATMUL, [a, b], [t],
                bo.BuiltinOptions.BatchMatMulOptions, S.BatchMatMulOptionsEnd(self.b))
        return t

    def logistic(self, inp, name, shape):
        t = self.act(name, shape)
        self.op(bo.BuiltinOperator.LOGISTIC, [inp], [t])
        return t

    def softmax(self, inp, name, shape):
        S = bo.S
        t = self.act(name, shape)
        S.SoftmaxOptionsStart(self.b)
        S.SoftmaxOptionsAddBeta(self.b, 1.0)
        self.op(bo.BuiltinOperator.SOFTMAX, [inp], [t],
                bo.BuiltinOptions.SoftmaxOptions, S.SoftmaxOptionsEnd(self.b))
        return t


def rmsnorm_nodes(g, x, name, gamma):
    """MUL(x,x) -> MEAN -> ADD eps -> RSQRT -> MUL(x) -> MUL(gamma)  (6 layers)."""
    sq = g.mul(x, name + "_sq", b=x)      # MUL(x,x): ticket 09's bug was here (b=None -> const 1)
    mean = g.mean_last(sq, name + "_mean")
    eps = g.const(name + "_eps", np.full((T, 1), EPS, np.float16))
    m_eps = g.add(mean, eps, name + "_mean_eps", (T, 1))
    rsq = g.rsqrt(m_eps, name + "_rsqrt")
    sc = g.mul(x, name + "_scaled", b=rsq)
    return g.mul(sc, name + "_norm", b=g.const(name + "_gamma", np.asarray(gamma, np.float16)))


def build_layer_graph(P, out_name="h1"):
    """One full decoder layer as one .tflite: x[T,D] in, out[T,D] out (61 native layers)."""
    g = G(initial_size=1 << 24)
    x = g.act("x", (T, D))
    desc = "two-layer chain: decoder layer T=%d fp16 14Q/2KV" % T

    n1 = rmsnorm_nodes(g, x, "n1", P["g1"])

    def qkv(W, b, heads, tag):
        y = g.fc(n1, tag + "_fc", W, b, heads * HD)
        t3 = g.reshape(y, tag + "_3d", (T, heads, HD), heads * HD)
        c16, s16 = tables(heads)
        tc = g.const(tag + "_cos", c16)
        ts = g.const(tag + "_sin", s16)
        a, bb = g.split(t3, tag + "_split", 2, 2, (T, heads, HALF))
        p1 = g.mul(a, tag + "_a_cos", b=tc, shape=(T, heads, HALF))
        p2 = g.mul(bb, tag + "_b_sin", b=ts, shape=(T, heads, HALF))
        p3 = g.mul(bb, tag + "_b_cos", b=tc, shape=(T, heads, HALF))
        p4 = g.mul(a, tag + "_a_sin", b=ts, shape=(T, heads, HALF))
        S = bo.S
        lo = g.act(tag + "_lo", (T, heads, HALF))
        S.SubOptionsStart(g.b)
        S.SubOptionsAddFusedActivationFunction(g.b, 0)
        g.op(bo.BuiltinOperator.SUB, [p1, p2], [lo], bo.BuiltinOptions.SubOptions,
             S.SubOptionsEnd(g.b))
        hi = g.act(tag + "_hi", (T, heads, HALF))
        S.AddOptionsStart(g.b)
        S.AddOptionsAddFusedActivationFunction(g.b, 0)
        g.op(bo.BuiltinOperator.ADD, [p3, p4], [hi], bo.BuiltinOptions.AddOptions,
             S.AddOptionsEnd(g.b))
        return g.concat([lo, hi], tag + "_rope", 2, (T, heads, HD))

    q3 = qkv(P["Wq"], P["bq"], H, "q")
    k3 = qkv(P["Wk"], P["bk"], KV, "k")
    v_fc = g.fc(n1, "v_fc", P["Wv"], P["bv"], KV * HD)
    v3 = g.reshape(v_fc, "v_3d", (T, KV, HD), KV * HD)

    def kv14(t3, tag):
        k0, k1 = g.split(t3, tag + "_split2", 1, 2, (T, 1, HD))
        return g.concat([k0] * KVR + [k1] * KVR, tag + "14", 1, (T, H, HD))

    k14 = kv14(k3, "k")
    v14 = kv14(v3, "v")

    qt = g.transpose(q3, "qt", [1, 0, 2], (T, H, HD))
    kt = g.transpose(k14, "kt", [1, 0, 2], (T, H, HD))
    vt = g.transpose(v14, "vt", [1, 0, 2], (T, H, HD))
    scores = g.bmm(qt, kt, "scores", (H, T, T), adj_y=True)
    sc = g.mul(scores, "scaled", b=g.const("scale", np.full((H, T, T), SCALE, np.float16)),
               shape=(H, T, T))
    masked = g.add(sc, g.const("mask", causal_mask()), "masked", (H, T, T))
    p = g.softmax(masked, "probs", (H, T, T))
    a3 = g.bmm(p, vt, "a3", (H, T, HD))
    at = g.transpose(a3, "at", [1, 0, 2], (H, T, HD))
    attn = g.reshape(at, "attn", (T, D), H * HD)
    o = g.fc(attn, "o", P["Wo"], np.zeros(D, np.float16), D)       # o_proj: no bias
    r1 = g.add(o, x, "r1")

    n2 = rmsnorm_nodes(g, r1, "n2", P["g2"])
    gate = g.fc(n2, "gate", P["Wg"], np.zeros(INTER, np.float16), INTER)
    up = g.fc(n2, "up", P["Wu"], np.zeros(INTER, np.float16), INTER)
    sig = g.logistic(gate, "sig", (T, INTER))
    s = g.mul(gate, "silu", b=sig, shape=(T, INTER))
    y = g.mul(s, "mlp_y", b=up, shape=(T, INTER))
    down = g.fc(y, "down", P["Wd"], np.zeros(D, np.float16), D)
    out = g.add(down, r1, out_name)
    return g.finish([x], [out], desc)


def build_logits_graph(w_embed, chunks=CHUNKS, out_name="logits"):
    """Tied LM head: chunks x FC(W=embed.weight slice, b=0) + CONCAT(axis=1) -> [T,VOCAB]."""
    if VOCAB % chunks:
        raise SystemExit("vocab %d not divisible by chunks=%d" % (VOCAB, chunks))
    n = VOCAB // chunks
    g = G(initial_size=int(w_embed.nbytes) + (16 << 20))
    x = g.act("x", (T, D))
    outs = []
    for i in range(chunks):
        v0, v1 = i * n, (i + 1) * n
        outs.append(g.fc(x, "fc_%d" % i, np.ascontiguousarray(w_embed[v0:v1]),
                         np.zeros(n, np.float16), n))
    out = g.concat(outs, out_name, 1, (T, VOCAB))
    return g.finish([x], [out], "two-layer chain: tied lm_head T=%d vocab=%d fp16 chunks=%d"
                    % (T, VOCAB, chunks))


# ------------------------------------------------------------------ stats helpers
def quantile_stats(a):
    a = np.asarray(a, np.float64)
    return {"min": float(a.min()), "max": float(a.max()),
            "abs_max": float(np.abs(a).max()),
            "rms": float(np.sqrt((a ** 2).mean()))}


def ulp_of(mx):
    return float(np.spacing(np.float16(mx)))


def weight_inventory(P, layer):
    rows, total = [], 0
    for name, key in [("q_proj", "Wq"), ("k_proj", "Wk"), ("v_proj", "Wv"),
                      ("o_proj", "Wo"), ("gate_proj", "Wg"), ("up_proj", "Wu"),
                      ("down_proj", "Wd")]:
        a = np.asarray(P[key], np.float16)
        total += int(a.nbytes)
        rows.append({"tensor": "layer%d.%s" % (layer, name),
                     "shape": list(a.shape), "bytes": int(a.nbytes),
                     "mib": round(int(a.nbytes) / 2 ** 20, 4)})
    small = 0
    for key in ("g1", "g2", "bq", "bk", "bv"):
        small += int(np.asarray(P[key], np.float16).nbytes)
    small += int((D + 2 * INTER + D) * 2)
    tables_bytes = int(2 * (T * H * HALF + T * KV * HALF) * 2 * 2)
    mask_bytes = int(H * T * T * 2)
    rows.append({"tensor": "norm gammas + qkv biases + zero FC biases",
                 "bytes": small, "mib": round(small / 2 ** 20, 4)})
    rows.append({"tensor": "rope cos/sin tables (q+k) + causal mask",
                 "bytes": tables_bytes + mask_bytes,
                 "mib": round((tables_bytes + mask_bytes) / 2 ** 20, 4)})
    return rows, total + small + tables_bytes + mask_bytes





# ------------------------------------------------------------------ tolerance
def tolerance_block(mx, floor_self, floor_chain, what):
    floor = max(floor_self, floor_chain)
    term_ulp = TOL_K * FP16_ULP_REL * mx
    term_floor = TOL_K * floor
    atol = rtol = max(term_ulp, term_floor)
    ulp = ulp_of(mx)
    basis = (
        "max|%s| = %.4f; the fp16 ULP at that magnitude is %.4g (%.2e relative). "
        "Device-independent floor = %.4e abs = %.2f ULP = %.2e relative, taken as "
        "max(floor_self=%.4e, floor_chain=%.4e): floor_self is this layer's own fp16 "
        "rounding floor (numpy chain with a fp16 rounding at EVERY tensor boundary vs the "
        "single-rounding golden -- what MDLA does, docs/OP-qkv.md 5.2), floor_chain is how "
        "far this layer's reference moves when it is fed the fp16-rounded previous-layer "
        "output instead of the exact one, i.e. the propagation the board handoff must "
        "introduce, estimated from numpy only. Registered atol = rtol = max(%g*2^-11*max|out|, "
        "%g*floor) = %.6g; the %s term dominates. NOTES 4's flat 2e-2 would be %.2e relative "
        "here, i.e. %s. No device measurement entered this number."
        % (what, mx, ulp, ulp / mx, floor, floor / ulp, floor / mx, floor_self, floor_chain,
           TOL_K, TOL_K, atol, "4*floor" if term_floor > term_ulp else "4*2^-11*max|out|",
           2e-2 / mx,
           "looser than the measured floor (weak)" if 2e-2 > 4 * floor
           else "tighter than the measured floor (unusable)"))
    self_atol = max(TOL_K * FP16_ULP_REL * mx, TOL_K * floor_self)
    block = {
        "revision": 1,
        "registered_before_run": True,
        "policy": "NOTES 4 / 10.1: atol = rtol = max(k*2^-11*max|out|, k*floor), k=%g; "
                  "the propagation allowance enters through floor_chain (see basis). The "
                  "own-error criterion B uses self_atol = self_rtol = max(k*2^-11*max|out|, "
                  "k*floor_self)." % TOL_K,
        "basis": basis,
        "self_atol": round(float(self_atol), 10),
        "self_rtol": round(float(self_atol), 10),
        "self_policy": "comparison B (CPU reference rebuilt from the board's own input, i.e. "
                       "this layer's own error with the propagation removed) is judged against "
                       "self_atol = self_rtol = max(k*2^-11*max|out|, k*floor_self), registered "
                       "here before any device run.",
        "max_abs_expected": mx,
        "ulp_at_max": ulp,
        "floor_self": floor_self,
        "floor_chain": floor_chain,
        "floor_used": floor,
        "floor_ulp": floor / ulp,
        "rel_equivalent": atol / mx + rtol,
        "under_2e-2_cap": bool(atol + rtol * mx < 2e-2),
        "top1": {
            "enabled": False,
            "why": "ADR 0003: under frozen random weights top-1 carries no semantics and one "
                   "fp16 ULP flips near-ties, so argmax is not a criterion. Frozen until the "
                   "real-weight stage. This ticket compares the logits VECTOR only.",
        },
    }
    return float(atol), float(rtol), block


# ------------------------------------------------------------------------ build
def build(kind, outdir=None, weights_dir=None):
    spec = GRAPHS[kind]
    outdir = outdir or spec["out"]
    os.makedirs(os.path.join(outdir, "golden"), exist_ok=True)
    ch = chain_refs()

    if kind == "l0":
        layer = spec["layer"]
        P = ch["P0"]
        x_in = ch["x"]
        exp = ch["h1"]
        sim = ch["h1_sim"]
        floor_self = float(np.abs(sim.astype(np.float64) - exp.astype(np.float64)).max())
        floor_chain = 0.0                      # chain root: nothing is handed to it
        golden64 = np.asarray(forward(ch["x"], P, np.float64, False)["out"], np.float16)
        blob = build_layer_graph(P, spec["output"])
        wrows, wtotal = weight_inventory(P, layer)
        trace = {k: quantile_stats(v) for k, v in
                 forward(ch["x"], P, np.float32, False, "out").items()}
        in_note = ("layer-0 input hidden = embed rows for token ids %s (fp16), exactly ticket "
                   "09's boundary" % TOKEN_IDS)
    elif kind == "l1":
        layer = spec["layer"]
        P = ch["P1"]
        x_in = ch["h1"]
        exp = ch["h2"]
        sim = ch["h2_sim"]
        floor_self = float(np.abs(forward(x_in, P, np.float32, True)["out"].astype(np.float64)
                                  - exp.astype(np.float64)).max())
        floor_chain = float(np.abs(forward(ch["h1_sim"], P, np.float32, False)["out"]
                                   .astype(np.float64) - exp.astype(np.float64)).max())
        golden64 = np.asarray(forward(x_in, P, np.float64, False)["out"], np.float16)
        blob = build_layer_graph(P, spec["output"])
        wrows, wtotal = weight_inventory(P, layer)
        trace = {k: quantile_stats(v) for k, v in
                 forward(x_in, P, np.float32, False, "out").items()}
        in_note = ("layer-1 input = h1_cpu, the fp16 handoff of the CPU chain "
                   "(tools/chain_two_layer.sh REPLACES this file with layer 0's device "
                   "out_device.bin before the board run)")
    else:
        layer = None
        w = embed_weight()
        x_in = ch["h2"]
        exp = logits_ref(x_in, w)
        sim = exp.copy()                       # a single output rounding -> no extra floor
        floor_self = 0.0
        floor_chain = float(np.abs(logits_ref(ch["h2_sim"], w).astype(np.float64)
                                   - exp.astype(np.float64)).max())
        golden64 = None
        blob = build_logits_graph(w)
        wrows, wtotal = [{"tensor": "embed.weight (tied lm_head), 4 chunks",
                          "shape": [VOCAB, D], "bytes": int(w.nbytes),
                          "mib": round(int(w.nbytes) / 2 ** 20, 4)}], int(w.nbytes) + VOCAB * 2
        trace = {"input": quantile_stats(x_in), "logits": quantile_stats(exp)}
        in_note = ("logits graph input = h2_cpu, the fp16 handoff of the CPU chain "
                   "(tools/chain_two_layer.sh REPLACES this file with layer 1's device "
                   "out_device.bin before the board run)")

    mx = float(np.abs(exp.astype(np.float64)).max())
    atol, rtol, tol = tolerance_block(mx, floor_self, floor_chain, spec["output"])
    if golden64 is not None:
        g64_codes = int((golden64 == exp).sum())
        f32_vs_f64 = float(np.abs(golden64.astype(np.float64) - exp.astype(np.float64)).max())
        ref_note = ("golden/expected.bin is an independent numpy float32 forward pass over the "
                    "stored fp16 weights/inputs/tables; it shares no code with the flatbuffer "
                    "writer (which contains no maths). A float64 rerun gives %d/%d identical "
                    "fp16 codes (max|f32-f64| = %.3e), so the reference is not precision-limited."
                    % (g64_codes, exp.size, f32_vs_f64))
    else:
        g64_codes = None
        ref_note = ("golden/expected.bin is a plain numpy float32 matmul of the stored fp16 "
                    "weights and input, in vocab chunks; it shares no code with the flatbuffer "
                    "writer.")

    with open(os.path.join(outdir, "model.tflite"), "wb") as f:
        f.write(blob)
    np.ascontiguousarray(x_in, np.float16).tofile(os.path.join(outdir, "golden", "input.bin"))
    np.ascontiguousarray(exp, np.float16).tofile(os.path.join(outdir, "golden", "expected.bin"))

    out_shape = [T, D] if layer is not None else [T, VOCAB]
    meta = {
        "op": spec["op"],
        "role": kind,
        "graph": ("x[%d,%d] -> decoder layer %d (layer%d.* weights, 61 native layers) -> %s[%d,%d]"
                  % (T, D, layer, layer, spec["output"], T, D) if layer is not None else
                  "x[%d,%d] -> %d x FULLY_CONNECTED(W=embed.weight slice, b=0) -> CONCAT(axis=1) "
                  "-> %s[%d,%d]" % (T, D, CHUNKS, spec["output"], T, VOCAB)),
        "shape": {"T": T, "D": D, "Hq": H, "Hkv": KV, "HD": HD, "I": INTER,
                  **({"layer": layer} if layer is not None else {"vocab": VOCAB})},
        "dtype": "fp16",
        "inputs": [{"name": "x", "shape": [T, D], "type": "float16",
                    "bytes": int(np.asarray(x_in).nbytes), "provenance": in_note}],
        "output": {"name": spec["output"], "shape": out_shape, "type": "float16",
                   "bytes": int(np.asarray(exp).nbytes)},
        "quant": {},
        "reference": "numpy_float32_from_stored_fp16_one_final_rounding",
        "reference_note": ref_note,
        "atol": round(atol, 10),
        "rtol": round(rtol, 10),
        "tolerance_revision": tol,
        "threshold_at_max": float(atol + rtol * mx),
        "fp16_floor": {"floor_self": floor_self, "floor_chain": floor_chain,
                       "floor_used": max(floor_self, floor_chain),
                       "ulp_at_max": ulp_of(mx), "max_abs_out": mx,
                       "max_abs_out_ulp": mx / ulp_of(mx)},
        "eps_register": {"graph_eps": EPS, "eps_stored_fp16": float(np.float16(EPS)),
                         "note": "Qwen2.5 rms_norm_eps = 1e-6, as in ticket 09; it is "
                                 "numerically unobservable at this dynamic range (ticket 09 "
                                 "section 7 measured eps=0 vs 1e-6 at 0.20x threshold)"},
        "mask": {"value": MASK, "position": "ADD after MUL(1/sqrt(HD))",
                 "causal": "0 at j<=i, MASK at j>i", "form": "const [H,T,T] baked in graph"},
        "rope": {"convention": "qwen2_split_half", "theta": THETA, "head_dim": HD,
                 "positions": "%d..%d" % (POS0, POS0 + T - 1),
                 "coherence": "both layers use the identical table (same THETA, POS0, T)"},
        "chain": {"role": kind,
                  "hands_to": {"l0": "two_layer_l1_t8_fp16", "l1": "two_layer_logits_t8_fp16",
                               "logits": None}[kind],
                  "handoff": "device out_device.bin is copied over the next op's "
                             "golden/input.bin (NOTES 10.4 item 5); no CPU correction in "
                             "between",
                  "weights_reload": "each layer is its own .dla, so the whole static weight "
                                    "block is pushed and loaded again per layer"},
        "weight_inventory": {"rows": wrows, "total_bytes": wtotal,
                             "total_mib": round(wtotal / 2 ** 20, 4)},
        "stats": {"input": quantile_stats(x_in), "output": quantile_stats(exp), "traces": trace},
        "generator": "tools/build_two_layer.py",
        "tflite": "model.tflite",
        "tflite_bytes": len(blob),
    }
    with open(os.path.join(outdir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("%s [%s]: in%s out%s  tflite=%dB  max|%s|=%.4f (ULP %.3g)  floor_self=%.3e "
          "floor_chain=%.3e -> atol=rtol=%.6g  weights=%.2f MiB"
          % (outdir, kind, list(np.asarray(x_in).shape), out_shape, len(blob),
             spec["output"], mx, ulp_of(mx), floor_self, floor_chain, atol, wtotal / 2 ** 20))
    return 0


# ------------------------------------------------------- device-free graph checker
def execcheck(kind, opdir):
    """Run the built .tflite in numpy and compare with the registered golden (no ncc, no board).

    exec_graph is ticket 09's reader: it reads the operator list, tensor shapes and constant
    buffers back out of the flatbuffer and applies standard TFLite semantics.  It is a WIRING
    check (ordered inputs of SPLIT/CONCAT, a MUL wired to the wrong operand, a missing
    TRANSPOSE, a constant that never made it into the file), not a numerical criterion --
    NOTES 10.2 / spec C4 / ticket 09 section 5.3.  Must be run BEFORE the board run, while
    golden/input.bin still holds the CPU-chain input.
    """
    import build_decoder_layer as bdl          # only for its numpy graph executor
    meta = json.load(open(os.path.join(opdir, "model.json")))
    x = np.fromfile(os.path.join(opdir, "golden", "input.bin"), dtype=np.float16)
    exp = np.fromfile(os.path.join(opdir, "golden", "expected.bin"), dtype=np.float16)
    shape = tuple(meta["inputs"][0]["shape"])
    got = bdl.exec_graph(os.path.join(opdir, "model.tflite"),
                         x.reshape(shape).astype(np.float32))
    if len(got) != 1:
        raise SystemExit("expected exactly one graph output, got %d" % len(got))
    g = np.asarray(got[0], np.float16).reshape(-1)
    e = exp.reshape(-1)
    if g.size != e.size:
        print("FAIL size: graph %s vs golden %s" % (g.shape, e.shape))
        return 1
    mx = float(np.abs(e.astype(np.float64)).max())
    ulp = ulp_of(mx)
    d = np.abs(g.astype(np.float64) - e.astype(np.float64))
    tol = TOL_K * ulp
    ok = d.max() <= tol
    print("execcheck %-42s max_abs=%.4e (%.2f ULP, %.2f ULP-tol)  rmse=%.3e  rel_l2=%.3e  "
          "codes_same=%d/%d -> %s"
          % (opdir, d.max(), d.max() / ulp, d.max() / tol, float(np.sqrt((d ** 2).mean())),
             float(np.linalg.norm(d) / np.linalg.norm(e.astype(np.float64))),
             int((g == e).sum()), e.size, "WIRING OK" if ok else "WIRING MISMATCH"))
    return 0 if ok else 1


# -------------------------------------------------------------------- --analyze
def analyze(kind, opdir):
    meta = json.load(open(os.path.join(opdir, "model.json")))
    ch = chain_refs()
    if kind == "logits":
        w = embed_weight()
        for name, h in (("h2_cpu", ch["h2"]), ("h2_sim", ch["h2_sim"])):
            a = logits_ref(h, w).astype(np.float64)
            print("%-8s max|v|=%.5g rms=%.5g" % (name, np.abs(a).max(), np.sqrt((a * a).mean())))
        print("tolerance in model.json: atol=rtol=%g" % meta["atol"])
        return 0
    layer = GRAPHS[kind]["layer"]
    P = ch["P0"] if layer == 0 else ch["P1"]
    x_in = ch["x"] if layer == 0 else ch["h1"]
    full = forward(x_in, P, np.float32, False, "out")
    sim = forward(x_in, P, np.float32, True, "out")
    g64 = forward(x_in, P, np.float64, False, "out")
    print("%-8s %12s %10s   %-30s %s" % ("stage", "max|v|", "rms", "floor sim-vs-golden",
                                         "golden f32 vs f64 (fp16 codes)"))
    for k in ["x", "n1", "q3", "k14", "v14", "scores", "probs", "attn", "r1", "n2",
              "mlp_y", "down", "out"]:
        a64 = np.asarray(full[k], np.float16)
        a = a64.astype(np.float64)
        s = np.asarray(sim[k], np.float16).astype(np.float64)
        g = np.asarray(g64[k], np.float16)
        mx = np.abs(a).max()
        ulp = ulp_of(mx)
        print("%-8s %12.5g %10.5g   abs=%.3e (%5.2f ULP of this stage)   diff=%.2e  same=%d/%d"
              % (k, mx, np.sqrt((a * a).mean()), np.abs(s - a).max(), np.abs(s - a).max() / ulp,
                 np.abs(g.astype(np.float64) - a).max(), int((g == a64).sum()), a.size))
    sc = np.asarray(full["scores"], np.float64)
    S = float(np.abs(sc).max() * SCALE)
    print("mask check: max|scores|=%.2f  S=max|scaled|=%.2f  limit S<=1996 -> %s"
          % (np.abs(sc).max(), S, "OK" if S <= 1996 else "MUST RE-DERIVE M"))
    print("tolerance in model.json: atol=rtol=%g (floor_used=%.3e)"
          % (meta["atol"], meta["fp16_floor"]["floor_used"]))
    return 0


# ------------------------------------------------------------------------ --diag
def _cmp(dev, ref, atol, rtol, label):
    d = np.abs(dev.astype(np.float64) - ref.astype(np.float64))
    mx = float(np.abs(ref.astype(np.float64)).max())
    ulp = ulp_of(mx)
    thr = atol + rtol * mx
    ev = np.abs(ref.astype(np.float64))
    ok = bool(np.all(d <= atol + rtol * ev)) if d.size else True
    print("%-30s max_abs=%.4e (%.2f ULP, rel %.3e)  rmse=%.3e  rel_l2=%.3e  threshold=%.4e  "
          "-> %s" % (label, d.max(), d.max() / ulp, d.max() / mx,
                     float(np.sqrt((d ** 2).mean())),
                     float(np.linalg.norm(d) / max(np.linalg.norm(ref.astype(np.float64)), 1e-12)),
                     thr, "PASS" if ok else "FAIL"))
    return ok, float(d.max())


def diag(kind, opdir):
    """After a board run: propagation error vs this layer's own error, noise floor, sim gap."""
    meta = json.load(open(os.path.join(opdir, "model.json")))
    exp = np.fromfile(os.path.join(opdir, "golden", "expected.bin"), dtype=np.float16)
    dev_p = os.path.join(opdir, "out_device.bin")
    if not os.path.exists(dev_p):
        raise SystemExit("no out_device.bin in %s (run tools/chain_two_layer.sh first)" % opdir)
    dev = np.fromfile(dev_p, dtype=np.float16)
    x_in = np.fromfile(os.path.join(opdir, "golden", "input.bin"), dtype=np.float16)
    shape = tuple(meta["inputs"][0]["shape"])
    x_in = x_in.reshape(shape)
    atol, rtol = float(meta["atol"]), float(meta["rtol"])

    if kind == "logits":
        ref_actual = logits_ref(x_in).reshape(-1)
        sim_actual = ref_actual
        golden_chain = exp.reshape(-1)
    else:
        layer = GRAPHS[kind]["layer"]
        P = fixture(layer)
        ref_actual = np.asarray(forward(x_in, P, np.float32, False)["out"], np.float16).reshape(-1)
        sim_actual = np.asarray(forward(x_in, P, np.float32, True)["out"], np.float16).reshape(-1)
        golden_chain = exp.reshape(-1)

    rev = meta["tolerance_revision"]
    s_atol, s_rtol = float(rev["self_atol"]), float(rev["self_rtol"])
    print("== %s (%s) ==" % (opdir, kind))
    print("registered tolerance : A(atol=rtol)=%.6g  B(self)=%.6g  (max|%s| = %.4f, "
          "floor_self = %.3e, floor_chain = %.3e)"
          % (atol, s_atol, GRAPHS[kind]["output"], meta["fp16_floor"]["max_abs_out"],
             meta["fp16_floor"]["floor_self"], meta["fp16_floor"]["floor_chain"]))
    ok_prop, _ = _cmp(dev, golden_chain, atol, rtol,
                      "A. propagation (vs CPU chain)")
    ok_self, _ = _cmp(dev, ref_actual, s_atol, s_rtol,
                      "B. own error (ref from board input)")
    _cmp(dev, sim_actual, atol, rtol, "C. dev vs fp16-sim (semantics)")
    print("   bit-identical codes vs CPU chain: %d/%d" % (int((dev == golden_chain).sum()), dev.size))

    r1, r2 = (os.path.join(opdir, n) for n in ("out_r1.bin", "out_r2.bin"))
    if os.path.exists(r1) and os.path.exists(r2):
        a1 = np.fromfile(r1, dtype=np.float16)
        a2 = np.fromfile(r2, dtype=np.float16)
        print("   noise floor (same .dla, same input, 2 runs): max|r1-r2| = %.4e  (%d/%d codes "
              "differing)" % (float(np.abs(a1.astype(np.float64) - a2.astype(np.float64)).max()),
                              int((a1 != a2).sum()), a1.size))
    print("   VERDICT: propagation %s / own-error %s"
          % ("PASS" if ok_prop else "FAIL", "PASS" if ok_self else "FAIL"))
    return 0 if (ok_prop and ok_self) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=list(GRAPHS), default="l0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--execcheck", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--weights-dir", default=None)
    a = ap.parse_args()
    outdir = a.out or GRAPHS[a.graph]["out"]

    if a.predict:
        tot = 61
        print("  %-40s %2d" % ("one decoder layer (measured by ticket 09)", tot))
        print("  %-40s %2d" % ("tied LM head: 4 x FC + 1 x CONCAT (ticket 08)", 5))
        print("  %-40s %2d" % ("three graphs, three execution steps at most", tot * 2 + 5))
        return 0
    if a.execcheck:
        return execcheck(a.graph, outdir)
    if a.analyze:
        return analyze(a.graph, outdir)
    if a.diag:
        return diag(a.graph, outdir)
    if a.weights_dir:
        os.environ["QWEN25_WEIGHTS_DIR"] = a.weights_dir
    return build(a.graph, outdir)


if __name__ == "__main__":
    sys.exit(main() or 0)
