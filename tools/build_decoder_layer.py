#!/usr/bin/env python3
"""One whole Qwen2.5-0.5B decoder layer, T=8 (or --T N), fp16, 14 query / 2 KV heads, hidden 896.

    /usr/bin/python3 tools/build_decoder_layer.py                       # -> ops/decoder_layer_t8_fp16
    /usr/bin/python3 tools/build_decoder_layer.py --T 128               # -> ops/decoder_layer_t128_fp16
    /usr/bin/python3 tools/build_decoder_layer.py --stage r1 --out ops/_decoder_bisect/r1
    /usr/bin/python3 tools/build_decoder_layer.py --variant ffn_half --out ops/_decoder_bisect/ffn
    /usr/bin/python3 tools/build_decoder_layer.py --diag                # after a board run

Graph (one .tflite, ONE runtime input x[T,896] fp16, ONE output out[T,896] fp16):

    x --RMSNorm(input_layernorm, eps=1e-6)--> n1
      --FC q/k/v (+bias)--> RESHAPE[T,H,64] --> RoPE(split-half, const tables)--> [T,H,64]
      --SPLIT+CONCAT (2 KV heads -> 14, mapping h//7)--> k14, v14
      --TRANSPOSE [T,H,HD]->[H,T,HD]--> BMM(adj_y) --> scores[14,8,8]
      --MUL(1/8) --> ADD(causal mask const, 0/-4096) --> SOFTMAX(beta=1) --> p
      --BMM(p, v14^T) --> [14,8,64] --TRANSPOSE[1,0,2]--> [8,14,64] --RESHAPE--> [8,896]
      --FC(o_proj, no bias)--> ADD(x)                       # residual 1
      --RMSNorm(post_attention_layernorm, eps=1e-6)
      --SwiGLU (gate/up -> LOGISTIC -> MUL -> MUL -> down)
      --ADD(residual 1)--> out

Why the shape choices: every one of them is copied from a ticket that has board evidence
(docs/OP-rmsnorm.md AE-variant, docs/OP-qkv.md, docs/OP-gqa-scores.md, docs/OP-masked-softmax.md,
docs/OP-pv-o.md, docs/OP-swiglu-ffn.md).  The interesting question of ticket 09 is not
"which op" but "does the assembly still fit in ONE execution step".

Three numpy references, all device-free (AGENTS R3: the golden shares no code with the
flatbuffer writer, which contains no maths at all):

  golden   float32 chain from the *stored fp16* inputs/weights/tables, ONE final rounding to
           fp16 -> golden/expected.bin (the registered reference)
  golden64 float64 chain -> diagnostic only (is the float32 golden precision-limited?)
  sim      float32 arithmetic with a fp16 rounding at EVERY tensor boundary (what MDLA does:
           docs/OP-qkv.md 5.2 measured 7167/7168 bit-identical) -> device-independent floor
           used to justify the tolerance (NOTES 10.1), never written to golden/

Tolerance is registered in model.json *before* the first board run:
    atol = rtol = max(4 * 2^-11 * max|out|, 4 * floor)      (NOTES 10.1 k=4 form)
with the components recorded so a reader can see which term dominates.

eps: the real Qwen2.5 rms_norm_eps = 1e-6 enters the graph as a [T,1] fp16 constant
(fp16(1e-6) = 9.537e-07, a subnormal: only ~6% accurate).  The older toy blocks
(docs/OP-block.md) used eps = 0; that difference is recorded in model.json
("eps_register") and in docs/OP-decoder-layer.md, because eps is *unobservable* at this
dynamic range (it moves the output by << 1 ULP) while being a real semantic difference.
"""
import argparse
import json
import os
import sys

import flatbuffers
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ops as bo          # flatbuffer toolbox (read-only use, AGENTS 8)
import qwen25_weights as qw     # frozen fixture loader (verify-on-load, sha256-checked)

T = 8                   # sequence length; --T overrides (static: the RoPE tables are data)
D = 896
H = 14                  # query heads
KV = 2                  # KV heads
HD = 64                 # head_dim
HALF = HD // 2
INTER = 4864            # SwiGLU intermediate width
THETA = 1e6
POS0 = 0
EPS = 1e-6              # transformers rms_norm_eps (NOT the toy block's 0)
SCALE = 1.0 / 8.0       # 1/sqrt(64), an exact power of two in fp16
MASK_REF = -4096.0       # ticket 05's value, 2^12; interval in docs/OP-masked-softmax.md 3.4
MASK = MASK_REF         # in effect; re-derived from THIS layer's S by derive_mask() (05 11.2)
LAYER = 0
KVR = H // KV           # 7

# fixed token ids so golden/input.bin is byte-reproducible (same list as ticket 03/06)
TOKEN_IDS_BASE = [9707, 11, 1879, 828, 11780, 13, 220, 16]
TOKEN_IDS = list(TOKEN_IDS_BASE)

DEFAULT_OUT = "ops/decoder_layer_t8_fp16"
STAGES = ["n1", "q3", "k14", "v14", "scores", "probs", "attn", "r1", "n2", "mlp_y", "down", "out"]


# ------------------------------------------------------------ T configuration
def token_ids(t):
    """t distinct token ids, fixed forever, so golden/input.bin is byte-reproducible.

    Positions 0..7 are ticket 03/06/09's list verbatim (that is what keeps the T=8 product
    byte-identical).  Beyond that the rule is arithmetic, not a hand-picked list:

        id(i) = (41 + 1103 * i) % vocab_size,  i = 0, 1, 2, ...

    stride 1103 is coprime to the 151936-entry vocabulary and 1103*119 < 151936, so the
    first 120 values are distinct, and none of them hits one of the 8 base ids (checked
    below).  Different rows of the embedding table => 128 genuinely different positions.
    """
    vocab = int(qw.get("embed.weight").shape[0])
    ids = list(TOKEN_IDS_BASE[:t])
    i = 0
    while len(ids) < t:
        cand = (41 + 1103 * i) % vocab
        i += 1
        if cand not in ids:
            ids.append(cand)
    assert len(set(ids)) == t, "token ids must be distinct"
    return ids


def derive_mask(s_scaled):
    """Mask constant M for this layer: M in [2S+104, 65504-S] (docs/OP-masked-softmax.md 3.4).

    Ticket 05's -4096 is kept whenever it is still inside the interval (that is the T=8
    case, and it keeps that product byte-identical).  When the layer gets longer the scores
    grow and S grows with them, so M has to be re-derived instead of reused: the smallest
    power of two >= 2S+104 is taken, because a power of two is exact in fp16 and auditable.
    """
    lo = 2.0 * s_scaled + 104.0
    hi = 65504.0 - s_scaled
    if lo <= -MASK_REF <= hi:
        return MASK_REF, "reused", lo, hi
    p = 1024.0
    while p < lo:
        p *= 2.0
    if p > hi:
        raise SystemExit("no valid mask constant: need M >= %.1f and M <= %.1f (S=%.1f); "
                         "fp16 cannot hold both => T is too long for this mask scheme"
                         % (lo, hi, s_scaled))
    return -p, "re-derived", lo, hi


def configure(t):
    """Re-bind every T-dependent module constant.  T=8 reproduces the T=8 product exactly."""
    global T, TOKEN_IDS, DEFAULT_OUT
    T = int(t)
    if T <= 0:
        raise SystemExit("--T must be positive")
    TOKEN_IDS = token_ids(T)
    DEFAULT_OUT = "ops/decoder_layer_t%d_fp16" % T


# ------------------------------------------------------------------ fixture data
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


def input_hidden():
    """Layer input hidden state: the raw embed rows for fixed token ids, fp16.

    This is the honest boundary of "one decoder layer": the graph owns its own RMSNorm, so
    it must be fed what the previous layer would hand it, i.e. the (unnormalised) hidden.
    """
    emb = qw.get("embed.weight")
    return np.ascontiguousarray(emb[np.asarray(TOKEN_IDS)].astype(np.float16))


def tables(heads, p0=POS0):
    """cos/sin for positions p0..p0+T-1, half head, duplicated over heads (no broadcast)."""
    inv = THETA ** (-2.0 * np.arange(HALF) / HD)
    ang = np.outer(p0 + np.arange(T), inv)
    c = np.broadcast_to(np.cos(ang)[:, None, :], (T, heads, HALF))
    s = np.broadcast_to(np.sin(ang)[:, None, :], (T, heads, HALF))
    return (np.ascontiguousarray(c).astype(np.float16),
            np.ascontiguousarray(s).astype(np.float16))


def causal_mask():
    """0 at visible positions, MASK where j > i (strictly upper triangle). [H,T,T] fp16."""
    m = np.zeros((T, T), np.float16)
    m[np.triu_indices(T, k=1)] = np.float16(MASK)
    return np.ascontiguousarray(np.broadcast_to(m, (H, T, T)).copy())


# ------------------------------------------------------------------------ maths
def _rmsnorm(x, g, store):
    """y = x / sqrt(mean(x^2) + eps) * gamma, with device rounding at every intermediate."""
    x = store(x)
    sq = store(x * x)
    mean = store(sq.mean(axis=-1, keepdims=True))
    mean_eps = store(mean + store(np.full_like(mean, EPS)))
    rsq = store(1.0 / np.sqrt(mean_eps))
    scaled = store(x * rsq)                     # broadcast [T,D] x [T,1]
    return store(scaled * store(g.astype(x.dtype)))


def _rope(y, c16, s16, store):
    """Split-half RoPE on a [T,h,HD] tensor.  out_lo = a*cos - b*sin ; out_hi = b*cos + a*sin."""
    dt = y.dtype
    y3 = store(y).reshape(T, -1, HD)
    a, b = y3[..., :HALF], y3[..., HALF:]
    C = np.repeat(c16.astype(dt)[:, :1, :], y3.shape[1], axis=1)
    S = np.repeat(s16.astype(dt)[:, :1, :], y3.shape[1], axis=1)
    lo = store(store(a * C) - store(b * S))
    hi = store(store(b * C) + store(a * S))
    return store(np.concatenate([lo, hi], axis=-1)).reshape(T, y3.shape[1], HD)


def _softmax(v, store):
    """Stable softmax over the last axis; the device's internal stabilisation is unknown
    (docs/OP-masked-softmax.md 9.2), so the reference uses the canonical max-subtracted form."""
    m = store(v - store(v.max(axis=-1, keepdims=True)))
    e = store(np.exp(m.astype(np.float64)).astype(v.dtype))
    return store(e / store(e.sum(axis=-1, keepdims=True)))


def forward(x16, P, dtype=np.float32, round_fp16=False, stage="out"):
    """The whole layer in numpy.  No flatbuffer, no ncc, no device (AGENTS R3).

    dtype=np.float32, round_fp16=False -> the registered golden (one final rounding)
    dtype=np.float32, round_fp16=True  -> device simulation (every boundary rounded)
    dtype=np.float64                    -> precision cross-check only
    Returns (dict of intermediates, list of tensor traces) for the requested stage.
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
        """FC(+bias) then RESHAPE [T, heads*HD] -> [T, heads, HD] (token-major)."""
        y = n1 @ W.astype(dtype).T + b.astype(dtype)
        return st(y).reshape(T, heads, HD)

    q3 = _rope(proj(P["Wq"], P["bq"], H), *tables(H), store=st)
    out["q3"] = q3
    k3 = _rope(st(n1 @ P["Wk"].astype(dtype).T + P["bk"].astype(dtype)).reshape(T, KV, HD),
               *tables(KV), store=st)
    v3 = st(st(n1 @ P["Wv"].astype(dtype).T + P["bv"].astype(dtype)).reshape(T, KV, HD))
    # KV replication, mapping head h -> KV head h // 7 (repeat_interleave), in-graph SPLIT+CONCAT
    k14 = np.repeat(k3, KVR, axis=1)
    v14 = np.repeat(v3, KVR, axis=1)
    out["k14"], out["v14"] = k14, v14
    if stage == "q3":
        return out
    if stage == "k14":
        return out
    if stage == "v14":
        return out

    qt = np.transpose(q3, (1, 0, 2))
    kt = np.transpose(k14, (1, 0, 2))
    vt = np.transpose(v14, (1, 0, 2))
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
    o = st(attn @ P["Wo"].astype(dtype).T)                 # o_proj: NO bias
    r1 = st(st(o) + st(x))
    out["r1"] = r1
    if stage == "r1":
        return out

    n2 = _rmsnorm(r1, P["g2"].astype(dtype), st)
    out["n2"] = n2
    if stage == "n2":
        return out
    gate = st(n2 @ P["Wg"].astype(dtype).T)                # MLP: no bias
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


def forward_ffn(r1_16, P, dtype=np.float32, round_fp16=False):
    """Second half only: r1 -> RMSNorm(post) -> SwiGLU -> + r1."""
    def st(a):
        a = np.asarray(a, dtype)
        return np.asarray(a, np.float16).astype(dtype) if round_fp16 else a
    r1 = st(r1_16.astype(dtype))
    n2 = _rmsnorm(r1, P["g2"].astype(dtype), st)
    gate = st(n2 @ P["Wg"].astype(dtype).T)
    up = st(n2 @ P["Wu"].astype(dtype).T)
    sig = st(1.0 / (1.0 + np.exp(-gate.astype(np.float64))).astype(dtype))
    y = st(st(st(gate * sig)) * up)
    down = st(y @ P["Wd"].astype(dtype).T)
    return st(st(down) + st(r1))


def counterfactuals(x16, P):
    """Plausible bugs; the device only matches the golden if none of them is what it computes."""
    g64 = lambda k, dt=np.float64: P[k].astype(dt)
    x = x16.astype(np.float64)
    n1 = x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * g64("g1")
    out = {}
    # (0) the real thing, exact float64 (sanity: equals the float32 golden to fp16 codes)
    out["exact_f64"] = None
    q3 = _rope(n1 @ g64("Wq").T + g64("bq"), *tables(H), store=np.asarray)
    k3 = _rope(n1 @ g64("Wk").T + g64("bk"), *tables(KV), store=np.asarray)
    v3 = (n1 @ g64("Wv").T + g64("bv")).reshape(T, KV, HD)
    k14, v14 = np.repeat(k3, KVR, 1), np.repeat(v3, KVR, 1)
    qt, kt, vt = (np.transpose(z, (1, 0, 2)) for z in (q3, k14, v14))
    s = qt @ np.transpose(kt, (0, 2, 1))
    p = _softmax(s * SCALE + causal_mask().astype(np.float64), np.asarray)
    attn = np.transpose(p @ vt, (1, 0, 2)).reshape(T, D)
    r1 = x + attn @ g64("Wo").T
    n2 = r1 / np.sqrt((r1 * r1).mean(-1, keepdims=True) + EPS) * g64("g2")
    gate = n2 @ g64("Wg").T
    y = (gate / (1 + np.exp(-gate))) * (n2 @ g64("Wu").T)
    return r1 + y @ g64("Wd").T, (x, n1, s, p, r1)


def cf_variants(x16, P):
    """Named wrong graphs, all numpy, used by --diag to show the judgement has teeth."""
    ref, (x, n1, s, p, r1) = counterfactuals(x16, P)
    g = lambda k: P[k].astype(np.float64)
    out = {}
    # (1) causal mask dropped (mask ADD silently constant-folded away)
    p0 = _softmax(s * SCALE, np.asarray)
    out["no_mask"] = _tail(x, P, np.transpose(p0 @ np.repeat(
        (n1 @ g("Wv").T + g("bv")).reshape(T, KV, HD), KVR, 1).transpose(1, 0, 2),
        (1, 0, 2)).reshape(T, D), np.asarray)
    # (2) no RoPE at all (tables ignored)
    q3 = (n1 @ g("Wq").T + g("bq")).reshape(T, H, HD)
    k3 = (n1 @ g("Wk").T + g("bk")).reshape(T, KV, HD)
    k14 = np.repeat(k3, KVR, 1)
    v14 = np.repeat((n1 @ g("Wv").T + g("bv")).reshape(T, KV, HD), KVR, 1)
    s2 = np.transpose(q3, (1, 0, 2)) @ np.transpose(np.transpose(k14, (1, 0, 2)), (0, 2, 1))
    p2 = _softmax(s2 * SCALE + causal_mask().astype(np.float64), np.asarray)
    out["no_rope"] = _tail(x, P, np.transpose(p2 @ np.transpose(v14, (1, 0, 2)),
                                              (1, 0, 2)).reshape(T, D), np.asarray)
    # (3) q/k/v bias dropped (config.json has no bias key -- the silent killer)
    qb = _rope(n1 @ g("Wq").T, *tables(H), store=np.asarray)
    kb = _rope(n1 @ g("Wk").T, *tables(KV), store=np.asarray)
    vb = (n1 @ g("Wv").T).reshape(T, KV, HD)
    sb = np.transpose(qb, (1, 0, 2)) @ np.transpose(
        np.transpose(np.repeat(kb, KVR, 1), (1, 0, 2)), (0, 2, 1))
    pb = _softmax(sb * SCALE + causal_mask().astype(np.float64), np.asarray)
    out["bias_dropped"] = _tail(x, P, np.transpose(
        pb @ np.transpose(np.repeat(vb, KVR, 1), (1, 0, 2)), (1, 0, 2)).reshape(T, D),
        np.asarray)
    # (4) residual 1 dropped (r1 replaced by the attention output)
    out["residual1_dropped"] = _tail(np.zeros_like(x), P, r1 - x, np.asarray)
    # (5) eps = 0 instead of 1e-6 (the toy-block convention)
    def noeps(xx, gam):
        return xx / np.sqrt((xx * xx).mean(-1, keepdims=True)) * gam
    n1e = noeps(x, g("g1"))
    q3e = _rope(n1e @ g("Wq").T + g("bq"), *tables(H), store=np.asarray)
    k3e = _rope(n1e @ g("Wk").T + g("bk"), *tables(KV), store=np.asarray)
    v14e = np.repeat((n1e @ g("Wv").T + g("bv")).reshape(T, KV, HD), KVR, 1)
    se = np.transpose(q3e, (1, 0, 2)) @ np.transpose(
        np.transpose(np.repeat(k3e, KVR, 1), (1, 0, 2)), (0, 2, 1))
    pe = _softmax(se * SCALE + causal_mask().astype(np.float64), np.asarray)
    r1e = x + np.transpose(pe @ np.transpose(v14e, (1, 0, 2)), (1, 0, 2)).reshape(T, D) @ g("Wo").T
    out["eps_zero"] = _tail(x, P, r1e - x, np.asarray, n2_override=(r1e, noeps))
    return ref, out


def _tail(x, P, attn, store, n2_override=None):
    g = lambda k: P[k].astype(np.float64)
    r1 = x + attn
    if n2_override is None:
        n2 = r1 / np.sqrt((r1 * r1).mean(-1, keepdims=True) + EPS) * g("g2")
    else:
        r1o, fn = n2_override
        n2 = fn(r1o, g("g2"))
    gate = n2 @ g("Wg").T
    y = (gate / (1 + np.exp(-gate))) * (n2 @ g("Wu").T)
    return r1 + y @ g("Wd").T


# ------------------------------------------------------------ flatbuffer builder
class G:
    """Accumulator with unique tensor names; buffer 0 is the activation placeholder."""

    def __init__(self):
        self.b = flatbuffers.Builder(1 << 20)
        self.buffers = [bo.buf_empty(self.b)]
        self.tensors, self.opcodes, self.ops = [], [], []
        self.idx = {}

    def _t(self, name, shape, ttype, buf):
        self.tensors.append(bo.tensor(self.b, name, list(shape), ttype, buf))
        self.idx[name] = len(self.tensors) - 1
        return self.idx[name]

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
        """SPLIT emits one SliceLayer *per output* (docs/OP-gqa-scores.md 4.1)."""
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
    sq = g.mul(x, name + "_sq", b=x)      # MUL(x,x): same tensor twice is accepted (OP-rmsnorm 2.2)
    mean = g.mean_last(sq, name + "_mean")
    eps = g.const(name + "_eps", np.full((T, 1), EPS, np.float16))
    m_eps = g.add(mean, eps, name + "_mean_eps", (T, 1))
    rsq = g.rsqrt(m_eps, name + "_rsqrt")
    sc = g.mul(x, name + "_scaled", b=rsq)
    return g.mul(sc, name + "_norm", b=g.const(name + "_gamma", np.asarray(gamma, np.float16)))


def build_graph(P, first="full", stage="out"):
    """first='full'|'attn_half'|'ffn_half'; stage = the full-chain tensor to stop at."""
    g = G()
    x = g.act("x", (T, D))
    desc = "decoder layer qwen2.5-0.5b T=%d fp16 14Q/2KV" % T

    if first == "ffn_half":
        # second half: the graph input is r1 (same [T,896] contract)
        n2 = rmsnorm_nodes(g, x, "n2", P["g2"])
        gate = g.fc(n2, "gate", P["Wg"], np.zeros(INTER, np.float16), INTER)
        up = g.fc(n2, "up", P["Wu"], np.zeros(INTER, np.float16), INTER)
        sig = g.logistic(gate, "sig", (T, INTER))
        s = g.mul(gate, "silu", b=sig, shape=(T, INTER))
        y = g.mul(s, "mlp_y", b=up, shape=(T, INTER))
        down = g.fc(y, "down", P["Wd"], np.zeros(D, np.float16), D)
        out = g.add(down, x, "out")
        return g.finish([x], [out], desc + " (ffn half)")

    n1 = rmsnorm_nodes(g, x, "n1", P["g1"])
    if stage == "n1":
        return g.finish([x], [n1], desc + " stage n1")

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
    if stage == "q3":
        return g.finish([x], [q3], desc + " stage q3")
    k3 = qkv(P["Wk"], P["bk"], KV, "k")
    v_fc = g.fc(n1, "v_fc", P["Wv"], P["bv"], KV * HD)
    v3 = g.reshape(v_fc, "v_3d", (T, KV, HD), KV * HD)

    # KV replication 2 -> 14 with one 14-input CONCAT each (docs/OP-gqa-scores.md 4.6:
    # nary14 compiles to 3 layers and is bit-exact on the board; the 3-CONCAT form costs 5)
    def kv14(t3, tag):
        k0, k1 = g.split(t3, tag + "_split2", 1, 2, (T, 1, HD))
        return g.concat([k0] * KVR + [k1] * KVR, tag + "14", 1, (T, H, HD))

    k14 = kv14(k3, "k")
    v14 = kv14(v3, "v")
    if stage == "k14":
        return g.finish([x], [k14], desc + " stage k14")
    if stage == "v14":
        return g.finish([x], [v14], desc + " stage v14")

    qt = g.transpose(q3, "qt", [1, 0, 2], (T, H, HD))
    kt = g.transpose(k14, "kt", [1, 0, 2], (T, H, HD))
    vt = g.transpose(v14, "vt", [1, 0, 2], (T, H, HD))
    scores = g.bmm(qt, kt, "scores", (H, T, T), adj_y=True)       # 3-D, 1 native layer
    if stage == "scores":
        return g.finish([x], [scores], desc + " stage scores")
    sc = g.mul(scores, "scaled", b=g.const("scale", np.full((H, T, T), SCALE, np.float16)),
               shape=(H, T, T))
    masked = g.add(sc, g.const("mask", causal_mask()), "masked", (H, T, T))
    p = g.softmax(masked, "probs", (H, T, T))
    if stage == "probs":
        return g.finish([x], [p], desc + " stage probs")
    a3 = g.bmm(p, vt, "a3", (H, T, HD))
    at = g.transpose(a3, "at", [1, 0, 2], (H, T, HD))
    attn = g.reshape(at, "attn", (T, D), H * HD)
    if stage == "attn":
        return g.finish([x], [attn], desc + " stage attn")
    o = g.fc(attn, "o", P["Wo"], np.zeros(D, np.float16), D)       # o_proj: no bias
    r1 = g.add(o, x, "r1")
    if stage == "r1" or first == "attn_half":
        return g.finish([x], [r1], desc + (" (attn half)" if first == "attn_half" else " stage r1"))

    n2 = rmsnorm_nodes(g, r1, "n2", P["g2"])
    if stage == "n2":
        return g.finish([x], [n2], desc + " stage n2")
    gate = g.fc(n2, "gate", P["Wg"], np.zeros(INTER, np.float16), INTER)
    up = g.fc(n2, "up", P["Wu"], np.zeros(INTER, np.float16), INTER)
    sig = g.logistic(gate, "sig", (T, INTER))
    s = g.mul(gate, "silu", b=sig, shape=(T, INTER))
    y = g.mul(s, "mlp_y", b=up, shape=(T, INTER))
    if stage == "mlp_y":
        return g.finish([x], [y], desc + " stage mlp_y")
    down = g.fc(y, "down", P["Wd"], np.zeros(D, np.float16), D)
    if stage == "down":
        return g.finish([x], [down], desc + " stage down")
    out = g.add(down, r1, "out")
    return g.finish([x], [out], desc)


# ------------------------------------------------------------------------ weights
def weight_inventory(P):
    rows, total = [], 0
    for name, key in [("q_proj", "Wq"), ("k_proj", "Wk"), ("v_proj", "Wv"),
                      ("o_proj", "Wo"), ("gate_proj", "Wg"), ("up_proj", "Wu"),
                      ("down_proj", "Wd")]:
        a = np.asarray(P[key], np.float16)
        b = int(a.nbytes)
        total += b
        rows.append({"tensor": name, "fixture": "layer%d.%s" % (
            LAYER, "mlp." + name if key in ("Wg", "Wu", "Wd") else "attn." + name),
            "shape": list(a.shape), "bytes": b, "mib": round(b / 2 ** 20, 4)})
    small = 0
    for key in ("g1", "g2", "bq", "bk", "bv"):
        small += int(np.asarray(P[key], np.float16).nbytes)
    zeros = int((D + 2 * INTER + D) * 2)
    small += zeros
    tables_bytes = int(2 * (T * H * HALF + T * KV * HALF) * 2 * 2)
    mask_bytes = int(H * T * T * 2)
    rows.append({"tensor": "norm gammas + qkv biases + zero FC biases",
                 "bytes": small, "mib": round(small / 2 ** 20, 4)})
    rows.append({"tensor": "rope cos/sin tables (q+k) + causal mask",
                 "bytes": tables_bytes + mask_bytes,
                 "mib": round((tables_bytes + mask_bytes) / 2 ** 20, 4)})
    return rows, total + small + tables_bytes + mask_bytes


# ------------------------------------------------------------------ layer budget
PREDICTED_LAYERS = [
    ("RMSNorm #1 (input_layernorm)", 6, "MUL, MEAN, ADD(eps), RSQRT, MUL(x), MUL(gamma)"),
    ("q projection + RoPE", 11, "FC, RESHAPE, SPLIT(2 Slice), MUL x4, SUB, ADD, CONCAT"),
    ("k projection + RoPE", 11, "same as q with H=2"),
    ("v projection", 2, "FC, RESHAPE (no RoPE)"),
    ("KV replication k (2->14)", 3, "SPLIT(2 Slice) + one 14-input CONCAT"),
    ("KV replication v (2->14)", 3, "SPLIT(2 Slice) + one 14-input CONCAT"),
    ("TRANSPOSE q, k14, v14", 3, "one per tensor [T,H,HD]->[H,T,HD]"),
    ("scores BMM (3-D, adj_y)", 1, "one native BatchMatMul"),
    ("MUL(1/8) + ADD(mask) + SoftMax", 3, "3 native layers (docs/OP-masked-softmax.md)"),
    ("PV BMM (3-D)", 1, "one native BatchMatMul"),
    ("TRANSPOSE + RESHAPE back to [8,896]", 2, ""),
    ("o_proj FC (no bias) + residual ADD", 2, ""),
    ("RMSNorm #2 (post_attention_layernorm)", 6, "same 6-layer chain"),
    ("SwiGLU FFN", 6, "FC, FC, LOGISTIC, MUL, MUL, FC (silu is not fused)"),
    ("residual ADD #2", 1, ""),
]


# ------------------------------------------------------------------------ traces
def quantile_stats(a):
    a = np.asarray(a, np.float64)
    return {"min": float(a.min()), "max": float(a.max()), "abs_max": float(np.abs(a).max()),
            "rms": float(np.sqrt((a ** 2).mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output dir (default ops/decoder_layer_t<T>_fp16)")
    ap.add_argument("--T", type=int, default=8,
                    help="sequence length (static shape; RoPE cos/sin tables are data)")
    ap.add_argument("--variant", choices=["full", "attn_half", "ffn_half"], default="full")
    ap.add_argument("--stage", choices=STAGES, default="out")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--execcheck", action="store_true",
                    help="run the built .tflite in numpy (wiring check) and compare to golden")
    ap.add_argument("--predict", action="store_true", help="print the layer budget only")
    a = ap.parse_args()
    configure(a.T)
    if a.out is None:
        a.out = DEFAULT_OUT

    if a.predict:
        tot = sum(n for _, n, _ in PREDICTED_LAYERS)
        for name, n, note in PREDICTED_LAYERS:
            print("  %-40s %2d  %s" % (name, n, note))
        print("  %-40s %2d" % ("TOTAL predicted native layers", tot))
        return 0

    if a.diag:
        return diag(a.out)
    if a.analyze:
        return analyze(a.out)
    if a.execcheck:
        return execcheck(a.out)

    P = fixture()
    x16 = input_hidden()
    stage = a.stage if a.variant == "full" else ("r1" if a.variant == "attn_half" else "out")

    # ---- the mask constant is a function of THIS layer's score range (ticket 05 11.2) ----
    # S = max|scaled score| is a property of q/k (bias-dominated), not of the mask, so it can
    # be measured before the mask is chosen: the "scores" stage stops right after the BMM.
    if a.variant == "ffn_half" or stage == "n1":
        S_scaled, mstatus, mlo, mhi = 0.0, "n/a (no attention in this graph)", 0.0, 0.0
    else:
        s_raw = float(np.abs(np.asarray(
            forward(x16, P, np.float32, False, "scores")["scores"], np.float64)).max())
        S_scaled = s_raw * SCALE
        MASK, mstatus, mlo, mhi = derive_mask(S_scaled)
        globals()["MASK"] = MASK
        print("mask: S=max|scaled score|=%.2f -> M=%.0f (%s), interval [%.1f, %.1f]"
              % (S_scaled, MASK, mstatus, mlo, mhi))

    if a.variant == "ffn_half":
        # the graph input is r1 (stored fp16), exactly what the first half would hand over
        ref_full = forward(x16, P, np.float32, False, "r1")["r1"]
        r1_16 = np.asarray(ref_full, np.float16)
        x_in = r1_16
        exp = np.asarray(forward_ffn(r1_16, P, np.float32, False), np.float16)
        sim = np.asarray(forward_ffn(r1_16, P, np.float32, True), np.float16)
        g64 = forward_ffn(r1_16, P, np.float64, False)
    else:
        x_in = x16
        full = forward(x16, P, np.float32, False, stage)
        exp = np.asarray(full[stage], np.float16)
        sim = np.asarray(forward(x16, P, np.float32, True, stage)[stage], np.float16)
        g64 = forward(x16, P, np.float64, False, stage)[stage]

    blob = build_graph(P, first=a.variant, stage=stage)

    mx = float(np.abs(exp.astype(np.float64)).max())
    ulp = float(np.spacing(np.float16(mx)))
    floor_abs = float(np.abs(sim.astype(np.float64) - exp.astype(np.float64)).max())
    floor_rel = floor_abs / mx
    g64_codes = int((np.asarray(g64, np.float16) == exp).sum())
    f32_vs_f64 = float(np.abs(np.asarray(g64, np.float64) - exp.astype(np.float64)).max())

    # ---- tolerance, registered BEFORE the board run (NOTES 4 / 10.1) --------------
    ULP_REL = 2 ** -11
    atol = max(4.0 * ULP_REL * mx, 4.0 * floor_abs)
    rtol = atol
    basis = (
        "max|out| = %.4f; fp16 ULP at that magnitude = %.4g (%.2e relative). "
        "Device-independent floor (numpy chain with a fp16 rounding at every tensor "
        "boundary, i.e. what MDLA does -- docs/OP-qkv.md 5.2: 7167/7168 bit-identical) "
        "= %.4e abs = %.2f ULP = %.2e relative. Registered atol = rtol = "
        "max(4*2^-11*max|out|, 4*floor) = %.6g. The %s term dominates. "
        "NOTES 4's flat 2e-2 would be %.2e relative here, i.e. %s."
        % (mx, ulp, ulp / mx, floor_abs, floor_abs / ulp, floor_rel, atol,
           "4*floor" if 4 * floor_abs > 4 * ULP_REL * mx else "4*2^-11*max|out|",
           2e-2 / mx,
           "looser than the measured floor (usable but weak)" if 2e-2 > 4 * floor_abs
           else "tighter than one fp16 ULP (unusable)"))

    trace = {}
    for k, v in (forward(x16, P, np.float32, False, "out") if a.variant == "full"
                 else {"out": exp}).items():
        trace[k] = quantile_stats(np.asarray(v, np.float64))

    wrows, wtotal = weight_inventory(P)
    mask_ok = S_scaled

    os.makedirs(os.path.join(a.out, "golden"), exist_ok=True)
    with open(os.path.join(a.out, "model.tflite"), "wb") as f:
        f.write(blob)
    np.ascontiguousarray(x_in, np.float16).tofile(os.path.join(a.out, "golden", "input.bin"))
    np.ascontiguousarray(exp, np.float16).tofile(os.path.join(a.out, "golden", "expected.bin"))

    meta = {
        "op": "DECODER_LAYER_T%d_FP16" % T,
        "variant": a.variant,
        "stage": stage,
        "shape": {"T": T, "D": D, "Hq": H, "Hkv": KV, "HD": HD, "I": INTER,
                  "in": [T, D], "out": [T, D]},
        "dtype": "fp16",
        "inputs": [{"name": "x", "shape": [T, D], "type": "float16",
                    "bytes": int(np.asarray(x_in).nbytes),
                    "what": "layer input hidden state (embed rows for fixed token ids)"
                            if a.variant != "ffn_half" else "r1 (first-half output, fp16)"}],
        "output": {"name": "out", "shape": list(np.asarray(exp).shape), "type": "float16",
                   "bytes": int(np.asarray(exp).nbytes)},
        "quant": {},
        "reference": "numpy_float32_from_stored_fp16_one_final_rounding",
        "reference_note": "golden/expected.bin is an independent numpy float32 forward pass "
                          "over the stored fp16 weights/inputs/tables; it shares no code with "
                          "the flatbuffer writer (which contains no maths).  A float64 rerun "
                          "gives %d/%d identical fp16 codes (max|f32-f64| = %.3e), so the "
                          "reference is not precision-limited."
                          % (g64_codes, np.asarray(exp).size, f32_vs_f64),
        "atol": round(atol, 10),
        "rtol": round(rtol, 10),
        "tolerance_basis": basis,
        "tolerance_revision": "registered before the first board run of this op; derived only "
                              "from the frozen fixture + a device-independent fp16 rounding "
                              "simulation (NOTES 4 / 10.1).  Not adjusted after any run.",
        "fp16_floor": {"abs": floor_abs, "rel": floor_rel, "ulp_at_max": ulp,
                       "max_abs_out": mx, "sim_vs_golden_identical_codes":
                       int((sim == exp).sum()), "codes": int(np.asarray(exp).size),
                       "note": "numpy chain with a fp16 rounding at every tensor boundary vs "
                               "the single-rounding golden; device-independent"},
        "threshold_at_max": float(atol + rtol * mx),
        "eps_register": {"graph_eps": EPS,
                         "eps_stored_fp16": float(np.float16(EPS)),
                         "eps_shape": [T, 1],
                         "toy_block_eps": 0,
                         "note": "Qwen2.5 rms_norm_eps = 1e-6 (transformers), NOT the earlier "
                                 "toy block's 0 (docs/OP-block.md).  The mean(x^2) of a real "
                                 "hidden is ~1e-4..1, so eps moves the output by << 1 ULP: the "
                                 "two conventions are numerically indistinguishable here and "
                                 "are NOT discriminated by this comparison.  The graph carries "
                                 "the real 1e-6 anyway (it is what the model means)."},
        "mask": {"value": MASK, "position": "ADD after MUL(1/sqrt(HD))",
                 "causal": "0 at j<=i, MASK at j>i", "form": "const [H,T,T] baked in graph",
                 "max_abs_scaled_score_S": mask_ok,
                 "S_limit": (abs(MASK) - 104.0) / 2.0,
                 "S_ok": bool(mlo <= abs(MASK) <= mhi) if mlo else True,
                 "derivation": "M in [2S+104, 65504-S] (docs/OP-masked-softmax.md 3.4); "
                               "M=%d = 2^%d is exact in fp16"
                               % (int(abs(MASK)), int(round(np.log2(abs(MASK))))),
                 "note": "re-measured for THIS layer before reuse, as ticket 05 11.2 requires"},
        "rope": {"convention": "qwen2_split_half", "theta": THETA, "head_dim": HD,
                 "positions": "%d..%d" % (POS0, POS0 + T - 1),
                 "table": "const cos/sin fp16, duplicated over heads (no broadcast)",
                 "static_note": "T must be static because the tables are data"},
        "kv_replication": {"from": KV, "to": H, "mapping": "query head h -> kv head h//7",
                           "form": "SPLIT(axis=1) + one 14-input CONCAT per tensor",
                           "layers": 3,
                           "why": "docs/OP-gqa-scores.md 4.6: nary14 compiles to 3 layers and "
                                  "is bit-exact on the board; the 3-CONCAT form costs 5"},
        "predicted_layers": {"items": [{"name": n, "layers": c, "note": s}
                                       for n, c, s in PREDICTED_LAYERS],
                             "total": sum(n for _, n, _ in PREDICTED_LAYERS),
                             "note": "counted from the component tickets' measured layer counts; "
                                     "the briefing's 40-50 estimate predates the nary14 CONCAT "
                                     "measurement and the 11-layer q/k RoPE"},
        "weight_inventory": {"rows": wrows, "total_bytes": wtotal,
                             "total_mib": round(wtotal / 2 ** 20, 4)},
        "stats": {"input": quantile_stats(x_in), "output": quantile_stats(exp),
                  "traces": trace},
        "generator": "tools/build_decoder_layer.py",
        "tflite": "model.tflite",
        "tflite_bytes": len(blob),
    }
    if mstatus != "reused":
        # only present when the constant had to move: keeps the T=8 product byte-identical
        meta["mask"]["interval"] = [mlo, mhi]
        meta["mask"]["selection"] = mstatus
    if T != 8:
        meta["input_tokens"] = {
            "ids": TOKEN_IDS,
            "rule": "positions 0-7 = ticket 03/06/09's list verbatim; the rest follow "
                    "id(i) = (41 + 1103*i) %% vocab_size, i = 0,1,2,... (stride coprime to the "
                    "151936-entry vocabulary, so all %d ids are distinct rows of embed.weight). "
                    "Emitted only for T != 8 so the T=8 product stays byte-identical." % T,
        }

    with open(os.path.join(a.out, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("%s: %s/%s  in%s out%s  tflite=%dB  max|out|=%.4f (ULP %.3g)  floor=%.3e "
          "(%.2f ULP)  atol=rtol=%.6g  weights=%.2f MiB"
          % (a.out, a.variant, stage, list(np.asarray(x_in).shape), list(np.asarray(exp).shape),
             len(blob), mx, ulp, floor_abs, floor_abs / ulp, atol, wtotal / 2 ** 20))
    return 0


# ------------------------------------------------------- device-free graph checker
import tflite as _tfl
from tflite import BuiltinOperator as _BO          # opcode -> name, no hand-written table

_OPNAME = {v: k for k, v in vars(_BO).items() if isinstance(v, int)}
_OPTNAME = {v: k for k, v in vars(_tfl.BuiltinOptions).items() if isinstance(v, int)}


def _opts(op):
    """Rebind the builtin-options bytes onto the generated table class (typed access)."""
    raw = op.BuiltinOptions()
    if raw is None:
        return None
    cls = getattr(_tfl, _OPTNAME.get(op.BuiltinOptionsType(), ""), None)
    if cls is None:
        return None
    t = cls()
    t.Init(raw.Bytes, raw.Pos)
    return t


def _tinfo(t):
    q = t.Quantization()
    return (t.Name().decode() if t.Name() else "", list(t.ShapeAsNumpy()),
            np.asarray(q.ScaleAsNumpy(), np.float64) if q is not None else np.array([]))


def exec_graph(path, feed, trace=False):
    """Evaluate the .tflite graph in numpy, straight from the flatbuffer.

    This is a *wiring* check, not a numerical one: it reads the operator list, tensor
    shapes and constant buffers out of the file and applies standard TFLite semantics.
    It therefore catches the whole class of "compiles fine, wrong maths" bugs that
    NOTES 10.2 / C4 warn about -- wrong opcode index, SPLIT input order, CONCAT input
    order, a MUL wired to a constant instead of the second operand, missing TRANSPOSE,
    reshape in the wrong order -- before spending a board cycle on them.
    """
    from tflite import Model
    buf = open(path, "rb").read()
    m = Model.GetRootAsModel(buf, 0)
    sg = m.Subgraphs(0)
    tensors, buffers = [], []
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        name, shape, _ = _tinfo(t)
        b = m.Buffers(t.Buffer())
        data = None
        if b is not None and b.DataLength():
            raw = bytes(b.DataAsNumpy())
            dt = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8,
                  9: np.int8, 7: np.int64}.get(t.Type())
            data = np.frombuffer(raw, dtype=dt).reshape(shape).copy()
        tensors.append({"name": name, "shape": shape, "data": data})
    codes = [m.OperatorCodes(i).BuiltinCode() for i in range(m.OperatorCodesLength())]
    vals = {}
    for i in range(sg.InputsLength()):
        ti = sg.Inputs(i)
        vals[ti] = np.asarray(feed, np.float32).reshape(tensors[ti]["shape"])
    out_idx = [sg.Outputs(i) for i in range(sg.OutputsLength())]

    def get(i):
        a = tensors[i]["data"]
        return a if a is not None else vals[i]

    def put(i, a):
        vals[i] = np.asarray(a, np.float32)
        if trace:
            print("  %-28s %-16s max=%.5g rms=%.5g" % (
                tensors[i]["name"], tensors[i]["shape"], np.abs(vals[i]).max(),
                float(np.sqrt((vals[i].astype(np.float64) ** 2).mean()))))

    def opt(op):
        return _opts(op)

    for k in range(sg.OperatorsLength()):
        op = sg.Operators(k)
        code = codes[op.OpcodeIndex()]
        ins = [op.Inputs(j) for j in range(op.InputsLength())]
        outs = [op.Outputs(j) for j in range(op.OutputsLength())]
        nm = _OPNAME.get(code, "code%d" % code)
        o = opt(op)
        if nm in ("MUL", "ADD", "SUB"):
            a, b = get(ins[0]), get(ins[1])
            put(outs[0], {"MUL": a * b, "ADD": a + b, "SUB": a - b}[nm])
        elif nm == "MEAN":
            ax = tuple(int(x) for x in np.atleast_1d(get(ins[1])))
            put(outs[0], get(ins[0]).mean(axis=ax, keepdims=bool(o.KeepDims())))
        elif nm == "RSQRT":
            put(outs[0], 1.0 / np.sqrt(get(ins[0])))
        elif nm == "LOGISTIC":
            put(outs[0], 1.0 / (1.0 + np.exp(-get(ins[0]).astype(np.float64))))
        elif nm == "SOFTMAX":
            v = get(ins[0]).astype(np.float64) * float(o.Beta())
            v = v - v.max(axis=-1, keepdims=True)
            e = np.exp(v)
            put(outs[0], e / e.sum(axis=-1, keepdims=True))
        elif nm == "FULLY_CONNECTED":
            x, W, b = get(ins[0]), get(ins[1]), get(ins[2])
            shp = x.shape
            put(outs[0], (x.reshape(-1, shp[-1]) @ W.T + b).reshape(shp[:-1] + (W.shape[0],)))
        elif nm == "RESHAPE":
            put(outs[0], get(ins[0]).reshape(list(o.NewShapeAsNumpy())))
        elif nm == "TRANSPOSE":
            put(outs[0], np.transpose(get(ins[0]), tuple(int(p) for p in get(ins[1]))))
        elif nm == "SPLIT":
            ax = int(np.atleast_1d(get(ins[0]))[0])
            parts = np.split(get(ins[1]), len(outs), axis=ax)
            for ti, part in zip(outs, parts):
                put(ti, part)
        elif nm == "CONCATENATION":
            put(outs[0], np.concatenate([get(i) for i in ins], axis=int(o.Axis())))
        elif nm == "BATCH_MATMUL":
            a, b = get(ins[0]), get(ins[1])
            if o.AdjX():
                a = np.swapaxes(a, -1, -2)
            if o.AdjY():
                b = np.swapaxes(b, -1, -2)
            put(outs[0], a @ b)
        else:
            raise SystemExit("exec_graph: unsupported op %s (code %d)" % (nm, code))
    return [vals[i] for i in out_idx]


# -------------------------------------------------------------------- --analyze
def _configure_from_meta(meta):
    """Bind T + MASK to what this op directory was actually built with."""
    global MASK
    configure(int(meta["shape"]["T"]))
    MASK = float(meta.get("mask", {}).get("value", MASK_REF))
    return MASK


def analyze(op_dir):
    """Per-stage magnitudes / fp16 floor / reference precision.  Device-free, no board needed."""
    meta = json.load(open(os.path.join(op_dir, "model.json")))
    _configure_from_meta(meta)
    P = fixture()
    x16 = input_hidden()
    full = forward(x16, P, np.float32, False, "out")
    sim = forward(x16, P, np.float32, True, "out")
    g64 = forward(x16, P, np.float64, False, "out")
    print("%-8s %12s %10s   %-28s %s" % ("stage", "max|v|", "rms", "floor sim-vs-golden",
                                         "golden f32 vs f64 (fp16 codes)"))
    for k in ["x", "n1", "q3", "k14", "v14", "scores", "probs", "attn", "r1", "n2",
              "mlp_y", "down", "out"]:
        a64 = np.asarray(full[k], np.float16)
        a = a64.astype(np.float64)
        s = np.asarray(sim[k], np.float16).astype(np.float64)
        g = np.asarray(g64[k], np.float16)
        mx = np.abs(a).max()
        ulp = float(np.spacing(np.float16(mx)))
        print("%-8s %12.5g %10.5g   abs=%.3e (%5.2f ULP of this stage)   diff=%.2e  same=%d/%d"
              % (k, mx, np.sqrt((a * a).mean()), np.abs(s - a).max(), np.abs(s - a).max() / ulp,
                 np.abs(g.astype(np.float64) - a).max(),
                 int((g == a64).sum()), a.size))
    sc = np.asarray(full["scores"], np.float64)
    S = float(np.abs(sc).max() / 8.0)
    lim = (abs(MASK) - 104.0) / 2.0
    print("mask check: max|scores|=%.2f  S=max|scaled|=%.2f  M=%.0f  limit S<=%.0f -> %s"
          % (np.abs(sc).max(), S, MASK, lim, "OK" if S <= lim else "MUST RE-DERIVE M"))
    p = np.asarray(full["probs"], np.float64)
    m = causal_mask()
    print("probs     : max=%.6f  frac>0.99=%.3f  masked positions max=%.3e  n_masked=%d"
          % (p.max(), float((p > 0.99).mean()), float(p[m == np.float16(MASK)].max()),
             int((m == np.float16(MASK)).sum())))
    # mask saturation: -4096 vs -8192 must give the same fp16 probabilities
    v = np.asarray(full["scores"], np.float64) * SCALE
    p2 = _softmax(v + np.where(causal_mask().astype(np.float64) == 0.0, 0.0, 2.0 * MASK),
                         np.asarray)
    print("mask sat  : %.0f vs %.0f fp16 probability codes differing = %d/%d"
          % (MASK, 2.0 * MASK,
             int((np.asarray(p, np.float16) != np.asarray(p2, np.float16)).sum()), p.size))
    inv = np.asarray(full["out"], np.float64)
    print("out       : min=%.4f max=%.4f absmax=%.4f  (tolerance in model.json: atol=rtol=%g)"
          % (inv.min(), inv.max(), np.abs(inv).max(), meta["atol"]))
    wrows = meta["weight_inventory"]["rows"]
    print("weights   : %.2f MiB static (fp16)" % meta["weight_inventory"]["total_mib"])
    for r in wrows:
        print("  %-42s %10d B %8.4f MiB %s"
              % (r["tensor"], r["bytes"], r["mib"], r.get("shape", "")))
    return 0


# -------------------------------------------------------------------- --execcheck
def execcheck(op_dir):
    """Evaluate the .tflite in numpy and compare with the registered golden.

    Device-free, no ncc, no board.  Any wiring bug (see exec_graph) shows up here as a
    large discrepancy, which is exactly the failure class ticket 09 must avoid burning
    board time on (NOTES 10.2 / spec C4).
    """
    meta = json.load(open(os.path.join(op_dir, "model.json")))
    _configure_from_meta(meta)
    x = np.fromfile(os.path.join(op_dir, "golden", "input.bin"), dtype=np.float16)
    exp = np.fromfile(os.path.join(op_dir, "golden", "expected.bin"), dtype=np.float16)
    shape = tuple(meta["inputs"][0]["shape"])
    got = exec_graph(os.path.join(op_dir, "model.tflite"), x.reshape(shape).astype(np.float32))
    if len(got) != 1:
        raise SystemExit("expected exactly one graph output, got %d" % len(got))
    g = np.asarray(got[0], np.float16).reshape(-1)
    e = exp.reshape(-1)
    if g.size != e.size:
        print("FAIL size: graph %s vs golden %s" % (g.shape, e.shape))
        return 1
    mx = float(np.abs(e.astype(np.float64)).max())
    d = np.abs(g.astype(np.float64) - e.astype(np.float64))
    tol = 4 * float(np.spacing(np.float16(mx)))
    print("numpy execution vs golden: max_abs=%.4e (%.2f ULP of max|out|, %.2f ULP of the "
          "fp16 floor idea)  rmse=%.3e  rel_l2=%.3e  codes_same=%d/%d -> %s"
          % (d.max(), d.max() / float(np.spacing(np.float16(mx))),
             d.max() / tol, float(np.sqrt((d ** 2).mean())),
             float(np.linalg.norm(d) / np.linalg.norm(e.astype(np.float64))),
             int((g == e).sum()), e.size,
             "WIRING OK (fp16 rounding level)" if d.max() <= tol else "WIRING MISMATCH"))
    return 0 if d.max() <= tol else 1


# ------------------------------------------------------------------------ --diag
def diag(op_dir):
    meta = json.load(open(os.path.join(op_dir, "model.json")))
    exp = np.fromfile(os.path.join(op_dir, "golden", "expected.bin"), dtype=np.float16)
    dev_path = os.path.join(op_dir, "out_device.bin")
    if not os.path.exists(dev_path):
        raise SystemExit("no out_device.bin in %s (run ./tools/e2e.sh first)" % op_dir)
    dev = np.fromfile(dev_path, dtype=np.float16)
    if dev.size != exp.size:
        print("FAIL size mismatch device=%d expected=%d" % (dev.size, exp.size))
        return 1
    _configure_from_meta(meta)
    P = fixture()
    x16 = input_hidden()
    mx = float(np.abs(exp.astype(np.float64)).max())
    ulp = float(np.spacing(np.float16(mx)))
    atol, rtol = float(meta["atol"]), float(meta["rtol"])
    thr = atol + rtol * mx

    d = np.abs(dev.astype(np.float64) - exp.astype(np.float64))
    ok = bool(np.all(d <= atol + rtol * np.abs(exp.astype(np.float64))))
    print("registered tolerance : atol=%.6g rtol=%.6g  threshold@max=%.6g" % (atol, rtol, thr))
    print("device vs golden     : max_abs=%.4e (%.2f ULP, rel=%.3e)  rmse=%.3e  rel_l2=%.3e "
          "-> %s" % (d.max(), d.max() / ulp, d.max() / mx,
                     float(np.sqrt((d ** 2).mean())),
                     float(np.linalg.norm(d) / np.linalg.norm(exp.astype(np.float64))),
                     "PASS" if ok else "FAIL"))
    print("bit-identical codes  : %d/%d" % (int((dev == exp).sum()), dev.size))
    r1 = os.path.join(op_dir, "out_r1.bin")
    r2 = os.path.join(op_dir, "out_r2.bin")
    if os.path.exists(r1) and os.path.exists(r2):
        a1 = np.fromfile(r1, dtype=np.float16)
        a2 = np.fromfile(r2, dtype=np.float16)
        nf = float(np.abs(a1.astype(np.float64) - a2.astype(np.float64)).max())
        print("noise floor          : max|r1-r2| = %.4e  (%d/%d codes differing)"
              % (nf, int((a1 != a2).sum()), a1.size))
    var = meta.get("variant", "full")
    if var != "full" or meta.get("stage") == "out":
        if var == "ffn_half":
            r1_16 = np.asarray(np.fromfile(os.path.join(op_dir, "golden", "input.bin"),
                                           dtype=np.float16)).reshape(T, D)
            sim = np.asarray(forward_ffn(r1_16, P, np.float32, True),
                             np.float16).reshape(-1)
        elif var == "attn_half":
            sim = np.asarray(forward(x16, P, np.float32, True, "r1")["r1"],
                             np.float16).reshape(-1)
        else:
            sim = np.asarray(forward(x16, P, np.float32, True, "out")["out"],
                             np.float16).reshape(-1)
        ds = np.abs(dev.astype(np.float64) - sim.astype(np.float64))
        print("device vs fp16 sim   : max_abs=%.4e (%.2f ULP)  identical=%d/%d"
              % (ds.max(), ds.max() / ulp, int((dev == sim).sum()), dev.size))
    if var == "full" and meta.get("stage") == "out":
        ref, cfs = cf_variants(x16, P)
        print("device vs counterfactuals (must be >> threshold):")
        for name, cf in cfs.items():
            dd = np.abs(dev.astype(np.float64) - np.asarray(cf, np.float64).reshape(-1))
            print("  %-22s max_abs=%.4e  (%7.2fx thr)" % (name, dd.max(), dd.max() / thr))
        print("device vs exact f64  : max_abs=%.4e"
              % np.abs(dev.astype(np.float64) - np.asarray(ref, np.float64).reshape(-1)).max())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
