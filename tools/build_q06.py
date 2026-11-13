#!/usr/bin/env python3
"""Ticket 06 -- prefill + decode graphs over ONE shared-weight bin, with a KV cache.

Ticket 05 left three constraints for this ticket (docs/OP-24layer-shared.md section 4.3):

  1. mask / RoPE tables must become GRAPH INPUTS, otherwise the T^2 mask constant lands in
     the extracted bin and the bin is tied to T (ticket 05's T=128 bin was 10.2 MB bigger
     for exactly that reason), so a prefill graph and a decode graph could never share one.
  2. ncc merges byte-identical static tensors, so "Static 15M" for 24 layers is an artefact
     of the fixture repeating layer 0; a real 24-layer stack is ~366 MB.
  3. per-tensor int16 gets pulled apart at T=128 (single layer 0.99658).  This ticket stays
     at T=16 (prefill) / T=1 (decode) and does not re-open that question.

This builder owns the second half of the seam: it builds two graphs of the same 24-layer
tower -- one prefill (T rows, no cache) and one decode (T query rows over a C-slot KV cache)
-- with every tensor scale taken from ONE frozen table, so that

  * the two graphs' weight tensors AND int32 biases AND gamma constants are byte-identical
    (they only depend on the frozen per-tensor activation scales, never on the data), i.e.
    extract-shared can move 100% of the statics into a single bin;
  * every golden step uses the same scales the device graph was compiled with, so
    "device == quantised golden" is a statement about the device, not about two different
    quantisers.

Freezing the scales is what makes the sequence test possible at all: a per-step numpy
rebuild of the same op sequence (the QL objects below recompute codes as they emit ops, so
the golden cannot drift from the graph) needs scales that do NOT move with the data.

Layout of the cache (ticket: TFLite is functional, so a KV write can only be an output):

    inputs   k_in [L*KV, C, 64] int16   v_in [L*KV, C, 64] int16     (L layers x KV heads)
    outputs  k_out [L*KV, T, 64]        v_out [L*KV, T, 64]          (the NEW rows only)

The host owns the cache: after each Run it writes the returned row into slot m (left
aligned, O(1) per token) and passes the whole [48, C, 64] buffer back next Run.  That
round trip is the per-token host copy cost the ticket asks to measure.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_w4a8 as bw

D, H, KV, HD, HALF, INTER, KVR = (bw.D, bw.H, bw.KV, bw.HD, bw.HALF, bw.INTER, bw.KVR)
EPS = bw.EPS
MASK_REAL = float(bw.MASK)                 # -4096, same value ticket 05 used
S_MASK = abs(MASK_REAL) / 32767.0          # mask input scale: code -32767 == -4096
S_TAB = 1.0 / 32767.0                      # RoPE table input scale (cos/sin in [-1, 1])
QMAX = 32767

# frozen global scales, filled by calibration (see calibrate())
S_K = None                                 # KV-cache scale for K rows
S_V = None                                 # KV-cache scale for V rows


# ------------------------------------------------------------------ scale table
class Ctx:
    """build_w4a8.QL plus a FROZEN output-scale table.

    learn=True  : the calibration pass, scale := pick_scale(this data), recorded in tab.
    learn=False : every op reads its output scale from tab; a missing tag is a hard error
                  (it would mean the build and the golden disagree about the graph).
    """

    def __init__(self, tab=None, learn=False, emit=True):
        self.q = bw.QL() if emit else None
        self.tab = {} if tab is None else tab
        self.learn = learn
        self.sat = {}

    def sc(self, key, real):
        if self.learn:
            self.tab[key] = bw.pick_scale(real)
            return self.tab[key]
        if key not in self.tab:
            raise SystemExit("scale %r was never frozen -- calibration did not reach it"
                             % key)
        return self.tab[key]

    def _sat(self, key, ts):
        n = int((np.abs(ts.c.astype(np.int64)) >= QMAX).sum())
        if n:
            self.sat[key] = self.sat.get(key, 0) + n
        return ts

    # ---- inputs
    def mk_in(self, name, shape, codes, s, zp=0):
        a = np.ascontiguousarray(np.asarray(codes, np.int16))
        return self.q.mk_in(name, shape, a, s, zp)

    # ---- ops with freezable output scales
    def fc(self, a, W, b, n, tag, so=None):
        w, sw, bq, _s, o_ref = bw.q_fc_tables(a.c, a.s, np.asarray(W, np.float64),
                                              np.asarray(b, np.float64))
        so = self.sc(tag + ".o", o_ref) if so is None else so
        return self._sat(tag, self.q.q_fc(a, w, sw, bq, so, n, tag))

    def mul(self, a, b, tag, so=None):
        so = self.sc(tag + ".o", a.real * b.real) if so is None else so
        return self._sat(tag, self.q.q_mul(a, b, so, tag))

    def add(self, a, b, tag, so=None, minus=False):
        so = self.sc(tag + ".o", a.real - b.real if minus else a.real + b.real) \
            if so is None else so
        return self._sat(tag, self.q.q_addsub(a, b, so, tag, minus=minus))

    def bmm(self, a, b, tag, so=None, adj_y=False):
        av = a.c.astype(np.int64) - a.zp
        bv = b.c.astype(np.int64) - b.zp
        acc = av @ np.swapaxes(bv, -1, -2) if adj_y else av @ bv
        so = self.sc(tag + ".o", acc * a.s * b.s) if so is None else so
        return self._sat(tag, self.q.q_bmm(a, b, so, tag, adj_y=adj_y))

    def softmax(self, a, tag):
        return self.q.q_softmax(a, tag)

    def logistic(self, a, tag):
        return self.q.q_logistic(a, tag)

    def mean(self, a, tag):
        return self._sat(tag, self.q.q_mean(a, self.sc(tag + ".o",
                                                       a.real.mean(axis=1, keepdims=True)),
                                           tag))

    def rsqrt(self, a, tag):
        real = 1.0 / np.sqrt(np.maximum(a.real, 1e-30))
        return self._sat(tag, self.q.q_rsqrt(a, self.sc(tag + ".o", real), tag))

    def reshape(self, a, shape, tag):
        return self.q.q_shape(a, tag, shape)

    def transpose(self, a, perm, tag):
        return self.q.q_transpose(a, tag, perm)

    def split(self, a, axis, num, shape, tag):
        return self.q.q_split(a, tag, axis, num, shape)

    def concat(self, ins, axis, shape, tag, so=None):
        return self._sat(tag, self.q.q_concat(ins, tag, axis, shape, so=so))


# --------------------------------------------------------------------- components
def rmsnorm5(ctx, a, gamma, tag):
    """int16 RMSNorm, 5 layers -- same sequence as build_w4a8._rmsnorm5, but every output
    scale comes from the frozen table (the original picks them from the data)."""
    sq = ctx.mul(a, a, tag + "_sq")
    mean = ctx.mean(sq, tag + "_mean")
    rs = ctx.rsqrt(mean, tag + "_rsqrt")
    sc = ctx.mul(a, rs, tag + "_scaled")
    gc, gs = bw.s8(np.asarray(gamma, np.float64))
    ty = bw.TS(ctx.q.const_i8(tag + "_gamma", gc, gs), gc, gs)
    return ctx.mul(sc, ty, tag + "_norm")


def rope_pair(ctx, y3, heads, rope, tag, s_out=None):
    """RoPE split-half with the cos/sin tables as GRAPH INPUTS (ticket 05 section 4.3)."""
    t = y3.c.shape[0]
    lo, hi = ctx.split(y3, 2, 2, (t, heads, HALF), tag + "_rs")
    c, s = rope["cos%d" % heads], rope["sin%d" % heads]
    p1 = ctx.mul(lo, c, tag + "_ac")
    p2 = ctx.mul(hi, s, tag + "_bs")
    p3 = ctx.mul(hi, c, tag + "_bc")
    p4 = ctx.mul(lo, s, tag + "_as")
    lo_r = lo.real * c.real - hi.real * s.real
    hi_r = hi.real * c.real + lo.real * s.real
    if s_out is None:
        s_out = ctx.sc(tag + "_rope.o", np.concatenate([lo_r, hi_r], axis=-1))
    a = ctx.q.q_addsub(p1, p2, s_out, tag + "_lo", minus=True)
    b = ctx.q.q_addsub(p3, p4, s_out, tag + "_hi")
    return ctx.concat([a, b], 2, (t, heads, HD), tag + "_rope", so=s_out)


def head_repeat(ctx, k2, L, tag):
    """[KV, L, 64] -> [H, L, 64] by repeating each KV head KVR times (GQA), no transpose."""
    k0, k1 = ctx.split(k2, 0, 2, (1, L, HD), tag + "_sp")
    return ctx.concat([k0] * KVR + [k1] * KVR, 0, (H, L, HD), tag)


def layer(ctx, x, P, p, rope, t, mask, kc=None, vc=None):
    """One decoder layer.  kc/vc are this layer's [KV, C, 64] cache slices (decode) or None
    (prefill: attention runs over this layer's own t rows)."""
    sK = S_K
    n1 = rmsnorm5(ctx, x, P["g1"], p + "n1")
    q3 = ctx.fc(n1, P["Wq"], P["bq"], H * HD, p + "q_fc")
    q3 = ctx.reshape(q3, (t, H, HD), p + "q_3d")
    q3 = rope_pair(ctx, q3, H, rope, p + "q")
    k3 = ctx.fc(n1, P["Wk"], P["bk"], KV * HD, p + "k_fc")
    k3 = ctx.reshape(k3, (t, KV, HD), p + "k_3d")
    k3 = rope_pair(ctx, k3, KV, rope, p + "k", s_out=sK)      # K rows live at S_K
    v3 = ctx.fc(n1, P["Wv"], P["bv"], KV * HD, p + "v_fc", so=S_V)
    v3 = ctx.reshape(v3, (t, KV, HD), p + "v_3d")             # V rows live at S_V

    k_new = ctx.transpose(k3, [1, 0, 2], p + "k_t")            # [KV, t, HD]
    v_new = ctx.transpose(v3, [1, 0, 2], p + "v_t")
    if kc is None:
        k_all, v_all, L = k_new, v_new, t
    else:
        C = kc.c.shape[1]
        k_all = ctx.concat([kc, k_new], 1, (KV, C + t, HD), p + "k_all", so=sK)
        v_all = ctx.concat([vc, v_new], 1, (KV, C + t, HD), p + "v_all", so=S_V)
        L = C + t

    kt = head_repeat(ctx, k_all, L, p + "kt")                  # [H, L, HD]
    vt = head_repeat(ctx, v_all, L, p + "vt")
    qt = ctx.transpose(q3, [1, 0, 2], p + "qt")                # [H, t, HD]

    scores = ctx.bmm(qt, kt, p + "scores", adj_y=True)         # [H, t, L]
    masked = ctx.add(scores, mask, p + "masked", so=scores.s)
    e = np.full(masked.c.shape, 1, bw.a_dtype())
    ec = bw.TS(ctx.q.const_i8(p + "eighth", e, 0.125), e, 0.125)
    scaled = ctx.mul(masked, ec, p + "scaled", so=ctx.sc(p + "scaled.o", masked.real / 8.0))
    pr = ctx.softmax(scaled, p + "probs")
    one = np.full(pr.c.shape, 1, bw.a_dtype())
    tt = bw.TS(ctx.q.const_i8(p + "one", one, 1.0), one, 1.0)
    p_sym = ctx.mul(pr, tt, p + "p_sym", so=1.0 / float(bw.aq_max()))

    a3 = ctx.bmm(p_sym, vt, p + "a3")                          # [H, t, HD]
    at = ctx.transpose(a3, [1, 0, 2], p + "at")                # [t, H, HD]
    attn = ctx.reshape(at, (t, D), p + "attn")
    o = ctx.fc(attn, P["Wo"], np.zeros(D), D, p + "o_proj")
    r1 = ctx.add(o, x, p + "r1")
    n2 = rmsnorm5(ctx, r1, P["g2"], p + "n2")
    gate = ctx.fc(n2, P["Wg"], np.zeros(INTER), INTER, p + "gate")
    up = ctx.fc(n2, P["Wu"], np.zeros(INTER), INTER, p + "up")
    sig = ctx.logistic(gate, p + "sig")
    s1 = ctx.mul(gate, sig, p + "silu")
    y = ctx.mul(s1, up, p + "mlp_y")
    down = ctx.fc(y, P["Wd"], np.zeros(D), D, p + "down")
    out = ctx.add(down, r1, p + "out")
    return out, k_new, v_new


def build_graph(P, cfg, tab, feed, emit=True, learn=False):
    """Assemble the whole tower.  Returns (ctx, hidden, k_out, v_out, blob)."""
    t, C, n = cfg["t"], cfg["cache"], cfg["layers"]
    deco = cfg["mode"] == "decode"
    ctx = Ctx(tab, learn=learn, emit=emit)
    q = ctx.q
    L = C + t if deco else t

    sx = ctx.sc("in.x", np.asarray(feed["x"], np.float64))
    x_in = x = ctx.mk_in("x", (t, D),
                  bw.clip_codes(np.round(np.asarray(feed["x"], np.float64) / sx)), sx)
    mask = ctx.mk_in("mask", (H, t, L), feed["mask"], S_MASK)
    rope = {}
    for hh in (H, KV):
        rope["cos%d" % hh] = ctx.mk_in("rope_cos%d" % hh, (t, hh, HALF),
                                       feed["cos%d" % hh], S_TAB)
        rope["sin%d" % hh] = ctx.mk_in("rope_sin%d" % hh, (t, hh, HALF),
                                       feed["sin%d" % hh], S_TAB)
    kp = vp = None
    if deco:
        if cfg.get("kv_split", "packed") == "packed":
            kin = ctx.mk_in("k_in", (n * KV, C, HD), feed["k_in"], S_K)
            vin = ctx.mk_in("v_in", (n * KV, C, HD), feed["v_in"], S_V)
            kp = q.q_split(kin, "k_in_split", 0, n, (KV, C, HD))
            vp = q.q_split(vin, "v_in_split", 0, n, (KV, C, HD))
        else:                      # one graph input per layer: no SPLIT on an input
            kp = [ctx.mk_in("k_in%02d" % i, (KV, C, HD), feed["k_in%02d" % i], S_K)
                  for i in range(n)]
            vp = [ctx.mk_in("v_in%02d" % i, (KV, C, HD), feed["v_in%02d" % i], S_V)
                  for i in range(n)]

    kouts, vouts = [], []
    for i in range(n):
        kc = None if not deco else bw.TS(kp[i].i, kp[i].c, S_K, 0)
        vc = None if not deco else bw.TS(vp[i].i, vp[i].c, S_V, 0)
        x, ko, vo = layer(ctx, x, P, "L%02d_" % i, rope, t, mask, kc=kc, vc=vc)
        kouts.append(ko)
        vouts.append(vo)

    k_out = ctx.concat(kouts, 0, (n * KV, t, HD), "k_out", so=S_K)
    v_out = ctx.concat(vouts, 0, (n * KV, t, HD), "v_out", so=S_V)
    out = [x.i, k_out.i, v_out.i]
    ins = [x_in.i, mask.i] + [rope[k].i for k in
                           ("cos%d" % H, "sin%d" % H, "cos%d" % KV, "sin%d" % KV)]
    if deco:
        if cfg.get("kv_split", "packed") == "packed":
            ins += [kin.i, vin.i]
        else:
            for i in range(n):
                ins += [kp[i].i, vp[i].i]
    blob = q.finish(ins, out, "qwen2.5-0.5b %s %dlayer T=%d cache=%d (ticket 06)"
                    % (cfg["mode"], n, t, C)) if emit else None
    return ctx, x, k_out, v_out, blob


# ------------------------------------------------------------------ feeds (host side)
def inv_freq():
    return bw.THETA ** (-2.0 * np.arange(HALF) / HD)


def rope_codes(t, heads, pos0=0):
    ang = np.outer(np.arange(pos0, pos0 + t), inv_freq())
    c = np.broadcast_to(np.cos(ang)[:, None, :], (t, heads, HALF))
    s = np.broadcast_to(np.sin(ang)[:, None, :], (t, heads, HALF))
    qc = np.clip(np.round(c / S_TAB), -QMAX, QMAX).astype(np.int16)
    qs = np.clip(np.round(s / S_TAB), -QMAX, QMAX).astype(np.int16)
    return np.ascontiguousarray(qc), np.ascontiguousarray(qs)


def prefill_feed(t):
    """Causal mask + RoPE tables for a t-token prefill.  The mask is a graph INPUT now, so
    what used to be a 458 KB/layer constant rides in as data instead of static data."""
    ids = bw.token_ids(t)
    x = np.asarray(bw.input_hidden(t), np.float64)
    iu = np.triu_indices(t, k=1)
    mc = np.zeros((H, t, t), np.int64)
    mc[:, iu[0], iu[1]] = -QMAX
    feed = dict(x=x, mask=mc.astype(np.int16), ids=ids)
    for hh in (H, KV):
        feed["cos%d" % hh], feed["sin%d" % hh] = rope_codes(t, hh)
    return feed


def decode_feed(t, C, pos, valid, x_rows):
    """Mask for one decode step: cache slots [0, valid) are real, slot C+r belongs to query
    row r, everything else is masked."""
    mc = np.full((H, t, C + t), -QMAX, np.int64)
    mc[:, :, :valid] = 0
    for r in range(t):
        mc[:, r, C + r] = 0
    feed = dict(x=np.asarray(x_rows, np.float64), mask=mc.astype(np.int16))
    for hh in (H, KV):
        feed["cos%d" % hh], feed["sin%d" % hh] = rope_codes(t, hh, pos0=pos)
    return feed


# --------------------------------------------------------------- fp64 reference
def ref_layer(P, x, khist, vhist, pos0=0):
    """Independent float64 decoder layer, written from the model definition (AGENTS R3).

    x [T, D]; khist/vhist [KV, Lh, HD] already RoPE-ed, holding every earlier position.
    Returns (out [T,D], k_new [KV,T,HD], v_new [KV,T,HD]).
    """
    T = x.shape[0]
    g1 = np.asarray(P["g1"], np.float64)
    n1 = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + EPS) * g1

    def proj(W, b, heads):
        return (n1 @ np.asarray(W, np.float64).T + np.asarray(b, np.float64)
                ).reshape(T, heads, HD)

    ang = np.outer(np.arange(pos0, pos0 + T), inv_freq())[:, None, :]
    c, s = np.cos(ang), np.sin(ang)

    def rope(y3):
        a, bb = y3[..., :HALF], y3[..., HALF:]
        return np.concatenate([a * c - bb * s, bb * c + a * s], axis=-1)

    q3 = rope(proj(P["Wq"], P["bq"], H))
    k3 = rope(proj(P["Wk"], P["bk"], KV))
    v3 = proj(P["Wv"], P["bv"], KV)
    k_all = np.concatenate([khist, np.transpose(k3, (1, 0, 2))], axis=1)   # [KV, L, HD]
    v_all = np.concatenate([vhist, np.transpose(v3, (1, 0, 2))], axis=1)
    k14 = np.repeat(k_all, KVR, axis=0)
    v14 = np.repeat(v_all, KVR, axis=0)
    qt = np.transpose(q3, (1, 0, 2))                                       # [H, T, HD]
    scores = (qt @ np.transpose(k14, (0, 2, 1))) / np.sqrt(float(HD))   # the 1/sqrt(d) scale
    L = k_all.shape[1]
    m = np.ones((T, L))
    for i in range(T):                     # query row i = position L-T+i
        m[i, L - T + i + 1:] = 0.0
    sc = np.where(m > 0, scores, -1e30)
    sc = sc - sc.max(axis=-1, keepdims=True)
    e = np.exp(sc)
    pr = e / e.sum(axis=-1, keepdims=True)
    a3 = pr @ v14
    attn = np.transpose(a3, (1, 0, 2)).reshape(T, D)
    o = attn @ np.asarray(P["Wo"], np.float64).T
    r1 = o + x
    g2 = np.asarray(P["g2"], np.float64)
    n2 = r1 / np.sqrt((r1 * r1).mean(axis=-1, keepdims=True) + EPS) * g2
    gate = n2 @ np.asarray(P["Wg"], np.float64).T
    up = n2 @ np.asarray(P["Wu"], np.float64).T
    y = gate / (1.0 + np.exp(-gate)) * up
    down = y @ np.asarray(P["Wd"], np.float64).T
    return down + r1, np.transpose(k3, (1, 0, 2)), np.transpose(v3, (1, 0, 2))


def ref_chain(P, x0, khist, vhist, n, pos0=0):
    """n chained fp64 layers; returns the stack output and EVERY layer's new K/V."""
    x = np.asarray(x0, np.float64)
    ks, vs = [], []
    for i in range(n):
        x, k, v = ref_layer(P, x, khist[i], vhist[i], pos0=pos0)
        ks.append(k)
        vs.append(v)
    return x, ks, vs


# ------------------------------------------------------------------------ logits
_EMB = {}


def emb():
    """Tied LM head: Qwen2.5-0.5B has no separate lm_head, logits = h @ embed.T."""
    if "e" not in _EMB:
        _EMB["e"] = bw.emb_weight()          # fp16: 272 MB, never widened
    return _EMB["e"]


def logits(h, chunk=16384):
    """Logits of ONE position: a 2-D input uses its last row (the predicting token)."""
    e = emb()
    h = np.asarray(h, np.float64)
    h = (h[-1] if h.ndim > 1 else h).reshape(-1)
    out = np.empty(e.shape[0], np.float32)
    for i in range(0, e.shape[0], chunk):
        out[i:i + chunk] = e[i:i + chunk].astype(np.float64) @ h
    return out


# ------------------------------------------------------------------- calibration
def calibrate(P, cfg):
    """Freeze one scale table from the PREFILL batch, then fix S_K / S_V.

    One table for both graphs is deliberate: the int32 bias constants and the gamma
    constants are functions of the activation scales only, so sharing the table is what
    makes the two graphs' statics byte-identical (extract-shared then reuses 100%).

    Calibration / evaluation are kept apart (NOTES 11.2): the table comes from the prefill
    batch of token ids 0..t-1; every number reported afterwards is a decode step, i.e. a
    different position and a different activation path.
    """
    global S_K, S_V
    t = cfg["t"]
    n = cfg["layers"]
    feed = prefill_feed(t)

    S_K = S_V = None
    tab = {}
    ctx, _x, k_out, v_out, _ = build_graph(P, dict(cfg, mode="prefill", cache=0), tab, feed,
                                           emit=True, learn=True)
    kmax = float(np.abs(k_out.real).max())
    vmax = float(np.abs(v_out.real).max())
    S_K = kmax / QMAX
    S_V = vmax / QMAX
    tab2 = {}
    ctx2, _x, k2, v2, _ = build_graph(P, dict(cfg, mode="prefill", cache=0), tab2, feed,
                                      emit=True, learn=True)
    return dict(tab=tab2, S_K=S_K, S_V=S_V, kmax=kmax, vmax=vmax,
                sat=dict(ctx2.sat), keys=len(tab2))


def load_tab(path):
    global S_K, S_V
    j = json.load(open(path))
    S_K = float(j["S_K"])
    S_V = float(j["S_V"])
    return j["tab"]


def save_tab(path, c):
    json.dump(dict(tab=c["tab"], S_K=c["S_K"], S_V=c["S_V"],
                   kmax=c["kmax"], vmax=c["vmax"]), open(path, "w"))


# ---------------------------------------------------------------------- building
def write_feed(d, feed, names):
    os.makedirs(d, exist_ok=True)
    for k, fn in names:
        np.ascontiguousarray(np.asarray(feed[k], np.int16)).tofile(os.path.join(d, fn))


PREFILL_IO = [("x", "x.bin"), ("mask", "mask.bin"), ("cos14", "cos14.bin"),
              ("sin14", "sin14.bin"), ("cos2", "cos2.bin"), ("sin2", "sin2.bin")]
DECODE_IO = PREFILL_IO + [("k_in", "k_in.bin"), ("v_in", "v_in.bin")]


def build_op(P, cfg, tab, out, decode):
    t, n, C = cfg["t"], cfg["layers"], cfg["cache"]
    if decode:
        xr = np.asarray(bw.input_hidden(t + 1), np.float64)[1:1 + t]
        feed = decode_feed(t, C, pos=1, valid=0, x_rows=xr)
        if cfg.get("kv_split", "packed") == "packed":
            feed["k_in"] = np.zeros((n * KV, C, HD), np.int16)
            feed["v_in"] = np.zeros((n * KV, C, HD), np.int16)
            names = DECODE_IO
        else:
            for i in range(n):
                feed["k_in%02d" % i] = np.zeros((KV, C, HD), np.int16)
                feed["v_in%02d" % i] = np.zeros((KV, C, HD), np.int16)
            kvn = []
            for i in range(n):
                kvn += [("k_in%02d" % i, "k_in%02d.bin" % i),
                        ("v_in%02d" % i, "v_in%02d.bin" % i)]
            names = PREFILL_IO + kvn
    else:
        feed = prefill_feed(t)
        names = PREFILL_IO
    ctx, x, k_out, v_out, blob = build_graph(P, cfg, tab, feed, emit=True)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "model.tflite"), "wb").write(blob)
    write_feed(os.path.join(out, "golden"), dict(feed, x=q_x(feed["x"], tab["in.x"])),
               names)
    np.ascontiguousarray(x.c).tofile(os.path.join(out, "golden", "hidden.bin"))
    meta = dict(
        op="PREFILL_DECODE", mode=cfg["mode"], layers=n, shape=dict(T=t, C=C, D=D, H=H,
                                                                   KV=KV, HD=HD),
        dtype="int16", weights_dtype="int8",
        builder="tools/build_q06.py --mode %s --t %d --cache %d --layers %d"
                % (cfg["mode"], t, C, n),
        inputs=[dict(name=a, file=b, shape=list(np.asarray(feed[a]).shape)) for a, b in names],
        outputs=[dict(name="hidden", file="hidden.bin", shape=[t, D]),
                 dict(name="k_out", file="k_out.bin", shape=[n * KV, t, HD]),
                 dict(name="v_out", file="v_out.bin", shape=[n * KV, t, HD])],
        quant=dict(S_K=S_K, S_V=S_V, S_MASK=S_MASK, S_TAB=S_TAB,
                   table=dict(n=len(tab))),
        mask=dict(value=MASK_REAL, note="graph INPUT, scale %.8f, code -32767" % S_MASK),
        rope=dict(note="cos/sin tables are graph INPUTS (scale %.8f)" % S_TAB),
        kv=dict(layout="k_in/v_in [layers*KV, C, HD] int16; host owns the cache",
                per_token_host_bytes=int(2 * n * KV * C * HD * 2)),
        weights_note=("fixture layer 0 repeated for all %d layers (ticket 05 allows weight "
                      "reuse): this is not a real 24-layer Qwen2.5-0.5B stack" % n),
        # ---- registered BEFORE the first board run ----
        int8_tol=1200,
        abs_dcos_device_vs_golden_max=0.003,
        rel_l2_device_vs_golden_max=0.05,
        quality=dict(
            hidden=("per-step device vs independent fp64 chain cosine; expected to decay as "
                    "the KV chain accumulates (ticket 05 measured 0.9962 for 24 chained "
                    "layers at T=16)"),
            logits=("final quality gate: cosine of h @ embed.T and argmax token id "
                    "(Qwen2.5-0.5B ties lm_head to embed.weight; the lm head is NOT on the "
                    "device in this ticket)")),
        tolerance_rationale=(
            "Registered before the first board run. Code tolerance 1200 = sqrt(24) x ticket "
            "04f single-layer 247 codes (the same propagation bound ticket 05 registered); "
            "abs(dcos) <= 0.003 is the ADR 0005 sixth-revision implementation gate. The "
            "decode chain is 8+ steps of the same 24-layer tower, so code drift may exceed "
            "the single-tower bound; that is reported as measured, not adjusted."),
    )
    meta["tflite_bytes"] = len(blob)
    json.dump(meta, open(os.path.join(out, "model.json"), "w"), indent=2)
    print("%-34s %d B  %s l=%d T=%d C=%d  in=%s"
          % (out, len(blob), cfg["mode"], n, t, C, [a for a, _ in names]))
    if ctx.sat:
        print("  saturation: %s" % dict(sorted(ctx.sat.items())[:8]))
    return feed


# ------------------------------------------------------------------- sequence test
def q_x(feed_x, s):
    return bw.clip_codes(np.round(np.asarray(feed_x, np.float64) / s))


def code_stats(dev, gold, s):
    d = np.asarray(dev, np.int64)
    g = np.asarray(gold, np.int64)
    a = np.abs(d - g)
    dr = d.astype(np.float64) * s
    gr = g.astype(np.float64) * s
    return dict(codes_max=int(a.max()), codes_mean=float(a.mean()),
                exact=int((a == 0).sum()), n=int(a.size),
                cos=float(bw.cosine(dr, gr)))


def seq_state(d):
    return json.load(open(os.path.join(d, "state.json")))


def save_state(d, st):
    json.dump(st, open(os.path.join(d, "state.json"), "w"), indent=2)


def seq_tokens(t, steps):
    return list(bw.token_ids(t + steps))


def golden_prefill(P, t, n, tab):
    ctx, x, k, v, _ = build_graph(P, dict(mode="prefill", t=t, cache=0, layers=n), tab,
                                 prefill_feed(t), emit=True)
    return dict(h=x.c, s_h=x.s, k=k.c, v=v.c)


def golden_step(P, cfg, tab, x_real, pos, m, kc, vc):
    feed = decode_feed(cfg["t"], cfg["cache"], pos, m, x_real)
    feed["k_in"] = np.asarray(kc, np.int16)
    feed["v_in"] = np.asarray(vc, np.int16)
    ctx, x, k, v, _ = build_graph(P, cfg, tab, feed, emit=True)
    return dict(h=x.c, s_h=x.s, k=k.c, v=v.c, sat=dict(ctx.sat))


def logit_pair(h_a, h_b):
    la, lb = logits(h_a), logits(h_b)
    cos = float(bw.cosine(la, lb))
    return dict(cos=cos, dcos=float(abs(cos - 1.0)),
                argmax_a=int(np.argmax(la)), argmax_b=int(np.argmax(lb)),
                hit=int(np.argmax(la) == np.argmax(lb)),
                top10=float(len(set(np.argsort(la)[-10:]) & set(np.argsort(lb)[-10:])) / 10.0))


DRY = False           # --dry: seed/absorb with the GOLDEN outputs (device-free rehearsal)


def seq_init(op_pre, op_dec, tab_path, steps, d):
    """Seed both caches from the PREFILL device output, then write step 0 inputs."""
    P = {k: np.asarray(v, np.float64) for k, v in bw.fixture().items()}
    tab = load_tab(tab_path)
    mp = json.load(open(os.path.join(op_pre, "model.json")))
    md = json.load(open(os.path.join(op_dec, "model.json")))
    t, n, C = mp["shape"]["T"], mp["layers"], md["shape"]["C"]
    os.makedirs(d, exist_ok=True)
    gp = golden_prefill(P, t, n, tab)
    if DRY:
        h_dev, k_dev, v_dev = gp["h"], gp["k"], gp["v"]
    else:
        h_dev = np.fromfile(os.path.join(op_pre, "out_hidden.bin"), np.int16).reshape(t, D)
        k_dev = np.fromfile(os.path.join(op_pre, "out_k.bin"), np.int16).reshape(n * KV, t, HD)
        v_dev = np.fromfile(os.path.join(op_pre, "out_v.bin"), np.int16).reshape(n * KV, t, HD)

    x0 = np.asarray(bw.input_hidden(t), np.float64)
    h_ref, k_ref, v_ref = ref_chain(P, x0, [np.zeros((KV, 0, HD))] * n,
                                    [np.zeros((KV, 0, HD))] * n, n)

    rep = dict(
        prefill_hidden=code_stats(h_dev, gp["h"], gp["s_h"]),
        prefill_logits=logit_pair(h_dev.astype(np.float64) * gp["s_h"], h_ref),
        prefill_hidden_vs_fp64=float(bw.cosine(h_dev.astype(np.float64) * gp["s_h"], h_ref)),
        prefill_kv_k=code_stats(k_dev, gp["k"], S_K),
        prefill_kv_v=code_stats(v_dev, gp["v"], S_V),
    )

    kd = np.zeros((n * KV, C, HD), np.int16)
    vd = np.zeros((n * KV, C, HD), np.int16)
    kg = np.zeros((n * KV, C, HD), np.int16)
    vg = np.zeros((n * KV, C, HD), np.int16)
    kd[:, 0:t] = k_dev
    vd[:, 0:t] = v_dev
    kg[:, 0:t] = gp["k"]
    vg[:, 0:t] = gp["v"]

    ids = seq_tokens(t, steps)
    st = dict(t=t, n=n, C=C, dt=md["shape"]["T"], m=t, pos=t, step=0, steps=steps,
              ids=ids, tab=tab_path, op_pre=op_pre, op_dec=op_dec)
    save_state(d, st)
    np.save(os.path.join(d, "k_dev.npy"), kd)
    np.save(os.path.join(d, "v_dev.npy"), vd)
    np.save(os.path.join(d, "k_gold.npy"), kg)
    np.save(os.path.join(d, "v_gold.npy"), vg)
    np.save(os.path.join(d, "k_ref.npy"), np.stack(k_ref))
    np.save(os.path.join(d, "v_ref.npy"), np.stack(v_ref))
    np.save(os.path.join(d, "h_ref_pre.npy"), h_ref)
    json.dump(rep, open(os.path.join(d, "prefill.json"), "w"), indent=2)
    print("prefill: hidden codes %s  |dcos| vs fp64 %.6f  logits cos %.7f"
          % (rep["prefill_hidden"], rep["prefill_hidden_vs_fp64"],
             rep["prefill_logits"]["cos"]))
    prepare_step(d, 0)
    return rep


def prepare_step(d, step):
    """Write the device inputs for this step plus the golden expectation."""
    st = seq_state(d)
    P = {k: np.asarray(v, np.float64) for k, v in bw.fixture().items()}
    tab = load_tab(st["tab"])
    t, n, C = st["t"], st["n"], st["C"]
    cfg_d = dict(mode="decode", t=st["dt"], cache=C, layers=n)
    pos = st["pos"]
    tok = st["ids"][pos]
    row = np.asarray(bw.emb_weight()[tok], np.float64).reshape(1, D)
    # padded query rows carry the same token: row 0 is the one that is used, and repeating
    # it keeps every intermediate finite (the T=1 graph does not compile, see docs)
    x_real = np.tile(row, (st["dt"], 1))
    kd = np.load(os.path.join(d, "k_dev.npy"))
    vd = np.load(os.path.join(d, "v_dev.npy"))
    feed = decode_feed(st["dt"], C, pos, min(st["m"], C), x_real)
    feed["k_in"] = kd
    feed["v_in"] = vd
    live = os.path.join(d, "live")
    write_feed(live, dict(feed, x=q_x(x_real, tab["in.x"])), DECODE_IO)
    g = golden_step(P, cfg_d, tab, x_real, pos, min(st["m"], C),
                    np.load(os.path.join(d, "k_gold.npy")),
                    np.load(os.path.join(d, "v_gold.npy")))
    np.save(os.path.join(d, "gold_h_%02d.npy" % step), g["h"])
    np.save(os.path.join(d, "gold_k_%02d.npy" % step), g["k"])
    np.save(os.path.join(d, "gold_v_%02d.npy" % step), g["v"])
    if DRY:                    # the rehearsal: the graph's own outputs play the device
        np.ascontiguousarray(g["h"], np.int16).tofile(os.path.join(live, "out_hidden.bin"))
        np.ascontiguousarray(g["k"], np.int16).tofile(os.path.join(live, "out_k.bin"))
        np.ascontiguousarray(g["v"], np.int16).tofile(os.path.join(live, "out_v.bin"))
    json.dump(dict(s_h=g["s_h"], sat=g["sat"], tok=int(tok), pos=int(pos), m=int(st["m"])),
              open(os.path.join(d, "gold_meta_%02d.json" % step), "w"), indent=2)


def absorb_step(d, step):
    """Compare one device step, advance both caches, write the next step inputs."""
    st = seq_state(d)
    P = {k: np.asarray(v, np.float64) for k, v in bw.fixture().items()}
    tab = load_tab(st["tab"])
    t, n, C = st["t"], st["n"], st["C"]
    gm = json.load(open(os.path.join(d, "gold_meta_%02d.json" % step)))
    gh = np.load(os.path.join(d, "gold_h_%02d.npy" % step))
    gk = np.load(os.path.join(d, "gold_k_%02d.npy" % step))
    gv = np.load(os.path.join(d, "gold_v_%02d.npy" % step))
    live = os.path.join(d, "live")
    dt = st["dt"]
    dh = np.fromfile(os.path.join(live, "out_hidden.bin"), np.int16).reshape(dt, D)[:1]
    dk = np.fromfile(os.path.join(live, "out_k.bin"), np.int16).reshape(n * KV, dt, HD)[:, :1]
    dv = np.fromfile(os.path.join(live, "out_v.bin"), np.int16).reshape(n * KV, dt, HD)[:, :1]
    gh, gk, gv = gh[:1], gk[:, :1], gv[:, :1]
    s_h = float(gm["s_h"])

    k_ref = np.load(os.path.join(d, "k_ref.npy"))
    v_ref = np.load(os.path.join(d, "v_ref.npy"))
    x_ref = np.asarray(bw.emb_weight()[gm["tok"]], np.float64).reshape(1, D)
    h_ref, kr, vr = ref_chain(P, x_ref, list(k_ref), list(v_ref), n, pos0=gm["pos"])
    np.save(os.path.join(d, "k_ref.npy"), np.concatenate([k_ref, np.stack(kr)], axis=2))
    np.save(os.path.join(d, "v_ref.npy"), np.concatenate([v_ref, np.stack(vr)], axis=2))

    hd = dh.astype(np.float64) * s_h
    hg = gh.astype(np.float64) * s_h
    rec = dict(
        step=int(step), pos=int(gm["pos"]), tok=int(gm["tok"]), m=int(gm["m"]),
        hidden=code_stats(dh, gh, s_h),
        hidden_vs_fp64=float(bw.cosine(hd, h_ref)),
        golden_vs_fp64=float(bw.cosine(hg, h_ref)),
        kv_k=code_stats(dk, gk, S_K), kv_v=code_stats(dv, gv, S_V),
        logits_dev=logit_pair(hd, h_ref), logits_gold=logit_pair(hg, h_ref),
        sat_gold=gm["sat"],
    )
    rec["abs_dcos"] = float(abs(rec["hidden"]["cos"] - 1.0))
    with open(os.path.join(d, "seq.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")

    m = st["m"]
    kd = np.load(os.path.join(d, "k_dev.npy"))
    vd = np.load(os.path.join(d, "v_dev.npy"))
    kg = np.load(os.path.join(d, "k_gold.npy"))
    vg = np.load(os.path.join(d, "v_gold.npy"))
    if m >= C:
        for a in (kd, vd, kg, vg):
            a[:, :-1] = a[:, 1:].copy()
            a[:, C - 1] = 0
        m = C - 1
    kd[:, m] = dk[:, 0]
    vd[:, m] = dv[:, 0]
    kg[:, m] = gk[:, 0]
    vg[:, m] = gv[:, 0]
    np.save(os.path.join(d, "k_dev.npy"), kd)
    np.save(os.path.join(d, "v_dev.npy"), vd)
    np.save(os.path.join(d, "k_gold.npy"), kg)
    np.save(os.path.join(d, "v_gold.npy"), vg)
    st.update(m=m + 1, pos=st["pos"] + 1, step=int(step) + 1)
    save_state(d, st)
    print("step %d pos %d: codes max %4d mean %6.2f | cos(dev,gold) %.7f  "
          "cos(dev,fp64) %.7f  logits cos %.7f hit %d"
          % (rec["step"], rec["pos"], rec["hidden"]["codes_max"],
             rec["hidden"]["codes_mean"], rec["hidden"]["cos"], rec["hidden_vs_fp64"],
             rec["logits_dev"]["cos"], rec["logits_dev"]["hit"]))
    if st["step"] < st["steps"]:
        prepare_step(d, st["step"])
    return rec


# ------------------------------------------------------------------------- report
def seq_report(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "seq.jsonl"))]
    pre = json.load(open(os.path.join(d, "prefill.json")))
    print("prefill : hidden codes max %d  |dcos| vs fp64 %.6f  logits cos %.7f"
          % (pre["prefill_hidden"]["codes_max"], pre["prefill_hidden_vs_fp64"],
             pre["prefill_logits"]["cos"]))
    print("%4s %5s %6s %6s %8s %10s %10s %10s %10s %6s"
          % ("step", "pos", "codmax", "codmean", "absdcos", "cos(d,gold)",
             "cos(d,fp64)", "cos(logit)", "cos(g,fp64)", "hit"))
    for r in recs:
        print("%4d %5d %6d %6.2f %8.5f %10.7f %10.7f %10.7f %10.7f %6d"
              % (r["step"], r["pos"], r["hidden"]["codes_max"], r["hidden"]["codes_mean"],
                 r["abs_dcos"], r["hidden"]["cos"], r["hidden_vs_fp64"],
                 r["logits_dev"]["cos"], r["golden_vs_fp64"], r["logits_dev"]["hit"]))
    import statistics as st
    print("summary : abs(dcos) max %.5f (gate 0.003)  codes max %d (gate 1200)  "
          "logits cos last %.7f  token hits %d/%d"
          % (max(r["abs_dcos"] for r in recs), max(r["hidden"]["codes_max"] for r in recs),
             recs[-1]["logits_dev"]["cos"],
             sum(r["logits_dev"]["hit"] for r in recs), len(recs)))
    return 0


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--tab", default="ops/q06_tab.json")
    ap.add_argument("--mode", default="prefill")
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--cache", type=int, default=0)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--kv-split", default="packed", choices=("packed", "perlayer"))
    ap.add_argument("--dry", action="store_true",
                    help="device-free rehearsal: golden outputs play the device")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seq-init", action="store_true")
    ap.add_argument("--seq-prepare", type=int, default=None)
    ap.add_argument("--seq-absorb", type=int, default=None)
    ap.add_argument("--seq-report", action="store_true")
    ap.add_argument("--seqdir", default="ops/q06_seq")
    ap.add_argument("--op-prefill", default="ops/q06_prefill_t16_24l")
    ap.add_argument("--op-decode", default="ops/q06_decode_t1_c2048_24l")
    a = ap.parse_args()

    global DRY
    DRY = a.dry
    bw.set_tier(8, 16)
    bw.SEAM = "native"
    bw.set_source("fixture")
    bw.set_input("same")

    if a.calib:
        P = {k: np.asarray(v, np.float64) for k, v in bw.fixture().items()}
        cfg = dict(t=a.t, layers=a.layers)
        c = calibrate(P, cfg)
        save_tab(a.tab, c)
        print("tab: %d scales  S_K=%.6e (kmax %.4f)  S_V=%.6e (vmax %.5f)  sat=%s"
              % (c["keys"], c["S_K"], c["kmax"], c["S_V"], c["vmax"], c["sat"]))
        return 0
    if a.seq_init:
        seq_init(a.op_prefill, a.op_decode, a.tab, a.steps, a.seqdir)
        return 0
    if a.seq_prepare is not None:
        prepare_step(a.seqdir, a.seq_prepare)
        return 0
    if a.seq_absorb is not None:
        absorb_step(a.seqdir, a.seq_absorb)
        return 0
    if a.seq_report:
        return seq_report(a.seqdir)

    P = {k: np.asarray(v, np.float64) for k, v in bw.fixture().items()}
    tab = load_tab(a.tab)
    cfg = dict(mode=a.mode, t=a.t, cache=a.cache, layers=a.layers, kv_split=a.kv_split)
    out = a.out or ("ops/q06_%s_t%d%s_%dl" % (a.mode, a.t,
                                              "" if a.mode == "prefill" else "_c%d" % a.cache,
                                              a.layers))
    build_op(P, cfg, tab, out, a.mode == "decode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
