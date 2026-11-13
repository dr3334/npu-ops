#!/usr/bin/env python3
"""Ticket 04b -- group-wise (per-(output channel, K-slab)) int4 weight scales on this stack.

Q: ticket 04 measured 14.0-15.8% rel_l2 per matrix with int4 per-output-channel scales.
   TFLite QuantizationParameters has one quantized_dimension -> a scale per output
   channel is the finest attribute-level expression.  A (n, group) grid can still be
   built by DECOMPOSITION: cut K into G slabs, run G FULLY_CONNECTED (each with its own
   per-channel int4 scales) and add the partials.  That is group-wise quantisation with
   group size K/G, in stock TFLite ops.

Usage
    python3 tools/build_q04b.py --list
    python3 tools/build_q04b.py errcurve
    python3 tools/build_q04b.py fc_g --group 4 --out ops/q04b_fc_g4
    python3 tools/build_q04b.py layer --group 4 --out ops/q04b_layer_g4
    python3 tools/build_q04b.py schema_scale2n --out ops/q04b_probe_scale2n
    python3 tools/build_q04b.py schema_axis1  --out ops/q04b_probe_axis1
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ops as bo
import build_w4a8 as bw

T, D, H, KV, HD, INTER = bw.T, bw.D, bw.H, bw.KV, bw.HD, bw.INTER
MATS = ["Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd"]


# --------------------------------------------------------------- group quantisation
def gq_int4(W, G):
    """int4 with a scale per (output channel, K-slab): the vendor-style group scale.

    W [N,K] -> codes [N,K] (values -8..7), scales [N,G], dequantised [N,K].
    G == 1 is exactly ticket 04 per-output-channel quantisation (G == K is per-weight,
    never used; G must divide K).
    """
    W = np.asarray(W, np.float64)
    N, K = W.shape
    if K % G:
        raise SystemExit("G=%d does not divide K=%d" % (G, K))
    KG = K // G
    Wg = W.reshape(N, G, KG)
    sc = np.maximum(np.abs(Wg).max(axis=2) / 7.0, 1e-12)          # [N,G]
    q = np.clip(np.round(Wg / sc[:, :, None]), -8, 7)
    deq = q * sc[:, :, None]
    return q.reshape(N, K).astype(np.int8), sc, deq.reshape(N, K)


def gq_int8(W, G):
    """Same grid with int8 weights (-127..127): used only for the 04c cross-reference."""
    W = np.asarray(W, np.float64)
    N, K = W.shape
    KG = K // G
    Wg = W.reshape(N, G, KG)
    sc = np.maximum(np.abs(Wg).max(axis=2) / 127.0, 1e-12)
    q = np.clip(np.round(Wg / sc[:, :, None]), -127, 127)
    deq = q * sc[:, :, None]
    return q.reshape(N, K).astype(np.int8), sc, deq.reshape(N, K)


def wmetrics(W, Wq):
    W = np.asarray(W, np.float64)
    d = W - np.asarray(Wq, np.float64)
    rel = float(np.linalg.norm(d) / np.linalg.norm(W))
    cos = float((W * Wq).sum() / (np.linalg.norm(W) * np.linalg.norm(Wq)))
    return rel, cos


def errcurve():
    """Weight-quantisation error vs group size, and the layer-level consequence.

    Two independent blocks, both numpy/float64, no device, no flatbuffer:
      1) per-matrix rel_l2 / cosine for int4 with group size K/G,
      2) the whole decoder layer run with group-quantised (dequantised) weights and
         everything else in float64 -> layer cosine / rel_l2 vs the fp64 chain.
    Block 2 is the number that decides the tier: it isolates the WEIGHT error (the
    14% main term of ticket 04) from the activation error.
    """
    P = bw.fixture()
    ch = bw.chain_f32(P)
    ref = ch["out"]
    # G must divide BOTH 896 (q/k/v/o) and 4864 (gate/up) and 896x4864 (down), because
    # every matrix of the layer is tiled with the same G.  gcd(896, 4864) = 128.
    groups = [1, 2, 4, 8, 16, 32, 64, 128]
    out = {"groups": groups, "per_matrix": {}, "layer": {}}
    print("=== int4 weight quantisation error vs group size (K=896, group = K/G) ===")
    for G in groups:
        row = {}
        for k in MATS:
            W = np.asarray(P[k], np.float64)
            _, _, deq = gq_int4(W, G)
            rel, cos = wmetrics(W, deq)
            row[k] = dict(G=G, group_size=W.shape[1] // G, rel_l2=rel, cosine=cos)
        out["per_matrix"][G] = row
        mean_rel = float(np.mean([row[k]["rel_l2"] for k in MATS]))
        print("  G=%-4d group=%-4d " % (G, D // G)
              + "  ".join("%s %.4f" % (k, row[k]["rel_l2"]) for k in MATS)
              + "   | mean %.4f" % mean_rel)
    print()
    print("=== whole layer: only the WEIGHTS are group-quantised, rest float64 ===")
    print("  G   group  out cosine   out rel_l2   probs cosine")
    for G in groups:
        P2 = dict(P)
        for k in MATS:
            _, _, deq = gq_int4(np.asarray(P[k], np.float64), G)
            P2[k] = deq.astype(np.float32)
        c2 = bw.chain_f32(P2)
        cos = bw.cosine(c2["out"], ref)
        rel = float(np.linalg.norm(c2["out"] - ref) / np.linalg.norm(ref))
        cosp = bw.cosine(c2["probs"], ch["probs"])
        out["layer"][G] = dict(cosine=cos, rel_l2=rel, probs_cosine=cosp)
        print("  %-4d %-6d %.7f  %.4e  %.7f" % (G, D // G, cos, rel, cosp))
    print()
    print("=== reference points ===")
    print("  ticket 04 per-channel int4 layer: cosine 0.9130 (quantised golden vs fp64)")
    print("  ticket 04 device layer          : cosine 0.9107")
    print("  vendor 4-bit bin                : 0.5075 B/param -> group ~32-64")
    dump = os.path.join("ops", "q04b_errcurve")
    os.makedirs(dump, exist_ok=True)
    json.dump(out, open(os.path.join(dump, "errcurve.json"), "w"), indent=1)
    print("  -> %s/errcurve.json" % dump)
    return 0

# --------------------------------------------------------- grouped FC (the trick)
def group_fc_tables(a_codes, a_scale, W_f64, b_f64, G):
    """Per-channel tables for each of the G K-slabs, plus the bias on slab 0 only.

    Returns (qg [G,N,KG] int4 codes, s_w [G,N], b_q [G,N] int32, s_o hint, o_ref).
    The bias is rounded once, against slab 0 s scale, and is zero for the other slabs.
    """
    W = np.asarray(W_f64, np.float64)
    N, K = W.shape
    KG = K // G
    q, sc, deq = gq_int4(W, G)                    # q [N,K], sc [N,G]
    qg = np.ascontiguousarray(q.reshape(N, G, KG).transpose(1, 0, 2))
    sw = np.ascontiguousarray(sc.T.copy())        # [G,N]
    bs = a_scale * sw
    b_q = np.zeros((G, N), np.int64)
    b_q[0] = np.round(np.asarray(b_f64, np.float64) / bs[0]).astype(np.int64)
    o_ref = (a_codes.astype(np.float64) * a_scale) @ deq.T + np.asarray(b_f64, np.float64)
    return qg, sw.astype(np.float32), b_q.astype(np.int32), bw.pick_scale(o_ref), o_ref


_BASE_QL = bw.QL          # captured once: build_layer_g swaps bw.QL later


class QLG(bw.QL):
    """QL whose FULLY_CONNECTED may carry a [G,N] scale grid (one scale per (n,slab)).

    Layer count per matrix: 1 K-split (shared by all consumers of the same input)
    + G FC + (G-1) ADD, i.e. 2G-1 layers plus the amortised split.
    """
    GROUPS = 1

    def __init__(self):
        super().__init__()
        self._kparts = {}

    def ksplit(self, a, name):
        if a.i in self._kparts:                   # q/k/v/o all eat the same [T,896]
            return self._kparts[a.i]
        K = a.c.shape[1]
        parts = self.q_split(a, name, 1, self.GROUPS, (a.c.shape[0], K // self.GROUPS))
        self._kparts[a.i] = parts
        return parts

    def q_fc(self, a, W_i4, s_w, b_q, so, n, name, version=9):
        s_w = np.asarray(s_w)
        if s_w.ndim == 1:
            return _BASE_QL.q_fc(self, a, W_i4, s_w, b_q, so, n, name, version)
        parts = self.ksplit(a, name + "_ksplit")
        running = None
        for gi, part in enumerate(parts):
            wg = np.ascontiguousarray(np.asarray(W_i4)[gi])
            swg = np.asarray(s_w[gi], np.float64)
            bqg = np.asarray(b_q[gi], np.int64)
            real = ((part.c.astype(np.float64) * part.s)
                    @ (wg.astype(np.float64) * swg[:, None]).T
                    + bqg * (part.s * swg)[None, :])
            sog = bw.pick_scale(real)             # each slab keeps its own resolution
            cur = _BASE_QL.q_fc(self, part, wg, swg, bqg, sog, n, name + "_g%d" % gi, version)
            if running is None:
                running = cur
            else:
                so2 = bw.pick_scale(running.real + cur.real)
                running = self.q_addsub(running, cur, so2, name + "_sum%d" % gi)
        return running


def build_fc_g(P, G):
    """q_proj [16,896]x[896,896] int4 with a group-wise K-slab scale grid."""
    ch = bw.chain_f32(P)
    n1 = ch["n1"]
    a_c, s_a = bw.s8(n1)
    qg, sw, bq, so, o_ref = group_fc_tables(a_c, s_a, P["Wq"], P["bq"], G)
    cls = type("QLGx%d" % G, (QLG,), dict(GROUPS=G))
    g = cls()
    a = g.mk_in("input", (T, D), a_c, s_a)
    y = g.q_fc(a, qg, sw, bq, so, D, "out")
    blob = g.finish([a.i], [y.i], "q04b q_proj int4 group-wise G=%d T=%d" % (G, T))
    _, _, deq = gq_int4(np.asarray(P["Wq"], np.float64), G)
    wrel, wcos = wmetrics(P["Wq"], deq)
    meta = dict(op="FULLY_CONNECTED", form="w4a8_group", stage="q_proj",
                shape=dict(T=T, K=D, N=D, G=G, group_size=D // G), dtype="int8",
                weights_dtype="int4",
                input={"name": "input", "shape": [T, D], "type": "int8"},
                output={"name": "out", "shape": [T, D], "type": "int8"},
                quant=dict(input=float(s_a), output=float(y.s),
                           slabs=G, scales_per_matrix=int(G * D),
                           weight_rel_l2=wrel, weight_cosine=wcos,
                           weights_bytes=int(bw.pack_i4(qg).size),
                           layers_per_matrix=2 * G - 1),
                fp32_ref="q_proj_pre_rope",
                int8_tol=1, tolerance_rationale=bw.ADR_TOL)
    extra = dict(fp32=n1 @ np.asarray(P["Wq"], np.float64).T
                 + np.asarray(P["bq"], np.float64))
    return blob, meta, a_c, y.c, extra


def build_layer_g(P, upto, G):
    """The whole 4W8A layer with every FC switched to the group-wise form."""
    orig_cls, orig_tab = bw.QL, bw.q_fc_tables
    try:
        bw.QL = type("QLGy%d" % G, (QLG,), dict(GROUPS=G))
        bw.q_fc_tables = lambda ac, asc, W, b: group_fc_tables(ac, asc, W, b, G)
        blob, meta, x, y, extra = bw.build_layer(P, upto)
    finally:
        bw.QL, bw.q_fc_tables = orig_cls, orig_tab
    meta["form"] = "w4a8_group"
    meta["stage"] = upto
    meta["group"] = G
    if isinstance(meta.get("shape"), dict):
        meta["shape"]["G"] = G
    wrel = {}
    for k in MATS:
        _, _, deq = gq_int4(np.asarray(P[k], np.float64), G)
        wrel[k] = wmetrics(P[k], deq)[0]
    meta["weight_rel_l2"] = {k: float(v) for k, v in wrel.items()}
    meta["weight_rel_l2_mean"] = float(np.mean(list(wrel.values())))
    meta["layers_note"] = ("1 K-split per distinct input tensor (n1, n2) + per matrix "
                           "G FC + (G-1) ADD")
    return blob, meta, x, y, extra

# ------------------------------------------------------------------ schema probes
def _probe_input(P):
    ch = bw.chain_f32(P)
    n1 = ch["n1"]
    a_c, s_a = bw.s8(n1)
    return ch, n1, a_c, s_a


def build_schema_scale2n(P):
    """Does the stack let a 896-wide int4 weight declare 2*896 scales (2 per channel)?

    Cheapest possible attempt at an attribute-level group scale: keep the tensor 2-D and
    simply give more scales than the quantized_dimension has entries.  Expected: the
    tflite verifier / ncc rejects it (schema: scale count == size of that dimension).
    """
    ch, n1, a_c, s_a = _probe_input(P)
    W = np.asarray(P["Wq"], np.float64)
    s_w = np.maximum(np.abs(W).max(axis=1) / 7.0, 1e-12)
    wq = np.clip(np.round(W / s_w[:, None]), -8, 7).astype(np.int8)
    two = np.stack([s_w, s_w], axis=1).reshape(-1).astype(np.float32)
    o_ref = (a_c.astype(np.float64) * s_a) @ (wq * s_w[:, None]).T
    so = bw.pick_scale(o_ref)
    g = QLG()
    a = g.mk_in("input", (T, D), a_c, s_a)
    tw = g.const_i4("W", wq, two, axis=0)
    g.buffers.append(bo.buf_data(g.b, np.zeros(D, np.int32)))
    tb = g._t("b", [D], bw.I32, len(g.buffers) - 1, (s_w * s_a).tolist(), [0] * D, axis=0)
    t = g.act("out", (T, D), so)
    S = bo.S
    S.FullyConnectedOptionsStart(g.b)
    S.FullyConnectedOptionsAddFusedActivationFunction(g.b, 0)
    S.FullyConnectedOptionsAddWeightsFormat(g.b, 0)
    S.FullyConnectedOptionsAddKeepNumDims(g.b, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(g.b, False)
    opt = S.FullyConnectedOptionsEnd(g.b)
    g._op(bo.BuiltinOperator.FULLY_CONNECTED, [a.i, tw, tb], [t],
          bo.BuiltinOptions.FullyConnectedOptions, opt, version=9)
    blob = g.finish([a.i], [t], "probe: int4 weight [896,896] with 1792 scales")
    codes = np.clip(np.round(o_ref / so), -128, 127).astype(np.int8)
    meta = dict(op="FULLY_CONNECTED", form="probe", probe="scale_count_2N",
                shape=dict(T=T, K=D, N=D, scales=2 * D), dtype="int8",
                input={"name": "input", "shape": [T, D], "type": "int8"},
                output={"name": "out", "shape": [T, D], "type": "int8"},
                quant=dict(input=float(s_a), output=float(so)),
                int8_tol=1, tolerance_rationale=bw.ADR_TOL,
                expected="COMPILE_FAIL: one scale per quantized_dimension entry",
                golden_note="golden assumes the first N scales are per-channel")
    return blob, meta, a_c, codes, dict(fp32=o_ref)


def build_schema_axis1(P):
    """Can the weight be per-INPUT-channel (quantized_dimension=1, K scales)?

    The only other axis the schema offers.  Not group-wise (shared across output rows),
    but if it compiles it pins down which axes exist.
    """
    ch, n1, a_c, s_a = _probe_input(P)
    W = np.asarray(P["Wq"], np.float64)
    s_c = np.maximum(np.abs(W).max(axis=0) / 7.0, 1e-12)
    wq = np.clip(np.round(W / s_c[None, :]), -8, 7).astype(np.int8)
    real = (a_c.astype(np.float64) * s_a) @ (wq.astype(np.float64) * s_c[None, :]).T
    so = bw.pick_scale(real)
    g = QLG()
    a = g.mk_in("input", (T, D), a_c, s_a)
    tw = g.const_i4("W_colscales", wq, s_c, axis=1)
    g.buffers.append(bo.buf_data(g.b, np.zeros(D, np.int32)))
    tb = g._t("b", [D], bw.I32, len(g.buffers) - 1, (s_c * s_a).tolist(), [0] * D, axis=1)
    t = g.act("out", (T, D), so)
    S = bo.S
    S.FullyConnectedOptionsStart(g.b)
    S.FullyConnectedOptionsAddFusedActivationFunction(g.b, 0)
    S.FullyConnectedOptionsAddWeightsFormat(g.b, 0)
    S.FullyConnectedOptionsAddKeepNumDims(g.b, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(g.b, False)
    opt = S.FullyConnectedOptionsEnd(g.b)
    g._op(bo.BuiltinOperator.FULLY_CONNECTED, [a.i, tw, tb], [t],
          bo.BuiltinOptions.FullyConnectedOptions, opt, version=9)
    blob = g.finish([a.i], [t], "probe: int4 weight per INPUT channel (dim=1)")
    codes = np.clip(np.round(real / so), -128, 127).astype(np.int8)
    meta = dict(op="FULLY_CONNECTED", form="probe", probe="quantized_dimension_1",
                shape=dict(T=T, K=D, N=D, scales=D), dtype="int8",
                input={"name": "input", "shape": [T, D], "type": "int8"},
                output={"name": "out", "shape": [T, D], "type": "int8"},
                quant=dict(input=float(s_a), output=float(so)),
                int8_tol=1, tolerance_rationale=bw.ADR_TOL,
                expected="UNKNOWN: accept/reject of a non-zero quantized_dimension",
                golden_note="golden assumes one scale per COLUMN, shared across rows")
    return blob, meta, a_c, codes, dict(fp32=real)

# ------------------------------------------------------- real Qwen2.5 weights (bf16)
REAL_KEYS = dict(Wq="self_attn.q_proj.weight", bq="self_attn.q_proj.bias",
                 Wk="self_attn.k_proj.weight", bk="self_attn.k_proj.bias",
                 Wv="self_attn.v_proj.weight", bv="self_attn.v_proj.bias",
                 Wo="self_attn.o_proj.weight",
                 Wg="mlp.gate_proj.weight", Wu="mlp.up_proj.weight",
                 Wd="mlp.down_proj.weight",
                 g1="input_layernorm.weight", g2="post_attention_layernorm.weight")


def _bf16_to_f32(buf):
    u = np.frombuffer(buf, dtype="<u2")
    return (u.astype(np.uint32) << 16).view(np.float32)


class Safetensors:
    """Minimal read-only safetensors reader (no torch, no safetensors package)."""

    def __init__(self, path):
        self.path = path
        f = open(path, "rb")
        n = int.from_bytes(f.read(8), "little")
        self.head = json.loads(f.read(n).decode("utf-8"))
        self.base = 8 + n
        f.close()

    def names(self):
        return [k for k in self.head if k != "__metadata__"]

    def get(self, name):
        meta = self.head[name]
        b, e = meta["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + b)
            buf = f.read(e - b)
        shape = meta["shape"]
        if meta["dtype"] == "BF16":
            return _bf16_to_f32(buf).reshape(shape)
        if meta["dtype"] == "F16":
            return np.frombuffer(buf, "<f2").astype(np.float32).reshape(shape)
        if meta["dtype"] == "F32":
            return np.frombuffer(buf, "<f4").reshape(shape)
        raise SystemExit("unsupported dtype " + meta["dtype"])


def real_layer(prefix="model.layers.0."):
    """The real layer-0 weights, plus a realistic input from the real embedding table."""
    path = os.environ.get("QWEN25_REAL_SAFETENSORS",
                          "/mnt/d/ai/Qwen2.5-0.5B-Instruct/model.safetensors")
    st = Safetensors(path)
    P = {}
    for k, suffix in REAL_KEYS.items():
        P[k] = np.ascontiguousarray(st.get(prefix + suffix))
    emb = st.get("model.embed_tokens.weight")
    vocab = int(emb.shape[0])
    ids = list(bw.TOKEN_IDS[:T])
    i = 0
    while len(ids) < T:            # same extras as build_w4a8.token_ids
        c = (41 + 1103 * i) % vocab
        i += 1
        if c not in ids:
            ids.append(c)
    x = np.ascontiguousarray(emb[np.asarray(ids)].astype(np.float32))
    return P, x, path


def errcurve_real(Gs=None):
    """The number that matters: int4 group-wise error on the REAL Qwen2.5-0.5B weights.

    The frozen fixture is Gaussian by construction (tools/gen_qwen25_weights.py), so its
    group-wise curve says nothing about the outlier structure of real weights.  This
    reads the real bf16 safetensors (read-only) and repeats the sweep, including the
    whole-layer cosine with real weights and a real embedding-row input.
    """
    Gs = Gs or [1, 2, 4, 8, 16, 32, 64, 128]
    P, x, path = real_layer()
    print("real weights: %s" % path)
    print("=== distribution (int4 cares about the tail, not the std) ===")
    print("  matrix      shape          std      max/std   kurtosis   int4 per-channel rel_l2")
    for k in MATS:
        W = np.asarray(P[k], np.float64)
        sd = float(W.std())
        mx = float(np.abs(W).max())
        kt = float(((W - W.mean()) ** 4).mean() / sd ** 4)
        rel, _ = wmetrics(W, gq_int4(W, 1)[2])
        print("  %-9s %-14s %8.4f %8.2f %9.1f   %.4f"
              % (k, "x".join(map(str, W.shape)), sd, mx / sd, kt, rel))
    print()
    print("=== int4 group-wise error on REAL weights (rel_l2 per matrix) ===")
    out = {"source": path, "groups": Gs, "per_matrix": {}, "layer": {}}
    for G in Gs:
        row = {}
        for k in MATS:
            W = np.asarray(P[k], np.float64)
            _, _, deq = gq_int4(W, G)
            row[k] = wmetrics(W, deq)[0]
        out["per_matrix"][G] = {k: float(v) for k, v in row.items()}
        print("  G=%-4d group=%-4s " % (G, "K/%d" % G)
              + "  ".join("%s %.4f" % (k, row[k]) for k in MATS)
              + "   | mean %.4f" % np.mean(list(row.values())))
    print()
    print("=== whole layer with REAL weights: only weights group-quantised, rest float64 ===")
    print("  G   group(K=896)  out cosine   out rel_l2   probs cosine")
    orig = bw.input_hidden
    try:
        bw.input_hidden = lambda t: np.ascontiguousarray(x[:t])
        ref = bw.chain_f32(P)
        for G in Gs:
            P2 = dict(P)
            for k in MATS:
                P2[k] = gq_int4(np.asarray(P[k], np.float64), G)[2]
            c2 = bw.chain_f32(P2)
            cos = bw.cosine(c2["out"], ref["out"])
            rel = float(np.linalg.norm(c2["out"] - ref["out"]) / np.linalg.norm(ref["out"]))
            cosp = bw.cosine(c2["probs"], ref["probs"])
            out["layer"][G] = dict(cosine=cos, rel_l2=rel, probs_cosine=cosp)
            print("  %-4d %-12d %.7f  %.4e  %.7f" % (G, 896 // G, cos, rel, cosp))
    finally:
        bw.input_hidden = orig
    print()
    print("  int8 per-channel weights for comparison (04c territory):")
    P3 = dict(P)
    for k in MATS:
        P3[k] = gq_int8(np.asarray(P[k], np.float64), 1)[2]
    orig = bw.input_hidden
    try:
        bw.input_hidden = lambda t: np.ascontiguousarray(x[:t])
        c3 = bw.chain_f32(P3)
        ref = bw.chain_f32(P)
    finally:
        bw.input_hidden = orig
    print("    int8 G=1 : out cosine %.7f  rel_l2 %.4e  probs cosine %.7f"
          % (bw.cosine(c3["out"], ref["out"]),
             float(np.linalg.norm(c3["out"] - ref["out"]) / np.linalg.norm(ref["out"])),
             bw.cosine(c3["probs"], ref["probs"])))
    d = os.path.join("ops", "q04b_errcurve")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "errcurve_real.json"), "w"), indent=1)
    print("  -> %s/errcurve_real.json" % d)
    return 0

def fcsim_real(Gs=None):
    """Real weights, FC-only: weight error vs the error of the WHOLE int8 group FC.

    Isolates the cost of the decomposition itself (int8 activation quantisation +
    requant of every partial sum + the ADD chain) from the weight error it buys.
    Same maths as QLG.q_fc, but in numpy only (no graph, no device).
    """
    Gs = Gs or [1, 2, 4, 8, 16, 32, 64]
    P, x, _ = real_layer()
    orig = bw.input_hidden
    try:
        bw.input_hidden = lambda t: np.ascontiguousarray(x[:t])
        ch = bw.chain_f32(P)
    finally:
        bw.input_hidden = orig
    # each matrix is fed its real upstream activation
    UPN = dict(Wq="n1", Wk="n1", Wv="n1", Wo="attn", Wg="n2", Wu="n2", Wd="mlp_y")
    print("=== real weights, 4W8A FC (int8 activation + int4 weights + int8 partial adds) ===")
    print("  matrix  G   weight rel_l2   projection rel_l2   projection cosine")
    for k in MATS:
        W = np.asarray(P[k], np.float64)
        b = np.asarray(P.get("b" + k[1:], np.zeros(W.shape[0])), np.float64)
        a_c, s_a = bw.s8(ch[UPN[k]])
        ref = (a_c.astype(np.float64) * s_a) @ W.T + b
        for G in Gs:
            qg, sw, bq, _, _ = group_fc_tables(a_c, s_a, W, b, G)
            K = W.shape[1]
            parts = [a_c[:, g * (K // G):(g + 1) * (K // G)] for g in range(G)]
            cur = None
            for g in range(G):
                wg = np.asarray(qg[g], np.float64)
                acc = parts[g].astype(np.int64) @ np.asarray(qg[g], np.int64).T + np.asarray(bq[g], np.int64)[None, :]
                real = (parts[g].astype(np.float64) * s_a) @ (wg * np.asarray(sw[g])[:, None]).T \
                    + np.asarray(bq[g], np.float64) * (s_a * np.asarray(sw[g]))[None, :]
                sog = bw.pick_scale(real)
                codes = np.clip(bw.requant_pc(acc, (s_a * np.asarray(sw[g], np.float64)) / sog),
                                -128, 127)
                val = codes.astype(np.float64) * sog
                if cur is None:
                    cur = val
                else:
                    so2 = bw.pick_scale(cur + val)
                    cur = np.round((cur + val) / so2) * so2
            rel = float(np.linalg.norm(cur - ref) / np.linalg.norm(ref))
            wrel = wmetrics(W, gq_int4(W, G)[2])[0]
            print("  %-6s  %-3d %.4f          %.4f              %.7f"
                  % (k, G, wrel, rel, float((cur * ref).sum() /
                                            (np.linalg.norm(cur) * np.linalg.norm(ref)))))
    return 0


def gptq_probe(Gs=None):
    """How much of the 4-bit gap is *rounding strategy* rather than scale granularity?

    Not a deliverable of this ticket, a bound for the next one: per-channel int4 codes are
    a buffer of numbers, so a hessian-aware / error-feedback rounding job can change them
    WITHOUT touching the graph (zero extra layers).  Compared here, in fp64, no int8
    activations, projection error only:
      RTN        - round-to-nearest (what tickets 04/04b do)
      EF         - round-to-nearest + per-row error feedback over K
      GPTQ       - hessian-aware sequential rounding (Frantar et al.), H from real X
    """
    P, x, _ = real_layer()
    orig = bw.input_hidden
    try:
        bw.input_hidden = lambda t: np.ascontiguousarray(x[:t])
        ch = bw.chain_f32(P)
    finally:
        bw.input_hidden = orig
    UPN = dict(Wq="n1", Wk="n1", Wv="n1", Wo="attn", Wg="n2", Wu="n2", Wd="mlp_y")
    print("=== real weights: int4 PER-CHANNEL projection error, rounding strategy ===")
    print("  calibration = first 8 of the 16 rows, evaluation = the other 8 (held out).")
    print("  the last column shows GPTQ on its OWN calibration rows: that number is a")
    print("  trap (16 rows fit 896 columns), it is printed to be seen and ignored.")
    print("  matrix  RTN(held)  GPTQ(held)  GPTQ(cal, trap)")
    for k in MATS:
        W = np.asarray(P[k], np.float64)
        A = np.asarray(ch[UPN[k]], np.float64)
        ref = A @ W.T
        def rel(Wq):
            return float(np.linalg.norm(A @ Wq.T - ref) / np.linalg.norm(ref))
        A_cal, A_ev = A[:8], A[8:]
        ref_ev = A_ev @ W.T
        rtn = gq_int4(W, 1)[2]
        gq = _gptq(A_cal, W)
        def r_ev(Wq):
            return float(np.linalg.norm(A_ev @ Wq.T - ref_ev) / np.linalg.norm(ref_ev))
        def r_cal(Wq):
            rc = A_cal @ W.T
            return float(np.linalg.norm(A_cal @ Wq.T - rc) / np.linalg.norm(rc))
        print("  %-6s  %.4f     %.4f      %.4f"
              % (k, r_ev(rtn), r_ev(gq), r_cal(gq)))
    return 0


def _scale_rows(W, limit):
    return np.maximum(np.abs(W).max(axis=1) / limit, 1e-12)


def _rtn_error_feedback(A, W, limit=7.0):
    """Round each row left to right, pushing the residual into the rest of the row.

    A hessian-free approximation of GPTQ (fixed 0.5 feedback weight).
    """
    N, K = W.shape
    sc = _scale_rows(W, limit)[:, None]
    out = np.zeros_like(W)
    Wc = W.copy()
    for kk in range(K):
        q = np.clip(np.round(Wc[:, kk] / sc[:, 0]), -int(limit) - 1, int(limit))
        out[:, kk] = q
        resid = Wc[:, kk] - q * sc[:, 0]
        if kk + 1 < K:
            Wc[:, kk + 1:] -= resid[:, None] * 0.5
    return out * sc


def _gptq(A, W, limit=7.0, damp=0.05):
    """GPTQ (Frantar et al. 2022) on Y = A @ W.T: sequential over K with H = A^T A."""
    N, K = W.shape
    H = (A.T @ A) / A.shape[0]
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    H += np.eye(K) * damp * float(np.mean(np.diag(H)))
    Hi = np.linalg.inv(H)
    Hi = np.linalg.cholesky(Hi).T          # upper Cholesky of H^-1
    sc = _scale_rows(W, limit)
    Q = np.zeros_like(W)
    Wc = W.copy()
    for kk in range(K):
        q = np.clip(np.round(Wc[:, kk] / sc), -int(limit) - 1, int(limit))
        Q[:, kk] = q
        err = (Wc[:, kk] - q * sc) / Hi[kk, kk]
        if kk + 1 < K:
            Wc[:, kk + 1:] -= err[:, None] * Hi[kk, kk + 1:][None, :]
    return Q * sc[:, None]

# ------------------------------------------------------------------------- driver
SPECS = dict(fc_g=build_fc_g, layer=build_layer_g, errcurve=errcurve,
             errcurve_real=errcurve_real, fcsim_real=fcsim_real, gptq_probe=gptq_probe,
             schema_scale2n=build_schema_scale2n, schema_axis1=build_schema_axis1)


def emit(d, blob, meta, x, y, extra):
    os.makedirs(os.path.join(d, "golden"), exist_ok=True)
    open(os.path.join(d, "model.tflite"), "wb").write(blob)
    np.ascontiguousarray(x).tofile(os.path.join(d, "golden", "input.bin"))
    np.ascontiguousarray(y).tofile(os.path.join(d, "golden", "expected.bin"))
    if extra.get("second") is not None:
        np.ascontiguousarray(extra["second"]).tofile(os.path.join(d, "golden", "input1.bin"))
    meta["tflite"] = "model.tflite"
    meta["tflite_bytes"] = len(blob)
    meta["quality_criterion"] = dict(
        cosine_min=0.9999, source="ADR 0005 (operator level, 8-bit and wider)",
        code_criterion="device vs quantised golden: |delta| <= int8_tol codes",
        four_bit_criterion="device == golden (0/1 code) + cosine(device,golden) - cosine(golden,fp64) <= 0.003")
    if "fp32" in extra:
        ref = np.asarray(extra["fp32"], np.float64)
        so = meta["quant"]["output"]
        dq = (np.asarray(y, np.float64) if so is None
              else np.asarray(y, np.float64) * float(so))
        rel = float(np.linalg.norm(dq - ref) / max(np.linalg.norm(ref), 1e-30))
        meta["accuracy_vs_fp32"] = dict(
            reference="independent float64 chain (tools/build_w4a8.py chain_f32)",
            cosine=bw.cosine(dq, ref), rel_l2=rel,
            max_abs=float(np.abs(dq - ref).max()),
            max_abs_over_maxabs_out=float(np.abs(dq - ref).max() / max(np.abs(ref).max(), 1e-30)),
            out_scale=so, one_code_real=so,
            note="quantisation error of the whole path, no device involved")
        print("  vs fp32 chain: cosine=%.7f rel_l2=%.4e"
              % (meta["accuracy_vs_fp32"]["cosine"], rel))
    json.dump(meta, open(os.path.join(d, "model.json"), "w"), indent=2)
    print("%s  %d B  in%s out%s" % (d, len(blob), np.shape(x), np.shape(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--upto", default="out", choices=bw.STAGES)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--real", action="store_true", help="real Qwen2.5-0.5B weights")
    a = ap.parse_args()
    if a.diag:
        return bw.diag(a.out)
    if a.list or not a.spec:
        print(" ".join(SPECS))
        return 0
    if a.spec not in SPECS:
        raise SystemExit("unknown spec %s" % a.spec)
    if a.spec == "errcurve":
        return errcurve()
    if a.spec == "errcurve_real":
        return errcurve_real()
    if a.spec == "fcsim_real":
        return fcsim_real()
    if a.spec == "gptq_probe":
        return gptq_probe()
    P = bw.fixture()
    if a.real:
        # Same graphs, REAL Qwen2.5-0.5B weights and a real embedding-row input.
        P, x_real, _ = real_layer()
        bw.input_hidden = lambda t: np.ascontiguousarray(x_real[:t])
    if a.spec == "fc_g":
        blob, meta, x, y, extra = build_fc_g(P, a.group)
        d = a.out or ("ops/q04b_fc_g%d" % a.group)
    elif a.spec == "layer":
        blob, meta, x, y, extra = build_layer_g(P, a.upto, a.group)
        d = a.out or ("ops/q04b_layer_g%d" % a.group)
        meta["fp32_ref"] = bw.STAGE_KEY[a.upto]
    else:
        blob, meta, x, y, extra = SPECS[a.spec](P)
        d = a.out or ("ops/q04b_" + a.spec)
    emit(d, blob, meta, x, y, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())

