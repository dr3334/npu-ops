#!/usr/bin/env python3
"""Ticket 05 -- N-layer prefill graph in ONE .dla, weights externalised to a shared bin.

Ticket 05 asks for "24 layers in a single graph + one external shared-weight bin", the
vendor deployment form (docs/OP-shared-weights.md): two graphs compiled normally, then
extract-shared moves the static tensors into one bin and each .dla keeps a single
"size = 0" input pointing at it.

This builder owns the *stacking* half of that seam.  It does NOT re-implement the layer:
every op call below is the same call sequence as tools/build_w4a8.py::build_layer
(w-bits 8 / a-bits 16 / seam native -- the tier ADR 0005 sixth revision locked), with two
changes only:

  1. the tensor-name tag carries a per-layer prefix ("L00_", "L01_", ...), so 24 copies of
     the same op names cannot collide in one flatbuffer;
  2. layer i's input tensor IS layer i-1's output tensor -- there is no graph input between
     layers, so the whole stack is one execution step (if the compiler agrees).

--layers 1 is a regression check: its golden must come out byte-identical to the official
ops/decoder_layer_t16_w8a16_* build of the same tier.

Fixture weights are the ticket-02 frozen fixture.  Layer 0 of it is repeated for every
layer (ticket 05 explicitly allows this): this is WEIGHT REUSE, not a real 24-layer model
-- the numbers say what 24 chained layers of that fixture cost, not what Qwen2.5-0.5B's
24 layers do.

Usage
    # build (writes model.tflite / model.json / golden/{input,expected}.bin)
    /usr/bin/python3 tools/build_24layer.py --layers 24 --t 16 --out ops/q05_prefill_t16_24l
    # device-free numbers: golden vs fp64 chain, per-layer growth, then --device adds the
    # board dump (out_device.bin) comparison
    /usr/bin/python3 tools/build_24layer.py --eval ops/q05_prefill_t16_24l --device
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_w4a8 as bw          # read-only reuse: the layer's validated op sequence


# ------------------------------------------------------------------ tier is locked
def lock_tier(w_bits, a_bits, seam):
    """Refuse anything but the tier ADR 0005 (sixth revision) locked: 8W16A + native."""
    if (w_bits, a_bits, seam) != (8, 16, "native"):
        raise SystemExit("ticket 05 is locked to --w-bits 8 --a-bits 16 --seam native "
                         "(ADR 0005 sixth revision); got %r" % ((w_bits, a_bits, seam),))
    bw.set_tier(w_bits, a_bits)
    bw.SEAM = seam
    bw.set_source("fixture")
    bw.set_input("same")


# --------------------------------------------------------- one layer (tagged, chained)
def _seam_native(g, scores, p):
    """int16 masked SOFTMAX, no fp16 seam -- same 4 layers as build_w4a8._seam_native
    (sym=False branch), only the tensor names carry the layer prefix."""
    mask_codes = np.round(bw.causal_mask() / scores.s).astype(np.int64)
    masked = g.q_add_const(scores, mask_codes, scores.s, p + "masked")
    e = np.full(masked.c.shape, 1, bw.a_dtype())
    ec = bw.TS(g.const_i8(p + "eighth", e, 0.125), e, 0.125)
    scaled = g.q_mul(masked, ec, bw.pick_scale(masked.real / 8.0), p + "scaled")
    pr = g.q_softmax(scaled, p + "probs")
    one = np.full(pr.c.shape, 1, bw.a_dtype())
    t = bw.TS(g.const_i8(p + "one", one, 1.0), one, 1.0)
    return g.q_mul(pr, t, 1.0 / float(bw.aq_max()), p + "p_sym")


def layer(g, x, P, p):
    """One decoder layer: x -> out, appended to the shared graph g.  Returns out TS.

    Call-for-call identical to tools/build_w4a8.py::build_layer(..., upto="out") at the
    locked tier; only p (the layer tag prefix) is new.
    """
    T, D, H, KV, HD, HALF = bw.T, bw.D, bw.H, bw.KV, bw.HD, bw.HALF
    n1 = bw._rmsnorm5(g, x, P["g1"], p + "n1")

    def qkv(W, b, heads, tg, rope=True):
        w, sw, bq, so, _ = bw.q_fc_tables(n1.c, n1.s, W, b)
        y = g.q_fc(n1, w, sw, bq, so, heads * HD, p + tg + "_fc")
        y3 = g.q_shape(y, p + tg + "_3d", (T, heads, HD))
        if not rope:
            return y3
        c, s = bw.rope_tables(heads)
        cc, sc = bw.s8(c)
        ss, sg = bw.s8(s)
        tcc = bw.TS(g.const_i8(p + tg + "_cos", cc, sc), cc, sc)
        tss = bw.TS(g.const_i8(p + tg + "_sin", ss, sg), ss, sg)
        lo_c, hi_c = g.q_split(y3, p + tg + "_split", 2, 2, (T, heads, HALF))
        p1 = g.q_mul(lo_c, tcc, bw.pick_scale(lo_c.real * c), p + tg + "_ac")
        p2 = g.q_mul(hi_c, tss, bw.pick_scale(hi_c.real * s), p + tg + "_bs")
        p3 = g.q_mul(hi_c, tcc, bw.pick_scale(hi_c.real * c), p + tg + "_bc")
        p4 = g.q_mul(lo_c, tss, bw.pick_scale(lo_c.real * s), p + tg + "_as")
        lo_r = lo_c.real * c - hi_c.real * s
        hi_r = hi_c.real * c + lo_c.real * s
        s_rope = bw.pick_scale(np.concatenate([lo_r, hi_r], axis=2))
        lo = g.q_addsub(p1, p2, s_rope, p + tg + "_lo", minus=True)
        hi = g.q_addsub(p3, p4, s_rope, p + tg + "_hi")
        return g.q_concat([lo, hi], p + tg + "_rope", 2, (T, heads, HD), so=s_rope)

    q3 = qkv(P["Wq"], P["bq"], H, "q")
    k3 = qkv(P["Wk"], P["bk"], KV, "k")
    wv, swv, bqv, s_vfc, _ = bw.q_fc_tables(n1.c, n1.s, P["Wv"], P["bv"])
    v_fc = g.q_fc(n1, wv, swv, bqv, s_vfc, KV * HD, p + "v_fc")
    v3 = g.q_shape(v_fc, p + "v_3d", (T, KV, HD))

    def kv14(t3, tg):
        k0, k1 = g.q_split(t3, p + tg + "_split2", 1, 2, (T, 1, HD))
        return g.q_concat([k0] * bw.KVR + [k1] * bw.KVR, p + tg + "14", 1, (T, H, HD))

    k14, v14 = kv14(k3, "k"), kv14(v3, "v")
    qt = g.q_transpose(q3, p + "qt", [1, 0, 2])
    kt = g.q_transpose(k14, p + "kt", [1, 0, 2])
    vt = g.q_transpose(v14, p + "vt", [1, 0, 2])
    scores = g.q_bmm(qt, kt, bw.pick_scale(scores_ref(n1, P)), p + "scores", adj_y=True)
    p_sym = _seam_native(g, scores, p)
    a3 = g.q_bmm(p_sym, vt, bw.pick_scale(np.transpose(p_sym.real, (0, 1, 2))
                                          @ np.transpose(v14.real, (1, 0, 2))), p + "a3")
    at = g.q_transpose(a3, p + "at", [1, 0, 2])
    attn = g.q_shape(at, p + "attn", (T, D))
    wo, swo, bqo, s_o, _ = bw.q_fc_tables(attn.c, attn.s, P["Wo"], np.zeros(D))
    o = g.q_fc(attn, wo, swo, bqo, s_o, D, p + "o_proj")
    r1 = g.q_addsub(o, x, bw.pick_scale(o.real + x.real), p + "r1")
    n2 = bw._rmsnorm5(g, r1, P["g2"], p + "n2")
    wg, swg, bqg, s_gate, _ = bw.q_fc_tables(n2.c, n2.s, P["Wg"], np.zeros(bw.INTER))
    gate = g.q_fc(n2, wg, swg, bqg, s_gate, bw.INTER, p + "gate")
    wu, swu, bqu, s_up, _ = bw.q_fc_tables(n2.c, n2.s, P["Wu"], np.zeros(bw.INTER))
    up = g.q_fc(n2, wu, swu, bqu, s_up, bw.INTER, p + "up")
    sig = g.q_logistic(gate, p + "sig")
    s1 = g.q_mul(gate, sig, bw.pick_scale(gate.real * sig.real), p + "silu")
    y = g.q_mul(s1, up, bw.pick_scale(s1.real * up.real), p + "mlp_y")
    wd, swd, bqd, s_down, _ = bw.q_fc_tables(y.c, y.s, P["Wd"], np.zeros(D))
    down = g.q_fc(y, wd, swd, bqd, s_down, D, p + "down")
    return g.q_addsub(down, r1, bw.pick_scale(down.real + r1.real), p + "out")


# ------------------------------------------------------------------- fp64 reference
def scores_ref(n1, P):
    """fp64 scores of this layer, used only to pick the scores output scale.

    build_layer picks it from the fp64 chain of the SAME layer; here the layer input is a
    quantised tensor, so the reference is recomputed from n1 (dequantised) directly.
    """
    T, H, KV, HD, HALF = bw.T, bw.H, bw.KV, bw.HD, bw.HALF

    def proj_rope(W, b, heads):
        y = (n1.real @ np.asarray(W, np.float64).T + np.asarray(b, np.float64)
             ).reshape(T, heads, HD)
        c, s = bw.rope_tables(heads)
        a, bb = y[..., :HALF], y[..., HALF:]
        return np.concatenate([a * c - bb * s, bb * c + a * s], axis=-1)

    q3 = proj_rope(P["Wq"], P["bq"], H)
    k3 = proj_rope(P["Wk"], P["bk"], KV)
    k14 = np.repeat(k3, bw.KVR, axis=1)
    return np.transpose(q3, (1, 0, 2)) @ np.transpose(k14, (1, 2, 0))


def fp64_chain(P, x0, n):
    """Independent float64 chain of the WHOLE stack (AGENTS R3: not the fb code path).

    build_w4a8.chain_f32() reads its input from input_hidden(T); feeding one layer's
    output into the next is done by rebinding that loader for the duration of the call,
    so the reference math stays exactly the validated code (nothing is copied).
    """
    orig = bw.input_hidden
    x = np.asarray(x0, np.float64)
    rows = []
    try:
        for i in range(n):
            bw.input_hidden = (lambda t, _x=x: _x)
            ch = bw.chain_f32(P)
            rows.append(dict(layer=i, max_abs=float(np.abs(ch["out"]).max()),
                             rms=float(np.sqrt((ch["out"] ** 2).mean())),
                             max_abs_scores=float(np.abs(ch["scores"]).max())))
            x = ch["out"]
    finally:
        bw.input_hidden = orig
    return x, rows


# ------------------------------------------------------------------------- the stack
WKEYS = ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd")


def build_stack(P, n, t, distinct=0.0):
    """One graph, one input, one output, n layers between them sharing P.

    distinct > 0 is a CONTROL, not a quality run: it scales layer i's FC weights by
    1 + distinct*i so the layers stop holding byte-identical tensors.  It answers one
    question only -- whether ncc deduplicated the 24 identical copies (Static 15M for 24
    layers says it did; docs/OP-24layer-shared.md section 2).
    """
    x0 = bw.input_hidden(t).astype(np.float64)
    codes, s_x = bw.s8(x0)
    g = bw.QL()
    x = g.mk_in("x", (t, bw.D), codes, s_x)
    for i in range(n):
        Pi = P if not distinct else dict(
            P, **{k: np.asarray(P[k], np.float64) * (1.0 + distinct * i) for k in WKEYS})
        x = layer(g, x, Pi, "L%02d_" % i)
    blob = g.finish([g.idx["x"]], [x.i],
                    "qwen2.5-0.5b prefill %dlayer T=%d 8W16A native (ticket 05)" % (n, t))
    return blob, x, codes, s_x, x0


def meta_for(n, t, blob, x, codes, s_x, ref, rows):
    """model.json: IO contract + tolerances REGISTERED BEFORE the first board run."""
    s_o = x.s
    dq = x.c.astype(np.float64) * s_o
    relg = float(np.linalg.norm(dq - ref) / max(np.linalg.norm(ref), 1e-30))
    cosg = bw.cosine(dq, ref)
    return dict(
        op="DECODER_LAYER_STACK", form=bw.FORM, stage="out", layers=n, stacked=True,
        graph="one tflite, no graph input/output between layers",
        shape=dict(T=t, D=bw.D, H=bw.H, KV=bw.KV, HD=bw.HD, I=bw.INTER),
        dtype=bw.a_name(), weights_dtype="int8",
        input={"name": "x", "shape": [t, bw.D], "type": bw.a_name()},
        output={"name": "out", "shape": [t, bw.D], "type": bw.a_name()},
        quant={"input": s_x, "output": s_o},
        weights_note=("fixture layer 0 repeated for all %d layers (ticket 05 allows weight "
                      "reuse): this graph is NOT a real 24-layer Qwen2.5-0.5B stack" % n),
        # ---- registered before the first board run (docs/OP-24layer-shared.md) ----
        int8_tol=1200,
        tolerance_rationale=(
            "Code level on a 24-layer CHAIN is not the single-layer contract. Registered "
            "before the first board run: int8_tol=1200 codes, derived as sqrt(24) x the "
            "per-layer device-vs-golden max of the same tier (ticket 04f measured 247 codes "
            "on one layer at 8W16A native; sqrt(24)*247 = 1210 -> 1200), i.e. the "
            "independent-rounding random-walk bound. It is a PROPAGATION bound, not a "
            "relaxation of the per-layer contract; the measured value is reported as-is "
            "and a miss is reported as FAIL."),
        abs_dcos_device_vs_golden_max=0.003,
        rel_l2_device_vs_golden_max=0.05,
        rel_l2_rationale=(
            "Registered before the first board run: one layer's device-vs-golden rel_l2 at "
            "this tier is ~1e-2 (from ticket 04f abs(dcos) 4.6e-5); a 24-layer chain adds "
            "rounding per layer, so 5e-2 is the registered cap (sqrt(24)*1e-2 = 4.9e-2)."),
        quality_criterion=dict(
            cosine_min=0.9998,
            source=("ADR 0005 sixth revision: int8 weights -> quality EXPECTATION 0.9998; "
                    "24 chained layers accumulate, so a miss is reported, not smoothed. "
                    "The real quality gate is the logits (ticket 06)."),
            code_criterion="device vs quantised golden: |delta| <= int8_tol codes"),
        accuracy_vs_fp32=dict(
            reference="independent float64 chain, all layers (tools/build_24layer.py)",
            cosine=cosg, rel_l2=relg,
            max_abs=float(np.abs(dq - ref).max()),
            out_scale=s_o, one_code_real=float(s_o)),
        per_layer_fp64=rows,
    )


def build(args):
    t = args.t
    bw.T = t
    P = bw.fixture()
    blob, x, codes, s_x, x0 = build_stack(P, args.layers, t, args.distinct)
    ref, rows = fp64_chain(P, x0, args.layers)
    d = args.out
    os.makedirs(os.path.join(d, "golden"), exist_ok=True)
    open(os.path.join(d, "model.tflite"), "wb").write(blob)
    codes.tofile(os.path.join(d, "golden", "input.bin"))
    x.c.tofile(os.path.join(d, "golden", "expected.bin"))
    meta = meta_for(args.layers, t, blob, x, codes, s_x, ref, rows)
    meta["tflite"] = "model.tflite"
    meta["tflite_bytes"] = len(blob)
    meta["precision_tier"] = dict(
        weights="int8, symmetric per-output-channel", activations="int16 per-tensor",
        bias="int32 per-channel", requant="tools/quant_contract.py (two roundings)",
        builder="tools/build_24layer.py --layers %d --t %d" % (args.layers, t))
    meta["weights_source"] = dict(
        source="fixture", fixture="weights/qwen25_frozen (ticket 02 frozen fixture)",
        layer=0, input_rows="embedding rows of token ids %s" % bw.token_ids(t))
    json.dump(meta, open(os.path.join(d, "model.json"), "w"), indent=2)
    print("%s  %d B  layers=%d T=%d  in%s out%s" % (d, len(blob), args.layers, t,
                                                    codes.shape, x.c.shape))
    print("golden vs fp64 chain: cosine=%.7f rel_l2=%.4e"
          % (bw.cosine(x.c.astype(np.float64) * x.s, ref),
             float(np.linalg.norm(x.c.astype(np.float64) * x.s - ref)
                   / np.linalg.norm(ref))))
    return 0


# ---------------------------------------------------------------------------- eval
def _codes(a, s):
    return a.astype(np.float64) * float(s)


def evaluate(opdir, use_device):
    meta = json.load(open(os.path.join(opdir, "model.json")))
    shape = tuple(meta["output"]["shape"])
    n = meta["layers"]
    t = meta["shape"]["T"]
    s_o = float(meta["quant"]["output"])
    bw.T = t
    lock_tier(8, 16, "native")
    P = bw.fixture()
    exp = np.fromfile(os.path.join(opdir, "golden", "expected.bin"), np.int16).reshape(shape)
    x0 = np.fromfile(os.path.join(opdir, "golden", "input.bin"), np.int16).reshape(shape)
    ref, rows = fp64_chain(P, _codes(x0, meta["quant"]["input"]), n)
    g = _codes(exp, s_o)
    cosg = bw.cosine(g, ref)
    relg = float(np.linalg.norm(g - ref) / np.linalg.norm(ref))
    print("stack      : %d layers, T=%d, %s, out scale %.6e (1 code = %.3e)"
          % (n, t, meta["dtype"], s_o, s_o))
    print("golden     : cosine vs fp64 %.7f   rel_l2 %.4e   max|fp64| %.4g"
          % (cosg, relg, float(np.abs(ref).max())))
    print("per-layer  : max|out| %s" % " ".join("%.2f" % r["max_abs"] for r in rows))
    out = dict(golden_vs_fp64=dict(cosine=cosg, rel_l2=relg))
    if not use_device:
        return out
    dev = np.fromfile(os.path.join(opdir, "out_device.bin"), np.int16).reshape(shape)
    a = np.abs(dev.astype(np.int64) - exp.astype(np.int64))
    d = _codes(dev, s_o)
    cosd = bw.cosine(d, ref)
    rel_df = float(np.linalg.norm(d - ref) / np.linalg.norm(ref))
    cos_dg = bw.cosine(d, g)
    rel_dg = float(np.linalg.norm(d - g) / max(np.linalg.norm(g), 1e-30))
    print("device     : cosine vs fp64 %.7f   rel_l2 vs fp64 %.4e" % (cosd, rel_df))
    print("device     : cosine vs golden %.7f   abs(dcos) %.4e   rel_l2 %.4e"
          % (cos_dg, abs(cosd - cosg), rel_dg))
    print("codes      : exact=%d/%d  max=%d  mean=%.2f  p99=%d"
          % (int((a == 0).sum()), a.size, int(a.max()), a.mean(),
             int(np.percentile(a, 99))))
    gate_codes = int(a.max()) <= int(meta["int8_tol"])
    gate_cos = abs(cosd - cosg) <= float(meta["abs_dcos_device_vs_golden_max"])
    gate_rel = rel_dg <= float(meta["rel_l2_device_vs_golden_max"])
    q_min = float(meta["quality_criterion"]["cosine_min"])
    print("gate       : abs(dcos)<=%.3f %s   rel_l2<=%.3f %s   codes<=%d %s"
          % (meta["abs_dcos_device_vs_golden_max"], "PASS" if gate_cos else "FAIL",
             meta["rel_l2_device_vs_golden_max"], "PASS" if gate_rel else "FAIL",
             meta["int8_tol"], "PASS" if gate_codes else "FAIL"))
    print("quality    : device cosine %.7f vs expectation %.4f -> %s"
          % (cosd, q_min, "PASS" if cosd >= q_min else "FAIL"))
    out.update(device_vs_fp64=dict(cosine=cosd, rel_l2=rel_df),
               device_vs_golden=dict(cosine=cos_dg, abs_dcos=abs(cosd - cosg),
                                     rel_l2=rel_dg, codes_max=int(a.max()),
                                     codes_mean=float(a.mean()),
                                     code_exact=int((a == 0).sum()), code_n=int(a.size)),
               gate=dict(codes=gate_codes, dcos=gate_cos, rel_l2=gate_rel,
                         quality=bool(cosd >= q_min)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=16)
    ap.add_argument("--seam", default="native")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval", default=None, metavar="OPDIR")
    ap.add_argument("--device", action="store_true")
    ap.add_argument("--distinct", type=float, default=0.0,
                    help="CONTROL only: scale layer i FC weights by 1+distinct*i so the "
                         "layers stop being byte-identical (dedupe probe, not a quality run)")
    a = ap.parse_args()
    lock_tier(a.w_bits, a.a_bits, a.seam)
    if a.eval:
        res = evaluate(a.eval, a.device)
        json.dump(res, open(os.path.join(a.eval, "eval.json"), "w"), indent=2)
        return 0
    a.out = a.out or ("ops/q05_prefill_t%d_%dl" % (a.t, a.layers))
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
