#!/usr/bin/env python3
"""GQA scores for Qwen2.5-0.5B: in-graph KV head replication + one 3-D BATCH_MATMUL.

What this op is (ticket 04, feature qwen25-attention)
-----------------------------------------------------
Real model: 14 query heads, 2 KV heads (GQA 7:1), head_dim 64, T=8, fp16.
The compiler does NOT broadcast the 2 KV heads to 14 -- the replication has to happen
inside the graph with already-proven shape ops, and the data must never return to host.

    q     [T=8, Hq=14, HD=64]  fp16   (graph input 0)
    k     [T=8, Hkv=2, HD=64]  fp16   (graph input 1)

      SPLIT(k, axis=1, 2)                     -> k0,k1        [8,1,64] each   (1 layer)
      CONCAT([k0]*7, axis=1)                  -> ka           [8,7,64]        (1 layer)
      CONCAT([k1]*7, axis=1)                  -> kb           [8,7,64]        (1 layer)
      CONCAT([ka,kb], axis=1)                 -> k14          [8,14,64]       (1 layer)
      TRANSPOSE(k14, perm=[1,0,2])            -> kt           [14,8,64]       (1 layer)
      TRANSPOSE(q,   perm=[1,0,2])            -> qt           [14,8,64]       (1 layer)
      BATCH_MATMUL(qt, kt, adj_y=1)           -> scores       [14,8,8]        (1 layer)

Head mapping: [k0 x7 | k1 x7] along the head axis is exactly repeat_interleave(7),
i.e. query head h reads KV head h//7 -- the real GQA mapping.

Why 3-D and not 4-D: [H,T,HD] / [H,T,T] keep one native BatchMatMulLayer;
4-D [1,H,T,HD] gets lowered to 5 layers (docs/OP-gqa.md, docs/OP-bmm.md).

Why the two TRANSPOSEs are mandatory (NOTES 10.2): feeding [T,H,HD] straight into the
BMM (or RESHAPEing [T,H] to [H,T,HD]) compiles and runs but computes the wrong
mathematics -- it silently scores the wrong axis pairs. The layout change has to be a real
TRANSPOSE [T,H,HD] -> [H,T,HD]. This builder always emits it.

Where q/k come from (recorded choice, the ticket asks explicitly)
-----------------------------------------------------------------
From the frozen fixture (ticket 02), same convention as ticket 03:
    x  = N(0,1) float32 [T,896]              (deterministic, seed below)
    q  = RoPE(q_proj x + b_q, split-half)    -> fp16 [8,14,64]
    k  = RoPE(k_proj x + b_k, split-half)    -> fp16 [8,2,64]
The bias part dominates the real Qwen2.5-0.5B attention (measured std 7.85 / 32.2, see
docs/OP-weights.md), so this keeps the dynamic range honest instead of the usual ~1.

The graph inputs are therefore golden/input.bin (q) and golden/input1.bin (k);
golden/expected.bin is a numpy float32 contraction of *those exact fp16 inputs*,
sharing no code with the flatbuffer writer (R3).

No scaling, no mask: those belong to ticket 05. This op is scores only.

Variants (layer-cost measurement, compile side only)
----------------------------------------------------
    main      the graph above (default, ops/gqa_scores_t8_fp16)
    bmm_only  same minus the replication (k arrives already 14-headed): isolates the
              cost of the SPLIT+CONCAT block
    twox7     the "2x7 heads" fallback: two 7-head BMMs, one KV head each
    unroll14  the "14 single-head BMM" fallback: 14 BMMs + CONCAT

Usage
-----
    /usr/bin/python3 tools/build_gqa_scores.py                       # main -> ops/gqa_scores_t8_fp16
    /usr/bin/python3 tools/build_gqa_scores.py --variant unroll14 --out ops/_gqa_variants/unroll14
    /usr/bin/python3 tools/build_gqa_scores.py --tol-analysis        # fp16 lower bound
"""
import argparse
import json
import os
import sys

import flatbuffers
import numpy as np

import build_ops as bo

T, HQ, HKV, HD = 8, 14, 2, 64
HIDDEN = 896
THETA = 1e6
SEED = 20261113
FP16_ULP_REL = 2.0 ** -11          # fp16 machine epsilon (11-bit significand)

MAIN = "main"
VARIANTS = ("main", "bmm_only", "nary14", "twox7", "unroll14", "wrong_reshape",
            "rank5")


# --------------------------------------------------------------------- q/k source
def _rope_split_half(x32, theta=THETA):
    """Qwen2 / HF RoPE, split-half convention (docs/OP-rope.md 2).  x32: [T,H,HD] f32."""
    t = x32.shape[0]
    half = x32.shape[-1] // 2
    inv = 1.0 / (theta ** (np.arange(0, half * 2, 2, dtype=np.float64) / (half * 2)))
    ang = np.outer(np.arange(t, dtype=np.float64), inv)            # [T, half]
    cos = np.cos(ang).astype(np.float32)[:, None, :]
    sin = np.sin(ang).astype(np.float32)[:, None, :]
    a, b = x32[..., :half], x32[..., half:]
    out = np.empty_like(x32)
    out[..., :half] = a * cos - b * sin     # out_lo = a*cos - b*sin
    out[..., half:] = b * cos + a * sin     # out_hi = b*cos + a*sin
    return out


def make_qk(weights_dir=None):
    """(q [T,HQ,HD] fp16, k [T,HKV,HD] fp16, provenance dict) from the frozen fixture."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import qwen25_weights as qw

    wq = qw.get_fp32("layer0.attn.q_proj.weight", path=weights_dir)
    bq = qw.get_fp32("layer0.attn.q_proj.bias", path=weights_dir)
    wk = qw.get_fp32("layer0.attn.k_proj.weight", path=weights_dir)
    bk = qw.get_fp32("layer0.attn.k_proj.bias", path=weights_dir)

    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((T, HIDDEN)).astype(np.float32)        # RMS ~ 1, as after a norm
    q32 = (x @ wq.T + bq).reshape(T, HQ, HD)
    k32 = (x @ wk.T + bk).reshape(T, HKV, HD)
    q = _rope_split_half(q32).astype(np.float16)
    k = _rope_split_half(k32).astype(np.float16)

    man = qw.manifest(path=weights_dir)
    prov = {"fixture": str(qw.resolve_dir(weights_dir)),
            "weights_digest": man.get("weights_digest"),
            "hidden": "N(0,1) float32 [%d,%d], numpy default_rng seed %d" % (T, HIDDEN, SEED),
            "rope": "split-half, theta=%g, positions 0..%d" % (THETA, T - 1),
            "tensors": ["layer0.attn.q_proj.weight", "layer0.attn.q_proj.bias",
                        "layer0.attn.k_proj.weight", "layer0.attn.k_proj.bias"]}
    return q, k, prov


# ------------------------------------------------------------------ golden (pure numpy)
def reference_scores(q16, k16):
    """scores[h,t,s] = sum_d q[t,h,d] * k[s,h//7,d]; float32 accumulate -> float16."""
    qt = np.transpose(q16.astype(np.float32), (1, 0, 2))                  # [Hq,T,HD]
    kt = np.transpose(k16.astype(np.float32), (1, 0, 2))                  # [Hkv,T,HD]
    kre = np.repeat(kt, HQ // HKV, axis=0)                                # [Hq,T,HD]
    assert kre.shape == qt.shape, (kre.shape, qt.shape)
    acc = np.einsum("htd,hsd->hts", qt, kre)                              # [Hq,T,T]
    return np.ascontiguousarray(acc.astype(np.float16))


def reference_scores_f16acc(q16, k16):
    """Independent fp16 lower bound: sequential fp16 accumulation over d (K=64).

    The device accumulates in unknown precision; this is the pessimistic bound (every
    partial sum rounded to fp16).  Used to justify the registered tolerance BEFORE the
    board run (NOTES 10.1) -- it never touches the compiled model.
    """
    qt = np.transpose(q16.astype(np.float32), (1, 0, 2))
    kt = np.repeat(np.transpose(k16.astype(np.float32), (1, 0, 2)), HQ // HKV, axis=0)
    acc = np.zeros((HQ, T, T), dtype=np.float16)
    for d in range(HD):
        prod = np.einsum("ht,hs->hts", qt[:, :, d], kt[:, :, d]).astype(np.float16)
        acc = (acc.astype(np.float32) + prod.astype(np.float32)).astype(np.float16)
    return acc


# ------------------------------------------------------------------ flatbuffer plumbing
class M:
    """Tiny fp16 flatbuffer accumulator (buffer 0 stays the empty placeholder)."""

    def __init__(self):
        self.b = flatbuffers.Builder(16384)
        self.buffers = [bo.buf_empty(self.b)]
        self.tensors = []
        self.opcodes = []
        self.ops = []
        self._oc = {}

    def buf(self, arr):
        self.buffers.append(bo.buf_data(self.b, arr))
        return len(self.buffers) - 1

    def const(self, name, shape, ttype, arr):
        self.tensors.append(bo.tensor(self.b, name, list(shape), ttype, self.buf(arr)))
        return len(self.tensors) - 1

    def io(self, name, shape):
        self.tensors.append(bo.tensor(self.b, name, list(shape), bo.F16, 0))
        return len(self.tensors) - 1

    def out(self, name, shape):
        self.buffers.append(bo.buf_empty(self.b))
        self.tensors.append(bo.tensor(self.b, name, list(shape), bo.F16,
                                      len(self.buffers) - 1))
        return len(self.tensors) - 1

    def oc(self, builtin):
        if builtin not in self._oc:
            self.opcodes.append(bo.opcode(self.b, builtin))
            self._oc[builtin] = len(self.opcodes) - 1
        return self._oc[builtin]

    def add_op(self, builtin, ins, outs, opt_type=None, opt=None):
        self.ops.append(bo.operator(self.b, self.oc(builtin), list(ins), list(outs),
                                    opt_type, opt))

    def finish(self, ins, outs, desc):
        return bo.finish(self.b, self.buffers, self.opcodes, self.tensors, self.ops,
                         ins, outs, desc)

    # ---- ops
    def split(self, x, shape, axis, num, names):
        b = self.b
        ax = self.const(names[0] + "_axis", [1], bo.I32, np.asarray([axis], np.int32))
        part = list(shape)
        part[axis] //= num
        outs = [self.out(nm, part) for nm in names]
        bo.S.SplitOptionsStart(b)
        bo.S.SplitOptionsAddNumSplits(b, num)
        opt = bo.S.SplitOptionsEnd(b)
        self.add_op(bo.BuiltinOperator.SPLIT, [ax, x], outs,
                    bo.BuiltinOptions.SplitOptions, opt)
        return outs

    def concat(self, xs, out_shape, axis, name):
        b = self.b
        t = self.out(name, out_shape)
        bo.S.ConcatenationOptionsStart(b)
        bo.S.ConcatenationOptionsAddAxis(b, axis)
        bo.S.ConcatenationOptionsAddFusedActivationFunction(b, 0)     # NONE
        opt = bo.S.ConcatenationOptionsEnd(b)
        self.add_op(bo.BuiltinOperator.CONCATENATION, xs, [t],
                    bo.BuiltinOptions.ConcatenationOptions, opt)
        return t

    def transpose(self, x, perm, out_shape, name):
        b = self.b
        tp = self.const(name + "_perm", [len(perm)], bo.I32, np.asarray(perm, np.int32))
        t = self.out(name, out_shape)
        bo.S.TransposeOptionsStart(b)
        opt = bo.S.TransposeOptionsEnd(b)
        self.add_op(bo.BuiltinOperator.TRANSPOSE, [x, tp], [t],
                    bo.BuiltinOptions.TransposeOptions, opt)
        return t

    def bmm(self, a, bt, out_shape, name, adj_y=True):
        b = self.b
        t = self.out(name, out_shape)
        bo.S.BatchMatMulOptionsStart(b)
        bo.S.BatchMatMulOptionsAddAdjX(b, False)
        bo.S.BatchMatMulOptionsAddAdjY(b, bool(adj_y))
        bo.S.BatchMatMulOptionsAddAsymmetricQuantizeInputs(b, False)
        opt = bo.S.BatchMatMulOptionsEnd(b)
        self.add_op(bo.BuiltinOperator.BATCH_MATMUL, [a, bt], [t],
                    bo.BuiltinOptions.BatchMatMulOptions, opt)
        return t

    def reshape(self, x, in_shape, out_shape, name):
        """RESHAPE: shape tensor + ReshapeOptions.new_shape, like the TFLite converter."""
        b = self.b
        shp = self.const(name + "_shape", [len(out_shape)], bo.I32,
                         np.asarray(out_shape, np.int32))
        nsv = b.CreateNumpyVector(np.asarray(out_shape, np.int32))
        t = self.out(name, out_shape)
        bo.S.ReshapeOptionsStart(b)
        bo.S.ReshapeOptionsAddNewShape(b, nsv)
        opt = bo.S.ReshapeOptionsEnd(b)
        self.add_op(bo.BuiltinOperator.RESHAPE, [x, shp], [t],
                    bo.BuiltinOptions.ReshapeOptions, opt)
        return t


# --------------------------------------------------------------------------- graphs
def build_main(q, k):
    """SPLIT -> CONCAT(x7,x7) -> TRANSPOSE -> 3-D BMM adj_y=1."""
    m = M()
    tq = m.io("q", [T, HQ, HD])
    tk = m.io("k", [T, HKV, HD])
    k0, k1 = m.split(tk, [T, HKV, HD], axis=1, num=2, names=["k0", "k1"])
    rep = HQ // HKV
    ka = m.concat([k0] * rep, [T, rep, HD], 1, "krep0")      # 7 identical inputs
    kb = m.concat([k1] * rep, [T, rep, HD], 1, "krep1")
    k14 = m.concat([ka, kb], [T, HQ, HD], 1, "k14")
    kt = m.transpose(k14, [1, 0, 2], [HQ, T, HD], "kt")
    qt = m.transpose(tq, [1, 0, 2], [HQ, T, HD], "qt")
    sc = m.bmm(qt, kt, [HQ, T, T], "scores", adj_y=True)
    return m.finish([0, 1], [sc], "gqa scores T8 14Q/2KV hd64 fp16: split+concat x7 + 3-D bmm")


def build_bmm_only(q, k14):
    """Baseline: replication removed (k arrives already 14-headed)."""
    m = M()
    tq = m.io("q", [T, HQ, HD])
    tk = m.io("k14", [T, HQ, HD])
    kt = m.transpose(tk, [1, 0, 2], [HQ, T, HD], "kt")
    qt = m.transpose(tq, [1, 0, 2], [HQ, T, HD], "qt")
    sc = m.bmm(qt, kt, [HQ, T, T], "scores", adj_y=True)
    return m.finish([0, 1], [sc], "bmm_only: no replication (cost baseline)")


def build_twox7(q, k):
    """Fallback '2x7 heads': two 7-head BMMs, one KV head each (NOT adopted)."""
    m = M()
    tq = m.io("q", [T, HQ, HD])
    tk = m.io("k", [T, HKV, HD])
    qg = m.split(tq, [T, HQ, HD], axis=1, num=2, names=["qg0", "qg1"])
    k0, k1 = m.split(tk, [T, HKV, HD], axis=1, num=2, names=["k0", "k1"])
    rep = HQ // HKV
    ka = m.concat([k0] * rep, [T, rep, HD], 1, "krep0")
    kb = m.concat([k1] * rep, [T, rep, HD], 1, "krep1")
    s = []
    for i, (qgi, ki) in enumerate(((qg[0], ka), (qg[1], kb))):
        kti = m.transpose(ki, [1, 0, 2], [rep, T, HD], "kt%d" % i)
        qti = m.transpose(qgi, [1, 0, 2], [rep, T, HD], "qt%d" % i)
        s.append(m.bmm(qti, kti, [rep, T, T], "scores%d" % i, adj_y=True))
    out = m.concat(s, [HQ, T, T], 0, "scores")
    return m.finish([0, 1], [out], "twox7 fallback: 2 groups of 7 heads")


def build_unroll14(q, k):
    """Fallback '14 single-head unroll': 14 3-D BMMs on [1,T,HD] + CONCAT (NOT adopted)."""
    m = M()
    tq = m.io("q", [T, HQ, HD])
    tk = m.io("k", [T, HKV, HD])
    qt = m.transpose(tq, [1, 0, 2], [HQ, T, HD], "qt")
    kt = m.transpose(tk, [1, 0, 2], [HKV, T, HD], "kt")
    qs = m.split(qt, [HQ, T, HD], axis=0, num=HQ, names=["q%d" % i for i in range(HQ)])
    ks = m.split(kt, [HKV, T, HD], axis=0, num=HKV, names=["k%d" % i for i in range(HKV)])
    s = []
    for h in range(HQ):
        s.append(m.bmm(qs[h], ks[h // (HQ // HKV)], [1, T, T], "s%d" % h, adj_y=True))
    out = m.concat(s, [HQ, T, T], 0, "scores")
    return m.finish([0, 1], [out], "unroll14 fallback: 14 single-head BMMs")


def build_wrong_reshape(q, k):
    """CONTROL (deliberately wrong): [T,H,HD] -> RESHAPE [H,T,HD] instead of TRANSPOSE.

    This is the failure mode NOTES 10.2 warns about: it compiles, runs in one execution
    step, and produces plausible-looking numbers that are mathematically wrong.  Kept in
    the tree as the control that proves the board comparison is not vacuous.
    """
    m = M()
    tq = m.io("q", [T, HQ, HD])
    tk = m.io("k", [T, HKV, HD])
    k0, k1 = m.split(tk, [T, HKV, HD], axis=1, num=2, names=["k0", "k1"])
    rep = HQ // HKV
    ka = m.concat([k0] * rep, [T, rep, HD], 1, "krep0")
    kb = m.concat([k1] * rep, [T, rep, HD], 1, "krep1")
    k14 = m.concat([ka, kb], [T, HQ, HD], 1, "k14")
    kt = m.reshape(k14, [T, HQ, HD], [HQ, T, HD], "kt_wrong")
    qt = m.reshape(tq, [T, HQ, HD], [HQ, T, HD], "qt_wrong")
    sc = m.bmm(qt, kt, [HQ, T, T], "scores", adj_y=True)
    return m.finish([0, 1], [sc], "CONTROL wrong layout: RESHAPE instead of TRANSPOSE")


def build_nary14(q, k):
    """Cheaper replication shape (compile-only probe): one 14-input CONCAT.

    main uses 3 concatenations ([k0]*7, [k1]*7, then merge).  TFLite allows a single
    CONCATENATION with N inputs, so the whole replication could be ONE layer.  Measured
    here to see what ncc actually emits (docs section on the replication cost).
    """
    m = M()
    tq = m.io("q", [T, HQ, HD])
    tk = m.io("k", [T, HKV, HD])
    k0, k1 = m.split(tk, [T, HKV, HD], axis=1, num=2, names=["k0", "k1"])
    rep = HQ // HKV
    k14 = m.concat([k0] * rep + [k1] * rep, [T, HQ, HD], 1, "k14")   # 14 inputs, 1 layer
    kt = m.transpose(k14, [1, 0, 2], [HQ, T, HD], "kt")
    qt = m.transpose(tq, [1, 0, 2], [HQ, T, HD], "qt")
    sc = m.bmm(qt, kt, [HQ, T, T], "scores", adj_y=True)
    return m.finish([0, 1], [sc], "nary14: single 14-input CONCAT replication (probe)")


def build_rank5(q, k):
    """Probe: the same BMM at rank 5 -- expected to be REJECTED (rank <= 4 is a hard limit).

    Kept because a quoted limit without a reproduction is not evidence (E-112 quotes the
    message; this re-derives it on this machine, ticket 04 asked to record the limit).
    """
    m = M()
    tq = m.io("q5", [1, 1, HQ, T, HD])
    tk = m.io("k5", [1, 1, HQ, T, HD])
    sc = m.bmm(tq, tk, [1, 1, HQ, T, T], "scores5", adj_y=True)
    return m.finish([0, 1], [sc], "rank5 probe: 5-D BATCH_MATMUL (expected reject)")


BUILDERS = {"main": build_main, "bmm_only": build_bmm_only, "nary14": build_nary14,
            "rank5": build_rank5,
            "twox7": build_twox7, "unroll14": build_unroll14,
            "wrong_reshape": build_wrong_reshape}


# ----------------------------------------------------------------------------- main
def emit(out_dir, blob, meta, arrays, expected):
    g = os.path.join(out_dir, "golden")
    os.makedirs(g, exist_ok=True)
    with open(os.path.join(out_dir, "model.tflite"), "wb") as f:
        f.write(blob)
    for nm, arr in arrays:
        arr.tofile(os.path.join(g, nm + ".bin"))
    expected.tofile(os.path.join(g, "expected.bin"))
    meta["tflite"] = "model.tflite"
    meta["tflite_bytes"] = len(blob)
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default=MAIN)
    ap.add_argument("--out")
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--tol-analysis", action="store_true",
                    help="print the independent fp16 lower bound and exit (no model)")
    args = ap.parse_args(argv)

    q, k, prov = make_qk(args.weights_dir)
    exp = reference_scores(q, k)

    if args.tol_analysis:
        sim = reference_scores_f16acc(q, k)
        d = np.abs(sim.astype(np.float64) - exp.astype(np.float64))
        mx = float(np.max(np.abs(exp.astype(np.float64))))
        print("q: shape=%s abs_max=%.6g" % (q.shape, float(np.abs(q.astype(np.float32)).max())))
        print("k: shape=%s abs_max=%.6g" % (k.shape, float(np.abs(k.astype(np.float32)).max())))
        print("scores: shape=%s abs_max=%.6g abs_min_nonzero=%.6g" %
              (exp.shape, mx, float(np.abs(exp.astype(np.float64)).min())))
        print("fp16-seq-acc vs fp32-acc golden: max_abs=%.6e max_rel=%.6e" %
              (float(d.max()), float((d / np.maximum(np.abs(exp.astype(np.float64)), 1e-12)).max())))
        print("2^-11*max|out| = %.6e   4*2^-11*max|out| = %.6e" %
              (FP16_ULP_REL * mx, 4 * FP16_ULP_REL * mx))
        return 0

    # k14 is what the graph must produce; used for bmm_only input layout
    k14 = np.repeat(np.transpose(k, (1, 0, 2)), HQ // HKV, axis=0)      # [HQ,T,HD]
    k14_t = np.ascontiguousarray(np.transpose(k14, (1, 0, 2)))          # [T,HQ,HD]

    out_shape = [HQ, T, T]
    if args.variant == "bmm_only":
        blob = build_bmm_only(q, k14_t)
        arrays = [("input", q), ("input1", k14_t)]
    elif args.variant == "twox7":
        blob = build_twox7(q, k)
        arrays = [("input", q), ("input1", k)]
    elif args.variant == "unroll14":
        blob = build_unroll14(q, k)
        arrays = [("input", q), ("input1", k)]
    elif args.variant == "rank5":
        blob = build_rank5(q, k)
        arrays = [("input", np.ascontiguousarray(q.reshape(1, 1, T, HQ, HD))),
                  ("input1", np.ascontiguousarray(
                      np.repeat(np.transpose(k, (1, 0, 2)), HQ // HKV, axis=0)
                      .reshape(1, 1, HQ, T, HD)))]
        out_shape = [1, 1, HQ, T, T]
    elif args.variant == "nary14":
        blob = build_nary14(q, k)
        arrays = [("input", q), ("input1", k)]
    elif args.variant == "wrong_reshape":
        blob = build_wrong_reshape(q, k)
        arrays = [("input", q), ("input1", k)]
    else:
        blob = build_main(q, k)
        arrays = [("input", q), ("input1", k)]

    sim = reference_scores_f16acc(q, k)
    dsim = float(np.max(np.abs(sim.astype(np.float64) - exp.astype(np.float64))))
    max_out = float(np.max(np.abs(exp.astype(np.float64))))
    atol = rtol = round(4.0 * FP16_ULP_REL * max_out, 6)

    meta = {
        "op": "GQA_SCORES",
        "variant": args.variant,
        "shape": {"T": T, "Hq": HQ, "Hkv": HKV, "HD": HD,
                  "kv_repeat": HQ // HKV, "adj_y": 1, "rank": 3},
        "dtype": "fp16",
        "inputs": [{"name": "q", "shape": [T, HQ, HD], "type": "fp16"},
                   {"name": "k" if args.variant != "bmm_only" else "k14",
                    "shape": [T, HQ if args.variant == "bmm_only" else HKV, HD],
                    "type": "fp16"}],
        "output": {"name": "scores", "shape": out_shape, "type": "fp16",
                   "bytes": int(exp.nbytes)},
        "quant": {"q": None, "k": None, "scores": None},
        "reference": "numpy_float32_accum_to_fp16",
        "qlayout": "graph inputs are [T,H,HD] (row-major FC output layout); the graph "
                   "does TRANSPOSE[1,0,2] -> [H,T,HD] before the 3-D BMM (NOTES 10.2)",
        "kv_replication": "SPLIT(axis=1,2) -> CONCAT([k0]*7) / CONCAT([k1]*7) -> "
                          "CONCAT -> [T,14,HD]; head h reads KV head h//7",
        "qk_source": prov,
        "seed": SEED,
        "fp16_lower_bound": {
            "method": "numpy sequential fp16 accumulation over d (K=64) vs float32 golden",
            "max_abs": dsim,
            "max_out_abs": max_out,
            "rel": (dsim / max_out) if max_out else None,
        },
        "tolerance_revision": "ticket-04 rev-1 (registered BEFORE the first board run)",
        "atol": atol,
        "rtol": rtol,
        "tolerance_reason": (
            "max|out| = %.4g >> 1 (q/k are bias-dominated, measured fixture std 7.85/32.2), "
            "so the '<2e-2 absolute' rule of NOTES 4 does not apply; per NOTES 10.1 the "
            "contract is atol=rtol=4*2^-11*max|out| = %.4g, i.e. 4 fp16 ULP at the output "
            "scale. The independent (device-free) lower bound from sequential fp16 "
            "accumulation is %.4g absolute (%.3g relative). Re-registering to make a run "
            "green is forbidden." % (max_out, atol, dsim, dsim / max_out)),
        "scope": "scores only -- no 1/sqrt(HD) scaling and no causal mask (ticket 05)",
    }
    out = args.out or ("ops/gqa_scores_t8_fp16" if args.variant == MAIN
                       else "ops/_gqa_variants/" + args.variant)
    os.makedirs(out, exist_ok=True)
    emit(out, blob, meta, arrays, np.ascontiguousarray(exp.reshape(out_shape)))
    print("%s: %d B  %s  q%s+k%s -> %s  max|out|=%.4g atol=rtol=%g" %
          (out, len(blob), args.variant, q.shape, arrays[1][1].shape, exp.shape,
           max_out, atol))
    return 0


if __name__ == "__main__":
    sys.exit(main())
