#!/usr/bin/env python3
"""Dual-dynamic matmul probes: the two matmuls inside attention, BOTH operands runtime.

    scores   = Q · Kᵀ      Q,K: [1,T,64]            -> [1,T,T]      (adj_x=0 adj_y=1)
    attn_out = P · V       P: [1,T,T], V: [1,T,64]  -> [1,T,64]     (adj 0/0)
    batched  = 2 heads     [1,2,T,64] × [1,2,T,64]  -> [1,2,T,T]

Three op forms are generated so the *compiler* can be asked which one it accepts with a
dynamic B (the question this file exists to answer, see docs/OP-bmm.md):

    form=bmm   BATCH_MATMUL                       A and B both subgraph inputs
    form=fc    FULLY_CONNECTED                    A is input, "weights" is input (S4 path)
    form=conv  CONV_2D 1x1                        A is input, "filter" is input  (expected reject)

Layout note for the FC form of P·V: FC computes out[m,n] = Σ_k a[m,k]·W[n,k], so the
weight tensor must be Vᵀ ([N,K]=[64,T]).  Without a preceding TRANSPOSE the caller has to
store V transposed.  This is a *consequence of the FC signature*, not of the quantisation.
BATCH_MATMUL needs no transpose at all for either matmul.

Quantisation contract (why some variants are bit-exact):
  * codes="small"  int8 codes ∈ {−1,0,1}, out_scale == s_in·s_w  -> real multiplier == 1,
    so out == clip(int64 exact accumulator, ±127) with no rounding  => int8_tol = 0.
    These are the "basis-vector-like" tests demanded by AGENTS.md §3.2: a transpose or an
    offset mistake cannot survive a ±1 code alphabet.
  * codes="full"   int8 codes over the whole range, out_scale = s_in·s_w·32 (multiplier 1/32);
    the device requant is a two-step rounding (OP-MATRIX §3, E-003), so a priori allow
    int8_tol = 1 and check the fraction of off-by-one codes.
  * fp16           tolerance registered BEFORE the run: K-term fp16 accumulation has a
    relative error bound ≈ K·2⁻¹¹  (K=64 → 3.1e-2).  atol=1e-2, rtol=3e-2 are written into
    model.json; tools/compare.py prefers the registered values over its defaults.

Golden data is pure numpy (int64 / float32→float16) and shares no code with the flatbuffer
serialiser (AGENTS.md R3).  Runtime inputs are `golden/input.bin` (A) and
`golden/input1.bin` (B); tools/e2e.sh already pushes both for non-"dynamic_weights" models.

    /usr/bin/python3 tools/build_bmm.py --list
    /usr/bin/python3 tools/build_bmm.py bmm_pv_t16_int8 --out ops/bmm_pv_t16_int8/model.tflite
    /usr/bin/python3 tools/build_bmm.py --form bmm --kind qk --T 16 --dtype int8 \
        --codes small --out ops/bmm_qk_t16_int8/model.tflite
"""
import argparse
import json
import os

import numpy as np

import build_ops as bo

SEED = 20261112
D = 64          # head dim (model width of the attention head)
S8 = 1.0 / 127.0


# ---------------------------------------------------------------- golden data
def make_operands(kind, batch, T, dtype, codes, rng):
    """Logical operands, independent of any serialisation.

    kind == "qk": (Q, K) each [batch, T, D]     -> scores [batch, T, T]
    kind == "pv": (P, V) P [batch, T, T], V [batch, T, D] -> [batch, T, D]
    """
    # batch > 1 means "H heads in one tensor": keep the real attention rank,
    # [1,H,T,D], instead of collapsing the leading 1 (a backend may care).
    lead = (1,) if batch == 1 else (1, batch)
    if kind == "qk":
        shp_a, shp_b = lead + (T, D), lead + (T, D)
    elif kind == "pv":
        shp_a, shp_b = lead + (T, T), lead + (T, D)
    else:
        raise SystemExit(f"unknown kind {kind}")

    if dtype == "int8":
        if codes == "small":
            a = rng.integers(-1, 2, size=shp_a).astype(np.int8)
            b = rng.integers(-1, 2, size=shp_b).astype(np.int8)
        else:
            a = np.clip(np.round(rng.uniform(-1, 1, shp_a) / S8), -127, 127).astype(np.int8)
            b = np.clip(np.round(rng.uniform(-1, 1, shp_b) / S8), -127, 127).astype(np.int8)
    elif dtype == "fp16":
        a = rng.uniform(-1, 1, shp_a).astype(np.float16)
        b = rng.uniform(-1, 1, shp_b).astype(np.float16)
    else:
        raise SystemExit(f"unknown dtype {dtype}")
    return a, b


def reference(kind, a, b, dtype, sa, sb, so):
    """Golden output; numpy only.  int8 -> int64 accumulate (+ optional requant).

    Works on the logical tensor rank: any leading dims (batch/heads) are flattened for
    the contraction and restored afterwards, so a 4-D [1,H,T,D] input is checked exactly
    like the 3-D single-head case.
    """
    lead = a.shape[:-2]
    a2 = a.reshape((-1,) + a.shape[-2:])
    b2 = b.reshape((-1,) + b.shape[-2:])
    if dtype == "int8":
        if kind == "qk":
            acc = np.einsum("btd,bsd->bts", a2.astype(np.int64), b2.astype(np.int64))
        else:
            acc = np.einsum("bts,bsd->btd", a2.astype(np.int64), b2.astype(np.int64))
        ratio = (sa * sb) / so
        # float64 exact product, single rounding; ±1 vs the device's two-step requant
        # rule is inside the registered int8_tol (OP-MATRIX §3 / E-003)
        y = np.round(acc.astype(np.float64) * ratio)
        out = np.clip(y, -128, 127).astype(np.int8)
        return out.reshape(lead + out.shape[-2:])
    if dtype == "fp16":
        a32, b32 = a2.astype(np.float32), b2.astype(np.float32)
        if kind == "qk":
            acc = np.einsum("btd,bsd->bts", a32, b32)
        else:
            acc = np.einsum("bts,bsd->btd", a32, b32)
        out = acc.astype(np.float16)
        return out.reshape(lead + out.shape[-2:])
    raise SystemExit(f"unknown dtype {dtype}")


def quant_contract(dtype, codes):
    """(sa, sb, so, tol_fields, reference_name)."""
    if dtype == "int8":
        sa = sb = S8
        so = sa * sb if codes == "small" else sa * sb * 32.0
        tol = 0 if codes == "small" else 1
        return sa, sb, so, {"int8_tol": tol}, (
            "int64_exact" if codes == "small" else "int64_exact_requant")
    if dtype == "fp16":
        # registered a priori: K·2^-11 relative bound for a K-term fp16 accumulation
        return None, None, None, {"atol": 1e-2, "rtol": 3e-2}, "float32_accum_to_fp16"
    raise SystemExit(f"unknown dtype {dtype}")


def _tout(dtype):
    return bo.I8 if dtype == "int8" else bo.F16


# --------------------------------------------------------------- model forms
def build_bmm(kind, batch, T, dtype, codes, alt_seed=None):
    """BATCH_MATMUL, both operands subgraph inputs."""
    rng = np.random.default_rng(SEED)
    a, b = make_operands(kind, batch, T, dtype, codes, rng)
    sa, sb, so, tolf, refname = quant_contract(dtype, codes)
    exp = reference(kind, a, b, dtype, sa, sb, so)
    alt = None
    if alt_seed is not None:
        # same A, a DIFFERENT B: the compiled .dla must not depend on B's values
        _, b_alt = make_operands(kind, batch, T, dtype, codes,
                                 np.random.default_rng(alt_seed))
        alt = {"b": b_alt, "expected": reference(kind, a, b_alt, dtype, sa, sb, so)}

    adj_y = 1 if kind == "qk" else 0
    # out[.., :] = a[..] · (b[..]ᵀ if adj_y else b[..])
    lead = [1, batch] if batch > 1 else [1]
    if kind == "qk":
        out_shape = lead + [T, T]
    else:
        out_shape = lead + [T, D]
    assert list(exp.shape) == out_shape, (exp.shape, out_shape)

    bb = bo.flatbuffers.Builder(4096)
    buffers = [bo.buf_empty(bb)]
    q = (lambda s: bo.quant_tensor(bb, [s], [0])) if dtype == "int8" else (lambda s: None)
    t_a = bo.tensor(bb, "a", list(a.shape), _tout(dtype), 0, q(sa))
    t_b = bo.tensor(bb, "b", list(b.shape), _tout(dtype), 0, q(sb))
    buffers.append(bo.buf_empty(bb))
    t_o = bo.tensor(bb, "out", out_shape, _tout(dtype), 1, q(so))

    S = bo.S
    S.BatchMatMulOptionsStart(bb)
    S.BatchMatMulOptionsAddAdjX(bb, False)
    S.BatchMatMulOptionsAddAdjY(bb, bool(adj_y))
    S.BatchMatMulOptionsAddAsymmetricQuantizeInputs(bb, False)
    opt = S.BatchMatMulOptionsEnd(bb)

    op = bo.operator(bb, 0, [0, 1], [2], bo.BuiltinOptions.BatchMatMulOptions, opt)
    blob = bo.finish(bb, buffers,
                     [bo.opcode(bb, bo.BuiltinOperator.BATCH_MATMUL)],
                     [t_a, t_b, t_o], [op], [0, 1], [2],
                     f"batch_matmul {kind} b{batch} T{T} {dtype} adj_y={adj_y}")

    meta = {
        "op": "BATCH_MATMUL",
        "form": "bmm",
        "matmul": "Q·Kᵀ" if kind == "qk" else "P·V",
        "shape": {"T": T, "D": D, "batch": batch if batch != 1 else 1,
                  "adj_x": 0, "adj_y": adj_y},
        "dtype": dtype,
        "inputs": [{"name": "a", "shape": list(a.shape), "type": dtype},
                   {"name": "b", "shape": list(b.shape), "type": dtype}],
        "output": {"name": "out", "shape": out_shape, "type": dtype,
                   "bytes": int(exp.nbytes)},
        "quant": {"a": sa, "b": sb, "out": so},
        "codes": codes,
        "reference": refname,
        "score": "repro",
        "seed": SEED,
    }
    meta.update(tolf)
    return blob, meta, a, b, exp, alt


def build_fc_dyn(kind, T, dtype, codes, alt_seed=None, batch=1):
    """FULLY_CONNECTED with runtime "weights" (the S4/E-005 dynamic-weight path).

    qk: A = Q [T,64], W = K [T,64]  (FC wants [N,K] and K already is [T,64])
    pv: A = P [T,T],  W = Vᵀ [64,T]  (layout consequence, see module docstring)

    batch > 1 (qk only) = the "union matrix" batching route: FC has ONE weight tensor
    shared by all rows, so H heads are stacked into a [H·T,64] input and a [H·T,64] weight
    (rows 0..T-1 = K of head 0, T..2T-1 = K of head 1).  The op then computes the whole
    [H·T, H·T] product; the H diagonal [T,T] blocks are the per-head scores and the
    off-diagonal blocks (cross-head products) are discarded -> ~H× wasted MACs but the
    head-per-head semantics are preserved.  Golden here is the full union matrix, i.e. no
    special-casing: if the device computes the union correctly, the diagonal blocks are the
    attention scores.
    """
    rng = np.random.default_rng(SEED)
    a, b = make_operands(kind, batch, T, dtype, codes, rng)
    a2 = a[0].reshape(batch * T, -1)
    b2 = b[0].reshape(batch * T, -1)
    sa, sb, so, tolf, refname = quant_contract(dtype, codes)

    if kind == "qk":
        in_shape, w_shape = list(a2.shape), list(b2.shape)
        w = b2                                   # W = K
    else:
        in_shape, w_shape = [T, T], [D, T]
        w = np.ascontiguousarray(b2.T)           # W = Vᵀ
    out_shape = [batch * T, batch * T] if kind == "qk" else [T, D]

    # golden through the *logical* matmul, not through the stored W layout
    exp = reference(kind, a2[None], b2[None], dtype, sa, sb, so)[0]
    assert list(exp.shape) == out_shape, (exp.shape, out_shape)

    alt = None
    if alt_seed is not None:
        _, b_alt = make_operands(kind, batch, T, dtype, codes,
                                 np.random.default_rng(alt_seed))
        b_alt2 = b_alt[0].reshape(batch * T, -1)
        exp_alt = reference(kind, a2[None], b_alt2[None], dtype, sa, sb, so)[0]
        w_alt = b_alt2 if kind == "qk" else np.ascontiguousarray(b_alt2.T)
        alt = {"b": w_alt, "expected": exp_alt}

    bb = bo.flatbuffers.Builder(4096)
    buffers = [bo.buf_empty(bb)]                        # 0: all runtime tensors
    quant = (lambda s: bo.quant_tensor(bb, [s], [0])) if dtype == "int8" else (lambda s: None)
    t_in = bo.tensor(bb, "input", in_shape, _tout(dtype), 0, quant(sa))
    t_w = bo.tensor(bb, "weights", w_shape, _tout(dtype), 0, quant(sb))
    # bias is a constant zero: isolates the dynamic-weight path from bias folding
    if dtype == "int8":
        bias = np.zeros(out_shape[1], np.int32)
        buffers.append(bo.buf_data(bb, bias))
        t_b = bo.tensor(bb, "bias", [out_shape[1]], bo.I32, 1, quant(sa * sb))
    else:
        bias = np.zeros(out_shape[1], np.float16)
        buffers.append(bo.buf_data(bb, bias))
        t_b = bo.tensor(bb, "bias", [out_shape[1]], bo.F16, 1)
    buffers.append(bo.buf_empty(bb))
    t_o = bo.tensor(bb, "output", out_shape, _tout(dtype), 2, quant(so))

    S = bo.S
    S.FullyConnectedOptionsStart(bb)
    S.FullyConnectedOptionsAddFusedActivationFunction(bb, 0)
    S.FullyConnectedOptionsAddWeightsFormat(bb, 0)     # DEFAULT [N,K]
    S.FullyConnectedOptionsAddKeepNumDims(bb, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(bb, False)
    opt = S.FullyConnectedOptionsEnd(bb)

    op = bo.operator(bb, 0, [0, 1, 2], [3], bo.BuiltinOptions.FullyConnectedOptions, opt)
    blob = bo.finish(bb, buffers,
                     [bo.opcode(bb, bo.BuiltinOperator.FULLY_CONNECTED)],
                     [t_in, t_w, t_b, t_o], [op], [0, 1], [3],
                     f"fc dynB {kind} T{T} {dtype}")

    meta = {
        "op": "FULLY_CONNECTED",
        "form": "fc",
        "matmul": "Q·Kᵀ" if kind == "qk" else "P·V",
        "shape": {"T": T, "D": D, "heads": batch, "M": in_shape[0], "K": in_shape[1],
                  "N": out_shape[1]},
        "dtype": dtype,
        "dynamic_weights": False,     # deliberately NOT the e2e "weights.bin" branch:
                                      # we want A=input.bin, B=input1.bin ordering
        "inputs": [{"name": "input", "shape": in_shape, "type": dtype},
                   {"name": "weights", "shape": w_shape, "type": dtype}],
        "output": {"name": "output", "shape": out_shape, "type": dtype,
                   "bytes": int(exp.nbytes)},
        "quant": {"input": sa, "weights": sb, "bias": None if sa is None else sa * sb,
                  "output": so},
        "codes": codes,
        "reference": refname,
        "weights_layout": "K rows = [N,K]" if kind == "qk" else "V transposed: [N,K]=[64,T]",
        "seed": SEED,
    }
    meta.update(tolf)
    return blob, meta, a2, w, exp, alt


def build_conv_dyn(kind, T, dtype, codes, alt_seed=None):
    """CONV_2D 1x1 with the FILTER as a runtime input.

    TFLite's CONV_2D signature has `filter` as input 1 but every backend treats it as the
    constant "weight" side, so H5 predicted a rejection.  It is generated anyway because a
    prediction that is never tested is not evidence (AGENTS R1/R4) -- and because, if ncc
    does accept it, the numbers have to prove the filter is read at runtime.

    qk: input [1,T,1,64] (NHWC, W=1) · filter [T,1,1,64]   -> out [1,T,1,T]
    pv: input [1,T,1,T]             · filter [64,1,1,T]    -> out [1,T,1,64]
    """
    rng = np.random.default_rng(SEED)
    a, b = make_operands(kind, 1, T, dtype, codes, rng)
    a2, b2 = a[0], b[0]
    sa, sb, so, tolf, refname = quant_contract(dtype, codes)
    cin = D if kind == "qk" else T
    oc = T if kind == "qk" else D
    exp = reference(kind, a2[None], b2[None], dtype, sa, sb, so)[0].reshape(1, T, 1, oc)

    # The RUNTIME buffer for `filter` is [OC,1,1,IC] (OHWI, IC innermost).  For P·V that
    # means the caller has to hand over Vᵀ — exactly the same "B must be transposed"
    # convention as the FC form, and the same trap: feeding V directly still *runs*
    # (1024 int8 either way) but computes A·(bytes read as [OC,IC]) = wrong numbers.
    # Measured: with V pushed verbatim the device output equals A·Fᵀ of the pushed bytes
    # bit-for-bit; with Vᵀ pushed it matches the golden.  See docs/OP-bmm.md §4.
    filt = b2 if kind == "qk" else np.ascontiguousarray(b2.T)

    alt = None
    if alt_seed is not None:
        _, b_alt = make_operands(kind, 1, T, dtype, codes,
                                 np.random.default_rng(alt_seed))
        b_alt2 = b_alt[0]
        exp_alt = reference(kind, a2[None], b_alt2[None],
                            dtype, sa, sb, so)[0].reshape(1, T, 1, oc)
        alt = {"b": b_alt2 if kind == "qk" else np.ascontiguousarray(b_alt2.T),
               "expected": exp_alt}

    bb = bo.flatbuffers.Builder(4096)
    buffers = [bo.buf_empty(bb)]
    quant = (lambda s: bo.quant_tensor(bb, [s], [0])) if dtype == "int8" else (lambda s: None)
    t_in = bo.tensor(bb, "input", [1, T, 1, cin], _tout(dtype), 0, quant(sa))
    t_f = bo.tensor(bb, "filter", [oc, 1, 1, cin], _tout(dtype), 0, quant(sb))
    if dtype == "int8":
        buffers.append(bo.buf_data(bb, np.zeros(oc, np.int32)))
        t_b = bo.tensor(bb, "bias", [oc], bo.I32, 1, quant(sa * sb))
    else:
        buffers.append(bo.buf_data(bb, np.zeros(oc, np.float16)))
        t_b = bo.tensor(bb, "bias", [oc], bo.F16, 1)
    buffers.append(bo.buf_empty(bb))
    t_o = bo.tensor(bb, "output", [1, T, 1, oc], _tout(dtype), 2, quant(so))

    S = bo.S
    S.Conv2DOptionsStart(bb)
    S.Conv2DOptionsAddPadding(bb, 1)          # SAME
    S.Conv2DOptionsAddStrideW(bb, 1)
    S.Conv2DOptionsAddStrideH(bb, 1)
    S.Conv2DOptionsAddFusedActivationFunction(bb, 0)
    S.Conv2DOptionsAddDilationWFactor(bb, 1)
    S.Conv2DOptionsAddDilationHFactor(bb, 1)
    opt = S.Conv2DOptionsEnd(bb)

    op = bo.operator(bb, 0, [0, 1, 2], [3], bo.BuiltinOptions.Conv2DOptions, opt)
    blob = bo.finish(bb, buffers,
                     [bo.opcode(bb, bo.BuiltinOperator.CONV_2D)],
                     [t_in, t_f, t_b, t_o], [op], [0, 1], [3],
                     f"conv1x1 dyn filter {kind} T{T} {dtype}")

    meta = {
        "op": "CONV_2D",
        "form": "conv",
        "matmul": "Q·Kᵀ" if kind == "qk" else "P·V",
        "shape": {"T": T, "D": D, "H": T, "W": 1, "C": cin, "OC": oc,
                  "FH": 1, "FW": 1, "pad": "SAME", "stride": 1},
        "dtype": dtype,
        "inputs": [{"name": "input", "shape": [1, T, 1, cin], "type": dtype},
                   {"name": "filter", "shape": [oc, 1, 1, cin], "type": dtype}],
        "output": {"name": "output", "shape": [1, T, 1, oc], "type": dtype,
                   "bytes": int(exp.nbytes)},
        "quant": {"input": sa, "weights": sb, "bias": None if sa is None else sa * sb,
                  "output": so},
        "codes": codes,
        "reference": refname,
        "filter_layout": ("[OC,1,1,IC] = K rows" if kind == "qk"
                          else "[OC,1,1,IC] = V transposed: rows are V columns"),
        "seed": SEED,
    }
    meta.update(tolf)
    return blob, meta, a2, filt, exp, alt


# ---------------------------------------------------------------------- specs
def _spec(form, kind, T, dtype, codes, batch=1):
    return dict(form=form, kind=kind, T=T, dtype=dtype, codes=codes, batch=batch)


SPECS = {
    # --- 1. BATCH_MATMUL: Q·Kᵀ (adj_y=1), int8 + fp16 -----------------------
    "bmm_qk_t8_int8":        _spec("bmm", "qk", 8, "int8", "small"),
    "bmm_qk_t16_int8":       _spec("bmm", "qk", 16, "int8", "small"),
    "bmm_qk_t16_int8_full":  _spec("bmm", "qk", 16, "int8", "full"),
    "bmm_qk_t32_int8":       _spec("bmm", "qk", 32, "int8", "small"),
    "bmm_qk_t16_fp16":       _spec("bmm", "qk", 16, "fp16", "-"),
    # --- 2. BATCH_MATMUL: P·V (adj 0/0) — the 16-byte-row boundary lives here
    "bmm_pv_t8_int8":        _spec("bmm", "pv", 8, "int8", "small"),
    "bmm_pv_t16_int8":       _spec("bmm", "pv", 16, "int8", "small"),
    "bmm_pv_t32_int8":       _spec("bmm", "pv", 32, "int8", "small"),
    "bmm_pv_t16_int8_full":  _spec("bmm", "pv", 16, "int8", "full"),
    "bmm_pv_t16_fp16":       _spec("bmm", "pv", 16, "fp16", "-"),
    "bmm_pv_t8_fp16":        _spec("bmm", "pv", 8, "fp16", "-"),   # 8 fp16 = 16 B row
    # --- 3. BATCH_MATMUL: 2 heads batched ---------------------------------
    "bmm_qk_b2h_t8_int8":    _spec("bmm", "qk", 8, "int8", "small", batch=2),
    "bmm_qk_b2h_t16_int8":   _spec("bmm", "qk", 16, "int8", "small", batch=2),
    "bmm_qk_b2h_t16_fp16":   _spec("bmm", "qk", 16, "fp16", "-", batch=2),
    "bmm_pv_b2h_t16_int8":   _spec("bmm", "pv", 16, "int8", "small", batch=2),
    # --- 4. FULLY_CONNECTED with runtime weights --------------------------
    "fc_bmm_qk_t8_int8":       _spec("fc", "qk", 8, "int8", "small"),
    "fc_bmm_qk_t16_int8":      _spec("fc", "qk", 16, "int8", "small"),
    "fc_bmm_qk_t16_int8_full": _spec("fc", "qk", 16, "int8", "full"),
    "fc_bmm_qk_t32_int8":      _spec("fc", "qk", 32, "int8", "small"),
    "fc_bmm_pv_t16_int8":      _spec("fc", "pv", 16, "int8", "small"),
    "fc_bmm_qk_b2h_t16_int8":  _spec("fc", "qk", 16, "int8", "small", batch=2),  # union
    "fc_bmm_pv_t8_int8":       _spec("fc", "pv", 8, "int8", "small"),   # boundary probe
    "fc_bmm_qk_t16_fp16":      _spec("fc", "qk", 16, "fp16", "-"),
    "fc_bmm_pv_t16_fp16":      _spec("fc", "pv", 16, "fp16", "-"),
    # --- 5. CONV_2D 1x1 with runtime filter -------------------------------
    "conv_bmm_qk_t16_int8":  _spec("conv", "qk", 16, "int8", "small"),
    "conv_bmm_qk_t16_fp16":  _spec("conv", "qk", 16, "fp16", "-"),
    "conv_bmm_pv_t16_int8":  _spec("conv", "pv", 16, "int8", "small"),
    "conv_bmm_pv_t8_int8":   _spec("conv", "pv", 8, "int8", "small"),   # boundary probe
    "conv_bmm_pv_t16_fp16":  _spec("conv", "pv", 16, "fp16", "-"),
}


def build(cfg, alt_seed=None):
    if cfg["form"] == "bmm":
        return build_bmm(cfg["kind"], cfg["batch"], cfg["T"], cfg["dtype"], cfg["codes"],
                         alt_seed)
    if cfg["form"] == "fc":
        return build_fc_dyn(cfg["kind"], cfg["T"], cfg["dtype"], cfg["codes"], alt_seed,
                            cfg["batch"])
    if cfg["form"] == "conv":
        return build_conv_dyn(cfg["kind"], cfg["T"], cfg["dtype"], cfg["codes"], alt_seed)
    raise SystemExit(f"unknown form {cfg['form']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", help="ops/<name>/model.tflite")
    ap.add_argument("--form", choices=["bmm", "fc", "conv"])
    ap.add_argument("--kind", choices=["qk", "pv"])
    ap.add_argument("--T", type=int)
    ap.add_argument("--dtype", choices=["int8", "fp16"])
    ap.add_argument("--codes", choices=["small", "full"], default="small")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--alt-b", action="store_true",
                    help="also emit golden/{input1_alt.bin,expected_alt.bin}: same A, "
                         "different B -> proves the compiled .dla is weight-agnostic (L3)")
    args = ap.parse_args()

    if args.list:
        for name in SPECS:
            print(name)
        return

    if args.spec in SPECS:
        cfg = dict(SPECS[args.spec])
    elif args.form and args.kind and args.T and args.dtype:
        cfg = _spec(args.form, args.kind, args.T, args.dtype,
                    args.codes if args.dtype == "int8" else "-", batch=args.batch)
    else:
        raise SystemExit("unknown spec; use --list or --form/--kind/--T/--dtype")

    alt_seed = SEED + 1000 if args.alt_b else None
    blob, meta, a, b, exp, alt = build(cfg, alt_seed)
    out = args.out or (f"ops/{args.spec}/model.tflite" if args.spec else None)
    if not out:
        raise SystemExit("--out is required for ad-hoc specs")
    d = os.path.dirname(os.path.abspath(out))
    g = os.path.join(d, "golden")
    os.makedirs(g, exist_ok=True)
    with open(out, "wb") as f:
        f.write(blob)
    a.tofile(os.path.join(g, "input.bin"))
    b.tofile(os.path.join(g, "input1.bin"))
    exp.tofile(os.path.join(g, "expected.bin"))
    if alt is not None:
        alt["b"].tofile(os.path.join(g, "input1_alt.bin"))
        alt["expected"].tofile(os.path.join(g, "expected_alt.bin"))
        meta["alt_b"] = {
            "seed": alt_seed,
            "files": ["golden/input1_alt.bin", "golden/expected_alt.bin"],
            "purpose": "same A + different B: one compiled .dla must serve any weights "
                       "(L3 re-bind check, replicates S4/E-005 on the dual-dynamic path)",
        }

    meta["spec"] = args.spec or "adhoc"
    meta["tflite"] = os.path.basename(out)
    meta["tflite_bytes"] = len(blob)
    with open(os.path.join(d, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)

    tol = meta.get("int8_tol", f"atol={meta.get('atol')},rtol={meta.get('rtol')}")
    print(f"{out}  {len(blob)} B  A{a.shape}×B{b.shape} -> {exp.shape}  "
          f"{meta['op']} {cfg['dtype']} codes={cfg['codes']}  tol={tol}")


if __name__ == "__main__":
    main()
