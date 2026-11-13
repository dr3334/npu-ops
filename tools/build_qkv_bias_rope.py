#!/usr/bin/env python3
"""Q / K / V projection (+bias) followed by RoPE, at the real Qwen2.5-0.5B shapes.

    /usr/bin/python3 tools/build_qkv_bias_rope.py --which q          # -> ops/qkv_q_bias_rope_t8_fp16
    /usr/bin/python3 tools/build_qkv_bias_rope.py --which k --out ops/<name>
    /usr/bin/python3 tools/build_qkv_bias_rope.py --which v --diag --out ops/<name>   # device diagnostics
    /usr/bin/python3 tools/build_qkv_bias_rope.py --which k --emit-neg-width896 --out ops/qkv_k_bias_rope_t8_fp16
                                                                  # -> <out>/neg_width896 (a second .tflite, no path argument)

Why three graphs (one per projection) and not one
-------------------------------------------------
tools/e2e.sh + tools/compare.py are single-input/single-output.  q is [8,14,64] and k/v are
[8,2,64], so one graph with three outputs cannot be compared by this harness.  Three graphs,
one builder: the only difference between them is (heads, out_width, weight tensor).

Graph (per projection), T=8, hidden D=896, head_dim 64, fp16:

    in  [8,896]                      <- post-input_layernorm hidden (see "input choice")
    FC  W[N,896] + bias[N]           -> [8,N]        N=896 (q) / 128 (k,v)
    RESHAPE                          -> [8,H,64]     H=14 (q) / 2 (k,v)   <- token-major
    SPLIT(axis=2)                    -> [8,H,32] x2  (a = x[...,:32], b = x[...,32:])
    MUL x4 with const cos/sin tables -> [8,H,32] x4
    SUB / ADD                        -> lo = a*cos - b*sin ; hi = b*cos + a*sin
    CONCAT(axis=2)                   -> [8,H,64]     output (rank 3)

Four things this probe is built to *pin down* (the ticket's checklist):

1. **the bias must be in the graph.**  Qwen2.5's q/k/v biases have no entry in config.json
   (they are hard-coded in the modelling code); dropping them shifts the output by ~7.8 (q)
   and ~29.8 (k) sigma while *every remaining number still looks plausible*.  The bias is the
   3rd FULLY_CONNECTED input, and the golden without it is one of the registered
   counterfactuals in --diag (it differs by 25.2 / 77.8 / 0.043, i.e. >> tolerance).
2. **k/v are 128 wide, not 896** (2 KV heads x 64).  --emit-neg-width896 builds the two
   "mis-wired to 896" graphs so the *failure mode* is a recorded artifact, not a guess.
3. **RoPE with pre-computed sin/cos tables.**  The ncc importer rejects SIN/COS, so the tables
   are data (const here: prefill of a fixed T=8 window; docs/OP-qkv.md keeps the dyn variant
   as the decode-time alternative).  T must be static for that reason.
4. **output rank 3.**  The device rejects rank > 4 and rank 4 costs ~4x the layers for the
   same work (docs/OP-rope.md §5.3), so [T,H,HD] stays 3-D.

Conventions that are easy to get *consistently* wrong, so they are stated here (AGENTS.md R4):
* RoPE is Qwen2 **split-half**: out_lo = a*cos - b*sin, out_hi = b*cos + a*sin.
  (The "b*sin + a*cos" form is NOT a rotation -- docs/OP-rope.md §2; build() asserts the
  pair norm is preserved, so a cos/sin swap cannot pass silently.)
* heads are split **token-major**: [T, 896] -> [T, 14, 64].  The wrong [14, T, 64] reading is
  a registered counterfactual (diff 35.4) -- NOTES §10.2 is exactly this failure class.
* positions are p0 = 0..7 (prefill of the first 8 positions).  Table row 0 is cos=1/sin=0,
  so a "rotation does nothing" bug is caught by rows 1..7 only; the tolerance is the same
  for all rows, and the no-rotation counterfactual is measured explicitly.

Input choice (recorded because the ticket asks which one was taken)
-------------------------------------------------------------------
The graph input is the **post-input_layernorm hidden** [8,896]: n1 = RMSNorm(embed[ids]) with
gamma = layer0.input_layernorm.weight and eps = 1e-6, computed once here in float64 from the
*stored fp16* fixture values and stored as golden/input.bin (fp16).  The RMSNorm itself is NOT
in the graph: it is its own operator with its own board evidence (docs/OP-rmsnorm.md), and
keeping it out keeps this probe's failure attribution single-cause.  eps = 1e-6 (transformers
rms_norm_eps); gamma is applied **inside the pre-graph norm**, i.e. it is NOT folded into the
FC weights -- so nobody has to remember to fold it in the numpy golden.
Token ids are fixed (TOKEN_IDS) so input.bin is reproducible byte for byte.

Golden (AGENTS.md R3)
---------------------
golden/expected.bin is numpy only: float64 from the *stored fp16* input/weights/bias/tables,
with exactly one rounding (float64 -> float16) at the end.  It never calls ncc, the TFLite
interpreter or the device, and it does not share code with the flatbuffer construction.

Tolerance (registered *before* the board run -- see model.json "tolerance_revision")
-----------------------------------------------------------------------------------
The device stores the FC output (and every MUL/ADD/SUB result) in fp16, so even a perfect
graph differs from the single-rounding golden by ~1 ULP of |out|.  |out| is 25.4 (q) and 77.6
(k) -- NOTES §4's flat 2e-2 would be 7.9e-4 / 2.6e-4 *relative*, i.e. tighter than one fp16
ULP, so the §10.1 exception applies: the floor is measured independently of the device
(numpy per-op fp16 rounding simulation, "fp16_floor" in model.json) and the registered
tolerance is k * 2^-11 * max|out| with k = 4 (atol) and k * 2^-11 (rtol).
v is a special case: |out|max = 0.128 < 1, so §4 applies unchanged (atol/rtol from the same
measured floor, well under 2e-2).

Usage / artifacts
-----------------
    model.tflite          reproducible from this script
    model.json            IO contract + atol/rtol + floor + conventions + provenance
    golden/input.bin      n1 [8,896] fp16
    golden/expected.bin   [8,H,64] fp16
    --diag                (after a board run) compares out_device.bin against the golden, the
                          fp16 simulation and the registered counterfactuals
"""
import argparse
import json
import os
import sys

import flatbuffers
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ops as bo          # flatbuffer toolbox (read-only use, AGENTS §8)

import qwen25_weights as qw     # frozen fixture loader (verify-on-load, sha256-checked)

T = 8                 # static token count (sin/cos are tables -> T must be static)
D = 896               # hidden width
HD = 64               # head_dim (Qwen2.5-0.5B)
HALF = HD // 2        # 32
THETA = 1e6           # Qwen2.5 rope_theta
POS0 = 0              # prefill: positions 0..T-1
EPS = 1e-6            # transformers rms_norm_eps
LAYER = 0
SEED = 20261113       # only for the reproducibility note; all values come from the fixture

# 8 token ids, fixed so golden/input.bin is byte-reproducible; arbitrary but documented.
TOKEN_IDS = [9707, 11, 1879, 828, 11780, 13, 220, 16]

F16, I32 = bo.F16, bo.I32
S = bo.S
BO, BOPT = bo.S.BuiltinOperator, bo.S.BuiltinOptions

SPECS = {
    # which: (kv-or-q heads, projected width N)
    "q": dict(heads=14, n=896),
    "k": dict(heads=2,  n=128),
    "v": dict(heads=2,  n=128),
}
DEFAULT_OUT = {w: f"ops/qkv_{w}_bias_rope_t8_fp16" for w in SPECS}


# ------------------------------------------------------------------ fixture data
def fixture():
    """Everything this probe needs from the frozen fixture, on CPU, reproducible."""
    gamma = qw.get(f"layer{LAYER}.input_layernorm.weight").astype(np.float64)
    emb = qw.get("embed.weight")
    x = emb[np.asarray(TOKEN_IDS)].astype(np.float64)         # [T, D] fp16-stored rows
    ms = (x * x).mean(axis=-1, keepdims=True)
    n1 = (x / np.sqrt(ms + EPS) * gamma).astype(np.float16)    # post-layernorm input
    out = {"input": n1}
    for w in SPECS:
        out[f"W{w}"] = qw.get(f"layer{LAYER}.attn.{w}_proj.weight")      # [N, D] fp16
        out[f"b{w}"] = qw.get(f"layer{LAYER}.attn.{w}_proj.bias")        # [N]    fp16
    return out


def tables(heads, p0=POS0):
    """cos/sin for positions p0..p0+T-1, half head (HD/2), already duplicated over heads.

    Duplicated instead of broadcast: a rank-3 [T,H,h] activation times a [T,1,h] table is a
    broadcast that this repo has no board evidence for (only [T,64]x[T,1] at D=64, docs/
    OP-rmsnorm.md).  The duplication costs 2*T*H*h*2 B of Static memory (7 KB for q).
    """
    inv = THETA ** (-2.0 * np.arange(HALF) / HD)
    ang = np.outer(p0 + np.arange(T), inv)                    # [T, HALF]
    c = np.broadcast_to(np.cos(ang)[:, None, :], (T, heads, HALF))
    s = np.broadcast_to(np.sin(ang)[:, None, :], (T, heads, HALF))
    return (np.ascontiguousarray(c).astype(np.float16),
            np.ascontiguousarray(s).astype(np.float16))


def tables_bs(c16, s16):
    """The stored fp16 tables in float64, broadcast-ready: [T,1,HALF]."""
    return c16.astype(np.float64)[:, :1, :], s16.astype(np.float64)[:, :1, :]


# ------------------------------------------------------------------------ maths
def fc_exact(n1, W, b):
    """float64 matmul on the *stored* fp16 values: the golden's FC."""
    return n1.astype(np.float64) @ W.astype(np.float64).T + b.astype(np.float64)


def rope_from_fc(y, c, s, store=True):
    """Split-half RoPE on an [T,H,HD]-shaped float array.

    out[...,:h] = a*cos - b*sin ;  out[...,h:] = b*cos + a*sin     (Qwen2 / HF rotate_half)
    c/s are [T,1,h] float64 (= the stored fp16 tables), broadcast over heads.
    """
    y3 = np.asarray(y, np.float64).reshape(T, -1, HD)
    a, bb = y3[..., :HALF], y3[..., HALF:]
    o = np.concatenate([a * c - bb * s, bb * c + a * s], axis=-1)
    o = o.reshape(T, -1)
    return o.astype(np.float16) if store else o


def golden(n1, W, b, c16, s16):
    """Registered reference: exact float64 from stored fp16 inputs, one final rounding."""
    c, s = tables_bs(c16, s16)
    return rope_from_fc(fc_exact(n1, W, b), c, s, store=True)


def fp16_sim(n1, W, b, c16, s16):
    """Device-semantics simulation: float32 FC, fp16 storage of the FC output, then every
    elementwise op rounds to fp16 (what MDLA does -- docs/OP-rope.md §6 measured it as
    bit-identical to the device for the rope chain).

    This is a *floor measurement*, independent of the device, used to justify the
    tolerance (NOTES §10.1).  It is never written to golden/expected.bin (R3).
    """
    y32 = n1.astype(np.float32) @ W.T + b                       # fp32 accumulate
    y = np.asarray(y32, np.float16).reshape(T, -1, HD)          # fp16 storage at FC output
    a, bb = y[..., :HALF], y[..., HALF:]
    C = np.repeat(c16[:, :1, :].astype(np.float32), y.shape[1], axis=1)
    Sn = np.repeat(s16[:, :1, :].astype(np.float32), y.shape[1], axis=1)

    def r(m):
        return np.asarray(m, np.float16)

    p1, p2 = r(a * C), r(bb * Sn)
    p3, p4 = r(bb * C), r(a * Sn)
    lo = r(p1.astype(np.float32) - p2.astype(np.float32))
    hi = r(p3.astype(np.float32) + p4.astype(np.float32))
    return np.concatenate([lo, hi], axis=-1).reshape(T, -1)


def check_rotation_invariant(y_in, y_out, label):
    """The golden must be a rotation: every (a,b) pair keeps its 2-norm (docs/OP-rope.md §2.3).
    fp16 storage of HD outputs costs <= ~1e-3 relative, so allow 3e-3."""
    yi = np.asarray(y_in, np.float64).reshape(T, -1, HD)
    yo = np.asarray(y_out, np.float64).reshape(T, -1, HD)
    ni = np.hypot(yi[..., :HALF], yi[..., HALF:])
    no = np.hypot(yo[..., :HALF], yo[..., HALF:])
    worst = float(np.abs(no - ni).max() / max(float(np.abs(ni).max()), 1e-12))
    if worst > 3e-3:
        raise SystemExit(f"{label}: pair norm not preserved (rel {worst:.2e}) -> formula is "
                         f"not a rotation (cos/sin swapped?)")
    return worst


# ------------------------------------------------------------ flatbuffer builder
class M:
    """Same accumulator shape as tools/build_rope.py (buffer 0 = activation placeholder)."""

    def __init__(self):
        self.b = flatbuffers.Builder(1 << 16)
        self.buffers = [bo.buf_empty(self.b)]
        self.tensors, self.opcodes, self.ops = [], [], []

    def const(self, name, shape, ttype, arr):
        self.buffers.append(bo.buf_data(self.b, np.ascontiguousarray(arr)))
        self.tensors.append(bo.tensor(self.b, name, list(shape), ttype,
                                      len(self.buffers) - 1))
        return len(self.tensors) - 1

    def act(self, name, shape, ttype):
        self.tensors.append(bo.tensor(self.b, name, list(shape), ttype, 0))
        return len(self.tensors) - 1

    def add_op(self, builtin, ins, outs, opt_type=None, opt=None):
        self.opcodes.append(bo.opcode(self.b, builtin))
        self.ops.append(bo.operator(self.b, len(self.opcodes) - 1, ins, outs,
                                    opt_type, opt))
        return len(self.ops) - 1

    def finalize(self, ins, outs, desc):
        return bo.finish(self.b, self.buffers, self.opcodes, self.tensors, self.ops,
                         ins, outs, desc)


def build_graph(n1, W, b, c16, s16, heads, n, neg=None):
    """Return the .tflite bytes for 'input -> FC(+bias) -> [T,H,HD] -> RoPE -> output'."""
    m = M()
    t_in = m.act("input", [T, D], F16)

    w_shape = [n, D]
    if neg == "declared_size":
        # the E-112 failure: the flatbuffer *declares* [896,896] but only [128,896] exist
        w_shape = [D, D]
    t_w = m.const("W", w_shape, F16, W)
    t_b = m.const("bias", [n], F16, b)
    t_fc = m.act("fc_out", [T, n], F16)
    S.FullyConnectedOptionsStart(m.b)
    S.FullyConnectedOptionsAddFusedActivationFunction(m.b, 0)     # NONE
    S.FullyConnectedOptionsAddWeightsFormat(m.b, 0)               # DEFAULT [N,K]
    S.FullyConnectedOptionsAddKeepNumDims(m.b, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(m.b, False)
    fc_opt = S.FullyConnectedOptionsEnd(m.b)
    m.add_op(BO.FULLY_CONNECTED, [t_in, t_w, t_b], [t_fc],
             BOPT.FullyConnectedOptions, fc_opt)

    out_shape = [T, heads, HD]
    t_shape = m.const("new_shape", [3], I32, np.asarray(out_shape, np.int32))
    nsv = m.b.CreateNumpyVector(np.asarray(out_shape, np.int32))
    S.ReshapeOptionsStart(m.b)
    S.ReshapeOptionsAddNewShape(m.b, nsv)
    rsh_opt = S.ReshapeOptionsEnd(m.b)
    t_r = m.act("x3", out_shape, F16)
    m.add_op(BO.RESHAPE, [t_fc, t_shape], [t_r], BOPT.ReshapeOptions, rsh_opt)

    hs = [T, heads, HALF]
    t_cos = m.const("cos", hs, F16, c16)
    t_sin = m.const("sin", hs, F16, s16)

    t_axis = m.const("axis", [1], I32, np.asarray([2], np.int32))
    t_a = m.act("a", hs, F16)
    t_b = m.act("b", hs, F16)
    S.SplitOptionsStart(m.b)
    S.SplitOptionsAddNumSplits(m.b, 2)
    split_opt = S.SplitOptionsEnd(m.b)
    m.add_op(BO.SPLIT, [t_axis, t_r], [t_a, t_b], BOPT.SplitOptions, split_opt)

    def mul(x, y, name):
        t_o = m.act(name, hs, F16)
        S.MulOptionsStart(m.b)
        S.MulOptionsAddFusedActivationFunction(m.b, 0)
        opt = S.MulOptionsEnd(m.b)
        m.add_op(BO.MUL, [x, y], [t_o], BOPT.MulOptions, opt)
        return t_o

    def binop(builtin, start, add, end, opt_type, x, y, name):
        t_o = m.act(name, hs, F16)
        start(m.b)
        add(m.b, 0)
        opt = end(m.b)
        m.add_op(builtin, [x, y], [t_o], opt_type, opt)
        return t_o

    p_a_cos = mul(t_a, t_cos, "a_cos")
    p_b_sin = mul(t_b, t_sin, "b_sin")
    p_b_cos = mul(t_b, t_cos, "b_cos")
    p_a_sin = mul(t_a, t_sin, "a_sin")
    t_lo = binop(BO.SUB, S.SubOptionsStart, S.SubOptionsAddFusedActivationFunction,
                 S.SubOptionsEnd, BOPT.SubOptions, p_a_cos, p_b_sin, "rot_lo")
    t_hi = binop(BO.ADD, S.AddOptionsStart, S.AddOptionsAddFusedActivationFunction,
                 S.AddOptionsEnd, BOPT.AddOptions, p_b_cos, p_a_sin, "rot_hi")
    t_out = m.act("output", out_shape, F16)
    S.ConcatenationOptionsStart(m.b)
    S.ConcatenationOptionsAddAxis(m.b, 2)
    S.ConcatenationOptionsAddFusedActivationFunction(m.b, 0)
    cat_opt = S.ConcatenationOptionsEnd(m.b)
    m.add_op(BO.CONCATENATION, [t_lo, t_hi], [t_out], BOPT.ConcatenationOptions, cat_opt)

    return m.finalize([t_in], [t_out],
                      f"qkv {n}x{D} bias + rope split-half T={T} H={heads} HD={HD} fp16")


# ---------------------------------------------------------------- counterfactuals
def counterfactuals(n1, W, b, c16, s16, which):
    """The registered "what if this were wrong" set.  Each one is a plausible bug; the
    device only matches the golden if none of them is what the graph actually computes."""
    c, s = tables_bs(c16, s16)
    heads = SPECS[which]["heads"]
    out = {}
    # (1) bias dropped (the headline bug: config.json has no bias key)
    out["bias_dropped"] = rope_from_fc(fc_exact(n1, W, np.zeros_like(b)), c, s).astype(np.float64)
    # (2) no rotation at all (tables ignored / sin=0 -> out == FC output)
    out["no_rotation"] = fc_exact(n1, W, b).astype(np.float16).astype(np.float64)
    # (3) head-major reshape [H,T,HD] instead of token-major [T,H,HD] (NOTES §10.2)
    y3 = fc_exact(n1, W, b).astype(np.float16).astype(np.float64).reshape(heads, T, HD)
    a, bb = y3[..., :HALF], y3[..., HALF:]
    C = c[:, :1, :].transpose(1, 0, 2)      # [HALF, T, 1] -> broadcast over [H, T, HALF]
    Sx = s[:, :1, :].transpose(1, 0, 2)
    out["head_major_reshape"] = np.concatenate([a * C - bb * Sx, bb * C + a * Sx],
                                               axis=-1).reshape(T, heads * HD)
    # (4) wrong projection tensor: k/v built from q_proj (the 896-wide mis-wiring)
    if which != "q":
        Wq = qw.get(f"layer{LAYER}.attn.q_proj.weight").astype(np.float64)
        bq = qw.get(f"layer{LAYER}.attn.q_proj.bias").astype(np.float64)
        yq = (n1.astype(np.float64) @ Wq.T + bq)[:, :SPECS[which]["n"]]
        out["k_from_q_proj_first_128"] = rope_from_fc(yq, c, s).astype(np.float64)
    return {k: v for k, v in out.items() if v is not None}


# ------------------------------------------------------------------------ main
def build(which, out_dir):
    heads, n = SPECS[which]["heads"], SPECS[which]["n"]
    fx = fixture()
    n1 = fx["input"]
    W, b = fx[f"W{which}"], fx[f"b{which}"]
    c16, s16 = tables(heads)

    exp = golden(n1, W, b, c16, s16)
    sim = fp16_sim(n1, W, b, c16, s16)
    yfc = fc_exact(n1, W, b)
    inv_err = check_rotation_invariant(yfc, exp, which)

    blob = build_graph(n1, W, b, c16, s16, heads, n)

    mx = float(np.abs(exp.astype(np.float64)).max())
    ulp = float(np.spacing(np.float16(mx)))
    floor_abs = float(np.abs(sim.astype(np.float64) - exp.astype(np.float64)).max())
    floor_rel = floor_abs / mx
    ident = int((sim == exp).sum())

    # ---- tolerance, registered BEFORE the board run (NOTES §4 / §10.1)
    ULP_REL = 2 ** -11            # fp16 relative granularity (half ULP at a binade top)
    if mx < 1.0:
        # §4 applies unchanged: |out| ~ O(1) -> a flat threshold is meaningful
        atol = max(4.0 * floor_abs, 4.0 * ULP_REL * mx)
        rtol = 4.0 * ULP_REL
        basis = ("NOTES §4 applies: max|out| = %.4f < 1, so a flat absolute threshold is "
                 "meaningful. floor = numpy per-op fp16 simulation (%.3e abs, %.2f ULP) "
                 "-> atol = 4x floor, rtol = 4*2^-11."
                 % (mx, floor_abs, floor_abs / ulp))
    else:
        # §10.1 exception: the device stores every op's result in fp16, so |dev-golden| is
        # >= ~1 ULP of max|out| no matter how right the graph is; a flat 2e-2 is tighter
        # than that (2e-2/max = few e-4 relative).
        atol = 4.0 * ULP_REL * mx
        rtol = 4.0 * ULP_REL
        basis = ("NOTES §10.1 exception: max|out| = %.2f >> 1, so a flat 2e-2 would be "
                 "%.2e relative (tighter than 1 fp16 ULP = %.2e relative). Floor measured "
                 "independently of the device (numpy per-op fp16 sim): %.3e abs = %.2f ULP "
                 "= %.2e relative. Registered atol = 4*2^-11*max|out| = %.5f, "
                 "rtol = 4*2^-11 = %.6g (= 4x the measured floor)."
                 % (mx, 2e-2 / mx, ulp / mx, floor_abs, floor_abs / ulp, floor_rel,
                    atol, rtol))
    thr_at_max = atol + rtol * mx

    os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
    with open(os.path.join(out_dir, "model.tflite"), "wb") as f:
        f.write(blob)
    n1.tofile(os.path.join(out_dir, "golden", "input.bin"))
    exp.tofile(os.path.join(out_dir, "golden", "expected.bin"))

    meta = {
        "op": f"QKV_{which.upper()}_BIAS_ROPE_SPLIT_HALF",
        "shape": {"T": T, "D": D, "heads": heads, "head_dim": HD, "out_width": n,
                  "in": [T, D], "out": [T, heads, HD]},
        "dtype": "fp16",
        "inputs": [{"name": "input", "shape": [T, D], "type": "float16",
                    "bytes": int(n1.nbytes),
                    "what": "post-input_layernorm hidden n1 (see input_choice)"}],
        "output": {"name": "output", "shape": [T, heads, HD], "type": "float16",
                   "bytes": int(exp.nbytes)},
        "quant": {},
        "reference": "numpy_float64_from_stored_fp16",
        "atol": round(atol, 8),
        "rtol": round(rtol, 8),
        "tolerance_basis": basis,
        "tolerance_revision": ("registered before the first board run of this op; derived "
                               "from a device-independent fp16 floor + NOTES §10.1 (no "
                               "post-hoc adjustment).  see docs/OP-qkv.md §5"),
        "fp16_floor": {"abs": floor_abs, "rel": floor_rel, "ulp_at_max": ulp,
                       "max_abs_out": mx, "sim_vs_golden_identical_codes": ident,
                       "codes": int(exp.size),
                       "note": "numpy per-op fp16 rounding simulation vs the exact golden; "
                               "device-independent"},
        "threshold_at_max": float(thr_at_max),
        "input_choice": {
            "graph_input": "post-input_layernorm hidden [8,896] (RMSNorm NOT in the graph)",
            "norm": "x / sqrt(mean(x^2) + eps) * gamma", "eps": EPS,
            "gamma": "layer0.input_layernorm.weight, applied in the pre-graph norm "
                     "(NOT folded into the FC weights)",
            "source": "embed.weight rows for fixed TOKEN_IDS, fp16-stored, float64 norm",
            "token_ids": TOKEN_IDS,
            "why": "RMSNorm has its own operator ticket + board evidence "
                   "(docs/OP-rmsnorm.md); keeping it out keeps a failure single-cause.  "
                   "Folding gamma into W was rejected because the numpy golden would then "
                   "have to fold it too (one more way to be silently wrong)."},
        "weights": {
            "fixture": str(qw.resolve_dir()),
            "weight_tensor": f"layer{LAYER}.attn.{which}_proj.weight",
            "weight_shape": [n, D],
            "weight_sha256": qw.spec(f"layer{LAYER}.attn.{which}_proj.weight")["sha256"],
            "bias_tensor": f"layer{LAYER}.attn.{which}_proj.bias",
            "bias_shape": [n],
            "bias_sha256": qw.spec(f"layer{LAYER}.attn.{which}_proj.bias")["sha256"],
            "bias_in_graph": True,
            "bias_note": "Qwen2.5 q/k/v all have biases and config.json has no key for "
                         "them; measured std 7.8 (q) / 29.8 (k) / 0.015 (v) >> weight std, "
                         "so dropping them shifts the output by >> tolerance "
                         "(docs/OP-weights.md §4)",
            "kv_width_note": ("k/v width is 128 = 2 KV heads x 64, NOT 896.  A 896-wide "
                              "mis-wiring is a different op: see neg_width896/ "
                              "(element count %d vs %d)" % (T * 896, T * n))},
        "rope": {"convention": "qwen2_split_half",
                 "formula": "out_lo = a*cos - b*sin ; out_hi = b*cos + a*sin",
                 "theta": THETA, "head_dim": HD, "half": HALF,
                 "pos_start": POS0, "positions": "%d..%d" % (POS0, POS0 + T - 1),
                 "table": "const pre-computed sin/cos duplicated over heads [T,H,HD/2]; "
                          "SIN/COS are rejected by the ncc importer",
                 "static_note": "T must be static because the tables are data",
                 "rotation_invariant_rel_err": inv_err,
                 "dyn_alternative": "a runtime input table costs +2 SliceLayers and a "
                                    "second input file (docs/OP-rope.md §4); not built here"},
        "decomposition": {"layers_expected": 11,
                          "ops": ["FULLY_CONNECTED", "RESHAPE", "SPLIT(x2)", "MUL(x4)",
                                  "SUB", "ADD", "CONCATENATION"],
                          "rank_note": "output rank 3; rank > 4 is rejected by ncc and rank 4 "
                                       "costs ~4x layers for the same work "
                                       "(docs/OP-rope.md §5.3)"},
        "generator": "tools/build_qkv_bias_rope.py",
        "tflite": "model.tflite",
        "tflite_bytes": len(blob),
    }
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("%s: %s [%d,%d] -> [%d,%d,%d]  tflite=%dB  max|out|=%.4f  floor=%.3e "
          "(%.2f ULP)  atol=%.6g rtol=%.6g thr@max=%.6g"
          % (out_dir, which, T, D, T, heads, HD, len(blob), mx, floor_abs, floor_abs / ulp,
             atol, rtol, thr_at_max))
    return meta


# ---------------------------------------------------------------------- --diag
def diag(which, op_dir):
    """Compare out_device.bin against the golden, the fp16 simulation and the registered
    counterfactuals.  Run after ./tools/e2e.sh; results go into docs/OP-qkv.md."""
    heads, n = SPECS[which]["heads"], SPECS[which]["n"]
    meta = json.load(open(os.path.join(op_dir, "model.json")))
    dev = np.fromfile(os.path.join(op_dir, "out_device.bin"), dtype=np.float16)
    exp = np.fromfile(os.path.join(op_dir, "golden", "expected.bin"), dtype=np.float16)
    if dev.size != exp.size:
        print("FAIL size mismatch device=%d expected=%d" % (dev.size, exp.size))
        return 1
    fx = fixture()
    n1, W, b = fx["input"], fx[f"W{which}"], fx[f"b{which}"]
    c16, s16 = tables(heads)
    sim = fp16_sim(n1, W, b, c16, s16).reshape(-1)
    cfs = counterfactuals(n1, W, b, c16, s16, which)

    mx = float(np.abs(exp.astype(np.float64)).max())
    ulp = float(np.spacing(np.float16(mx)))
    atol, rtol = float(meta["atol"]), float(meta["rtol"])
    thr = atol + rtol * mx

    print("op      : %s  out %s" % (meta["op"], meta["output"]["shape"]))
    print("tolerance registered: atol=%g rtol=%g  threshold@max=%.6g" % (atol, rtol, thr))
    d = np.abs(dev.astype(np.float64) - exp.astype(np.float64))
    ok = bool(np.all(d <= atol + rtol * np.abs(exp.astype(np.float64))))
    print("device vs golden      : max_abs=%.4e (%.2f ULP, rel=%.3e) rmse=%.3e "
          "rel_l2=%.3e -> %s"
          % (d.max(), d.max() / ulp, d.max() / mx, np.sqrt((d ** 2).mean()),
             np.linalg.norm(d) / np.linalg.norm(exp.astype(np.float64)),
             "PASS" if ok else "FAIL"))
    ds = np.abs(dev.astype(np.float64) - sim.astype(np.float64))
    print("device vs fp16 sim    : max_abs=%.4e (%.2f ULP)  identical_codes=%d/%d"
          % (ds.max(), ds.max() / ulp, int((dev == sim).sum()), dev.size))
    print("device vs counterfactuals (must be >> threshold):")
    for name, cf in cfs.items():
        dd = np.abs(dev.astype(np.float64) - cf.reshape(-1).astype(np.float64))
        print("  %-24s max_abs=%.4e  (%6.1fx thr)" % (name, dd.max(), dd.max() / thr))
    return 0 if ok else 1


# ------------------------------------------------------- negative control: 896
def emit_neg_width896(which, op_dir):
    """Build the two "mis-wired to 896" graphs so the failure mode is recorded evidence.

    (a) a_896wide_fc_to_kv_reshape : FC declared 896 wide -> RESHAPE to [8,2,64] is
                                     impossible (7168 elements vs 1024)
    (b) b_declared_896_with_128_data: weight tensor declared [896,896] but only [128,896]
                                     values written -> the E-112 root cause (NOTES §5 pit 2)
    """
    heads, n = SPECS[which]["heads"], SPECS[which]["n"]
    os.makedirs(op_dir, exist_ok=True)
    fx = fixture()
    n1 = fx["input"]
    Wq = qw.get(f"layer{LAYER}.attn.q_proj.weight")            # [896,896] fp16
    bq = qw.get(f"layer{LAYER}.attn.q_proj.bias")              # [896]
    c16q, s16q = tables(14)
    blob = build_graph(n1, Wq, bq, c16q, s16q, heads, D)   # out [8,896] -> reshape [8,2,64]
    open(os.path.join(op_dir, "a_896wide_fc_to_kv_reshape.tflite"), "wb").write(blob)

    c16, s16 = tables(heads)
    W_real, b_real = fx[f"W{which}"], fx[f"b{which}"]
    blob2 = build_graph(n1, W_real, b_real, c16, s16, heads, n, neg="declared_size")
    open(os.path.join(op_dir, "b_declared_896_with_128_data.tflite"), "wb").write(blob2)
    print("%s: neg controls written (a=896-wide FC -> [%d,%d,%d] reshape; "
          "b=declared [896,896] with [%d,%d] data)" % (op_dir, T, heads, HD, n, D))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=sorted(SPECS))
    ap.add_argument("--out")
    ap.add_argument("--diag", action="store_true", help="compare out_device.bin (after e2e)")
    ap.add_argument("--emit-neg-width896", dest="neg896", action="store_true",
                    help="emit the 'k/v mis-wired to 896' negative-control graphs")
    a = ap.parse_args()
    if not a.which:
        raise SystemExit("--which q|k|v is required")
    out = a.out or DEFAULT_OUT[a.which]
    if a.diag:
        return diag(a.which, out)
    if a.neg896:
        emit_neg_width896(a.which, os.path.join(out, "neg_width896"))
        return 0
    build(a.which, out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
