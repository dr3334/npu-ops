#!/usr/bin/env python3
"""RoPE (rotary position embedding), Qwen2 **split-half** convention, as a TFLite probe.

    /usr/bin/python3 tools/build_rope.py --list
    /usr/bin/python3 tools/build_rope.py rope_1x8x64_cosconst_fp16 [--out ops/<name>]
    /usr/bin/python3 tools/build_rope.py --verify-hf     # convention audit (needs torch)

Scope / why it is built this way
--------------------------------
The ncc importer **rejects SIN / COS** ("not supported"), so a rotary embedding cannot
be expressed with a transcendental in the graph: the cos/sin tables must be *data*.
This generator therefore only uses operators this repo has already proven to reach
MDLA_5_0 without a bridge: SPLIT, MUL, SUB, ADD, CONCATENATION (docs/OP-shapeops.md,
docs/OP-elementwise.md).

Qwen2 split-half rotary, head_dim D = 64, half h = 32, a = x[..., :h], b = x[..., h:]:

    out[..., :h] = a*cos - b*sin
    out[..., h:] = b*cos + a*sin          <-- NOTE, see "convention" below

with angle(p, i) = (p0 + p) * theta^(-2i/D), theta = 1e6 (Qwen2).

CONVENTION (this cost one full board pass, keep the note - AGENTS.md R4)
------------------------------------------------------------------------
The task statement this worktree started from wrote the second line as
"b' = b*sin + a*cos".  That is **not a rotation**: applied to the unit vector (a,b) =
(1,0) at theta=0.7 it returns (0.7648, 0.7648), norm 1.08, while a rotation must return
(cos, sin) = (0.7648, 0.6442), norm 1.  HuggingFace's Qwen2 code is
`q*cos + rotate_half(q)*sin` with `rotate_half = cat((-x2, x1))`, which is exactly
(a*cos - b*sin, b*cos + a*sin).  The first implementation of this file followed the
literal (swapped) text and *agreed with its own golden to 0 codes* - a textbook R3
"both sides wrong together" result.  Two guards are now permanent:
  * `--verify-hf` audits the formula against transformers' Qwen2 rotary code;
  * `build()` asserts the rotation invariant (per-pair 2-norm preserved) before it
    writes any artifact.  A cos/sin swap breaks that invariant immediately.

Deliberate choices (experiment design, not incidental code)
-----------------------------------------------------------
* **positions start at p0 = 5, not 0.**  At p = 0 the table is cos = 1, sin = 0, so
  out == in and *every* bug in the sign/swizzle ordering would still pass.
* **the correct rotation needs all four products** (a*cos, b*sin, b*cos, a*sin); the
  tempting "2 MUL, re-use the products" form is only valid for the *swapped* (wrong)
  formula.  Two decompositions are therefore kept:
    - `mul=4mul`  : SPLIT -> 4 MUL -> SUB / ADD -> CONCAT   (works with const *and*
                    runtime tables; 9 real MDLA layers)
    - `mul=2mul`  : SPLIT -> CONCAT(b,a) swap -> 2 MUL -> ADD  (const only, because it
                    needs the tables pre-folded as [cos|cos] and [-sin|sin]; 6 layers)
* **table modes** - the experiment for "can cos/sin be a runtime input?":
    const2 = cos, sin as two const buffers (half-width)
    const1 = one packed const [cos|sin] table, split in-graph  (control for dyn1)
    dyn1   = the same packed table as a **runtime input**       (golden/input1.bin)
    const_dh = two *full-width* const tables [cos|cos], [-sin|sin] (2mul path)
  const1 vs dyn1 differ in exactly one variable: the table tensor is a const buffer or
  a subgraph input.
* dtype is **fp16** (rotary activations in an LLM block are fp16).
* every model is far above the 16-byte innermost-row rule (docs/OP-MATRIX.md §2):
  fp16 last dim 64 -> 128 B/row, split halves -> 64 B/row.

Golden (AGENTS.md R3)
---------------------
`golden/expected.bin` is computed HERE with numpy from the **stored** fp16 inputs
(cast to float64), never from ncc / the TFLite interpreter / the device.  The one
allowed rounding is float64 -> float16 at the end (storage type).

Tolerance is **registered before the board run** (SPECS + model.json):
    atol = 2e-3, rtol = 1e-3
Derivation (fp16, |x| <= 1, |cos|,|sin| <= 1 => |out| <= 2):
    p1..p4 = fl16(a*cos), fl16(b*sin), fl16(b*cos), fl16(a*sin) : 0.5 ULP(1.0) each
    lo/hi  = fl16(p1 -/+ p2)   : 0.5 ULP(|out| <= 2) + the propagated product error
    expected = fl16(exact)     : the reference's own storage rounding
=> every term is a *relative* fp16 rounding, so the bound scales with |out|: atol is
the absolute floor (~2 ULP at 1.0), rtol covers the 1-2 ULP slop at |out| ~ 2.
Anything bigger is an ordering/layout bug, not fp16 noise.
"""
import argparse
import json
import os

import flatbuffers
import numpy as np

import build_ops as bo

SEED = 20261112
D_HEAD = 64                     # Qwen2 head_dim in scope
THETA = 1e6                     # Qwen2 rope_theta
POS0 = 5                        # first position (see "positions start at p0")
F16, I32 = bo.F16, bo.I32
S = bo.S
BO, BOPT = bo.S.BuiltinOperator, bo.S.BuiltinOptions


# ------------------------------------------------------------------ spec table
# keys: shape / table (const2|const1|dyn1|const_dh) / mul (4mul|2mul) / table_values
SPECS = {
    # T=1, single head: the decode step.  Smallest graph that is still non-degenerate.
    "rope_1x1x64_cosconst_fp16":  dict(shape=[1, 1, D_HEAD],  table="const2"),
    # T=8, single head: 8 distinct positions -> a position mix-up cannot hide.
    "rope_1x8x64_cosconst_fp16":  dict(shape=[1, 8, D_HEAD],  table="const2"),
    # T=8, 2 heads as a 4-D tensor [1,T,H,D]: 2 heads must not be mixed
    "rope_1x8x2x64_cosconst_fp16": dict(shape=[1, 8, 2, D_HEAD], table="const2"),
    # one packed const table + in-graph SPLIT: control for the dyn1 variant
    "rope_1x8x64_tableconst_fp16": dict(shape=[1, 8, D_HEAD], table="const1"),
    # one packed *runtime input* table + in-graph SPLIT: the position-dependent case
    "rope_1x8x64_tabledyn_fp16":  dict(shape=[1, 8, D_HEAD],  table="dyn1"),
    "rope_1x1x64_tabledyn_fp16":  dict(shape=[1, 1, D_HEAD],  table="dyn1"),
    # discriminating control: the runtime table is random noise, not a rotary table.
    # If the device still matches this golden, the table is consumed as real data (it
    # cannot be recomputed, const-folded, or ignored).  marked table_values=wild_random
    "rope_1x8x64_tabledyn_wild_fp16": dict(shape=[1, 8, D_HEAD], table="dyn1",
                                           table_values="wild_random"),
    # ---- layer-count controls for the 4-D blow-up -------------------------------
    # (c1) rank 4 with ONE head: distinguishes "rank 4" from "2 heads"
    "rope_1x8x1x64_cosconst_fp16": dict(shape=[1, 8, 1, D_HEAD], table="const2"),
    # (c2) rank 3 with as many rows as the 2-head case: heads folded into an earlier dim
    "rope_1x16x64_cosconst_fp16":  dict(shape=[1, 16, D_HEAD], table="const2"),
    # ---- cheapest *correct* const decomposition (pre-folded signed tables) -------
    "rope_1x8x64_dupconst_2mul_fp16": dict(shape=[1, 8, D_HEAD], table="const_dh",
                                           mul="2mul"),
}


# --------------------------------------------------------------- rotary tables
def tables(shape, p0=POS0, theta=THETA):
    """cos/sin of the *half* head vector (shape[:-1] + [h]), fp16.

    angle(p, i) = (p0 + p) * theta ** (-2i / D), i = 0..h-1.
    Returned as the *stored* fp16 arrays: the golden uses exactly these values.
    """
    D = shape[-1]
    h = D // 2
    T = int(np.prod(shape[:-1]))          # all "position-like" leading dims
    inv = theta ** (-2.0 * np.arange(h) / D)
    ang = np.outer(p0 + np.arange(T), inv)              # T x h
    hs = list(shape[:-1]) + [h]
    return (np.cos(ang).astype(np.float16).reshape(hs),
            np.sin(ang).astype(np.float16).reshape(hs))


def golden(x16, cos16, sin16, store=True):
    """Independent reference: numpy, float64 on the stored fp16 values.

    Qwen2 split-half:  out_lo = a*cos - b*sin ,  out_hi = b*cos + a*sin.
    `store=True` applies the single allowed rounding (float64 -> float16 storage type);
    `store=False` keeps float64 (used by the HF convention audit, so the comparison is
    not dominated by the fp16 storage rounding).
    """
    h = x16.shape[-1] // 2
    xn = x16.astype(np.float64)
    c, s = cos16.astype(np.float64), sin16.astype(np.float64)
    a, b = xn[..., :h], xn[..., h:]
    y = np.concatenate([a * c - b * s, b * c + a * s], axis=-1)
    return y.astype(np.float16) if store else y


def check_rotation_invariant(x16, y16, label):
    """A rotation preserves the 2-norm of every (a,b) pair.  Assert it on the golden.

    This is the guard that the swapped (non-rotation) formula cannot pass: for a unit
    (a,b)=(1,0) the swapped form gives |out| = 1.08.  Tolerance is dominated by fp16
    storage of the 64 outputs, so allow 3e-3 relative.
    """
    h = x16.shape[-1] // 2
    xn, yn = x16.astype(np.float64), y16.astype(np.float64)
    nin = np.hypot(xn[..., :h], xn[..., h:])
    nout = np.hypot(yn[..., :h], yn[..., h:])
    worst = float(np.abs(nout - nin).max() / max(float(np.abs(nin).max()), 1e-9))
    if worst > 3e-3:
        raise SystemExit(
            f"{label}: golden does NOT preserve the pair norm (rel err {worst:.2e}) "
            f"-> the formula is not a rotation (cos/sin swapped?)")
    return worst


# --------------------------------------------------------- flatbuffer helpers
class M:
    """Same accumulator shape as tools/build_shapeops.py (buffer 0 = placeholder)."""

    def __init__(self):
        self.b = flatbuffers.Builder(16384)
        self.buffers = [bo.buf_empty(self.b)]
        self.tensors, self.opcodes, self.ops = [], [], []

    def const(self, name, shape, ttype, arr):
        self.buffers.append(bo.buf_data(self.b, np.ascontiguousarray(arr)))
        self.tensors.append(bo.tensor(self.b, name, list(shape), ttype,
                                      len(self.buffers) - 1))
        return len(self.tensors) - 1

    def act(self, name, shape, ttype):
        """Activation / model-IO tensor: no data buffer (AGENTS §7.3)."""
        self.tensors.append(bo.tensor(self.b, name, list(shape), ttype, 0))
        return len(self.tensors) - 1

    def add_op(self, builtin, ins, outs, opt_type=None, opt=None):
        self.opcodes.append(bo.opcode(self.b, builtin))
        self.ops.append(bo.operator(self.b, len(self.opcodes) - 1, ins, outs,
                                    opt_type, opt))

    # options tables must exist before the Operator that references them
    def mul_opt(self):
        S.MulOptionsStart(self.b)
        S.MulOptionsAddFusedActivationFunction(self.b, 0)
        return BOPT.MulOptions, S.MulOptionsEnd(self.b)

    def add_opt(self):
        S.AddOptionsStart(self.b)
        S.AddOptionsAddFusedActivationFunction(self.b, 0)
        return BOPT.AddOptions, S.AddOptionsEnd(self.b)

    def sub_opt(self):
        S.SubOptionsStart(self.b)
        S.SubOptionsAddFusedActivationFunction(self.b, 0)
        return BOPT.SubOptions, S.SubOptionsEnd(self.b)

    def split_opt(self, n):
        S.SplitOptionsStart(self.b)
        S.SplitOptionsAddNumSplits(self.b, n)
        return BOPT.SplitOptions, S.SplitOptionsEnd(self.b)

    def concat_opt(self, axis):
        S.ConcatenationOptionsStart(self.b)
        S.ConcatenationOptionsAddAxis(self.b, axis)
        S.ConcatenationOptionsAddFusedActivationFunction(self.b, 0)
        return BOPT.ConcatenationOptions, S.ConcatenationOptionsEnd(self.b)

    def finish(self, ins, outs, desc):
        return bo.finish(self.b, self.buffers, self.opcodes, self.tensors,
                         self.ops, ins, outs, desc)


def build(cfg, out_dir):
    shape = list(cfg["shape"])
    mode = cfg["table"]
    mul = cfg.get("mul", "4mul")
    D, h = shape[-1], shape[-1] // 2
    axis = len(shape) - 1
    hs = shape[:-1] + [h]                 # half-width activation shape

    # ---- data (stored fp16 = what the device and the golden both see)
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-1.0, 1.0, size=shape).astype(np.float16)
    # tripwires: exact 0, both signs, +-1 and fractions -> a sign flip or half swap
    # cannot hide in random noise
    flat = x.reshape(-1)
    flat[:8] = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75], np.float16)
    x = flat.reshape(shape)

    cos16, sin16 = tables(shape)                        # half-width, fp16
    if cfg.get("table_values") == "wild_random":
        # control: values unrelated to any position/angle
        r2 = np.random.default_rng(SEED + 1)
        cos16 = r2.uniform(-1.0, 1.0, size=hs).astype(np.float16)
        sin16 = r2.uniform(-1.0, 1.0, size=hs).astype(np.float16)
    exp = golden(x, cos16, sin16)
    if cfg.get("table_values") != "wild_random":
        inv_err = check_rotation_invariant(x, exp, os.path.basename(out_dir))
    else:
        inv_err = None

    # ---- graph
    m = M()
    t_x = m.act("input", shape, F16)
    t_axis = m.const("axis", [1], I32, np.array([axis], np.int32))
    feed = [("input", x)]
    model_inputs = [t_x]

    t_a = m.act("a", hs, F16)
    t_b = m.act("b", hs, F16)
    ot, opt = m.split_opt(2)
    m.add_op(BO.SPLIT, [t_axis, t_x], [t_a, t_b], ot, opt)
    t_y = m.act("output", shape, F16)

    if mul == "2mul":
        # ---- cheapest *correct* const form: out = x*[cos|cos] + swap(x)*[-sin|sin]
        # only legal with pre-folded full-width const tables (const_dh)
        cc = np.concatenate([cos16, cos16], axis=-1)     # [.., h]*2 = [.., D]
        ms = np.concatenate([-sin16, sin16], axis=-1)
        t_cc = m.const("cos_cos", shape, F16, cc)
        t_ms = m.const("neg_sin_sin", shape, F16, ms)
        t_sw = m.act("swap", shape, F16)
        ot, opt = m.concat_opt(axis)
        m.add_op(BO.CONCATENATION, [t_b, t_a], [t_sw], ot, opt)   # [b, a]
        t_u = m.act("u", shape, F16)
        t_v = m.act("v", shape, F16)
        ot, opt = m.mul_opt()
        m.add_op(BO.MUL, [t_x, t_cc], [t_u], ot, opt)             # [a*c, b*c]
        ot, opt = m.mul_opt()
        m.add_op(BO.MUL, [t_sw, t_ms], [t_v], ot, opt)            # [-b*s, a*s]
        ot, opt = m.add_opt()
        m.add_op(BO.ADD, [t_u, t_v], [t_y], ot, opt)
        n_mul = 2
        note = "2 MUL + swap CONCAT + ADD (const tables pre-folded, one of them signed)"
    else:
        # ---- faithful split/4-MUL/SUB/ADD/concat form (const or runtime tables)
        if mode == "const2":
            t_cos = m.const("cos", hs, F16, cos16)
            t_sin = m.const("sin", hs, F16, sin16)
        elif mode in ("const1", "dyn1"):
            packed = np.concatenate([cos16, sin16], axis=-1)      # [cos|sin]
            if mode == "const1":
                t_tab = m.const("rope_table", shape, F16, packed)
            else:
                t_tab = m.act("rope_table", shape, F16)
                model_inputs.append(t_tab)
                feed.append(("rope_table", packed))
            t_cos = m.act("cos", hs, F16)
            t_sin = m.act("sin", hs, F16)
            ot, opt = m.split_opt(2)
            m.add_op(BO.SPLIT, [t_axis, t_tab], [t_cos, t_sin], ot, opt)
        else:
            raise SystemExit(f"bad table mode {mode}")

        t_p = [m.act(nm, hs, F16) for nm in ("a_cos", "b_sin", "b_cos", "a_sin")]
        ot, opt = m.mul_opt()
        m.add_op(BO.MUL, [t_a, t_cos], [t_p[0]], ot, opt)         # a*cos
        ot, opt = m.mul_opt()
        m.add_op(BO.MUL, [t_b, t_sin], [t_p[1]], ot, opt)         # b*sin
        ot, opt = m.mul_opt()
        m.add_op(BO.MUL, [t_b, t_cos], [t_p[2]], ot, opt)         # b*cos
        ot, opt = m.mul_opt()
        m.add_op(BO.MUL, [t_a, t_sin], [t_p[3]], ot, opt)         # a*sin
        t_lo = m.act("rot_lo", hs, F16)
        t_hi = m.act("rot_hi", hs, F16)
        ot, opt = m.sub_opt()
        m.add_op(BO.SUB, [t_p[0], t_p[1]], [t_lo], ot, opt)       # a*cos - b*sin
        ot, opt = m.add_opt()
        m.add_op(BO.ADD, [t_p[2], t_p[3]], [t_hi], ot, opt)       # b*cos + a*sin
        ot, opt = m.concat_opt(axis)
        m.add_op(BO.CONCATENATION, [t_lo, t_hi], [t_y], ot, opt)
        n_mul = 4
        note = "SPLIT -> 4 MUL -> SUB/ADD -> CONCAT (faithful decomposition)"

    blob = m.finish(model_inputs, [t_y],
                    f"rope split-half {'x'.join(map(str, shape))} fp16 {mode} {mul}")

    # ---- artifacts
    os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
    with open(os.path.join(out_dir, "model.tflite"), "wb") as f:
        f.write(blob)
    for i, (nm, arr) in enumerate(feed):
        arr.tofile(os.path.join(out_dir, "golden",
                                "input.bin" if i == 0 else f"input{i}.bin"))
    exp.tofile(os.path.join(out_dir, "golden", "expected.bin"))

    meta = {
        "op": "ROPE_SPLIT_HALF",
        "shape": {"in": shape},
        "dtype": "fp16",
        "inputs": [{"name": nm, "shape": list(arr.shape), "type": "float16",
                    "bytes": int(arr.nbytes)} for nm, arr in feed],
        "output": {"name": "output", "shape": shape, "type": "float16",
                   "bytes": int(exp.nbytes)},
        "quant": {},
        "reference": "numpy_float64_from_stored_fp16",
        "atol": 2e-3,
        "rtol": 1e-3,
        "tolerance_rationale": (
            "fp16 chain: 4 MUL roundings (0.5 ULP each at |x*cos| <= 1) + 1 ADD/SUB "
            "rounding (0.5 ULP at |out| <= 2) + the reference's own storage rounding; "
            "every term is a relative fp16 rounding, so atol covers the floor at "
            "|out|~1 and rtol covers the 1-2 ULP slop at |out|~2.  Beyond that = "
            "ordering/layout bug."),
        "rope": {"convention": "qwen2_split_half",
                 "formula": "out_lo = a*cos - b*sin ; out_hi = b*cos + a*sin",
                 "convention_note": (
                     "the 'b*sin + a*cos' variant written in the original task text is "
                     "NOT a rotation (it does not preserve the pair norm); this "
                     "generator asserts norm preservation and audits against "
                     "transformers Qwen2 rotate_half -- see docs/OP-rope.md"),
                 "theta": THETA, "head_dim": D, "half": h, "pos_start": POS0,
                 "table_values": cfg.get("table_values", "rotary")},
        "decomposition": {"table": mode, "mul": mul, "muls": n_mul, "note": note,
                          "table_note": {
                              "const2": "cos/sin as two half-width const buffers",
                              "const1": "one packed const [cos|sin] table, split in-graph",
                              "dyn1": "packed [cos|sin] table is a RUNTIME input "
                                      "(golden/input1.bin)",
                              "const_dh": "two full-width const tables [cos|cos], "
                                          "[-sin|sin] (signed table folded in)"}[mode]},
        "rotation_invariant_rel_err": inv_err,
        "seed": SEED,
        "pos_start_note": ("positions start at 5: at p=0 cos=1, sin=0 and out==in, so "
                           "a p=0 table would let sign/swizzle bugs pass"),
        "generator": "tools/build_rope.py",
        "tflite": "model.tflite",
        "tflite_bytes": len(blob),
    }
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{out_dir}: ROPE {shape} {mode}/{mul}  tflite={len(blob)}B  "
          f"inputs={[nm for nm, _ in feed]}  norm_invariant_rel={inv_err}  "
          f"(atol=2e-3 rtol=1e-3 registered)")


def verify_hf(specs=None):
    """Convention audit against HuggingFace transformers' *Qwen2* rotary code.

    This is NOT the golden (R3: expected.bin stays numpy and is written before any
    board run).  It answers one question: is the formula this probe implements really
    the Qwen2 split-half convention?  transformers:
        rotate_half(x) = cat((-x2, x1)) ;  q_embed = q*cos + rotate_half(q)*sin
    Two comparisons per shape:
      (1) ours with float64 tables vs HF -> agreement = same convention
      (2) ours with the fp16-stored tables vs HF -> what the fp16 table itself costs
    """
    import torch
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2RotaryEmbedding, apply_rotary_pos_emb)

    for name in (specs or SPECS):
        cfg_s = SPECS[name]
        shape = list(cfg_s["shape"])
        h = shape[-1] // 2
        T = int(np.prod(shape[:-1]))
        rng = np.random.default_rng(SEED)
        x = rng.uniform(-1.0, 1.0, size=shape).astype(np.float16)

        inv = THETA ** (-2.0 * np.arange(h) / shape[-1])
        ang = np.outer(POS0 + np.arange(T), inv)
        c64, s64 = np.cos(ang), np.sin(ang)
        c16, s16 = c64.astype(np.float16), s64.astype(np.float16)
        hs = shape[:-1] + [h]
        g64 = golden(x, c64.reshape(hs), s64.reshape(hs), store=False)  # no fp16 round
        g16 = golden(x, c16.reshape(hs), s16.reshape(hs))       # the stored tables

        config = Qwen2Config(head_dim=shape[-1], rope_theta=THETA,
                             max_position_embeddings=4096, num_attention_heads=1,
                             hidden_size=shape[-1], intermediate_size=shape[-1],
                             num_hidden_layers=1)
        rot = Qwen2RotaryEmbedding(config)
        xt = torch.from_numpy(x.astype(np.float32)).reshape(1, 1, T, shape[-1])
        pos = torch.arange(POS0, POS0 + T).unsqueeze(0)
        cos, sin = rot(xt, pos)
        hf = apply_rotary_pos_emb(xt, xt, cos, sin, 1)[0].reshape(shape)
        hf = hf.numpy().astype(np.float64)
        d64 = float(np.abs(g64.astype(np.float64) - hf).max())
        d16 = float(np.abs(g16.astype(np.float64) - hf).max())
        print(f"{name:34s} {str(shape):18s} ours(f64 tables) vs HF: {d64:.3e}   "
              f"ours(fp16 tables) vs HF: {d16:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--verify-hf", action="store_true",
                    help="convention audit vs transformers Qwen2 rotary (not the golden)")
    a = ap.parse_args()
    if a.list:
        print("\n".join(SPECS))
        return
    if a.verify_hf:
        verify_hf()
        return
    if a.spec not in SPECS:
        raise SystemExit("unknown spec; use --list")
    build(SPECS[a.spec], a.out or f"ops/{a.spec}")


if __name__ == "__main__":
    main()
