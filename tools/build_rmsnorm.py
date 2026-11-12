#!/usr/bin/env python3
"""RMSNorm and its primitive (RSQRT) on MDLA_5_0 (MT6897) — branch `feat/rmsnorm`.

Scope (docs/AGENT-BRIEF.md):
    y = x * rsqrt(mean(x^2, axis=-1) + eps) * gamma        (per token, hidden = 64)

Three questions, in this order:
  1. PRIMITIVE  — does ncc-tflite accept TFLite RSQRT for mdla5.0 (fp16 / int8), or does it
     need a CPU bridge?  (If rejected, the fallback candidates are SQRT+RECIPROCAL or DIV;
     that path is only built when the probe fails, see docs/OP-rmsnorm.md.)
  2. COMPOSED GRAPH — MEAN(last axis) -> ADD eps -> RSQRT -> MUL x -> MUL gamma, compiled and
     run on the NPU as ONE graph, compared against an independent numpy golden.
  3. SHAPE/ALIGNMENT — [1,64] and [8,64] (T=8 = block prefill), int8 and fp16.
     The workspace rule (docs/CAPABILITY.md §6) is: the innermost row of a DRAM-boundary
     tensor must be >= 16 bytes and %16 == 0.  [T,64] fp16 = 128 B, int8 = 64 B, both fine;
     `rmsnorm_8x4_fp16_row8b` is the deliberate negative control (row = 8 B).

Method rules honoured here (AGENTS.md):
  R1  every numeric contract (atol/rtol for float, int8_tol for quantised) is registered in
      SPECS **before** any compile or board run, and copied verbatim to model.json.
      The rationale per entry is written out in `tolerance_rationale`.
  R3  the golden is computed HERE from the *stored* input constants in float32/int64, and
      never passes through ncc, the TFLite interpreter or the device runtime.
  R7  this script only produces artifacts; compile / device / numerics are reported by
      tools/compile.sh and tools/compare.py (five conclusions kept separate).
  R6  no CPU fallback: the graph is compiled with --disallow-bridge, and every layer that
      does not land on MDLA_5_0 shows up in model.compile.log as "gap is detected".

Quantisation choices (int8 chain) — all zero-points are 0 and every scale is a power of two,
so each requant multiplier is an exact power of two (the only rounding is the rounding of the
*result*, no multiplier approximation):
    x      s_x   = 1/16      |x| <= 2        (test data limited to |code| <= 32)
    x*x    s_sq  = 1/8       mult 1/32       code_q  = round(q_x^2 / 32)  <= 32
    mean   s_m   = 1/8       (same as sq -> MEAN does one divide by 64)
    +eps   s_m2  = 1/8       eps_code = 0    -> the int8 eps ADD is an *identity* in code
                                              space (eps = 1e-5 < 0.5 * s = 0.0625).  That is
                                              a real property of the int8 path, recorded, not
                                              a convenience: it means int8 RMSNorm cannot
                                              honour eps, only fp16 can.
    rsqrt  s_r   = 1/32      |r| <= 4        r = 1/sqrt(mean) in [0.7, 1.2] for our data
    x*r    s_y1  = 1/32      mult 1/16
    gamma  s_g   = 1/64      gamma ~ 1 +- 0.1
    y      s_y   = 1/32      mult 1/64       |y| <= ~2

Usage (WSL python that has the `tflite` flatbuffer schema package):
    /usr/bin/python3 tools/build_rmsnorm.py --list
    /usr/bin/python3 tools/build_rmsnorm.py rmsnorm_8x64_fp16 \
        --out ops/rmsnorm_8x64_fp16/model.tflite
"""
import argparse
import json
import os

import flatbuffers
import numpy as np

import build_ops as bo

SEED = 20261112

F16, I8, I32, F32T = bo.F16, bo.I8, bo.I32, bo.F32
BO, BOPT = bo.S.BuiltinOperator, bo.S.BuiltinOptions
TYPE_NAME = {F16: "float16", I8: "int8", I32: "int32", F32T: "float32"}

# int8 scale table (see the module docstring; one place, used by both model and golden)
Q = dict(s_x=1 / 16.0, s_sq=1 / 8.0, s_m=1 / 8.0, s_m2=1 / 8.0,
         s_r=1 / 32.0, s_y1=1 / 32.0, s_g=1 / 64.0, s_y=1 / 32.0)


# --------------------------------------------------------------- spec table
# Tolerances are REGISTERED HERE, before running (AGENTS.md R1).  Do not edit retroactively;
# if a number turns out to be wrong, say so in docs/OP-rmsnorm.md instead of loosening it.
SPECS = {
    # ---------------- 1. the primitive: RSQRT alone ----------------------------------
    "rsqrt_1x64_fp16": dict(
        kind="rsqrt", shape=[1, 64], dtype="fp16", in_lo=0.25, in_hi=4.0,
        atol=1e-2, rtol=1e-2,
        prediction=(
            "H1 (native): RSQRT maps to an MDLA elementwise/rsqrt layer, EXIT=0. "
            "H2 (bridge): 'Bridging is disallowed but gap is detected between Input and "
            "MDLA_5_0'. Static prior for H1: ncc-tflite contains the strings MTKEXT_RSQRT / "
            "'RSqrt: ParamSetup fails.' / XFL_RSQRT_QUANT, and libmvpu_isa.so exports the "
            "instruction v_rsqrt_f32. Numeric prediction: |y - 1/sqrt(x)| <= 1 fp16 ULP "
            "(<= 1e-3), i.e. far inside the registered 1e-2."),
    ),
    "rsqrt_1x64_int8": dict(
        kind="rsqrt", shape=[1, 64], dtype="int8", in_lo=0.5, in_hi=7.9,
        int8_tol=2,
        prediction=(
            "Weaker prior than fp16: the int8 path needs a table/approximation (the "
            "XFL_RSQRT_QUANT string suggests a quantised variant exists). Prediction: "
            "EXIT=0 with <= 2 codes of 1/64 error; if it needs a requant the compiler may "
            "instead report IN_GAP. Recorded either way."),
    ),
    # ---------------- 2. the composed graph (fp16: the reference path) ---------------
    "rmsnorm_1x64_fp16": dict(
        kind="rmsnorm", shape=[1, 64], dtype="fp16", eps=1e-5,
        atol=1e-2, rtol=1e-2,
        prediction=(
            "Prediction: 5-6 layers on MDLA_5_0 in one execution step "
            "(MulLayer, MeanLayer, AddLayer, RSqrt/RsqrtLayer, MulLayer x2), EXIT=0, max_abs "
            "<= ~2e-3. Main risks, in order: (a) broadcast MUL [T,64]x[T,1] needs a CPU "
            "relayout -> IN_GAP; (b) MUL(x,x) lists the same tensor twice -> importer may "
            "reject or require a copy; (c) the [T,1] intermediate has a 2-byte innermost row, "
            "which violates the 16-byte rule *if* intermediates are DRAM-boundary tensors."),
    ),
    "rmsnorm_8x64_fp16": dict(
        kind="rmsnorm", shape=[8, 64], dtype="fp16", eps=1e-5,
        atol=1e-2, rtol=1e-2,
        prediction=(
            "Same layer list as [1,64]; T=8 is the block-prefill shape. Predict EXIT=0 and "
            "the same accuracy, since only the row count changes (rows are independent, the "
            "reduction is still over the 64-wide axis)."),
    ),
    "rmsnorm_8x64_fp16_eps1": dict(
        kind="rmsnorm", shape=[8, 64], dtype="fp16", eps=1.0,
        atol=1e-2, rtol=1e-2,
        prediction=(
            "STEP-OBSERVABILITY ABLATION (one variable: eps). With eps=1.0 the ADD term is "
            "13% of the denominator, so if the compiler folds/drops/reorders the ADD eps step "
            "the output moves by >1e-1 and the 1e-2 contract fails loudly. With the realistic "
            "eps=1e-5 the ADD is a 3e-6 relative perturbation of a mean(x^2)~3 signal and is "
            "numerically invisible. Prediction: EXIT=0 and PASS -- i.e. the eps ADD really is "
            "executed, not optimised away."),
    ),
    # ---------------- 2b. composed graph, int8 (row = 64 B, the other half of rule) ---
    "rmsnorm_1x64_int8": dict(
        kind="rmsnorm", shape=[1, 64], dtype="int8", eps=1e-5,
        int8_tol=4,
        prediction=(
            "Compile: expect the same MDLA layer list as fp16. Numerics: the int8 chain has "
            "four requant/rounding stages (x*x, MEAN /64, RSQRT table, x*r, y*gamma) so a "
            "code-exact result is not expected; registered contract is 4 codes of 1/32. "
            "eps is *not* representable in int8 (code rounds to 0) -> the ADD is an identity."),
    ),
    "rmsnorm_8x64_int8": dict(
        kind="rmsnorm", shape=[8, 64], dtype="int8", eps=1e-5,
        int8_tol=4,
        prediction="Same as rmsnorm_1x64_int8 with 8 rows; expect the same <=4 code spread.",
    ),
    # ---------------- 3. alignment rule controls ------------------------------------
    "rmsnorm_8x8_fp16": dict(
        kind="rmsnorm", shape=[8, 8], dtype="fp16", eps=1e-5,
        atol=1e-2, rtol=1e-2,
        prediction=(
            "POSITIVE CONTROL for the 16-byte rule: row = 8 elements * 2 B = 16 B -> predict "
            "EXIT=0. Also narrows the reduce axis to 8, which is the smallest useful hidden "
            "size and makes the [T,1] intermediate relatively more expensive."),
    ),
    "rmsnorm_8x4_fp16_row8b": dict(
        kind="rmsnorm", shape=[8, 4], dtype="fp16", eps=1e-5,
        atol=1e-2, rtol=1e-2, expect_compile_fail=True,
        prediction=(
            "NEGATIVE CONTROL for the 16-byte rule: row = 4 elements * 2 B = 8 B < 16 -> "
            "predict EXIT=1 with 'Bridging is disallowed but gap is detected between Input "
            "and MDLA_5_0' *after* the layer list has already been printed (docs/CAPABILITY.md "
            "§6: the exec plan is printed before the bridge check, so the layer name alone "
            "must not be read as success)."),
    ),
    "rsqrt_1x64_fp32": dict(
        kind="rsqrt", shape=[1, 64], dtype="fp32", in_lo=0.25, in_hi=4.0,
        atol=1e-5, rtol=1e-4, expect_compile_fail=True,
        prediction=(
            "DISCRIMINATOR for the fp32 question (one variable: keep the op, change the "
            "dtype of the composed graph's most exotic layer). Prediction: also rejected at "
            "import ('Fail to process the given model'), i.e. fp32 is not importable for "
            "MDLA_5_0 elementwise ops at all; if instead this one compiles, then the fp32 "
            "blocker lives in another op of the chain (MEAN/broadcast MUL) and the dossier "
            "says so instead of blaming 'fp32' in general."),
    ),
    # ---------------- 4. fp32-ish path (asked for explicitly) ------------------------
    "rmsnorm_1x64_fp32": dict(
        kind="rmsnorm", shape=[1, 64], dtype="fp32", eps=1e-5,
        atol=1e-5, rtol=1e-4, expect_compile_fail=True,
        prediction=(
            "fp32 is not directly connectable on mdla5.0 (docs/CAPABILITY.md §5, E-004): "
            "predict IN_GAP at compile time, so the 1e-5 tolerance can never be exercised. "
            "Recorded so the 'is there an fp32 path?' question has evidence, not folklore."),
    ),
}


# --------------------------------------------------------------- flatbuffers
class M:
    """Tiny flatbuffer accumulator (buffer 0 = the empty placeholder, per TFLite)."""

    def __init__(self):
        self.b = flatbuffers.Builder(16384)
        self.buffers = [bo.buf_empty(self.b)]
        self.tensors = []
        self._oc = []          # opcode table offsets; index == opcode index
        self._oci = {}         # builtin -> opcode index (dedup, like TFLite does)
        self.ops = []

    def buf(self, arr):
        self.buffers.append(bo.buf_data(self.b, arr))
        return len(self.buffers) - 1

    def const(self, name, shape, ttype, arr, q=None):
        self.tensors.append(bo.tensor(self.b, name, list(shape), ttype,
                                      self.buf(arr), q))
        return len(self.tensors) - 1

    def act(self, name, shape, ttype, q=None):
        """Intermediate activation.  Buffer 0 (empty) = 'not a constant'."""
        self.tensors.append(bo.tensor(self.b, name, list(shape), ttype, 0, q))
        return len(self.tensors) - 1

    def out(self, name, shape, ttype, q=None):
        """Graph output; own empty buffer, mirroring tools/build_shapeops.py's chain model."""
        self.buffers.append(bo.buf_empty(self.b))
        self.tensors.append(bo.tensor(self.b, name, list(shape), ttype,
                                      len(self.buffers) - 1, q))
        return len(self.tensors) - 1

    def op(self, builtin, ins, outs, opt_type=None, opt=None):
        if builtin in self._oci:
            idx = self._oci[builtin]
        else:
            idx = len(self._oc)
            self._oc.append(bo.opcode(self.b, builtin))
            self._oci[builtin] = idx
        self.ops.append(bo.operator(self.b, idx, ins, outs, opt_type, opt))

    def finish(self, ins, outs, desc):
        return bo.finish(self.b, self.buffers, self._oc, self.tensors, self.ops,
                         ins, outs, desc)


def _q(m, s, zp=0):
    return bo.quant_tensor(m.b, [s], [zp])


def _opt(m, kind, keep_dims=None):
    """Build the BuiltinOptions table for `kind` (must precede the Operator table)."""
    S, b = bo.S, m.b
    if kind in ("MulOptions", "AddOptions"):
        getattr(S, kind + "Start")(b)
        # NONE: ncc would otherwise fold a ReLU into the elementwise layer
        getattr(S, kind + "AddFusedActivationFunction")(b, 0)
        return getattr(BOPT, kind), getattr(S, kind + "End")(b)
    if kind == "ReducerOptions":
        S.ReducerOptionsStart(b)
        S.ReducerOptionsAddKeepDims(b, bool(keep_dims))
        return BOPT.ReducerOptions, S.ReducerOptionsEnd(b)
    raise SystemExit(f"unknown options {kind}")


# --------------------------------------------------------------- golden helpers
def _fp16(x):
    return np.asarray(x, np.float32).astype(np.float16)


def _rha(v):
    """round-half-away-from-zero, i.e. TFLite's MultiplyByQuantizedMultiplier rule."""
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def _clip8(v):
    return np.clip(v, -128, 127)


def _gamma(rng, n, dtype):
    """RMSNorm gains are ~1: N(1, 0.1).  The golden uses the *stored* values."""
    g = rng.normal(1.0, 0.1, size=(n,))
    if dtype == "int8":
        codes = _clip8(_rha(g / Q["s_g"])).astype(np.int8)
        codes = np.where(codes == 0, np.int8(64), codes).astype(np.int8)  # never a dead gain
        return codes, codes.astype(np.float64) * Q["s_g"]
    g16 = _fp16(g)
    return g16, g16.astype(np.float64)


# --------------------------------------------------------------- golden
def golden(cfg, rng):
    """Return dict with stored input, expected output, quant meta, tensors to write."""
    kind, dtype, shape = cfg["kind"], cfg["dtype"], cfg["shape"]
    T, H = shape

    if kind == "rsqrt":
        if dtype == "fp16":
            x = _fp16(rng.uniform(cfg["in_lo"], cfg["in_hi"], size=shape))
            x.flat[0], x.flat[1], x.flat[2] = np.float16(cfg["in_lo"]), np.float16(1.0), \
                np.float16(cfg["in_hi"])
            x32 = x.astype(np.float32)
            y = _fp16(np.float32(1.0) / np.sqrt(x32))            # float32 golden
            y64 = (1.0 / np.sqrt(x.astype(np.float64))).astype(np.float16)
            return dict(x=x, y=y, qm={}, in_t=F16, out_t=F16,
                        extra={"golden_dtype": "float32",
                               "golden_f32_vs_f64_max_abs": float(np.max(np.abs(
                                   y.astype(np.float64) - y64.astype(np.float64))))})
        s_in, s_out = 1 / 16.0, 1 / 64.0
        if dtype == "fp32":
            x = rng.uniform(cfg["in_lo"], cfg["in_hi"], size=shape).astype(np.float32)
            x.flat[0], x.flat[1], x.flat[2] = cfg["in_lo"], 1.0, cfg["in_hi"]
            y = (1.0 / np.sqrt(x.astype(np.float64))).astype(np.float32)
            return dict(x=x, y=y, qm={}, in_t=F32T, out_t=F32T,
                        extra={"golden_dtype": "float64_then_float32"})
        codes = rng.integers(int(round(cfg["in_lo"] / s_in)),
                            int(round(cfg["in_hi"] / s_in)) + 1,
                            size=shape).astype(np.int8)
        codes.flat[0], codes.flat[1], codes.flat[2] = 8, 16, 127        # 0.5 / 1.0 / 7.94
        real = codes.astype(np.float64) * s_in
        y = _clip8(_rha((1.0 / np.sqrt(real)) / s_out)).astype(np.int8)
        qm = {"input": s_in, "output": s_out, "input_zp": 0, "output_zp": 0}
        return dict(x=codes, y=y, qm=qm, in_t=I8, out_t=I8,
                    extra={"golden_dtype": "float64_exact_quantised",
                           "quant_rule": "y = clip(round(rsqrt(q*s_in)/s_out))"})

    if kind != "rmsnorm":
        raise SystemExit(f"no golden for {kind}")

    eps = float(cfg["eps"])

    if dtype == "fp16":
        x = _fp16(rng.uniform(-3.0, 3.0, size=shape))
        # pin a few magnitudes so the row is never accidentally degenerate
        x[0, 0], x[0, 1], x[0, 2] = np.float16(-3.0), np.float16(0.0), np.float16(3.0)
        g_store, g_ref = _gamma(rng, H, dtype)
        eps_store = float(np.float16(eps))          # the constant as it is stored in the model
        x32 = x.astype(np.float32)
        sq = x32 * x32
        mean = sq.mean(axis=1, keepdims=True, dtype=np.float32)
        m2 = mean + np.float32(eps_store)
        r = np.float32(1.0) / np.sqrt(m2)
        y = _fp16(x32 * r * g_ref.astype(np.float32))
        # second, independent golden in float64 (reported, not the pass criterion)
        xd = x.astype(np.float64)
        y64 = (xd * (1.0 / np.sqrt((xd * xd).mean(axis=1, keepdims=True) + eps_store))
               * g_ref).astype(np.float16)
        return dict(x=x, y=y, qm={}, in_t=F16, out_t=F16,
                    extra={"eps": eps, "eps_stored": eps_store,
                           "gamma": [float(v) for v in np.asarray(g_store).ravel()],
                           "golden_dtype": "float32",
                           "golden_f32_vs_f64_max_abs": float(np.max(np.abs(
                               y.astype(np.float64) - y64.astype(np.float64)))),
                           "graph": ["MUL(x,x)", "MEAN(axis=-1,keep_dims)",
                                     "ADD(eps)", "RSQRT", "MUL(x)", "MUL(gamma)"]})

    if dtype == "int8":
        # |code| <= 32 keeps x*x (mult 1/32) inside int8 and mean(x^2) ~ 1.4 -> r ~ 0.85
        codes = rng.integers(-32, 33, size=shape).astype(np.int8)
        codes[0, :8] = np.array([32, -32, 30, -30, 8, -8, 0, 16], np.int8)
        g_codes, g_ref = _gamma(rng, H, dtype)
        eps_code = int(_clip8(_rha(eps / Q["s_m2"])))
        qx = codes.astype(np.int64)
        q_sq = _clip8(_rha(qx.astype(np.float64) * qx.astype(np.float64)
                           * (Q["s_x"] ** 2 / Q["s_sq"])))
        q_m = _clip8(_rha(q_sq.sum(axis=1, keepdims=True) / float(H)))
        q_m2 = _clip8(q_m + eps_code)
        real_m2 = q_m2.astype(np.float64) * Q["s_m2"]
        q_r = _clip8(_rha((1.0 / np.sqrt(real_m2)) / Q["s_r"]))
        q_y1 = _clip8(_rha(qx.astype(np.float64) * q_r * (Q["s_x"] * Q["s_r"] / Q["s_y1"])))
        q_y = _clip8(_rha(q_y1 * g_codes.astype(np.float64)
                          * (Q["s_y1"] * Q["s_g"] / Q["s_y"]))).astype(np.int8)
        # ideal float32 RMSNorm on the same quantised input, for a quantisation-vs-device split
        xr = codes.astype(np.float32) * np.float32(Q["s_x"])
        ideal = xr * (np.float32(1.0) / np.sqrt((xr * xr).mean(axis=1, keepdims=True,
                                                             dtype=np.float32)
                                                + np.float32(eps))) * g_ref.astype(np.float32)
        return dict(x=codes, y=q_y, qm={"input": Q["s_x"], "output": Q["s_y"],
                                       "input_zp": 0, "output_zp": 0},
                    in_t=I8, out_t=I8,
                    extra={"gamma_codes": [int(v) for v in np.asarray(g_codes).ravel()],
                           "eps_code": eps_code,
                           "quant_scales": {k: float(v) for k, v in Q.items()},
                           "ideal_float_output": [float(v) for v in ideal.ravel()],
                           "golden_dtype": "int64_exact_quantised",
                           "quant_rule": (
                               "per stage: round-half-away-from-zero on an exact power-of-two "
                               "multiplier (x*x /32, MEAN /64, rsqrt table, x*r /16, y*g /64)"),
                           "graph": ["MUL(x,x)", "MEAN(axis=-1,keep_dims)",
                                     "ADD(eps_code)", "RSQRT", "MUL(x)", "MUL(gamma)"]})

    if dtype == "fp32":
        x = rng.uniform(-3.0, 3.0, size=shape).astype(np.float32)
        g_store, g_ref = _gamma(rng, H, "fp16")
        eps_store = eps
        xd = x.astype(np.float64)
        y32 = (x.astype(np.float64)
               * (1.0 / np.sqrt((xd * xd).mean(axis=1, keepdims=True) + eps_store))
               * g_ref).astype(np.float32)
        return dict(x=x, y=y32, qm={}, in_t=F32T, out_t=F32T,
                    extra={"eps": eps, "eps_stored": eps_store,
                           "gamma": [float(v) for v in np.asarray(g_store).ravel()],
                           "golden_dtype": "float64_then_float32",
                           "graph": ["MUL(x,x)", "MEAN(axis=-1,keep_dims)",
                                     "ADD(eps)", "RSQRT", "MUL(x)", "MUL(gamma)"]})

    raise SystemExit(f"no golden for {dtype}")


# --------------------------------------------------------------- model builders
def build_rsqrt(cfg, g):
    """Single-op RSQRT: the primitive probe (question 1)."""
    m = M()
    int8 = cfg["dtype"] == "int8"
    qin = _q(m, g["qm"]["input"]) if int8 else None
    qout = _q(m, g["qm"]["output"]) if int8 else None
    t_in = m.act("input", cfg["shape"], g["in_t"], qin)
    t_out = m.out("output", cfg["shape"], g["out_t"], qout)
    # TFLite RSQRT has no BuiltinOptions table at all -> opt_type stays NONE
    m.op(BO.RSQRT, [t_in], [t_out])
    blob = m.finish([t_in], [t_out],
                    f"rsqrt {'x'.join(map(str, cfg['shape']))} {cfg['dtype']}")
    meta = {"op": "RSQRT", "shape": dict(d0=cfg["shape"][0], d1=cfg["shape"][1])}
    return blob, meta, [t_in], t_out


def build_rmsnorm(cfg, g):
    """MEAN(last axis) -> ADD eps -> RSQRT -> MUL x -> MUL gamma -> MUL gamma.

    Two deliberate structural choices, both flagged here because they are *graph*
    semantics and not compiler appeasement (AGENTS.md R2):
      * the eps constant has the full [T,1] shape instead of a scalar, so the ADD needs no
        broadcasting (only the two MULs do, and those are irreducible: the RMSNorm scalar
        must multiply a whole row);
      * MUL(x, x) reuses the *same tensor* as both operands (TFLite allows it); if the
        importer rejects that, the documented fallback is a second graph input fed with a
        copy of the data (tools/e2e.sh already pushes golden/input1.bin when present) or the
        SQUARE op.  See docs/OP-rmsnorm.md §failures.
    """
    T, H = cfg["shape"]
    int8 = cfg["dtype"] == "int8"
    eps_code = float(g["extra"]["eps_code"]) if int8 else 0.0
    m = M()

    if int8:
        q_in = _q(m, Q["s_x"]); q_sq = _q(m, Q["s_sq"]); q_m = _q(m, Q["s_m"])
        q_m2 = _q(m, Q["s_m2"]); q_r = _q(m, Q["s_r"]); q_y1 = _q(m, Q["s_y1"])
        q_g = _q(m, Q["s_g"]); q_y = _q(m, Q["s_y"])
        eps_arr = np.full((T, 1), eps_code, np.int8)
        g_arr = np.asarray(g["extra"]["gamma_codes"], np.int8).reshape(H)
    else:
        q_in = q_sq = q_m = q_m2 = q_r = q_y1 = q_g = q_y = None
        eps_arr = np.full((T, 1), g["extra"]["eps_stored"], np.float16)
        g_arr = np.asarray(g["extra"]["gamma"], np.float16).reshape(H)

    tt = g["in_t"]
    t_in = m.act("input", [T, H], tt, q_in)
    t_sq = m.act("x_sq", [T, H], tt, q_sq)
    t_axis = m.const("mean_axis", [1], I32, np.asarray([1], np.int32))
    t_mean = m.act("mean", [T, 1], tt, q_m)
    t_eps = m.const("eps", [T, 1], g["out_t"], eps_arr, q_m2)
    t_m2 = m.act("mean_pe", [T, 1], tt, q_m2)
    t_r = m.act("rsqrt", [T, 1], tt, q_r)
    t_y1 = m.act("x_norm", [T, H], tt, q_y1)
    t_g = m.const("gamma", [H], g["out_t"], g_arr, q_g)
    t_out = m.out("output", [T, H], g["out_t"], q_y)

    b = m.b
    ot, o = _opt(m, "MulOptions");  m.op(BO.MUL, [t_in, t_in], [t_sq], ot, o)   # x*x
    ot, o = _opt(m, "ReducerOptions", keep_dims=True)
    m.op(BO.MEAN, [t_sq, t_axis], [t_mean], ot, o)
    ot, o = _opt(m, "AddOptions");  m.op(BO.ADD, [t_mean, t_eps], [t_m2], ot, o)
    m.op(BO.RSQRT, [t_m2], [t_r])
    ot, o = _opt(m, "MulOptions");  m.op(BO.MUL, [t_in, t_r], [t_y1], ot, o)     # x * r
    ot, o = _opt(m, "MulOptions");  m.op(BO.MUL, [t_y1, t_g], [t_out], ot, o)    # * gamma

    blob = m.finish([t_in], [t_out],
                    f"rmsnorm {'x'.join(map(str, cfg['shape']))} {cfg['dtype']}")
    meta = {"op": "RMSNORM",
            "shape": {f"d{i}": int(d) for i, d in enumerate(cfg["shape"])},
            "op_shape": [int(d) for d in cfg["shape"]]}
    return blob, meta, [t_in], t_out


# ---------------------------------------------------------------------- main
def emit(out_dir, blob, meta, x, y, g, tflite_name="model.tflite"):
    os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
    path = os.path.join(out_dir, tflite_name)
    with open(path, "wb") as f:
        f.write(blob)
    x.tofile(os.path.join(out_dir, "golden", "input.bin"))
    y.tofile(os.path.join(out_dir, "golden", "expected.bin"))
    meta["tflite"] = tflite_name
    meta["tflite_bytes"] = len(blob)
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", help="model.tflite path (its dir gets model.json + golden/)")
    args = ap.parse_args()

    if args.list:
        for k in SPECS:
            print(k)
        return
    if args.spec not in SPECS:
        raise SystemExit(f"unknown spec {args.spec}; use --list")

    cfg = SPECS[args.spec]
    rng = np.random.default_rng(SEED)
    g = golden(cfg, rng)
    blob, meta, ins, out_idx = (build_rsqrt(cfg, g) if cfg["kind"] == "rsqrt"
                                else build_rmsnorm(cfg, g))

    out = args.out or f"ops/{args.spec}/model.tflite"
    d = os.path.dirname(os.path.abspath(out))
    meta.update({
        "dtype": cfg["dtype"],
        "input": {"name": "input", "shape": [int(v) for v in cfg["shape"]],
                  "type": TYPE_NAME[g["in_t"]], "bytes": int(np.asarray(g["x"]).nbytes)},
        "output": {"name": "output", "shape": [int(v) for v in np.asarray(g["y"]).shape],
                   "type": TYPE_NAME[g["out_t"]], "bytes": int(np.asarray(g["y"]).nbytes)},
        "quant": g["qm"],
        "seed": SEED,
        "prediction": cfg["prediction"],
        "generator": "tools/build_rmsnorm.py",
        "spec": args.spec,
        "expect_compile_fail": bool(cfg.get("expect_compile_fail", False)),
    })
    meta.update({k: v for k, v in g["extra"].items() if k != "ideal_float_output"})
    if cfg["dtype"] == "int8":
        meta["int8_tol"] = int(cfg["int8_tol"])
        meta["reference"] = "int64_exact_quantised"
        meta["tolerance_rationale"] = (
            "chained int8 graph, registered before running: 4 rounding/requant stages "
            "(x*x /32, MEAN /64, RSQRT table, x*r /16, y*gamma /64), each <=0.5 code of the "
            "stage scale, plus the +-1 code of the rsqrt table amplified by the MUL-by-x "
            "(<=2 codes for |code(x)|<=32).  Budget: 1+1+1+2 = 4 codes of s_y=1/32 "
            "(0.125 absolute on a signal with |y|<=2).  NOTE the ideal-float output is also "
            "stored in golden/ for a quantisation-vs-device split, but the PASS criterion is "
            "the quantised golden above.")
    elif cfg["dtype"] == "fp16":
        meta["atol"] = float(cfg["atol"])
        meta["rtol"] = float(cfg["rtol"])
        meta["reference"] = "float32"
        meta["tolerance_rationale"] = (
            "registered before the run, per the branch brief: fp16 atol 1e-2 with rtol 1e-2 "
            "on a signal with |y| <= ~3 (>= 8 fp16 ULP at 1.0 = 4.9e-4; the budget covers a "
            "chained 6-op graph with a division-like rsqrt whose input is an accumulated "
            "mean).  The actual max_abs is reported alongside so the contract can be seen to "
            "be coarse rather than retro-fitted.")
    else:
        meta["atol"] = float(cfg["atol"])
        meta["rtol"] = float(cfg["rtol"])
        meta["reference"] = "float64_then_float32"
        meta["tolerance_rationale"] = (
            "fp32 registered at 1e-5/1e-4; only meaningful if an fp32 graph compiles without "
            "a CPU bridge, which is the open question for this dtype.")

    emit(d, blob, meta, np.asarray(g["x"]), np.asarray(g["y"]), g)
    print(f"{out}  {len(blob)} B  in{list(np.asarray(g['x']).shape)} "
          f"-> out{list(np.asarray(g['y']).shape)}  dtype={cfg['dtype']}  kind={cfg['kind']}")


if __name__ == "__main__":
    main()
