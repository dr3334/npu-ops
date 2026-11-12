#!/usr/bin/env python3
"""SwiGLU activation path for the Qwen2.5 MLP:  silu(x) = x*sigmoid(x),  y = silu(gate)*up.

Scope (branch feat/silu, worktree wt-silu).  Four model families, all built from the
TFLite spec with the flatbuffers helpers of tools/build_ops.py (nothing is built by
calling ncc or the device):

  A "silu"   : x                -> LOGISTIC(x)=t ; MUL(x,t)                    -> y  [1 model input]
  B "gated"  : gate, up         -> LOGISTIC(gate)=t ; MUL(gate,t)=s ; MUL(s,up) -> y  [2 model inputs]
  C "fused"  : x[T,128]         -> FC(x,Wg)=gate ; FC(x,Wu)=up ; then B's three ops     [const weights]
  D "gelu"   : x                -> GELU (native TFLite builtin; this schema has GELU=150, no SILU)

A/B/C answer "what does the composition cost, and does ncc fuse any of it"; D answers
"is there a native fused activation we should prefer" -- it is *tested*, never assumed:
ncc-tflite may reject the builtin outright.

Method rules (AGENTS.md):
  R3  the golden is computed HERE, in float64 / int64, from the **stored** input values
      (fp16 values as stored; int8 codes dequantised with the declared scale).  It never
      passes through ncc, the TFLite interpreter or the device runtime.
  R1  the numeric contract is registered in SPECS *before* any compile or board run and
      is copied verbatim into model.json (tools/compare.py reads it from there).
  R7  compile / device-run / numerics are separate conclusions; this script only makes
      artifacts, tools/compare.py gives the numeric verdict.
  R9  the comments below record *why* each quantisation constant was chosen; they were
      written before the board run and must survive refactoring.

int8 quantisation contracts -- every requant multiplier is an exact power of two, so the
only rounding left is a shift and the golden can predict the result in codes:

  A silu     x      : scale 1/16 , zp 0     (|x| <= 8)
             sigmoid: scale 1/256, zp -128  (TFLite LOGISTIC spec)
             out    : scale 1/16 , zp 0     -> mult (1/16 * 1/256)/(1/16) = 2^-8
  B gated    gate   : scale 1/16 , zp 0     (|gate| <= 8)
             up     : scale 1/128, zp 0     (|up| <= ~1)
             sigmoid: scale 1/256, zp -128
             s      : scale 1/16 , zp 0     -> mult 2^-8
             out    : scale 1/8  , zp 0     -> mult (1/16 * 1/128)/(1/8) = 2^-8
  C fused    x      : scale 1/16 , zp 0 ; weights 1/127, zp 0 ; bias int32 zeros
             gate/up: scale 1/16 , zp 0     -> FC mult (1/16*1/127)/(1/16) = 1/127
                                               (NOT a power of two: the known FC requant
                                               rule of OP-MATRIX §3 applies)
             sigmoid: 1/256, zp -128 ; s: 1/16 -> mult 2^-8
             out    : scale 1/2  , zp 0     -> mult (1/16 * 1/16)/(1/2) = 2^-7
                                               (a product of two +-8 tensors needs a
                                                wider output range -- that is exactly
                                                why int8 is unattractive here)

Three reference arrays per model (the device dump is compared against all three by
--analyze, so we can say *which* rounding model the hardware implements):
  golden/expected.bin       exact math on the stored inputs, ONE final rounding to the
                            storage type  -> this is what tools/compare.py uses
  golden/expected_alt.bin   hypothesis 1: the graph as TFLite reference kernels would run
                            it (fp16: every intermediate tensor rounded to fp16; int8:
                            two-step rounding through a 31-bit multiplier + shift)
  golden/expected_alt2.bin  hypothesis 2 (int8, pow2 muls only): the same graph if the
                            backend implements a power-of-two multiplier as a *single*
                            RoundingDivideByPOT shift instead of high-mul+shift

Usage (WSL python that has the `tflite` flatbuffer package):
    /usr/bin/python3 tools/build_silu.py --list
    /usr/bin/python3 tools/build_silu.py silu_1x696_fp16 --out ops/silu_1x696_fp16
    /usr/bin/python3 tools/build_silu.py --analyze ops/silu_1x696_fp16
"""
import argparse
import json
import math
import os

import flatbuffers
import numpy as np

import build_ops as bo

SEED = 20261112                      # same deterministic seed family as the rest of the repo

F16 = bo.F16
I8 = bo.I8
I32 = bo.I32

BO = bo.S.BuiltinOperator
BOPT = bo.S.BuiltinOptions
TYPE_NAME = {F16: "float16", I8: "int8", I32: "int32"}

# ---- int8 quantisation constants (header explains why) ------------------------------
S_X = 1.0 / 16.0                     # model-input activation scale (|x| <= 8)
S_UP_GATED = 1.0 / 128.0             # "up" scale in the two-input gated model
S_SIG = 1.0 / 256.0                  # TFLite LOGISTIC output scale
S_SIG_ZP = -128
S_OUT_SILU = 1.0 / 16.0              # silu: same scale as x (silu(x) <= x for x > 0)
S_OUT_GATED = 1.0 / 8.0              # gated: covers |silu(gate)*up| <= 8
S_W = 1.0 / 127.0                    # FC weight scale
S_OUT_FUSED = 1.0 / 2.0              # fused: product of two +-8 tensors
K_FUSED = 128                        # hidden size of the LLM residual stream

# Physical input ranges.  Both dtypes use the SAME ranges so fp16 and int8 results are
# comparable: int8 pins the range (1/16 * 127 ~= 8).
X_LO, X_HI = -8.0, 8.0
UP_LO, UP_HI = -1.0, 1.0

# Values pinned into the first elements of each sample: both tails, the SiLU minimum
# (-1.2783 -> -0.2785), an exact zero, a unit value.  Random sampling does not reliably
# hit the minimum or the clamp corners (AGENTS.md §3.3: find the edge, not one sample).
X_PINS = [-8.0, -3.0, -1.2783, 0.0, 1.0, 8.0]
UP_PINS = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.25]

# ------------------------------------------------------------------------ spec table
# Registered BEFORE any compile or board run (R1).  Rationale:
#  * fp16: atol 2e-3 / rtol 1e-2 -- the repo's registered fp16 contract (OP-actreduce
#    §1.3).  A composed graph rounds intermediate tensors to fp16, so the honest
#    tolerance is "a few fp16 ULP at the output magnitude": rtol covers large |y|,
#    atol covers small ones.  Exactness is still reported separately.
#  * int8: tolerance in *codes* of the output tensor, bounded analytically before the run:
#      (a) sigmoid is quantised to 1/256 -> |dsig| <= 1/512; through x*sig that is
#          |x|/512 <= 8/512 = 0.0156 real = 0.25 code at out scale 1/16;
#      (b) each requant is a power-of-two shift: <= 0.5 code of rounding;
#      (c) the exact-math golden itself rounds once: 0.5 code.
#    A silu  : 0.25 + 0.5 + 0.5 = 1.25                    -> int8_tol 2
#    B gated : 0.25 + 2*0.5 (two muls) + 0.5 = 1.75        -> int8_tol 2
#    C fused : B + FC requant with the non-pow2 1/127 multiplier (>=1 code,
#              OP-MATRIX §3) = 2.75                       -> int8_tol 3
SPECS = {
    # ---- A: silu(x) = x*sigmoid(x), single input -----------------------------------
    "silu_1x696_fp16": dict(kind="silu", T=1, W=696, dtype="fp16",
                            atol=2e-3, rtol=1e-2,
                            prediction="LOGISTIC+MUL -> 2 MDLA layers; fp16 within a few ULP"),
    "silu_8x696_fp16": dict(kind="silu", T=8, W=696, dtype="fp16",
                            atol=2e-3, rtol=1e-2,
                            prediction="same graph, real batch; 696*2 = 1392 B rows are 16-aligned"),
    "silu_1x696_int8": dict(kind="silu", T=1, W=696, dtype="int8", int8_tol=2,
                            prediction="696 % 16 == 8 -> expected COMPILE FAIL (16-byte row rule)"),
    "silu_8x696_int8": dict(kind="silu", T=8, W=696, dtype="int8", int8_tol=2,
                            prediction="same row rule -> expected COMPILE FAIL"),
    "silu_1x704_fp16": dict(kind="silu", T=1, W=704, dtype="fp16", atol=2e-3, rtol=1e-2,
                            prediction="control for the 704 column"),
    "silu_1x704_int8": dict(kind="silu", T=1, W=704, dtype="int8", int8_tol=2,
                            prediction="704 % 16 == 0 -> the padded-width workaround should compile"),
    # ---- B: gated epilogue, gate and up as two model inputs -------------------------
    "gated_1x696_fp16": dict(kind="gated", T=1, W=696, dtype="fp16",
                             atol=2e-3, rtol=1e-2,
                             prediction="3 MDLA layers (sigmoid + 2 mul); no fusion expected"),
    "gated_8x696_fp16": dict(kind="gated", T=8, W=696, dtype="fp16",
                             atol=2e-3, rtol=1e-2,
                             prediction="same, batch 8"),
    "gated_1x696_int8": dict(kind="gated", T=1, W=696, dtype="int8", int8_tol=2,
                             prediction="expected COMPILE FAIL on the 696-wide model inputs"),
    "gated_1x704_int8": dict(kind="gated", T=1, W=704, dtype="int8", int8_tol=2,
                             prediction="704-wide two-input epilogue; 3 layers, exact shifts"),
    # ---- C: fused, const GATE_PROJ/UP_PROJ weights, single input [T,128] ------------
    "fused_1x128x696_fp16": dict(kind="fused", T=1, W=696, dtype="fp16", K=K_FUSED,
                                 atol=2e-3, rtol=1e-2,
                                 prediction="5 layers (2 FC + sigmoid + 2 mul); FC followed by "
                                            "LOGISTIC is probably NOT fused"),
    "fused_8x128x696_fp16": dict(kind="fused", T=8, W=696, dtype="fp16", K=K_FUSED,
                                 atol=2e-3, rtol=1e-2,
                                 prediction="same graph, batch 8"),
    "fused_1x128x696_int8": dict(kind="fused", T=1, W=696, dtype="int8", K=K_FUSED, int8_tol=3,
                                 prediction="the only DRAM boundary tensor is [1,128] (16-aligned), "
                                            "so int8 may compile even though 696-wide tensors are internal"),
    "fused_8x128x696_int8": dict(kind="fused", T=8, W=696, dtype="int8", K=K_FUSED, int8_tol=3,
                                 prediction="same, batch 8"),
    # ---- D: native GELU (schema has GELU=150; there is no SILU builtin at all) ------
    "gelu_1x696_fp16": dict(kind="gelu", T=1, W=696, dtype="fp16", atol=2e-3, rtol=1e-2,
                            prediction="ncc may reject the GELU builtin; tested, not assumed"),
    # same op with the tanh approximation the compiler's error message points at
    # ("Not support ERF approximation, please set the 'approximate' parameter to true").
    # The golden stays the *exact* erf GELU, so passing atol=2e-3 also measures how far
    # the tanh approximation is from the exact function; expected_alt holds the tanh
    # approximation so --analyze can say which one the device ran.
    "gelu_1x696_fp16_tanh": dict(kind="gelu", T=1, W=696, dtype="fp16", approximate=True,
                                 atol=2e-3, rtol=1e-2,
                                 prediction="accepted only in the tanh-approximated form, "
                                            "if at all; numeric diff vs exact erf is the "
                                            "approximation error, not a kernel error"),
}


# ------------------------------------------------------------------ math (golden side)
def sigmoid(x):
    """float64 sigmoid, written out here so no shared library code can leak into it."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def fp16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16)


_erf = np.vectorize(math.erf)


def gelu_exact(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def gelu_tanh(x):
    """The tanh approximation of GELU (what TFLite's GeluOptions:approximate=true means)."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _round_half_away(x):
    """numpy round() is half-to-even; int8 quantisation is half-away-from-zero.
    Keeping the two apart on purpose (this is where silent 1-code errors come from)."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def enc(v, scale, zp=0):
    """Real value -> stored int8 code, clipped to the int8 range."""
    return np.clip(_round_half_away(np.asarray(v, dtype=np.float64) / scale) + zp,
                   -128, 127).astype(np.int8)


def dec(q, scale, zp=0):
    return (np.asarray(q, dtype=np.float64) - zp) * scale


def rdtp(x, exponent):
    """TFLite RoundingDivideByPOT (round half away from zero)."""
    x = np.asarray(x, dtype=np.int64)
    if exponent == 0:
        return x.copy()
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + np.where(x < 0, 1, 0)
    return (x >> exponent) + (remainder > threshold).astype(np.int64)


def qmult(mult):
    """TFLite QuantizeMultiplier -> (M0, shift) with mult ~= M0 * 2^-31 * 2^shift."""
    m, shift = math.frexp(mult)
    q = int(round(m * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        shift += 1
    return q, shift


def requant(acc, mult, mode="tflite"):
    """acc (int64 codes) * mult, in one of the two hypotheses the repo cares about."""
    x = np.asarray(acc, dtype=np.int64)
    M0, shift = qmult(mult)
    if mode == "shift":                      # backend turns a pow2 multiplier into one shift
        return rdtp(x, -shift)
    ab = x * M0                              # SaturatingRoundingDoublingHighMul + shift
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    return rdtp((ab + nudge) >> 31, -shift)


def clip8(x):
    return np.clip(x, -128, 127).astype(np.int8)


# ------------------------------------------------------------------------- data sets
def _pin(arr, values):
    """Overwrite the first len(values) flat elements with `values`."""
    f = arr.reshape(-1)
    for i, v in enumerate(values[: f.size]):
        f[i] = v
    return arr


def sample(kind, dtype, T, W, K, rng):
    """Stored fp16 arrays or int8 codes for one model."""
    d = {}
    if kind in ("silu", "gated", "gelu"):
        if dtype == "fp16":
            d["x"] = _pin(fp16(rng.uniform(X_LO, X_HI, size=(T, W))), X_PINS)
        else:
            q = rng.integers(-128, 128, size=(T, W)).astype(np.int8)
            d["x"] = _pin(q, [enc(v, S_X) for v in X_PINS])
        if kind == "gated":
            if dtype == "fp16":
                d["up"] = _pin(fp16(rng.uniform(UP_LO, UP_HI, size=(T, W))), UP_PINS)
            else:
                q = rng.integers(-128, 128, size=(T, W)).astype(np.int8)
                d["up"] = _pin(q, [enc(v, S_UP_GATED) for v in UP_PINS])
        return d

    # kind == "fused": weights are constants, drawn in the real range that matches the
    # activation contract.  fp16: +-1 like a real gate_proj.  int8: codes +-6 so that the
    # FC output lands inside the declared +-8 gate range -- i.e. the weights a calibrator
    # would have to produce for the int8 variant to be usable at all.
    if dtype == "fp16":
        d["x"] = _pin(fp16(rng.uniform(-1.0, 1.0, size=(T, K))), [-1.0, 0.0, 1.0])
        d["wg"] = fp16(rng.uniform(-1.0, 1.0, size=(W, K)))
        d["wu"] = fp16(rng.uniform(-1.0, 1.0, size=(W, K)))
        d["bg"] = np.zeros(W, dtype=np.float16)
        d["bu"] = np.zeros(W, dtype=np.float16)
    else:
        q = rng.integers(-128, 128, size=(T, K)).astype(np.int8)
        d["x"] = _pin(q, [enc(v, S_X) for v in (-8.0, 0.0, 8.0)])
        d["wg"] = rng.integers(-6, 7, size=(W, K)).astype(np.int8)
        d["wu"] = rng.integers(-6, 7, size=(W, K)).astype(np.int8)
        d["bg"] = np.zeros(W, dtype=np.int32)
        d["bu"] = np.zeros(W, dtype=np.int32)
    return d


# --------------------------------------------------------------- golden computation
def compute(spec, d):
    """Return (x_in, exact, alt, alt2, quant_meta).

    exact : float64 exact math on the stored inputs + one final rounding  (compare.py)
    alt   : TFLite-reference-kernel model (two-step rounding)               (hypothesis 1)
    alt2  : pow2-multiplier-as-plain-shift model, int8 pow2 graphs only     (hypothesis 2)
    """
    kind, dtype = spec["kind"], spec["dtype"]
    qm = {}

    # ------------------------------------------------------------------ fp16 branch
    if dtype == "fp16":
        if kind == "silu":
            x = d["x"].astype(np.float64)
            s = sigmoid(x)
            exact = fp16(x * s)
            alt = fp16(x * fp16(s))                  # sigmoid tensor stored as fp16 first
            return x, exact, alt, alt, qm
        if kind == "gated":
            x = d["x"].astype(np.float64)
            u = d["up"].astype(np.float64)
            s = sigmoid(x)
            exact = fp16(x * s * u)
            alt = fp16(fp16(x * fp16(s)) * u)        # s stored fp16, product stored fp16
            return x, exact, alt, alt, qm
        if kind == "fused":
            x = d["x"].astype(np.float64)
            gate = x @ d["wg"].astype(np.float64).T + d["bg"].astype(np.float64)
            up = x @ d["wu"].astype(np.float64).T + d["bu"].astype(np.float64)
            s = sigmoid(gate)
            exact = fp16(gate * s * up)
            g16, u16 = fp16(gate), fp16(up)          # every tensor materialised as fp16
            alt = fp16(fp16(g16 * fp16(sigmoid(g16))) * u16)
            return x, exact, alt, alt, qm
        x = d["x"].astype(np.float64)
        exact = fp16(gelu_exact(x))
        alt = fp16(gelu_tanh(x)) if spec.get("approximate") else exact
        return x, exact, alt, alt, qm

    # ------------------------------------------------------------------ int8 branch
    if kind == "silu":
        qx = d["x"]
        xr = dec(qx, S_X)
        exact = enc(xr * sigmoid(xr), S_OUT_SILU)
        qs = enc(sigmoid(xr), S_SIG, S_SIG_ZP)                 # LOGISTIC = 1/256, zp -128
        acc = qx.astype(np.int64) * (qs.astype(np.int64) + 128)  # (a-azp)*(b-bzp)
        mult = (S_X * S_SIG) / S_OUT_SILU                      # = 2^-8
        qm = {"input": S_X, "input_zp": 0, "sigmoid": S_SIG, "sigmoid_zp": S_SIG_ZP,
              "output": S_OUT_SILU, "output_zp": 0,
              "requant": {"mul": f"{mult:.6g} = 2^-8"}}
        return qx, exact, clip8(requant(acc, mult)), clip8(requant(acc, mult, "shift")), qm

    if kind == "gated":
        qg, qu = d["x"], d["up"]
        gr, ur = dec(qg, S_X), dec(qu, S_UP_GATED)
        exact = enc(gr * sigmoid(gr) * ur, S_OUT_GATED)
        qs = enc(sigmoid(gr), S_SIG, S_SIG_ZP)
        m1 = (S_X * S_SIG) / S_X                               # = 2^-8
        qs_mul = clip8(requant(qg.astype(np.int64) * (qs.astype(np.int64) + 128), m1))
        m2 = (S_X * S_UP_GATED) / S_OUT_GATED                  # = 2^-8
        qs2 = clip8(requant(qg.astype(np.int64) * (qs.astype(np.int64) + 128), m1, "shift"))
        acc = qs_mul.astype(np.int64) * qu.astype(np.int64)
        acc2 = qs2.astype(np.int64) * qu.astype(np.int64)
        qm = {"input": S_X, "input_zp": 0, "up": S_UP_GATED, "up_zp": 0,
              "sigmoid": S_SIG, "sigmoid_zp": S_SIG_ZP, "s": S_X, "s_zp": 0,
              "output": S_OUT_GATED, "output_zp": 0,
              "requant": {"mul1": f"{m1:.6g} = 2^-8", "mul2": f"{m2:.6g} = 2^-8"}}
        return qg, exact, clip8(requant(acc, m2)), clip8(requant(acc2, m2, "shift")), qm

    if kind == "fused":
        qx = d["x"]
        xr = dec(qx, S_X)
        gate_r = xr @ (d["wg"].astype(np.float64) * S_W).T
        up_r = xr @ (d["wu"].astype(np.float64) * S_W).T
        exact = enc(gate_r * sigmoid(gate_r) * up_r, S_OUT_FUSED)
        m_fc = (S_X * S_W) / S_X                               # = 1/127
        acc_g = qx.astype(np.int64) @ d["wg"].astype(np.int64).T + d["bg"].astype(np.int64)
        acc_u = qx.astype(np.int64) @ d["wu"].astype(np.int64).T + d["bu"].astype(np.int64)
        qg = clip8(requant(acc_g, m_fc))
        qu = clip8(requant(acc_u, m_fc))
        qs = enc(sigmoid(dec(qg, S_X)), S_SIG, S_SIG_ZP)
        m1 = (S_X * S_SIG) / S_X                               # = 2^-8
        qs_mul = clip8(requant(qg.astype(np.int64) * (qs.astype(np.int64) + 128), m1))
        m2 = (S_X * S_X) / S_OUT_FUSED                          # = 2^-7
        alt = clip8(requant(qs_mul.astype(np.int64) * qu.astype(np.int64), m2))
        qm = {"input": S_X, "input_zp": 0, "weights": S_W, "weights_zp": 0,
              "gate": S_X, "up": S_X, "sigmoid": S_SIG, "sigmoid_zp": S_SIG_ZP,
              "s": S_X, "s_zp": 0, "output": S_OUT_FUSED, "output_zp": 0,
              "requant": {"fc": "1/127 (two-step rounding, OP-MATRIX §3)",
                          "mul1": "2^-8", "mul2": "2^-7"}}
        return qx, exact, alt, alt, qm

    if kind == "gelu":
        # TFLite has no int8 GELU spec; out scale 1/128 zp 0 fits |gelu| <= 8.
        qx = d["x"]
        xr = dec(qx, S_X)
        exact = enc(gelu_exact(xr), 1.0 / 128.0)
        alt = enc(gelu_tanh(xr), 1.0 / 128.0) if spec.get("approximate") else exact
        qm = {"input": S_X, "input_zp": 0, "output": 1.0 / 128.0, "output_zp": 0,
              "note": "no TFLite int8 GELU spec; scale chosen to fit |gelu| <= 8"}
        return qx, exact, alt, alt, qm

    raise SystemExit(f"no golden for {kind}/{dtype}")


# --------------------------------------------------------------------- flatbuffers
def q8(b, scale, zp=0):
    return bo.quant_tensor(b, [scale], [zp])


def _mul_node(b, oc, ins, outs):
    S = bo.S
    S.MulOptionsStart(b)
    S.MulOptionsAddFusedActivationFunction(b, 0)          # NONE
    opt = S.MulOptionsEnd(b)
    return bo.operator(b, oc, ins, outs, BOPT.MulOptions, opt)


def _fc_node(b, oc, ins, outs):
    S = bo.S
    S.FullyConnectedOptionsStart(b)
    S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)
    S.FullyConnectedOptionsAddWeightsFormat(b, 0)
    S.FullyConnectedOptionsAddKeepNumDims(b, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
    opt = S.FullyConnectedOptionsEnd(b)
    return bo.operator(b, oc, ins, outs, BOPT.FullyConnectedOptions, opt)


def build_blob(spec, d):
    kind, dtype, T, W = spec["kind"], spec["dtype"], spec["T"], spec["W"]
    b = flatbuffers.Builder(1 << 17)
    # buffer 0 must stay empty: a model input carrying a data buffer gets constant-folded
    # by ncc into "IdentityLayer on NVM" (AGENTS.md §7.3).
    buffers = [bo.buf_empty(b)]
    tensors = []
    S = bo.S
    i8 = dtype == "int8"

    def add(name, shape, quant=None, data=None, ttype=None):
        buffers.append(bo.buf_empty(b) if data is None else bo.buf_data(b, data))
        tensors.append(bo.tensor(b, name, shape, ttype or (I8 if i8 else F16),
                                 len(buffers) - 1, quant))
        return len(tensors) - 1

    ops = []                                   # (builtin name, inputs, outputs)
    sig_quant = q8(b, S_SIG, S_SIG_ZP) if i8 else None
    act_q = (lambda s: q8(b, s, 0)) if i8 else (lambda s: None)

    if kind == "silu":
        t_x = add("input", [T, W], act_q(S_X))                       # buffer 0 = empty
        t_sig = add("sigmoid", [T, W], sig_quant)
        t_y = add("output", [T, W], act_q(S_OUT_SILU))
        ops += [("LOGISTIC", [t_x], [t_sig]), ("MUL", [t_x, t_sig], [t_y])]
        mi, mo = [t_x], [t_y]

    elif kind == "gated":
        t_g = add("gate", [T, W], act_q(S_X))                        # buffer 0 = empty
        t_u = add("up", [T, W], act_q(S_UP_GATED))
        t_sig = add("sigmoid", [T, W], sig_quant)
        t_s = add("s", [T, W], act_q(S_X))
        t_y = add("output", [T, W], act_q(S_OUT_GATED))
        ops += [("LOGISTIC", [t_g], [t_sig]),
                ("MUL", [t_g, t_sig], [t_s]),
                ("MUL", [t_s, t_u], [t_y])]
        mi, mo = [t_g, t_u], [t_y]

    elif kind == "fused":
        K = spec["K"]
        t_x = add("input", [T, K], act_q(S_X))                       # buffer 0 = empty
        t_wg = add("weights_gate", [W, K], act_q(S_W), d["wg"])
        t_bg = add("bias_gate", [W], act_q(S_X * S_W) if i8 else None, d["bg"], I32 if i8 else F16)
        t_g = add("gate", [T, W], act_q(S_X))
        t_wu = add("weights_up", [W, K], act_q(S_W), d["wu"])
        t_bu = add("bias_up", [W], act_q(S_X * S_W) if i8 else None, d["bu"], I32 if i8 else F16)
        t_u = add("up", [T, W], act_q(S_X))
        t_sig = add("sigmoid", [T, W], sig_quant)
        t_s = add("s", [T, W], act_q(S_X))
        t_y = add("output", [T, W], act_q(S_OUT_FUSED))
        ops += [("FULLY_CONNECTED", [t_x, t_wg, t_bg], [t_g]),
                ("FULLY_CONNECTED", [t_x, t_wu, t_bu], [t_u]),
                ("LOGISTIC", [t_g], [t_sig]),
                ("MUL", [t_g, t_sig], [t_s]),
                ("MUL", [t_s, t_u], [t_y])]
        mi, mo = [t_x], [t_y]

    elif kind == "gelu":
        t_x = add("input", [T, W], act_q(S_X))
        t_y = add("output", [T, W], act_q(1.0 / 128.0))
        ops += [("GELU", [t_x], [t_y])]
        mi, mo = [t_x], [t_y]

    else:
        raise SystemExit(f"unknown kind {kind}")

    names = []
    for name, _, _ in ops:
        if name not in names:
            names.append(name)
    opcode_objs = [bo.opcode(b, getattr(BO, name)) for name in names]
    nodes = []
    for name, ins, outs in ops:
        oc = names.index(name)
        if name == "MUL":
            nodes.append(_mul_node(b, oc, ins, outs))
        elif name == "FULLY_CONNECTED":
            nodes.append(_fc_node(b, oc, ins, outs))
        elif name == "GELU":
            S.GeluOptionsStart(b)
            S.GeluOptionsAddApproximate(b, bool(spec.get("approximate", False)))
            nodes.append(bo.operator(b, oc, ins, outs, BOPT.GeluOptions, S.GeluOptionsEnd(b)))
        else:                                        # LOGISTIC: no options table
            nodes.append(bo.operator(b, oc, ins, outs))

    desc = (f"{kind} {dtype} " + (f"{T}x{spec['K']}x{W}" if kind == "fused" else f"{T}x{W}"))
    return bo.finish(b, buffers, opcode_objs, tensors, nodes, mi, mo, desc)


# ------------------------------------------------------------------------------ main
GRAPH = {"silu": ["LOGISTIC", "MUL"],
         "gated": ["LOGISTIC", "MUL", "MUL"],
         "fused": ["FULLY_CONNECTED", "FULLY_CONNECTED", "LOGISTIC", "MUL", "MUL"],
         "gelu": ["GELU"]}
OPNAME = {"silu": "SILU(LOGISTIC+MUL)", "gated": "GATED_SILU_MUL",
          "fused": "FUSED_FC_SILU_MUL", "gelu": "GELU"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--analyze", metavar="OPDIR")
    args = ap.parse_args()

    if args.list:
        for k in SPECS:
            print(k)
        return
    if args.analyze:
        return analyze(args.analyze)
    if args.spec not in SPECS:
        raise SystemExit(f"unknown spec {args.spec}; use --list")

    cfg = SPECS[args.spec]
    rng = np.random.default_rng(SEED)
    d = sample(cfg["kind"], cfg["dtype"], cfg["T"], cfg["W"], cfg.get("K"), rng)
    x_in, exact, alt, alt2, qm = compute(cfg, d)
    blob = build_blob(cfg, d)

    out = args.out or f"ops/{args.spec}/model.tflite"
    if os.path.isdir(out):
        out = os.path.join(out, "model.tflite")
    odir = os.path.dirname(os.path.abspath(out))
    os.makedirs(os.path.join(odir, "golden"), exist_ok=True)
    with open(out, "wb") as f:
        f.write(blob)
    x_in.tofile(os.path.join(odir, "golden", "input.bin"))
    if "up" in d:
        d["up"].tofile(os.path.join(odir, "golden", "input1.bin"))
    exact.tofile(os.path.join(odir, "golden", "expected.bin"))
    alt.tofile(os.path.join(odir, "golden", "expected_alt.bin"))
    alt2.tofile(os.path.join(odir, "golden", "expected_alt2.bin"))

    tn = "float16" if cfg["dtype"] == "fp16" else "int8"
    shape = [cfg["T"], cfg["W"]]
    meta = {
        "op": OPNAME[cfg["kind"]],
        "graph": GRAPH[cfg["kind"]],
        "shape": {"T": cfg["T"], "W": cfg["W"], **({"K": cfg["K"]} if "K" in cfg else {})},
        "op_shape": shape,
        "dtype": cfg["dtype"],
        "input": {"name": "input", "shape": list(np.asarray(x_in).shape), "type": tn,
                  "bytes": int(np.asarray(x_in).nbytes)},
        "inputs": [{"name": "gate" if cfg["kind"] == "gated" else "input",
                    "shape": list(np.asarray(x_in).shape), "type": tn}]
                  + ([{"name": "up", "shape": shape, "type": tn}] if "up" in d else []),
        "output": {"name": "output", "shape": shape, "type": tn,
                   "bytes": int(np.asarray(exact).nbytes)},
        "quant": qm,
        "reference": "float64_exact_then_store" if cfg["dtype"] == "fp16"
                     else "float64_exact_then_encode",
        "alt_reference": "unfused_fp16_rounding" if cfg["dtype"] == "fp16"
                         else "tflite_reference_two_step_then_plain_shift",
        "seed": SEED,
        "prediction": cfg["prediction"],
        **({"gelu_approximate": bool(cfg.get("approximate", False))}
           if cfg["kind"] == "gelu" else {}),
        "generator": "tools/build_silu.py",
        "tflite": os.path.basename(out),
        "tflite_bytes": len(blob),
    }
    if cfg["dtype"] == "int8":
        meta["int8_tol"] = int(cfg["int8_tol"])
        meta["tolerance_rationale"] = (
            "codes of the output tensor; bound registered before the run = sigmoid 8-bit "
            "quantisation (<=0.25 code) + one shift rounding per requant (0.5 each) + the "
            "golden's own rounding (0.5); the fused variant adds the FC 1/127 requant "
            "(>=1 code, OP-MATRIX §3).  1 code = out_scale in absolute terms.")
    else:
        meta["atol"] = float(cfg["atol"])
        meta["rtol"] = float(cfg["rtol"])
        meta["tolerance_rationale"] = (
            "fp16: atol is a small multiple of the ULP of the last rounded tensor; the "
            "graph stores intermediate tensors as fp16, so rtol covers large |y|.")
    with open(os.path.join(odir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{out}  {len(blob)} B  in {list(np.asarray(x_in).shape)} -> out {shape} "
          f"dtype={cfg['dtype']}  kind={cfg['kind']}  ops={len(meta['graph'])}")


def analyze(opdir):
    """Device dump vs the three reference models: which rounding model does the device use?"""
    meta = json.load(open(os.path.join(opdir, "model.json")))
    dt = np.float16 if meta["dtype"] == "fp16" else np.int8
    dev = np.fromfile(os.path.join(opdir, "out_device.bin"), dtype=dt)
    shape = tuple(meta["output"]["shape"])
    g = {}
    for tag, fn in (("exact", "expected.bin"), ("alt(TFLite-ref)", "expected_alt.bin"),
                    ("alt2(plain shift)", "expected_alt2.bin")):
        p = os.path.join(opdir, "golden", fn)
        g[tag] = np.fromfile(p, dtype=dt) if os.path.exists(p) else None
    if dev.size != g["exact"].size:
        print(f"SIZE MISMATCH dev={dev.size} expected={g['exact'].size}")
        return 1
    print(f"{opdir}: {meta['op']} {meta['shape']} dtype={meta['dtype']}  n={dev.size}")
    for tag, ref in g.items():
        if ref is None or ref.size != dev.size:
            continue
        if np.issubdtype(dt, np.integer):
            ne = np.abs(ref.astype(np.int64) - dev.astype(np.int64))
            hist = np.bincount(ne, minlength=4)[:4].tolist()
            print(f"  vs {tag:20s} mismatched={int((ne != 0).sum()):5d}/{ne.size}  "
                  f"max|d|={int(ne.max())}  hist(0,1,2,3+)={hist}")
        else:
            d = np.abs(ref.astype(np.float64) - dev.astype(np.float64))
            print(f"  vs {tag:20s} max|d|={d.max():.3e}  rmse={np.sqrt((d ** 2).mean()):.3e}  "
                  f"exact={int((d == 0).sum())}/{d.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
