#!/usr/bin/env python3
"""SwiGLU FFN of the Qwen2.5-0.5B decoder block (ticket 07).

Graph (all fp16, weights = the frozen fixture, layer0.mlp.*):

    x [8,896] --FC(gate_proj)--> gate [8,4864]
              --FC(up_proj)  --> up   [8,4864]
    sig = LOGISTIC(gate)
    s   = MUL(gate, sig)
    y   = MUL(s, up)                      # silu(gate) * up
    out = FC(down_proj)(y) -> [8,896]

Why it is built this way
------------------------
* The three projections are FULLY_CONNECTED with **static** fp16 weights taken from the
  frozen fixture (tools/qwen25_weights.get -> sha256-verified bytes).  TFLite FC is
  out[m,n] = sum_k in[m,k]*W[n,k] + b[n], and the fixture already stores [N,K]
  (gate/up 4864x896, down 896x4864), so the weights go in unchanged (no transpose).
* The Qwen2.5 MLP has **no bias** (Llama-style), but TFLite FC *requires* a third input:
  a 2-input FULLY_CONNECTED fails at model load (NOTES section 5, pitfall 1).  A zero
  constant bias is therefore fed in -- a format requirement, not a numerical one.
* This schema has **no SILU builtin** (docs/OP-silu.md section 7.1), so silu is
  LOGISTIC+MUL and the gate is multiplied by up a second time.  Whether ncc folds that
  epilogue into the preceding FULLY_CONNECTED is answered by the layer list in
  model.compile.log; the compile-only control spec (swiglu_ffn_t8_fp16_nosilu_ctrl,
  the same graph minus the two silu ops) measures the layer-count delta.

Golden (R3): computed here in float64 from the **stored** fp16 input and the stored fp16
weights; it never goes through ncc, the TFLite interpreter or the device.
  golden/expected.bin      exact float64 chain, ONE final rounding to fp16  (compare.py)
  golden/expected_alt.bin  every tensor boundary materialised as fp16 (TFLite reference
                           kernel semantics).  max|alt-exact| is a device-independent
                           lower bound for any implementation that stores intermediate
                           tensors in fp16 -- the tolerance policy leans on that number.

Tolerance registration (at build time, BEFORE any compile or board run; copied into
model.json, which tools/compare.py reads).  Policy, fixed before the first measurement:

    m = max|expected| (exact golden)
    m <= 1    -> atol=2e-3, rtol=1e-2   (the repo's registered fp16 contract; at this
                                         magnitude it is looser than 4 fp16 ULP, so no
                                         NOTES section 10.1 exception is needed)
    m >  1    -> atol=rtol=4*2^-11*m    (NOTES section 10.1 exception; the report also
                                         gives max_abs/max|exp| and rel_l2)

Usage (WSL python that has the tflite flatbuffer package):
    /usr/bin/python3 tools/build_swiglu_ffn.py --list
    /usr/bin/python3 tools/build_swiglu_ffn.py --dryrun          # stats, writes nothing
    /usr/bin/python3 tools/build_swiglu_ffn.py swiglu_ffn_t8_fp16 --out ops/swiglu_ffn_t8_fp16
    /usr/bin/python3 tools/build_swiglu_ffn.py --analyze ops/swiglu_ffn_t8_fp16
"""
import argparse
import json
import os
import sys

import flatbuffers
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ops as bo                      # noqa: E402  (read-only toolbox, NOTES s5)
import qwen25_weights as qw                 # noqa: E402

F16 = bo.F16
BO = bo.S.BuiltinOperator
BOPT = bo.S.BuiltinOptions

T = 8                 # decoder tokens per step
H = 896               # residual stream width (hidden_size)
I = 4864              # FFN inner width (intermediate_size)
SEED = 20261113       # fixture seed (docs/OP-weights.md); input sampling only
TOL_K = 4.0           # NOTES 10.1: k * 2^-11 * max|out|
FP16_EPS = 2.0 ** -11

WEIGHT_NAMES = {
    "gate_proj": "layer0.mlp.gate_proj.weight",
    "up_proj": "layer0.mlp.up_proj.weight",
    "down_proj": "layer0.mlp.down_proj.weight",
}

SPECS = {
    "swiglu_ffn_t8_fp16": dict(
        kind="swiglu", ctrl=False,
        prediction=(
            "6 MDLA layers: FC(gate), FC(up), Sigmoid, Mul, Mul, FC(down).  silu is NOT "
            "fused into the preceding FC (TFLite FC fused_activation has no SILU slot; "
            "OP-silu s5).  One execution step, no bridge.  Static weights ~24.9 MiB must "
            "reside in DRAM; Temp expected 0 because the [8,4864] fp16 intermediates "
            "(77.8 KB each) fit in L1=3 MB.")),
    "swiglu_ffn_t8_fp16_nosilu_ctrl": dict(
        kind="swiglu", ctrl=True,
        prediction=(
            "4 MDLA layers (FC, FC, Mul, FC) -> the silu epilogue costs exactly 2 layers "
            "and is absorbed by no FC.")),
}


# ------------------------------------------------------------------ tolerance policy
def tolerance_for(max_out):
    """Registered before any device run; rule fixed in the module docstring."""
    m = float(max_out)
    if m <= 1.0:
        return dict(atol=2e-3, rtol=1e-2, tolerance_revision="repo-fp16-default",
                    tolerance_rationale=(
                        "max|expected|<=1, so the repo's registered fp16 contract "
                        "(atol=2e-3, rtol=1e-2; OP-actreduce s1.3, reused by OP-silu) is "
                        "already looser than 4 fp16 ULP at this magnitude "
                        f"(4*2^-11*{m:.3g} = {4 * FP16_EPS * m:.3g}), and it bounds the "
                        "fp16-layer-boundary lower bound measured by the numpy simulation "
                        "(--sim).  No NOTES section 10.1 exception needed."))
    k = TOL_K * FP16_EPS * m
    return dict(atol=k, rtol=k, tolerance_revision="NOTES-10.1-exception",
                tolerance_rationale=(
                    f"max|expected|={m:.4g} > 1, so NOTES section 10.1 applies: "
                    f"atol=rtol={TOL_K:g}*2^-11*max|out|={k:.4g}, supported by a "
                    "device-independent measurement (numpy per-tensor fp16 simulation, "
                    "--sim), not by the device output.  Report max_abs/max|exp| and rel_l2."))


# ------------------------------------------------------------------------ fixture IO
def load_weights():
    w = {}
    for short, name in WEIGHT_NAMES.items():
        a = qw.get(name)                       # read-only, sha256-verified on load
        assert a.dtype == np.float16, (name, a.dtype)
        w[short] = a
    return w


def weight_inventory(w):
    inv = []
    for short, name in WEIGHT_NAMES.items():
        a = w[short]
        inv.append({"tensor": short, "fixture_name": name,
                    "shape": list(a.shape), "dtype": "float16",
                    "bytes": int(a.nbytes), "mib": round(a.nbytes / 2 ** 20, 4)})
    inv.append({"tensor": "total_3_projections",
                "shape": None, "dtype": "float16",
                "bytes": int(sum(i["bytes"] for i in inv)),
                "mib": round(sum(i["bytes"] for i in inv) / 2 ** 20, 4)})
    return inv


# ------------------------------------------------------------------- input sampling
def make_input():
    """A realistically scaled activation for this op: the post-attention RMSNorm output.

    z ~ N(0,1) is normalised per row to RMS 1, then scaled by the fixture's
    layer0.post_attention_layernorm.weight (mean 0.678, std 0.151; docs/OP-weights.md
    section 4) -- that is what a real post-norm activation looks like.  A few elements
    are pinned to +-6 / 0 so the FFN sees a wider range than the gaussian body.
    """
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((T, H))
    z = z / np.sqrt((z ** 2).mean(axis=1, keepdims=True))
    gamma = qw.get_fp32("layer0.post_attention_layernorm.weight").astype(np.float64)
    x = z * gamma
    x[0, 0], x[0, 1], x[1, 0], x[2, 2] = 6.0, -6.0, 0.0, 3.0
    return x.astype(np.float16)


def sigmoid(x):
    """float64 sigmoid, spelled out so no shared library can leak into the golden."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def f16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16)


def goldens(x, w, ctrl=False):
    """Return (exact, alt, intermediates).  float64 chain from the stored fp16 operands."""
    xf = x.astype(np.float64)
    gate = xf @ w["gate_proj"].astype(np.float64).T
    up = xf @ w["up_proj"].astype(np.float64).T
    y = gate * up if ctrl else gate * sigmoid(gate) * up
    exact = f16(y @ w["down_proj"].astype(np.float64).T)
    if ctrl:
        return exact, exact, {"gate": gate, "up": up, "y": y}

    # hypothesis: the device stores every tensor boundary as fp16 (TFLite reference
    # kernels do; OP-silu measured that the composition's residual equals the sigmoid
    # layer's own 1-code error, i.e. the MULs are correctly rounded fp16 multiplies).
    g16, u16 = f16(gate), f16(up)
    s16 = f16(g16.astype(np.float64) * sigmoid(g16.astype(np.float64)))
    y16 = f16(s16.astype(np.float64) * u16.astype(np.float64))
    alt = f16(y16.astype(np.float64) @ w["down_proj"].astype(np.float64).T)
    return exact, alt, {"gate": gate, "up": up, "y": y}


# --------------------------------------------------------------------- flatbuffers
def _fc_node(b, oc, ins, outs):
    S = bo.S
    S.FullyConnectedOptionsStart(b)
    S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)      # NONE
    S.FullyConnectedOptionsAddWeightsFormat(b, 0)                # DEFAULT [N,K]
    S.FullyConnectedOptionsAddKeepNumDims(b, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
    return bo.operator(b, oc, ins, outs, BOPT.FullyConnectedOptions,
                       S.FullyConnectedOptionsEnd(b))


def _mul_node(b, oc, ins, outs):
    S = bo.S
    S.MulOptionsStart(b)
    S.MulOptionsAddFusedActivationFunction(b, 0)                 # NONE
    return bo.operator(b, oc, ins, outs, BOPT.MulOptions, S.MulOptionsEnd(b))


def build_blob(spec, w):
    b = flatbuffers.Builder(1 << 25)          # ~26 MB of weights: pre-size, avoid recopies
    buffers = [bo.buf_empty(b)]               # buffer 0 stays empty: a model input with a
    tensors = []                              # data buffer gets constant-folded by ncc

    def add(name, shape, data=None):
        buffers.append(bo.buf_empty(b) if data is None else bo.buf_data(b, data))
        tensors.append(bo.tensor(b, name, shape, F16, len(buffers) - 1))
        return len(tensors) - 1

    t = {}
    t["x"] = add("input", [T, H])                                     # buffer 0 = empty
    t["wg"] = add("weights_gate", [I, H], w["gate_proj"])
    t["bg"] = add("bias_gate", [I], np.zeros(I, np.float16))           # FC needs 3 inputs
    t["gate"] = add("gate", [T, I])
    t["wu"] = add("weights_up", [I, H], w["up_proj"])
    t["bu"] = add("bias_up", [I], np.zeros(I, np.float16))
    t["up"] = add("up", [T, I])
    t["wd"] = add("weights_down", [H, I], w["down_proj"])
    t["bd"] = add("bias_down", [H], np.zeros(H, np.float16))
    t["y"] = add("y", [T, I])
    t["out"] = add("output", [T, H])
    if not spec["ctrl"]:
        t["sig"] = add("sigmoid", [T, I])
        t["s"] = add("s", [T, I])

    graph = [("FULLY_CONNECTED", ["x", "wg", "bg"], ["gate"]),
             ("FULLY_CONNECTED", ["x", "wu", "bu"], ["up"])]
    if spec["ctrl"]:
        graph += [("MUL", ["gate", "up"], ["y"]),
                  ("FULLY_CONNECTED", ["y", "wd", "bd"], ["out"])]
    else:
        graph += [("LOGISTIC", ["gate"], ["sig"]),
                  ("MUL", ["gate", "sig"], ["s"]),
                  ("MUL", ["s", "up"], ["y"]),
                  ("FULLY_CONNECTED", ["y", "wd", "bd"], ["out"])]

    names = []
    for n, _, _ in graph:
        if n not in names:
            names.append(n)
    opcode_objs = [bo.opcode(b, getattr(BO, n)) for n in names]
    nodes = []
    for n, ins, outs in graph:
        oc = names.index(n)
        ii, oo = [t[k] for k in ins], [t[k] for k in outs]
        if n == "FULLY_CONNECTED":
            nodes.append(_fc_node(b, oc, ii, oo))
        elif n == "MUL":
            nodes.append(_mul_node(b, oc, ii, oo))
        else:                                                          # LOGISTIC: no options
            nodes.append(bo.operator(b, oc, ii, oo))

    desc = f"swiglu ffn {T}x{H}->{T}x{I}->{T}x{H} fp16" + (" ctrl" if spec["ctrl"] else "")
    return bo.finish(b, buffers, opcode_objs, tensors, nodes,
                     [t["x"]], [t["out"]], desc), [n for n, _, _ in graph]


# --------------------------------------------------------------------------- report
def stats(a):
    a = a.astype(np.float64)
    return {"min": float(a.min()), "max": float(a.max()),
            "abs_max": float(np.abs(a).max()),
            "rms": float(np.sqrt((a ** 2).mean()))}


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--analyze", metavar="OPDIR")
    args = ap.parse_args()

    if args.list:
        print("\n".join(SPECS))
        return 0
    if args.analyze:
        return analyze(args.analyze)
    if args.spec and args.spec not in SPECS:
        raise SystemExit(f"unknown spec {args.spec}; use --list")

    spec_name = args.spec or "swiglu_ffn_t8_fp16"
    cfg = SPECS[spec_name]
    w = load_weights()
    x = make_input()
    exact, alt, inter = goldens(x, w, ctrl=cfg["ctrl"])

    m = float(np.abs(exact.astype(np.float64)).max())
    tol = tolerance_for(m)
    sim = float(np.abs(alt.astype(np.float64) - exact.astype(np.float64)).max())
    rel_sim = sim / max(m, 1e-30)

    report = {
        "spec": spec_name, "T": T, "H": H, "I": I,
        "input": stats(x), "output": stats(exact),
        "intermediates": {k: stats(v) for k, v in inter.items()},
        "fp16_layer_boundary_lower_bound_max_abs": sim,
        "fp16_layer_boundary_lower_bound_rel": rel_sim,
        "tolerance": tol,
        "weights": weight_inventory(w),
    }
    if args.dryrun:
        print(json.dumps(report, indent=2))
        return 0

    blob, graph = build_blob(cfg, w)

    out = args.out or f"ops/{spec_name}"
    if not out.endswith(".tflite"):
        out = os.path.join(out, "model.tflite")
    odir = os.path.dirname(os.path.abspath(out))
    os.makedirs(os.path.join(odir, "golden"), exist_ok=True)
    with open(out, "wb") as f:
        f.write(blob)
    x.tofile(os.path.join(odir, "golden", "input.bin"))
    exact.tofile(os.path.join(odir, "golden", "expected.bin"))
    alt.tofile(os.path.join(odir, "golden", "expected_alt.bin"))

    meta = {
        "op": "SWIGLU_FFN",
        "graph": graph,
        "shape": {"T": T, "H": H, "I": I},
        "op_shape": [T, H],
        "dtype": "fp16",
        "input": {"name": "input", "shape": [T, H], "type": "float16",
                  "bytes": int(x.nbytes)},
        "inputs": [{"name": "input", "shape": [T, H], "type": "float16"}],
        "output": {"name": "output", "shape": [T, H], "type": "float16",
                   "bytes": int(exact.nbytes)},
        "quant": {},
        "weights": weight_inventory(w),
        "weights_source": "QWEN25_WEIGHTS_DIR / weights/qwen25_frozen (frozen fixture, "
                          "layer0.mlp.*, sha256-verified by tools/qwen25_weights.get)",
        "seed": SEED,
        "prediction": cfg["prediction"],
        "generator": "tools/build_swiglu_ffn.py",
        "tflite": os.path.basename(out),
        "tflite_bytes": len(blob),
        "reference": "float64_exact_then_store",
        "alt_reference": "per_tensor_fp16_rounding (TFLite reference kernel semantics)",
        "fp16_layer_boundary_lower_bound": {
            "max_abs": sim, "rel": rel_sim,
            "how": "numpy chain where every tensor boundary is rounded to fp16 (golden/"
                   "expected_alt.bin vs golden/expected.bin); device-independent"},
        "stats": report,
        "dryrun_control": cfg["ctrl"],
    }
    meta.update(tol)
    with open(os.path.join(odir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{out}  {len(blob)} B  in [{T},{H}] -> [{T},{H}] fp16  graph={graph}")
    print(f"max|out|={m:.4g}  fp16-layer lower bound={sim:.3e} ({rel_sim:.2e} rel)  "
          f"tolerance: atol={tol['atol']:g} rtol={tol['rtol']:g} "
          f"[{tol['tolerance_revision']}]")
    print("weights: " + ", ".join(
        f"{i['tensor']}={i['bytes']} B" for i in weight_inventory(w)[:3]) +
        f"  total={weight_inventory(w)[-1]['bytes']} B")
    return 0


def analyze(opdir):
    """Device dump vs both reference models, plus the relative form of the error."""
    meta = json.load(open(os.path.join(opdir, "model.json")))
    dt = np.float16
    dev = np.fromfile(os.path.join(opdir, "out_device.bin"), dtype=dt)
    shape = tuple(meta["output"]["shape"])
    exact = np.fromfile(os.path.join(opdir, "golden", "expected.bin"), dtype=dt)
    alt = np.fromfile(os.path.join(opdir, "golden", "expected_alt.bin"), dtype=dt)
    if dev.size != exact.size:
        print(f"SIZE MISMATCH dev={dev.size} expected={exact.size}")
        return 1
    d = dev.reshape(shape)
    e = exact.reshape(shape)
    print(f"{opdir}: {meta['op']} {meta['shape']} dtype=fp16  n={dev.size}")
    # noise floor (NOTES section 4): the same .dla + same input run twice
    r1p = os.path.join(opdir, "out_r1.bin")
    r2p = os.path.join(opdir, "out_r2.bin")
    if os.path.exists(r1p) and os.path.exists(r2p):
        r1 = np.fromfile(r1p, dtype=dt).astype(np.float64)
        r2 = np.fromfile(r2p, dtype=dt).astype(np.float64)
        nf = float(np.abs(r1 - r2).max())
        print(f"  noise floor         max|r1-r2|={nf:.3e}  runs_identical="
              f"{bool(np.array_equal(r1, r2))}")
    m = float(np.abs(e).max())
    dev_vs_exact = np.abs(e.astype(np.float64) - d.astype(np.float64)).max()
    print(f"  relative form       max|exp|={m:.6g}  max_abs/max|exp|="
          f"{dev_vs_exact / max(m, 1e-30):.3e}  "
          f"ULP16@max|exp|={2.0 ** -11 * 2.0 ** np.floor(np.log2(max(m, 1e-30))):.3e}")
    for tag, ref in (("exact(fp64)", e), ("alt(per-tensor fp16)", alt.reshape(shape))):
        r = ref.astype(np.float64)
        dd = np.abs(r - d.astype(np.float64))
        print(f"  vs {tag:20s} max|d|={dd.max():.3e}  rel(max)= "
              f"{dd.max() / max(np.abs(r).max(), 1e-30):.2e}  "
              f"rel_l2={np.linalg.norm(r - d) / max(np.linalg.norm(r), 1e-30):.2e}  "
              f"exact={int((dd == 0).sum())}/{dd.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
