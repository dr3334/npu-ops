#!/usr/bin/env python3
"""Noise floor + error anatomy for ops/decoder_layer_t128_fp16 (NOTES 4 / 10.1 protocol).

    /usr/bin/python3 ops/decoder_layer_t128_fp16/noise_floor.py

Reads (all gitignored, produced by the two locked e2e runs):
    out_r1.bin / out_r2.bin   the SAME .dla with the SAME input, run twice
    golden/expected.bin       the registered numpy float32 golden
Writes noise_floor.json and prints the numbers quoted in docs/OP-decoder-layer.md 14.4.

The point of the file is the three-way split that makes a loose-looking tolerance
defensible:

    |device - golden|  vs  |fp16-sim - golden|  vs  |device - fp16-sim|

The middle term is the *device-independent* floor: a numpy chain that rounds to fp16 at
every tensor boundary (what MDLA does, docs/OP-qkv.md 5.2).  A device that is only a
fraction of a ULP worse than that floor is not "wrong", and NOTES 10.1 says exactly that.
"""
import json
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(D))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_decoder_layer as bdl          # numpy graph + fixture loader (read-only use)

META = json.load(open(os.path.join(D, "model.json")))
SHAPE = tuple(META["output"]["shape"])


def load(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return None
    return np.fromfile(p, dtype=np.float16).reshape(SHAPE)


def stats(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    nz = int((d > 0).sum())
    return {"max_abs": float(d.max()), "n_diff": nz, "n_total": int(d.size),
            "frac_diff": nz / d.size, "rmse": float(np.sqrt(np.mean(d ** 2)))}


exp = load(os.path.join("golden", "expected.bin"))
r1, r2 = load("out_r1.bin"), load("out_r2.bin")
mx = float(np.abs(exp.astype(np.float64)).max())
ulp = float(np.spacing(np.float16(mx)))

out = {"op": META["op"], "output_shape": list(SHAPE), "output_bytes": int(exp.nbytes),
       "max_abs_expected": mx, "ulp_at_max_expected": ulp,
       "atol": META["atol"], "rtol": META["rtol"], "threshold_at_max": META["threshold_at_max"]}

noise = {}
if r1 is not None and r2 is not None:
    noise["r1_vs_r2"] = stats(r1, r2)
    out["noise_floor_value"] = noise["r1_vs_r2"]["max_abs"]
    out["noise_floor_is_zero"] = out["noise_floor_value"] == 0.0
out["noise_floor"] = noise

# ---- device vs golden -------------------------------------------------------------
dev = r1 if r1 is not None else load("out_device.bin")
if dev is not None:
    s = stats(dev, exp)
    out["device_vs_golden"] = dict(s)
    out["device_vs_golden"].update({
        "max_abs_in_ulp": s["max_abs"] / ulp,
        "max_abs_relative": s["max_abs"] / mx,
        "rel_l2": float(np.linalg.norm(dev.astype(np.float64) - exp.astype(np.float64))
                        / np.linalg.norm(exp.astype(np.float64))),
        "bit_identical_codes": int((dev == exp).sum()),
        "passes_registered_tolerance": bool(np.all(
            np.abs(dev.astype(np.float64) - exp.astype(np.float64))
            <= META["atol"] + META["rtol"] * np.abs(exp.astype(np.float64)))),
    })

# ---- the device-independent floor + the sim ---------------------------------------
bdl._configure_from_meta(META)
P = bdl.fixture()
x16 = bdl.input_hidden()
golden = np.asarray(bdl.forward(x16, P, np.float32, False, "out")["out"], np.float16).reshape(SHAPE)
sim = np.asarray(bdl.forward(x16, P, np.float32, True, "out")["out"], np.float16).reshape(SHAPE)
fs = stats(sim, golden)
out["fp16_sim_vs_golden"] = dict(fs)
out["fp16_sim_vs_golden"].update({
    "max_abs_in_ulp": fs["max_abs"] / ulp,
    "max_abs_relative": fs["max_abs"] / mx,
    "note": "device-independent floor: numpy chain with a fp16 rounding at every tensor "
            "boundary vs the single-rounding golden",
})
out["registered_floor"] = META["fp16_floor"]
if dev is not None:
    ds = stats(dev, sim)
    out["device_vs_fp16_sim"] = dict(ds)
    out["device_vs_fp16_sim"].update({
        "max_abs_in_ulp": ds["max_abs"] / ulp,
        "bit_identical_codes": int((dev == sim).sum()),
    })
    out["device_minus_floor_in_ulp"] = (out["device_vs_golden"]["max_abs_in_ulp"]
                                        - out["fp16_sim_vs_golden"]["max_abs_in_ulp"])

# ---- where the floor comes from (per-stage) ---------------------------------------
stages = {}
for k in ["n1", "q3", "scores", "probs", "attn", "r1", "n2", "mlp_y", "down", "out"]:
    a = np.asarray(bdl.forward(x16, P, np.float32, False, "out")[k], np.float16).astype(np.float64)
    b = np.asarray(bdl.forward(x16, P, np.float32, True, "out")[k], np.float16).astype(np.float64)
    m = float(np.abs(a).max())
    u = float(np.spacing(np.float16(m)))
    stages[k] = {"max_abs": m, "sim_vs_golden": float(np.abs(a - b).max()),
                 "sim_vs_golden_in_ulp_of_stage": float(np.abs(a - b).max()) / u}
out["stage_floor"] = stages

path = os.path.join(D, "noise_floor.json")
json.dump(out, open(path, "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "stage_floor"}, indent=2))
print("stage floor (sim vs golden, in ULP of that stage):")
for k, v in stages.items():
    print("  %-8s max|v|=%-10.4g diff=%.3e (%.1f ULP)" % (k, v["max_abs"], v["sim_vs_golden"],
                                                           v["sim_vs_golden_in_ulp_of_stage"]))
print("wrote", path)
