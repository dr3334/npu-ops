#!/usr/bin/env python3
"""Noise floor + error anatomy for ops/lm_head_t8_fp16 (NOTES 4 protocol).

    /usr/bin/python3 ops/lm_head_t8_fp16/noise_floor.py

Reads (all gitignored, produced by board_run.sh):
    out_r1.bin, out_r2.bin   the two e2e runs of the SAME .dla with the SAME input
    out_r3.bin               third run from the timed push/run/pull block
    golden/expected.bin      independent numpy float32 reference (cast to fp16)

Writes noise_floor.json and prints the numbers quoted in docs/OP-lm-head.md.
"""
import json
import os

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
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
    return {
        "max_abs": float(d.max()),
        "n_diff": nz,
        "n_total": int(d.size),
        "frac_diff": nz / d.size,
        "rmse": float(np.sqrt(np.mean(d ** 2))),
    }


def ulp_at(x):
    """fp16 spacing at |x| (2**-11 relative is the floor; exact binade spacing otherwise)."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    e = np.floor(np.log2(np.maximum(x, 2.0 ** -14)))
    return np.maximum(2.0 ** (e - 10), 2.0 ** -24)


exp = load(os.path.join("golden", "expected.bin"))
r1, r2, r3 = load("out_r1.bin"), load("out_r2.bin"), load("out_r3.bin")
mx = float(np.abs(exp.astype(np.float64)).max())

out = {"output_shape": list(SHAPE), "output_bytes": int(exp.nbytes),
       "max_abs_expected": mx, "atol": META["atol"], "rtol": META["rtol"]}

noise = {}
for name, other in (("r1_vs_r2", r2), ("r1_vs_r3", r3), ("r2_vs_r3", r3)):
    if other is None:
        continue
    noise[name] = stats(r1 if name.startswith("r1") else r2, other)
out["noise_floor"] = noise

if r2 is not None:
    nf = noise["r1_vs_r2"]["max_abs"]
    out["noise_floor_value"] = nf
    out["noise_floor_is_zero"] = nf == 0.0

# device vs golden, with ULP accounting
s = stats(r1, exp)
err = np.abs(r1.astype(np.float64) - exp.astype(np.float64))
ulp = ulp_at(exp)
ratio = err / ulp
# Per-element ULP ratio is only meaningful where the element is not near zero: for a
# ~0 element one fp16 ULP is a huge *relative* step while the absolute error stays tiny.
near_zero = np.abs(exp.astype(np.float64)) < 0.01
out["device_vs_golden"] = dict(s)
out["device_vs_golden"].update({
    "max_abs_relative": s["max_abs"] / mx,
    "ulp_at_max_expected": float(ulp_at(mx)),
    "max_err_in_ulp_at_max": s["max_abs"] / float(ulp_at(mx)),
    "exact_fraction": float((err == 0).sum()) / err.size,
    "within_1ulp_fraction": float((ratio <= 1.0).sum()) / err.size,
    "max_ulp_ratio_abs_ge_0.01": float(ratio[~near_zero].max()),
    "n_abs_lt_0.01": int(near_zero.sum()),
    "max_abs_err_abs_lt_0.01": float(err[near_zero].max()) if near_zero.any() else 0.0,
    "rel_atol": META["atol"] / mx,
    "contract_headroom": float(META["atol"] + META["rtol"] * mx) / max(s["max_abs"], 1e-30),
    "verdict": "PASS" if s["max_abs"] <= META["atol"] + META["rtol"] * mx else "FAIL",
})

json.dump(out, open(os.path.join(D, "noise_floor.json"), "w"), indent=2)

print("output          : %s  %d B (%.3f MB)" % (SHAPE, exp.nbytes, exp.nbytes / 1e6))
print("max|expected|   : %.6f" % mx)
for k, v in noise.items():
    print("noise %-9s: max_abs=%.3e  differ=%d/%d (%.4f%%)  rmse=%.3e"
          % (k, v["max_abs"], v["n_diff"], v["n_total"], 100 * v["frac_diff"], v["rmse"]))
g = out["device_vs_golden"]
print("device vs golden: max_abs=%.3e  differ=%d/%d (%.4f%%)  rmse=%.3e"
      % (g["max_abs"], g["n_diff"], g["n_total"], 100 * g["frac_diff"], g["rmse"]))
print("  max_abs/max|exp|          : %.3e" % g["max_abs_relative"])
print("  ULP at max|exp|           : %.3e  => max err = %.2f ULP there" % (g["ulp_at_max_expected"], g["max_err_in_ulp_at_max"]))
print("  bit-identical elements    : %.4f%%   within 1 ULP: %.4f%%" % (100 * g["exact_fraction"], 100 * g["within_1ulp_fraction"]))
print("  max ULP ratio (|exp|>=0.01): %.2f   [|exp|<0.01: n=%d, max abs err %.2e]"
      % (g["max_ulp_ratio_abs_ge_0.01"], g["n_abs_lt_0.01"], g["max_abs_err_abs_lt_0.01"]))
print("  contract atol+rtol*max|exp|      : %.4e  (headroom %.2fx)" % (META["atol"] + META["rtol"] * mx, g["contract_headroom"]))
print("  VERDICT: %s" % g["verdict"])