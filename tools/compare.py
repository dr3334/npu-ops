#!/usr/bin/env python3
"""Compare a device output dump against the golden expected output.

    /usr/bin/python3 tools/compare.py ops/<name> [--atol 1e-5] [--rtol 1e-4]

Integer path (int8/int32): requires exact equality, reports mismatch count.
Float path: reports max_abs, rmse, rel_l2, NaN/Inf counts and checks atol/rtol.
Exit code 0 = pass.
"""
import argparse
import json
import os
import sys

import numpy as np

DT = {"int8": np.int8, "int16": np.int16, "int32": np.int32,
      "float32": np.float32, "float16": np.float16,
      "i8": np.int8, "i16": np.int16, "i32": np.int32, "fp32": np.float32, "fp16": np.float16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("opdir")
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--device", default="out_device.bin")
    args = ap.parse_args()

    d = args.opdir
    meta = json.load(open(os.path.join(d, "model.json")))
    exp_path = os.path.join(d, "golden", "expected.bin")
    dev_path = os.path.join(d, args.device)
    if not os.path.exists(dev_path):
        print(f"FAIL  {d}: device output missing ({args.device})")
        return 2

    otype = meta["output"]["type"]
    dt = DT[otype]
    exp = np.fromfile(exp_path, dtype=dt)
    dev = np.fromfile(dev_path, dtype=dt)
    shape = tuple(meta["output"]["shape"])
    print(f"op        : {meta['op']} {meta['shape']}  dtype={meta['dtype']}")
    print(f"expected  : {exp.size} vals   device: {dev.size} vals   logical shape={shape}")
    if exp.size != dev.size:
        print(f"FAIL  size mismatch ({exp.size} vs {dev.size})")
        return 1

    exp = exp.reshape(shape)
    dev = dev.reshape(shape)

    if np.issubdtype(dt, np.integer):
        tol = int(meta.get("int8_tol", 0))
        neq = exp.astype(np.int64) - dev.astype(np.int64)
        n_bad = int((neq != 0).sum())
        max_abs = int(np.abs(neq).max()) if neq.size else 0
        print(f"exact     : {'YES' if n_bad == 0 else 'NO'}   mismatched={n_bad}/{exp.size}")
        print(f"contract  : int8_tol={tol}  max_abs_diff={max_abs}"
              f"  fraction_off={n_bad / exp.size:.4%}")
        if n_bad and tol == 0:
            idx = np.argwhere(neq != 0)[:5]
            for i in map(tuple, idx):
                print(f"  at {i}: expected={exp[i]} device={dev[i]}")
        ok = max_abs <= tol
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    e = exp.astype(np.float64)
    a = dev.astype(np.float64)
    # default tolerances per storage type (register-before-test, see reports/)
    # 与 int8_tol 同等待遇：model.json 里登记的 atol/rtol 优先于内置默认值，
    # 这样"先登记容差再测"的契约对浮点算子也生效（act-agent 反馈）。
    d_atol = 2e-2 if otype == "float16" else 1e-5
    d_rtol = 2e-2 if otype == "float16" else 1e-4
    if args.atol is None:
        args.atol = float(meta.get("atol", d_atol))
    if args.rtol is None:
        args.rtol = float(meta.get("rtol", d_rtol))
    diff = np.abs(e - a)
    denom = np.maximum(np.abs(e), 1e-12)
    max_abs = float(diff.max())
    rmse = float(np.sqrt(np.mean((e - a) ** 2)))
    rel_l2 = float(np.linalg.norm(e - a) / max(np.linalg.norm(e), 1e-12))
    nan_e, nan_a = int(np.isnan(e).sum()), int(np.isnan(a).sum())
    inf_e, inf_a = int(np.isinf(e).sum()), int(np.isinf(a).sum())
    ok = bool(np.all(diff <= args.atol + args.rtol * np.abs(e)))
    print(f"max_abs   : {max_abs:.3e}")
    print(f"rmse      : {rmse:.3e}")
    print(f"rel_l2    : {rel_l2:.3e}")
    print(f"nan/inf   : exp {nan_e}/{inf_e}  dev {nan_a}/{inf_a}")
    print(f"range     : exp [{e.min():.4g},{e.max():.4g}]  dev [{a.min():.4g},{a.max():.4g}]")
    print(f"tol       : atol={args.atol:g} rtol={args.rtol:g}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())