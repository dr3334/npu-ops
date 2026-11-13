#!/usr/bin/env python3
"""Capability scan for the conv/pool family: build a config matrix, compile each
with the fixed policy (--disallow-bridge) and report where the native single-step
MDLA mapping stops working.

Motivation (AGENTS.md 3.3 "find the boundary, not a sample"): the first 16 probes
showed MAX/AVG_POOL_2D, stride-2 and dilation-2 convs failing with

    Bridging is disallowed but gap is detected between Input and MDLA_5_0

while 3x3/1x1 stride-1 convs compiled to a single MDLA step.  Compiling the same
models *with* bridging allowed showed the gap is filled by an **eDPA 1.0
"Relayout"** step (MDLA's own DMA engine, 64 B of code), not by CPU code.  This
scan maps which (op, C, H, stride, dilation, multiplier) combinations still fit
in one MDLA step under the fixed --disallow-bridge policy.

Output: ops/_scan/convpool_scan.tsv (one row per config) + per-config compile logs
under ops/_scan/<name>/model.compile.log.  Exit code 0 always (this is a probe).

Usage:  /usr/bin/python3 tools/scan_convpool.py [--fast]
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_convpool as B  # noqa: E402


def compile_one(name, cfg):
    d = os.path.join(ROOT, "ops", "_scan", name)
    os.makedirs(os.path.join(d, "golden"), exist_ok=True)
    blob, meta, x, y, consts = B.build(cfg.pop("kind"), B.C(**dict(cfg)))
    open(os.path.join(d, "model.tflite"), "wb").write(blob)
    x.tofile(os.path.join(d, "golden", "input.bin"))
    y.tofile(os.path.join(d, "golden", "expected.bin"))
    meta["tflite"] = "model.tflite"
    json.dump(meta, open(os.path.join(d, "model.json"), "w"), indent=2)
    r = subprocess.run([os.path.join(ROOT, "tools", "compile.sh"),
                        os.path.join("ops", "_scan", name, "model.tflite"),
                        os.path.join("ops", "_scan", name, "model.dla")],
                       capture_output=True, text=True)
    log = open(os.path.join(d, "model.compile.log")).read()
    layers = re.findall(r"^   [├└] (\w+<\d+>\{\w+\})", log, re.M)
    mem = re.search(r"\[\s*0\]:\s*(\S+\s*\S*)\s+(\d+\S*)\s+(\d+\S*)\s+(\d+\S*)\s+(\d+\S*)\s+(\d+\S*)\s+(\S+)", log)
    gap = "yes" if "gap is detected" in log else "no"
    err = ""
    if "gap is detected" in log:
        err = re.search(r"gap is detected between (.+)", log).group(1)
    return dict(name=name, rc=r.returncode, layers=",".join(layers) or "-",
                gap=gap, gap_between=err,
                mem=(mem.group(0).strip() if mem else "-"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer configs")
    args = ap.parse_args()

    cfgs = []
    Cs = [4, 8, 16, 32] if not args.fast else [4, 16]
    for C in Cs:
        # pooling: does the input relayout disappear with wider channels?
        for kind in ("maxpool", "avgpool"):
            cfgs.append((f"{kind}_8x8x{C}_2x2s2", dict(
                kind=kind, h=8, w=8, c=C, fh=2, fw=2, s=2, pad="VALID")))
            cfgs.append((f"{kind}_16x16x{C}_3x3s1same", dict(
                kind=kind, h=16, w=16, c=C, fh=3, fw=3, s=1, pad="SAME")))
        # conv: stride and dilation
        cfgs.append((f"conv_16x16x{C}_f3s2v", dict(
            kind="conv", h=16, w=16, c=C, oc=max(8, C), fh=3, fw=3, s=2, pad="VALID")))
        cfgs.append((f"conv_8x8x{C}_f3d2", dict(
            kind="conv", h=8, w=8, c=C, oc=max(8, C), fh=3, fw=3, dil=2)))
        cfgs.append((f"conv_8x8x{C}_f3s1same", dict(
            kind="conv", h=8, w=8, c=C, oc=max(8, C), fh=3, fw=3)))
        cfgs.append((f"dwconv_8x8x{C}_f3m2", dict(
            kind="dwconv", h=8, w=8, c=C, fh=3, fw=3, mult=2)))
        cfgs.append((f"dwconv_8x8x{C}_f3m1", dict(
            kind="dwconv", h=8, w=8, c=C, fh=3, fw=3, mult=1)))

    rows = []
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    out = os.path.join(ROOT, "ops", "_scan", "convpool_scan.tsv")
    with open(out, "w") as f:
        f.write("config\tcompile_rc\tlayers\tgap\tgap_between\tmdla_mem_row\n")
        for name, cfg in cfgs:
            try:
                r = compile_one(name, cfg)
            except Exception as e:                    # build-side failure
                r = dict(name=name, rc="BUILD-ERR", layers=str(e)[:60],
                         gap="-", gap_between="", mem="-")
            print(f"{r['name']:34s} rc={str(r['rc']):4s} gap={r['gap']:3s} "
                  f"{r['layers']}")
            f.write("\t".join(str(r[k]) for k in
                              ("name", "rc", "layers", "gap", "gap_between", "mem")) + "\n")
            f.flush()
            rows.append(r)
    print(f"\nwritten: {out}")
    ok = sum(1 for r in rows if str(r["rc"]) == "0")
    print(f"native single-step OK: {ok}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
