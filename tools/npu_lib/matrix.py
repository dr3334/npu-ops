#!/usr/bin/env python3
"""跑票面要求的最小矩阵，并把证据写成 reports/npu-lib-matrix.{json,md}。

    PYTHONPATH=tools /usr/bin/python3 -m npu_lib.matrix            # add/mul/sub x 64/896/96
    PYTHONPATH=tools /usr/bin/python3 -m npu_lib.matrix --lens 64,896 --ops add,mul

每个组合都走完整链路：Graph -> model.tflite -> .dla -> 资产捕获 -> plan -> npu_run(自建提交器)
-> 两道验收（同源逐字节 + CPU golden 逐位）。所有板端动作都在外层 devlock 里（NPU_LOCK_HELD=1）。
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import env
from .graph import Graph, _sha256_file, _sha256_bytes
from .dtype import DType

REPORTS = env.REPO_ROOT / "reports"
SCALE = 1.0 / 127.0


def sha_file(p):
    p = Path(p)
    return _sha256_file(p) if p.exists() else ""


def prove_no_vendor_lib(name, device_run_dir):
    """证据：提交进程存活期间 /proc/<pid>/maps 里没有 libneuron/neuronrt；二进制本身也没有
    dlopen / DT_NEEDED。做法：npu_run 跑完多活 1.5 s，再从另一个 shell 采样。"""
    remote = "%s/bin/npu_run" % env.DEV_ROOT
    script = ("cd %s && NPU_RUN_HOLD_MS=1500 %s plan.txt > /dev/null 2>&1 & "
              "sleep 0.7; "
              "pids=$(pidof npu_run); "
              "echo \"PIDS=$pids\"; "
              "for p in $pids; do cat /proc/$p/maps; done > %s/maps.txt 2>/dev/null; "
              "echo '--- vendor-ish lines ---'; "
              "grep -iE 'neuron|mdla' %s/maps.txt || echo NONE; "
              "echo \"MAP_LINES=$(wc -l < %s/maps.txt)\"; "
              "wait" % (device_run_dir, remote, device_run_dir, device_run_dir, device_run_dir))
    r = env.dev_shell(script, check=False, capture=True)
    out = (r.stdout or "")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    hits = [ln for ln in lines if "neuron" in ln.lower() or "mdla" in ln.lower()]
    info = {
        "sample": lines[-6:],
        "vendor_lib_mapped": bool([h for h in hits if h.strip() != "NONE"]),
        "map_lines": next((ln for ln in lines if ln.startswith("MAP_LINES=")), ""),
    }
    # 二进制层面：静态链接（无 DT_NEEDED）+ 没有任何厂商库的字样
    probe = ("cd '%s'; "
             "echo STATIC_HDR=$(readelf -d probe/npu_run 2>/dev/null | grep -c 'no dynamic section'); "
             "echo NEEDED=$(readelf -d probe/npu_run 2>/dev/null | grep -c NEEDED); "
             "echo VENDOR_STRINGS=$(strings -a probe/npu_run | grep -cE 'libneuron|neuronrt'); "
             "echo DLOPEN_STRINGS=$(strings -a probe/npu_run | grep -E 'dlopen' | head -2 | paste -sd, -)"
             % env.REPO_ROOT)
    r2 = env.run(["bash", "-lc", probe], cwd=env.REPO_ROOT, capture=True, check=False)
    checks = {}
    for ln in (r2.stdout or "").splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            checks[k.strip()] = v.strip()
    info["binary_check"] = checks
    info["static_link"] = checks.get("STATIC_HDR") == "1"
    info["dt_needed"] = checks.get("NEEDED", "?")
    info["vendor_strings"] = checks.get("VENDOR_STRINGS", "?")
    return info


def run_combo(op, length, do_maps=False):
    name = "%s_1x%d_int8" % (op, length)
    q = DType.int8(SCALE)
    g = Graph(name)
    a = g.input("a", (1, length), q)
    b = g.input("b", (1, length), q)
    t0 = time.time()
    getattr(g, op)(a, b)
    art = g.compile()
    cap_a = np.fromfile(art.golden / "a.bin", dtype=np.int8)
    cap_b = np.fromfile(art.golden / "b.bin", dtype=np.int8)
    dev = g.run({"a": cap_a, "b": cap_b})
    # run() 每次覆盖 run/out.bin，先把这一跑的输出留档，免得被第二跑冲掉
    out_path = art.root / "run" / "out.bin"
    primary = art.root / "run" / "out_primary.bin"
    if out_path.exists():
        shutil.copy(out_path, primary)
    res = g.verify(dev)
    # 第二跑：**换一对输入**，证明数值门不是"重放抓取时那对输入"才成立
    # （这一跑"同源"按定义不适用：厂商那次只跑过 capture 的输入）
    rng = np.random.default_rng(7 + length)
    alt_a = rng.integers(-100, 101, size=length).astype(np.int8)
    alt_b = rng.integers(-100, 101, size=length).astype(np.int8)
    alt = g.run({"a": alt_a, "b": alt_b})
    res_alt = g.verify(alt)
    if out_path.exists():
        shutil.copy(out_path, art.root / "run" / "out_alt.bin")
    dt = time.time() - t0

    plan_txt = art.plan_path.read_text()
    run_log = art.root / "run" / "npu_run.log"
    npu_log = run_log.read_text() if run_log.exists() else ""
    out_file = art.root / "run" / "out.bin"

    ev = {
        "name": name,
        "op": op,
        "shape": [1, length],
        "numel": length,
        "dtype": "int8",
        "scales": {"a": SCALE, "b": SCALE, "out": dev.scale},
        "requant_ratio": res.detail["requant_ratio"],
        "plan": {"path": str(art.plan_path), "sha256": _sha256_bytes(plan_txt.encode()),
                 "text": plan_txt},
        "sha256": {
            "model.tflite": sha_file(art.root / "model.tflite"),
            "model.dla": sha_file(art.root / "model.dla"),
            "golden/input_a": sha_file(art.golden / "a.bin"),
            "golden/input_b": sha_file(art.golden / "b.bin"),
            "golden/expected": sha_file(art.golden / "expected.bin"),
            "vendor_out": sha_file(art.capture / "vendor_out.bin"),
            "device_out": sha_file(primary),          # 抓取输入那一跑的输出（留档件）
            "device_out_alt": sha_file(art.root / "run" / "out_alt.bin"),
        },
        "bytes": {"device_out": primary.stat().st_size if primary.exists() else 0},
        "verify": res.as_dict(),
        "alt_inputs": {
            "numeric_ok": res_alt.numeric_ok,
            "same_source_ok": res_alt.same_source_ok,
            "note": "fresh random inputs; the vendor run only ever saw the captured pair, "
                    "so the same-source gate is not defined here",
            "sha256_out": res_alt.detail["sha256_out"],
            "sha256_expected": res_alt.detail["sha256_expected"],
        },
        "pass": bool(res.ok and res_alt.numeric_ok),
        "npu_run_log": npu_log.strip().splitlines()[:8],
        "seconds": round(dt, 1),
    }
    if do_maps:
        ev["no_vendor_lib"] = prove_no_vendor_lib(name, art.manifest["device_run_dir"])
    return ev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="add,mul,sub")
    ap.add_argument("--lens", default="64,896,96")
    ap.add_argument("--maps", action="store_true", help="额外采一次 /proc/<pid>/maps 证据")
    ap.add_argument("--out-json", default=str(REPORTS / "npu-lib-matrix.json"))
    ap.add_argument("--out-md", default=str(REPORTS / "npu-lib-matrix.md"))
    args = ap.parse_args(argv)

    ops = [x.strip() for x in args.ops.split(",") if x.strip()]
    lens = [int(x) for x in args.lens.split(",") if x.strip()]
    results = []
    for op in ops:
        for n in lens:
            try:
                ev = run_combo(op, n, do_maps=args.maps and not results)
            except Exception as e:                     # 失败也留证据（仓库规则：失败保留）
                ev = {"name": "%s_1x%d_int8" % (op, n), "op": op, "numel": n,
                      "pass": False, "error": "%s: %s" % (type(e).__name__, e)}
                print("[matrix] %s_1x%d_int8 FAILED: %s" % (op, n, ev["error"]), file=sys.stderr)
            results.append(ev)
            ev_path = Path(env.artifacts_root()) / ev["name"] / "evidence.json"
            if ev_path.parent.exists():
                ev_path.write_text(json.dumps(ev, indent=2))

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "runner": "probe/npu_run (plan-driven, /dev/apusys, no libneuron/neuronrt)",
        "results": results,
        "passed": sum(1 for r in results if r.get("pass")),
        "total": len(results),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2))

    md = ["# npu_lib 板端矩阵（自动生成，勿手改）", "",
          "生成：`PYTHONPATH=tools /usr/bin/python3 -m npu_lib.matrix`（外层 `tools/devlock.sh run`）", "",
          "| 算子 | 形状 | 元素 | requant | 同源逐字节 | golden 逐位 | 结果 | 证据 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "verify" in r:
            md.append("| %s | %s | %d | %s | %s | %s | **%s** | `artifacts/%s/evidence.json` |" % (
                r["op"].upper(), r["shape"], r["numel"], r["requant_ratio"],
                "PASS" if r["verify"]["same_source_ok"] else "FAIL",
                "PASS" if r["verify"]["numeric_ok"] else "FAIL",
                "PASS" if r["pass"] else "FAIL", r["name"]))
        else:
            md.append("| %s | [1,%d] | %d | - | - | - | **FAIL** | %s |"
                      % (r["op"].upper(), r["numel"], r["numel"], r.get("error", "")))
    md += ["", "共 %d/%d 组合 PASS。" % (summary["passed"], summary["total"])]
    Path(args.out_md).write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
