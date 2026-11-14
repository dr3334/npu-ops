#!/usr/bin/env python3
"""bbdiff —— 受控差分探查工具（把黑盒编译器当参考编译器用）。

    /usr/bin/python3 tools/bbdiff.py --matrix tools/bbdiff-matrix-blackbox03.json

职责（票 03 的「唯一新接缝」）：跑编译 -> 收集产物 -> 两次编译一致性检查 -> 差分 -> 出报告。

设计约束（每条都对着票面/规格）：
  * 编译只走 tools/compile.sh（唯一入口，不另开第二条路径）；
  * 实验矩阵是**数据**：加变体只加一行 variants[]，判定也只写 checks[]，改它不动本文件；
  * 产物落 ops/_probe/<experiment>/<variant>/，沿用既有证据形态（*.compile.log + json）；
  * **先做一致性检查再做差分**：同一份模型重复编译 N 次，把跨运行不稳定的字节记成「已知噪声区」，
    后续差分把噪声区排除；
  * 失败实验也留证据（命令、日志、退出码、产物快照），不删。

两条来自实测、写进设计的环境事实（证据见 reports/compiler-blackbox-03.md）：
  1. .dla 里嵌了模型文件名/路径（长度影响产物大小）-> 每个变体都从同一个固定路径编译
     （<exp>/_build/model.tflite），否则跨变体差分里混进路径字符串，看不出谁真的变了；
  2. 同一份输入的重复编译不是逐字节可复现的（静态数据段长度抖动 + 产物内嵌的哈希串），
     所以「噪声区」不是可选项，而是差分的前提。

退出码：0 = 所有 check 都符合预期（expect 默认 pass）；1 = 有 check 不符；2 = 用法/环境错误。
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
BASE_SEED = 20261112            # 与 tools/build_ops.py 的 SEED 一致（票 01/02 的 fc 夹具）
A_SCALE = 1.0 / 127.0           # 与 build_ops.py 的 fc int8 量化参数一致
W_SCALE = 1.0 / 127.0
B_SCALE = A_SCALE * W_SCALE     # 1/16129
O_SCALE = B_SCALE * 32.0        # requant 倍率 1/32（设备契约）


# ----------------------------------------------------------------- 模型（数据 -> tflite）
def _codes(spec):
    """按变体声明生成 FC 的三个常量张量（输入码 / 权重码 / bias 码）。

    量化参数固定成 build_ops.py 的 fc int8 那一档，所以 weights.kind=base 的变体
    与 ops/fc_1x64x64_int8/model.tflite 是同一份夹具（同 seed、同量化、同张量命名）。
    """
    import numpy as np
    m, k, n = int(spec["m"]), int(spec["k"]), int(spec["n"])
    rng = np.random.default_rng(int(spec.get("seed", BASE_SEED)))
    a_f = rng.uniform(-1.0, 1.0, size=(m, k)).astype(np.float32)
    w_f = rng.uniform(-1.0, 1.0, size=(n, k)).astype(np.float32)
    b_f = rng.uniform(-0.5, 0.5, size=(n,)).astype(np.float32)
    wspec = spec.get("weights") or {"kind": "base"}
    kind = wspec.get("kind", "base")
    if kind == "rng":                       # 换一组权重（形状与 bias 不变）
        w_f = np.random.default_rng(int(wspec["seed"])).uniform(
            -1.0, 1.0, size=(n, k)).astype(np.float32)
    elif kind == "pattern":                 # 全部同一个码值（标定「权重区」用：只动权重）
        w_f = np.full((n, k), float(wspec["code"]) * W_SCALE, dtype=np.float32)
    elif kind == "base":
        pass
    else:
        raise SystemExit("weights.kind 未知: %r" % kind)
    a_q = np.clip(np.round(a_f / A_SCALE), -127, 127).astype(np.int8)
    w_q = np.clip(np.round(w_f / W_SCALE), -127, 127).astype(np.int8)
    b_q = np.round(b_f / B_SCALE).astype(np.int32)
    return a_q, w_q, b_q


def build_model(spec):
    """写出 FULLY_CONNECTED 的 tflite 字节（复用 tools/build_ops.py 的 flatbuffer 助手，不改它）。"""
    import flatbuffers
    sys.path.insert(0, str(TOOLS))
    import build_ops as bo            # 只 import，不改（公共工具）
    S = bo.S
    m, k, n = int(spec["m"]), int(spec["k"]), int(spec["n"])
    a_q, w_q, b_q = _codes(spec)
    b = flatbuffers.Builder(4096)
    buffers = [bo.buf_empty(b)]                       # 0: IO 占位（输入张量不带数据）
    buffers.append(bo.buf_data(b, w_q))               # 1: 权重常量
    buffers.append(bo.buf_data(b, b_q))               # 2: bias 常量
    buffers.append(bo.buf_empty(b))                   # 3: 输出占位
    t_in = bo.tensor(b, "input", [m, k], bo.I8, 0, bo.quant_tensor(b, [A_SCALE], [0]))
    t_w = bo.tensor(b, "weights", [n, k], bo.I8, 1, bo.quant_tensor(b, [W_SCALE], [0]))
    t_b = bo.tensor(b, "bias", [n], bo.I32, 2, bo.quant_tensor(b, [B_SCALE], [0]))
    t_out = bo.tensor(b, "output", [m, n], bo.I8, 3, bo.quant_tensor(b, [O_SCALE], [0]))
    S.FullyConnectedOptionsStart(b)
    S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)
    S.FullyConnectedOptionsAddWeightsFormat(b, 0)
    S.FullyConnectedOptionsAddKeepNumDims(b, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
    opt = S.FullyConnectedOptionsEnd(b)
    op = bo.operator(b, 0, [0, 1, 2], [3], S.BuiltinOptions.FullyConnectedOptions, opt)
    blob = bo.finish(b, buffers,
                     [bo.opcode(b, S.BuiltinOperator.FULLY_CONNECTED)],
                     [t_in, t_w, t_b, t_out], [op], [0], [3],
                     "bbdiff FC %dx%dx%d int8" % (m, k, n))
    meta = {
        "op": "FULLY_CONNECTED", "m": m, "k": k, "n": n,
        "shapes": {"input": [m, k], "weights": [n, k], "bias": [n], "output": [m, n]},
        "dtype": "int8",
        "weights": spec.get("weights") or {"kind": "base"},
        "weight_bytes": int(w_q.nbytes),
        "bias_bytes": int(b_q.nbytes),
        "quant": {"input": A_SCALE, "weights": W_SCALE, "bias": B_SCALE, "output": O_SCALE},
        "tflite_bytes": len(blob),
        "sha256_tflite": hashlib.sha256(blob).hexdigest(),
    }
    return blob, meta


# ----------------------------------------------------------------- 编译（唯一入口 compile.sh）
def compile_once(model_bytes, build_dir, extra_flags=()):
    """把模型写到固定路径再编译一次；返回 (dla_bytes|None, rc, log, cmd, seconds)。

    固定路径是刻意的：.dla 里嵌了模型文件名/路径，换路径就等于换了输入。
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    tflite = build_dir / "model.tflite"
    dla = build_dir / "model.dla"
    tflite.write_bytes(model_bytes)
    if dla.exists():
        dla.unlink()
    flags = [str(f).replace("{build}", str(build_dir)) for f in extra_flags]
    cmd = ["bash", "tools/compile.sh", str(tflite), str(dla)] + flags
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    took = time.time() - started
    log_path = build_dir / (dla.stem + ".compile.log")
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    data = dla.read_bytes() if dla.exists() else None
    return data, proc.returncode, log, " ".join(cmd), took


# ----------------------------------------------------------------- 差分
def diff_offsets(a, b):
    """逐字节差分：返回 (不同的偏移集合, 长度差 a-b)。

    长度不同时，公共前缀逐字节比，尾部多出来的字节全部算「不同」。
    """
    n = min(len(a), len(b))
    offs = {i for i in range(n) if a[i] != b[i]}
    if len(a) != len(b):
        offs.update(range(n, max(len(a), len(b))))
    return offs, len(a) - len(b)


def runs_of(offsets):
    """偏移集合 -> 连续区间列表 [[start, end], ...]。"""
    if not offsets:
        return []
    out = []
    order = sorted(offsets)
    s = p = order[0]
    for i in order[1:]:
        if i == p + 1:
            p = i
            continue
        out.append([s, p])
        s = p = i
    out.append([s, p])
    return out


def merge_runs(runs, gap=0):
    """把间隔 <= gap 的相邻区间并成一个 —— 差分区里的「巧合相等」不该把区间切碎。"""
    if not runs:
        return []
    gap = max(0, int(gap))
    out = [list(runs[0])]
    for s, e in runs[1:]:
        if s - out[-1][1] - 1 <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def size_of_runs(runs):
    return sum(e - s + 1 for s, e in runs)


def as_mask(runs, size):
    m = bytearray(size)
    for s, e in runs:
        for i in range(s, min(e + 1, size)):
            m[i] = 1
    return m


# ----------------------------------------------------------------- 实验
class Variant(object):
    def __init__(self, spec, exp_dir):
        self.id = spec["id"]
        self.spec = spec
        self.note = spec.get("note", "")
        self.flags = spec.get("flags") or []
        self.dir = exp_dir / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.blob = None
        self.meta = {}
        self.attempts = []      # 所有编译尝试（含被判成另一「产物类别」的）
        self.runs = []          # 同一「产物类别」里保留的 repeats 次（差分只用它们）
        self.classes = {}       # 类别 -> 尝试次数
        self.dropped = 0        # 落在非主类别、被排除的尝试数
        self.error = ""

    @property
    def ok(self):
        return bool(self.runs) and all(r["rc"] == 0 and r["dla"] for r in self.runs)

    def build(self):
        try:
            self.blob, self.meta = build_model(self.spec["model"])
        except Exception as e:                                  # 失败也留证据
            self.error = "%s: %s" % (type(e).__name__, e)
            (self.dir / "model.error.txt").write_text(self.error + "\n")
            return
        (self.dir / "model.tflite").write_bytes(self.blob)
        (self.dir / "model.json").write_text(json.dumps(self.meta, indent=2) + "\n")

    def compile(self, build_dir, repeats, slack=2, threshold=0.05):
        """重复编译 attempts 次，再把它们按「产物类别」分簇，只保留主类别的前 repeats 次。

        为什么要分簇：同一份输入重复编译，产物不是逐字节稳定的；而且偶尔会整块换一种排布
        （实测跨运行能差 4 KB 以上，占产物一半）。把那种整块换类当成「噪声区」会把噪声区
        撑到半个文件，后面所有差分都失去意义 —— 所以先分簇，跨类的差异单列成证据，不混进噪声区。
        """
        if self.blob is None:
            return
        attempts = repeats + max(0, int(slack))
        for i in range(1, attempts + 1):
            data, rc, log, cmd, took = compile_once(self.blob, build_dir, self.flags)
            (self.dir / ("attempt%d.compile.log" % i)).write_text(log)
            if data:
                (self.dir / ("attempt%d.dla" % i)).write_bytes(data)
            else:
                self.error = self.error or "attempt %d 没有产出 .dla（rc=%s）" % (i, rc)
            self.attempts.append({
                "attempt": i, "rc": rc, "cmd": cmd, "seconds": round(took, 2),
                "bytes": len(data) if data else 0,
                "sha256": hashlib.sha256(data).hexdigest() if data else "",
                "dla": data, "class": None, "kept": False,
            })
        self._select_class(repeats, threshold)

    def _select_class(self, repeats, threshold):
        """按两两差分距离分簇（并查集），主类别 = 成员最多的簇（并列取最早的）。"""
        good = [a for a in self.attempts if a["dla"]]
        if not good:
            return
        limit = max(64, int(threshold * max(a["bytes"] for a in good)))
        parent = list(range(len(good)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)

        for i in range(len(good)):
            for j in range(i + 1, len(good)):
                offs, _ = diff_offsets(good[i]["dla"], good[j]["dla"])
                if len(offs) <= limit:
                    union(i, j)
        groups = {}
        for i in range(len(good)):
            groups.setdefault(find(i), []).append(i)
        # 主类别：成员最多；并列取成员最早出现的那簇
        modal_key = sorted(groups, key=lambda k: (-len(groups[k]), min(groups[k])))[0]
        self.classes = {("class%d" % n): len(g) for n, g in enumerate(
            sorted(groups.values(), key=lambda g: min(g)))}
        kept = [good[i] for i in sorted(groups[modal_key])][:repeats]
        kept_ids = {id(a) for a in kept}
        self.dropped = len(good) - len(kept)
        for a in self.attempts:
            a["kept"] = id(a) in kept_ids
            a["class"] = "modal" if a["kept"] else ("off-class" if a["dla"] else "failed")
        self.runs = kept
        for n, a in enumerate(kept, 1):
            a["run"] = n

    def write_evidence(self):
        lines = ["# %s" % self.id, "note: %s" % (self.note or "-"),
                 "flags: %s" % (" ".join(self.flags) if self.flags else "-"),
                 "model.tflite bytes=%d sha256=%s" % (
                     self.meta.get("tflite_bytes", 0), self.meta.get("sha256_tflite", "-")),
                 "weight_bytes=%d bias_bytes=%d" % (
                     self.meta.get("weight_bytes", 0), self.meta.get("bias_bytes", 0)),
                 "attempts=%d  classes=%s  dropped(off-class)=%d" % (
                     len(self.attempts), json.dumps(self.classes), self.dropped),
                 "kept runs: %s" % (", ".join("attempt%d" % r["attempt"] for r in self.runs) or "-")]
        for a in self.attempts:
            lines.append("attempt%d [%s%s]: rc=%s bytes=%d sha256=%s seconds=%.2f" % (
                a["attempt"], a["class"], "" if a["kept"] else "",
                a["rc"], a["bytes"], a["sha256"] or "-", a["seconds"]))
            lines.append("  cmd: %s" % a["cmd"])
        if self.error:
            lines.append("error: %s" % self.error)
        (self.dir / "evidence.txt").write_text("\n".join(lines) + "\n")
        return {"id": self.id, "note": self.note, "flags": self.flags,
                "model": self.meta, "error": self.error, "ok": self.ok,
                "classes": self.classes, "dropped_off_class": self.dropped,
                "attempts": [{k: v for k, v in a.items() if k != "dla"} for a in self.attempts],
                "runs": [{k: v for k, v in r.items() if k != "dla"} for r in self.runs]}


def measure_noise(variants, upto=None):
    """噪声区：同一变体重复编译之间**不稳定**的偏移集合（所有变体、所有配对的并集）。

    这就是「已知噪声区」的定义：只要两次相同的编译在这里给出不同字节，差分就不能用它下结论。

    upto 是「标定窗口」：只用前 upto 次编译来定噪声区，剩下的那次留作 holdout ——
    否则「第 3 次与第 1 次除噪声区外相同」就是同义反复（噪声区本来就是从它推出来的）。
    """
    noise = set()
    detail = {}
    for v in variants:
        runs = v.runs[:upto] if upto else v.runs
        if len(runs) < 2:
            continue
        local = set()
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a, b = runs[i]["dla"], runs[j]["dla"]
                if a is None or b is None:
                    continue
                offs, _ = diff_offsets(a, b)
                local |= offs
        detail[v.id] = len(local)
        noise |= local
    return noise, detail


def classify(offsets, noise_mask):
    inside = [i for i in offsets if i < len(noise_mask) and noise_mask[i]]
    outside = [i for i in offsets if not (i < len(noise_mask) and noise_mask[i])]
    return inside, outside


# ----------------------------------------------------------------- 判定（数据驱动）
def run_check(check, ctx):
    """check 是矩阵里的一行数据；返回带 ok / as_expected 的结果 dict。

    kind:
      diff_outside_noise  两次编译的差分离噪后必须还有足够变化（该变的必须变）
      repeatable          某变体第 holdout_run 次编译必须与第 1 次在噪声区外逐字节相同（不该变的必须不变）
      noise_budget        噪声区本身必须小（否则它把差分吃掉，后面的结论都不可信）
      diff_within_zone    差分离噪后必须落进指定 zone（区域分层没写错）
      size_delta          两变体产物大小差必须落在「某个字节量」的倍数区间内
    expect: pass（默认）| fail —— 后者是对照项，用来证明判定不是恒真。
    """
    kind = check["kind"]
    exp = check.get("expect", "pass")
    res = {"id": check["id"], "kind": kind, "expect": exp, "note": check.get("note", "")}
    try:
        if kind == "diff_outside_noise":
            d = ctx["diff"](check["a"], check["b"])
            total = len(d["offsets"])
            outside = d["outside"]
            share = (d["inside"] / total) if total else 0.0
            ok = (len(outside) >= int(check.get("min_bytes", 1))
                  and (len(outside) / max(d["size_a"], 1) >= float(check.get("min_fraction", 0.0)))
                  and share <= float(check.get("max_noise_share", 1.0)))
            res.update({"a": check["a"], "b": check["b"], "diff_bytes": total,
                        "outside_noise": len(outside), "inside_noise": d["inside"],
                        "noise_share": round(share, 4),
                        "size_a": d["size_a"], "size_b": d["size_b"],
                        "runs_outside": runs_of(outside)[:12]})
        elif kind == "repeatable":
            v = ctx["variants"][check["variant"]]
            hold = int(check.get("holdout_run", 2))
            if len(v.runs) < hold or v.runs[0]["dla"] is None or v.runs[hold - 1]["dla"] is None:
                raise ValueError("变体 %s 的 run1/run%d 不存在" % (v.id, hold))
            offs, dlen = diff_offsets(v.runs[0]["dla"], v.runs[hold - 1]["dla"])
            inside, outside = classify(offs, ctx["noise_mask"])
            ok = len(outside) <= int(check.get("allow_outside_bytes", 0))
            res.update({"variant": v.id, "holdout_run": hold, "diff_bytes": len(offs),
                        "outside_noise": len(outside), "inside_noise": len(inside),
                        "size_delta": dlen, "runs_outside": runs_of(outside)[:12]})
        elif kind == "diff_within_zone":
            d = ctx["diff"](check["a"], check["b"])
            zone = ctx["zones"].get(check["zone"])
            if zone is None:
                raise ValueError("zone %r 没有标定出来" % check["zone"])
            zmask = as_mask(zone["runs"], max(d["size_a"], d["size_b"]))
            esc = [i for i in d["outside"] if not (i < len(zmask) and zmask[i])]
            ok = len(esc) <= int(check.get("allow_outside_bytes", 0)) and len(d["outside"]) > 0
            res.update({"a": check["a"], "b": check["b"], "zone": check["zone"],
                        "zone_bytes": size_of_runs(zone["runs"]),
                        "diff_bytes": len(d["offsets"]), "outside_noise": len(d["outside"]),
                        "inside_noise": d["inside"], "escaped_zone": len(esc),
                        "escape_runs": runs_of(esc)[:12]})
        elif kind == "noise_budget":
            nb = ctx["noise_bytes"]
            size = max(ctx["max_size"], 1)
            share = nb / float(size)
            ok = share <= float(check.get("max_share", 0.05)) and 0 < nb < size
            res.update({"noise_bytes": nb, "max_size": size, "share": round(share, 4)})
        elif kind == "size_delta":
            a, b = ctx["variants"][check["a"]], ctx["variants"][check["b"]]
            sa, sb = a.runs[0]["bytes"], b.runs[0]["bytes"]
            delta = sa - sb
            ref = int(a.meta.get("weight_bytes", 0))
            lo = float(check.get("lo", 0.5)) * ref
            hi = float(check.get("hi", 1.5)) * ref
            ok = lo <= delta <= hi
            res.update({"a": a.id, "b": b.id, "size_a": sa, "size_b": sb, "delta": delta,
                        "weight_bytes": ref, "ratio": round(delta / ref, 3) if ref else None,
                        "window": [round(lo, 1), round(hi, 1)]})
        else:
            raise ValueError("check.kind 未知: %r" % kind)
    except Exception as e:
        res.update({"ok": False, "as_expected": False, "error": "%s: %s" % (type(e).__name__, e)})
        return res
    res["ok"] = ok
    res["as_expected"] = (ok if exp == "pass" else not ok)
    return res


# ----------------------------------------------------------------- 报告
def write_diff_file(path, a_id, b_id, d):
    lines = ["# diff %s vs %s" % (a_id, b_id),
             "size: %d vs %d (delta %+d)" % (d["size_a"], d["size_b"], d["size_b"] - d["size_a"]),
             "diff bytes=%d   inside noise=%d   outside noise=%d" % (
                 len(d["offsets"]), d["inside"], len(d["outside"])),
             "ranges (outside noise, top 40):"]
    for s, e in runs_of(d["outside"])[:40]:
        lines.append("  %6d..%-6d  len=%d" % (s, e, e - s + 1))
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="受控差分探查：编译 -> 一致性检查 -> 差分 -> 报告")
    ap.add_argument("--matrix", required=True, help="实验矩阵 JSON（数据；加变体只加一行）")
    ap.add_argument("--out-root", default="ops/_probe", help="产物根目录（缺省 ops/_probe）")
    ap.add_argument("--repeats", type=int, default=None, help="覆盖矩阵里的 repeats（每变体编译次数）")
    ap.add_argument("--only", default="", help="只跑这些变体（逗号分隔）")
    ap.add_argument("--report", default="", help="额外把自动报告写到这里（如 reports/xxx.md）")
    args = ap.parse_args(argv)

    matrix = json.loads(Path(args.matrix).read_text())
    experiment = matrix["experiment"]
    repeats = max(2, int(args.repeats or matrix.get("repeats", 3)))
    slack = int(matrix.get("class_slack", 2))
    threshold = float(matrix.get("class_threshold_share", 0.05))
    exp_dir = (Path(args.out_root) / experiment) if os.path.isabs(args.out_root) \
        else (REPO_ROOT / args.out_root / experiment)
    exp_dir.mkdir(parents=True, exist_ok=True)
    build_dir = exp_dir / "_build"          # 所有变体共用同一条编译路径（避免路径混进差分）
    diff_dir = exp_dir / "diff"
    diff_dir.mkdir(exist_ok=True)

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    specs = [v for v in matrix["variants"] if not only or v["id"] in only]

    print("[bbdiff] %s  repeats=%d  out=%s" % (experiment, repeats, exp_dir))
    variants = {}
    order = []
    for spec in specs:
        v = Variant(spec, exp_dir)
        v.build()
        v.compile(build_dir, repeats, slack=slack, threshold=threshold)
        variants[v.id] = v
        order.append(v.id)
        v.write_evidence()
        print("  %-14s tflite=%-5d dla=%s  attempts=%d kept=%d dropped=%d %s" % (
            v.id, v.meta.get("tflite_bytes", 0),
            "/".join(str(r["bytes"]) for r in v.runs) or "-",
            len(v.attempts), len(v.runs), v.dropped,
            "" if v.ok else "  FAIL(%s)" % (v.error or "rc")))

    # 一致性检查（先于差分）
    noise, per_variant_noise = measure_noise([variants[i] for i in order],
                                              upto=max(2, repeats - 1))
    noise_runs = runs_of(noise)
    max_size = max([r["bytes"] for v in variants.values() for r in v.runs] + [0])
    noise_mask = as_mask(noise_runs, max_size)
    (exp_dir / "noise.txt").write_text(
        "noise bytes=%d  runs=%d  (union over %d variants x %d kept repeats, "
        "calibration window = first %d)\n" % (
            len(noise), len(noise_runs), len(order), repeats, max(2, repeats - 1))
        + "per-variant noisy bytes: %s\n" % json.dumps(per_variant_noise)
        + "runs:\n" + "".join("  %6d..%-6d len=%d\n" % (s, e, e - s + 1) for s, e in noise_runs))

    # zone 标定（数据里声明 from/to）；差分会话内缓存
    zones = {}
    cache = {}

    def diff_of(a_id, b_id):
        key = (a_id, b_id)
        if key in cache:
            return cache[key]
        va, vb = variants[a_id], variants[b_id]
        if not va.ok or not vb.ok:
            raise ValueError("变体 %s/%s 编译失败，无法差分" % (a_id, b_id))
        a, b = va.runs[0]["dla"], vb.runs[0]["dla"]
        offs, dlen = diff_offsets(a, b)
        inside, outside = classify(offs, noise_mask)
        d = {"offsets": offs, "inside": len(inside), "outside": outside,
             "size_a": len(a), "size_b": len(b), "size_delta": dlen}
        cache[key] = d
        write_diff_file(diff_dir / ("%s__vs__%s.txt" % (a_id, b_id)), a_id, b_id, d)
        return d

    for name, spec in (matrix.get("zones") or {}).items():
        d = diff_of(spec["from"], spec["to"])
        zone_runs = merge_runs(runs_of(d["outside"]), int(spec.get("gap", 0)))
        zones[name] = {"runs": zone_runs, "bytes": size_of_runs(zone_runs),
                       "source": "%s vs %s（离噪）" % (spec["from"], spec["to"]),
                       "note": spec.get("note", "")}
        (exp_dir / ("zone-%s.txt" % name)).write_text(
            "zone %s: bytes=%d  source=%s\n%s\n" % (
                name, zones[name]["bytes"], zones[name]["source"], spec.get("note", ""))
            + "runs:\n" + "".join("  %6d..%-6d len=%d\n" % (s, e, e - s + 1) for s, e in zone_runs))

    ctx = {"variants": variants, "noise_mask": noise_mask, "zones": zones, "diff": diff_of,
           "noise_bytes": len(noise), "max_size": max_size}
    results = [run_check(c, ctx) for c in (matrix.get("checks") or [])]

    manifest = {
        "experiment": experiment,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "matrix": args.matrix,
        "repeats": repeats,
        "noise": {"bytes": len(noise), "runs": noise_runs,
                  "per_variant": per_variant_noise, "max_size": max_size},
        "zones": zones,
        "variants": [variants[i].write_evidence() for i in order],
        "checks": results,
        "ok": all(r["as_expected"] for r in results),
    }
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # 自动报告（人读）
    md = ["# %s —— 差分探查自动报告" % experiment, "",
          "生成：/usr/bin/python3 tools/bbdiff.py --matrix %s（WSL /usr/bin/python3）" % args.matrix,
          "时间：%s · 每变体保留 %d 次同类别编译（另跑 %d 次做类别校验）· 编译入口 tools/compile.sh"
          % (manifest["generated_at"], repeats, max(0, slack)),
          "", "## 变体", "",
          "| 变体 | tflite B | 权重 B | .dla B（保留各次） | rc | 尝试/保留/丢弃 | 噪声内字节 |",
          "|---|---|---|---|---|---|---|"]
    for i in order:
        v = variants[i]
        md.append("| %s | %d | %d | %s | %s | %d/%d/%d | %s |" % (
            v.id, v.meta.get("tflite_bytes", 0), v.meta.get("weight_bytes", 0),
            " / ".join(str(r["bytes"]) for r in v.runs) or "-",
            "/".join(str(r["rc"]) for r in v.runs),
            len(v.attempts), len(v.runs), v.dropped, per_variant_noise.get(v.id, "-")))
    md += ["", "## 已知噪声区（两次编译不稳定）", "",
           "共 **%d B** / %d 段（占最大产物 %.1f%%）；由每变体前 %d 次编译两两比对取并集"
           "（第 %d 次留作 holdout，见 case2）。逐段见 noise.txt。" % (
               len(noise), len(noise_runs), 100.0 * len(noise) / max(max_size, 1),
               max(2, repeats - 1), repeats),
           "", "| zone | 字节 | 来源 |", "|---|---|---|"]
    for name, z in zones.items():
        md.append("| %s | %d | %s |" % (name, z["bytes"], z["source"]))
    md += ["", "## 判定", "", "| check | kind | 结果 | 关键数字 |", "|---|---|---|---|"]
    for r in results:
        nums = {k: v for k, v in r.items()
                if k in ("diff_bytes", "outside_noise", "inside_noise", "noise_share",
                         "escaped_zone", "size_delta", "delta", "ratio", "zone_bytes",
                         "noise_bytes", "share")}
        md.append("| %s | %s | **%s** | %s |" % (
            r["id"], r["kind"], "PASS" if r["as_expected"] else "FAIL", json.dumps(nums)))
    md += ["", "## 单个差分", ""]
    for (a_id, b_id) in sorted(cache):
        d = cache[(a_id, b_id)]
        md.append("- %s vs %s：size %d -> %d（%+d），不同 %d B，噪声内 %d B，噪声外 %d B -> diff/%s__vs__%s.txt"
                  % (a_id, b_id, d["size_a"], d["size_b"], d["size_b"] - d["size_a"],
                     len(d["offsets"]), d["inside"], len(d["outside"]), a_id, b_id))
    md.append("")
    (exp_dir / "report.md").write_text("\n".join(md) + "\n")
    if args.report:
        Path(args.report).write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
