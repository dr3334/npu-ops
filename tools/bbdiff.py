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

06 起的三个口径变化（证据见 reports/compiler-blackbox-06.md）：
  a. **噪声按变体各自标定**（矩阵键 default_noise_scope / noise_scopes）：跨变体差分只豁免
     「参与这次差分的两个变体」自己的噪声；旧的「全体并集」保留为对照列，因为全局掩码会跨变体
     豁免真信号、也会替别人兜底假绿；
  b. **稳定字节口径**内置（stable_diff）：只在两侧都逐字节稳定的偏移上比较，04 §4.2 那个 1 B
     信号现在是矩阵里的一行数据（expect_offsets: [6343]），不再需要临时脚本；
  c. **排布类**：同一个模型连编多次时，常量块的落址会整块翻转（bias/weights 互换），跨了两种
     排布的线性差分是「同一组字节换位置」而不是内容变化 —— 工具比常量块落址签名，不同就剔出常量区。

新增的 check kind 都在 run_check 的 docstring 里；新增的可选矩阵字段（都向后兼容）：
  prewrite{}（编译前往 build 目录写辅助文件，喂 --dla-metadata）、pack{levels[]}（.dlb 打包等级）、
  default_noise_scope / noise_scopes（噪声口径）、flags[]（照旧）。

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
def _floats(spec):
    """按变体声明生成 FC 的浮点源数据（量化/转 fp16 之前的那一份）。

    weights.kind=base 的变体与 build_ops.py 的 fc 夹具同 seed、同张量命名，所以
    _codes() 出来的就是 ops/fc_1x64x64_int8/model.tflite 那一份夹具。
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
        raise ValueError("weights.kind 未知: %r" % kind)
    return a_f, w_f, b_f


def _codes(spec):
    """按变体声明生成 FC 的三个常量张量（输入码 / 权重码 / bias 码）。

    码值始终按 A_SCALE/W_SCALE/B_SCALE 这一档生成（与 build_ops.py 的 fc int8 一致）；
    **声明**出来的量化参数另外由 model.quant 决定（见 _declared_quant）——
    这样「只改量化配置」的变体可以保持码值逐字节不变，差异只能来自元数据。
    """
    import numpy as np
    a_f, w_f, b_f = _floats(spec)
    a_q = np.clip(np.round(a_f / A_SCALE), -127, 127).astype(np.int8)
    w_q = np.clip(np.round(w_f / W_SCALE), -127, 127).astype(np.int8)
    b_q = np.round(b_f / B_SCALE).astype(np.int32)
    return a_q, w_q, b_q


def _declared_quant(spec):
    """写进模型里的**声明**量化参数（int8 用）。

    model.quant = {"input": scale, "weights": scale, "output": scale}（都可缺省）：
    bias 的 scale 由 input*weights 推出；output 缺省 = bias*32（requant 倍率 1/32）。
    缺省全部不写时与票 03 的基准逐字节相同。
    """
    q = spec.get("quant") or {}
    qa = float(q.get("input", A_SCALE))
    qw = float(q.get("weights", W_SCALE))
    qb = qa * qw
    qo = float(q.get("output", qb * 32.0))
    return qa, qw, qb, qo


def build_model(spec):
    """写出 FULLY_CONNECTED 的 tflite 字节（复用 tools/build_ops.py 的 flatbuffer 助手，不改它）。

    model.dtype（可选）: "int8"（缺省）| "fp16" —— 票 04 的 FC-E 用来做 int8<->fp16 对照。
    """
    import flatbuffers
    sys.path.insert(0, str(TOOLS))
    import build_ops as bo            # 只 import，不改（公共工具）
    import numpy as np
    S = bo.S
    m, k, n = int(spec["m"]), int(spec["k"]), int(spec["n"])
    dtype = str(spec.get("dtype", "int8"))
    a_f, w_f, b_f = _floats(spec)
    b = flatbuffers.Builder(4096)
    buffers = [bo.buf_empty(b)]                       # 0: IO 占位（输入张量不带数据）
    if dtype == "fp16":
        a16 = a_f.astype(np.float16)
        w16 = w_f.astype(np.float16)
        b16 = b_f.astype(np.float16)
        buffers.append(bo.buf_data(b, w16))           # 1: 权重常量（fp16）
        buffers.append(bo.buf_data(b, b16))           # 2: bias 常量（fp16）
        buffers.append(bo.buf_empty(b))               # 3: 输出占位
        t_in = bo.tensor(b, "input", [m, k], bo.F16, 0)
        t_w = bo.tensor(b, "weights", [n, k], bo.F16, 1)
        t_b = bo.tensor(b, "bias", [n], bo.F16, 2)
        t_out = bo.tensor(b, "output", [m, n], bo.F16, 3)
        quant = {"input": None, "weights": None, "bias": None, "output": None}
        weight_bytes, bias_bytes = int(w16.nbytes), int(b16.nbytes)
    elif dtype == "int8":
        a_q, w_q, b_q = _codes(spec)
        qa, qw, qb, qo = _declared_quant(spec)
        buffers.append(bo.buf_data(b, w_q))           # 1: 权重常量
        buffers.append(bo.buf_data(b, b_q))           # 2: bias 常量
        buffers.append(bo.buf_empty(b))               # 3: 输出占位
        t_in = bo.tensor(b, "input", [m, k], bo.I8, 0, bo.quant_tensor(b, [qa], [0]))
        t_w = bo.tensor(b, "weights", [n, k], bo.I8, 1, bo.quant_tensor(b, [qw], [0]))
        t_b = bo.tensor(b, "bias", [n], bo.I32, 2, bo.quant_tensor(b, [qb], [0]))
        t_out = bo.tensor(b, "output", [m, n], bo.I8, 3, bo.quant_tensor(b, [qo], [0]))
        quant = {"input": qa, "weights": qw, "bias": qb, "output": qo}
        weight_bytes, bias_bytes = int(w_q.nbytes), int(b_q.nbytes)
    else:
        raise ValueError("model.dtype 未知: %r" % dtype)
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
                     "bbdiff FC %dx%dx%d %s" % (m, k, n, dtype))
    meta = {
        "op": "FULLY_CONNECTED", "m": m, "k": k, "n": n,
        "shapes": {"input": [m, k], "weights": [n, k], "bias": [n], "output": [m, n]},
        "dtype": dtype,
        "weights": spec.get("weights") or {"kind": "base"},
        "weight_bytes": weight_bytes,
        "bias_bytes": bias_bytes,
        "quant": quant,
        "quant_codes": ({"input": A_SCALE, "weights": W_SCALE, "bias": B_SCALE}
                        if dtype == "int8" else None),
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
        self.packs = {}         # .dlb 压缩等级 -> {bytes, rc, log}（spec.pack 声明时才跑）
        self.prewrite = spec.get("prewrite") or {}   # 每次编译前写进 build 目录的辅助文件



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

        spec.prewrite 是「编译前要落在 build 目录里的辅助文件」（06 的 --dla-metadata 用它喂
        一个 key:file）；写成数据而不是代码，加一个新开关不用动工具。
        """
        if self.blob is None:
            return
        build_dir.mkdir(parents=True, exist_ok=True)
        for name, content in self.prewrite.items():
            (build_dir / name).write_text(content)
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

    def constant_layout(self):
        """每个常量 buffer 落在产物的哪个偏移（精确子串优先，否则按字节多重集）。

        这是「排布类」的**可比较**签名：同一份模型连编多次时，常量块的先后顺序会整块翻转
        （06 §4 实测：同一个 fc_a，两次编译 bias@1671/weights@1951，另一次 weights@1671/bias@5791）。
        跨变体差分一旦跨了这两种排布，线性差分就是「同一组字节换了位置」，不是内容差异。
        """
        if getattr(self, "_layout", None) is not None:
            return self._layout
        try:
            import bbmap
            consts = bbmap.constant_buffers(self.blob)
        except Exception:
            self._layout = None
            return None
        blob = self.runs[0]["dla"] if self.runs else b""
        lay = []
        for name, data in consts:
            off = blob.find(data)
            how = "exact"
            if off < 0:
                hits = bbmap.find_multiset(blob, data)
                off = hits[0] if hits else None
                how = "multiset"
            lay.append({"tensor": name, "bytes": len(data), "offset": off, "how": how})
        self._layout = lay
        return lay

    def constant_area(self):
        """常量块整体占据的区间 [start, end]；定位不到就 None。"""
        lay = self.constant_layout()
        if not lay or any(r["offset"] is None for r in lay):
            return None
        return [min(r["offset"] for r in lay),
                max(r["offset"] + r["bytes"] - 1 for r in lay)]

    def pack(self, levels):
        """把保留的第一次产物打成 .dlb（每个等级一次），记下大小与 readdlb 的原文。

        .dlb 是容器层的变量（04 §5.4 点名没覆盖的那一项）：它不进 .dla 的字节差分，
        所以只有「打包后多大、readdlb 读到什么」这两个可测事实。
        """
        if not levels or not self.runs:
            return
        src = self.dir / ("attempt%d.dla" % self.runs[0]["attempt"])
        if not src.exists():
            return
        for lv in levels:
            out = self.dir / ("level%d.dlb" % int(lv))
            cmd = ["bash", "tools/pack.sh", str(int(lv)), str(src), str(out)]
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
            log = ""
            logp = self.dir / ("level%d.pack.log" % int(lv))
            if logp.exists():
                log = logp.read_text(errors="replace")
            self.packs[int(lv)] = {"bytes": out.stat().st_size if out.exists() else 0,
                                   "rc": proc.returncode, "cmd": " ".join(cmd)}
            if proc.returncode != 0:
                self.error = self.error or "pack level %s 失败（rc=%s）" % (lv, proc.returncode)

    def write_evidence(self):
        if getattr(self, "_ev", None) is not None:
            return self._ev          # 同一变体只写一次盘（调用方两处都要返回值的习惯导致的重复调用）
        lines = ["# %s" % self.id, "note: %s" % (self.note or "-"),
                 "flags: %s" % (" ".join(self.flags) if self.flags else "-"),
                 "model.tflite bytes=%d sha256=%s" % (
                     self.meta.get("tflite_bytes", 0), self.meta.get("sha256_tflite", "-")),
                 "weight_bytes=%d bias_bytes=%d" % (
                     self.meta.get("weight_bytes", 0), self.meta.get("bias_bytes", 0)),
                 "attempts=%d  classes=%s  dropped(off-class)=%d" % (
                     len(self.attempts), json.dumps(self.classes), self.dropped),
                 "kept runs: %s" % (", ".join("attempt%d" % r["attempt"] for r in self.runs) or "-")]
        if self.packs:
            lines.append(".dlb packs: %s" % json.dumps(
                {str(k): v["bytes"] for k, v in sorted(self.packs.items())}))
        for a in self.attempts:
            lines.append("attempt%d [%s%s]: rc=%s bytes=%d sha256=%s seconds=%.2f" % (
                a["attempt"], a["class"], "" if a["kept"] else " [off-class]",
                a["rc"], a["bytes"], a["sha256"] or "-", a["seconds"]))
            lines.append("  cmd: %s" % a["cmd"])
        if self.error:
            lines.append("error: %s" % self.error)
        (self.dir / "evidence.txt").write_text("\n".join(lines) + "\n")
        self._ev = {"id": self.id, "note": self.note, "flags": self.flags,
                "model": self.meta, "error": self.error, "ok": self.ok,
                "classes": self.classes, "dropped_off_class": self.dropped,
                "packs": {str(k): v for k, v in sorted(self.packs.items())},
                "attempts": [{k: v for k, v in a.items() if k != "dla"} for a in self.attempts],
                "runs": [{k: v for k, v in r.items() if k != "dla"} for r in self.runs]}
        return self._ev


def measure_noise(variants, upto=None):
    """噪声区：同一变体重复编译之间**不稳定**的偏移集合。

    返回 (并集, 每变体集合)：并集是 03/04 的旧口径（跨变体豁免同一批偏移），每变体集合是 06 起的新口径。
    **每变体口径**的理由：全局并集会让 A 的抖动在 B 的差分里也被豁免 —— 只要某个变体在偏移 X 上抖过一次，
    所有其它变体的真实变化在 X 上就被一起免掉（04 §5.3 那次 1 B 漏网就是这么来的：噪声区没覆盖到，
    却靠 allow_outside_bytes 余量遮住）。各变体自己标定以后，「豁免」只能由参与这次差分的两个变体给出。

    upto 是「标定窗口」：只用前 upto 次编译来定噪声区，剩下的那次留作 holdout ——
    否则「第 3 次与第 1 次除噪声区外相同」就是同义反复（噪声区本来就是从它推出来的）。
    """
    noise = set()
    detail = {}
    per_variant = {}
    for v in variants:
        runs = v.runs[:upto] if upto else v.runs
        local = set()
        if len(runs) >= 2:
            for i in range(len(runs)):
                for j in range(i + 1, len(runs)):
                    a, b = runs[i]["dla"], runs[j]["dla"]
                    if a is None or b is None:
                        continue
                    offs, _ = diff_offsets(a, b)
                    local |= offs
        detail[v.id] = len(local)
        per_variant[v.id] = local
        noise |= local
    return noise, detail, per_variant


def stable_offsets(runs):
    """**稳定字节**口径（04 §4.2 的做法，本文件自 06 起内置）：

    一个变体的若干次编译里取值**逐字节一致**的偏移集合 —— 只有这些偏移上的跨变体差异才可能是真信号，
    其余偏移（UUID、索引字节、整块重排）一律不参与判定。返回 (偏移集合, 可比前缀长度)。
    """
    blobs = [r["dla"] for r in runs if r["dla"] is not None]
    if len(blobs) < 2:
        return None, 0
    n = min(len(b) for b in blobs)
    stable = set()
    for i in range(n):
        v = blobs[0][i]
        if all(b[i] == v for b in blobs[1:]):
            stable.add(i)
    return stable, n


def multiset_distance(a, b, s, e):
    """[s, e] 区间内两侧字节**多重集**的距离：有多少个字节在对面找不到配对。

    0 表示「同一组字节换了顺序」= 整块重排（不是内容变化）。04 §4.1 的常量块顺序翻转就是这类。
    """
    ca, cb = {}, {}
    for i in range(s, min(e + 1, len(a))):
        ca[a[i]] = ca.get(a[i], 0) + 1
    for i in range(s, min(e + 1, len(b))):
        cb[b[i]] = cb.get(b[i], 0) + 1
    diff = 0
    for k in set(ca) | set(cb):
        diff += abs(ca.get(k, 0) - cb.get(k, 0))
    return diff // 2


def split_reorder(a, b, runs, min_run=256, max_share=0.02):
    """把差分区间分成「整块重排」与「真内容变化」两类，返回 (reorder_runs, signal_runs)。

    判据：一段 >= min_run 的差分里，多重集距离 <= max(64, max_share*长度) —— 也就是这段字节
    在对面**一个不少地存在**、只是换了位置。这不是内容变化，不能算信号（04 §4.2 的 1660..7634 就是它）。
    """
    reorder, signal = [], []
    for s, e in runs:
        n = e - s + 1
        if n >= min_run and multiset_distance(a, b, s, e) <= max(64, int(max_share * n)):
            reorder.append([s, e])
        else:
            signal.append([s, e])
    return reorder, signal


def cmdline_region(blob):
    """产物里那段**嵌进去的完整命令行**的区间 [start, end]（可打印 ASCII 的连续串，含 --disallow-bridge）。

    用途：判断一个「只加开关」的变体到底是改了产物内容，还是只改了这段回显文本（06 §2.3）。
    """
    i = blob.find(b"--disallow-bridge")
    if i < 0:
        return None
    s = i
    while s > 0 and 32 <= blob[s - 1] <= 126:
        s -= 1
    e = i
    while e < len(blob) and 32 <= blob[e] <= 126:
        e += 1
    return [s, e - 1]


def classify(offsets, noise_mask):
    inside = [i for i in offsets if i < len(noise_mask) and noise_mask[i]]
    outside = [i for i in offsets if not (i < len(noise_mask) and noise_mask[i])]
    return inside, outside


# ----------------------------------------------------------------- 判定（数据驱动）
SCOPE_INDEPENDENT = {"size_delta", "stable_diff", "flag_payload", "pack_size"}


def mask_for(ctx, a_id, b_id, scope):
    """这次差分该豁免哪些偏移。

    scope="union"       ：所有变体噪声的并集（03/04 的旧口径，保留只为对照）
    scope="per-variant" ：只豁免**参与这次差分的两个变体**自己的噪声（06 起的缺省口径）
    """
    if scope == "union":
        return ctx["noise_mask"]
    m = bytearray(max(1, ctx["max_size"]))
    for vid in {a_id, b_id}:
        for off in ctx["noise_per_variant"].get(vid, ()):
            if off < len(m):
                m[off] = 1
    return m


def stable_diff_of(ctx, a_id, b_id):
    """稳定字节口径的差分（04 §4.2 的做法，内置）。返回带 reorder/signal 拆分的 dict。"""
    key = ("stable", a_id, b_id)
    cache = ctx["stable_cache"]
    if key in cache:
        return cache[key]
    va, vb = ctx["variants"][a_id], ctx["variants"][b_id]
    if not va.runs or not vb.runs:
        raise ValueError("变体 %s/%s 没有保留的编译" % (a_id, b_id))
    sa, na = stable_offsets(va.runs)
    sb, nb = stable_offsets(vb.runs)
    if sa is None or sb is None:
        raise ValueError("变体 %s/%s 的保留编译不足 2 次，无法判稳定字节" % (a_id, b_id))
    n = min(na, nb)
    a, b = va.runs[0]["dla"], vb.runs[0]["dla"]
    offs = [i for i in range(n) if (i in sa and i in sb and a[i] != b[i])]
    runs = runs_of(offs)
    reorder, signal = split_reorder(a, b, runs)
    # 排布类：两侧常量块的落址不同时，常量区里那段「同一组字节换了位置」不能算信号，
    # 单独剔出来记在 excluded_area_bytes 里（06 §4 的 layout 类，能差 4 KB）。
    la, lb = va.constant_layout(), vb.constant_layout()
    same_layout = la is not None and lb is not None and \
        [(r["tensor"], r["offset"]) for r in la] == [(r["tensor"], r["offset"]) for r in lb]
    area = None
    if not same_layout:
        areas = [x for x in (va.constant_area(), vb.constant_area()) if x]
        if areas:
            area = [min(x[0] for x in areas), max(x[1] for x in areas)]
    excluded = 0
    if area is not None:
        keep = []
        for s, e in signal:
            lo, hi = max(s, area[0]), min(e, area[1])
            excluded += max(0, hi - lo + 1)
            if s < area[0]:
                keep.append([s, min(e, area[0] - 1)])
            if e > area[1]:
                keep.append([max(s, area[1] + 1), e])
        signal = keep
    d = {"a": a_id, "b": b_id, "size_a": len(a), "size_b": len(b),
         "stable_a": len(sa), "stable_b": len(sb), "comparable": n,
         "stable_diff_offsets": offs, "diff_bytes": len(offs),
         "reorder_runs": reorder, "reorder_bytes": size_of_runs(reorder),
         "signal_runs": signal, "signal_offsets": [i for s, e in signal for i in range(s, e + 1)],
         "signal_bytes": size_of_runs(signal),
         "layout_a": la, "layout_b": lb, "same_layout": same_layout,
         "constant_area": area, "excluded_area_bytes": excluded,
         "tail_uncomparable": max(len(a), len(b)) - n}
    ctx["stable_cache"][key] = d
    return d


def eval_check(check, ctx, scope):
    """按指定噪声口径跑一条 check；返回数字 + ok（不处理 expect）。"""
    kind = check["kind"]
    out = {}
    try:
        if kind == "diff_outside_noise":
            d = ctx["diff"](check["a"], check["b"], scope)
            total = len(d["offsets"])
            outside = d["outside"]
            share = (d["inside"] / total) if total else 0.0
            ok = (len(outside) >= int(check.get("min_bytes", 1))
                  and (len(outside) / max(d["size_a"], 1) >= float(check.get("min_fraction", 0.0)))
                  and share <= float(check.get("max_noise_share", 1.0)))
            out.update({"a": check["a"], "b": check["b"], "diff_bytes": total,
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
            inside, outside = classify(offs, mask_for(ctx, v.id, v.id, scope))
            ok = len(outside) <= int(check.get("allow_outside_bytes", 0))
            out.update({"variant": v.id, "holdout_run": hold, "diff_bytes": len(offs),
                        "outside_noise": len(outside), "inside_noise": len(inside),
                        "noise_own_bytes": len(ctx["noise_per_variant"].get(v.id, ())),
                        "size_delta": dlen, "runs_outside": runs_of(outside)[:12]})
        elif kind == "diff_within_zone":
            d = ctx["diff"](check["a"], check["b"], scope)
            zone = ctx["zones"].get(check["zone"])
            if zone is None:
                raise ValueError("zone %r 没有标定出来" % check["zone"])
            zmask = as_mask(zone["runs"], max(d["size_a"], d["size_b"]))
            esc = [i for i in d["outside"] if not (i < len(zmask) and zmask[i])]
            ok = len(esc) <= int(check.get("allow_outside_bytes", 0)) and len(d["outside"]) > 0
            out.update({"a": check["a"], "b": check["b"], "zone": check["zone"],
                        "zone_bytes": size_of_runs(zone["runs"]),
                        "diff_bytes": len(d["offsets"]), "outside_noise": len(d["outside"]),
                        "inside_noise": d["inside"], "escaped_zone": len(esc),
                        "escape_runs": runs_of(esc)[:12]})
        elif kind == "noise_budget":
            if scope == "union":
                nb = ctx["noise_bytes"]
            else:
                nb = max([0] + [len(s) for s in ctx["noise_per_variant"].values()])
            size = max(ctx["max_size"], 1)
            share = nb / float(size)
            ok = share <= float(check.get("max_share", 0.05)) and 0 < nb < size
            out.update({"noise_bytes": nb, "max_size": size, "share": round(share, 4)})
        elif kind == "size_delta":
            a, b = ctx["variants"][check["a"]], ctx["variants"][check["b"]]
            sa, sb = a.runs[0]["bytes"], b.runs[0]["bytes"]
            delta = sa - sb
            ref = int(a.meta.get("weight_bytes", 0))
            lo = float(check.get("lo", 0.5)) * ref
            hi = float(check.get("hi", 1.5)) * ref
            ok = lo <= delta <= hi
            out.update({"a": a.id, "b": b.id, "size_a": sa, "size_b": sb, "delta": delta,
                        "weight_bytes": ref, "ratio": round(delta / ref, 3) if ref else None,
                        "window": [round(lo, 1), round(hi, 1)]})
        elif kind == "stable_diff":
            d = stable_diff_of(ctx, check["a"], check["b"])
            want = check.get("expect_offsets")
            sig = set(d["signal_offsets"])
            ok = d["signal_bytes"] >= int(check.get("min_bytes", 1))
            if check.get("max_bytes") is not None:
                ok = ok and d["signal_bytes"] <= int(check["max_bytes"])
            if want is not None:
                ok = ok and sig == set(int(x) for x in want)
            out.update({"a": d["a"], "b": d["b"], "diff_bytes": d["diff_bytes"],
                        "stable_diff_bytes": d["diff_bytes"], "reorder_bytes": d["reorder_bytes"],
                        "signal_bytes": d["signal_bytes"],
                        "signal_runs": d["signal_runs"][:12],
                        "reorder_runs": d["reorder_runs"][:6],
                        "stable_offsets": [d["stable_a"], d["stable_b"]],
                        "same_layout": d["same_layout"], "excluded_area_bytes": d["excluded_area_bytes"],
                        "expect_offsets": want, "tail_uncomparable": d["tail_uncomparable"]})
            if want is not None:
                out["signal_offsets_match"] = sorted(sig) == sorted(int(x) for x in want)
        elif kind == "flag_payload":
            # 「只加开关」的变体：把两侧产物里嵌的**命令行回显整段抹掉**，再比内容。
            # 用字节直方图而不是逐字节：整块换排布（同一组字节换了位置）不算内容变化 ——
            # 这正是本口径要扛住的假信号（06 §4 的 layout 类，一次能差 4 KB）。
            va, vb = ctx["variants"][check["a"]], ctx["variants"][check["b"]]
            a, b = va.runs[0]["dla"], vb.runs[0]["dla"]
            ra, rb = cmdline_region(a), cmdline_region(b)
            sa, na = stable_offsets(va.runs)
            sb, nb = stable_offsets(vb.runs)
            big = 1 << 60
            cut = min([na, nb] + [r[0] for r in (ra, rb) if r is not None])
            # 只在「两侧都稳定、且都在命令行回显之前」的偏移上比字节多重集：
            # 这样既排掉每次编译都变的 UUID/索引字节，也排掉整块重排，剩下的差异只能是内容。
            delta = len(b) - len(a)
            diff = [i for i in range(cut)
                    if sa is not None and sb is not None and i in sa and i in sb and a[i] != b[i]]
            # 只改了长度时，产物里那些「写着自己多大 / 相对文件尾的偏移」的字段必然跟着位移，
            # 位移量恰好等于 size delta（04 §4.1 观测到的那类字段）。它们不是内容变化，单列出来。
            size_fields, payload = [], []
            for i in diff:
                d = (b[i] - a[i]) % 256
                if i == 0 or d == (delta % 256):
                    size_fields.append(i)
                else:
                    payload.append(i)
            # 判定用 body_size_delta：把两侧的命令行回显长度扣掉以后，产物本体长了/短了多少。
            # 它不受每次编译的抖动影响（抖动不改长度），也不受整块重排影响 —— 只有真正写进
            # 产物内容的开关才会让它非零。payload_bytes 只作为诊断给出（可能混进少量抖动字节）。
            la = None if ra is None else ra[1] - ra[0] + 1
            lb = None if rb is None else rb[1] - rb[0] + 1
            body_delta = (len(b) - (lb or 0)) - (len(a) - (la or 0))
            extra = check.get("expected_extra_bytes")
            ok = (abs(body_delta) <= int(check.get("allow_payload_bytes", 0))
                  and (extra is None or delta == int(extra)))
            out.update({"a": va.id, "b": vb.id, "size_delta": delta,
                        "expected_extra_bytes": extra,
                        "cmdline_bytes": [la, lb], "body_size_delta": body_delta,
                        "payload_bytes": len(payload), "size_field_bytes": len(size_fields),
                        "payload_runs": runs_of(payload)[:8], "compared_offsets": cut,
                        "stable_compared": len([i for i in range(cut)
                                                if sa is not None and sb is not None
                                                and i in sa and i in sb])})
        elif kind == "pack_size":
            v = ctx["variants"][check["variant"]]
            lo, hi = int(check.get("level_low", 0)), int(check.get("level_high", 9))
            pl = v.packs.get(lo, {})
            ph = v.packs.get(hi, {})
            if not pl.get("bytes") or not ph.get("bytes"):
                raise ValueError("变体 %s 缺 .dlb 打包结果（levels %s/%s）" % (v.id, lo, hi))
            ratio = ph["bytes"] / float(pl["bytes"])
            ok = ratio <= float(check.get("max_ratio", 0.85)) and ph["bytes"] < pl["bytes"]
            out.update({"variant": v.id, "level_low": lo, "level_high": hi,
                        "dlb_low": pl["bytes"], "dlb_high": ph["bytes"],
                        "ratio": round(ratio, 3), "dla_bytes": v.runs[0]["bytes"]})
        else:
            raise ValueError("check.kind 未知: %r" % kind)
    except Exception as e:
        out.update({"ok": False, "error": "%s: %s" % (type(e).__name__, e)})
        return out
    out["ok"] = bool(ok)
    return out


def run_check(check, ctx):
    """check 是矩阵里的一行数据；返回带 ok / as_expected 的结果 dict。

    kind:
      diff_outside_noise  两次编译的差分离噪后必须还有足够变化（该变的必须变）
      repeatable          某变体第 holdout_run 次编译必须与第 1 次在噪声区外逐字节相同（不该变的必须不变）
      noise_budget        噪声区本身必须小（否则它把差分吃掉，后面的结论都不可信）
      diff_within_zone    差分离噪后必须落进指定 zone（区域分层没写错）
      size_delta          两变体产物大小差必须落在「某个字节量」的倍数区间内
      stable_diff         稳定字节口径：两侧都逐字节稳定的偏移上，取值的差异才是信号（06 新增）
      flag_payload        只加开关的变体：差异必须全落在「嵌进产物的命令行回显」里（06 新增）
      pack_size           .dlb 压缩等级确实压缩（06 新增）

    scope: 噪声口径（union / per-variant），缺省取矩阵的 default_noise_scope；
           除三种用噪声掩码判定的 kind 以外，其余 kind 与口径无关，只跑一次。
    expect: pass（默认）| fail —— 后者是对照项，用来证明判定不是恒真。
    """
    kind = check["kind"]
    exp = check.get("expect", "pass")
    declared = check.get("scope", ctx["default_scope"])
    scopes = [declared] if kind in SCOPE_INDEPENDENT else ctx["scopes"]
    res = {"id": check["id"], "kind": kind, "expect": exp, "note": check.get("note", ""),
           "scope": declared}
    by_scope = {sc: eval_check(check, ctx, sc) for sc in scopes}
    main = by_scope[declared]
    keys = ("diff_bytes", "outside_noise", "inside_noise", "noise_share", "stable_diff_bytes",
            "reorder_bytes", "signal_bytes", "payload_bytes", "size_delta", "delta", "error")
    for k, v in main.items():
        if k != "ok" and k not in res:
            res[k] = v
    res["ok"] = main["ok"]
    res["as_expected"] = main["ok"] if exp == "pass" else (not main["ok"])
    if len(by_scope) > 1:
        res["by_scope"] = {sc: dict({"ok": d["ok"]},
                                    **{k: d[k] for k in keys if k in d})
                           for sc, d in by_scope.items()}
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
        if spec.get("pack"):
            v.pack(spec["pack"].get("levels") or [0, 9])
        variants[v.id] = v
        order.append(v.id)
        v.write_evidence()
        print("  %-14s tflite=%-5d dla=%s  attempts=%d kept=%d dropped=%d %s" % (
            v.id, v.meta.get("tflite_bytes", 0),
            "/".join(str(r["bytes"]) for r in v.runs) or "-",
            len(v.attempts), len(v.runs), v.dropped,
            "" if v.ok else "  FAIL(%s)" % (v.error or "rc")))

    # 一致性检查（先于差分）：并集口径（旧）+ 每变体口径（新）
    calibration = max(2, repeats - 1)
    noise, per_variant_noise, noise_per_variant = measure_noise(
        [variants[i] for i in order], upto=calibration)
    noise_runs = runs_of(noise)
    max_size = max([r["bytes"] for v in variants.values() for r in v.runs] + [0])
    noise_mask = as_mask(noise_runs, max_size)
    stable_per_variant = {}
    for i in order:
        st, n = stable_offsets(variants[i].runs)
        stable_per_variant[i] = {"stable_bytes": len(st) if st is not None else None,
                                 "comparable_prefix": n}
    (exp_dir / "noise.txt").write_text(
        "noise bytes=%d  runs=%d  (union over %d variants x %d kept repeats, "
        "calibration window = first %d)\n" % (
            len(noise), len(noise_runs), len(order), repeats, calibration)
        + "per-variant noisy bytes: %s\n" % json.dumps(per_variant_noise)
        + "per-variant stable bytes (all kept runs identical): %s\n" % json.dumps(stable_per_variant)
        + "union runs:\n"
        + "".join("  %6d..%-6d len=%d\n" % (s, e, e - s + 1) for s, e in noise_runs)
        + "per-variant runs:\n"
        + "".join("%s:\n" % vid
                  + "".join("  %6d..%-6d len=%d\n" % (s, e, e - s + 1)
                            for s, e in runs_of(noise_per_variant[vid]))
                  for vid in order))

    # zone 标定（数据里声明 from/to）；差分会话内缓存
    zones = {}
    cache = {}

    def diff_of(a_id, b_id, scope="per-variant"):
        key = (a_id, b_id, scope)
        if key in cache:
            return cache[key]
        va, vb = variants[a_id], variants[b_id]
        if not va.ok or not vb.ok:
            raise ValueError("变体 %s/%s 编译失败，无法差分" % (a_id, b_id))
        a, b = va.runs[0]["dla"], vb.runs[0]["dla"]
        offs, dlen = diff_offsets(a, b)
        mask = mask_for({"noise_mask": noise_mask, "noise_per_variant": noise_per_variant,
                         "max_size": max_size}, a_id, b_id, scope)
        inside, outside = classify(offs, mask)
        d = {"offsets": offs, "inside": len(inside), "outside": outside, "scope": scope,
             "size_a": len(a), "size_b": len(b), "size_delta": dlen}
        cache[key] = d
        write_diff_file(diff_dir / ("%s__vs__%s%s.txt" % (
            a_id, b_id, "" if scope == "union" else "." + scope)), a_id, b_id, d)
        return d

    for name, spec in (matrix.get("zones") or {}).items():
        try:
            d = diff_of(spec["from"], spec["to"])
        except (KeyError, ValueError) as e:      # --only 只跑了一部分变体：zone 标不出来就跳过
            print("  [zone %s] 跳过：%s" % (name, e))
            continue
        zone_runs = merge_runs(runs_of(d["outside"]), int(spec.get("gap", 0)))
        zones[name] = {"runs": zone_runs, "bytes": size_of_runs(zone_runs),
                       "source": "%s vs %s（离噪）" % (spec["from"], spec["to"]),
                       "note": spec.get("note", "")}
        (exp_dir / ("zone-%s.txt" % name)).write_text(
            "zone %s: bytes=%d  source=%s\n%s\n" % (
                name, zones[name]["bytes"], zones[name]["source"], spec.get("note", ""))
            + "runs:\n" + "".join("  %6d..%-6d len=%d\n" % (s, e, e - s + 1) for s, e in zone_runs))

    default_scope = str(matrix.get("default_noise_scope", "per-variant"))
    scopes = [str(s) for s in (matrix.get("noise_scopes") or ["union", "per-variant"])]
    if default_scope not in scopes:
        scopes.append(default_scope)
    ctx = {"variants": variants, "noise_mask": noise_mask, "zones": zones, "diff": diff_of,
           "noise_bytes": len(noise), "max_size": max_size,
           "noise_per_variant": {k: v for k, v in noise_per_variant.items()},
           "stable_cache": {}, "default_scope": default_scope, "scopes": scopes}
    results = [run_check(c, ctx) for c in (matrix.get("checks") or [])]

    manifest = {
        "experiment": experiment,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "matrix": args.matrix,
        "repeats": repeats,
        "noise": {"bytes": len(noise), "runs": noise_runs,
                  "per_variant": per_variant_noise, "max_size": max_size,
                  "per_variant_offsets": {k: sorted(v) for k, v in noise_per_variant.items()},
                  "stable_per_variant": stable_per_variant,
                  "default_scope": default_scope, "scopes": scopes},
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
          "噪声口径：判定用 **%s**；对照列同时给出 union（03/04 旧口径）" % default_scope,
          "", "## 变体", "",
          "| 变体 | tflite B | 权重 B | .dla B（保留各次） | rc | 尝试/保留/丢弃 | 自身噪声 B | 稳定字节 B |",
          "|---|---|---|---|---|---|---|---|"]
    for i in order:
        v = variants[i]
        md.append("| %s | %d | %d | %s | %s | %d/%d/%d | %s | %s |" % (
            v.id, v.meta.get("tflite_bytes", 0), v.meta.get("weight_bytes", 0),
            " / ".join(str(r["bytes"]) for r in v.runs) or "-",
            "/".join(str(r["rc"]) for r in v.runs),
            len(v.attempts), len(v.runs), v.dropped,
            per_variant_noise.get(v.id, "-"),
            stable_per_variant.get(i, {}).get("stable_bytes", "-")))
    packed = [i for i in order if variants[i].packs]
    if packed:
        lv = sorted({k for i in packed for k in variants[i].packs})
        md += ["", "### .dlb 打包（容器层，不进 .dla 字节差分）", "",
               "| 变体 | .dla B | " + " | ".join("level%d" % l for l in lv) + " |",
               "|---|---|" + "---|" * len(lv)]
        for i in packed:
            v = variants[i]
            md.append("| %s | %d | %s |" % (
                v.id, v.runs[0]["bytes"],
                " | ".join(str(v.packs.get(l, {}).get("bytes", "-")) for l in lv)))
    md += ["", "## 已知噪声区（两次编译不稳定）", "",
           "并集口径 **%d B** / %d 段（占最大产物 %.1f%%）；由每变体前 %d 次编译两两比对取并集"
           "（第 %d 次留作 holdout）。判定已改用**每变体各自标定**：跨变体差分只豁免"
           "**参与这次差分的两个变体**自己的噪声，逐段见 noise.txt。" % (
               len(noise), len(noise_runs), 100.0 * len(noise) / max(max_size, 1),
               calibration, repeats),
           "", "每变体噪声 B：%s" % json.dumps(per_variant_noise),
           "", "| zone | 字节 | 来源 |", "|---|---|---|"]
    for name, z in zones.items():
        md.append("| %s | %d | %s |" % (name, z["bytes"], z["source"]))
    md += ["", "## 判定", "",
           "「结果」按 %s 口径；「union 旧口径」列是同一份产物在旧口径下的判定（改口径前后对照）。" % default_scope,
           "", "| check | kind | 结果 | union 旧口径 | 关键数字 |", "|---|---|---|---|---|"]
    for r in results:
        nums = {k: v for k, v in r.items()
                if k in ("diff_bytes", "outside_noise", "inside_noise", "noise_share",
                         "escaped_zone", "size_delta", "delta", "ratio", "zone_bytes",
                         "noise_bytes", "share", "signal_bytes", "stable_diff_bytes",
                         "reorder_bytes", "payload_bytes", "size_field_bytes",
                         "body_size_delta", "expect_offsets")}
        legacy = "-"
        if r.get("by_scope") and "union" in r["by_scope"]:
            u = r["by_scope"]["union"]["ok"]
            legacy = ("PASS" if u else "FAIL") if r["expect"] == "pass" \
                else ("PASS(按预期被判红)" if not u else "FAIL(该红却绿)")
        md.append("| %s | %s | **%s** | %s | %s |" % (
            r["id"], r["kind"], "PASS" if r["as_expected"] else "FAIL", legacy,
            json.dumps(nums, ensure_ascii=False)))
    md += ["", "## 单个差分（噪声口径 / 稳定字节口径）", ""]
    for key in sorted(cache, key=lambda k: (k[0], k[1], k[2])):
        d = cache[key]
        a_id, b_id, scope = key
        md.append("- [%s] %s vs %s：size %d -> %d（%+d），不同 %d B，噪声内 %d B，噪声外 %d B"
                  % (scope, a_id, b_id, d["size_a"], d["size_b"], d["size_b"] - d["size_a"],
                     len(d["offsets"]), d["inside"], len(d["outside"])))
    if ctx["stable_cache"]:
        md += ["", "## 稳定字节口径（两侧都逐字节稳定的偏移上取值不同）", "",
               "| a | b | 稳定字节差 | 整块重排 | 信号 | 常量区剔除 | 排布类 | 信号区间（前 8） |",
               "|---|---|---|---|---|---|---|---|"]
        for key in sorted(ctx["stable_cache"]):
            d = ctx["stable_cache"][key]
            md.append("| %s | %s | %d | %d | %d | %d | %s | %s |" % (
                d["a"], d["b"], d["diff_bytes"], d["reorder_bytes"], d["signal_bytes"],
                d["excluded_area_bytes"], "同" if d["same_layout"] else "不同",
                d["signal_runs"][:8]))
    md.append("")
    (exp_dir / "report.md").write_text("\n".join(md) + "\n")
    if args.report:
        Path(args.report).write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
