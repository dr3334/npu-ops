"""Graph —— 定义 -> 编译 -> 运行 -> 验收。

接口形状是票面定死的（便于后面扩算子/扩精度）：

    DType.int8(scale, zero_point=0)
    g = Graph(name)
    t = g.input(name, shape, dtype)
    g.add(a, b) / g.mul(a, b) / g.sub(a, b)      # 同形状，无广播（广播显式报错）
    g.compile()  -> Artifacts
    g.run(inputs) -> Output
    g.verify(out) -> VerifyResult

compile() 的四步（票面 ①②③④）：
  1. 用 TFLite 写出 model.tflite（dtype/shape/scale 来自 Graph，数值由 golden.py 独立给出）
  2. tools/compile.sh 编 model.dla（ncc-tflite，--disallow-bridge）
  3. **用厂商 runtime 跑一次**抓命令流资产（ADR 0002 允许的「命令捕获」，见 capture.py）
  4. 资产/plan/golden/manifest 落到 artifacts/<name>/

run() 只走自建提交器：plan.txt 驱动 probe/npu_run（/dev/apusys），不 dlopen 任何厂商库。
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import capture as capture_mod
from . import env
from . import golden as golden_mod
from .dtype import DType
from .errors import BroadCastError, NpuLibError, UnsupportedError
from .plan import Plan
from .tensor import Tensor

# 16 B 行规则：int8 每行的字节数必须 >= 16 且 % 16 == 0（docs/OP-MATRIX.md）。
MIN_ROW_BYTES = 16
MAX_NPU_RUN_BYTES = 4 * 1024 * 1024   # 一块数据缓冲的上限（抓取时 runtime 用 4 KiB，这里留余量）

# FC 的输出 scale 约定：s_out = s_a * s_w * 32（与 ops/fc_dynb_* / tools/build_dynb.py 同一条
# 已上板的约定，requant multiplier = 1/32）。改这个数等于改设备端算子的量化语义，先要证据。
FC_OUT_SCALE_FACTOR = 32.0


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


ALLOC_ALIGN = 4096   # 抓取时 runtime 对数据缓冲就是 4 KiB；大张量（FC 权重）按 4 KiB 向上取整


def _alloc_for(nbytes):
    """一块数据缓冲的分配大小：至少 4 KiB、按 4 KiB 向上取整。

    逐元素算子的张量只有几十~几千字节，所以以前写死 4096 够用；Qwen 的 FC 权重
    （4864x896 int8 = 4.36 MB）必须真的给够，否则 npu_run 会在尺寸检查处退出。
    """
    n = max(int(nbytes), ALLOC_ALIGN)
    return (n + ALLOC_ALIGN - 1) // ALLOC_ALIGN * ALLOC_ALIGN


def _np_dtype(kind):
    return np.float16 if kind == "fp16" else np.int8


def _row_bytes(shape, itemsize):
    """最后一维的字节数（逐元素算子按行成组，行规则作用在它上面）。"""
    return (shape[-1] if shape else 1) * itemsize


class Output:
    """run() 的结果：码值（int8）或 fp16 数值 + 形状。"""

    __slots__ = ("codes", "shape", "scale", "kind")

    def __init__(self, codes, shape, scale, kind="int8"):
        self.kind = kind
        self.codes = np.asarray(codes, dtype=_np_dtype(kind)).reshape(shape)
        self.shape = tuple(shape)
        self.scale = scale

    def tobytes(self):
        return self.codes.tobytes()

    def __len__(self):
        return self.codes.size

    def __repr__(self):
        return "Output(shape=%r, %s, first=%r)" % (self.shape, self.kind,
                                                    self.codes.ravel()[:8].tolist())


class VerifyResult:
    """两道门的结果。ok == (同源逐字节) and (golden 逐位)。"""

    def __init__(self, name, same_source_ok, numeric_ok, detail):
        self.name = name
        self.same_source_ok = same_source_ok
        self.numeric_ok = numeric_ok
        self.detail = detail

    @property
    def ok(self):
        return bool(self.same_source_ok and self.numeric_ok)

    def as_dict(self):
        d = {"name": self.name, "ok": self.ok,
             "same_source_ok": self.same_source_ok, "numeric_ok": self.numeric_ok}
        d.update(self.detail)
        return d

    def __repr__(self):
        return "VerifyResult(%s, ok=%s, same_source=%s, numeric=%s)" % (
            self.name, self.ok, self.same_source_ok, self.numeric_ok)


class Artifacts:
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.manifest = manifest
        self.asset = self.root / "asset"
        self.capture = self.root / "capture"
        self.golden = self.root / "golden"
        self.plan_path = self.root / "plan.txt"

    def __repr__(self):
        return "Artifacts(%s)" % self.root


class Graph:
    def __init__(self, name):
        if not name or "/" in name or "\\" in name:
            raise ValueError("Graph name must be a simple token, got %r" % (name,))
        self.name = name
        self._inputs = []
        self._tensors = []
        self._ops = []
        self._artifacts = None
        self._last_run = None

    # ------------------------------------------------------------ 定义
    def input(self, name, shape, dtype):
        if not isinstance(dtype, DType):
            raise TypeError("dtype must be npu_lib.DType, got %r" % (dtype,))
        if any(t.name == name for t in self._tensors):
            raise ValueError("duplicate tensor name %r" % (name,))
        shape = tuple(int(d) for d in shape)
        if not shape or any(d <= 0 for d in shape):
            raise ValueError("shape must be a non-empty tuple of positive ints, got %r" % (shape,))
        row = _row_bytes(shape, dtype.itemsize)
        if row % MIN_ROW_BYTES != 0 or row < MIN_ROW_BYTES:
            raise ValueError(
                "16 B row rule: last dim of %r with %s is %d B; need >= %d and %% %d == 0 "
                "(docs/OP-MATRIX.md)" % (shape, dtype.name, row, MIN_ROW_BYTES, MIN_ROW_BYTES))
        t = Tensor(name, shape, dtype)
        self._inputs.append(t)
        self._tensors.append(t)
        return t

    def _check_tensor(self, t, what):
        if not isinstance(t, Tensor):
            raise TypeError("%s: expected Tensor, got %r" % (what, t))
        if t not in self._tensors:
            raise ValueError("%s: tensor %r belongs to another graph" % (what, t.name))
        if t.dtype.kind not in ("int8", "fp16"):
            raise UnsupportedError(
                "%s: only int8 and fp16 are implemented (got %s); DType.int16 still raises"
                % (what, t.dtype.kind))

    def _binop(self, kind, a, b):
        self._check_tensor(a, kind)
        self._check_tensor(b, kind)
        if a.shape != b.shape:
            raise BroadCastError(
                "%s: shapes %r and %r differ -- this library does NOT broadcast "
                "(same-shape only, by design); add an explicit RESHAPE/broadcast op later"
                % (kind, a.shape, b.shape))
        if a.dtype.kind != b.dtype.kind:
            raise UnsupportedError("%s: mixed dtypes %s/%s" % (kind, a.dtype, b.dtype))
        if a.dtype.kind == "fp16":
            out_dtype = DType.fp16()
        elif kind in ("add", "sub"):
            if a.dtype != b.dtype:
                raise UnsupportedError(
                    "%s: input scales differ (%r vs %r). That needs a requant with ratio != 1, "
                    "which has no board evidence for elementwise yet -- rescale explicitly."
                    % (kind, a.dtype.scale, b.dtype.scale))
            out_dtype = a.dtype
        else:  # mul（int8）
            out_dtype = DType.int8(a.dtype.scale * b.dtype.scale, 0)
        # 一条图只有一个算子：单层 .dla = 一份命令流 + 3 块缓冲（见 docs/OWN-SUBMITTER-LIB.md）
        if self._ops:
            raise UnsupportedError(
                "graph %r already has an op: this submitter replays ONE command stream with "
                "A/B/C buffers, so a chained graph needs an N-buffer plan (documented gap). "
                "Compile each op as its own Graph for now." % self.name)
        idx = len(self._ops)
        out = Tensor("%s_%d" % (kind, idx), a.shape, out_dtype, producer=None)
        self._ops.append({"kind": kind, "a": a, "b": b, "out": out, "index": idx})
        out._producer = self._ops[-1]
        self._tensors.append(out)
        return out

    def add(self, a, b):
        return self._binop("add", a, b)

    def sub(self, a, b):
        return self._binop("sub", a, b)

    def mul(self, a, b):
        return self._binop("mul", a, b)

    def fc(self, x, w, out_scale=None):
        """FULLY_CONNECTED，**权重是运行时输入**（不是常量）。

        x [M,K] int8，w [N,K] int8（TFLite 的权重布局）-> out [M,N] int8。
        s_out 缺省是 s_a*s_w*32（ops/fc_dynb_* 那条已上板的老约定）；K 一大就不够 —— 例如
        K=896 时满量程码的累加能到 1e7 量级，输出会 97.8% 饱和（见 reports/npu-lib-fc-qwen.md），
        所以真实维度上必须按数据显式给 out_scale = max|acc|/127 之类的值。
        权重为什么必须动态：常量权重在自建提交器上只有落在抓取那次的 IOVA 才对
        （docs/ISSUE-const-fc.md，3/11）；动态权重有板端证据（docs/OWN-SUBMITTER.md §6）。
        """
        self._check_tensor(x, "fc")
        self._check_tensor(w, "fc")
        if len(x.shape) != 2 or len(w.shape) != 2:
            raise UnsupportedError(
                "fc: rank-2 only (got %r and %r); RESHAPE stays on the host" % (x.shape, w.shape))
        m, k = x.shape
        n, k2 = w.shape
        if k != k2:
            raise BroadCastError(
                "fc: inner dims differ (x [M,K]=%r vs w [N,K]=%r)" % (x.shape, w.shape))
        if self._ops:
            raise UnsupportedError(
                "graph %r already has an op: one op per graph (3-buffer submitter). "
                "Compile each op as its own Graph for now." % self.name)
        if out_scale is None:
            out_scale = x.dtype.scale * w.dtype.scale * FC_OUT_SCALE_FACTOR
        out_dtype = DType.int8(out_scale, 0)
        out = Tensor("fc_0", (m, n), out_dtype, producer=None)
        self._ops.append({"kind": "fc", "a": x, "b": w, "out": out, "index": 0})
        out._producer = self._ops[-1]
        self._tensors.append(out)
        return out

    def softmax(self, x, out_scale=None):
        """SOFTMAX（原生 SoftMaxLayer），沿最后一维。

        int8 的 softmax 输出按 TFLite 惯例是 `s_out = 1/256`、`zero_point = -128`（非对称），
        这一档有板端证据：[1,8,16] 上我们的重放结果与厂商设备输出**逐字节相同**
        （reports/npu-lib-softmax.md）。判据按登记容差比（缺省 int8_tol = 2，与厂商那条 compare 一致），
        不要求逐位：设备是 LUT 定点实现。
        """
        self._check_tensor(x, "softmax")
        if self._ops:
            raise UnsupportedError(
                "graph %r already has an op: one op per graph (3-buffer submitter)." % self.name)
        if out_scale is None:
            out_scale = 1.0 / 256.0
        out_dtype = DType.int8_asym(out_scale, -128)
        out = Tensor("softmax_0", x.shape, out_dtype, producer=None)
        self._ops.append({"kind": "softmax", "a": x, "b": None, "out": out, "index": 0,
                          "tol": 2})
        out._producer = self._ops[-1]
        self._tensors.append(out)
        return out

    def mean(self, x, axis=-1, out_scale=None):
        """MEAN（keep_dims=True），沿 axis 归约。int8。

        输出尺度缺省取 `s_in`（= ops/mean_4x16_int8 那条已上板的厂商约定：out scale == in scale，
        于是 multiplier = 1/K、码值 = round(sum/K)）；真实用途按数据显式给。
        """
        self._check_tensor(x, "mean")
        if self._ops:
            raise UnsupportedError(
                "graph %r already has an op: one op per graph (3-buffer submitter)." % self.name)
        ax = int(axis) % len(x.shape)
        shape = list(x.shape)
        shape[ax] = 1
        if out_scale is None:
            out_scale = x.dtype.scale
        out = Tensor("mean_0", tuple(shape), DType.int8(out_scale, 0), producer=None)
        self._ops.append({"kind": "mean", "a": x, "b": None, "out": out, "index": 0,
                          "axis": ax, "tol": 0})
        out._producer = self._ops[-1]
        self._tensors.append(out)
        return out

    def rsqrt(self, x, out_scale=None):
        """RSQRT（一元），int8。输入码值必须 > 0。"""
        self._check_tensor(x, "rsqrt")
        if self._ops:
            raise UnsupportedError(
                "graph %r already has an op: one op per graph (3-buffer submitter)." % self.name)
        if out_scale is None:
            # 缺省把 y 的满量程放在 1.0 上（y = 1/sqrt(x) 常见量级 O(1)）
            out_scale = 1.0 / 127.0
        out = Tensor("rsqrt_0", x.shape, DType.int8(out_scale, 0), producer=None)
        self._ops.append({"kind": "rsqrt", "a": x, "b": None, "out": out, "index": 0,
                          "tol": 0})
        out._producer = self._ops[-1]
        self._tensors.append(out)
        return out

    # ------------------------------------------------------------ 编译
    def compile(self, rebuild=False):
        if len(self._ops) != 1:
            raise UnsupportedError(
                "graph %r has %d ops; this ticket ships the single-op path "
                "(one .dla layer, one command stream, 3 buffers)" % (self.name, len(self._ops)))
        op = self._ops[0]
        out = op["out"]
        art = env.artifacts_root() / self.name
        art.mkdir(parents=True, exist_ok=True)
        started = time.time()

        model_tflite = art / "model.tflite"
        model_dla = art / "model.dla"
        spec_path = art / "spec.json"
        unary = op["b"] is None
        spec = {
            "name": self.name,
            "op": op["kind"],
            "shape": list(op["a"].shape),
            "in_name": op["a"].name,
            "in1_name": "" if unary else op["b"].name,
            "out_name": out.name,
            "in_scale": op["a"].dtype.scale,
            "in1_scale": 1.0 if unary else op["b"].dtype.scale,
            "dtype": op["a"].dtype.kind,
            "out_scale": out.dtype.scale,
            "out_zero_point": out.dtype.zero_point,
            "int8_tol": op.get("tol", 0),
            "seed": 20261113,
        }
        if op["kind"] == "mean":
            spec["axis"] = op["axis"]
        if op["kind"] == "fc":
            # FC 的两个输入与输出是三个不同形状：binop 的单 "shape" 不够用
            spec["shapes"] = {"a": list(op["a"].shape), "b": list(op["b"].shape),
                              "out": list(op["out"].shape)}
        # 同一个名字 + 不同 spec（形状/scale）= 拿旧命令流套新量化参数 —— 这类错会被
        # golden 悄悄吸收，必须当场失败而不是复用（教训：Qwen 维度的 gate_proj 分块实验）。
        if spec_path.exists() and not rebuild:
            old = json.loads(spec_path.read_text())
            # 只比**两边都有**的键：后来新增的元数据（如 int8_tol）不该让老资产报错
            diff = {k: (old[k], spec[k]) for k in spec if k in old and old[k] != spec[k]}
            if diff:
                raise NpuLibError(
                    "graph %r: artifacts already exist with a different spec %r; "
                    "use another name or compile(rebuild=True)" % (self.name, diff))
        spec_path.write_text(json.dumps(spec, indent=2))

        # ① TFLite
        if rebuild or not model_tflite.exists():
            env.run([env.python3(), "-m", "npu_lib.build_model",
                     "--spec", str(spec_path), "--out", str(art)], cwd=env.REPO_ROOT)
        # ② .dla
        if rebuild or not model_dla.exists():
            env.run(["bash", "tools/compile.sh", str(model_tflite), str(model_dla)],
                    cwd=env.REPO_ROOT)
            log = art / "model.compile.log"
            txt = log.read_text() if log.exists() else ""
            if "EXIT=0" not in txt:
                raise NpuLibError("compile failed; see %s\n%s" % (log, txt[-2000:]))

        # ③ 资产捕获（厂商 runtime 只用来录一次命令流）
        cap_json = art / "capture.json"
        if rebuild or not cap_json.exists():
            capture_mod.capture(self.name, art, model_dla,
                                art / "golden" / (op["a"].name + ".bin"),
                                None if unary else art / "golden" / (op["b"].name + ".bin"))
        cap = json.loads(cap_json.read_text())

        # ④ plan + manifest
        run_dev_dir = cap["device_run_dir"]
        # 一元算子的第三块缓冲是 LUT/scratch：它的内容在资产里（extra_file），不是我们推的输入
        scratch = (art / "asset" / "scratch.bin").exists()
        if unary:
            bytes_b = 4096 if scratch else 0
            n_inputs = 2 if scratch else 1
            extra = "scratch.bin" if scratch else ""
            in_b = ""
        else:
            bytes_b, n_inputs, extra = op["b"].nbytes, 2, ""
            in_b = "%s/b.bin" % run_dev_dir
        # 命令流里有几个地址字 = iova.txt 里几个非 0 槽（一元且没有 LUT 时只有 A 和 C 两处）
        iova_txt = art / "asset" / "iova.txt"
        iovas = [int(x, 16) for x in iova_txt.read_text().split()[:3]]
        expect_patches = sum(1 for v in iovas if v)
        plan = Plan.from_asset(
            asset_dir=cap["asset_dir"], iova_txt=iova_txt,
            alloc_a=_alloc_for(op["a"].nbytes), alloc_b=_alloc_for(max(bytes_b, 1)),
            alloc_c=_alloc_for(op["out"].nbytes),
            dir0=1, dir1=2, dir2=1, patch_idx=2, expect_patches=expect_patches,
            bytes_a=op["a"].nbytes, bytes_b=bytes_b, bytes_out=op["out"].nbytes,
            n_inputs=n_inputs, extra_file=extra,
            in_a="%s/a.bin" % run_dev_dir, in_b=in_b,
            out="%s/out.bin" % run_dev_dir)
        plan.write(art / "plan.txt")

        manifest = {
            "name": self.name,
            "op": op["kind"],
            "shape": list(op["a"].shape),
            "dtype": "int8",
            "numel": op["a"].numel,
            "scales": {"a": spec["in_scale"], "b": spec["in1_scale"], "out": spec["out_scale"]},
            "requant_ratio": golden_mod.ratio_for(op["kind"], spec["in_scale"],
                                                  spec["in1_scale"], spec["out_scale"]),
            "int8_tol": op.get("tol", 0),
            "unary": unary,
            "artifacts": {},
            "device_run_dir": run_dev_dir,
            "compile_log": str(art / "model.compile.log"),
            "capture": cap,
            "seconds": round(time.time() - started, 2),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        gold_rel = ["golden/%s.bin" % op["a"].name, "golden/expected.bin"]
        if not unary:
            gold_rel.append("golden/%s.bin" % op["b"].name)
        for rel in ["model.tflite", "model.dla", "plan.txt"] + gold_rel + \
                   ["capture/vendor_out.bin"] + \
                   ["asset/%s" % p.name for p in sorted((art / "asset").iterdir())
                    if p.name not in ("a.bin", "b.bin", "vendor_out.bin")]:
            f = art / rel
            if f.exists():
                manifest["artifacts"][rel] = {"bytes": f.stat().st_size, "sha256": _sha256_file(f)}
        (art / "manifest.json").write_text(json.dumps(manifest, indent=2))

        self._artifacts = Artifacts(art, manifest)
        return self._artifacts

    # ------------------------------------------------------------ 运行
    def _artifacts_or_compile(self):
        if self._artifacts is None:
            if (env.artifacts_root() / self.name / "manifest.json").exists():
                art = env.artifacts_root() / self.name
                self._artifacts = Artifacts(art, json.loads((art / "manifest.json").read_text()))
            else:
                self.compile()
        return self._artifacts

    def _ensure_runner(self, art):
        binary = env.REPO_ROOT / "probe" / "npu_run"
        if not binary.exists():
            env.run(["bash", "tools/build_npu_run.sh"], cwd=env.REPO_ROOT)
        local_sha = _sha256_file(binary)
        remote = "%s/bin/npu_run" % env.DEV_ROOT
        probe = env.dev_shell("sha256sum %s 2>/dev/null | cut -d' ' -f1" % remote,
                              check=False, capture=True)
        if (probe.stdout or "").strip() != local_sha:
            env.dev_shell("mkdir -p %s/bin" % env.DEV_ROOT)
            env.dev_push(binary, remote)
        env.dev_shell("chmod 755 %s" % remote)   # adb push 不保留可执行位
        return remote

    def run(self, inputs):
        """把 inputs 推给自建提交器跑一次；返回 Output(int8)。"""
        art = self._artifacts_or_compile()
        op = self._ops[0]
        want = {op["a"].name} if op["b"] is None else {op["a"].name, op["b"].name}
        if set(inputs) != want:
            raise ValueError("inputs must be exactly %r, got %r" % (want, set(inputs)))
        blobs = {}
        for t in [x for x in (op["a"], op["b"]) if x is not None]:
            want_dt = _np_dtype(t.dtype.kind)
            arr = np.asarray(inputs[t.name])
            if arr.dtype != want_dt:
                arr = arr.astype(want_dt)
            if arr.size != t.numel:
                raise ValueError("input %r: expected %d elements, got %d"
                                 % (t.name, t.numel, arr.size))
            blobs[t.name] = arr.astype(want_dt, copy=False).tobytes()

        run_dev_dir = art.manifest["device_run_dir"]
        with env.board_lock("npu_lib.run %s" % self.name):
            runner = self._ensure_runner(art)
            local_in = art.root / "run"
            local_in.mkdir(exist_ok=True)
            (local_in / "a.bin").write_bytes(blobs[op["a"].name])
            env.dev_push(local_in / "a.bin", "%s/a.bin" % run_dev_dir)
            if op["b"] is not None:
                (local_in / "b.bin").write_bytes(blobs[op["b"].name])
                env.dev_push(local_in / "b.bin", "%s/b.bin" % run_dev_dir)
            env.dev_push(art.plan_path, "%s/plan.txt" % run_dev_dir)
            r = env.dev_shell("cd %s && %s plan.txt" % (run_dev_dir, runner),
                              check=False, capture=True)
            (local_in / "npu_run.log").write_text((r.stdout or "") + (r.stderr or ""))
            if r.returncode != 0:
                raise NpuLibError("npu_run rc=%d: %s"
                                  % (r.returncode, (r.stdout or "")[-2000:] + (r.stderr or "")[-2000:]))
            out_local = local_in / "out.bin"
            env.dev_pull("%s/out.bin" % run_dev_dir, out_local)
        kind = op["out"].dtype.kind
        codes = np.frombuffer(out_local.read_bytes(), dtype=_np_dtype(kind))
        if codes.size != op["out"].numel:
            raise NpuLibError("device returned %d bytes, expected %d"
                              % (codes.size, op["out"].numel))
        out = Output(codes, op["out"].shape, op["out"].dtype.scale, kind)
        self._last_run = {"inputs": blobs, "out": out.tobytes(),
                          "runner_log": str(local_in / "npu_run.log")}
        return out

    # ------------------------------------------------------------ 验收
    def verify(self, out):
        """两道门：① 与抓资产那次厂商输出逐字节相同；② 与 CPU golden 逐位相同。"""
        art = self._artifacts_or_compile()
        op = self._ops[0]
        if isinstance(out, Output):
            got = out.tobytes()
        else:
            got = np.asarray(out).astype(np.int8).tobytes()
        if len(got) != op["out"].nbytes:
            raise ValueError("verify: got %d bytes, expected %d" % (len(got), op["out"].nbytes))

        # ② 数值：用**这次实际喂进去的**输入重算独立 golden
        dt = _np_dtype(op["a"].dtype.kind)
        src = self._last_run["inputs"] if self._last_run is not None else None
        if src is not None:
            a_codes = np.frombuffer(src[op["a"].name], dtype=dt)
            b_codes = (None if op["b"] is None
                       else np.frombuffer(src[op["b"].name], dtype=dt))
        else:
            a_codes = np.fromfile(art.golden / (op["a"].name + ".bin"), dtype=dt)
            b_codes = (None if op["b"] is None
                       else np.fromfile(art.golden / (op["b"].name + ".bin"), dtype=dt))
        # golden 要按张量形状算：fc 是矩阵乘，softmax 沿最后一维归约
        # （一维码流直接算会把整块当成一行 —— softmax 就变成全张量归一化了）
        if op["kind"] in ("fc", "softmax", "mean", "rsqrt"):
            a_codes = a_codes.reshape(op["a"].shape)
        if op["kind"] == "fc":
            b_codes = b_codes.reshape(op["b"].shape)
        if op["a"].dtype.kind == "fp16":
            # fp16：按 **ULP** 比（设备是 fp16 算术，与 float32 参考之间本来就有舍入）
            max_ulp = int(op.get("max_ulp", 1))
            a_f = np.asarray(a_codes, dtype=np.float16).reshape(op["a"].shape)
            b_f = (None if op["b"] is None
                   else np.asarray(b_codes, dtype=np.float16).reshape(op["b"].shape))
            exp_f16 = golden_mod.reference_fp16(op["kind"], a_f, b_f, axis=op.get("axis"))
            got_f16 = np.frombuffer(got, dtype=np.float16).reshape(op["out"].shape)
            u = golden_mod.ulp_diff(got_f16, exp_f16)
            bad = u > max_ulp
            numeric_ok = not bool(bad.any())
            n_bad = int(bad.sum())
            first_bad = int(np.argmax(bad)) if n_bad else None
            exp_bytes = exp_f16.tobytes()
            num_detail = {"numeric_rule": "ulp", "ulp_tol": max_ulp, "max_ulp": int(u.max()),
                          "requant_ratio": None}
        else:
            expected, acc, ratio = golden_mod.reference(
                op["kind"], a_codes, b_codes,
                op["a"].dtype.scale, 1.0 if op["b"] is None else op["b"].dtype.scale,
                op["out"].dtype.scale, zp_out=op["out"].dtype.zero_point, axis=op.get("axis"))
            exp_arr = np.asarray(expected, dtype=np.int8).ravel()
            exp_bytes = exp_arr.tobytes()
            # 容差：int8_tol 为 0 时逐位比；>0 时按 |diff| <= tol（softmax 这类定点/LUT 实现）
            tol = int(op.get("tol", 0))
            got_arr = np.frombuffer(got, dtype=np.int8)
            diff = np.abs(got_arr.astype(np.int16) - exp_arr.astype(np.int16))
            bad = diff > tol
            numeric_ok = not bool(bad.any())
            n_bad = int(bad.sum())
            first_bad = int(np.argmax(bad)) if n_bad else None
            num_detail = {"numeric_rule": "int8_code", "int8_tol": tol,
                          "max_abs_diff": int(diff.max()) if diff.size else 0,
                          "requant_ratio": ratio}

        # ① 同源：只有"喂的就是抓取时那对输入"时才谈得上逐字节比
        vendor_path = art.capture / "vendor_out.bin"
        vendor_bytes = vendor_path.read_bytes() if vendor_path.exists() else b""
        cap_in_a = (art.golden / (op["a"].name + ".bin")).read_bytes()
        if self._last_run is None:
            same_inputs = True
        elif op["b"] is None:
            same_inputs = self._last_run["inputs"][op["a"].name] == cap_in_a
        else:
            cap_in_b = (art.golden / (op["b"].name + ".bin")).read_bytes()
            same_inputs = (self._last_run["inputs"][op["a"].name] == cap_in_a and
                           self._last_run["inputs"][op["b"].name] == cap_in_b)
        if not vendor_bytes:
            same_source_ok = False
            same_note = "capture/vendor_out.bin missing -- capture step did not run"
        elif not same_inputs:
            same_source_ok = False
            same_note = ("inputs differ from the captured pair, so a byte-for-byte comparison "
                         "with the vendor run is not defined (run the captured golden inputs)")
        else:
            same_source_ok = got == vendor_bytes
            same_note = "byte-for-byte vs capture/vendor_out.bin"

        detail = {
            "bytes": len(got),
            "numeric_mismatches": n_bad,
            "numeric_first_bad_index": first_bad,
            "same_source_note": same_note,
            "sha256_out": _sha256_bytes(got),
            "sha256_expected": _sha256_bytes(exp_bytes),
            "sha256_vendor": _sha256_bytes(vendor_bytes) if vendor_bytes else "",
        }
        detail.update(num_detail)
        res = VerifyResult(self.name, same_source_ok, numeric_ok, detail)
        unit = "ulp" if op["a"].dtype.kind == "fp16" else "mismatches"
        print("[verify] %-22s same_source=%-5s numeric=%-5s (%d/%d %s)"
              % (self.name, same_source_ok, numeric_ok, n_bad, len(got), unit))
        return res
