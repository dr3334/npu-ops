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


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _row_bytes(shape, itemsize):
    """最后一维的字节数（逐元素算子按行成组，行规则作用在它上面）。"""
    return (shape[-1] if shape else 1) * itemsize


class Output:
    """run() 的结果：int8 码值 + 形状。"""

    __slots__ = ("codes", "shape", "scale")

    def __init__(self, codes, shape, scale):
        self.codes = np.asarray(codes, dtype=np.int8).reshape(shape)
        self.shape = tuple(shape)
        self.scale = scale

    def tobytes(self):
        return self.codes.tobytes()

    def __len__(self):
        return self.codes.size

    def __repr__(self):
        return "Output(shape=%r, int8, first=%r)" % (self.shape, self.codes.ravel()[:8].tolist())


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
        if t.dtype.kind != "int8":
            raise UnsupportedError(
                "%s: only int8 is implemented in this ticket (got %s); "
                "DType.int16/fp16 are entry points that raise for now" % (what, t.dtype.kind))

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
        if kind in ("add", "sub"):
            if a.dtype != b.dtype:
                raise UnsupportedError(
                    "%s: input scales differ (%r vs %r). That needs a requant with ratio != 1, "
                    "which has no board evidence for elementwise yet -- rescale explicitly."
                    % (kind, a.dtype.scale, b.dtype.scale))
            out_dtype = a.dtype
        else:  # mul
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
        spec = {
            "name": self.name,
            "op": op["kind"],
            "shape": list(op["a"].shape),
            "in_name": op["a"].name,
            "in1_name": op["b"].name,
            "out_name": out.name,
            "in_scale": op["a"].dtype.scale,
            "in1_scale": op["b"].dtype.scale,
            "out_scale": out.dtype.scale,
            "seed": 20261113,
        }
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
                                art / "golden" / (op["b"].name + ".bin"))
        cap = json.loads(cap_json.read_text())

        # ④ plan + manifest
        run_dev_dir = cap["device_run_dir"]
        plan = Plan.from_asset(
            asset_dir=cap["asset_dir"], iova_txt=art / "asset" / "iova.txt",
            alloc_a=4096, alloc_b=4096, alloc_c=4096,
            dir0=1, dir1=2, dir2=1, patch_idx=2, expect_patches=3,
            n_elems=op["a"].numel, n_inputs=2,
            in_a="%s/a.bin" % run_dev_dir, in_b="%s/b.bin" % run_dev_dir,
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
            "artifacts": {},
            "device_run_dir": run_dev_dir,
            "compile_log": str(art / "model.compile.log"),
            "capture": cap,
            "seconds": round(time.time() - started, 2),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        for rel in ["model.tflite", "model.dla", "plan.txt",
                    "golden/%s.bin" % op["a"].name, "golden/%s.bin" % op["b"].name,
                    "golden/expected.bin", "capture/vendor_out.bin"] + \
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
        if set(inputs) != {op["a"].name, op["b"].name}:
            raise ValueError("inputs must be exactly %r, got %r"
                             % ({op["a"].name, op["b"].name}, set(inputs)))
        blobs = {}
        for t in (op["a"], op["b"]):
            arr = np.asarray(inputs[t.name])
            if arr.dtype != np.int8:
                arr = arr.astype(np.int8)
            if arr.size != t.numel:
                raise ValueError("input %r: expected %d elements, got %d"
                                 % (t.name, t.numel, arr.size))
            blobs[t.name] = arr.astype(np.int8, copy=False).tobytes()

        run_dev_dir = art.manifest["device_run_dir"]
        with env.board_lock("npu_lib.run %s" % self.name):
            runner = self._ensure_runner(art)
            local_in = art.root / "run"
            local_in.mkdir(exist_ok=True)
            (local_in / "a.bin").write_bytes(blobs[op["a"].name])
            (local_in / "b.bin").write_bytes(blobs[op["b"].name])
            env.dev_push(local_in / "a.bin", "%s/a.bin" % run_dev_dir)
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
        codes = np.frombuffer(out_local.read_bytes(), dtype=np.int8)
        if codes.size != op["out"].numel:
            raise NpuLibError("device returned %d bytes, expected %d"
                              % (codes.size, op["out"].numel))
        out = Output(codes, op["out"].shape, op["out"].dtype.scale)
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
        if len(got) != op["out"].numel:
            raise ValueError("verify: got %d bytes, expected %d" % (len(got), op["out"].numel))

        # ② 数值：用**这次实际喂进去的**输入重算独立 golden
        if self._last_run is not None:
            a_codes = np.frombuffer(self._last_run["inputs"][op["a"].name], dtype=np.int8)
            b_codes = np.frombuffer(self._last_run["inputs"][op["b"].name], dtype=np.int8)
        else:
            a_codes = np.fromfile(art.golden / (op["a"].name + ".bin"), dtype=np.int8)
            b_codes = np.fromfile(art.golden / (op["b"].name + ".bin"), dtype=np.int8)
        expected, acc, ratio = golden_mod.reference(
            op["kind"], a_codes, b_codes,
            op["a"].dtype.scale, op["b"].dtype.scale, op["out"].dtype.scale)
        exp_bytes = np.asarray(expected, dtype=np.int8).tobytes()
        numeric_ok = got == exp_bytes
        n_bad = sum(1 for x, y in zip(got, exp_bytes) if x != y)
        first_bad = next((i for i, (x, y) in enumerate(zip(got, exp_bytes)) if x != y), None)

        # ① 同源：只有"喂的就是抓取时那对输入"时才谈得上逐字节比
        vendor_path = art.capture / "vendor_out.bin"
        vendor_bytes = vendor_path.read_bytes() if vendor_path.exists() else b""
        cap_in_a = (art.golden / (op["a"].name + ".bin")).read_bytes()
        cap_in_b = (art.golden / (op["b"].name + ".bin")).read_bytes()
        same_inputs = (self._last_run is None or
                       (self._last_run["inputs"][op["a"].name] == cap_in_a and
                        self._last_run["inputs"][op["b"].name] == cap_in_b))
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
            "requant_ratio": ratio,
            "same_source_note": same_note,
            "sha256_out": _sha256_bytes(got),
            "sha256_expected": _sha256_bytes(exp_bytes),
            "sha256_vendor": _sha256_bytes(vendor_bytes) if vendor_bytes else "",
        }
        res = VerifyResult(self.name, same_source_ok, numeric_ok, detail)
        print("[verify] %-22s same_source=%-5s numeric=%-5s (%d/%d mismatches)"
              % (self.name, same_source_ok, numeric_ok, n_bad, len(got)))
        return res
