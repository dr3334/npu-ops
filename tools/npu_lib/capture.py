"""命令流资产捕获 —— compile() 的第 3 步。

**这一步用厂商 runtime 跑一次**：把 .dla 交给 neuronrt，在 LD_PRELOAD 的 ioctl 记录器
(probe/libiolog2.so) 下录下一次合法提交，然后把命令流 / subcmd / cmdin 模板 / 三个旧 IOVA
抽成资产目录。这是 docs/adr/0002 明确划进「自建」范围的**命令捕获**：命令仍来自厂商编译器+一次
录制，但执行路径（握手/分配/重定位/提交/同步）完全是我们自己的。

run() **不**碰 neuronrt —— 见 docs/OWN-SUBMITTER-LIB.md 的「怎么证明没碰厂商库」。
"""
import json
import os
import re
import shutil
from pathlib import Path

from . import env
from .errors import NpuLibError

DEV = env.DEV_ROOT


def _sh(script, check=True):
    return env.dev_shell(script, check=check, capture=True)


def capture(name, artifacts_dir, dla_path, a_path, b_path, timeout_note=""):
    """跑一次厂商 runtime 抓资产；返回 dict（含 iova/cmdbufs/vendor_out 路径）。"""
    art = Path(artifacts_dir)
    cap = art / "capture"
    bufdir = cap / "buf"
    assert_dir = art / "asset"
    if cap.exists():
        shutil.rmtree(cap)
    bufdir.mkdir(parents=True)

    run_dir = "%s/runs/%s" % (DEV, name)
    _sh("mkdir -p %s/asset" % run_dir)
    env.dev_push(dla_path, "%s/model.dla" % run_dir)
    env.dev_push(a_path, "%s/a.bin" % run_dir)
    env.dev_push(b_path, "%s/b.bin" % run_dir)
    # mkasset.py 也要看这两个文件（按前 64 字节认 A/B 缓冲）
    shutil.copy(a_path, cap / "a.bin")
    shutil.copy(b_path, cap / "b.bin")
    vendor_remote = "%s/vendor_out.bin" % run_dir

    # 记录器默认写 probe/io2.<pid>.log 与 probe/buf/（相对 cwd），所以必须先 cd $DEV
    cmd = ("cd %s && rm -f probe/io2.*.log && rm -rf probe/buf && mkdir -p probe/buf && "
           "LD_PRELOAD=%s/probe/libiolog2.so LD_LIBRARY_PATH=%s/lib "
           "%s/bin/neuronrt -m hw -a %s/model.dla -i %s/a.bin -i %s/b.bin -o %s -c 1 2>&1"
           % (DEV, DEV, DEV, DEV, run_dir, run_dir, run_dir, vendor_remote))
    r = _sh(cmd, check=False)
    if r.returncode != 0:
        raise NpuLibError("vendor capture run failed (rc=%d): %s" % (r.returncode, r.stdout))

    log_remote = _sh("ls -t %s/probe/io2.*.log 2>/dev/null | head -1" % DEV, check=False)
    log = (log_remote.stdout or "").strip().splitlines()
    if not log:
        raise NpuLibError("no iolog produced (shim not loaded?) -- capture failed")
    _sh("cp %s %s/probe/cap_%s.log" % (log[0], DEV, name))
    cap_log = cap / "cap.log"
    env.dev_pull("%s/probe/cap_%s.log" % (DEV, name), cap_log)

    names = _sh("cd %s/probe/buf && ls *.bin 2>/dev/null" % DEV, check=False).stdout or ""
    nbuf = 0
    for f in names.split():
        env.dev_pull("%s/probe/buf/%s" % (DEV, f), bufdir / f)
        nbuf += 1
    if nbuf == 0:
        raise NpuLibError("no buffer dumps pulled from %s/probe/buf" % DEV)

    vendor_local = cap / "vendor_out.bin"
    env.dev_pull(vendor_remote, vendor_local)

    # 资产目录：mkasset.py 是公共工具（不改），它的输入全给绝对路径
    if assert_dir.exists():
        shutil.rmtree(assert_dir)
    assert_dir.mkdir(parents=True)
    shutil.copy(vendor_local, assert_dir / "vendor_out.bin")   # 它用它反查真正的输出缓冲
    r = env.run([env.python3(), "probe/mkasset.py", str(cap_log), str(bufdir),
                 str(cap / "a.bin"), str(cap / "b.bin"), str(assert_dir)],
                cwd=env.REPO_ROOT, capture=True)
    shutil.copy(a_path, assert_dir / "a.bin")
    shutil.copy(b_path, assert_dir / "b.bin")

    iova = (assert_dir / "iova.txt").read_text().split()
    files = sorted(p.name for p in assert_dir.iterdir())
    for fname in files:
        if fname in ("a.bin", "b.bin", "vendor_out.bin"):
            continue
        env.dev_push(assert_dir / fname, "%s/asset/%s" % (run_dir, fname))

    info = {
        "device_run_dir": run_dir,
        "asset_dir": "%s/asset" % run_dir,
        "iova": ["0x%x" % int(x, 16) for x in iova[:3]],
        "cmdbufs": (assert_dir / "cmdsizes.txt").read_text().split(),
        "buffers_dumped": nbuf,
        "vendor_out": str(vendor_local),
        "capture_log": str(cap_log),
        "vendor_stdout": (r.stdout or "")[-4000:],
        "note": timeout_note or "one vendor-runtime run, only to record the command stream (ADR 0002)",
    }
    (art / "capture.json").write_text(json.dumps(info, indent=2))
    return info
