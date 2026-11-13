"""路径/设备/锁 —— 库与外部世界的唯一接缝。

设计约束（见 .qwen25-impl/NOTES.md §2 与 docs/AGENT-BRIEF.md）：
  * 本库在 **WSL 的 /usr/bin/python3** 下运行（numpy/flatbuffers/tflite 都在那里）；
  * 设备是 Windows 侧的 adb 管的：WSL 里通过 interop 调 adb.exe，push 的**源路径必须是
    Windows 形式**，所以本地路径一律过 wslpath -w；
  * 板子是共享的，所有接触 /dev/apusys 的动作用 tools/devlock.sh 的同一把锁串行
    （lock 是目录，WSL 与 git-bash 看到的是同一个 /mnt/d 路径上的同一个目录）。
"""
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from .errors import NpuLibError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_ROOT = "/data/local/tmp/npuops"
NPU_LOCK = os.environ.get("NPU_LOCK", "/mnt/d/deepseek_ workspace/mtk/.npu-board.lock")

IS_WSL = (os.name == "posix" and "microsoft" in platform.uname().release.lower())
IS_WIN = (os.name == "nt")


def adb():
    """adb 的可执行文件（Windows 二进制；WSL 里走 interop）。"""
    explicit = os.environ.get("NPU_ADB")
    if explicit:
        return explicit
    for cand in ("/mnt/d/adb/platform-tools/adb.exe", "/d/adb/platform-tools/adb",
                 shutil.which("adb") or ""):
        if cand and os.path.exists(cand):
            return cand
    return "adb"


def python3():
    """带 numpy/flatbuffers/tflite schema 的解释器。"""
    return os.environ.get("NPU_PYTHON", "/usr/bin/python3")


def to_adb_path(path):
    """adb push 的源路径：WSL 下必须是 Windows 形式（adb.exe 是 Windows 进程）。"""
    p = os.path.abspath(str(path))
    if not IS_WSL:
        return p
    out = subprocess.run(["wslpath", "-w", p], capture_output=True, text=True)
    if out.returncode != 0:
        raise NpuLibError("wslpath -w %s failed: %s" % (p, out.stderr.strip()))
    return out.stdout.strip()


def artifacts_root():
    root = os.environ.get("NPU_LIB_ARTIFACTS")
    return Path(root) if root else (REPO_ROOT / "artifacts")


def run(cmd, cwd=None, check=True, capture=False, env=None, timeout=None):
    """跑一个外部命令；默认把输出透传到 stderr，失败抛 NpuLibError。"""
    printable = " ".join(str(c) for c in cmd)
    if os.environ.get("NPU_LIB_VERBOSE"):
        print("[cmd] " + printable, file=__import__("sys").stderr)
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd or REPO_ROOT),
                       text=True, capture_output=capture, env=e, timeout=timeout)
    if check and p.returncode != 0:
        tail = ""
        if capture:
            tail = "\n" + (p.stdout or "")[-2000:] + (p.stderr or "")[-2000:]
        raise NpuLibError("command failed (rc=%d): %s%s" % (p.returncode, printable, tail))
    if not capture:
        return p.returncode
    return p


# ------------------------------------------------------------------ board
def dev_shell(script, check=True, capture=False):
    """在设备上跑一段 shell。"""
    return run([adb(), "shell", script], check=check, capture=capture)


def dev_push(local, remote, check=True):
    return run([adb(), "push", to_adb_path(local), remote], check=check, capture=True)


def dev_pull(remote, local, check=True):
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    # 目的地也要 Windows 形式：adb.exe 是 Windows 进程，看不到 /mnt/d
    return run([adb(), "pull", remote, to_adb_path(local)], check=check, capture=True)


class board_lock:
    """板锁：与 tools/devlock.sh 共用同一个目录锁。

    * 自己在跑（默认）：mkdir 抢锁，退出时释放；
    * 外层已经用 <code>devlock.sh run -- </code> 包住了本次进程：设 NPU_LOCK_HELD=1，本类就只
      记录不抢锁（devlock 不可重入，重复抢会挂到超时）。
    """

    def __init__(self, what="npu_lib", timeout=1800):
        self.what = what
        self.timeout = timeout
        self.held = False

    def __enter__(self):
        if os.environ.get("NPU_LOCK_HELD") == "1":
            self.held = True          # 外层 devlock 已持有
            return self
        waited = 0
        while True:
            try:
                os.mkdir(NPU_LOCK)
                break
            except FileExistsError:
                if waited >= self.timeout:
                    raise NpuLibError("board lock timeout (%ds) at %s" % (self.timeout, NPU_LOCK))
                time.sleep(2)
                waited += 2
        with open(os.path.join(NPU_LOCK, "owner"), "w") as f:
            f.write("pid=%d host=%s ts=%d cmd=%s\n"
                    % (os.getpid(), platform.node(), int(time.time()), self.what))
        self.held = True
        return self

    def __exit__(self, *exc):
        if os.environ.get("NPU_LOCK_HELD") == "1":
            return False
        try:
            with open(os.path.join(NPU_LOCK, "owner")) as f:
                owner = f.read()
        except OSError:
            return False
        if ("pid=%d " % os.getpid()) in owner:
            shutil.rmtree(NPU_LOCK, ignore_errors=True)
        return False


def healthcheck(quick=True):
    """前后各一次的 healthcheck（公共工具，不改它；只在 WSL 里换个壳调用）。"""
    script = "./tools/healthcheck.sh" + (" --quick" if quick else "")
    if IS_WSL:
        # healthcheck -> e2e.sh 用了 git-bash 的 'pwd -W'，在 WSL 里跑不了；
        # 从 WSL 里借 Windows 的 git-bash 跑（interop）。
        win_repo = subprocess.run(["wslpath", "-w", str(REPO_ROOT)],
                                  capture_output=True, text=True).stdout.strip()
        bash = r"C:\Program Files\Git\bin\bash.exe"
        inner = 'cd "%s" && %s' % (win_repo.replace("\\", "/"), script)
        return run(["cmd.exe", "/c", bash, "-lc", inner], capture=True)
    return run(["bash", "-lc", script], capture=True)
