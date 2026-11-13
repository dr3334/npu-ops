#!/usr/bin/env bash
# 板端矩阵的完整跑法（票面纪律：前后各一次 healthcheck --quick，全程持 NPU_LOCK）。
#
#   ./tools/run_npu_lib_matrix.sh                     # add/mul/sub x 64/896/96
#   ./tools/run_npu_lib_matrix.sh --lens 64 --maps    # 子集 + 采一次 /proc/<pid>/maps 证据
#
# 为什么要这一层：库本身在 WSL 的 /usr/bin/python3 里跑（numpy/flatbuffers/tflite），
# 而 healthcheck/e2e 用的是 git-bash 的 'pwd -W'，只能在 Windows 侧跑。
# 锁是同一个目录（/mnt/d/... = /d/...），两次 acquire 之间不会互相卡住。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
export NPU_LOCK="${NPU_LOCK:-/d/deepseek_ workspace/mtk/.npu-board.lock}"

_win="$(pwd -W 2>/dev/null || echo /d/x)"
_drv="$(printf '%s' "${_win:0:1}" | tr 'A-Z' 'a-z')"
WSLROOT="/mnt/$_drv${_win:2}"

ARGS=""
for a in "$@"; do ARGS="$ARGS $a"; done

rc=0
echo "== [1/3] healthcheck --quick (before) =="
./tools/devlock.sh run -- ./tools/healthcheck.sh --quick || { echo "PRE-HEALTHCHECK FAILED"; exit 10; }

echo "== [2/3] matrix (WSL python3, plan-driven self-built submitter) =="
./tools/devlock.sh run -- wsl.exe -e bash -lc \
  "cd '$WSLROOT' && NPU_LOCK_HELD=1 PYTHONPATH=tools /usr/bin/python3 -m npu_lib.matrix$ARGS"
rc=$?

echo "== [3/3] healthcheck --quick (after) =="
./tools/devlock.sh run -- ./tools/healthcheck.sh --quick || { echo "POST-HEALTHCHECK FAILED"; exit 11; }

echo "== done rc=$rc (reports/npu-lib-matrix.md) =="
exit $rc
