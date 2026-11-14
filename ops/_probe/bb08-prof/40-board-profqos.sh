#!/usr/bin/env bash
# 40-board-profqos.sh — the runtime's own per-sub-command API (07's harness), used for ONE purpose
# here: get the runtime's exec_us for the *same product* so the cycle->us clock is pinned inside a
# single run instead of across runs (07 9 left exactly this open).
#
#   NPU_LOCK_HELD=1 ./tools/devlock.sh run -- bash ops/_probe/bb08-prof/40-board-profqos.sh fc
#
# Two passes per product:
#   clean    : no profiler env          -> exec_us for reconciliation
#   L3       : MTKNN_PROFILING_LEVEL=3  -> SAME RUN also writes execution_result.json/dla_info.json
#              (this is the run whose JSON and exec_us belong together)
set -uo pipefail
M="${1:?model fc|dec}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EV="$ROOT/ops/_probe/bb08-prof/board"
ADB="${ADB:-/d/adb/platform-tools/adb}"
DEV=/data/local/tmp/npuops
export MSYS_NO_PATHCONV=1
mkdir -p "$EV"
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }

sh_ "ls -l $DEV/bin/profqos_v" || { echo "NO profqos_v on device"; exit 9; }

: > "$EV/profqos-$M.txt"
for l in 0 1 2 3; do
  pass() {
    local tag="$1" envs="$2"
    local dir="$DEV/bb08/pq_${M}_${l}_${tag}"
    sh_ "rm -rf $dir && mkdir -p $dir"
    {
      echo "=== $M p$l $tag ==="
      sh_ "cd $dir && timeout 180 env $envs LD_LIBRARY_PATH=$DEV/lib $DEV/bin/profqos_v \
             ../${M}_p${l}.dla -i ../${M}.in.bin -o pq_out.bin -c 20"
      sh_ "cd $dir && ls -l"
    } >> "$EV/profqos-$M.txt"
    for f in execution_result.json dla_info.json; do
      if sh_ "test -f $dir/$f && echo yes" | grep -q yes; then
        "$ADB" pull "$dir/$f" "$(cygpath -w "$EV/${f%.json}-pq-${M}-p${l}-${tag}.json")" >/dev/null
        echo "  pulled $f ($tag p$l)"
      fi
    done
  }
  pass clean ""
  pass L3 "MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3"
done
cat "$EV/profqos-$M.txt"
echo "----- dmesg verdict -----"
BAD=$(sh_ "dmesg | grep -icE 'MDLA timeout|mdw_cmd_complete.*ret\(' " || echo 0)
echo "BADMARKERS=$BAD"
