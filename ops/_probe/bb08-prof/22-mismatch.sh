#!/usr/bin/env bash
# 22-mismatch.sh — capture the level-gating text verbatim: a product compiled with a lower -P level
# than the runtime asks for, on the board AND on the host (-m null). Console + logcat tag=neuron.
#
#   NPU_LOCK_HELD=1 ./tools/devlock.sh run -- bash ops/_probe/bb08-prof/22-mismatch.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EV="$ROOT/ops/_probe/bb08-prof/board"
ADB="${ADB:-/d/adb/platform-tools/adb}"
DEV=/data/local/tmp/npuops
export MSYS_NO_PATHCONV=1
mkdir -p "$EV"
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }

: > "$EV/level-gating-board.txt"
"$ADB" logcat -c >/dev/null 2>&1
for pair in "fc_p0 1" "fc_p0 2" "fc_p0 3" "fc_p1 2" "fc_p1 3" "fc_p2 3" "fc_p3 3"; do
  set -- $pair; D="$1"; L="$2"
  {
    echo "=== DLA=$D  ask=$L ==="
    sh_ "cd $DEV/bb08 && timeout 90 env MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=$L $DEV/bin/neuronrt -m hw -a $D.dla -i fc.in.bin -o gating.bin -c 3"
    echo "--- rc/console above; logcat tag=neuron since last clear: ---"
    "$ADB" logcat -d -v time -s neuron 2>&1 | tr -d '\r'
    echo
  } >> "$EV/level-gating-board.txt"
  "$ADB" logcat -c >/dev/null 2>&1
done
cat "$EV/level-gating-board.txt"
echo "----- dmesg verdict -----"
BAD=$(sh_ "dmesg | grep -icE 'MDLA timeout|ret\\(-5'" | tail -1)
BAD=${BAD:-0}
echo "BADMARKERS=$BAD"
