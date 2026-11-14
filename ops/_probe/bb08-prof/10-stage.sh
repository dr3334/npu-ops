#!/usr/bin/env bash
# 10-stage.sh — push the eight products + inputs to the board.  BOARD: caller holds devlock.
#
#   NPU_LOCK_HELD=1 ./tools/devlock.sh run -- bash ops/_probe/bb08-prof/10-stage.sh
#
# Layout on device: /data/local/tmp/npuops/bb08/{fc_p0..3.dla,dec_p0..3.dla,fc.in.bin,dec.in.bin}
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
B="$ROOT/ops/_probe/bb08-prof/build"
EV="$ROOT/ops/_probe/bb08-prof/board"
ADB="${ADB:-/d/adb/platform-tools/adb}"
DEV=/data/local/tmp/npuops
export MSYS_NO_PATHCONV=1
mkdir -p "$EV"
wp() { cygpath -w "$1"; }
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }

{
echo "== device =="
"$ADB" devices | tr -d '\r'
sh_ 'getprop ro.board.platform; getprop ro.build.version.release; getprop ro.build.version.incremental'
echo "== uptime / id =="
sh_ 'uptime; id'
echo "== abi / runtime =="
sh_ "ls -l $DEV/bin/neuronrt $DEV/lib/libneuron_runtime.so 2>&1"
} 2>&1 | tee "$EV/state-before.txt"

echo "== stage =="
sh_ "mkdir -p $DEV/bb08"
for m in fc dec; do
  for l in 0 1 2 3; do
    "$ADB" push "$(wp "$B/${m}_p${l}.dla")" "$DEV/bb08/${m}_p${l}.dla" >/dev/null || exit 5
  done
done
"$ADB" push "$(wp "$B/golden/input.bin")"     "$DEV/bb08/fc.in.bin"  >/dev/null || exit 5
"$ADB" push "$(wp "$B/dec/golden/input.bin")" "$DEV/bb08/dec.in.bin" >/dev/null || exit 5
sh_ "cd $DEV/bb08 && ls -l"
echo "== does the harness binary survive from bb07? =="
sh_ "ls -l $DEV/bin/profqos_v $DEV/bin/profqos 2>&1"
