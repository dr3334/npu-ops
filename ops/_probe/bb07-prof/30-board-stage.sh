#!/usr/bin/env bash
# 30-board-stage.sh — stage the harness + assets on the board and do ONE profqos run.
# BOARD: caller must hold the lock (tools/devlock.sh run -- bash this).
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
EV="$R/ops/_probe/bb07-prof/board"
mkdir -p "$EV"
export MSYS_NO_PATHCONV=1

echo "== device =="
"$ADB" devices | tr -d '\r'
"$ADB" shell 'getprop ro.board.platform; getprop ro.build.version.release' | tr -d '\r'

echo "== runtime libs on device =="
"$ADB" shell "ls $DEV/lib | head -20" | tr -d '\r'
echo "== does the runtime .so there export getProfiledQoSData? (name check only) =="
"$ADB" shell "ls -l $DEV/lib/libneuron_runtime.so*" | tr -d '\r'

echo "== stage assets =="
push() { "$ADB" push "$(wslpath -w "$1")" "$2" >/dev/null && echo "  pushed $2"; }
"$ADB" shell "mkdir -p $DEV/bb07" >/dev/null
push /mnt/d/bb07-tmp/profqos "$DEV/bin/profqos"
for f in q000 q001 q002 q003 q100; do push "$R/ops/_probe/bb07-prof/build2/$f.dla" "$DEV/bb07/$f.dla"; done
push "$R/ops/_probe/bb07-prof/build/golden/input.bin" "$DEV/bb07/in.bin"
"$ADB" shell "chmod +x $DEV/bin/profqos" >/dev/null

echo
echo "== profqos on hardware: q003 (-P3), -c 20 =="
"$ADB" shell "cd $DEV && LD_LIBRARY_PATH=\$PWD/lib ./bin/profqos bb07/q003.dla -i bb07/in.bin -o bb07/o.bin -c 20" \
  2>&1 | tr -d '\r' | tee "$EV/profqos-q003-first.txt"
echo "  rc=${PIPESTATUS[0]}"
