#!/usr/bin/env bash
# 07-board-check.sh — device reachable? (BOARD: caller must hold the lock)
set -uo pipefail
ADB=/mnt/d/adb/platform-tools/adb.exe
echo "== adb devices =="
"$ADB" devices 2>&1 | tr -d '\r'
echo "== board platform =="
"$ADB" shell 'getprop ro.board.platform; getprop ro.build.version.release' 2>&1 | tr -d '\r'
echo "== /data/local/tmp/npuops =="
"$ADB" shell 'ls /data/local/tmp/npuops; echo ---; ls /data/local/tmp/npuops/lib | head' 2>&1 | tr -d '\r'
