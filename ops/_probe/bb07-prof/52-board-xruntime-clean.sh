#!/usr/bin/env bash
# 52-board-xruntime-clean.sh — trial 1 on a HEALTHY board (no profiler env, no -P).
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
EV="$R/ops/_probe/bb07-prof/board"
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }
{
for o in "" "garbage" "profiling-level:3" "nbThreads:2" "{\"parallelismOption\":\"single\"}"; do
  echo "--- opt=[$o] ---"
  if [ -z "$o" ]; then
    sh_ "cd $DEV && timeout 60 ./bin/neuronrt -m hw -a bb07/full_p0.dla -i bb07/din.bin -o bb07/x0.bin -c 5"
  else
    sh_ "cd $DEV && timeout 60 ./bin/neuronrt -m hw -a bb07/full_p0.dla -i bb07/din.bin -o bb07/x1.bin -c 5 --Xruntime '$o'"
  fi
done
} 2>&1 | tee "$EV/xruntime-hw-clean.txt"
