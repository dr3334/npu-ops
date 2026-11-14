#!/usr/bin/env bash
# 44-board-xruntime-and-levels.sh — trial 1 on the board + level gating, full stdout.
# BOARD: caller must hold the lock.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
EV="$R/ops/_probe/bb07-prof/board"
mkdir -p "$EV"
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }

echo "########## 1. --Xruntime on hardware: anything observable? ##########"
{
for o in "" "profiling-level:3" "profilingLevel=3" "garbage" "nbThreads:2"; do
  echo "--- opt=[$o] ---"
  if [ -z "$o" ]; then
    sh_ "cd $DEV && ./bin/neuronrt -m hw -a bb07/full_p3.dla -i bb07/din.bin -o bb07/x.bin -c 3"
  else
    sh_ "cd $DEV && ./bin/neuronrt -m hw -a bb07/full_p3.dla -i bb07/din.bin -o bb07/x.bin -c 3 --Xruntime '$o'"
  fi
done
} 2>&1 | tee "$EV/xruntime-hw.txt"

echo
echo "########## 2. level gating on hardware (does -P matter at run time?) ##########"
{
for d in full_p0 full_p3; do
  for lvl in 1 3; do
    echo "--- $d, MTKNN_PROFILING_LEVEL=$lvl ---"
    sh_ "cd $DEV && MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=$lvl ./bin/neuronrt -m hw -a bb07/$d.dla -i bb07/din.bin -o bb07/g.bin -c 3" | grep -iE "profil|only supports|Total inference|FPS"
  done
done
} 2>&1 | tee "$EV/level-gating-hw.txt"

echo
echo "########## 3. full console at level 3 on hardware (any tables printed?) ##########"
sh_ "cd $DEV && rm -rf lvl3 && mkdir lvl3 && MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3 ./bin/neuronrt -m hw -a bb07/full_p3.dla -i bb07/din.bin -o bb07/l3.bin -c 3" > "$EV/level3-console-hw.txt" 2>&1
echo "  lines: $(wc -l < "$EV/level3-console-hw.txt")"
grep -c "Summary" "$EV/level3-console-hw.txt" || true
head -12 "$EV/level3-console-hw.txt"
