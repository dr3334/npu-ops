#!/usr/bin/env bash
# 38-board-perop.sh — the last shot at per-LAYER numbers: the runtime's own profiler tables
# with MTKNN_PER_OP_PROFILE on hardware.  BOARD: caller must hold the lock.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
EV="$R/ops/_probe/bb07-prof/board"
mkdir -p "$EV"
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }

for lvl in 1 2 3; do
  echo "===== MTKNN_PROFILING_LEVEL=$lvl + PER_OP, full_p3, -c 5 ====="
  sh_ "cd $DEV && rm -rf prof$lvl && mkdir -p prof$lvl && cd prof$lvl && MTKNN_ENABLE_PROFILER=1 MTKNN_PER_OP_PROFILE=1 MTKNN_PROFILING_LEVEL=$lvl ../bin/neuronrt -m hw -a ../bb07/full_p3.dla -i ../bb07/din.bin -o ../bb07/p$lvl.bin -c 5" | tee "$EV/perop-matrix-L$lvl.txt" | head -45
  echo
  echo "--- files written ---"
  sh_ "cd $DEV/prof$lvl && ls -la"
  echo
done

echo "===== MTKNN_PROFILER_LOG_PATH=<dir> ====="
sh_ "cd $DEV && rm -rf plog && mkdir -p plog && MTKNN_ENABLE_PROFILER=1 MTKNN_PER_OP_PROFILE=1 MTKNN_PROFILING_LEVEL=3 MTKNN_PROFILER_LOG_PATH=$DEV/plog ./bin/neuronrt -m hw -a bb07/full_p3.dla -i bb07/din.bin -o bb07/pl.bin -c 5" | tail -5
sh_ "cd $DEV/plog && ls -la"
echo
echo "===== pull the JSON reports ====="
for f in dla_info.json execution_result.json; do
  "$ADB" pull "$(sh_ "ls $DEV/plog/$f 2>/dev/null || ls $DEV/$f 2>/dev/null" | head -1)" "$(wslpath -w "$EV/$f")" 2>&1 | tail -1
done
