#!/usr/bin/env bash
# 53-board-repro.sh — second sample, healthy board, no profiler env (safe: no level-3 PMU run).
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
EV="$R/ops/_probe/bb07-prof/board"
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }
{
echo "== profqos full_p3 -c 20 (no profiler env) =="
sh_ "cd $DEV && timeout 90 ./bin/profqos_v bb07/full_p3.dla -i bb07/din.bin -o bb07/r1.bin -c 20"
echo "== neuronrt full_p3 -c 20 x2 =="
for i in 1 2; do sh_ "cd $DEV && timeout 90 ./bin/neuronrt -m hw -a bb07/full_p3.dla -i bb07/din.bin -o bb07/r2.bin -c 20" | grep -E "ms/inf"; done
echo "== profqos full_p0 -c 20 =="
sh_ "cd $DEV && timeout 90 ./bin/profqos_v bb07/full_p0.dla -i bb07/din.bin -o bb07/r3.bin -c 20"
} 2>&1 | tee "$EV/repro-healthy.txt"
