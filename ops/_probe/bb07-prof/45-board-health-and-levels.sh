#!/usr/bin/env bash
# 45-board-health-and-levels.sh — is the board still healthy after the level-3 runs? then the
# level-gating matrix (with per-command timeout so nothing can hang the lock).
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
EV="$R/ops/_probe/bb07-prof/board"
mkdir -p "$EV"
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }

echo "== health: known-good op on hardware =="
sh_ "cd $DEV && timeout 40 ./bin/neuronrt -m hw -a dla/fc_1x64x64_int8.dla -i dla/fc_1x64x64_int8.in.bin -o dla/hc.bin -c 5" | tail -4

echo
echo "== health: full_p3 again, -c 5 =="
sh_ "cd $DEV && timeout 60 ./bin/neuronrt -m hw -a bb07/full_p3.dla -i bb07/din.bin -o bb07/h3.bin -c 5" | tail -4

echo
echo "== level gating (profiler env + -P levels) =="
{
for d in full_p0 full_p3; do
  for lvl in 1 3; do
    echo "--- $d :: MTKNN_PROFILING_LEVEL=$lvl ---"
    sh_ "cd $DEV && timeout 90 env MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=$lvl ./bin/neuronrt -m hw -a bb07/$d.dla -i bb07/din.bin -o bb07/g.bin -c 3" \
      | grep -iE "profil|only supports|Total inference|FPS|Summary|LEVEL" | head -8
  done
done
} 2>&1 | tee "$EV/level-gating-hw.txt"

echo
echo "== full console at level 3 (any tables printed on the board?) =="
sh_ "cd $DEV && rm -rf lvl3 && mkdir lvl3 && cd lvl3 && timeout 90 env MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3 ../bin/neuronrt -m hw -a ../bb07/full_p3.dla -i ../bb07/din.bin -o ../bb07/l3.bin -c 3" > "$EV/level3-console-hw.txt" 2>&1
echo "  lines: $(wc -l < "$EV/level3-console-hw.txt")  summaries: $(grep -c Summary "$EV/level3-console-hw.txt" || true)"
head -16 "$EV/level3-console-hw.txt"
