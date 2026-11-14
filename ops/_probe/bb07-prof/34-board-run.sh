#!/usr/bin/env bash
# 34-board-run.sh — the real run: per-sub-command profile + neuronrt totals, on hardware.
# BOARD: caller must hold the lock.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
EV="$R/ops/_probe/bb07-prof/board"
mkdir -p "$EV"
export MSYS_NO_PATHCONV=1
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }

"$ADB" push "$(wslpath -w /mnt/d/bb07-tmp/profqos_vendor)" "$DEV/bin/profqos_v" >/dev/null
sh_ "chmod +x $DEV/bin/profqos_v"

echo "===== A. profqos (vendor-linked) on hardware ====="
for d in q003 q000 q001; do
  echo "--- $d  -c 20 ---"
  sh_ "cd $DEV && ./bin/profqos_v bb07/$d.dla -i bb07/in.bin -o bb07/out_$d.bin -c 20" | tee "$EV/profqos-$d.txt"
done

echo
echo "===== B. neuronrt -c 20 totals (same DLAs) ====="
for d in q003 q000 q001 q100; do
  echo "--- $d ---"
  sh_ "cd $DEV && ./bin/neuronrt -m hw -a bb07/$d.dla -i bb07/in.bin -o bb07/nrt_$d.bin -c 20" | grep -E "Inference|ms/inf|FPS" | tee "$EV/neuronrt-$d.txt"
done

echo
echo "===== C. do the two output files agree? ====="
sh_ "cd $DEV/bb07 && for d in q003 q000 q001; do echo -n \"\$d \"; cmp -s out_\$d.bin nrt_\$d.bin && echo same || echo DIFF; done"
