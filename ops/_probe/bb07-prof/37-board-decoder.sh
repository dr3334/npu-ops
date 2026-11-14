#!/usr/bin/env bash
# 37-board-decoder.sh — per-layer profile of the multi-layer decoder graph, on hardware.
# BOARD: caller must hold the lock.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
EV="$R/ops/_probe/bb07-prof/board"
B="$R/ops/_probe/bb07-prof/build3"
mkdir -p "$EV"
export MSYS_NO_PATHCONV=1
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }
push() { "$ADB" push "$(wslpath -w "$1")" "$2" >/dev/null; }

push "$B/full_p0.dla" "$DEV/bb07/full_p0.dla"
push "$B/full_p3.dla" "$DEV/bb07/full_p3.dla"
push "$B/full.dir/golden/input.bin" "$DEV/bb07/din.bin"

echo "===== A. profqos on full_p3 (-P3), -c 5 ====="
sh_ "cd $DEV && ./bin/profqos_v bb07/full_p3.dla -i bb07/din.bin -o bb07/o_full3.bin -c 5" | tee "$EV/profqos-full_p3.txt" | head -40

echo
echo "===== B. profqos on full_p0 (no -P), -c 5 ====="
sh_ "cd $DEV && ./bin/profqos_v bb07/full_p0.dla -i bb07/din.bin -o bb07/o_full0.bin -c 5" | tee "$EV/profqos-full_p0.txt" | tail -6

echo
echo "===== C. neuronrt -c 20, alternating p0 / p3, 3 rounds ====="
for r in 1 2 3; do
  for v in full_p0 full_p3; do
    printf '  round%s %-8s ' "$r" "$v"
    sh_ "cd $DEV && ./bin/neuronrt -m hw -a bb07/$v.dla -i bb07/din.bin -o bb07/n_$v.bin -c 20" | grep -E "ms/inf" | tr -d '\n'
    echo
  done
done | tee "$EV/neuronrt-alternating.txt"
