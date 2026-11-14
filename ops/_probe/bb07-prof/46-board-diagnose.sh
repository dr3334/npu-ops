#!/usr/bin/env bash
# 46-board-diagnose.sh — the board stopped running anything; what does the kernel say?
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
EV="$R/ops/_probe/bb07-prof/board"
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }

echo "== dmesg: apusys / mdw / mdla (last 40 matching) =="
sh_ "dmesg | grep -iE 'apusys|mdw|mdla|apu' | tail -40" | tee "$EV/dmesg-apu.txt"
echo
echo "== /dev/apusys present? =="
sh_ "ls -l /dev/apusys"
echo
echo "== processes holding the APU =="
sh_ "ps -A | grep -iE 'neuron|npu' | head"
echo
echo "== retry the known-good op after 20 s =="
sleep 20
sh_ "cd $DEV && timeout 40 ./bin/neuronrt -m hw -a dla/fc_1x64x64_int8.dla -i dla/fc_1x64x64_int8.in.bin -o dla/hc2.bin -c 5" | tail -3
echo
echo "== and with the profiler env off/on, -c 1 =="
sh_ "cd $DEV && timeout 30 env -u MTKNN_ENABLE_PROFILER -u MTKNN_PROFILING_LEVEL -u MTKNN_PER_OP_PROFILE ./bin/neuronrt -m hw -a dla/fc_1x64x64_int8.dla -i dla/fc_1x64x64_int8.in.bin -o dla/hc3.bin -c 1" | tail -3
