#!/usr/bin/env bash
# 49-board-state.sh — uptime / vendor lib presence / one retry
set -uo pipefail
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }
echo "== uptime =="; sh_ "uptime"
echo "== vendor lib =="; sh_ "ls -l /vendor/lib64/libneuron_runtime.so /vendor/lib64/mt6897/libneuron_runtime.so 2>&1"
echo "== /data/local/tmp/npuops/lib =="; sh_ "ls -l $DEV/lib | head -5"
echo "== retry with explicit LD_LIBRARY_PATH =="
sh_ "cd $DEV && LD_LIBRARY_PATH=/vendor/lib64 timeout 40 ./bin/neuronrt -m hw -a dla/fc_1x64x64_int8.dla -i dla/fc_1x64x64_int8.in.bin -o dla/hc5.bin -c 5" | tail -3
echo "== retry plain again =="
sh_ "cd $DEV && timeout 40 ./bin/neuronrt -m hw -a dla/fc_1x64x64_int8.dla -i dla/fc_1x64x64_int8.in.bin -o dla/hc6.bin -c 5" | tail -3
