#!/usr/bin/env bash
# 48-board-recheck.sh — did the APU recover on its own? (locked, short)
set -uo pipefail
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }
"$ADB" devices | tr -d '\r'
sh_ "cd $DEV && timeout 40 ./bin/neuronrt -m hw -a dla/fc_1x64x64_int8.dla -i dla/fc_1x64x64_int8.in.bin -o dla/hc4.bin -c 5" | tail -3
