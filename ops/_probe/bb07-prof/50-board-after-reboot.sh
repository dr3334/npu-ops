#!/usr/bin/env bash
# 50-board-after-reboot.sh — the box rebooted by itself; is the APU healthy again?
set -uo pipefail
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }
echo "== uptime =="; sh_ "uptime"
echo "== known-good op =="
sh_ "cd $DEV && timeout 60 ./bin/neuronrt -m hw -a dla/fc_1x64x64_int8.dla -i dla/fc_1x64x64_int8.in.bin -o dla/hc7.bin -c 10" | tail -3
echo "== dmesg: any apusys timeout since boot? =="
sh_ "dmesg | grep -ciE 'MDLA timeout|mdw_cmd_complete.*ret\(-5'"
