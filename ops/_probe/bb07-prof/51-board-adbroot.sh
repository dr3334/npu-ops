#!/usr/bin/env bash
# 51-board-adbroot.sh — after the reboot, can we still get a privileged shell?
set -uo pipefail
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }
echo "== id before =="; sh_ "id"
echo "== adb root =="; "$ADB" root 2>&1 | tr -d '\r'; sleep 4; "$ADB" wait-for-device
echo "== id after =="; sh_ "id"
echo "== dmesg access =="; sh_ "dmesg | tail -2"
echo "== known-good op =="
sh_ "cd $DEV && timeout 60 ./bin/neuronrt -m hw -a dla/fc_1x64x64_int8.dla -i dla/fc_1x64x64_int8.in.bin -o dla/hc8.bin -c 10" | tail -3
