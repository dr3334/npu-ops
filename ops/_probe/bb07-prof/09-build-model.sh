#!/usr/bin/env bash
# 09-build-model.sh — build the tflite for fc_1x64x64_int8 + a .dla (offline).
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
cd "$R" || exit 2
B=ops/_probe/bb07-prof/build
mkdir -p "$B"
M=$B/fc.tflite
if [ ! -f "$M" ]; then
  /usr/bin/python3 tools/build_ops.py fc_1x64x64_int8 --out "$M" || exit 3
fi
ls -l "$M"
./tools/compile.sh "$M" "$B/fc_v0.dla" || exit 4
ls -l "$B"
