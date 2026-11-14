#!/usr/bin/env bash
# 28-gdb.sh — why does the harness SIGFPE? backtrace it.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
T=/mnt/d/bb07-tmp/smoke
H=/mnt/d/bb07-tmp/profqos_host
B="$R/ops/_probe/bb07-prof/build2"
IN="$R/ops/_probe/bb07-prof/build/golden/input.bin"
cd "$T" || exit 2
gdb -q -batch \
  -ex "set pagination off" \
  -ex run \
  -ex "bt 12" \
  -ex "info registers rip" \
  --args "$H" "$B/q003.dla" -i "$IN" -o "$T/dbg.out" -c 1 -n 2>&1 | tail -30
