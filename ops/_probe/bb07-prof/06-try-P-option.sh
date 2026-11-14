#!/usr/bin/env bash
# 06-try-P-option.sh — does ncc-tflite accept -P / --profiling-level? What does it change?
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
cd "$R" || exit 2
B=ops/_probe/bb07-prof/build
mkdir -p "$B"
M=ops/fc_1x64x64_int8/model.tflite
ls -l "$M" || exit 3

try() { # name, extra flags...
  local name="$1"; shift
  ./tools/compile.sh "$M" "$B/$name.dla" "$@" > "$B/$name.stdout" 2>&1
  local rc=$?
  local sz=$(stat -c %s "$B/$name.dla" 2>/dev/null || echo 0)
  printf '%-14s rc=%-2s size=%-7s flags=%s\n' "$name" "$rc" "$sz" "$*"
  grep -iE "profil" "$B/$name.compile.log" 2>/dev/null | head -6
}

try v0
try p1 -P1
try p2 -P2
try p3 -P3
try pl_eq2 --profiling-level=2
try dbg --gen-debug-info
echo "== sizes =="
ls -l "$B"/*.dla 2>/dev/null | awk '{print $5, $9}'
