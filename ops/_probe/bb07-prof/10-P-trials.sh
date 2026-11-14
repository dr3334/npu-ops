#!/usr/bin/env bash
# 10-P-trials.sh — probe -P / --profiling-level on ncc-tflite (offline, compile only).
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
cd "$R" || exit 2
B=ops/_probe/bb07-prof/build
M=$B/fc.tflite
try() {
  local name="$1"; shift
  ./tools/compile.sh "$M" "$B/$name.dla" "$@" > "$B/$name.stdout" 2>&1
  local rc=$?
  local sz=$(stat -c %s "$B/$name.dla" 2>/dev/null || echo 0)
  printf '=== %-10s rc=%-2s size=%-7s flags=[%s]\n' "$name" "$rc" "$sz" "$*"
  grep -iE "profil|invalid option|does not exist|error" "$B/$name.compile.log" | head -5
}
try v0b
try p1 -P1
try p1sp "-P 1"
try p2 -P2
try p3 -P3
try pl23 --profiling-level=2
try pl3 --profiling-level=3
try dbg --gen-debug-info
try dbgp2 --gen-debug-info -P2
echo
echo "== sizes =="
for f in "$B"/*.dla; do printf '%8s  %s\n' "$(stat -c %s "$f")" "$(basename "$f")"; done
