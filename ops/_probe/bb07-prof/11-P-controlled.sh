#!/usr/bin/env bash
# 11-P-controlled.sh — same-length output names so path length cancels; measure -P effect.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
cd "$R" || exit 2
B=ops/_probe/bb07-prof/build2
mkdir -p "$B"
M=ops/_probe/bb07-prof/build/fc.tflite
run() {
  local name="$1"; shift
  ./tools/compile.sh "$M" "$B/$name.dla" "$@" > "$B/$name.stdout" 2>&1
  local rc=$?
  local sz=$(stat -c %s "$B/$name.dla" 2>/dev/null || echo 0)
  printf '%-6s rc=%-2s size=%-7s flags=[%s]\n' "$name" "$rc" "$sz" "$*"
}
run q000
run q001 -P1
run q002 -P2
run q003 -P3
run q012 --profiling-level=2
run q013 --profiling-level=3
run q100 --gen-debug-info
run q103 --gen-debug-info -P3
echo
echo "== any 'profil' line in logs =="
grep -ihE "profil" "$B"/*.compile.log | sort | uniq -c | head -20
echo
echo "== file sizes sorted =="
for f in q000 q001 q002 q003 q012 q013 q100 q103; do printf '%s %s\n' "$(stat -c %s "$B/$f.dla")" "$f"; done
