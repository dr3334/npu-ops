#!/usr/bin/env bash
# 27-profqos-smoke.sh — run the harness against the host runtime (null device). Validates the
# API plumbing and shows what getProfiledQoSData() returns without hardware.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
cd "$R" || exit 2
T=/mnt/d/bb07-tmp/smoke
rm -rf "$T"; mkdir -p "$T"
H=/mnt/d/bb07-tmp/profqos_host
IN="$R/ops/_probe/bb07-prof/build/golden/input.bin"
B="$R/ops/_probe/bb07-prof/build2"

for d in q000 q003; do
  echo "===== $d : null device, no profiler env ====="
  ( cd "$T" && "$H" "$B/$d.dla" -i "$IN" -o "$T/$d.out" -c 3 -n ) > "$T/$d.log" 2>&1
  echo "  rc=$?"
  cat "$T/$d.log"
  echo
done

echo "===== q003 : MTKNN_PROFILING_LEVEL=3 ====="
( cd "$T" && MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3 "$H" "$B/q003.dla" -i "$IN" -o "$T/q003b.out" -c 3 -n ) > "$T/q003b.log" 2>&1
echo "  rc=$?"
sed -n '1,40p' "$T/q003b.log"
echo
echo "== output bytes identical to the neuronrt run? =="
ls -l "$T"/*.out 2>/dev/null
