#!/usr/bin/env bash
# 21-latency.sh — CLEAN total latency (-c N, no profiler env) for every level of one model,
# PAIRED AND ALTERNATING: p0,p1,p2,p3,p0,p1,p2,p3,... so a drifting board cannot favour a level
# (reports/timings.md discipline: median is the headline, pairs are the only cross-run comparison).
#
#   NPU_LOCK_HELD=1 ./tools/devlock.sh run -- bash ops/_probe/bb08-prof/21-latency.sh fc 20 3
#
# Writes board/latency-<model>.tsv:  round  product  total_ms  ms_per_inf
set -uo pipefail
M="${1:?model}"; N="${2:-20}"; ROUNDS="${3:-3}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EV="$ROOT/ops/_probe/bb08-prof/board"
ADB="${ADB:-/d/adb/platform-tools/adb}"
DEV=/data/local/tmp/npuops
export MSYS_NO_PATHCONV=1
mkdir -p "$EV"
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }
TSV="$EV/latency-$M.tsv"
: > "$TSV"
printf 'round\tproduct\ttotal_ms\tms_per_inf\n' >> "$TSV"

for r in $(seq 1 "$ROUNDS"); do
  for l in 0 1 2 3; do
    OUT=$(sh_ "cd $DEV/bb08 && timeout 120 ./bin/neuronrt -m hw -a ${M}_p${l}.dla -i ${M}.in.bin -o lat_${M}_p${l}.bin -c $N" | tee "$EV/lat-${M}-p${l}-r${r}.txt")
    LINE=$(echo "$OUT" | grep -m1 "Total inference time")
    TOT=$(echo "$LINE" | sed -n 's/.*= \([0-9.]*\) ms.*/\1/p')
    PER=$(echo "$LINE" | sed -n 's/.*(\([0-9.]*\) ms\/inf).*/\1/p')
    printf '%s\t%s\t%s\t%s\n' "$r" "${M}_p${l}" "${TOT:-NA}" "${PER:-NA}" >> "$TSV"
    echo "round$r ${M}_p${l}  total=${TOT:-NA} ms  per_inf=${PER:-NA} ms"
  done
done

echo "----- output equality across levels (same input, same math) -----"
sh_ "cd $DEV/bb08 && for f in lat_${M}_p*.bin; do echo \"\$f \$(md5sum \$f | cut -c1-16)\"; done"
echo "----- dmesg verdict -----"
BAD=$(sh_ "dmesg | grep -icE 'MDLA timeout|mdw_cmd_complete.*ret\(' " || echo 0)
echo "BADMARKERS=$BAD"
[ "$BAD" = "0" ] && echo "BOARD_STATE=dmesg-clean" || echo "BOARD_STATE=BAD(need adb reboot)"
