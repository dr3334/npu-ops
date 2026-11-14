#!/usr/bin/env bash
# 23-c-sweep.sh — does the repeat count N itself move ms/inf?  (control for ticket 08a)
#
#   NPU_LOCK_HELD=1 ./tools/devlock.sh run -- bash ops/_probe/bb08-prof/23-c-sweep.sh fc 0 "20 50 100 200" 3
#
# 08 measured `fc -c 20` and got 0.3519 ms/inf; 08a measured `fc -c 200` and saw ~0.65 ms/inf.
# Before either number can be quoted, we must know whether that gap is the level, the day,
# or N itself.  Same product, same input, only N changes — paired and alternating, so a
# drifting board cannot favour one N (reports/timings.md discipline).
#
# Writes board/08a/23-c-sweep-<model>.tsv: round  n  total_ms  ms_per_inf
set -uo pipefail
M="${1:?model}"; L="${2:-0}"; NS="${3:-20 50 100 200}"; ROUNDS="${4:-3}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EV="$ROOT/ops/_probe/bb08-prof/board/08a"
ADB="${ADB:-/d/adb/platform-tools/adb}"
DEV=/data/local/tmp/npuops
export MSYS_NO_PATHCONV=1
mkdir -p "$EV"
sh_() { "$ADB" shell "$1" 2>&1 | tr -d "\r"; }
TSV="$EV/23-c-sweep-$M.tsv"
: > "$TSV"
printf "round\tn\ttotal_ms\tms_per_inf\n" >> "$TSV"

for r in $(seq 1 "$ROUNDS"); do
  for n in $NS; do
    OUT=$(sh_ "cd $DEV/bb08 && timeout 120 $DEV/bin/neuronrt -m hw -a ${M}_p${L}.dla -i ${M}.in.bin -o csw_${M}_${n}.bin -c $n" | tee "$EV/23-csw-${M}-n${n}-r${r}.txt")
    LINE=$(echo "$OUT" | grep -m1 "Total inference time")
    TOT=$(echo "$LINE" | sed -n "s/.*= \([0-9.]*\) ms.*/\1/p")
    PER=$(echo "$LINE" | sed -n "s/.*(\([0-9.]*\) ms\/inf).*/\1/p")
    printf "%s\t%s\t%s\t%s\n" "$r" "$n" "${TOT:-NA}" "${PER:-NA}" >> "$TSV"
    echo "round$r ${M}_p${L}  c=$n  total=${TOT:-NA} ms  per_inf=${PER:-NA} ms"
  done
done

echo "----- dmesg verdict -----"
BAD=$(sh_ "dmesg | grep -icE 'MDLA timeout|ret\(-5'" | tail -1)
BAD=${BAD:-0}
echo "BADMARKERS=$BAD"
[ "$BAD" = "0" ] && echo "BOARD_STATE=dmesg-clean" || echo "BOARD_STATE=BAD(need adb reboot)"
