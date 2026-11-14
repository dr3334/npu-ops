#!/usr/bin/env bash
# 24-latency-08a.sh — ticket 08a: the total-latency measurement, writing only into board/08a/.
#
#   NPU_LOCK_HELD=1 ./tools/devlock.sh run -- bash ops/_probe/bb08-prof/24-latency-08a.sh
#
# Same discipline as 21-latency.sh (paired and alternating p0,p1,p2,p3 inside every round,
# median as the headline, pairs as the only cross-run comparison), but:
#   * the products are driven with -c 200 (fc) / -c 50 (dec), not 08's -c 20 / -c 10 --
#     a larger N averages the per-run jitter into total_ms before we divide;
#   * 9 rounds instead of 3, because 08's 3-round table was swamped by its own MAD;
#   * output lands in board/08a/ so 08's board/ evidence (which reports/compiler-blackbox-08.md
#     quotes) is never overwritten by a re-measurement.
#
# Writes board/08a/latency-<model>.tsv: round  product  total_ms  ms_per_inf
#        board/08a/lat-<model>-p<level>-r<round>.txt: the raw console line
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EV="$ROOT/ops/_probe/bb08-prof/board/08a"
ADB="${ADB:-/d/adb/platform-tools/adb}"
DEV=/data/local/tmp/npuops
export MSYS_NO_PATHCONV=1
mkdir -p "$EV"
sh_() { "$ADB" shell "$1" 2>&1 < /dev/null | tr -d "\r"; }

# model:repeats:rounds
PLAN=${PLAN:-"fc:200:9 dec:50:9"}

for spec in $PLAN; do
  IFS=":" read -r M N ROUNDS <<< "$spec"
  [ -n "${M:-}" ] || continue
  TSV="$EV/latency-$M.tsv"
  : > "$TSV"
  printf "round\tproduct\ttotal_ms\tms_per_inf\n" >> "$TSV"
  echo "===== $M  -c $N  x $ROUNDS rounds (paired alternating) ====="
  for r in $(seq 1 "$ROUNDS"); do
    for l in 0 1 2 3; do
      OUT=$(sh_ "cd $DEV/bb08 && timeout 120 $DEV/bin/neuronrt -m hw -a ${M}_p${l}.dla -i ${M}.in.bin -o lat_${M}_p${l}.bin -c $N" | tee "$EV/lat-${M}-p${l}-r${r}.txt")
      LINE=$(echo "$OUT" | grep -m1 "Total inference time")
      TOT=$(echo "$LINE" | sed -n "s/.*= \([0-9.]*\) ms.*/\1/p")
      PER=$(echo "$LINE" | sed -n "s/.*(\([0-9.]*\) ms\/inf).*/\1/p")
      printf "%s\t%s\t%s\t%s\n" "$r" "${M}_p${l}" "${TOT:-NA}" "${PER:-NA}" >> "$TSV"
      echo "round$r ${M}_p${l}  total=${TOT:-NA} ms  per_inf=${PER:-NA} ms"
    done
  done
  echo "----- output equality across levels ($M) -----"
  sh_ "cd $DEV/bb08 && for f in lat_${M}_p*.bin; do echo \"\$f \$(md5sum \$f | cut -c1-16)\"; done"
done

echo "----- dmesg verdict -----"
BAD=$(sh_ "dmesg | grep -icE 'MDLA timeout|ret\(-5'" | tail -1)
BAD=${BAD:-0}
echo "BADMARKERS=$BAD"
[ "$BAD" = "0" ] && echo "BOARD_STATE=dmesg-clean" || echo "BOARD_STATE=BAD(need adb reboot)"
