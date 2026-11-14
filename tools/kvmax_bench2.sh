#!/usr/bin/env bash
# Ticket kv-maxlen: bench v2 -- the device clock `date +%s%N` produced a NEGATIVE delta at
# C=8192 (see ops/kvmax/timing.jsonl round 1..3), so time the adb round trip with the HOST
# clock (monotonic, and the constant adb overhead cancels in the (T_N - T_1)/(N-1) form)
# and print the device-side raw epochs next to it, so the anomaly is visible, not hidden.
#   ./tools/devlock.sh run -- ./tools/kvmax_bench2.sh "2048 4096 8192"
set -uo pipefail
CS="${1:-2048 4096 8192}"
ROUNDS="${ROUNDS:-3}"
N="${N:-10}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV=/data/local/tmp/npuops
ADB=/d/adb/platform-tools/adb
export MSYS_NO_PATHCONV=1
cd "$ROOT" || exit 2
J=ops/kvmax/timing2.jsonl
: > "$J"
now() { date +%s%N; }

run_c() {   # prints: host_ms=<n> dev=<raw "rc=.. ms=..">
  local C=$1 cnt=$2
  local ins="-i kvmax/c$C/x.bin -i kvmax/c$C/mask.bin -i kvmax/c$C/cos14.bin -i kvmax/c$C/sin14.bin -i kvmax/c$C/cos2.bin -i kvmax/c$C/sin2.bin -i kvmax/c$C/k_in.bin -i kvmax/c$C/v_in.bin"
  local ots="-o kvmax/c$C/out_hidden.bin -o kvmax/c$C/out_k.bin -o kvmax/c$C/out_v.bin"
  local a b
  a=$(now)
  local r; r=$("$ADB" shell "cd $DEV && LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a kvmax/c$C/model.dla $ins $ots -c $cnt >/dev/null 2>&1; echo rc=\$?" | tr -d "\r")
  b=$(now)
  echo "host_ms=$(( (b-a)/1000000 )) dev=$r"
}

r=1
while [ "$r" -le "$ROUNDS" ]; do
  if [ $((r % 2)) = 1 ]; then ORD="$CS"; else ORD="$(printf "%s\n" $CS | tac)"; fi
  for C in $ORD; do
    A="$(run_c "$C" 1)"; B="$(run_c "$C" "$N")"
    TA=$(printf "%s" "$A" | sed -n "s/host_ms=\([0-9]*\).*/\1/p")
    TB=$(printf "%s" "$B" | sed -n "s/host_ms=\([0-9]*\).*/\1/p")
    PI=$(awk -v a="$TA" -v b="$TB" -v n="$N" "BEGIN{printf \"%.2f\", (b-a)/(n-1)}")
    RA=$(printf "%s" "$A" | sed -n "s/.*dev=\(rc=[0-9]*\).*/\1/p")
    RB=$(printf "%s" "$B" | sed -n "s/.*dev=\(rc=[0-9]*\).*/\1/p")
    echo "[r$r] C=$C t1=${TA}ms tN=${TB}ms($N) per_invoke=${PI}ms  ($A | $B)"
    printf '{"round":%d,"C":%d,"t1_host_ms":%s,"tN_host_ms":%s,"n":%d,"per_invoke_ms":%s,"rc1":"%s","rcN":"%s"}\n' \
      "$r" "$C" "$TA" "$TB" "$N" "$PI" "$RA" "$RB" >> "$J"
  done
  r=$((r + 1))
done
echo "[bench2] wrote $J"
