#!/usr/bin/env bash
# Ticket kv-maxlen: board bench for the decode graph at several cache lengths C.
#   ./tools/devlock.sh run -- ./tools/kvmax_bench.sh "64 2048 4096 8192 16384"
#
# Discipline (NOTES 11.5): every C is measured in the SAME locked session, in ALTERNATION
# (round 1 ascending, round 2 descending, ...) and only RATIOS are reported; absolute ms is
# magnitude only.  per-invoke = (T(-c N) - T(-c 1)) / (N-1)   [ticket 06 method, which
# removes the one-off load of the .dla + the KV input buffers].
set -uo pipefail
CS="${1:-2048 4096 8192 16384}"
ROUNDS="${ROUNDS:-3}"
N="${N:-20}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV=/data/local/tmp/npuops
DEVD=$DEV/kvmax
ADB=/d/adb/platform-tools/adb
OUTD=ops/kvmax
export MSYS_NO_PATHCONV=1
cd "$ROOT" || exit 2
winpath() { local p="$ROOT/$1"; ( cd "$(dirname "$p")" && printf "%s/%s" "$(pwd -W)" "$(basename "$p")" ); }
J=$OUTD/timing.jsonl
: > "$J"
FILES="x mask cos14 sin14 cos2 sin2 k_in v_in"
push_c() {   # push_c <C>
  local C=$1 d=$DEVD/c$1
  "$ADB" shell "mkdir -p $d" >/dev/null
  "$ADB" push "$(winpath "ops/kvmax_decode_t8_c${C}_24l/model.dla")" "$d/model.dla" >/dev/null || return 5
  for f in $FILES; do
    "$ADB" push "$(winpath "ops/kvmax/chk_c${C}/live/$f.bin")" "$d/$f.bin" >/dev/null || return 5
  done
}

run_c() {    # run_c <C> <count> -> "rc=<n> ms=<device ms>"
  local C=$1 cnt=$2
  local ins="-i kvmax/c$C/x.bin -i kvmax/c$C/mask.bin -i kvmax/c$C/cos14.bin -i kvmax/c$C/sin14.bin -i kvmax/c$C/cos2.bin -i kvmax/c$C/sin2.bin -i kvmax/c$C/k_in.bin -i kvmax/c$C/v_in.bin"
  local ots="-o kvmax/c$C/out_hidden.bin -o kvmax/c$C/out_k.bin -o kvmax/c$C/out_v.bin"
  "$ADB" shell "cd $DEV && a=\$(date +%s%N); LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a kvmax/c$C/model.dla $ins $ots -c $cnt >/dev/null 2>&1; rc=\$?; b=\$(date +%s%N); echo rc=\$rc ms=\$(( (b-a)/1000000 ))" | tr -d "\r"
}

echo "[bench] C set: $CS  rounds=$ROUNDS  N=$N  $(date -Iseconds)"
for C in $CS; do push_c "$C" || { echo "PUSH_FAIL C=$C"; exit 5; }; done

rep() { printf "%s\n" "$1" | sed -n "s/.*rc=\([0-9]*\).*/\1/p"; }
msm() { printf "%s\n" "$1" | sed -n "s/.*ms=\([0-9]*\).*/\1/p"; }

r=1
while [ "$r" -le "$ROUNDS" ]; do
  if [ $((r % 2)) = 1 ]; then ORD="$CS"; else ORD="$(printf "%s\n" $CS | tac)"; fi
  for C in $ORD; do
    R1="$(run_c "$C" 1)";  R1C=$(rep "$R1"); T1=$(msm "$R1")
    RN="$(run_c "$C" "$N")"; RNC=$(rep "$RN"); TN=$(msm "$RN")
    PI=""
    if [ "$R1C" = "0" ] && [ "$RNC" = "0" ]; then
      PI=$(awk -v a="$T1" -v b="$TN" -v n="$N" 'BEGIN{printf "%.2f", (b-a)/(n-1)}')
    fi
    echo "[r$r] C=$C t1=${T1}ms tN=${TN}ms($N) per_invoke=${PI:-NA}ms  ($R1 / $RN)"
    printf '{"round":%d,"C":%d,"t1_ms":%s,"tN_ms":%s,"n":%d,"per_invoke_ms":%s,"rc1":"%s","rcN":"%s"}\n' \
      "$r" "$C" "${T1:-null}" "${TN:-null}" "$N" "${PI:-null}" "${R1C:-x}" "${RNC:-x}" >> "$J"
  done
  r=$((r + 1))
done
echo "[bench] wrote $J"
