#!/usr/bin/env bash
# Ticket kv-maxlen: ONE locked board session -- healthcheck, paired timing bench over the
# cache lengths that compiled, then the device-vs-quantised-golden check at width C.
#   ./tools/devlock.sh run -- ./tools/kvmax_board.sh "2048 4096 8192"
#   ROUNDS=3 N=20 MEM=1  (MEM=1 also runs the resident-memory probe on the largest C)
set -uo pipefail
CS="${1:-2048 4096 8192}"
ROUNDS="${ROUNDS:-3}"
N="${N:-20}"
MEM="${MEM:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV=/data/local/tmp/npuops
ADB=/d/adb/platform-tools/adb
OUTD=ops/kvmax
export MSYS_NO_PATHCONV=1
cd "$ROOT" || exit 2
mkdir -p "$OUTD"
echo "########## HEALTHCHECK BEFORE $(date -Iseconds)"
./tools/healthcheck.sh --quick; echo "HC_BEFORE_RC=$?"

########## 1) timing bench (paired alternating, into ops/kvmax/timing.jsonl)
ROUNDS="$ROUNDS" N="$N" ./tools/kvmax_bench.sh "$CS"

########## 2) numeric check at width C: device vs quantised golden (non-empty cache)
for C in $CS; do
  D=kvmax/c$C
  "$ADB" shell "cd $DEV && a=\$(date +%s%N); LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a $D/model.dla \
    -i $D/x.bin -i $D/mask.bin -i $D/cos14.bin -i $D/sin14.bin -i $D/cos2.bin -i $D/sin2.bin \
    -i $D/k_in.bin -i $D/v_in.bin -o $D/out_hidden.bin -o $D/out_k.bin -o $D/out_v.bin -c 1 >/dev/null 2>&1; \
    rc=\$?; b=\$(date +%s%N); echo rc=\$rc ms=\$(( (b-a)/1000000 ))" | tr -d "\r"
  for f in out_hidden out_k out_v; do
    "$ADB" pull "$DEV/$D/$f.bin" "$OUTD/chk_c$C/live/$f.bin" >/dev/null || echo "PULL_FAIL $C $f"
  done
  wsl.exe -e bash -lc "cd /mnt/d/deepseek_\\ workspace/mtk/wt-kv2 && /usr/bin/python3 tools/kvmax_check.py --compare --dir ops/kvmax/chk_c$C" \
    | tee "$OUTD/compare_c$C.txt"
  tail -1 "$OUTD/compare_c$C.txt"
done

if [ "$MEM" = "1" ]; then
  BIG="$(printf "%s\n" $CS | sort -n | tail -1)"
  echo "########## 3) resident memory at C=$BIG"
  ./tools/kvmax_mem.sh "$BIG" 20
fi

echo "########## HEALTHCHECK AFTER $(date -Iseconds)"
./tools/healthcheck.sh --quick; echo "HC_AFTER_RC=$?"
echo "SESSION_DONE $(date -Iseconds)"
