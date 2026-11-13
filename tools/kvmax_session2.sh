#!/usr/bin/env bash
# Ticket kv-maxlen: lock session 2 -- host-clock bench + the device-clock anomaly probe.
#   ./tools/devlock.sh run -- ./tools/kvmax_session2.sh "2048 4096 8192"
set -uo pipefail
CS="${1:-2048 4096 8192}"
ROUNDS="${ROUNDS:-3}"
N="${N:-10}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV=/data/local/tmp/npuops
ADB=/d/adb/platform-tools/adb
export MSYS_NO_PATHCONV=1
cd "$ROOT" || exit 2
echo "########## HEALTHCHECK BEFORE $(date -Iseconds)"
./tools/healthcheck.sh --quick; echo "HC_BEFORE_RC=$?"
ROUNDS="$ROUNDS" N="$N" ./tools/kvmax_bench2.sh "$CS"

echo "########## device-clock anomaly probe (raw epochs, C=8192, -c 10)"
C=8192
INS="-i kvmax/c$C/x.bin -i kvmax/c$C/mask.bin -i kvmax/c$C/cos14.bin -i kvmax/c$C/sin14.bin -i kvmax/c$C/cos2.bin -i kvmax/c$C/sin2.bin -i kvmax/c$C/k_in.bin -i kvmax/c$C/v_in.bin"
OTS="-o kvmax/c$C/out_hidden.bin -o kvmax/c$C/out_k.bin -o kvmax/c$C/out_v.bin"
for i in 1 2; do
  "$ADB" shell "cd $DEV && a=\$(date +%s%N); LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a kvmax/c$C/model.dla $INS $OTS -c 10 >/dev/null 2>&1; rc=\$?; b=\$(date +%s%N); echo rc=\$rc a=\$a b=\$b diff_ns=\$((b-a))" | tr -d "\r"
done
echo "########## HEALTHCHECK AFTER $(date -Iseconds)"
./tools/healthcheck.sh --quick; echo "HC_AFTER_RC=$?"
echo "SESSION2_DONE $(date -Iseconds)"
