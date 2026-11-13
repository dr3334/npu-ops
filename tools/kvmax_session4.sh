#!/usr/bin/env bash
# Ticket kv-maxlen: is the C=8192 graph stable at the boundary?  3 independent Runs of the
# same inputs, sha256 of all three outputs (ticket 06 established a 0-byte noise floor at
# C=2048; this repeats the test at the largest C that exists today).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV=/data/local/tmp/npuops
ADB=/d/adb/platform-tools/adb
C="${1:-8192}"
export MSYS_NO_PATHCONV=1
cd "$ROOT" || exit 2
echo "########## HEALTHCHECK BEFORE $(date -Iseconds)"
./tools/healthcheck.sh --quick; echo "HC_BEFORE_RC=$?"
D=kvmax/c$C
: > ops/kvmax/noise_c$C.txt
for i in 1 2 3; do
  "$ADB" shell "cd $DEV && LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a $D/model.dla \
    -i $D/x.bin -i $D/mask.bin -i $D/cos14.bin -i $D/sin14.bin -i $D/cos2.bin -i $D/sin2.bin \
    -i $D/k_in.bin -i $D/v_in.bin -o $D/out_hidden.bin -o $D/out_k.bin -o $D/out_v.bin -c 1 >/dev/null 2>&1; \
    echo rc=\$?" | tr -d "\r"
  "$ADB" shell "cd $DEV && sha256sum $D/out_hidden.bin $D/out_k.bin $D/out_v.bin" | tr -d "\r" | awk "{print \$1}" | tr "\n" " " >> ops/kvmax/noise_c$C.txt
  echo >> ops/kvmax/noise_c$C.txt
done
echo "[noise] C=$C, three Runs (hidden k v sha256):"
cat "ops/kvmax/noise_c$C.txt"
echo "########## HEALTHCHECK AFTER $(date -Iseconds)"
./tools/healthcheck.sh --quick; echo "HC_AFTER_RC=$?"
echo "SESSION4_DONE $(date -Iseconds)"
