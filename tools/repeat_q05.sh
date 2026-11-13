#!/usr/bin/env bash
# Ticket 05: "one load, many Runs" stability -- N SEPARATE neuronrt invocations (each loads
# the graph + bin once) writing N output files, so byte equality can be checked.
#   ./tools/repeat_q05.sh <opdir> [n]        # run inside ./tools/devlock.sh run --
set -uo pipefail
OPDIR="${1:?usage: repeat_q05.sh <opdir> [n]}"
N="${2:-3}"
NAME="$(basename "$OPDIR")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV="${DEV:-/data/local/tmp/npuops}"
SUB="$DEV/q05"
ADB="${ADB:-/d/adb/platform-tools/adb}"
export MSYS_NO_PATHCONV=1
winpath() { local p="$ROOT/$1"; ( cd "$(dirname "$p")" && printf '%s/%s' "$(pwd -W)" "$(basename "$p")" ); }
cd "$ROOT" || exit 2
i=1
while [ "$i" -le "$N" ]; do
  "$ADB" shell "cd $DEV && LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a q05/$NAME.dla -i q05/$NAME.in.bin -i q05/shared.bin -o q05/$NAME.r$i.bin -c 1 2>&1 | grep -E 'Inference|ms/inf'" | tr -d '\r'
  "$ADB" pull "$SUB/$NAME.r$i.bin" "$(winpath "$OPDIR")/out_r$i.bin" >/dev/null || exit 6
  i=$((i+1))
done
(cd "$OPDIR" && sha256sum out_r*.bin)
