#!/usr/bin/env bash
# Ticket 05: push tools/mem_q05_device.sh and sample neuronrt's resident set while it runs
# one 24-layer graph with the external bin (one load, many Runs).
#   ./tools/mem_q05.sh <opdir> [iterations]        # run inside ./tools/devlock.sh run --
set -uo pipefail
OPDIR="${1:?usage: mem_q05.sh <opdir> [iterations]}"
ITER="${2:-400}"
NAME="$(basename "$OPDIR")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV="${DEV:-/data/local/tmp/npuops}"
SUB="$DEV/q05"
ADB="${ADB:-/d/adb/platform-tools/adb}"
export MSYS_NO_PATHCONV=1
winpath() { local p="$ROOT/$1"; ( cd "$(dirname "$p")" && printf '%s/%s' "$(pwd -W)" "$(basename "$p")" ); }

cd "$ROOT" || exit 2
mkdir -p "$OPDIR/mem"
"$ADB" push "$(winpath tools/mem_q05_device.sh)" "$SUB/mem.sh" >/dev/null || exit 5
"$ADB" shell "dmabuf_dump -b" | tr -d '\r' > "$OPDIR/mem/dmabuf_before.txt"
"$ADB" shell "sh $SUB/mem.sh '$NAME' '$ITER'" | tr -d '\r' > "$OPDIR/mem/run.txt"
"$ADB" shell "dmabuf_dump -b" | tr -d '\r' > "$OPDIR/mem/dmabuf_after.txt"
"$ADB" pull "$SUB/$NAME.mem.out.bin" "$(winpath "$OPDIR")/out_mem.bin" >/dev/null 2>&1
grep -E '^===|^Rss|^Pss|^rchar|^read_bytes' "$OPDIR/mem/run.txt" | head -60
echo "[mem] wrote $OPDIR/mem/"
