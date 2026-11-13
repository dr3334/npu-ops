#!/usr/bin/env bash
# Ticket kv-maxlen: resident-memory probe for one cache length C (one load, N Runs).
#   ./tools/devlock.sh run -- ./tools/kvmax_mem.sh 8192 [iterations]
set -uo pipefail
C="${1:?usage: kvmax_mem.sh <C> [iterations]}"
ITER="${2:-20}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV=/data/local/tmp/npuops
ADB=/d/adb/platform-tools/adb
OUT=ops/kvmax/mem_c$C
export MSYS_NO_PATHCONV=1
winpath() { local p="$ROOT/$1"; ( cd "$(dirname "$p")" && printf "%s/%s" "$(pwd -W)" "$(basename "$p")" ); }
cd "$ROOT" || exit 2
mkdir -p "$OUT"
"$ADB" push "$(winpath tools/kvmax_mem_device.sh)" "$DEV/kvmax/mem.sh" >/dev/null || exit 5
"$ADB" shell "dmabuf_dump -b" | tr -d "\r" > "$OUT/dmabuf_before.txt"
"$ADB" shell "sh $DEV/kvmax/mem.sh $C $ITER" | tr -d "\r" > "$OUT/run.txt"
"$ADB" shell "dmabuf_dump -b" | tr -d "\r" > "$OUT/dmabuf_after.txt"
grep -E "^===|^Rss|^Pss|^rchar|^read_bytes" "$OUT/run.txt" | head -60
echo "[mem] wrote $OUT/"
