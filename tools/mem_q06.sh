#!/usr/bin/env bash
# Ticket 06: resident-memory probe for the decode graph (one load, N Runs).
#   ./tools/mem_q06.sh [iterations]        # run inside ./tools/devlock.sh run --
set -uo pipefail
ITER="$1"
[ -z "$ITER" ] && ITER=200
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV=/data/local/tmp/npuops
SUB=$DEV/q06
ADB=/d/adb/platform-tools/adb
OUT=ops/q06_decode_t8_c2048_24l/mem
export MSYS_NO_PATHCONV=1
winpath() { local p="$ROOT/$1"; ( cd "$(dirname "$p")" && printf '%s/%s' "$(pwd -W)" "$(basename "$p")" ); }
cd "$ROOT" || exit 2
mkdir -p "$OUT"
"$ADB" push "$(winpath tools/mem_q06_device.sh)" "$SUB/mem.sh" >/dev/null || exit 5
"$ADB" shell "dmabuf_dump -b" | tr -d '\r' > "$OUT/dmabuf_before.txt"
"$ADB" shell "sh $SUB/mem.sh decode.dla '$ITER'" | tr -d '\r' > "$OUT/run.txt"
"$ADB" shell "dmabuf_dump -b" | tr -d '\r' > "$OUT/dmabuf_after.txt"
grep -E '^===|^Rss|^Pss|^rchar|^read_bytes' "$OUT/run.txt" | head -70
echo "[mem] wrote $OUT/"
