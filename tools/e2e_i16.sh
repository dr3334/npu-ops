#!/usr/bin/env bash
# Ticket 04d companion of tools/e2e.sh: identical flow, int16-aware compare.
#
#   ./tools/e2e_i16.sh ops/<name> [--repeat N]
#
# The only difference from tools/e2e.sh is step 6: tools/compare_i16.py instead of
# tools/compare.py (the public tool's dtype map has no int16 -- AGENTS 8 keeps us from
# editing it from a ticket; the one-line change is listed in the ticket report).
set -uo pipefail

OPDIR="${1:?usage: e2e_i16.sh ops/<name> [--repeat N]}"
REPEAT=3
[ "${2:-}" = "--repeat" ] && REPEAT="${3:-3}"
NAME="$(basename "$OPDIR")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
_win="$(cd "$ROOT" && pwd -W 2>/dev/null || echo /d/x)"
_drv="$(printf '%s' "${_win:0:1}" | tr 'A-Z' 'a-z')"
WSLROOT="${WSLROOT:-/mnt/$_drv${_win:2}}"
DEV=/data/local/tmp/npuops
ADB="${ADB:-/d/adb/platform-tools/adb}"
WSL="${WSL:-wsl.exe}"
export MSYS_NO_PATHCONV=1

winpath() { local p="$ROOT/$1"; ( cd "$(dirname "$p")" && printf '%s/%s' "$(pwd -W)" "$(basename "$p")" ); }
wslrun()  { "$WSL" -e bash -lc "cd '$WSLROOT' && $*"; }

cd "$ROOT" || exit 2

if [ ! -f "$OPDIR/model.tflite" ]; then
  echo "[build] $NAME"
  wslrun "/usr/bin/python3 tools/build_w4a8.py '$NAME' --out '$OPDIR/model.tflite' --a-bits 16 --w-bits 4" || exit 3
fi
if [ ! -f "$OPDIR/model.dla" ]; then
  echo "[compile] $NAME"
  wslrun "./tools/compile.sh '$OPDIR/model.tflite' '$OPDIR/model.dla'" || { echo "COMPILE_FAIL $NAME"; exit 4; }
fi

INS="-i dla/$NAME.in.bin"
"$ADB" shell "mkdir -p $DEV/dla" >/dev/null
"$ADB" push "$(winpath "$OPDIR/model.dla")"        "$DEV/dla/$NAME.dla"    >/dev/null || exit 5
"$ADB" push "$(winpath "$OPDIR/golden/input.bin")" "$DEV/dla/$NAME.in.bin" >/dev/null || exit 5
if [ -f "$OPDIR/golden/input1.bin" ]; then
  "$ADB" push "$(winpath "$OPDIR/golden/input1.bin")" "$DEV/dla/$NAME.in1.bin" >/dev/null || exit 5
  INS="$INS -i dla/$NAME.in1.bin"
fi

RUNLOG="$("$ADB" shell "cd $DEV && LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a dla/$NAME.dla $INS -o dla/$NAME.out.bin -c $REPEAT 2>&1" | tr -d '\r')"
echo "$RUNLOG" | grep -E "Inference|FPS|time|fail|error" | head -4

"$ADB" pull "$DEV/dla/$NAME.out.bin" "$(winpath "$OPDIR")/out_device.bin" >/dev/null || exit 6

wslrun "/usr/bin/python3 tools/compare_i16.py '$OPDIR'" > "$OPDIR/compare.txt" 2>&1
CRC=$?
cat "$OPDIR/compare.txt"
TS=$(date -Iseconds)
printf '{"op":"%s","repeat":%d,"compare_exit":%d,"ts":"%s"}\n' "$NAME" "$REPEAT" "$CRC" "$TS" > "$OPDIR/e2e.json"
exit $CRC
