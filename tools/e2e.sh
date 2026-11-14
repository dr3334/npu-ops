#!/usr/bin/env bash
# End-to-end: build (if missing) -> compile (if missing) -> push -> run on NPU -> pull -> 对拍.
#
#   ./tools/e2e.sh ops/fc_1x64x64_int8 [--repeat 3]
#
# Split of duties: adb is a Windows binary (this Git Bash), ncc-tflite / python live in WSL.
# MSYS_NO_PATHCONV keeps adb destinations POSIX; source paths are passed as D:/...
set -uo pipefail

OPDIR="${1:?usage: e2e.sh ops/<name> [--repeat N]}"
REPEAT=3
[ "${2:-}" = "--repeat" ] && REPEAT="${3:-3}"
NAME="$(basename "$OPDIR")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# WSL path must follow the *current* checkout (worktrees included), not a hardcoded root.
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

# 1) build tflite + golden (WSL python that has the tflite schema package)
if [ ! -f "$OPDIR/model.tflite" ]; then
  echo "[build] $NAME"
  wslrun "/usr/bin/python3 tools/build_ops.py '$NAME' --out '$OPDIR/model.tflite'" || exit 3
fi

# 2) compile to .dla (WSL: ncc-tflite is an x86-64 Linux ELF)
if [ ! -f "$OPDIR/model.dla" ]; then
  echo "[compile] $NAME"
  wslrun "./tools/compile.sh '$OPDIR/model.tflite' '$OPDIR/model.dla'" || { echo "COMPILE_FAIL $NAME"; exit 4; }
fi

# 3) push model + input(s); a dynamic-B model takes weights as a second runtime input
DYN=0
grep -q '"dynamic_weights": true' "$OPDIR/model.json" 2>/dev/null && DYN=1
INS="-i dla/$NAME.in.bin"
"$ADB" shell "mkdir -p $DEV/dla" >/dev/null
"$ADB" push "$(winpath "$OPDIR/model.dla")"        "$DEV/dla/$NAME.dla"    >/dev/null || exit 5
"$ADB" push "$(winpath "$OPDIR/golden/input.bin")" "$DEV/dla/$NAME.in.bin" >/dev/null || exit 5
if [ "$DYN" = "1" ]; then
  "$ADB" push "$(winpath "$OPDIR/golden/weights.bin")" "$DEV/dla/$NAME.w.bin" >/dev/null || exit 5
  INS="$INS -i dla/$NAME.w.bin"
fi
# generic second runtime input (e.g. elementwise ADD: a + b)
if [ "$DYN" = "0" ] && [ -f "$OPDIR/golden/input1.bin" ]; then
  "$ADB" push "$(winpath "$OPDIR/golden/input1.bin")" "$DEV/dla/$NAME.in1.bin" >/dev/null || exit 5
  INS="$INS -i dla/$NAME.in1.bin"
fi

# 4) run on NPU hardware
RUNLOG="$("$ADB" shell "cd $DEV && LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a dla/$NAME.dla $INS -o dla/$NAME.out.bin -c $REPEAT 2>&1" | tr -d '\r')"
echo "$RUNLOG" | grep -E "Inference|FPS|time|fail|error" | head -4

# 5) pull output
"$ADB" pull "$DEV/dla/$NAME.out.bin" "$(winpath "$OPDIR")/out_device.bin" >/dev/null || exit 6

# 6) compare
wslrun "/usr/bin/python3 tools/compare.py '$OPDIR'" > "$OPDIR/compare.txt" 2>&1
CRC=$?
cat "$OPDIR/compare.txt"
TS=$(date -Iseconds)
printf '{"op":"%s","repeat":%d,"compare_exit":%d,"ts":"%s"}\n' "$NAME" "$REPEAT" "$CRC" "$TS" > "$OPDIR/e2e.json"
exit $CRC
