#!/usr/bin/env bash
# Ticket 05 companion of tools/e2e_i16.sh: the SHARED-WEIGHT flow (two .dla + one bin).
#
#   ./tools/e2e_q05.sh <opdir> [<opdir> ...]        # env: REPEAT=3, BIN=ops/q05_shared.bin
#
# tools/e2e.sh supports one model.dla and at most two runtime inputs; the shared-weight
# form is "N graphs + one extracted bin", which it does not cover (docs/OP-shared-weights.md
# section 5 asks the Lead to decide whether the public tool grows a --shared branch).
# This script is the ticket-local equivalent: same push/run/pull/compare contract, one
# extra "-i <bin>" argument, and tools/build_24layer.py --eval --device as the comparator
# (the public compare.py has no int16 dtype and no chain metrics).
#
# Every graph directory is expected to hold:
#   model_shared.dla        extract-shared output (the .dla that references the bin)
#   golden/input.bin        activations
#   out_device.bin          pulled back here
set -uo pipefail

[ "$#" -ge 1 ] || { echo "usage: $0 <opdir> [<opdir> ...]" >&2; exit 2; }
REPEAT="${REPEAT:-3}"
BIN="${BIN:-ops/q05_shared.bin}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
_win="$(cd "$ROOT" && pwd -W 2>/dev/null || echo /d/x)"
_drv="$(printf '%s' "${_win:0:1}" | tr 'A-Z' 'a-z')"
WSLROOT="${WSLROOT:-/mnt/$_drv${_win:2}}"
DEV="${DEV:-/data/local/tmp/npuops}"
SUB="$DEV/q05"
ADB="${ADB:-/d/adb/platform-tools/adb}"
WSL="${WSL:-wsl.exe}"
export MSYS_NO_PATHCONV=1

winpath() { local p="$ROOT/$1"; ( cd "$(dirname "$p")" && printf '%s/%s' "$(pwd -W)" "$(basename "$p")" ); }
WD="${WD:-/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen}"
wslrun()  { "$WSL" -e bash -lc "cd '$WSLROOT' && export QWEN25_WEIGHTS_DIR='$WD' && $*"; }

cd "$ROOT" || exit 2
[ -f "$BIN" ] || { echo "missing shared bin $BIN" >&2; exit 2; }

"$ADB" shell "mkdir -p $SUB" >/dev/null
"$ADB" push "$(winpath "$BIN")" "$SUB/shared.bin" >/dev/null || exit 5
BSHA="$("$ADB" shell "sha256sum $SUB/shared.bin" | tr -d '\r' | cut -d' ' -f1)"
echo "[bin] $BIN -> $SUB/shared.bin  sha256=$BSHA"

RC_ALL=0
for OPDIR in "$@"; do
  NAME="$(basename "$OPDIR")"
  DLA="$OPDIR/model_shared.dla"
  [ -f "$DLA" ] || { echo "MISSING $DLA" >&2; RC_ALL=6; continue; }
  "$ADB" push "$(winpath "$DLA")" "$SUB/$NAME.dla" >/dev/null || exit 5
  "$ADB" push "$(winpath "$OPDIR/golden/input.bin")" "$SUB/$NAME.in.bin" >/dev/null || exit 5
  DSHA="$("$ADB" shell "sha256sum $SUB/$NAME.dla" | tr -d '\r' | cut -d' ' -f1)"
  echo "[dla] $DLA -> $SUB/$NAME.dla  sha256=$DSHA"
  RUNLOG="$("$ADB" shell "cd $DEV && LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a q05/$NAME.dla -i q05/$NAME.in.bin -i q05/shared.bin -o q05/$NAME.out.bin -c $REPEAT 2>&1" | tr -d '\r')"
  echo "$RUNLOG" | grep -E "Inference|FPS|time|fail|error|size" | head -6
  if ! echo "$RUNLOG" | grep -qi "inference"; then
    echo "RUN_FAIL $NAME"; RC_ALL=7; continue
  fi
  "$ADB" pull "$SUB/$NAME.out.bin" "$(winpath "$OPDIR")/out_device.bin" >/dev/null || exit 6
  OSHA="$(sha256sum "$OPDIR/out_device.bin" | cut -d' ' -f1)"
  wslrun "/usr/bin/python3 tools/build_24layer.py --eval '$OPDIR' --device" | tee "$OPDIR/compare.txt"
  CRC="${PIPESTATUS[0]}"
  TS="$(date -Iseconds)"
  printf '{"op":"%s","repeat":%d,"bin_sha256":"%s","dla_sha256":"%s","out_sha256":"%s","ts":"%s"}\n' \
    "$NAME" "$REPEAT" "$BSHA" "$DSHA" "$OSHA" "$TS" > "$OPDIR/e2e.json"
  RC_ALL=$(( RC_ALL | CRC ))
done
exit $RC_ALL
