#!/usr/bin/env bash
# Unary variant of mkasset.sh: capture one operator's command-stream asset (A -> C).
#
#   ./probe/mkasset_u.sh <opname> <device_dla> <device_a.bin> [local_a.bin]
#
# Same idea as mkasset.sh, but the model takes exactly one runtime input and produces
# one output, so the command stream has 2 address words (A, C) instead of 3.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
OP="$1"; DLA="$2"; ABIN="$3"; AL="${4:-}"
export MSYS_NO_PATHCONV=1
ADB="${ADB:-/d/adb/platform-tools/adb}"
DEV=/data/local/tmp/npuops
LOCAL="probe/assets/$OP"
rm -rf "$LOCAL"; mkdir -p "$LOCAL/buf"

[ -n "$AL" ] || AL="$(ls ops/${OP}_*/golden/input.bin 2>/dev/null | head -1)"
[ -n "$AL" ] || { echo "local golden input not found for $OP"; exit 2; }

echo "[capture-u] $OP  (neuronrt + iolog2, one run, single input)"
NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" ./tools/devlock.sh run -- \
  "$ADB" shell "cd $DEV && rm -f probe/io2.*.log && rm -rf probe/buf && mkdir -p probe/buf && \
    LD_PRELOAD=$DEV/probe/libiolog2.so LD_LIBRARY_PATH=$DEV/lib \
    ./bin/neuronrt -m hw -a $DLA -i $ABIN -o $DEV/probe/cap_u.bin -c 1" >/dev/null 2>&1

LOG="$("$ADB" shell "ls -t $DEV/probe/io2.*.log 2>/dev/null | head -1" | tr -d '\r')"
[ -n "$LOG" ] || { echo "no iolog produced"; exit 3; }
"$ADB" shell "cp $LOG $DEV/probe/cap_u.log" >/dev/null
"$ADB" pull "$DEV/probe/cap_u.log" "$LOCAL/cap.log" >/dev/null 2>&1 || { echo "pull log failed"; exit 3; }
"$ADB" shell "cd $DEV/probe/buf && ls *.bin 2>/dev/null" | tr -d '\r' | while read -r f; do
  [ -n "$f" ] || continue
  "$ADB" pull "$DEV/probe/buf/$f" "$LOCAL/buf/$f" >/dev/null 2>&1
done
"$ADB" pull "$DEV/probe/cap_u.bin" "$LOCAL/vendor_out.bin" >/dev/null 2>&1
cp "$AL" "$LOCAL/a.bin"

wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && \
  /usr/bin/python3 probe/mkasset.py '$LOCAL/cap.log' '$LOCAL/buf' '$LOCAL/a.bin' - '$LOCAL'" || exit 4

"$ADB" shell "mkdir -p $DEV/assets/$OP" >/dev/null
for f in cmd0.bin cmd1.bin cmd2.bin cmdin.bin subcmd.bin iova.txt cmdsizes.txt; do
  "$ADB" push "$LOCAL/$f" "$DEV/assets/$OP/$f" >/dev/null 2>&1 || { echo "push $f failed"; exit 5; }
done
echo "[capture-u] asset -> $DEV/assets/$OP  ($(cat "$LOCAL/iova.txt"))"
