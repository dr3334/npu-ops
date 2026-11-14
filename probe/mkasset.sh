#!/usr/bin/env bash
# Capture one operator's command-stream asset for the self-made submitter.
#
#   ./probe/mkasset.sh <opname> <device_dla> <device_a.bin> <device_b.bin>
#
# Runs neuronrt ONCE under the ioctl recorder (vendor runtime is used only to *record* one
# legal submission; the asset is then replayed by our own submitter), pulls back the log +
# buffer dump, builds the asset dir locally, and pushes it to /data/local/tmp/npuops/assets/<op>.
#
# NOTE: neuronrt scrubs the environment, so IOLOG/IODIR never reach the shim. The shim then
# uses its defaults: probe/io2.<pid>.log and probe/buf/. We clean those and pick up the
# newest log after the run.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
OP="$1"; DLA="$2"; ABIN="$3"; BBIN="$4"
export MSYS_NO_PATHCONV=1
ADB="${ADB:-/d/adb/platform-tools/adb}"
DEV=/data/local/tmp/npuops
LOCAL="probe/assets/$OP"
rm -rf "$LOCAL"; mkdir -p "$LOCAL/buf"

echo "[capture] $OP  (neuronrt + iolog2, one run)"
NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" ./tools/devlock.sh run -- \
  "$ADB" shell "cd $DEV && rm -f probe/io2.*.log && rm -rf probe/buf && mkdir -p probe/buf && \
    LD_PRELOAD=$DEV/probe/libiolog2.so LD_LIBRARY_PATH=$DEV/lib \
    ./bin/neuronrt -m hw -a $DLA -i $ABIN -i $BBIN -o $DEV/probe/cap_out.bin -c 1" >/dev/null 2>&1

LOG="$("$ADB" shell "ls -t $DEV/probe/io2.*.log 2>/dev/null | head -1" | tr -d '\r')"
[ -n "$LOG" ] || { echo "no iolog produced (shim not loaded?)"; exit 3; }
"$ADB" shell "cp $LOG $DEV/probe/cap.log" >/dev/null
"$ADB" pull "$DEV/probe/cap.log" "$LOCAL/cap.log" >/dev/null 2>&1 || { echo "pull log failed"; exit 3; }

"$ADB" shell "cd $DEV/probe/buf && ls *.bin 2>/dev/null" | tr -d '\r' | while read -r f; do
  [ -n "$f" ] || continue
  "$ADB" pull "$DEV/probe/buf/$f" "$LOCAL/buf/$f" >/dev/null 2>&1
done
"$ADB" pull "$DEV/probe/cap_out.bin" "$LOCAL/vendor_out.bin" >/dev/null 2>&1

AL="${5:-}"; BL="${6:-}"
[ -n "$AL" ] || AL="$(ls ops/${OP}_*/golden/input.bin 2>/dev/null | head -1)"
[ -n "$BL" ] || BL="$(ls ops/${OP}_*/golden/input1.bin 2>/dev/null | head -1)"
cp "$AL" "$LOCAL/a.bin" 2>/dev/null
cp "$BL" "$LOCAL/b.bin" 2>/dev/null

wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && \
  /usr/bin/python3 probe/mkasset.py '$LOCAL/cap.log' '$LOCAL/buf' '$LOCAL/a.bin' '$LOCAL/b.bin' '$LOCAL'" || exit 4

"$ADB" shell "mkdir -p $DEV/assets/$OP" >/dev/null
for f in cmd0.bin cmd1.bin cmd2.bin cmdin.bin subcmd.bin iova.txt cmdsizes.txt; do
  "$ADB" push "$LOCAL/$f" "$DEV/assets/$OP/$f" >/dev/null 2>&1 || { echo "push $f failed"; exit 5; }
done
echo "[capture] asset -> $DEV/assets/$OP  ($(cat "$LOCAL/iova.txt"))"
