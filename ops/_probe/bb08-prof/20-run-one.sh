#!/usr/bin/env bash
# 20-run-one.sh — ONE profiled inference batch (one model, one level). BOARD: caller holds devlock.
#
#   NPU_LOCK_HELD=1 ./tools/devlock.sh run -- bash ops/_probe/bb08-prof/20-run-one.sh fc 3 20
#   ... 20-run-one.sh dec 2 10
#
# Always executed in a fresh CWD on the device, because level 3 writes its JSON into CWD.
# dmesg is checked after every run: MDLA timeout / ret(-5) => the board is in the bad state
# (docs/OP-toolchain.md 6.1) and the caller must reboot, not retry.
set -uo pipefail
M="${1:?model fc|dec}"; L="${2:?level 1|2|3}"; N="${3:-20}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EV="$ROOT/ops/_probe/bb08-prof/board"
ADB="${ADB:-/d/adb/platform-tools/adb}"
DEV=/data/local/tmp/npuops
export MSYS_NO_PATHCONV=1
mkdir -p "$EV"
sh_() { "$ADB" shell "$1" 2>&1 | tr -d '\r'; }
RUN="$DEV/bb08/run_${M}_L${L}"

sh_ "rm -rf $RUN && mkdir -p $RUN"
"$ADB" logcat -c >/dev/null 2>&1
echo "===== profiled: $M level=$L -c $N ====="
sh_ "cd $RUN && timeout 180 env MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=$L \
       LD_LIBRARY_PATH=$DEV/lib ../../bin/neuronrt -m hw -a ../${M}_p${L}.dla -i ../${M}.in.bin -o out.bin -c $N" \
  | tee "$EV/console-${M}-L${L}.txt"
RC=${PIPESTATUS[0]}
echo "  rc=$RC"

echo "----- logcat tag=neuron -----"
"$ADB" logcat -d -v time -s neuron 2>&1 | tr -d '\r' | tee "$EV/logcat-neuron-${M}-L${L}.txt"
echo "----- logcat grep (fallback, any tag) -----"
"$ADB" logcat -d -v time 2>&1 | tr -d '\r' | grep -iE "profiling|profil|neuron|MVPU|Per-OP|MDLA" \
  | grep -vE "logcat|^-+ beginning" | tail -60 > "$EV/logcat-grep-${M}-L${L}.txt"
wc -l < "$EV/logcat-grep-${M}-L${L}.txt"

echo "----- files written to CWD -----"
sh_ "cd $RUN && ls -l"
for f in execution_result.json dla_info.json; do
  if sh_ "test -f $RUN/$f && echo yes" | grep -q yes; then
    "$ADB" pull "$RUN/$f" "$(cygpath -w "$EV/${f%.json}-${M}-L${L}.json")" >/dev/null && echo "  pulled $f"
  fi
done
sh_ "cd $RUN && for f in *.bin; do echo \"\$f \$(md5sum \$f | cut -c1-16) \$(stat -c %s \$f)\"; done"

echo "----- dmesg (apusys / mdw) -----"
sh_ "dmesg | grep -iE 'apusys|mdw|MDLA' | tail -12"
echo "----- verdict -----"
BAD=$(sh_ "dmesg | grep -icE 'MDLA timeout|ret\\(-5'" | tail -1)
BAD=${BAD:-0}
echo "BADMARKERS=$BAD"
[ "$BAD" = "0" ] && echo "BOARD_STATE=dmesg-clean" || echo "BOARD_STATE=BAD(need adb reboot)"
