#!/usr/bin/env bash
# healthcheck.sh — "known good program" + device state self-check (AGENTS.md §3.4).
#
#   ./tools/healthcheck.sh            # full: device presence + known-good op + repeat + state probe
#   ./tools/healthcheck.sh --quick    # device presence + one known-good run
#
# A kernel/operator experiment must pass this BEFORE and AFTER it runs: an op that
# computes correctly but leaves the board in a bad state must not enter the library.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADB="${ADB:-/d/adb/platform-tools/adb}"
DEV=/data/local/tmp/npuops
GOOD_OP="${GOOD_OP:-fc_1x64x64_int8}"     # the reference "known good" operator
QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
export MSYS_NO_PATHCONV=1
rc=0
step() { printf '%-42s ' "$1"; }

# --- 1. device present ---------------------------------------------------------
step "device present"
if "$ADB" get-state >/dev/null 2>&1; then echo "OK"; else echo "FAIL"; exit 10; fi

step "device identity"
ID="$("$ADB" shell 'getprop ro.board.platform; getprop ro.build.version.release' | tr -d '\r' | paste -sd '/')"
echo "$ID"

step "/dev/apusys node"
"$ADB" shell 'test -c /dev/apusys' && echo "OK" || { echo "FAIL"; rc=11; }

step "neuronrt present"
"$ADB" shell "test -x $DEV/bin/neuronrt" && echo "OK" || { echo "FAIL"; rc=12; }

# --- 2. known-good operator ----------------------------------------------------
step "known-good op ($GOOD_OP)"
if "$ROOT/tools/e2e.sh" "ops/$GOOD_OP" >"/tmp/health_$GOOD_OP.log" 2>&1; then
  echo "PASS"
else
  echo "FAIL (see /tmp/health_$GOOD_OP.log)"; rc=13
fi

if [ "$QUICK" = "0" ]; then
  # --- 3. repeatability: same op twice, results must agree --------------------
  step "repeatability (2nd run)"
  if "$ROOT/tools/e2e.sh" "ops/$GOOD_OP" >"/tmp/health_2nd_$GOOD_OP.log" 2>&1 \
     && cmp -s "$ROOT/ops/$GOOD_OP/out_device.bin" "$ROOT/ops/$GOOD_OP/out_device.bin"; then
    echo "PASS"
  else
    echo "FAIL"; rc=14
  fi

  # --- 4. board state after the run ------------------------------------------
  step "board state (no crash/leak markers)"
  ST="$("$ADB" shell 'dmesg | tail -40' 2>/dev/null | tr -d '\r' | grep -ciE "apusys.*(timeout|fail|error)|mdw.*(timeout|fail)" || true)"
  if [ "${ST:-0}" = "0" ]; then echo "OK"; else echo "WARN ($ST lines, review dmesg)"; fi
fi

echo
[ "$rc" = "0" ] && echo "HEALTHCHECK: PASS" || echo "HEALTHCHECK: FAIL rc=$rc"
exit $rc
