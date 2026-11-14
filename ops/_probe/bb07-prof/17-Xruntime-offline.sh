#!/usr/bin/env bash
# 17-Xruntime-offline.sh — what does --Xruntime accept? Probe with the host neuronrt.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
cd "$R" || exit 2
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
RT="$S/host/bin/neuronrt"
export LD_LIBRARY_PATH="$S/host/lib"
D=ops/_probe/bb07-prof/build2/q000.dla
IN=ops/_probe/bb07-prof/build/golden/input.bin
ls -l "$D" "$IN" || exit 3
T=/mnt/d/bb07-tmp
mkdir -p "$T"

probe() { # label, mode, extra args...
  local label="$1" mode="$2"; shift 2
  local log="$T/rt_$label.log"
  "$RT" -m "$mode" -a "$D" -i "$IN" -o "$T/out_$label.bin" -c 1 "$@" > "$log" 2>&1
  local rc=$?
  printf '=== %-16s mode=%-4s rc=%-2s args=[%s]\n' "$label" "$mode" "$rc" "$*"
  head -12 "$log" | sed 's/^/    /'
  echo
}
probe null_plain null
probe null_x1 null --Xruntime 'nbThreads:2'
probe null_x2 null --Xruntime 'nbThreads=2'
probe null_x3 null --Xruntime 'garbage'
probe hw_plain hw
