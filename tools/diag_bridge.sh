#!/usr/bin/env bash
# Diagnostic: compile with bridging ALLOWED so the exec plan reveals what ncc
# inserts into the "gap" between Input/Output and the MDLA step (read-only use of
# ncc; writes only to /tmp, never to the repo).
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-convpool"
SDK="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
NCC="$SDK/host/bin/ncc-tflite"
export LD_LIBRARY_PATH="$SDK/host/lib:${LD_LIBRARY_PATH:-}"
cd "$R" || exit 2
for s in "$@"; do
  echo "########## $s"
  "$NCC" --arch=mdla5.0 --opt=3 --num-mdla=2 --l1-size-kb=3072 \
    --show-exec-plan --show-memory-summary \
    "ops/$s/model.tflite" -d "/tmp/diag_$s.dla" 2>&1 |
    sed -n '/Graph\[0\]/,/TargetExecutionOrder/p'
done
