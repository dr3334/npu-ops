#!/usr/bin/env bash
# 19-full-report.sh — full stdout of the host runtime at profiling level 3 (no truncation).
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
cd "$R" || exit 2
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
export LD_LIBRARY_PATH="$S/host/lib"
RT="$S/host/bin/neuronrt"
T=/mnt/d/bb07-tmp/full
rm -rf "$T"; mkdir -p "$T"; cd "$T" || exit 3
D="/mnt/d/deepseek_ workspace/mtk/wt-bb07/ops/_probe/bb07-prof/build2/q003.dla"
IN="/mnt/d/deepseek_ workspace/mtk/wt-bb07/ops/_probe/bb07-prof/build/golden/input.bin"
echo "===== level 3, per-op, -c 4 ====="
MTKNN_ENABLE_PROFILER=1 MTKNN_PER_OP_PROFILE=1 MTKNN_PROFILING_LEVEL=3   "$RT" -m null -a "$D" -i "$IN" -o "$T/o.bin" -c 4 2>&1
echo "===== rc=$? ====="
echo "===== files ====="
ls -la "$T"
