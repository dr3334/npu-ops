#!/usr/bin/env bash
# 18-envvars-offline.sh — do the MTKNN_* profiler env vars do anything without a device?
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
cd "$R" || exit 2
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
export LD_LIBRARY_PATH="$S/host/lib"
RT="$S/host/bin/neuronrt"
D=ops/_probe/bb07-prof/build2/q003.dla     # compiled with -P3
IN=ops/_probe/bb07-prof/build/golden/input.bin
T=/mnt/d/bb07-tmp/envtest
rm -rf "$T"; mkdir -p "$T"; cd "$T" || exit 3
for lvl in 1 2 3; do
  echo "== profiling level $lvl =="
  MTKNN_ENABLE_PROFILER=1 MTKNN_PER_OP_PROFILE=1 MTKNN_PROFILING_LEVEL=$lvl     "$RT" -m null -a "/mnt/d/deepseek_ workspace/mtk/wt-bb07/$D" -i "/mnt/d/deepseek_ workspace/mtk/wt-bb07/$IN"     -o "$T/out$lvl.bin" -c 2 2>&1 | head -10
  echo "  rc=$?"
done
echo "== files produced in $T =="
ls -la "$T"
echo "== and with an explicit log path =="
MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILER_LOG_PATH="$T/prof.json" MTKNN_PROFILING_LEVEL=3   "$RT" -m null -a "/mnt/d/deepseek_ workspace/mtk/wt-bb07/$D" -i "/mnt/d/deepseek_ workspace/mtk/wt-bb07/$IN" -o "$T/o.bin" -c 1 2>&1 | head -10
ls -la "$T"
