#!/usr/bin/env bash
# 22-hw-path-and-P12.sh — (a) does -m hw on the x86 host do anything? (b) -P1 vs -P2 bytes.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
export LD_LIBRARY_PATH="$S/host/lib"
RT="$S/host/bin/neuronrt"
B="$R/ops/_probe/bb07-prof/build2"
IN="$R/ops/_probe/bb07-prof/build/golden/input.bin"
T=/mnt/d/bb07-tmp/hw
rm -rf "$T"; mkdir -p "$T"; cd "$T" || exit 3
echo "== -m hw on the x86 host, profiler on, full output =="
MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3 "$RT" -m hw -a "$B/q003.dla" -i "$IN" -o "$T/o.bin" -c 2 2>&1 | head -20
echo "  rc=$?"
echo
echo "== -m cmodel =="
"$RT" -m cmodel -a "$B/q003.dla" -i "$IN" -o "$T/o2.bin" -c 1 2>&1 | head -10
echo "  rc=$?"
echo
echo "== -P1 vs -P2 byte identity =="
cmp -s "$B/q001.dla" "$B/q002.dla" && echo "q001 == q002 (byte-identical)" || { echo "q001 != q002"; cmp -l "$B/q001.dla" "$B/q002.dla" | head -20; }
echo
echo "== -P1 vs -P3 (first 30 differing byte positions) =="
cmp -l "$B/q001.dla" "$B/q003.dla" | head -10
echo "  total differing: $(cmp -l "$B/q001.dla" "$B/q003.dla" | wc -l)"
