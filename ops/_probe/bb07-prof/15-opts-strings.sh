#!/usr/bin/env bash
# 15-opts-strings.sh — what keys does the runtime's options string accept?
set -uo pipefail
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
R="$S/host/lib/libneuron_runtime.so"
T=/mnt/d/bb07-tmp
mkdir -p "$T"
strings -a "$R" > "$T/runtime.strings"
echo "== total strings: $(wc -l < "$T/runtime.strings") =="
echo "== neighbourhood of 'parallelismOption' =="
grep -n "parallelismOption" "$T/runtime.strings" | head
echo
echo "== neighbourhood of 'Runtime options' =="
n=$(grep -n "Runtime options: " "$T/runtime.strings" | head -1 | cut -d: -f1)
echo "line $n"
sed -n "$((n-25)),$((n+25))p" "$T/runtime.strings"
echo
echo "== keys that look like camelCase option names (heuristic) =="
grep -nE "^[a-z][a-zA-Z]{3,24}Option$|^[a-z]+[A-Z][a-zA-Z]{2,}$" "$T/runtime.strings" | sort -u -t: -k2 | head -40
