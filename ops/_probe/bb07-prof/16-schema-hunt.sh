#!/usr/bin/env bash
# 16-schema-hunt.sh — is there an embedded JSON schema for the runtime options string?
set -uo pipefail
T=/mnt/d/bb07-tmp
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
F="$T/runtime.strings"
[ -f "$F" ] || strings -a "$S/host/lib/libneuron_runtime.so" > "$F"
echo "== lines containing 'properties' =="
grep -n "properties" "$F" | head -20
echo
echo "== lines containing '"type"' =="
grep -n '"type"' "$F" | head -20
echo
echo "== lines 3830-3850 (parallelismOption neighbourhood) =="
sed -n '3830,3850p' "$F"
echo
echo "== search for keys: nbThreads / dlaBuffer / parallelism =="
grep -nE "nbThreads|dlaBuffer|parallelism|nThreads|deviceId" "$F" | head -20
