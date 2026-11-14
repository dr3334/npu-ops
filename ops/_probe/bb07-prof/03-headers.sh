#!/usr/bin/env bash
# 03-headers.sh — public API headers: profiling entry points + option-string vocabulary.
set -uo pipefail
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
H="$S/host/include/neuron/api"
echo "== grep -n -i prof in headers =="
grep -n -i "prof" "$H"/*.h
echo
echo "== RuntimeAPI.h: getProfiledQoSData context =="
grep -n -B12 -A25 "getProfiledQoSData" "$H/RuntimeAPI.h"
