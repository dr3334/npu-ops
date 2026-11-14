#!/usr/bin/env bash
# 33-pull-vendor.sh — pull the board's runtime lib (Windows dest path!) and inspect it.
set -uo pipefail
ADB=/mnt/d/adb/platform-tools/adb.exe
D=/mnt/d/bb07-tmp
"$ADB" pull /vendor/lib64/mt6897/libneuron_runtime.so "$(wslpath -w "$D/vendor-runtime.so")" 2>&1 | tr -d '\r'
ls -l "$D/vendor-runtime.so" || exit 2
file "$D/vendor-runtime.so"
readelf -d "$D/vendor-runtime.so" | grep -E "SONAME|NEEDED"
echo "== exports =="
nm -D --defined-only "$D/vendor-runtime.so" | grep -E "NeuronRuntime_(getProfiledQoSData|inference|create|loadNetworkFromFile)" || echo "  (nm found nothing)"
