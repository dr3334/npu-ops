#!/usr/bin/env bash
# 32-soname.sh — SONAME/NEEDED of the SDK runtime vs the one on the board.
set -uo pipefail
ADB=/mnt/d/adb/platform-tools/adb.exe
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
echo "== SDK aarch64 libneuron_runtime.so =="
readelf -d "$S/mt6897/lib/libneuron_runtime.so" | grep -E "SONAME|NEEDED"
echo
echo "== board /vendor/lib64/mt6897/libneuron_runtime.so =="
"$ADB" pull /vendor/lib64/mt6897/libneuron_runtime.so /mnt/d/bb07-tmp/vendor-runtime.so >/dev/null 2>&1 && echo "  pulled"
readelf -d /mnt/d/bb07-tmp/vendor-runtime.so 2>/dev/null | grep -E "SONAME|NEEDED"
echo
echo "== exported getProfiledQoSData on the vendor lib? =="
nm -D --defined-only /mnt/d/bb07-tmp/vendor-runtime.so 2>/dev/null | grep -E "getProfiledQoSData|NeuronRuntime_inference"
