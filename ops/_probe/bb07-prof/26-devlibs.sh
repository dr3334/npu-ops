#!/usr/bin/env bash
# 26-devlibs.sh — the aarch64 runtime libs on the device side of the SDK + NDK presence.
set -uo pipefail
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
echo "== mt6897 tree =="
ls -la "$S/mt6897"
echo "== mt6897/lib (first 30) =="
ls -la "$S/mt6897/lib" 2>/dev/null | head -30
echo "== getProfiledQoSData / inference / load exported by the aarch64 runtime? =="
nm -D --defined-only "$S/mt6897/lib/libneuron_runtime.so" 2>/dev/null | grep -iE "ProfiledQoS|NeuronRuntime_inference|loadNetworkFromFile"
echo "== NDK =="
ls -d /mnt/d/liteRT/android-ndk-r27c /mnt/d/liteRT/android-ndk-r26d 2>/dev/null
ls /mnt/d/liteRT/android-ndk-r27c/toolchains/llvm/prebuilt/ 2>/dev/null
echo "== probe/ contents =="
ls -l "/mnt/d/deepseek_ workspace/mtk/wt-bb07/probe/" 2>/dev/null | head
