#!/usr/bin/env bash
# 31-board-layout.sh — where do the runtime .so files actually live on this board?
set -uo pipefail
ADB=/mnt/d/adb/platform-tools/adb.exe
DEV=/data/local/tmp/npuops
echo "== $DEV tree (2 levels) =="
"$ADB" shell "ls -la $DEV; echo ---; ls -la $DEV/bin | head -20" | tr -d '\r'
echo "== does neuronrt still run? =="
"$ADB" shell "cd $DEV && ./bin/neuronrt -v" 2>&1 | tr -d '\r'
echo "== which libneuron_runtime does it see? =="
"$ADB" shell "ls -l /vendor/lib64/libneuron_runtime.so* 2>/dev/null; ls -l /system/lib64/libneuron_runtime.so* 2>/dev/null; ls -l /vendor/lib64/ | grep -i neuron" 2>&1 | tr -d '\r' | head -20
echo "== strace-free probe: what does the loader resolve =="
"$ADB" shell "cd $DEV && ldd ./bin/neuronrt 2>&1 | head -20" | tr -d '\r'
