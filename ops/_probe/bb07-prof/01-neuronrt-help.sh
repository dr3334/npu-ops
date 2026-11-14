#!/usr/bin/env bash
# 01-neuronrt-help.sh — host neuronrt help + strings around --Xruntime (offline).
set -uo pipefail
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
echo "== host neuronrt --help =="
LD_LIBRARY_PATH="$S/host/lib" "$S/host/bin/neuronrt" --help 2>&1
echo "== rc=$? =="
echo
echo "== strings | grep -i xruntime (host) =="
strings -a "$S/host/bin/neuronrt" | grep -i -n "xruntime" | head -40
echo "== strings | grep -iE 'prof|trace|dump|log' (host, unique) =="
strings -a "$S/host/bin/neuronrt" | grep -iE "profil|trace|dump|log_|_log|verbose|debug" | sort -u | head -60
echo
echo "== device neuronrt (mt6897) size/type =="
file "$S/mt6897/bin/neuronrt"
echo "== strings | grep -i xruntime (device) =="
strings -a "$S/mt6897/bin/neuronrt" | grep -i "xruntime" | head -40
