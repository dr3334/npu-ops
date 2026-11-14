#!/usr/bin/env bash
# 05-compiler-profiling-opt.sh — does ncc-tflite have a profiling-level option? (offline)
set -uo pipefail
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
echo "== ncc-tflite --help | grep -i -B2 -A4 profil =="
LD_LIBRARY_PATH="$S/host/lib" "$S/host/bin/ncc-tflite" --help 2>&1 | grep -i -B2 -A4 "profil" 
echo "== rc=$? =="
echo
echo "== ncc-tflite --help | grep -iE '^\s*-P|profiling-level|debug-info|urate' =="
LD_LIBRARY_PATH="$S/host/lib" "$S/host/bin/ncc-tflite" --help 2>&1 | grep -nE "\-P|profiling|urate|debug-info|-D," | head -20
echo
echo "== strings ncc-tflite: profiling-level / -P =="
strings -a "$S/host/bin/ncc-tflite" | grep -iE "profiling.level|profiling_level|^-P" | sort -u | head -20
