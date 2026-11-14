#!/usr/bin/env bash
# 35-build-decoder.sh — build+compile the decoder-layer graph and its two halves (offline).
# These give a MULTI-LAYER graph, which is what per-layer profiling needs to be shown on.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
cd "$R" || exit 2
B=ops/_probe/bb07-prof/build3
mkdir -p "$B"

build_one() { # name, builder args...
  local name="$1"; shift
  if [ ! -f "$B/$name.tflite" ]; then
    /usr/bin/python3 tools/build_decoder_layer.py "$@" --out "$B/$name.dir" || { echo "BUILD_FAIL $name"; return 1; }
    cp "$B/$name.dir/model.tflite" "$B/$name.tflite" 2>/dev/null || { echo "NO_TFLITE $name"; ls "$B/$name.dir"; return 1; }
  fi
  ls -l "$B/$name.tflite"
}
build_one full      --stage r1
build_one attn_half --variant attn_half
build_one ffn_half  --variant ffn_half
