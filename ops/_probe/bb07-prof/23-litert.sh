#!/usr/bin/env bash
# 23-litert.sh — trial 3: the LiteRT delegate path. What is in the zips and the runtime dirs?
set -uo pipefail
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
T=/mnt/d/bb07-tmp/litert
rm -rf "$T"; mkdir -p "$T"
echo "== runtime dirs (recursive) =="
find "$S/litert_mediatek_runtime" "$S/litert_mediatek_runtime_jit" -type f | head -40
echo
echo "== zips: sizes =="
ls -l "$S"/litert_npu_*.zip
echo
echo "== zip listings =="
for z in "$S"/litert_npu_*.zip; do
  echo "----- $(basename "$z") -----"
  unzip -l "$z" | head -25
done
