#!/usr/bin/env bash
set -u
cd '/mnt/d/deepseek_ workspace/mtk/wt-t128' || exit 1
export QWEN25_WEIGHTS_DIR='/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen'
for s in n1 q3 k14 v14 scores probs attn r1 n2 mlp_y down out; do
  d="ops/_t128_bisect/s_$s"
  /usr/bin/python3 tools/build_decoder_layer.py --T 128 --stage "$s" --out "$d" >/dev/null 2>&1 || { echo "BUILD_FAIL $s"; continue; }
  printf 'stage %-8s ' "$s"
  ./tools/compile.sh "$d/model.tflite" "$d/model.dla" 2>&1 | tail -1
  grep -c ERROR "$d/model.compile.log" 2>/dev/null | sed 's/^/   errors=/'
done
