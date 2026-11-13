#!/usr/bin/env bash
set -u
cd '/mnt/d/deepseek_ workspace/mtk/wt-t128' || exit 1
export QWEN25_WEIGHTS_DIR='/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen'
run() {
  k="$1"; t="$2"; d="ops/_t128_bisect/p_${k}_${t}"
  /usr/bin/python3 ops/_t128_bisect/probe_fc.py "$k" "$t" "$d" >/dev/null 2>&1 || { echo "BUILD_FAIL $k $t"; return; }
  printf '%-18s T=%-4s ' "$k" "$t"
  ./tools/compile.sh "$d/model.tflite" "$d/model.dla" 2>&1 | tail -1
}
for k in fc_only fc_reshape fc_add_reshape fc_reshape_rope; do run "$k" 128; done
run fc_reshape 8
