#!/usr/bin/env bash
# Ticket kv-maxlen: 1-layer probes -- is the wall a per-tensor/dimension limit (would show
# up with ONE layer too) or the 24-layer memory planning (only with 24)?
#   bash tools/kvmax_probe1l.sh            # CS + caps from the environment
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
export QWEN25_WEIGHTS_DIR="${QWEN25_WEIGHTS_DIR:-/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen}"
TAB=ops/q06_tab.json
LOGD=ops/kvmax
BT="${BT:-900}"
CT="${CT:-1800}"
CS="${CS:-8192 16384 32768 65536}"
for C in $CS; do
  OUT=ops/kvmax_1l_t8_c${C}
  echo "===== 1L C=${C} build $(date -Iseconds) ====="
  rm -rf "$OUT"
  T0=$(date +%s%N)
  timeout "$BT" /usr/bin/python3 tools/build_q06.py --mode decode --t 8 --cache "$C" --layers 1 \
      --tab "$TAB" --out "$OUT" > "$LOGD/build1l_c${C}.log" 2>&1
  BRC=$?; T1=$(date +%s%N)
  echo "1L C=${C} build_rc=${BRC} build_s=$(( (T1-T0)/1000000000 ))"
  tail -3 "$LOGD/build1l_c${C}.log"
  [ "$BRC" = "0" ] && [ -f "$OUT/model.tflite" ] || { echo "1L C=${C} BUILD_FAIL"; continue; }
  echo "1L C=${C} TFLITE=$(stat -c %s "$OUT/model.tflite")"
  T2=$(date +%s%N)
  timeout "$CT" ./tools/compile.sh "$OUT/model.tflite" "$OUT/model.dla" > "$LOGD/compile1l_c${C}.out" 2>&1
  CRC=$?; T3=$(date +%s%N)
  echo "1L C=${C} compile_rc=${CRC} compile_s=$(( (T3-T2)/1000000000 ))"
  tail -2 "$LOGD/compile1l_c${C}.out"
  if [ -f "$OUT/model.compile.log" ]; then
    grep -m3 -E "EXIT=|SIZE=" "$OUT/model.compile.log"
    grep -m4 -E "Execution Step|Bridging is disallowed|gap is detected|Cannot find an execution plan|Fail to process|rank of shape|ERROR" "$OUT/model.compile.log"
    grep -A3 "DRAM Usage" "$OUT/model.compile.log" | tail -2
  fi
  rm -f "$OUT/model.tflite"
  echo "===== 1L C=${C} done $(date -Iseconds) ====="
done
echo "PROBE1L_DONE $(date -Iseconds)"
