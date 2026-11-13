#!/usr/bin/env bash
# Ticket kv-maxlen: sweep part 2 -- same as kvmax_sweep.sh but every build/compile is
# wrapped in a wall-clock cap, so an ncc memory-planning blow-up is recorded as a result
# instead of burning the session.
#   CS="16384 32768" BT=900 CT=2400 bash tools/kvmax_sweep2.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
export QWEN25_WEIGHTS_DIR="${QWEN25_WEIGHTS_DIR:-/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen}"
TAB=ops/q06_tab.json
LOGD=ops/kvmax
BT="${BT:-1200}"     # build cap (s)
CT="${CT:-2700}"     # compile cap (s)
CS="${CS:-16384 32768}"
mkdir -p "$LOGD"
for C in $CS; do
  OUT=ops/kvmax_decode_t8_c${C}_24l
  echo "===== C=${C} start $(date -Iseconds) build_cap=${BT}s compile_cap=${CT}s ====="
  rm -rf "$OUT"
  T0=$(date +%s%N)
  timeout "$BT" /usr/bin/python3 tools/build_q06.py --mode decode --t 8 --cache "$C" --layers 24 \
      --tab "$TAB" --out "$OUT" > "$LOGD/build_c${C}.log" 2>&1
  BRC=$?
  T1=$(date +%s%N)
  echo "C=${C} build_rc=${BRC} build_s=$(( (T1-T0)/1000000000 ))  (124 = timeout)"
  tail -3 "$LOGD/build_c${C}.log"
  if [ "$BRC" != "0" ] || [ ! -f "$OUT/model.tflite" ]; then echo "C=${C} BUILD_FAIL rc=$BRC"; continue; fi
  echo "C=${C} TFLITE=$(stat -c %s "$OUT/model.tflite")"
  T2=$(date +%s%N)
  timeout "$CT" ./tools/compile.sh "$OUT/model.tflite" "$OUT/model.dla" > "$LOGD/compile_c${C}.out" 2>&1
  CRC=$?
  T3=$(date +%s%N)
  echo "C=${C} compile_rc=${CRC} compile_s=$(( (T3-T2)/1000000000 ))  (124 = timeout)"
  tail -2 "$LOGD/compile_c${C}.out"
  if [ -f "$OUT/model.compile.log" ]; then
    grep -m3 -E "EXIT=|SIZE=|SHA256=" "$OUT/model.compile.log"
    grep -m3 -E "Execution Step|Bridging is disallowed|gap is detected|Cannot find an execution plan|Fail to process" "$OUT/model.compile.log"
    grep -A3 "DRAM Usage" "$OUT/model.compile.log" | tail -3
  fi
  rm -f "$OUT/model.tflite"
  echo "===== C=${C} done $(date -Iseconds) ====="
done
echo "SWEEP2_DONE $(date -Iseconds)"
