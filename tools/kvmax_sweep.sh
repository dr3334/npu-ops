#!/usr/bin/env bash
# Ticket kv-maxlen: sweep the decode graph's KV-cache length C (build + compile only, no board).
#   wsl.exe -e bash -lc "cd '/mnt/d/.../wt-kv2' && bash tools/kvmax_sweep.sh"
# One .dla per C; model.tflite deleted after compiling (regenerable, and tflite is ~390 MB each).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
TAB=ops/q06_tab.json
export QWEN25_WEIGHTS_DIR="${QWEN25_WEIGHTS_DIR:-/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen}"
LOGD=ops/kvmax
mkdir -p "$LOGD"
CS="${CS:-2048 4096 8192 16384 32768}"
for C in $CS; do
  OUT=ops/kvmax_decode_t8_c${C}_24l
  echo "===== C=${C} start $(date -Iseconds) ====="
  rm -rf "$OUT"
  T0=$(date +%s%N)
  /usr/bin/python3 tools/build_q06.py --mode decode --t 8 --cache "${C}" --layers 24 \
      --tab "$TAB" --out "$OUT" > "$LOGD/build_c${C}.log" 2>&1
  BRC=$?
  T1=$(date +%s%N)
  echo "C=${C} build_rc=${BRC} build_s=$(( (T1-T0)/1000000000 ))"
  tail -4 "$LOGD/build_c${C}.log"
  if [ "${BRC}" != "0" ] || [ ! -f "$OUT/model.tflite" ]; then echo "C=${C} BUILD_FAIL"; continue; fi
  echo "C=${C} TFLITE=$(stat -c %s "$OUT/model.tflite")"
  T2=$(date +%s%N)
  ./tools/compile.sh "$OUT/model.tflite" "$OUT/model.dla" > "$LOGD/compile_c${C}.out" 2>&1
  CRC=$?
  T3=$(date +%s%N)
  echo "C=${C} compile_rc=${CRC} compile_s=$(( (T3-T2)/1000000000 ))"
  tail -2 "$LOGD/compile_c${C}.out"
  if [ -f "$OUT/model.compile.log" ]; then
    grep -m2 -E 'EXIT=|SIZE=|SHA256=' "$OUT/model.compile.log"
    echo "--- exec steps / gap ---"
    grep -m4 -E 'Execution Step|Bridging|gap is detected' "$OUT/model.compile.log"
    echo "--- memory summary ---"
    grep -A12 -m1 -E 'Memory Summary|memory summary' "$OUT/model.compile.log" | head -20
  fi
  rm -f "$OUT/model.tflite"
  echo "===== C=${C} done $(date -Iseconds) ====="
done
echo "SWEEP_DONE $(date -Iseconds)"
