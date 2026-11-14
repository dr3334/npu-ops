#!/usr/bin/env bash
# Ticket kv-maxlen: build+compile the C=64 decode graph -- the "tower floor" reference that
# makes the attention-width share a within-session measurement instead of a borrowed number.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
export QWEN25_WEIGHTS_DIR="${QWEN25_WEIGHTS_DIR:-/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen}"
C="${1:-64}"
OUT=ops/kvmax_decode_t8_c${C}_24l
echo "== build C=$C $(date -Iseconds)"
/usr/bin/python3 tools/build_q06.py --mode decode --t 8 --cache "$C" --layers 24 \
    --tab ops/q06_tab.json --out "$OUT" 2>&1 | tail -2
echo "== compile C=$C $(date -Iseconds)"
./tools/compile.sh "$OUT/model.tflite" "$OUT/model.dla" 2>&1 | tail -2
grep -A3 "DRAM Usage" "$OUT/model.compile.log" | tail -2
rm -f "$OUT/model.tflite"
echo "== prepare chk $C $(date -Iseconds)"
/usr/bin/python3 tools/kvmax_check.py --prepare --dir ops/kvmax/chk_c$C --cache "$C" --t 8 --layers 24 --tab ops/q06_tab.json
echo "== done $(date -Iseconds)"
