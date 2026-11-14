#!/usr/bin/env bash
# 00-build.sh — host side (WSL): build the two .tflite inputs, then compile eight DLAs.
#
#   wsl.exe -e bash -lc 'bash "ops/_probe/bb08-prof/00-build.sh"'
#
# Products (one file per level, never recompiled in place — ticket 08 needs three *distinct*
# products because the -P level is baked into the .dla):
#   fc_p0/1/2/3.dla     fc_1x64x64_int8            (int8, 1 layer)
#   dec_p0/1/2/3.dla    decoder_layer_t8_fp16      (fp16, 61 native layers, 29.9 MB tflite)
# p0 = no -P flag (the un-profiled reference; 07 used full_p0 for the same purpose).
#
# Evidence: build/*.compile.log (cmd / ncc version / exec-plan / memory-summary / EXIT / SIZE / SHA256).
set -uo pipefail
R="${R:-/mnt/d/deepseek_ workspace/mtk/wt-bb08}"
B="$R/ops/_probe/bb08-prof/build"
cd "$R" || exit 2
mkdir -p "$B"

if [ ! -f "$B/fc.tflite" ]; then
  echo "== build tflite: fc_1x64x64_int8 =="
  /usr/bin/python3 tools/build_ops.py fc_1x64x64_int8 --out "$B/fc.tflite" || exit 3
fi
if [ ! -f "$B/dec/model.tflite" ]; then
  echo "== build tflite: decoder_layer_t8_fp16 (29.9 MB, slow) =="
  /usr/bin/python3 tools/build_decoder_layer.py --out "$B/dec" || exit 3
fi
ls -l "$B/fc.tflite" "$B/dec/model.tflite" "$B/golden/input.bin" "$B/dec/golden/input.bin"

for lvl in 0 1 2 3; do
  FLAG=""; [ "$lvl" -gt 0 ] && FLAG="-P$lvl"
  echo "== compile fc $FLAG =="
  ./tools/compile.sh "$B/fc.tflite" "$B/fc_p$lvl.dla" $FLAG
done
for lvl in 0 1 2 3; do
  FLAG=""; [ "$lvl" -gt 0 ] && FLAG="-P$lvl"
  echo "== compile decoder $FLAG =="
  ./tools/compile.sh "$B/dec/model.tflite" "$B/dec_p$lvl.dla" $FLAG
done

echo "== sizes / sha (the -P level must change the product) =="
for f in "$B"/fc_p?.dla "$B"/dec_p?.dla; do
  printf '%-16s %10s  %s\n' "$(basename "$f")" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -c1-16)"
done
