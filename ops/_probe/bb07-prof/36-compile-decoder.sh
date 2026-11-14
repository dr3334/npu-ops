#!/usr/bin/env bash
# 36-compile-decoder.sh — compile the multi-layer graphs (offline).
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
cd "$R" || exit 2
B=ops/_probe/bb07-prof/build3
ls -l "$B"/full.dir/golden/ "$B"/ffn_half.dir/golden/ 2>/dev/null | head -20
comp() { # tag, tflite, flags...
  local tag="$1" t="$2"; shift 2
  ./tools/compile.sh "$t" "$B/$tag.dla" "$@" | tail -1
}
comp full_p0  "$B/full.tflite"
comp full_p3  "$B/full.tflite" -P3
comp ffn_p3   "$B/ffn_half.tflite" -P3
comp attn_p3  "$B/attn_half.tflite" -P3
echo "== sizes =="
for f in "$B"/*.dla; do printf '%10s %s\n' "$(stat -c %s "$f")" "$(basename "$f")"; done
echo "== exec plan of full: how many layers/steps? =="
grep -cE "Layer|layer" "$B/full_p3.compile.log" 2>/dev/null
grep -E "^ *(step|[0-9]+:)" "$B/full_p3.compile.log" | head -10
