#!/usr/bin/env bash
set -uo pipefail
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH
cd "/mnt/d/deepseek_ workspace/mtk/wt-t08"
/usr/bin/python3 ops/lm_head_t8_fp16/probe/make_probes.py || exit 3
echo "=================== compile ==================="
for d in ops/lm_head_t8_fp16/probe/*/; do
  n=$(basename "$d")
  ./tools/compile.sh "$d/model.tflite" "$d/model.dla" >/dev/null 2>&1
  rc=$?
  reason=$(grep -m1 -E "Dimension should be|unsupported|Bridging|Fail to process|Cannot find" "$d/model.compile.log" | sed 's/^ *//' || true)
  printf '%-16s rc=%d  %s\n' "$n" "$rc" "$reason"
done
