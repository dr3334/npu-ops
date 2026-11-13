#!/usr/bin/env bash
# Board sequence for ticket 03 (held entirely inside ONE devlock acquisition).
#   ./tools/devlock.sh run -- ./tools/board_t03.sh
# Runs healthcheck, then 2 e2e passes per op (2nd = noise floor), then healthcheck.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
echo "=== healthcheck BEFORE ($(date -Iseconds)) ==="
./tools/healthcheck.sh --quick; echo "hc_before_rc=$?"
for w in q k v; do
  d="ops/qkv_${w}_bias_rope_t8_fp16"
  echo "=== ${w} run 1 ==="
  ./tools/e2e.sh "$d"; echo "${w}_r1_rc=$?"
  cp -f "$d/out_device.bin" "$d/out_r1.bin" 2>/dev/null && cp -f "$d/compare.txt" "$d/compare_r1.txt" 2>/dev/null
  echo "=== ${w} run 2 (noise floor) ==="
  ./tools/e2e.sh "$d"; echo "${w}_r2_rc=$?"
  cp -f "$d/out_device.bin" "$d/out_r2.bin" 2>/dev/null
done
echo "=== healthcheck AFTER ($(date -Iseconds)) ==="
./tools/healthcheck.sh --quick; echo "hc_after_rc=$?"
echo "=== BOARD_T03_DONE ==="
