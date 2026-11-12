#!/usr/bin/env bash
# E-103 shape boundary scan (W3).
# For each (M,K,N): build the tflite, compile with --disallow-bridge, record the stage
# that fails. Compile-stage results are collected for every shape; device runs are done
# separately for a subset (see run_devices()).
#
#   ./tools/scan_shapes.sh > docs/CAPABILITY.tsv
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
OUT=${OUT:-/tmp/scan}
mkdir -p "$OUT"

emit() { printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5"; }
printf "M\tK\tN\ttflite\tcompile\n"

scan() {
  local m=$1 k=$2 n=$3
  local d="$OUT/fc_${m}x${k}x${n}"
  mkdir -p "$d"
  if ! /usr/bin/python3 tools/build_ops.py fc --m "$m" --k "$k" --n "$n" --dtype int8 \
        --out "$d/model.tflite" > "$d/build.log" 2>&1; then
    emit "$m" "$k" "$n" "FAIL" "-"; return
  fi
  if ./tools/compile.sh "$d/model.tflite" "$d/model.dla" > /dev/null 2>&1; then
    emit "$m" "$k" "$n" "OK" "OK"
  else
    local why
    why=$(grep -m1 -oE "(Unsupport[^,]*|Bridging is disallowed[^,]*|ERROR: [^,]*)" "$d/model.compile.log" 2>/dev/null | head -1)
    emit "$m" "$k" "$n" "OK" "FAIL:${why:-unknown}"
  fi
}

for k in 8 16 24 32 48 64 96 128 192 256 384 512; do scan 1 "$k" 64; done
for n in 8 16 24 32 48 64 96 128 192 256 384 512; do scan 1 64 "$n"; done
for m in 2 7 8 31 32 127 128; do scan "$m" 128 128; done
