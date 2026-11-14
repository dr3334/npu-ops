#!/usr/bin/env bash
# 20-matrix.sh — DLA variant x profiler env matrix, host runtime, full stdout captured.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
export LD_LIBRARY_PATH="$S/host/lib"
RT="$S/host/bin/neuronrt"
B="$R/ops/_probe/bb07-prof/build2"
IN="$R/ops/_probe/bb07-prof/build/golden/input.bin"
T=/mnt/d/bb07-tmp/matrix
rm -rf "$T"; mkdir -p "$T"; cd "$T" || exit 3

run() { # tag, dla, envprefix...
  local tag="$1" dla="$2"; shift 2
  local d="$T/run_$tag"; mkdir -p "$d"
  ( cd "$d" && env "$@" "$RT" -m null -a "$B/$dla.dla" -i "$IN" -o "$d/out.bin" -c 3 ) > "$T/$tag.log" 2>&1
  printf '%-22s rc=%-2s lines=%-4s  %s\n' "$tag" "$?" "$(wc -l < "$T/$tag.log")" "$(grep -ciE 'debuginfo|per-op|profil' "$T/$tag.log") prof-lines"
}

run q000_none        q000
run q000_lvl3        q000 MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3
run q000_lvl3_perop  q000 MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3 MTKNN_PER_OP_PROFILE=1
run q001_lvl3        q001 MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3
run q100_none        q100
run q100_lvl3        q100 MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3
run q100_lvl3_perop  q100 MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3 MTKNN_PER_OP_PROFILE=1
run q003_lvl3        q003 MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3
run q003_lvl3_perop  q003 MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3 MTKNN_PER_OP_PROFILE=1
run q003_lvl2        q003 MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=2
echo
echo "== grep: debuginfo / per-op / level messages across all logs =="
grep -hniE "debuginfo|per-op|profiling level|only supports" "$T"/*.log | sort | uniq -c
echo
echo "== section headers seen (q003_lvl3_perop) =="
grep -nE "^-{10,}| Summary| Report|LEVEL" "$T/q003_lvl3_perop.log" | head -30
